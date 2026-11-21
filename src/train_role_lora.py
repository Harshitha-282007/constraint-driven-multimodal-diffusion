"""
Trains the Role LoRA (src/rolelora/lora.py): the model learns, from conflict examples with exact targets, which input
governs which attribute. Nothing is neutralised or switched off at training or at inference: the role gates stay at their
initial values with conflict flags 0 (no detector), and the caption keeps whatever colour or material word it has.

Roles and where their supervision comes from:

    sketch -> shape, placement   every row: the reference is a neutral object (vase, mug ...) with another shape, and the target
                                 has the photo's shape; also the preservation loss on normal rows keeps the sketch path
    text   -> object identity    every row (the caption names the object); the preservation loss keeps identity intact
    ref    -> colour, material   col / mat rows (the caption names a wrong colour / material, target = photo) and hue rows
                                 (photo object and reference hue-rotated together, caption now stale, target = recoloured photo)

No exact target exists for "the caption names another object than the sketch shows", so that conflict has no training rows
(it is evaluated, not trained).

    L = L_diff(masked) + 2 L_colour(decoded x0, object mask) + 0.5 L_texture + 0.5 L_keep   (L_keep: normal rows only)

    L_keep = || eps_lora - eps_base ||^2, eps_base from the same model with the LoRA switched off: it ties the LoRA to the
    baseline wherever the baseline was already right (identity, sketch layout), so the LoRA only has to learn the conflicts.

    python src/train_role_lora.py --data /content/pilot_v1/pilot_v1 --gov .../governance.pt --out .../role_lora --groups text ref
"""
import argparse
import gc
import json
import os
import random
import time

import numpy as np
import torch
from PIL import Image

import train_role_gates as TR
from governance import model as GM
from pipeline import BASE_ID, CENTER, Pipeline, cs_shape, ip_shape
from rolegate import losses as LS
from rolegate.augment import hue_pair
from rolegate.gates import RoleGates
from rolelora.lora import RoleLoRA
from train_colour_adapter import hue_angles, safe_save
from train_fusion_lora import load_rows

W = dict(colour=2.0, texture=0.5, keep=0.5)
VERSION = 1


# ----------------------------------------------------------------------------- data
def build_specs(rows, data_dir, res, seed=0, extra_hue=1):
    """normal + col + mat + hue rows (train_role_gates.build_rows) plus `extra_hue` more hue rotations per photo."""
    specs = TR.build_rows(rows, data_dir, res, seed)
    from governance.labels import _rng
    for s in [s for s in specs if s["kind"] == "normal"]:
        for k, deg in enumerate(hue_angles(extra_hue, _rng(seed, s["key"], "hue_extra"))):
            hp = hue_pair(np.array(s["target"]), s["mask"], np.array(s["ref"]), deg)
            if hp is not None:
                specs.append(dict(s, key=f"{s['key']}_hx{k}_{int(deg)}", kind="hue", ref=Image.fromarray(hp[1]), target=Image.fromarray(hp[0])))
    return specs


# ----------------------------------------------------------------------------- one forward
def forward(P, role, batch, sched, t, noise=None):
    """Plain forward of the baseline architecture (no neutralisation, flags 0). Returns (pred eps, noise, noisy).
    Pass the returned `noise` back in to compare two passes (LoRA on / off) on the identical noisy input."""
    dev, dt, res = P.dev, P.dt, P.res
    B = len(batch)
    lat = torch.cat([d["lat"] for d in batch]).to(dev, dt)
    noise = torch.randn_like(lat) if noise is None else noise
    tt = t.repeat(B)
    noisy = sched.add_noise(lat, noise, tt)
    f = 1 - t.item() / sched.config.num_train_timesteps
    pe = torch.cat([d["txt"][0] for d in batch]).to(dev); pooled = torch.cat([d["txt"][2] for d in batch]).to(dev)
    groups = torch.stack([d["groups"] for d in batch]).to(dev)
    emb = torch.cat([d["emb"] for d in batch]).to(dev, dt)
    tids = torch.tensor([[res, res, 0, 0, res, res]], device=dev, dtype=dt).repeat(B, 1)
    cs, ips = CENTER["cs"] * cs_shape(f, CENTER["hold"]), CENTER["ip"] * ip_shape(f)
    role.set_context(torch.zeros(B, 3, device=dev), tt, groups, ips)
    role.ctx.collect = None
    added = {"text_embeds": role.mix_pooled(pooled), "time_ids": tids}
    ctrl = torch.cat([d["ctrl"] for d in batch]).to(dev, dt)
    with torch.autocast("cuda", dtype=torch.float16):
        with torch.no_grad():
            down, mid = P.controlnet(noisy, tt, encoder_hidden_states=pe, controlnet_cond=ctrl, conditioning_scale=float(cs),
                                     guess_mode=False, added_cond_kwargs=added, return_dict=False)
        down, mid = role.gate_controlnet(down, mid)
        pred = P.unet(noisy, tt, encoder_hidden_states=pe, added_cond_kwargs={**added, "image_embeds": [emb]},
                      down_block_additional_residuals=down, mid_block_additional_residual=mid, return_dict=False)[0]
    return pred, noise, noisy


# ----------------------------------------------------------------------------- training
def train(data_dir, gov_path, out_dir, res=512, lr=3e-4, accum=4, steps=400, save_every=100, seed=0, groups=("text", "ref"),
          rank=16, resume=None, memory_test=False, P=None, log=print):
    from diffusers import DDPMScheduler
    random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    P = P or Pipeline(res=res, log=log)
    net = GM.load(gov_path)
    rows = load_rows(data_dir, "train", only_clean=True)
    specs = build_specs(rows, data_dir, res, seed)
    if memory_test:
        keep = {s["base"] for s in specs[:6]}
        specs = [s for s in specs if s["base"] in keep]
    data = TR.prepare(P, specs, net, log)
    kinds = {k: sum(d["kind"] == k for d in data) for k in ("normal", "col", "mat", "hue")}
    normal = [d for d in data if d["kind"] == "normal"]
    dev = P.dev

    lora = RoleLoRA.attach(P.unet, groups, rank, alpha=rank).to(dev).float()          # BEFORE the gates are installed
    start = 0
    if resume:
        d = torch.load(resume, map_location=dev, weights_only=False)
        lora.load_state_dict(d["state"]); start = int(d["meta"].get("step", 0))
        log(f"resumed from {resume} at step {start}")
    role = RoleGates.for_unet(P.unet, P.controlnet_res_channels()).to(dev).float().eval()
    role.requires_grad_(False)
    role.install(P.unet)
    params = list(lora.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    sched = DDPMScheduler.from_pretrained(BASE_ID, subfolder="scheduler")
    T = sched.config.num_train_timesteps
    P.unet.enable_gradient_checkpointing(); P.unet.train(); P.vae.enable_gradient_checkpointing()
    log(f"LoRA groups {groups} rank {rank}: {lora.n_params() / 1e6:.2f} M params over {len(lora.names)} projections | {kinds} | "
        f"GPU {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    torch.cuda.reset_peak_memory_stats()

    total = start + (2 if memory_test else steps)
    hist, micro, step, t0 = [], 0, start, time.time()
    opt.zero_grad(set_to_none=True)
    try:
        while step < total:
            for _ in range(accum):
                micro += 1
                do_attr = micro % 2 == 0 or memory_test
                do_keep = micro % 3 == 0 or memory_test
                d = random.choice(normal) if do_keep else random.choice(data)
                t = torch.randint(200, 601, (1,), device=dev) if do_attr else torch.randint(0, T, (1,), device=dev)
                pred, noise, noisy = forward(P, role, [d], sched, t)
                parts = dict(diff=LS.masked_diffusion(pred, noise, d["mask_lat"].to(dev)))
                if do_keep:
                    with torch.no_grad(), lora.disabled():
                        pred0, _, _ = forward(P, role, [d], sched, t, noise=noise)
                    parts["keep"] = W["keep"] * ((pred.float() - pred0.float()) ** 2).mean()
                if do_attr:
                    img = TR.decode_x0(P, sched, noisy, pred, t)
                    tgt, m = d["tgt256"].to(dev).float(), d["mask256"].to(dev)
                    parts["colour"] = W["colour"] * LS.colour_loss(img, tgt, m)
                    parts["texture"] = W["texture"] * LS.texture_loss(img, tgt, m)
                loss = sum(parts.values())
                scaler.scale(loss / accum).backward()
                hist.append(dict(step=step, micro=micro, kind=d["kind"], t=int(t), loss=float(loss), **{k: float(v) for k, v in parts.items()}))
                del pred, noise, noisy, loss
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0 or memory_test:
                rec = hist[-10 * accum:]
                mean = lambda k: np.nanmean([h.get(k, np.nan) for h in rec]) if any(k in h for h in rec) else float("nan")
                log(f"step {step:4d}/{total} loss {mean('loss'):.4f} diff {mean('diff'):.4f} colour {mean('colour'):.4f} texture {mean('texture'):.4f} "
                    f"keep {mean('keep'):.4f} | peak GPU {torch.cuda.max_memory_allocated() / 1e9:.2f} GB | {(time.time() - t0) / 60:.1f} min")
            if not memory_test and step % save_every == 0:
                safe_save(lora, out_dir, f"role_lora_step{step:04d}.pt", dict(step=step), hist, log)
    finally:
        role.remove(P.unet)
        P.unet.disable_gradient_checkpointing(); P.unet.eval(); P.vae.disable_gradient_checkpointing()
    lora.eval()
    if memory_test:
        log(f"MEMORY TEST OK at res={res}: peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    else:
        cfg = dict(version=VERSION, res=res, lr=lr, accum=accum, steps=step, groups=list(groups), rank=rank, weights=W, n_samples=len(data),
                   kinds=kinds, minutes=(time.time() - t0) / 60)
        where = safe_save(lora, out_dir, "role_lora.pt", dict(step=step, train_cfg=cfg), hist, log)
        if where:
            json.dump(cfg, open(os.path.join(where, "role_lora_config.json"), "w"), indent=1)
        log(f"saved {where}/role_lora.pt")
    gc.collect(); torch.cuda.empty_cache()
    return P, lora, hist


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--gov", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--groups", nargs="+", default=["text", "ref"], choices=["text", "ref"])
    ap.add_argument("--resume", default=None)
    ap.add_argument("--memory_test", action="store_true")
    a = ap.parse_args()
    train(a.data, a.gov, a.out, res=a.res, steps=a.steps, lr=a.lr, rank=a.rank, groups=tuple(a.groups), resume=a.resume, memory_test=a.memory_test)
