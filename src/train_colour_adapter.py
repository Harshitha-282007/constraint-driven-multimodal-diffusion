"""
Trains the ColourAdapter (src/coloradapter/) inside frozen SDXL + scribble ControlNet + IP-Adapter.

Rule being learned: the reference governs colour, even when the caption names another colour.

Training data (every row has an exact target):

    normal   photo, caption, reference                          target = photo           (palette of the reference)
    hue      photo object AND reference hue-rotated together     target = recoloured photo (palette of the recoloured
             (CIELAB, rolegate.augment.hue_pair), caption kept        reference); the caption now names the WRONG colour

Each step uses either the stale caption (the wrong colour word is still there: the adapter has to win) or the
colour-neutral caption (colour words -> "colored": the adapter is the only colour source). Inference uses the
neutral caption, so the reference's colour is the only colour in the prompt path.

What is frozen: SDXL, ControlNet, IP-Adapter (its colour channel is switched off with the role-gate knock; its texture
channel stays), the role gates (initial values, never trained). What is trained: ColourAdapter only.

    L = L_diff(masked, object weighted) + 0.5 * L_colour(decoded x0, object region, CIELAB; t in [200, 600])

    python src/train_colour_adapter.py --data /content/pilot_v1/pilot_v1 --gov .../governance.pt --out .../colour_adapter
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
from coloradapter.adapter import ColourAdapter
from coloradapter.palette import palette
from governance import model as GM
from governance.labels import _rng, gates_for
from pipeline import BASE_ID, CENTER, Pipeline, cs_shape, ip_shape
from rolegate import losses as LS
from rolegate.augment import hue_pair
from rolegate.gates import RoleGates
from train_fusion_lora import load_rows

W_COLOUR = 0.5
N_HUE = 4
VERSION = 1


# ----------------------------------------------------------------------------- data
def hue_angles(n, rng, lo=40.0, hi=320.0):
    """n hue rotations spread over [lo, hi] degrees, one random angle per equal slice (a rotation near 0 / 360 is no conflict)."""
    w = (hi - lo) / n
    return [lo + w * (i + rng.random()) for i in range(n)]


def build_specs(rows, data_dir, res, seed=0, n_hue=N_HUE):
    op = lambda r, k: Image.open(os.path.join(data_dir, r[k]))
    specs = []
    for r in rows:
        tgt = op(r, "target").convert("RGB").resize((res, res), Image.LANCZOS)
        mask = np.array(op(r, "mask").convert("L").resize((res, res), Image.NEAREST)) > 127
        ref = op(r, "reference").convert("RGB")
        common = dict(sketch=op(r, "sketch"), obj=r["category"], seed=TR.seed_for(r["id"]), ref_object=r.get("ref_object") or "object",
                      base=r["id"], target=tgt, mask=mask, rule=gates_for(r))
        specs.append(dict(common, key=r["id"], kind="normal", caption=r["caption"], ref=ref, pal=palette(np.array(ref))))
        for deg in hue_angles(n_hue, _rng(seed, r["id"], "hue")):
            hp = hue_pair(np.array(tgt), mask, np.array(ref), deg)
            if hp is not None:
                specs.append(dict(common, key=f"{r['id']}_hue{int(deg)}", kind="hue", caption=r["caption"], ref=Image.fromarray(hp[1]),
                                  target=Image.fromarray(hp[0]), pal=palette(hp[1])))
    return specs


def prepare(P, specs, net, log=print):
    data = TR.prepare(P, specs, net, log)
    for d, s in zip(data, specs):                       # TR.prepare keeps the order of `specs`
        assert d["key"] == s["key"]
        d["pal"] = torch.from_numpy(s["pal"])
    return data


# ----------------------------------------------------------------------------- one forward
def forward(P, role, ca, batch, sched, t, neutral):
    dev, dt, res = P.dev, P.dt, P.res
    B = len(batch)
    lat = torch.cat([d["lat"] for d in batch]).to(dev, dt)
    noise = torch.randn_like(lat)
    tt = t.repeat(B)
    noisy = sched.add_noise(lat, noise, tt)
    f = 1 - t.item() / sched.config.num_train_timesteps

    def caption(d):                                     # (prompt embeds, pooled); neutral version when the caption has colour words
        n = d["txt_neu"]["colour"] if neutral else None
        return n if n is not None else (d["txt"][0], d["txt"][2])

    pe = torch.cat([caption(d)[0] for d in batch]).to(dev)
    pooled = torch.cat([caption(d)[1] for d in batch]).to(dev)
    groups = torch.stack([d["groups"] for d in batch]).to(dev)
    emb = torch.cat([d["emb"] for d in batch]).to(dev, dt)
    tids = torch.tensor([[res, res, 0, 0, res, res]], device=dev, dtype=dt).repeat(B, 1)
    cs, ips = CENTER["cs"] * cs_shape(f, CENTER["hold"]), CENTER["ip"] * ip_shape(f)
    role.set_context(torch.zeros(B, 3, device=dev), tt, groups, ips, knock=("ref_colour",))   # IP colour channel off; texture stays
    role.ctx.collect = None
    ca.set_palette(torch.stack([d["pal"] for d in batch]).to(dev))
    added = {"text_embeds": pooled, "time_ids": tids}
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
def train(data_dir, gov_path, out_dir, res=512, lr=5e-4, lr_alpha=1e-2, accum=4, steps=300, save_every=100, seed=0,
          resume=None, memory_test=False, P=None, log=print):
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
    data = prepare(P, specs, net, log)
    by_kind = {k: sum(d["kind"] == k for d in data) for k in ("normal", "hue")}
    dev = P.dev

    role = RoleGates.for_unet(P.unet, P.controlnet_res_channels()).to(dev).float().eval()
    role.requires_grad_(False)
    role.install(P.unet)
    ca = ColourAdapter.for_unet(P.unet).to(dev).float()
    start = 0
    if resume:
        d = torch.load(resume, map_location=dev, weights_only=False)
        ca.load_state_dict(d["state"]); start = int(d["meta"].get("step", 0))
        log(f"resumed from {resume} at step {start}")
    ca.install(P.unet); ca.train()
    alpha = list(ca._alpha.parameters())
    rest = [p for n, p in ca.named_parameters() if not n.startswith("_alpha")]
    opt = torch.optim.AdamW([dict(params=rest, lr=lr), dict(params=alpha, lr=lr_alpha)], weight_decay=0.0)
    scaler = torch.cuda.amp.GradScaler()
    sched = DDPMScheduler.from_pretrained(BASE_ID, subfolder="scheduler")
    T = sched.config.num_train_timesteps
    P.unet.enable_gradient_checkpointing(); P.unet.train(); P.vae.enable_gradient_checkpointing()
    log(f"adapter params: {sum(p.numel() for p in ca.parameters()) / 1e6:.2f} M over {len(ca.layer_dims)} cross-attention layers | "
        f"{by_kind} | GPU {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    torch.cuda.reset_peak_memory_stats()

    total = start + (2 if memory_test else steps)
    hist, micro, step, t0 = [], 0, start, time.time()
    opt.zero_grad(set_to_none=True)
    try:
        while step < total:
            for _ in range(accum):
                micro += 1
                do_attr = micro % 2 == 0 or memory_test
                d = random.choice(data)
                t = torch.randint(200, 601, (1,), device=dev) if do_attr else torch.randint(0, T, (1,), device=dev)
                neutral = random.random() < 0.5
                pred, noise, noisy = forward(P, role, ca, [d], sched, t, neutral)
                parts = dict(diff=LS.masked_diffusion(pred, noise, d["mask_lat"].to(dev)))
                if do_attr:
                    img = TR.decode_x0(P, sched, noisy, pred, t)
                    parts["colour"] = W_COLOUR * LS.colour_loss(img, d["tgt256"].to(dev).float(), d["mask256"].to(dev))
                loss = sum(parts.values())
                scaler.scale(loss / accum).backward()
                hist.append(dict(step=step, micro=micro, kind=d["kind"], neutral=neutral, t=int(t), loss=float(loss),
                                 **{k: float(v) for k, v in parts.items()}))
                del pred, noise, noisy, loss
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(ca.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0 or memory_test:
                rec = hist[-10 * accum:]
                mean = lambda k: np.nanmean([h.get(k, np.nan) for h in rec]) if any(k in h for h in rec) else float("nan")
                am = float(torch.stack([p.detach().abs().mean() for p in alpha]).mean())
                log(f"step {step:4d}/{total} loss {mean('loss'):.4f} diff {mean('diff'):.4f} colour {mean('colour'):.4f} | mean |alpha| {am:.4f} | "
                    f"peak GPU {torch.cuda.max_memory_allocated() / 1e9:.2f} GB | {(time.time() - t0) / 60:.1f} min")
            if not memory_test and step % save_every == 0:
                ca.save(os.path.join(out_dir, f"colour_adapter_step{step:04d}.pt"), dict(step=step))
                json.dump(hist, open(os.path.join(out_dir, "train_log.json"), "w"))
    finally:
        ca.remove(P.unet); role.remove(P.unet)
        P.unet.disable_gradient_checkpointing(); P.unet.eval(); P.vae.disable_gradient_checkpointing()
    ca.eval()
    if memory_test:
        log(f"MEMORY TEST OK at res={res}: peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    else:
        cfg = dict(version=VERSION, res=res, lr=lr, lr_alpha=lr_alpha, accum=accum, steps=step, w_colour=W_COLOUR, n_samples=len(data),
                   kinds=by_kind, minutes=(time.time() - t0) / 60)
        ca.save(os.path.join(out_dir, "colour_adapter.pt"), dict(step=step, train_cfg=cfg))
        json.dump(cfg, open(os.path.join(out_dir, "colour_adapter_config.json"), "w"), indent=1)
        json.dump(hist, open(os.path.join(out_dir, "train_log.json"), "w"))
        log(f"saved {out_dir}/colour_adapter.pt")
    gc.collect(); torch.cuda.empty_cache()
    return P, ca, hist


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--gov", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--memory_test", action="store_true")
    a = ap.parse_args()
    train(a.data, a.gov, a.out, res=a.res, steps=a.steps, lr=a.lr, resume=a.resume, memory_test=a.memory_test)
