"""
Trains the role gates (src/rolegate/gates.py) inside frozen SDXL + scribble ControlNet + IP-Adapter, so that the
network itself decides, per layer, per pixel and per attribute, how much of each input acts:

    sketch     -> shape, placement          (ControlNet residual gates: contour, placement)
    text       -> object identity           (text token-group gates: identity, colour words, material words, other)
    reference  -> colour and material       (reference channel gates: colour, texture)

Training data (125 clean training photos; the correct output is known for every row):

    normal   original (photo, sketch, reference, caption)                         target = photo
    col      caption names a wrong colour (labels.make_color)                     target = photo
    mat      caption names a wrong material (labels.make_material)                target = photo
    hue      photo object and reference both hue-rotated in CIELAB, caption kept  target = recoloured photo
    drop     negative prompt, no reference (keeps classifier-free guidance valid) target = photo

Gates are conditioned on the governance network's PREDICTED flags (notebook 05), so training sees detector errors;
the prior loss uses the rule labels.

    L = L_diff(masked) + 0.5 L_attr + 0.2 L_swap + 0.5 L_prior (annealed to 0.1 by step 450) + 0.1 L_keep + 0.01 L_reg

v2 (after the first run): sketch gates read layer-normalised residuals, train at 0.2x the learning rate and may not
close (L_keep); text colour / material act through neutral-caption channels (rolegate/gates.py), including the
pooled embedding; the reference prior keeps each layer's initial level; the contour term is down-weighted (0.1),
because on the 32-latent decode it gave no usable signal.

v3 (after v2 kept the stated colour): flagged conflicts start with that text channel closed (gates.FLAG_INIT),
and the reference colour channel starts stronger in the up-blocks (gates.REF_COLOUR_INIT).

Phases (optimizer steps): 0-150 warm-up (normal + drop rows, diff + prior), 150-450 all rows + attribute losses
(every 2nd micro-step, t in [200, 600]), 450-600 + swap loss (every 4th micro-step, hue pair batched with its
original in one UNet call - required because gradient checkpointing re-runs the processors during backward).

    python src/train_role_gates.py --data /content/pilot_v1/pilot_v1 --gov .../governance.pt --out .../role_gates
    python src/train_role_gates.py ... --memory_test                   # one step of every kind, prints peak memory
"""
import argparse
import gc
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from governance import model as GM
from governance.features import object_phrase
from governance.labels import _rng, caption_colors, gates_for, make_color, make_material
from pipeline import BASE_ID, CENTER, Pipeline, cs_shape, ip_shape, seed_for
from rolegate import losses as LS
from rolegate.augment import hue_pair
from rolegate.gates import RoleGates
from train_fusion_lora import load_rows

HUE_DEGREES = (90, 150, 210, 270)
W = dict(attr=0.5, swap=0.2, prior=0.5, prior_floor=0.2, keep=0.1, reg=0.01)    # prior_floor: fraction kept after the anneal
ATTR_W = dict(colour=1.0, texture=1.0, contour=0.1, identity=1.0)
SKETCH_LR = 0.2                                                                   # sketch gates' lr multiplier
VERSION = 3
PHASES = dict(warm=150, conflict=450, total=600)
FLAG_KEYS = ("g_obj", "g_col", "g_mat")


# ----------------------------------------------------------------------------- data
def build_rows(rows, data_dir, res, seed=0, hue=True):
    """Training rows -> list of sample specs (PIL images + metadata). Hue rows point at their original with `pair`."""
    op = lambda r, k: Image.open(os.path.join(data_dir, r[k]))
    specs = []
    for r in rows:
        tgt = op(r, "target").convert("RGB").resize((res, res), Image.LANCZOS)
        mask = np.array(op(r, "mask").convert("L").resize((res, res), Image.NEAREST)) > 127
        ref = op(r, "reference").convert("RGB")
        common = dict(sketch=op(r, "sketch"), obj=r["category"], seed=seed_for(r["id"]), ref_object=r.get("ref_object") or "object",
                      base=r["id"], target=tgt, mask=mask)
        specs.append(dict(common, key=r["id"], kind="normal", caption=r["caption"], ref=ref, rule=gates_for(r)))
        for kind, maker in (("col", make_color), ("mat", make_material)):
            v = maker(r, rows, _rng(seed, r["id"], kind))
            if v is not None:
                specs.append(dict(common, key=v["id"], kind=kind, caption=v["caption"], ref=ref, rule=gates_for(v)))
        if hue:
            deg = _rng(seed, r["id"], "hue").choice(HUE_DEGREES)
            hp = hue_pair(np.array(tgt), mask, np.array(ref), deg)
            if hp is not None:
                rule = gates_for(r)
                rule["g_col"] = int(bool(rule["g_col"]) or bool(caption_colors(r["caption"])))   # stated colour is now wrong
                specs.append(dict(common, key=f"{r['id']}_hue{deg}", kind="hue", caption=r["caption"], ref=Image.fromarray(hp[1]),
                                  target=Image.fromarray(hp[0]), rule=rule, pair=r["id"]))
    return specs


def _img_t(pil, size):
    return torch.from_numpy(np.array(pil.convert("RGB").resize((size, size), Image.BICUBIC))).permute(2, 0, 1)[None].float() / 255


def _mask_t(m, size, dilate=0):
    t = F.interpolate(torch.from_numpy(np.asarray(m, np.float32))[None, None], size=(size, size), mode="area")
    t = (t > 0.0).float()                       # any coverage counts: small objects must not vanish at latent size
    return F.max_pool2d(t, 2 * dilate + 1, 1, dilate) if dilate else t


@torch.no_grad()
def prepare(P, specs, net, log=print, chunk=16):
    """Encode every sample once (on CPU afterwards): text, reference channels, flags, latents, masks, control image."""
    res, dev, dt = P.res, P.dev, P.dt
    P.prepare_object_dirs([s["ref_object"] for s in specs])
    ctrl_cache, data = {}, []
    for i in range(0, len(specs), chunk):
        part = specs[i:i + chunk]
        items = P.make_items([{k: s[k] for k in ("key", "sketch", "ref", "caption", "obj", "seed", "ref_object")} for s in part])
        ref_obj = P.clip.text([object_phrase(s["ref_object"]) for s in part])
        for s, it, ro in zip(part, items, ref_obj):
            x = (_img_t(s["target"], res) * 2 - 1).to(dev, dt)
            lat = (P.vae.encode(x).latent_dist.sample() * P.vae.config.scaling_factor).cpu()
            if s["base"] not in ctrl_cache:
                ctrl_cache[s["base"]] = P.pipe.prepare_image(image=it["sketch"], width=res, height=res, batch_size=1, num_images_per_prompt=1,
                                                             device=dev, dtype=dt, do_classifier_free_guidance=False, guess_mode=False).cpu()
            pred = GM.predict_gates(net, it["x"])
            data.append(dict(
                key=s["key"], kind=s["kind"], base=s["base"], pair=s.get("pair"), caption=s["caption"],
                txt=it["txt"], groups=torch.tensor(it["text_groups"], dtype=torch.long), emb=P.ref_channels(it).half(),
                flags_pred=torch.tensor([pred[k] for k in FLAG_KEYS]), flags_rule=torch.tensor([float(s["rule"][k]) for k in FLAG_KEYS]),
                lat=lat, ctrl=ctrl_cache[s["base"]], mask_lat=_mask_t(s["mask"], res // 8, dilate=2), mask256=_mask_t(s["mask"], 256),
                tgt256=_img_t(s["target"], 256).half(), sk256=P._soft_sketch(it["sketch"], 256).cpu(),
                obj_emb=it["clip"]["obj"].float(), ref_obj_emb=ro.float(), txt_neu=it["txt_neutral"]))
            it.pop("S_gh", None)
        del items; torch.cuda.empty_cache()
    kinds = {}
    for d in data:
        kinds[d["kind"]] = kinds.get(d["kind"], 0) + 1
    n_neu = {w: sum(d["txt_neu"][w] is not None for d in data) for w in ("colour", "material")}
    log(f"prepared {len(data)} samples {kinds} | {len(ctrl_cache)} sketches | neutral captions {n_neu}")
    return data


# ----------------------------------------------------------------------------- one forward
def forward(P, role, batch, sched, t, drop=False, need_x0=False):
    """batch: list of samples sharing nothing but t. Returns (pred eps, noise, noisy, collected gates)."""
    dev, dt, res = P.dev, P.dt, P.res
    B = len(batch)
    lat = torch.cat([d["lat"] for d in batch]).to(dev, dt)
    noise = torch.randn_like(lat[:1]).repeat(B, 1, 1, 1)               # same noise for every row (swap pairs need it)
    tt = t.repeat(B)
    noisy = sched.add_noise(lat, noise, tt)
    f = 1 - t.item() / sched.config.num_train_timesteps
    if drop:
        pe = torch.cat([d["txt"][1] for d in batch]).to(dev); pooled = torch.cat([d["txt"][3] for d in batch]).to(dev)
        groups = torch.zeros(B, 77, dtype=torch.long, device=dev)
        emb = torch.zeros(B, 2, batch[0]["emb"].shape[-1], device=dev, dtype=dt)
    else:
        pe = torch.cat([d["txt"][0] for d in batch]).to(dev); pooled = torch.cat([d["txt"][2] for d in batch]).to(dev)
        groups = torch.stack([d["groups"] for d in batch]).to(dev)
        emb = torch.cat([d["emb"] for d in batch]).to(dev, dt)
    flags = torch.stack([d["flags_pred"] for d in batch]).to(dev)
    tids = torch.tensor([[res, res, 0, 0, res, res]], device=dev, dtype=dt).repeat(B, 1)
    tn, pn = {}, {}
    if not drop:                                       # neutral-caption channels; rows without one use their own caption
        for w in ("colour", "material"):
            if any(d["txt_neu"][w] is not None for d in batch):
                tn[w] = torch.cat([(d["txt_neu"][w] or (d["txt"][0], d["txt"][2]))[0] for d in batch]).to(dev)
                pn[w] = torch.cat([(d["txt_neu"][w] or (d["txt"][0], d["txt"][2]))[1] for d in batch]).to(dev)
    cs, ips = CENTER["cs"] * cs_shape(f, CENTER["hold"]), CENTER["ip"] * ip_shape(f)    # same base schedule as inference
    role.set_context(flags, tt, groups, ips, text_neutral=tn)
    role.ctx.collect = []
    added = {"text_embeds": role.mix_pooled(pooled, pn.get("colour"), pn.get("material")), "time_ids": tids}
    ctrl = torch.cat([d["ctrl"] for d in batch]).to(dev, dt)
    with torch.autocast("cuda", dtype=torch.float16):
        with torch.no_grad():
            down, mid = P.controlnet(noisy, tt, encoder_hidden_states=pe, controlnet_cond=ctrl, conditioning_scale=float(cs),
                                     guess_mode=False, added_cond_kwargs=added, return_dict=False)
        down, mid = role.gate_controlnet(down, mid)
        pred = P.unet(noisy, tt, encoder_hidden_states=pe, added_cond_kwargs={**added, "image_embeds": [emb]},
                      down_block_additional_residuals=down, mid_block_additional_residual=mid, return_dict=False)[0]
    return pred, noise, noisy, role.ctx.collect


def decode_x0(P, sched, noisy, pred, t, size=256):
    """Predicted clean image at `size` px in [0,1], differentiable (latents area-downsampled before decoding)."""
    a = sched.alphas_cumprod.to(noisy.device)[t].float().view(-1, 1, 1, 1)
    x0 = (noisy.float() - (1 - a).sqrt() * pred.float()) / a.sqrt()
    z = F.interpolate(x0, size=(size // 8, size // 8), mode="area")
    with torch.autocast("cuda", dtype=torch.float16):
        img = P.vae.decode((z / P.vae.config.scaling_factor).to(P.dt), return_dict=False)[0]
    return ((img.float() + 1) / 2).clamp(0, 1)


def attr_losses(P, img, batch):
    dev = img.device
    tgt = torch.cat([d["tgt256"] for d in batch]).to(dev).float()
    m = torch.cat([d["mask256"] for d in batch]).to(dev)
    sk = torch.cat([d["sk256"] for d in batch]).to(dev).float()
    emb = P.clip.image(img)
    parts = dict(colour=LS.colour_loss(img, tgt, m), texture=LS.texture_loss(img, tgt, m), contour=LS.contour_loss(img, sk),
                 identity=LS.identity_loss(emb, torch.stack([d["obj_emb"] for d in batch]).to(dev),
                                           torch.stack([d["ref_obj_emb"] for d in batch]).to(dev)))
    return sum(ATTR_W[k] * v for k, v in parts.items()), parts, emb


# ----------------------------------------------------------------------------- training
def train(data_dir, gov_path, out_dir, res=512, lr=5e-4, accum=4, phases=PHASES, save_every=150, seed=0,
          memory_test=False, P=None, log=print):
    from diffusers import DDPMScheduler
    random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    P = P or Pipeline(res=res, log=log)
    net = GM.load(gov_path)
    rows = load_rows(data_dir, "train", only_clean=True)
    specs = build_rows(rows, data_dir, res, seed)
    if memory_test:
        keep = {specs[i]["base"] for i in range(min(3, len(specs)))}
        hue_bases = [s["base"] for s in specs if s["kind"] == "hue"][:1]
        specs = [s for s in specs if s["base"] in keep | set(hue_bases)]
    data = prepare(P, specs, net, log)
    by_key = {d["key"]: d for d in data}
    normal = [d for d in data if d["kind"] == "normal"]
    allrows = [d for d in data if d["kind"] in ("normal", "col", "mat", "hue")]
    hue = [d for d in data if d["kind"] == "hue" and d["pair"] in by_key]

    dev = P.dev
    role = RoleGates.for_unet(P.unet, P.controlnet_res_channels()).to(dev).float()
    init = {k: v.detach().clone() for k, v in role.named_parameters()}
    params = list(role.parameters())
    sk_ids = {id(p) for p in role.sketch.parameters()}
    opt = torch.optim.AdamW([dict(params=[p for p in params if id(p) not in sk_ids], lr=lr),
                             dict(params=list(role.sketch.parameters()), lr=lr * SKETCH_LR)], weight_decay=0.0)
    scaler = torch.cuda.amp.GradScaler()
    sched = DDPMScheduler.from_pretrained(BASE_ID, subfolder="scheduler")
    T = sched.config.num_train_timesteps
    P.unet.enable_gradient_checkpointing(); P.unet.train()
    P.vae.enable_gradient_checkpointing()
    role.install(P.unet); role.train()
    log(f"gate params: {sum(p.numel() for p in params) / 1e6:.3f} M | {len(role.cross)} cross-attention layers, "
        f"{len(role.sketch)} ControlNet residuals | GPU {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    torch.cuda.reset_peak_memory_stats()

    total = 4 if memory_test else phases["total"]
    hist, micro, step, t0 = [], 0, 0, time.time()
    opt.zero_grad(set_to_none=True)
    try:
        while step < total:
            for _ in range(accum):
                micro += 1
                phase = "warm" if step < phases["warm"] else ("conflict" if step < phases["conflict"] else "swap")
                if memory_test:                                          # exercise every kind of step once
                    phase = ["warm", "conflict", "conflict", "swap"][step]
                do_swap = phase == "swap" and hue and (micro % 4 == 0 or memory_test)
                do_attr = phase != "warm" and (micro % 2 == 0 or do_swap or memory_test)
                drop = (not do_attr) and random.random() < 0.10
                if do_swap:
                    h = random.choice(hue); batch = [by_key[h["pair"]], h]
                else:
                    batch = [random.choice(normal if phase == "warm" else allrows)]
                t = torch.randint(200, 601, (1,), device=dev) if do_attr else torch.randint(0, T, (1,), device=dev)
                pred, noise, noisy, coll = forward(P, role, batch, sched, t, drop=drop)
                lat = torch.cat([d["lat"] for d in batch]).to(dev)
                mlat = torch.cat([d["mask_lat"] for d in batch]).to(dev)
                parts = dict(diff=LS.masked_diffusion(pred, noise, mlat))
                prior_w = W["prior"] * max(W["prior_floor"], 1 - step / phases["conflict"])
                if prior_w > 0 and not drop:
                    m = torch.cat([d["mask256"] for d in batch]).to(dev)
                    parts["prior"] = prior_w * LS.prior_loss(coll, m, torch.stack([d["flags_rule"] for d in batch]).to(dev))
                parts["keep"] = W["keep"] * LS.keep_loss(coll).to(dev)
                parts["reg"] = W["reg"] * LS.reg_loss(role, init)
                if do_attr:
                    img = decode_x0(P, sched, noisy, pred, t.repeat(len(batch)))
                    la, ap, emb = attr_losses(P, img, batch)
                    parts["attr"] = W["attr"] * la
                    parts.update({f"a_{k}": v.detach() for k, v in ap.items()})
                    if do_swap:
                        parts["swap"] = W["swap"] * LS.swap_loss(img[:1], img[1:], emb[:1], emb[1:], torch.cat([d["mask256"] for d in batch[:1]]).to(dev))
                loss = sum(v for k, v in parts.items() if not k.startswith("a_"))
                role.ctx.collect = None                                   # recomputation during backward must not re-collect
                scaler.scale(loss / accum).backward()
                hist.append(dict(step=step, micro=micro, phase=phase, kind="+".join(d["kind"] for d in batch), t=int(t), drop=drop,
                                 loss=float(loss), **{k: float(v) for k, v in parts.items()}))
                del pred, noise, noisy, coll, loss
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0 or memory_test:
                rec = hist[-10 * accum:]
                mean = lambda k: np.nanmean([h.get(k, np.nan) for h in rec]) if any(k in h for h in rec) else float("nan")
                log(f"step {step:4d}/{total} [{phase}] loss {mean('loss'):.4f} diff {mean('diff'):.4f} attr {mean('attr'):.4f} "
                    f"swap {mean('swap'):.4f} prior {mean('prior'):.4f} keep {mean('keep'):.4f} | peak GPU {torch.cuda.max_memory_allocated() / 1e9:.2f} GB | "
                    f"{(time.time() - t0) / 60:.1f} min")
            if not memory_test and step % save_every == 0:
                role.save(os.path.join(out_dir, f"role_gates_step{step:04d}.pt"), dict(step=step))
                json.dump(hist, open(os.path.join(out_dir, "train_log.json"), "w"))
    finally:
        role.remove(P.unet); role.ctx.collect = None
        P.unet.disable_gradient_checkpointing(); P.unet.eval(); P.vae.disable_gradient_checkpointing()
    role.eval()
    if memory_test:
        log(f"MEMORY TEST OK at res={res}: peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    else:
        cfg = dict(version=VERSION, sketch_lr=SKETCH_LR, res=res, lr=lr, accum=accum, phases=phases, weights=W, attr_weights=ATTR_W, n_samples=len(data),
                   kinds={k: sum(d["kind"] == k for d in data) for k in ("normal", "col", "mat", "hue")}, minutes=(time.time() - t0) / 60)
        role.save(os.path.join(out_dir, "role_gates.pt"), dict(step=step, train_cfg=cfg))
        json.dump(cfg, open(os.path.join(out_dir, "role_gates_config.json"), "w"), indent=1)
        json.dump(hist, open(os.path.join(out_dir, "train_log.json"), "w"))
        log(f"saved {out_dir}/role_gates.pt")
    gc.collect(); torch.cuda.empty_cache()
    return P, role, hist


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder that contains metadata.jsonl (the unzipped pilot_v1)")
    ap.add_argument("--gov", required=True, help="governance.pt from notebook 05")
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--memory_test", action="store_true")
    a = ap.parse_args()
    train(a.data, a.gov, a.out, res=a.res, lr=a.lr, accum=a.accum, memory_test=a.memory_test)
