"""
M1 forward passes. One function, three modes, all on the SAME frozen SDXL + ControlNet + IP-Adapter + role gates:

    base      caption as written. Text colour/material gates frozen at 1, no neutralisation.   == Pipeline.run_role(it, gates, [0,0,0])
    teacher   the rule: colour and material words neutral in the UNet text path, the pooled     == Pipeline.run_role(it, gates, [0,1,1],
              embedding and the ControlNet caption.                                                knock=("text_colour","text_material"), cn_text="mix")
    student   learned gates g(s) with s = presence of a reference. The ControlNet caption is the full caption minus
              alpha * (full - neutral), alpha = clamp(1 - g_pooled, 0, 1), computed without gradient.

`generate` is a copy of Pipeline.run_role's loop that calls `predict`, so that checking generate(teacher) against run_role verifies
`predict` against the real implementation.
"""
from contextlib import contextmanager

import numpy as np
import torch


def presence(emb):
    """Presence of a reference, from the input itself: 1 if the IP embedding is non-zero (a real reference), else 0."""
    return float(emb.abs().sum() > 0)


@contextmanager
def installed(P, role):
    role.to(P.dev).eval()
    role.install(P.unet)
    try:
        yield
    finally:
        role.remove(P.unet)
        role.ctx.record = None


def make_ctx(P, it, with_ref=True):
    """Everything about one item that the forward needs, on the device, un-batched."""
    dev, dt = P.dev, P.dt
    pe, npe, ppe, nppe = [t.to(dev) for t in it["txt"]]
    neu = {w: (None if v is None else (v[0].to(dev), v[1].to(dev))) for w, v in (it.get("txt_neutral") or {}).items()}
    emb = P.ref_channels(it).to(dev, dt)
    if not with_ref:
        emb = torch.zeros_like(emb)
    ctrl = P.pipe.prepare_image(image=it["sketch"], width=P.res, height=P.res, batch_size=1, num_images_per_prompt=1, device=dev, dtype=dt,
                                do_classifier_free_guidance=False, guess_mode=False)
    return dict(pe=pe, npe=npe, ppe=ppe, nppe=nppe, neu=neu, emb=emb, ctrl=ctrl, groups=torch.tensor([list(it["text_groups"])], device=dev, dtype=torch.long))


def batch(ctx, cfg):
    """Batched tensors. cfg=False: the conditional element only. cfg=True: [unconditional (negative prompt, no reference), conditional]."""
    z = torch.zeros_like(ctx["emb"])
    pres = presence(ctx["emb"])
    out = {}
    for w, key in (("colour", "col"), ("material", "mat")):
        v = ctx["neu"].get(w)
        if cfg:
            out["tn_" + key] = None if v is None else torch.cat([ctx["npe"], v[0]])
            out["pn_" + key] = None if v is None else torch.cat([ctx["nppe"], v[1]])
        else:
            out["tn_" + key] = None if v is None else v[0]
            out["pn_" + key] = None if v is None else v[1]
    if cfg:
        out.update(pe=torch.cat([ctx["npe"], ctx["pe"]]), pooled=torch.cat([ctx["nppe"], ctx["ppe"]]),
                   groups=torch.cat([torch.zeros_like(ctx["groups"]), ctx["groups"]]), emb=torch.cat([z, ctx["emb"]]),
                   ctrl=torch.cat([ctx["ctrl"]] * 2), s=torch.tensor([0.0, pres], device=ctx["emb"].device))
    else:
        out.update(pe=ctx["pe"], pooled=ctx["ppe"], groups=ctx["groups"], emb=ctx["emb"], ctrl=ctx["ctrl"], s=torch.tensor([pres], device=ctx["emb"].device))
    return out


def predict(P, role, c, x, t, f, mode, ip=None, autocast=False):
    """Noise prediction for the batched context `c` at model input `x`, timestep t (a 0-dim tensor), progress f."""
    from pipeline import CENTER, cs_shape, ip_shape
    dev, dt, res = P.dev, P.dt, P.res
    B = x.shape[0]
    cs = CENTER["cs"] * cs_shape(f, CENTER["hold"])
    ips = (CENTER["ip"] if ip is None else ip) * ip_shape(f)
    s = c["s"].float()
    zeros, ones = torch.zeros_like(s), torch.ones_like(s)
    if mode == "student":
        flags, knock, freeze = torch.stack([zeros, s, s], 1), (), False
    elif mode == "teacher":
        flags, knock, freeze = torch.stack([zeros, ones, ones], 1), ("text_colour", "text_material"), False
    elif mode == "base":
        flags, knock, freeze = torch.stack([zeros, zeros, zeros], 1), (), True
    else:
        raise ValueError(mode)
    tt = t.view(1).repeat(B).to(dev)
    tn = {"colour": c["tn_col"], "material": c["tn_mat"]}
    role.set_context(flags.to(dev), tt, c["groups"], ips, knock, freeze, text_neutral=tn)
    role.ctx.collect = None; role.ctx.record = None
    pooled = role.mix_pooled(c["pooled"], c["pn_col"], c["pn_mat"])
    if mode == "student":
        alpha = (1 - role.pooled(role.ctx.cond.to(dev))).clamp(0, 1).detach()           # (B, 2): colour, material; no gradient
    elif mode == "teacher":
        alpha = torch.ones(B, 2, device=dev)
    else:
        alpha = torch.zeros(B, 2, device=dev)
    pe_cn = c["pe"]
    for k, n in ((0, c["tn_col"]), (1, c["tn_mat"])):
        if n is not None:
            pe_cn = pe_cn - alpha[:, k].view(B, 1, 1).to(pe_cn.dtype) * (c["pe"] - n)
    tids = torch.tensor([[res, res, 0, 0, res, res]], device=dev, dtype=dt).repeat(B, 1)
    added = {"text_embeds": pooled, "time_ids": tids}
    with torch.autocast("cuda", dtype=torch.float16, enabled=autocast):
        with torch.no_grad():
            down, mid = P.controlnet(x, tt, encoder_hidden_states=pe_cn, controlnet_cond=c["ctrl"], conditioning_scale=float(cs),
                                     guess_mode=False, added_cond_kwargs=added, return_dict=False)
        down, mid = role.gate_controlnet(down, mid)
        eps = P.unet(x, tt, encoder_hidden_states=c["pe"], added_cond_kwargs={**added, "image_embeds": [c["emb"]]},
                     down_block_additional_residuals=down, mid_block_additional_residual=mid, return_dict=False)[0]
    return eps


@torch.no_grad()
def generate(P, role, it, mode, with_ref=True, ip=None, seed_off=0):
    """Copy of Pipeline.run_role's sampling loop (classifier-free guidance) built on `predict`. Returns a PIL image."""
    from pipeline import CENTER
    pipe, dev, dt, res = P.pipe, P.dev, P.dt, P.res
    c = batch(make_ctx(P, it, with_ref), cfg=True)
    pipe.scheduler.set_timesteps(P.steps, device=dev)
    ts = pipe.scheduler.timesteps
    gen = torch.Generator("cpu").manual_seed(it["seed"] + seed_off)
    lat = torch.randn((1, 4, res // 8, res // 8), generator=gen).to(dev, dt) * pipe.scheduler.init_noise_sigma
    with installed(P, role):
        for i, t in enumerate(ts):
            li = pipe.scheduler.scale_model_input(torch.cat([lat] * 2), t)
            eps = predict(P, role, c, li, t, i / len(ts), mode, ip=ip)
            eu, ec = eps.chunk(2)
            lat = pipe.scheduler.step(eu + CENTER["cfg"] * (ec - eu), t, lat, return_dict=False)[0]
    img = P.vae.decode(lat / P.vae.config.scaling_factor, return_dict=False)[0]
    return pipe.image_processor.postprocess(img, output_type="pil")[0]


def encode_target(P, row, data_dir):
    """VAE latent (mean) of the row's real photo at the current resolution."""
    import os
    from PIL import Image
    im = Image.open(os.path.join(data_dir, row["target"])).convert("RGB").resize((P.res, P.res), Image.LANCZOS)
    x = torch.from_numpy(np.array(im)).permute(2, 0, 1)[None].float().div(127.5).sub(1).to(P.dev, P.dt)
    with torch.no_grad():
        return P.vae.encode(x).latent_dist.mean * P.vae.config.scaling_factor


def noisy_input(sched, z0, t, seed):
    g = torch.Generator("cpu").manual_seed(int(seed))
    noise = torch.randn(z0.shape, generator=g).to(z0.device, z0.dtype)
    tt = torch.tensor([int(t)], device=z0.device)
    return sched.add_noise(z0, noise, tt), noise
