"""
Generation pipeline: frozen SDXL + scribble ControlNet + IP-Adapter with a custom Euler loop.

Lifted from app/app.py and notebook 04 (`run_item`) so that the notebooks and the interface share one
implementation. Two ways to drive it:

  legacy arms   run(item, p, shaped=False/True)            fixed / scheduled / router, exactly as before
  governed      run(item, p, gates={'g_obj','g_col','g_mat'})

What the gates change (each mechanism is taken from published work, see README):
  reference   the IP-Adapter image embedding is split into a colour part (E(ref) - E(grey ref)) and a texture
              part (E(grey ref)) as in SADis; a part is injected only if the reference governs that attribute,
              and only into the style block of the UNet as in InstantStyle
  sketch      the ControlNet scale is relaxed after a (shortened) hold time only in proportion to g_obj,
              instead of always, following the idea of SmartControl (gate-driven scalar, not a learned map)
  text        unchanged: an attribute the caption states stays in the prompt
"""
import gc
import hashlib
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from governance.features import ClipL, build_x, letterbox, object_phrase, on_white, ref_for_clip, sketch_for_clip, to_grey
from rolegate.tokens import neutral_caption, token_groups

BASE_ID = "stabilityai/stable-diffusion-xl-base-1.0"
VAE_ID = "madebyollin/sdxl-vae-fp16-fix"
CN_ID = "xinsir/controlnet-scribble-sdxl-1.0"
IP_REPO, IP_SUBFOLDER, IP_WEIGHT, ENC_SUBFOLDER = "h94/IP-Adapter", "sdxl_models", "ip-adapter_sdxl_vit-h.safetensors", "models/image_encoder"
NEG = "blurry, low quality, text, watermark, multiple objects, cropped"

CENTER = dict(cs=1.0, hold=0.7, ip=0.5, cfg=7.0)
RELAX = 0.3            # sketch strength multiplier after the hold time (value used in every reported run)
HOLD_CONFLICT = 0.4    # hold time when the object gate is fully on (lowest level of the old router grid)
GUIDE_W, GUIDE_START, GUIDE_END, GUIDE_EVERY = 0.25, 0.25, 0.75, 2


def seed_for(s):
    return int(hashlib.md5(str(s).encode()).hexdigest()[:8], 16)


def prep_sketch(im, res):
    """Any sketch -> white lines on black, letterboxed (what the scribble ControlNet expects)."""
    g = on_white(im).convert("L")
    if np.array(g).mean() > 127:
        g = ImageOps.invert(g)
    return letterbox(g.convert("RGB"), res, (0, 0, 0))


def prep_ref(im, res):
    return letterbox(on_white(im), res, (127, 127, 127))


def cs_shape(f, hold, relax=RELAX):
    return 1.0 if f < hold else relax


def ip_shape(f):
    return 0.0 if f < 0.15 else min(1.0, (f - 0.15) / 0.25)


def style_block(s):
    """IP-Adapter scale on the style block of SDXL only (InstantStyle)."""
    return {"up": {"block_0": [0.0, float(s), 0.0]}}


def compose_ip(full, grey, flat, g_col, g_mat):
    """Reference embedding with only the governed parts.

    texture part = E(grey ref), colour part = E(ref) - E(grey ref)  (SADis, without its SVD rescaling).
    g = 0 keeps a part, g = 1 removes it. When the texture is removed, the colour part is added to the
    embedding of a flat grey image instead, so the projector still receives an in-distribution vector.
    Both gates 0 returns `full` exactly.
    """
    g_col, g_mat = float(g_col), float(g_mat)
    return grey * (1 - g_mat) + flat * g_mat + (full - grey) * (1 - g_col)


def sketch_schedule(f, cs, hold, g_obj):
    """ControlNet scale at step fraction f. No object conflict -> the sketch stays at full strength."""
    hold_eff = hold - g_obj * max(hold - HOLD_CONFLICT, 0.0)
    return cs if f < hold_eff else cs * (1 - g_obj * (1 - RELAX))


class Pipeline:
    def __init__(self, res=640, steps=20, device="cuda", lora=None, clip=None, log=print):
        from diffusers import (AutoencoderKL, ControlNetModel, EulerDiscreteScheduler,
                               StableDiffusionXLControlNetPipeline, UNet2DConditionModel)
        from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer, CLIPVisionModelWithProjection
        self.res, self.steps, self.dev, self.dt = res, steps, device, torch.float16
        dt = self.dt
        self.vae = AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=dt).to(device)
        self.unet = UNet2DConditionModel.from_pretrained(BASE_ID, subfolder="unet", torch_dtype=dt, variant="fp16").to(device)
        self.controlnet = ControlNetModel.from_pretrained(CN_ID, torch_dtype=dt).to(device)
        te1 = CLIPTextModel.from_pretrained(BASE_ID, subfolder="text_encoder", torch_dtype=dt, variant="fp16")
        te2 = CLIPTextModelWithProjection.from_pretrained(BASE_ID, subfolder="text_encoder_2", torch_dtype=dt, variant="fp16")
        enc = CLIPVisionModelWithProjection.from_pretrained(IP_REPO, subfolder=ENC_SUBFOLDER, torch_dtype=dt)
        self.pipe = StableDiffusionXLControlNetPipeline(
            vae=self.vae, text_encoder=te1, text_encoder_2=te2,
            tokenizer=CLIPTokenizer.from_pretrained(BASE_ID, subfolder="tokenizer"),
            tokenizer_2=CLIPTokenizer.from_pretrained(BASE_ID, subfolder="tokenizer_2"),
            unet=self.unet, controlnet=self.controlnet,
            scheduler=EulerDiscreteScheduler.from_pretrained(BASE_ID, subfolder="scheduler"), image_encoder=enc)
        self.pipe.load_ip_adapter(IP_REPO, subfolder=IP_SUBFOLDER, weight_name=IP_WEIGHT)
        self.pipe.set_progress_bar_config(disable=True)
        self.has_lora = False
        if lora and os.path.exists(lora):
            self.pipe.load_lora_weights(os.path.dirname(lora) or ".", weight_name=os.path.basename(lora))
            self.has_lora = True
            log(f"fusion LoRA loaded: {lora}")
        for m in (self.vae, self.unet, self.controlnet, te1, te2, enc):
            m.requires_grad_(False); m.eval()
        self._enc = (te1, te2, enc)                       # kept on the CPU, moved to the GPU only while encoding
        self.pipe.text_encoder = self.pipe.text_encoder_2 = self.pipe.image_encoder = None
        self.pipe.vae.enable_slicing()
        self.clip = None if clip is False else (clip or ClipL(device))     # clip=False: training only, no CLIP-L needed
        self.GL = res // 16; self.GH = self.GL * 8
        self.flat_ip = self.encode(images=[Image.new("RGB", (res, res), (127, 127, 127))])[1][0]
        log(f"pipeline ready | GPU {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # ------------------------------------------------------------------ encoding
    @torch.no_grad()
    def encode(self, captions=(), images=()):
        """SDXL text embeddings and IP-Adapter image embeddings, with the encoders on the GPU only for this call.

        Returns ([ (prompt, negative, pooled, negative pooled) ... ], [ positive image embedding (1,1,1024) ... ]), on CPU.
        """
        p = self.pipe
        p.text_encoder, p.text_encoder_2, p.image_encoder = self._enc
        for m in self._enc:
            m.to(self.dev)
        try:
            txt = [[t.cpu() for t in p.encode_prompt(prompt=c, device=self.dev, num_images_per_prompt=1,
                                                     do_classifier_free_guidance=True, negative_prompt=NEG)] for c in captions]
            ips = [p.prepare_ip_adapter_image_embeds(ip_adapter_image=im, ip_adapter_image_embeds=None, device=self.dev,
                                                     num_images_per_prompt=1, do_classifier_free_guidance=True)[0][1:2].cpu()
                   for im in images]
        finally:
            for m in self._enc:
                m.to("cpu")
            p.text_encoder = p.text_encoder_2 = p.image_encoder = None
            gc.collect(); torch.cuda.empty_cache()
        return txt, ips

    def use_lora(self, on):
        if self.has_lora:
            (self.pipe.enable_lora if on else self.pipe.disable_lora)()

    # ------------------------------------------------------------------ items
    def _soft_sketch(self, sk, size):
        t = torch.from_numpy(np.array(sk.convert("L"))).float()[None, None].to(self.dev) / 255
        t = F.interpolate(t, size=(size, size), mode="area")
        return F.max_pool2d((t > 0.35 * t.max()).float(), 9, 1, 4)

    def make_items(self, specs):
        """specs: dicts with key, sketch (PIL), ref (PIL), caption, obj (object name), seed, and optionally attr.

        All captions and references are encoded in one pass. Each item gets everything `run` needs plus `x`,
        the governance-network input.
        """
        tok = self.pipe.tokenizer
        neu = {c: {w: neutral_caption(tok, c, w) for w in ("colour", "material")} for c in dict.fromkeys(s["caption"] for s in specs)}
        caps = list(dict.fromkeys([s["caption"] for s in specs] + [n for d in neu.values() for n in d.values() if n]))
        refs = [prep_ref(s["ref"], self.res) for s in specs]
        txt, ips = self.encode(caps, refs + [to_grey(r) for r in refs])
        txt = dict(zip(caps, txt))
        items = []
        for i, s in enumerate(specs):
            sk = prep_sketch(s["sketch"], self.res)
            obj = (s.get("obj") or "").strip() or s["caption"]
            cap_e, obj_e = self.clip.text([s["caption"], object_phrase(obj)])
            sk_e = self.clip.image_pil(sketch_for_clip(s["sketch"]))
            ref_c = ref_for_clip(s["ref"])
            ref_e, grey_e = self.clip.image_pil(ref_c), self.clip.image_pil(to_grey(ref_c))
            it = dict(key=s.get("key", "ui"), caption=s["caption"], obj=obj, seed=int(s.get("seed", 0)),
                      sketch=sk, ref=refs[i], txt=txt[s["caption"]], ip_full=ips[i], ip_grey=ips[len(specs) + i],
                      x=build_x(cap_e, obj_e, sk_e, ref_e, grey_e),
                      clip=dict(cap=cap_e, obj=obj_e, sketch=sk_e, ref=ref_e), S_gh=self._soft_sketch(sk, self.GH))
            if s.get("attr"):
                it["attr"], it["attr_emb"] = s["attr"], self.clip.text([s["attr"]])[0]
            it["ref_object"] = s.get("ref_object") or "object"
            it["text_groups"] = token_groups(tok, s["caption"], obj)
            # caption with colour / material words neutralised: (prompt embeds, pooled) or None
            it["txt_neutral"] = {w: (txt[n][0], txt[n][2]) if n else None for w, n in neu[s["caption"]].items()}
            items.append(it)
        return items

    # ------------------------------------------------------------------ role-gated model: reference channels
    OBJ_TEXT_ID = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"          # text tower of IP-Adapter's image encoder

    @torch.no_grad()
    def prepare_object_dirs(self, names):
        """Unit text directions of 'a photo of a {name}' in IP-Adapter's ViT-H embedding space (InstantStyle's
        content subtraction). Loaded once, then the text tower is freed."""
        names = [n for n in dict.fromkeys(names) if n not in getattr(self, "obj_dirs", {})]
        if not hasattr(self, "obj_dirs"):
            self.obj_dirs = {}
        if not names:
            return self.obj_dirs
        from transformers import CLIPTextModelWithProjection, CLIPTokenizer
        tok = CLIPTokenizer.from_pretrained(self.OBJ_TEXT_ID)
        txt = CLIPTextModelWithProjection.from_pretrained(self.OBJ_TEXT_ID, torch_dtype=self.dt).to(self.dev).eval()
        for i in range(0, len(names), 64):
            chunk = names[i:i + 64]
            t = tok([object_phrase(n) for n in chunk], padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(self.dev)
            e = F.normalize(txt(**t).text_embeds.float(), dim=-1).cpu()
            self.obj_dirs.update(zip(chunk, e))
        del txt; gc.collect(); torch.cuda.empty_cache()
        return self.obj_dirs

    def ref_channels(self, it):
        """(1, 2, 1024): [colour channel, texture channel] of the reference.

        colour  = E(flat grey) + E(ref) - E(grey ref)          colour on a neutral base (SADis)
        texture = E(grey ref) minus its component along the reference object's text direction (InstantStyle)
        """
        full, grey, flat = it["ip_full"].float().view(-1), it["ip_grey"].float().view(-1), self.flat_ip.float().view(-1)
        d = getattr(self, "obj_dirs", {}).get(it.get("ref_object", ""))
        col = flat + (full - grey)
        tex = grey - (grey @ d) * d if d is not None else grey
        return torch.stack([col, tex])[None]

    @torch.no_grad()
    def run_role(self, it, role, flags, p=None, knock=(), freeze_text=False, record_at=None, seed_off=0):
        """Generate with the role gates installed. flags = (object, colour, material) conflict scores in [0,1].

        The base schedule is the existing 'scheduled' arm (centre strengths, geometry first, appearance later);
        the gates then decide per layer, per pixel and per attribute how much of each input acts.
        Returns (image, gate record or None).
        """
        p = dict(CENTER, **(p or {}))
        pipe, dev, dt, res = self.pipe, self.dev, self.dt, self.res
        pe, npe, ppe, nppe = [t.to(dev) for t in it["txt"]]
        pr, pooled = torch.cat([npe, pe]), torch.cat([nppe, ppe])
        tids = torch.tensor([[res, res, 0, 0, res, res]], device=dev, dtype=dt).repeat(2, 1)
        emb = self.ref_channels(it)
        ipe = [torch.cat([torch.zeros_like(emb), emb]).to(dev, dt)]
        groups = torch.tensor([[0] * 77, list(it["text_groups"])], device=dev, dtype=torch.long)
        fl = torch.tensor([list(flags)] * 2, device=dev, dtype=torch.float32)
        tn, pn = {}, {}
        for w, v in (it.get("txt_neutral") or {}).items():               # unconditional half: negative prompt itself
            tn[w] = None if v is None else torch.cat([npe, v[0].to(dev)])
            pn[w] = None if v is None else torch.cat([nppe, v[1].to(dev)])
        ctrl = pipe.prepare_image(image=it["sketch"], width=res, height=res, batch_size=1, num_images_per_prompt=1,
                                  device=dev, dtype=dt, do_classifier_free_guidance=True, guess_mode=False)
        pipe.scheduler.set_timesteps(self.steps, device=dev)
        ts = pipe.scheduler.timesteps
        gen = torch.Generator("cpu").manual_seed(it["seed"] + seed_off)
        lat = torch.randn((1, 4, res // 8, res // 8), generator=gen).to(dev, dt) * pipe.scheduler.init_noise_sigma
        added = {"text_embeds": pooled, "time_ids": tids}
        record = None
        role.to(dev).eval()
        role.install(self.unet)
        try:
            for i, t in enumerate(ts):
                f = i / len(ts)
                cs, ips = p["cs"] * cs_shape(f, p["hold"]), p["ip"] * ip_shape(f)
                role.set_context(fl, t.view(1).repeat(2).to(dev), groups, ips, knock, freeze_text, text_neutral=tn)
                role.ctx.record = {} if record_at is not None and i == record_at else None
                added = {"text_embeds": role.mix_pooled(pooled, pn.get("colour"), pn.get("material")), "time_ids": tids}
                li = pipe.scheduler.scale_model_input(torch.cat([lat] * 2), t)
                down, mid = pipe.controlnet(li, t, encoder_hidden_states=pr, controlnet_cond=ctrl, conditioning_scale=float(cs),
                                            guess_mode=False, added_cond_kwargs=added, return_dict=False)
                down, mid = role.gate_controlnet(down, mid)
                eps = self.unet(li, t, encoder_hidden_states=pr, added_cond_kwargs={**added, "image_embeds": ipe},
                                down_block_additional_residuals=down, mid_block_additional_residual=mid, return_dict=False)[0]
                if role.ctx.record is not None:
                    record = role.ctx.record
                eu, ec = eps.chunk(2)
                eps = eu + p["cfg"] * (ec - eu)
                lat = pipe.scheduler.step(eps, t, lat, return_dict=False)[0]
        finally:
            role.remove(self.unet)
            role.ctx.record = None
        img = self.vae.decode(lat / self.vae.config.scaling_factor, return_dict=False)[0]
        return pipe.image_processor.postprocess(img, output_type="pil")[0], record

    def controlnet_res_channels(self):
        return [m.out_channels for m in self.controlnet.controlnet_down_blocks] + [self.controlnet.controlnet_mid_block.out_channels]

    # ------------------------------------------------------------------ old constraint guidance (baseline arm only)
    def _guidance_delta(self, x0, it, lam_geo, lam_app):
        x0g = x0.detach().float().requires_grad_(True)
        with torch.enable_grad():
            z = F.interpolate(x0g, size=(self.GL, self.GL), mode="area")
            img = self.vae.decode((z / self.vae.config.scaling_factor).to(self.dt), return_dict=False)[0]
            x = ((img.float() + 1) / 2).clamp(0, 1)
            g = x.mean(1, keepdim=True)
            kx = torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]], device=x.device).view(1, 1, 3, 3)
            E = (F.conv2d(g, kx, padding=1) ** 2 + F.conv2d(g, kx.transpose(2, 3), padding=1) ** 2 + 1e-6).sqrt()
            E = E / (E.amax() + 1e-6)
            S = it["S_gh"]
            l_geo = (1 - (E * S).sum() / (S.sum() + 1e-6)) + 0.5 * (E * (1 - S)).mean()
            l_app = 1 - (self.clip.image(x) * it["attr_emb"].to(x.device)).sum()
            grad, = torch.autograd.grad(lam_geo * l_geo + lam_app * l_app, x0g)
        return grad / (grad.abs().mean() + 1e-8)

    # ------------------------------------------------------------------ generation
    @torch.no_grad()
    def run(self, it, p=None, shaped=True, gates=None, split=True, style_only=None, guided=False, seed_off=0):
        """Generate one image.

        gates=None  legacy behaviour: shaped=False is the 'fixed' arm, shaped=True the 'scheduled' / 'router' arms.
        gates=dict  governed behaviour (see module docstring). split=False keeps the full reference embedding
                    (ablation); style_only defaults to True when gates are given.
        """
        p = dict(CENTER, **(p or {}))
        pipe, dev, dt, res = self.pipe, self.dev, self.dt, self.res
        pe, npe, ppe, nppe = [t.to(dev) for t in it["txt"]]
        pr, pooled = torch.cat([npe, pe]), torch.cat([nppe, ppe])
        tids = torch.tensor([[res, res, 0, 0, res, res]], device=dev, dtype=dt).repeat(2, 1)
        governed = gates is not None
        style_only = governed if style_only is None else style_only
        if governed and split:
            pos = compose_ip(it["ip_full"], it["ip_grey"], self.flat_ip, gates["g_col"], gates["g_mat"])
            ip_amp = p["ip"] * (1 - float(gates["g_col"]) * float(gates["g_mat"]))     # nothing left to take -> off
        else:
            pos, ip_amp = it["ip_full"], p["ip"]
        ipe = [torch.cat([torch.zeros_like(pos), pos]).to(dev, dt)]
        ctrl = pipe.prepare_image(image=it["sketch"], width=res, height=res, batch_size=1, num_images_per_prompt=1,
                                  device=dev, dtype=dt, do_classifier_free_guidance=True, guess_mode=False)
        pipe.scheduler.set_timesteps(self.steps, device=dev)
        ts = pipe.scheduler.timesteps
        gen = torch.Generator("cpu").manual_seed(it["seed"] + seed_off)
        lat = torch.randn((1, 4, res // 8, res // 8), generator=gen).to(dev, dt) * pipe.scheduler.init_noise_sigma
        added = {"text_embeds": pooled, "time_ids": tids}
        for i, t in enumerate(ts):
            f = i / len(ts)
            if governed:
                cs, ips = sketch_schedule(f, p["cs"], p["hold"], float(gates["g_obj"])), ip_amp * ip_shape(f)
            else:
                cs = p["cs"] * (cs_shape(f, p["hold"]) if shaped else 1.0)
                ips = ip_amp * (ip_shape(f) if shaped else 1.0)
            li = pipe.scheduler.scale_model_input(torch.cat([lat] * 2), t)
            pipe.set_ip_adapter_scale(style_block(ips) if style_only else float(ips))
            kw = {}
            if cs > 1e-3:
                down, mid = pipe.controlnet(li, t, encoder_hidden_states=pr, controlnet_cond=ctrl, conditioning_scale=float(cs),
                                            guess_mode=False, added_cond_kwargs=added, return_dict=False)
                kw = dict(down_block_additional_residuals=down, mid_block_additional_residual=mid)
            eps = pipe.unet(li, t, encoder_hidden_states=pr, added_cond_kwargs={**added, "image_embeds": ipe},
                            return_dict=False, **kw)[0]
            eu, ec = eps.chunk(2)
            eps = eu + p["cfg"] * (ec - eu)
            if guided and "attr_emb" in it and GUIDE_START <= f <= GUIDE_END and i % GUIDE_EVERY == 0:
                x0 = lat - pipe.scheduler.sigmas[i].to(dev) * eps
                lam_geo = float(np.clip(p["cs"] / CENTER["cs"], 0.3, 1.5)); lam_app = float(np.clip(p["ip"] / CENTER["ip"], 0.3, 1.6))
                lat = lat - GUIDE_W * self._guidance_delta(x0, it, lam_geo, lam_app).to(dt)
            lat = pipe.scheduler.step(eps, t, lat, return_dict=False)[0]
        img = self.vae.decode(lat / self.vae.config.scaling_factor, return_dict=False)[0]
        return pipe.image_processor.postprocess(img, output_type="pil")[0]
