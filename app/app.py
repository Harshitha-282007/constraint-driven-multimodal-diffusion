"""
Attribute-routed multimodal diffusion (SDXL + scribble ControlNet + IP-Adapter) -- standalone test interface
Team One Sheep Army

Extracted from notebook 04_attribute_router so the interface uses the SAME model, the SAME conflict features,
the SAME router (router.pt) and the SAME denoising loop as the experiments.

Colab T4:
    !pip uninstall -y torchao -q
    !pip install -q -U diffusers transformers accelerate safetensors gradio psutil
    # put router.pt (from your router_v3 run) next to this file, then:
    !python app.py
Inputs : sketch, reference image, caption (+ optional object name / attribute phrase)
Methods: fixed / scheduled / router / router + constraint guidance (or compare all 4)
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch, json, glob, zipfile, hashlib, time, gc, psutil, random
import numpy as np, pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
ROUTER_PATH = "router.pt"

# ================= config =================

# ---- CONFIG ----
PROFILE = "fast"          # "fast" (~1 h) or "full" (~2 h). Same images are reused when you switch.
PROFILES = {
    "fast": dict(N_TRAIN=dict(normal=6,  sem=4, color=2, material=2, app=4), N_EVAL=dict(normal=8,  sem=4, color=2, material=2, app=4), EVAL_SEEDS=[0]),
    "full": dict(N_TRAIN=dict(normal=10, sem=6, color=3, material=3, app=6), N_EVAL=dict(normal=12, sem=6, color=3, material=3, app=6), EVAL_SEEDS=[0, 1]),
}
DRIVE_ROOT = "/content/drive/MyDrive/sketchy_dataset_one_sketch"
ZIP_PATH = f"{DRIVE_ROOT}/pilot_v1.zip"
LOCAL_DATA = "/content/pilot_v1"
CONFLICT_FILE = f"{DRIVE_ROOT}/metadata_with_conflicts.jsonl"
RUN = "router_v2"
OUT = f"{DRIVE_ROOT}/sdxl_runs/{RUN}"

RES, STEPS = 640, 20
NEG = "blurry, low quality, text, watermark, multiple objects, cropped"
KIND_OF = {"object_text_sketch": "sem", "color_text_ref": "color", "material_text_ref": "material", "appearance_ref_swap": "app"}
GROUP = dict(normal="normal", sem="object_conflict", color="attribute_conflict", material="attribute_conflict", app="ref_swap_conflict")

# the four routed factors, each tied to one attribute / modality
#   cs   : sketch strength            -> geometry (sketch)
#   hold : fraction of steps the sketch stays strong -> shape / layout persistence (sketch)
#   ip   : reference strength         -> appearance (reference)
#   cfg  : text guidance strength     -> semantics (text)
LEVELS = dict(cs=[0.5, 1.0, 1.4], hold=[0.4, 0.7, 1.0], ip=[0.2, 0.5, 0.8], cfg=[4.0, 7.0, 10.0])
CENTER = dict(cs=1.0, hold=0.7, ip=0.5, cfg=7.0)
FACTORS = list(LEVELS)
RANGES = {k: (min(v), max(v)) for k, v in LEVELS.items()}
RELAX = 0.3                                    # sketch strength multiplier after the hold time
W = dict(geo=1.0, app=1.0, sem=1.0)           # composite utility weights used for the grid targets
SOFT_TEMP = 0.5

GUIDE_W, GUIDE_START, GUIDE_END, GUIDE_EVERY = 0.25, 0.25, 0.75, 2
LAMBDA_GEO, LAMBDA_APP = 1.0, 1.0

BASE_ID = "stabilityai/stable-diffusion-xl-base-1.0"
VAE_ID = "madebyollin/sdxl-vae-fp16-fix"
CN_ID = "xinsir/controlnet-scribble-sdxl-1.0"
IP_REPO, IP_SUBFOLDER, IP_WEIGHT, ENC_SUBFOLDER = "h94/IP-Adapter", "sdxl_models", "ip-adapter_sdxl_vit-h.safetensors", "models/image_encoder"
CLIP_ID, DINO_ID = "openai/clip-vit-large-patch14", "facebook/dinov2-small"      # used for targets + guidance
CLIPB_ID = "openai/clip-vit-base-patch32"                                       # held-out evaluation only

# ================= image helpers =================

def letterbox(im, size, fill):
    im = ImageOps.contain(im, (size, size), Image.LANCZOS)
    c = Image.new("RGB", (size, size), fill)
    c.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    return c

def prep_ref(path):
    return letterbox(Image.open(path).convert("RGB"), RES, (127, 127, 127))



# ================= models =================

# ---- load models (each straight onto the GPU so system RAM stays low) ----
dt = torch.float16
from diffusers import (UNet2DConditionModel, ControlNetModel, AutoencoderKL,
                       StableDiffusionXLControlNetPipeline, EulerDiscreteScheduler)
from transformers import (CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer,
                          CLIPVisionModelWithProjection, AutoModel)
def ram(): return round(psutil.virtual_memory().used / 1e9, 1)
def gpu(): return round(torch.cuda.memory_allocated() / 1e9, 2)
def to_gpu(m, name):
    m = m.to("cuda"); gc.collect()
    print(f"{name:10s} | RAM {ram()} GB | GPU {gpu()} GB")
    return m

vae = to_gpu(AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=dt), "vae")
unet = to_gpu(UNet2DConditionModel.from_pretrained(BASE_ID, subfolder="unet", torch_dtype=dt, variant="fp16"), "unet")
controlnet = to_gpu(ControlNetModel.from_pretrained(CN_ID, torch_dtype=dt), "controlnet")
te1 = CLIPTextModel.from_pretrained(BASE_ID, subfolder="text_encoder", torch_dtype=dt, variant="fp16")
te2 = CLIPTextModelWithProjection.from_pretrained(BASE_ID, subfolder="text_encoder_2", torch_dtype=dt, variant="fp16")
enc = CLIPVisionModelWithProjection.from_pretrained(IP_REPO, subfolder=ENC_SUBFOLDER, torch_dtype=dt)
tok1 = CLIPTokenizer.from_pretrained(BASE_ID, subfolder="tokenizer")
tok2 = CLIPTokenizer.from_pretrained(BASE_ID, subfolder="tokenizer_2")
sched = EulerDiscreteScheduler.from_pretrained(BASE_ID, subfolder="scheduler")
pipe = StableDiffusionXLControlNetPipeline(vae=vae, text_encoder=te1, text_encoder_2=te2, tokenizer=tok1, tokenizer_2=tok2,
                                           unet=unet, controlnet=controlnet, scheduler=sched, image_encoder=enc)
pipe.load_ip_adapter(IP_REPO, subfolder=IP_SUBFOLDER, weight_name=IP_WEIGHT)
pipe.set_progress_bar_config(disable=True)

CLIPV = to_gpu(CLIPVisionModelWithProjection.from_pretrained(CLIP_ID, torch_dtype=dt), "clip vis")
CLIPT = to_gpu(CLIPTextModelWithProjection.from_pretrained(CLIP_ID, torch_dtype=dt), "clip txt")
CLIPTOK = CLIPTokenizer.from_pretrained(CLIP_ID)
DINO = to_gpu(AutoModel.from_pretrained(DINO_ID, torch_dtype=dt), "dino")
for m in (vae, unet, controlnet, CLIPV, CLIPT, DINO):
    m.requires_grad_(False); m.eval()
print("all models ready | GPU", gpu(), "GB | RAM", ram(), "GB")

# ================= helpers =================

dev = "cuda"
MEAN_C = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=dev).view(1, 3, 1, 1)
STD_C = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=dev).view(1, 3, 1, 1)
MEAN_I = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
STD_I = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

def to_t(pil, size=None):
    if size: pil = pil.resize((size, size), Image.BICUBIC)
    return torch.from_numpy(np.array(pil.convert("RGB"))).permute(2, 0, 1)[None].float().to(dev) / 255

def clip_img(x):
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    return F.normalize(CLIPV(pixel_values=((x - MEAN_C) / STD_C).to(dt)).image_embeds.float(), dim=-1)

@torch.no_grad()
def clip_txt(strs):
    tk = CLIPTOK(strs, padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(dev)
    return F.normalize(CLIPT(input_ids=tk.input_ids, attention_mask=tk.attention_mask).text_embeds.float(), dim=-1)

@torch.no_grad()
def dino_img(x):
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    return F.normalize(DINO(pixel_values=((x - MEAN_I) / STD_I).to(dt)).last_hidden_state[:, 0].float(), dim=-1)

def edge_map(x):
    g = x.mean(1, keepdim=True)
    kx = torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]], device=x.device).view(1, 1, 3, 3)
    gx, gy = F.conv2d(g, kx, padding=1), F.conv2d(g, kx.transpose(2, 3), padding=1)
    return (gx ** 2 + gy ** 2 + 1e-6).sqrt()

def soft_sketch(sk_pil, size):
    t = torch.from_numpy(np.array(sk_pil.convert("L"))).float()[None, None].to(dev) / 255
    t = F.interpolate(t, size=(size, size), mode="area")
    t = (t > 0.35 * t.max()).float()
    return F.max_pool2d(t, 9, 1, 4)

def thin_np(sk_pil, size):         # undilated sketch mask as numpy bool (for the held-out edge metrics)
    t = torch.from_numpy(np.array(sk_pil.convert("L"))).float()[None, None] / 255
    t = F.interpolate(t, size=(size, size), mode="area")[0, 0].numpy()
    return t > 0.35 * t.max()

def geo_stats(x, S):
    E = edge_map(x); E = E / (E.amax() + 1e-6)
    return (E * S).sum() / (S.sum() + 1e-6), (E * (1 - S)).mean()

GL = RES // 16; GH = GL * 8
BASE = {}


BASE = {}


# ================= encoding + conflict features =================

@torch.no_grad()
def encode_text(p):
    return [t.cpu() for t in pipe.encode_prompt(prompt=p, device=dev, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=NEG)]

@torch.no_grad()
def encode_ip(img):
    return [t.cpu() for t in pipe.prepare_ip_adapter_image_embeds(ip_adapter_image=img, ip_adapter_image_embeds=None, device=dev,
                                                                   num_images_per_prompt=1, do_classifier_free_guidance=True)]

def conflict_features(it, sk_emb, ref_emb):
    cos = lambda a, b: float((a * b).sum())
    cat_emb = clip_txt([f"a photo of a {it['cap_category']}"])
    return [1 - cos(cat_emb, sk_emb),                  # object conflict: text vs sketch
            1 - cos(cat_emb, ref_emb),                 # object conflict: text vs reference
            1 - cos(it["cap_emb"], it["attr_emb"]),    # appearance conflict: text vs reference attributes
            1 - cos(sk_emb, ref_emb),                  # object conflict: sketch vs reference
            cos(it["cap_emb"], sk_emb), cos(it["cap_emb"], ref_emb)]



# ================= denoising loop =================

def cs_shape(f, hold): return 1.0 if f < hold else RELAX
def ip_shape(f): return 0.0 if f < 0.15 else min(1.0, (f - 0.15) / 0.25)

def guidance_delta(x0, b, it, lam_geo, lam_app):
    x0g = x0.detach().float().requires_grad_(True)
    with torch.enable_grad():
        z = F.interpolate(x0g, size=(GL, GL), mode="area")
        img = vae.decode((z / vae.config.scaling_factor).to(dt), return_dict=False)[0]
        x = ((img.float() + 1) / 2).clamp(0, 1)
        rec, out = geo_stats(x, b["S_gh"])
        l_geo = (1 - rec) + 0.5 * out                                  # geometry: edges follow the sketch
        l_app = 1 - (clip_img(x) * it["attr_emb"]).sum()               # appearance: match reference attributes (not its object)
        g, = torch.autograd.grad(LAMBDA_GEO * lam_geo * l_geo + LAMBDA_APP * lam_app * l_app, x0g)
    return g / (g.abs().mean() + 1e-8)

@torch.no_grad()
def run_item(it, p, shaped=True, guided=False, seed_off=0):
    """p = dict(cs, hold, ip, cfg). shaped=False -> constant cs/ip over all steps (the 'fixed' arm)."""
    b = BASE[it["base"]]
    pe, npe, ppe, nppe = [t.to(dev) for t in TXT[it["caption"]]]
    pr, pooled = torch.cat([npe, pe]), torch.cat([nppe, ppe])
    tid = torch.tensor([[RES, RES, 0, 0, RES, RES]], device=dev, dtype=dt)
    tids = torch.cat([tid, tid])
    ipe = [e.to(dev) for e in IPC[it["ref"]]]
    ctrl = pipe.prepare_image(image=b["sketch"], width=RES, height=RES, batch_size=1, num_images_per_prompt=1,
                              device=dev, dtype=dt, do_classifier_free_guidance=True, guess_mode=False)
    pipe.scheduler.set_timesteps(STEPS, device=dev)
    ts = pipe.scheduler.timesteps
    g = torch.Generator("cpu").manual_seed(it["seed"] + seed_off)
    lat = torch.randn((1, 4, RES // 8, RES // 8), generator=g).to(dev, dt) * pipe.scheduler.init_noise_sigma
    lam_geo = float(np.clip(p["cs"] / CENTER["cs"], 0.3, 1.5))        # guidance weights follow the router's modality weights
    lam_app = float(np.clip(p["ip"] / CENTER["ip"], 0.3, 1.6))
    for i, t in enumerate(ts):
        f = i / len(ts)
        cs = p["cs"] * (cs_shape(f, p["hold"]) if shaped else 1.0)
        ips = p["ip"] * (ip_shape(f) if shaped else 1.0)
        li = pipe.scheduler.scale_model_input(torch.cat([lat] * 2), t)
        pipe.set_ip_adapter_scale(float(ips))
        kw = {}
        if cs > 1e-3:
            down, mid = pipe.controlnet(li, t, encoder_hidden_states=pr, controlnet_cond=ctrl, conditioning_scale=float(cs),
                                        guess_mode=False, added_cond_kwargs={"text_embeds": pooled, "time_ids": tids}, return_dict=False)
            kw = dict(down_block_additional_residuals=down, mid_block_additional_residual=mid)
        eps = pipe.unet(li, t, encoder_hidden_states=pr, added_cond_kwargs={"text_embeds": pooled, "time_ids": tids, "image_embeds": ipe},
                        return_dict=False, **kw)[0]
        eu, ec = eps.chunk(2)
        eps = eu + p["cfg"] * (ec - eu)                                 # text (semantics) strength is routed too
        if guided and GUIDE_START <= f <= GUIDE_END and i % GUIDE_EVERY == 0:
            x0 = lat - pipe.scheduler.sigmas[i].to(dev) * eps
            lat = lat - GUIDE_W * guidance_delta(x0, b, it, lam_geo, lam_app).to(dt)
        lat = pipe.scheduler.step(eps, t, lat, return_dict=False)[0]
    img = vae.decode(lat / vae.config.scaling_factor, return_dict=False)[0]
    return pipe.image_processor.postprocess(img, output_type="pil")[0]




ARMS = {"fixed":         dict(src="center", shaped=False, guided=False),
        "scheduled":     dict(src="center", shaped=True,  guided=False),
        "router":        dict(src="router", shaped=True,  guided=False),
        "router_guided": dict(src="router", shaped=True,  guided=True)}
def params_for(it, arm): return dict(CENTER) if ARMS[arm]["src"] == "center" else predict_params(it)



# ---- router (same architecture as notebook 04: 6 conflict features -> 4 outputs, sigmoid, 5-model ensemble) ----
class Router(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, len(FACTORS)), nn.Sigmoid())
    def forward(self, x): return self.net(x)

assert os.path.exists(ROUTER_PATH), "router.pt not found next to app.py (use the one saved by your router_v3 run)"
_ck = torch.load(ROUTER_PATH, map_location="cpu", weights_only=False)
ENS = []
for _sd in _ck["states"]:
    _m = Router(); _m.load_state_dict(_sd); ENS.append(_m.eval())
MU, SD = torch.as_tensor(_ck["mu"]).float(), torch.as_tensor(_ck["sd"]).float()
print("router loaded:", len(ENS), "models | factors", _ck["factors"], "| ranges", _ck["ranges"])
def denorm(u, k): lo, hi = RANGES[k]; return lo + float(u) * (hi - lo)
def predict_params(it):
    with torch.no_grad():
        o = torch.stack([m((torch.tensor([it["feat"]], dtype=torch.float32) - MU) / SD)[0] for m in ENS]).mean(0)
    return {f: denorm(o[j], f) for j, f in enumerate(FACTORS)}

TXT, IPC = {}, {}
BASE = {}


# ================= interface =================

import gradio as gr
for m in (te1, te2, enc): m.requires_grad_(False).eval()

def ui_encode(caption, ref_img):
    pipe.text_encoder, pipe.text_encoder_2, pipe.image_encoder = te1, te2, enc
    for m in (te1, te2, enc): m.to("cuda")
    try:
        txt, ipe = encode_text(caption), encode_ip(ref_img)
    finally:
        for m in (te1, te2, enc): m.to("cpu")
        pipe.text_encoder = pipe.text_encoder_2 = pipe.image_encoder = None
        gc.collect(); torch.cuda.empty_cache()
    return txt, ipe

UI_COLORS = ["red", "blue", "green", "yellow", "orange", "purple", "pink", "black", "white", "brown", "gray", "silver", "gold", "beige"]
UI_MATERIALS = ["wood", "metal", "plastic", "ceramic", "glass", "leather", "fabric", "stone"]
UI_FINISHES = ["glossy", "matte", "shiny metallic", "rough", "polished", "satin"]
@torch.no_grad()
def zero_shot(ref_emb, words, tmpl):
    return words[int((clip_txt([tmpl.format(w) for w in words]) @ ref_emb.T).argmax())]
def on_white(im):
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA"); im = Image.alpha_composite(Image.new("RGBA", im.size, (255, 255, 255, 255)), im)
    return im.convert("RGB")
def prep_sketch_pil(im):
    g = on_white(im).convert("L")
    if np.array(g).mean() > 127: g = ImageOps.invert(g)
    return letterbox(g.convert("RGB"), RES, (0, 0, 0))

@torch.no_grad()
def build_item(sketch_im, ref_im, caption, obj, attr_override, seed):
    sk = prep_sketch_pil(sketch_im); ref = letterbox(on_white(ref_im), RES, (127, 127, 127))
    sk_emb = clip_img(to_t(letterbox(ImageOps.invert(sk.convert("L")).convert("RGB"), 224, (255, 255, 255)))); ref_emb = clip_img(to_t(ref))
    attr = attr_override.strip() or f"a {zero_shot(ref_emb, UI_FINISHES, '{} surface')} {zero_shot(ref_emb, UI_MATERIALS, 'made of {}')} object in {zero_shot(ref_emb, UI_COLORS, '{} colored')} colors"
    BASE["ui"] = dict(sketch=sk, sk_emb=sk_emb, S_gh=soft_sketch(sk, GH), S_256=soft_sketch(sk, 256), thin256=thin_np(sk, 256), t_clip=None, t_dino=None)
    TXT[caption], IPC["ui"] = ui_encode(caption, ref)
    it = dict(key="ui", kind="ui", group="ui", base="ui", caption=caption, cap_category=(obj.strip() or caption), sk_category="", ref="ui", attr=attr, seed=int(seed))
    it["cap_emb"], it["attr_emb"] = clip_txt([caption]), clip_txt([attr]); it["obj_emb"] = clip_txt([f"a photo of a {it['cap_category']}"])
    it["feat"] = conflict_features(it, sk_emb, ref_emb)
    return it, sk

LABELS = {"fixed": "Fixed weights (baseline)", "scheduled": "Scheduled (geometry first)", "router": "Router (learned weights)", "router_guided": "Router + constraint guidance"}
def synthesize(sketch_im, ref_im, caption, obj, attr_override, choice, seed):
    if sketch_im is None or ref_im is None or not caption.strip(): raise gr.Error("Provide a sketch, a reference image and a caption.")
    it, sk = build_item(sketch_im, ref_im, caption.strip(), obj, attr_override, seed)
    arms = list(ARMS) if choice == "Compare all 4" else [k for k, v in LABELS.items() if v == choice]
    out, info = [], [f"reference attributes: {it['attr']}"]
    for arm in arms:
        p = params_for(it, arm)
        out.append((run_item(it, p, ARMS[arm]["shaped"], ARMS[arm]["guided"]), f"{LABELS[arm]} | sketch {p['cs']:.2f} hold {p['hold']:.2f} | ref {p['ip']:.2f} | text {p['cfg']:.1f}"))
        info.append(f"{LABELS[arm]}: sketch strength {p['cs']:.2f}, sketch hold {p['hold']:.2f}, reference strength {p['ip']:.2f}, text strength {p['cfg']:.1f}")
        torch.cuda.empty_cache()
    return out, sk, "\n".join(info)

with gr.Blocks(title="Attribute-routed multimodal diffusion") as demo:
    gr.Markdown("## Attribute-routed multimodal diffusion (SDXL)\nSketch -> geometry, caption -> semantics, reference -> appearance. The router sets how strongly each acts.")
    with gr.Row():
        with gr.Column():
            i_sk = gr.Image(type="pil", label="Sketch"); i_ref = gr.Image(type="pil", label="Reference image")
        with gr.Column():
            i_cap = gr.Textbox(label="Caption"); i_obj = gr.Textbox(label="Object name (optional)")
            i_attr = gr.Textbox(label="Reference attributes (optional; empty = predicted automatically)")
            i_arm = gr.Dropdown(["Compare all 4"] + list(LABELS.values()), value=LABELS["router"], label="Method"); i_seed = gr.Number(value=0, precision=0, label="Seed")
            btn = gr.Button("Generate", variant="primary")
    o_gal = gr.Gallery(label="Results", columns=2, height=520)
    with gr.Row():
        o_sk = gr.Image(label="Sketch given to the model"); o_txt = gr.Textbox(label="What the router decided", lines=7)
    btn.click(synthesize, [i_sk, i_ref, i_cap, i_obj, i_attr, i_arm, i_seed], [o_gal, o_sk, o_txt])
demo.queue(default_concurrency_limit=1).launch(share=True, show_error=True)