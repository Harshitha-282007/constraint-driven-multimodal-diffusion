"""
Role-dropout fusion LoRA: fine-tunes the SDXL UNet (LoRA on its attention layers) so that the generator obeys
the governance gates.

Every training sample is an aligned quadruple from pilot_v1: (target photo, sketch, reference, caption). The
reference is a DIFFERENT object with the target's colour and material, so the target is the correct answer for
"shape from the sketch, identity from the text, appearance from the reference". Each step draws one regime:

    full         caption as is,               full reference             (40%)
    text_colour  caption states the colour,   greyscale reference only   (25%)  -> learns "colour from the text"
    ref_colour   colour words removed,        full reference             (25%)  -> learns "colour from the reference"
    drop         negative prompt,             no reference               (10%)  -> keeps classifier-free guidance valid

The first three are exactly the inputs the governed pipeline feeds at inference (see pipeline.compose_ip), and
in all of them the real photo is the supervision. Training all conditions together follows the denoising LoRA of
UniCombine (arXiv 2503.09277); the role dropout is ours.

Frozen: VAE, text encoders, IP-Adapter, ControlNet (run under no_grad), UNet base weights. Trained: LoRA only.
Memory recipe follows the diffusers SDXL DreamBooth-LoRA example (fp16 frozen weights, fp32 LoRA, gradient
checkpointing, 8-bit Adam when bitsandbytes is installed).

    python src/train_fusion_lora.py --data /content/pilot_v1/pilot_v1 --out /content/drive/MyDrive/.../fusion_lora
    python src/train_fusion_lora.py --data ... --out ... --max_steps 5 --memory_test      # day-1 feasibility check
"""
import argparse
import gc
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from governance.features import to_grey
from governance.labels import ensure_color, ref_colors, strip_colors
from pipeline import BASE_ID, Pipeline, compose_ip, ip_shape, prep_ref, prep_sketch, style_block

REGIMES = [("full", 0.40), ("text_colour", 0.25), ("ref_colour", 0.25), ("drop", 0.10)]


def load_rows(data_dir, split="train", only_clean=True):
    rows = [json.loads(l) for l in open(os.path.join(data_dir, "metadata.jsonl")) if l.strip()]
    rows = [r for r in rows if r["split"] == split and r.get("conflict_type", "none") == "none"]
    if only_clean:      # notebook 01 kept all 200 rows; train only on references that passed the quality filters
        clean = [r for r in rows if r.get("ref_filters_passed", True) and r.get("seg_ok", True)]
        rows = clean or rows
    return rows


@torch.no_grad()
def prepare(P, rows, data_dir, log=print):
    """Encode everything once: target latents, three caption variants, full + grey reference, control image."""
    res, dev, dt = P.res, P.dev, P.dt
    op = lambda r, k: Image.open(os.path.join(data_dir, r[k]))
    caps = []
    for r in rows:
        r["_caps"] = dict(full=r["caption"], text_colour=ensure_color(r["caption"], ref_colors(r)), ref_colour=strip_colors(r["caption"]))
        caps += list(r["_caps"].values())
    caps = list(dict.fromkeys(caps))
    refs = [prep_ref(op(r, "reference"), res) for r in rows]
    txt, ips = P.encode(caps, refs + [to_grey(x) for x in refs])
    txt = dict(zip(caps, txt))
    data = []
    for i, r in enumerate(rows):
        tgt = op(r, "target").convert("RGB").resize((res, res), Image.LANCZOS)
        x = (torch.from_numpy(np.array(tgt)).permute(2, 0, 1)[None].float() / 127.5 - 1).to(dev, dt)
        lat = (P.vae.encode(x).latent_dist.sample() * P.vae.config.scaling_factor).cpu()
        ctrl = P.pipe.prepare_image(image=prep_sketch(op(r, "sketch"), res), width=res, height=res, batch_size=1,
                                    num_images_per_prompt=1, device=dev, dtype=dt,
                                    do_classifier_free_guidance=False, guess_mode=False).cpu()
        data.append(dict(id=r["id"], lat=lat, ctrl=ctrl, ip_full=ips[i], ip_grey=ips[len(rows) + i],
                         txt={k: txt[c] for k, c in r["_caps"].items()}))
    log(f"prepared {len(data)} samples | caption variants: {len(caps)}")
    return data


def add_lora(unet, rank=8, scope="all"):
    from diffusers.training_utils import cast_training_params
    from peft import LoraConfig
    unet.requires_grad_(False)
    suffixes = ("to_k", "to_q", "to_v", "to_out.0")
    targets = list(suffixes) if scope == "all" else [
        n for n, _ in unet.named_modules() if n.endswith(suffixes) and n.startswith("up_blocks")]
    unet.add_adapter(LoraConfig(r=rank, lora_alpha=rank, init_lora_weights="gaussian", target_modules=targets))
    cast_training_params(unet, dtype=torch.float32)        # LoRA weights in fp32, frozen base stays fp16
    unet.enable_gradient_checkpointing()
    return [p for p in unet.parameters() if p.requires_grad]


def make_optimizer(params, lr):
    try:
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(params, lr=lr, weight_decay=1e-2)
    except Exception:
        return torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)


def save_lora(unet, out_dir, name="fusion_lora.safetensors"):
    from diffusers import StableDiffusionXLControlNetPipeline
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft.utils import get_peft_model_state_dict
    os.makedirs(out_dir, exist_ok=True)
    StableDiffusionXLControlNetPipeline.save_lora_weights(
        save_directory=out_dir, weight_name=name,
        unet_lora_layers=convert_state_dict_to_diffusers(get_peft_model_state_dict(unet)))
    return os.path.join(out_dir, name)


def train(data_dir, out_dir, res=512, rank=8, scope="all", lr=1e-4, max_steps=800, accum=4, ip_scale=0.5,
          seed=0, only_clean=True, memory_test=False, save_every=200, P=None, log=print):
    """Returns (Pipeline with the LoRA attached, list of losses). Pass an existing Pipeline(res=res, clip=False) to reuse it."""
    from diffusers import DDPMScheduler
    random.seed(seed); torch.manual_seed(seed)
    P = P or Pipeline(res=res, clip=False, log=log)
    dev, dt = P.dev, P.dt
    rows = load_rows(data_dir, "train", only_clean)
    data = prepare(P, rows, data_dir, log)
    P.vae.to("cpu"); gc.collect(); torch.cuda.empty_cache()                     # not needed while training

    params = add_lora(P.unet, rank, scope)
    opt = make_optimizer(params, lr)
    scaler = torch.cuda.amp.GradScaler()
    noise_sched = DDPMScheduler.from_pretrained(BASE_ID, subfolder="scheduler")
    T = noise_sched.config.num_train_timesteps
    tids = torch.tensor([[res, res, 0, 0, res, res]], device=dev, dtype=dt)
    names, probs = zip(*REGIMES)
    log(f"trainable LoRA params: {sum(p.numel() for p in params) / 1e6:.2f} M | optimizer: {type(opt).__name__} | "
        f"GPU before training {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    torch.cuda.reset_peak_memory_stats()

    P.unet.train()
    losses, step = [], 0
    opt.zero_grad(set_to_none=True)
    while step < max_steps:
        for _ in range(accum):
            d = random.choice(data)
            regime = random.choices(names, probs)[0]
            lat = d["lat"].to(dev, dt)
            noise = torch.randn_like(lat)
            t = torch.randint(0, T, (1,), device=dev)
            noisy = noise_sched.add_noise(lat, noise, t)
            f = 1 - t.item() / T                                            # fraction of denoising already done
            if regime == "drop":
                _, pe, _, pooled = d["txt"]["full"]                         # negative prompt embeddings
                ip, amp = torch.zeros_like(d["ip_full"]), 0.0
            else:
                pe, _, pooled, _ = d["txt"][regime]
                g_col = 1.0 if regime == "text_colour" else 0.0
                ip, amp = compose_ip(d["ip_full"], d["ip_grey"], P.flat_ip, g_col, 0.0), ip_scale
            pe, pooled, ip = pe.to(dev), pooled.to(dev), ip.to(dev, dt)
            added = {"text_embeds": pooled, "time_ids": tids}
            P.pipe.set_ip_adapter_scale(style_block(amp * ip_shape(f)))     # same schedule and block as at inference
            with torch.autocast("cuda", dtype=torch.float16):
                with torch.no_grad():
                    down, mid = P.controlnet(noisy, t, encoder_hidden_states=pe, controlnet_cond=d["ctrl"].to(dev, dt),
                                             conditioning_scale=1.0, guess_mode=False, added_cond_kwargs=added, return_dict=False)
                pred = P.unet(noisy, t, encoder_hidden_states=pe, added_cond_kwargs={**added, "image_embeds": [ip]},
                              down_block_additional_residuals=down, mid_block_additional_residual=mid, return_dict=False)[0]
            loss = F.mse_loss(pred.float(), noise.float())
            scaler.scale(loss / accum).backward()
            losses.append(loss.item())
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        step += 1
        if step % 20 == 0 or memory_test:
            log(f"step {step:4d}/{max_steps} | loss (last {20 * accum}) {np.mean(losses[-20 * accum:]):.4f} | "
                f"peak GPU {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        if not memory_test and step % save_every == 0:
            save_lora(P.unet, out_dir)
    P.unet.eval()
    if memory_test:
        log(f"MEMORY TEST OK at res={res}, rank={rank}, scope={scope}: peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    else:
        path = save_lora(P.unet, out_dir)
        json.dump(dict(res=res, rank=rank, scope=scope, lr=lr, steps=max_steps, accum=accum, ip_scale=ip_scale,
                       n_train=len(data), regimes=REGIMES, final_loss=float(np.mean(losses[-100:]))),
                  open(os.path.join(out_dir, "fusion_lora_config.json"), "w"), indent=1)
        log(f"saved {path}")
    P.vae.to(dev); P.has_lora = True
    return P, losses


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder that contains metadata.jsonl (the unzipped pilot_v1)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--scope", choices=["all", "up"], default="all", help="'up' = LoRA on up_blocks only (less memory)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_steps", type=int, default=800)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--memory_test", action="store_true")
    a = ap.parse_args()
    train(a.data, a.out, res=a.res, rank=a.rank, scope=a.scope, lr=a.lr, max_steps=a.max_steps, accum=a.accum,
          memory_test=a.memory_test)
