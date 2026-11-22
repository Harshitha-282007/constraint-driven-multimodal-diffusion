"""Inference with the learned model. No conflict flags, no per-modality weights: constants below are fixed for ALL inputs.
python governance/infer.py --weights runs/cas_v1/cas_weights.pt --sketch s.png --ref r.png --caption "a red bicycle" --out out.png
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import torch
from cas_data import prep_sketch, prep_ref
NEG = "blurry, low quality, text, watermark, multiple objects, cropped"
FIXED = dict(controlnet_conditioning_scale=1.0, guidance_scale=7.0, num_inference_steps=20)   # identical for every input
CAS_IP_SCALE, BASELINE_IP_SCALE = 1.0, 0.6      # baseline value = notebook 02/04 "fixed" arm


def apply_weights(unet, path):
    """Copy trained tensors into the UNet; returns backup so the original (baseline) weights can be restored."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    sd = unet.state_dict(); backup = {}
    missing_lora = [n for n in blob["weights"] if n not in sd]
    if missing_lora:
        from peft import LoraConfig
        r = next(v.shape[0] for n, v in blob["weights"].items() if "lora_A" in n)
        unet.add_adapter(LoraConfig(r=r, lora_alpha=r, target_modules=r".*attn2\.to_(k|v)$")); sd = unet.state_dict()
    for n, v in blob["weights"].items():
        backup[n] = sd[n].detach().clone(); sd[n].copy_(v.to(sd[n].dtype))
    return backup, blob["step"]


def restore_weights(unet, backup):
    sd = unet.state_dict()
    for n, v in backup.items(): sd[n].copy_(v)


def generate(pipe, sketch_path, ref_path, caption, seed, res=640, ip_scale=CAS_IP_SCALE):
    pipe.set_ip_adapter_scale(ip_scale)
    g = torch.Generator("cpu").manual_seed(seed)
    return pipe(prompt=caption, negative_prompt=NEG, image=prep_sketch(sketch_path, res), ip_adapter_image=prep_ref(ref_path, res),
                height=res, width=res, generator=g, **FIXED).images[0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True); ap.add_argument("--sketch", required=True); ap.add_argument("--ref", required=True)
    ap.add_argument("--caption", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--res", type=int, default=640)
    a = ap.parse_args()
    from cas_stack import load_pipe
    pipe = load_pipe()
    apply_weights(pipe.unet, a.weights)
    generate(pipe, a.sketch, a.ref, a.caption, a.seed, a.res).save(a.out)
