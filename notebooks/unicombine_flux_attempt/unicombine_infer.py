"""
Inference for the Sketchy pilot with UniCombine (sketch + reference Condition-LoRAs, optional Denoising-LoRA).
Low-memory version: 4-bit transformer, text embeddings come from the Part-1 cache (no T5/CLIP loaded).

IMPORTANT (from UniCombine's attention code): the image branch applies every *active* adapter that is not
the type of a condition passed in. So the active adapter list must always match the conditions you pass.
`set_mode` does that; never call the pipe with a mismatched adapter list.

Modes (these are also your ablation rows in the report):
    sketch          sketch Condition-LoRA only
    reference       reference Condition-LoRA only
    both_free       both Condition-LoRAs, no Denoising-LoRA  (UniCombine "training-free" union)
    both_trained    both Condition-LoRAs + trained Denoising-LoRA   (the baseline we train)
    text_only       plain FLUX.1-schnell, caption only (separate pipeline: load_text_only / run_split_text_only)
"""
import json, os, sys
import torch
from PIL import Image


def _dtype():
    return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


def _qconfig(dt):
    from diffusers import BitsAndBytesConfig
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dt, bnb_4bit_use_double_quant=True)


def load(uni_dir, flux_dir, cond_lora_dir, denoise_dir=None, denoise_name=None,
         cond_types=("sketch", "reference"), device="cuda", quantize=True):
    sys.path.insert(0, uni_dir)
    from src.UniCombineTransformer2DModel import UniCombineTransformer2DModel
    from src.UniCombinePipeline import UniCombinePipeline

    dt = _dtype()
    tr = UniCombineTransformer2DModel.from_pretrained(os.path.join(flux_dir, "transformer"), torch_dtype=dt,
                                                      quantization_config=_qconfig(dt) if quantize else None)
    if not quantize:
        tr = tr.to(device)
    for t in cond_types:
        tr.load_lora_adapter(os.path.join(cond_lora_dir, f"{t}.safetensors"), adapter_name=t)
    if denoise_name:
        tr.load_lora_adapter(os.path.join(denoise_dir, denoise_name), adapter_name=denoise_name, use_safetensors=True)
    # no text encoders / tokenizers: prompts come in as cached embeddings
    pipe = UniCombinePipeline.from_pretrained(flux_dir, torch_dtype=dt, transformer=None, text_encoder=None,
                                              text_encoder_2=None, tokenizer=None, tokenizer_2=None)
    pipe.transformer = tr
    pipe.vae.to(device)                      # (do not pipe.to(): the 4-bit transformer is already on the GPU)
    pipe.denoise_name, pipe.dt = denoise_name, dt
    return pipe


def set_mode(pipe, mode, denoise_weight=1.0):
    types = {"sketch": ["sketch"], "reference": ["reference"],
             "both_free": ["sketch", "reference"], "both_trained": ["sketch", "reference"]}[mode]
    if mode == "both_trained":
        assert pipe.denoise_name, "load() was called without a Denoising-LoRA"
        pipe.transformer.set_adapters(types + [pipe.denoise_name], [1.0] * len(types) + [denoise_weight])
    else:
        pipe.transformer.set_adapters(types)
    return types


def _embeds(cache, caption, dt, device="cuda"):
    pe, pooled = cache["text"][caption]
    return pe.to(device, dt)[None], pooled.to(device, dt)[None]


@torch.no_grad()
def generate(pipe, cache, caption, sketch=None, reference=None, mode="both_trained",
             seed=0, steps=8, denoise_weight=1.0):
    from src.condition import Condition
    res = cache["res"]
    types = set_mode(pipe, mode, denoise_weight)
    imgs = {"sketch": sketch, "reference": reference}
    conds = [Condition(t, raw_img=imgs[t].convert("RGB").resize((res, res), Image.LANCZOS), need_preprocess=False) for t in types]
    pe, pooled = _embeds(cache, caption, pipe.dt)
    g = torch.Generator("cpu").manual_seed(int(seed))
    return pipe(prompt=None, prompt_embeds=pe, pooled_prompt_embeds=pooled, conditions=conds, height=res, width=res,
                num_inference_steps=steps, max_sequence_length=cache["max_seq"], model_config={}, generator=g).images[0]


def run_split(pipe, cache, root, split, mode, out_dir, meta_file="metadata.jsonl", steps=8, seed=0,
              denoise_weight=1.0, limit=None):
    """Generate for every row of a split (resumable). Reads the conditions the row names, so the same function
    runs the aligned pilot now and the conflict subset later (rows there point at mismatched assets/captions)."""
    os.makedirs(out_dir, exist_ok=True)
    rows = [json.loads(l) for l in open(os.path.join(root, meta_file))]
    rows = [r for r in rows if r["split"] == split][:limit]
    for k, r in enumerate(rows):
        out = os.path.join(out_dir, f"{r['id']}.png")
        if os.path.exists(out):
            continue
        sk, rf = Image.open(os.path.join(root, r["sketch"])), Image.open(os.path.join(root, r["reference"]))
        generate(pipe, cache, r["caption"], sk, rf, mode, seed + k, steps, denoise_weight).save(out)
    return out_dir


# ---------------------------------------------------------------------------------------------- text-only baseline
def load_text_only(flux_dir, device="cuda"):
    from diffusers import FluxPipeline, FluxTransformer2DModel
    dt = _dtype()
    tr = FluxTransformer2DModel.from_pretrained(os.path.join(flux_dir, "transformer"), torch_dtype=dt, quantization_config=_qconfig(dt))
    pipe = FluxPipeline.from_pretrained(flux_dir, torch_dtype=dt, transformer=tr, text_encoder=None, text_encoder_2=None,
                                        tokenizer=None, tokenizer_2=None)
    pipe.vae.to(device)
    pipe.dt = dt
    return pipe


@torch.no_grad()
def run_split_text_only(pipe, cache, root, split, out_dir, meta_file="metadata.jsonl", steps=4, seed=0, limit=None):
    os.makedirs(out_dir, exist_ok=True)
    res = cache["res"]
    rows = [json.loads(l) for l in open(os.path.join(root, meta_file))]
    rows = [r for r in rows if r["split"] == split][:limit]
    for k, r in enumerate(rows):
        out = os.path.join(out_dir, f"{r['id']}.png")
        if os.path.exists(out):
            continue
        pe, pooled = _embeds(cache, r["caption"], pipe.dt)
        pipe(prompt=None, prompt_embeds=pe, pooled_prompt_embeds=pooled, height=res, width=res, num_inference_steps=steps,
             guidance_scale=0.0, max_sequence_length=cache["max_seq"],
             generator=torch.Generator("cpu").manual_seed(seed + k)).images[0].save(out)
    return out_dir
