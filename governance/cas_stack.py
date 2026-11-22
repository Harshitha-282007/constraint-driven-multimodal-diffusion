"""Loads the real SDXL + scribble ControlNet + IP-Adapter stack (same checkpoints as notebooks 02/04)."""
import torch
BASE_ID = "stabilityai/stable-diffusion-xl-base-1.0"
VAE_ID = "madebyollin/sdxl-vae-fp16-fix"
CN_ID = "xinsir/controlnet-scribble-sdxl-1.0"
IP_REPO, IP_SUB, IP_W, ENC_SUB = "h94/IP-Adapter", "sdxl_models", "ip-adapter_sdxl_vit-h.safetensors", "models/image_encoder"


def load_pipe(dtype=torch.float16, device="cuda"):
    from diffusers import (UNet2DConditionModel, ControlNetModel, AutoencoderKL,
                           StableDiffusionXLControlNetPipeline, EulerDiscreteScheduler)
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer, CLIPVisionModelWithProjection
    vae = AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=dtype)
    unet = UNet2DConditionModel.from_pretrained(BASE_ID, subfolder="unet", torch_dtype=dtype, variant="fp16")
    cn = ControlNetModel.from_pretrained(CN_ID, torch_dtype=dtype)
    te1 = CLIPTextModel.from_pretrained(BASE_ID, subfolder="text_encoder", torch_dtype=dtype, variant="fp16")
    te2 = CLIPTextModelWithProjection.from_pretrained(BASE_ID, subfolder="text_encoder_2", torch_dtype=dtype, variant="fp16")
    enc = CLIPVisionModelWithProjection.from_pretrained(IP_REPO, subfolder=ENC_SUB, torch_dtype=dtype)
    pipe = StableDiffusionXLControlNetPipeline(
        vae=vae, text_encoder=te1, text_encoder_2=te2,
        tokenizer=CLIPTokenizer.from_pretrained(BASE_ID, subfolder="tokenizer"),
        tokenizer_2=CLIPTokenizer.from_pretrained(BASE_ID, subfolder="tokenizer_2"),
        unet=unet, controlnet=cn, scheduler=EulerDiscreteScheduler.from_pretrained(BASE_ID, subfolder="scheduler"),
        image_encoder=enc)
    pipe.load_ip_adapter(IP_REPO, subfolder=IP_SUB, weight_name=IP_W)
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(device)


def components_from_pipe(pipe, res, dtype, dev):
    """Build the callables train_step needs from a loaded pipeline."""
    from diffusers import DDPMScheduler
    from transformers import CLIPImageProcessor
    proc = CLIPImageProcessor()

    @torch.no_grad()
    def encode_text(prompts):
        pe, _, pooled, _ = pipe.encode_prompt(prompts, device=dev, num_images_per_prompt=1, do_classifier_free_guidance=False)
        return pe.to(dtype), pooled.to(dtype)

    @torch.no_grad()
    def encode_image(img):
        px = proc(images=img, return_tensors="pt").pixel_values.to(dev, dtype)
        return pipe.image_encoder(px).image_embeds

    return dict(unet=pipe.unet, controlnet=pipe.controlnet, vae=pipe.vae,
                scheduler=DDPMScheduler.from_pretrained(BASE_ID, subfolder="scheduler"),
                encode_text=encode_text, encode_image=encode_image, dtype=dtype, dev=dev, res=res)
