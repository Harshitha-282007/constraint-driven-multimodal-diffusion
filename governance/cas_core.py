"""One CAS training step, independent of how the (frozen) components were loaded (real SDXL or tiny test stack)."""
import random
import numpy as np
import torch
from cas_data import to_tensor, shape_perturb
from cas_model import weighted_eps_loss


@torch.no_grad()
def _enc_latents(vae, img, dtype, dev, gen):
    x = to_tensor(img).unsqueeze(0).to(dev, dtype)
    return vae.encode(x).latent_dist.sample(generator=gen) * vae.config.scaling_factor


def train_step(C, sample, step, cfg, rng):
    """C: dict(unet, controlnet, vae, scheduler, encode_text, encode_image, dtype, dev, res).
    Returns dict of scalar losses (with 'loss' = tensor to backprop)."""
    dev, dt = C["dev"], C["dtype"]
    unet, cn, sch = C["unet"], C["controlnet"], C["scheduler"]
    g = torch.Generator(device="cpu").manual_seed(cfg["seed"] * 1000003 + step)
    lat = _enc_latents(C["vae"], sample["target"], dt, dev, g if dev == "cpu" else None)
    noise = torch.randn(lat.shape, generator=g).to(dev, lat.dtype)
    t = torch.randint(0, sch.config.num_train_timesteps, (1,), generator=g).to(dev)
    noisy = sch.add_noise(lat, noise, t)

    # modality dropout (keeps classifier-free guidance usable for every pathway)
    cap = "" if rng.random() < cfg["p_drop_text"] else sample["caption"]
    drop_ref = rng.random() < cfg["p_drop_ref"]
    drop_sk = rng.random() < cfg["p_drop_sketch"]
    pe, pooled = C["encode_text"]([cap])
    res = C["res"]
    time_ids = torch.tensor([[res, res, 0, 0, res, res]], device=dev, dtype=pe.dtype)
    added = {"text_embeds": pooled, "time_ids": time_ids}

    sk = to_tensor(sample["sketch"]).unsqueeze(0).to(dev, dt) * 0.5 + 0.5
    if drop_sk:
        sk = torch.zeros_like(sk)
    with torch.no_grad():
        down, mid = cn(noisy, t, encoder_hidden_states=pe, controlnet_cond=sk, conditioning_scale=1.0,
                       added_cond_kwargs=added, return_dict=False)

    def predict(ref_img, drop):
        ie = C["encode_image"](ref_img)
        if drop:
            ie = torch.zeros_like(ie)
        kw = dict(added, image_embeds=[ie.unsqueeze(1)])
        return unet(noisy, t, encoder_hidden_states=pe, added_cond_kwargs=kw,
                    down_block_additional_residuals=[d.to(dt) for d in down], mid_block_additional_residual=mid.to(dt)).sample

    pred = predict(sample["ref"], drop_ref)
    m = sample["mask"]
    H, W = lat.shape[-2:]
    if m is not None:
        mm = torch.from_numpy(np.asarray(m.resize((W, H))).astype(np.float32) / 255.0)[None, None].to(dev)
    else:
        mm = torch.zeros(1, 1, H, W, device=dev)
    l_diff = weighted_eps_loss(pred, noise, mm, cfg["obj_weight"])
    out = dict(l_diff=l_diff.detach().item())
    loss = l_diff
    # shape-invariance: same appearance, different silhouette => same noise prediction (target = unperturbed branch)
    if cfg["lambda_inv"] > 0 and step % cfg["inv_every"] == 0 and not drop_ref:
        pred2 = predict(shape_perturb(sample["ref"], random.Random(step)), False)
        l_inv = ((pred2.float() - pred.detach().float()) ** 2).mean()
        loss = loss + cfg["lambda_inv"] * l_inv
        out["l_inv"] = l_inv.detach().item()
    out["loss"] = loss
    return out
