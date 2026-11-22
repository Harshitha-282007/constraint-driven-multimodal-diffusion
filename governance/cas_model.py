"""Trainable-parameter selection, loss, checkpointing for Counterfactual Attribute Supervision (CAS).
Everything frozen except the REFERENCE pathway (IP-Adapter image projection + to_k_ip/to_v_ip) and,
optionally, a low-rank LoRA on the TEXT cross-attention K/V. ControlNet (sketch) is untouched."""
import os, random
import numpy as np
import torch
import torch.nn.functional as F


def select_trainable(unet, text_lora_rank=0):
    for p in unet.parameters():
        p.requires_grad_(False)
    names = []
    for n, p in unet.named_parameters():
        if n.startswith("encoder_hid_proj.") or ".to_k_ip." in n or ".to_v_ip." in n:
            names.append(n)
    if text_lora_rank > 0:
        from peft import LoraConfig
        unet.add_adapter(LoraConfig(r=text_lora_rank, lora_alpha=text_lora_rank, init_lora_weights="gaussian",
                                    target_modules=r".*attn2\.to_(k|v)$"))
        names += [n for n, p in unet.named_parameters() if "lora_" in n]
    pmap = dict(unet.named_parameters())
    assert names, "no trainable parameters found: is an IP-Adapter loaded into the UNet?"
    params = []
    for n in names:
        p = pmap[n]; p.data = p.data.float(); p.requires_grad_(True); params.append(p)
    return names, params


def trainable_state(unet, names):
    sd = unet.state_dict()
    return {n: sd[n].detach().cpu().float().clone() for n in names}


def load_trainable_state(unet, state):
    sd = unet.state_dict()
    for n, v in state.items():
        assert n in sd, f"unknown tensor {n}"
        sd[n].copy_(v.to(sd[n].dtype))


def weighted_eps_loss(pred, noise, mask_lat, obj_weight):
    """MSE on epsilon, up-weighted inside the object mask (where colour/material live)."""
    w = 1.0 + obj_weight * mask_lat
    return (w * (pred.float() - noise.float()) ** 2).mean()


def l2sp(params, anchors):
    return sum(((p - a.to(p.device)) ** 2).sum() for p, a in zip(params, anchors)) / max(1, sum(p.numel() for p in params))


def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp); os.replace(tmp, path)


def rng_state():
    return dict(py=random.getstate(), np=np.random.get_state(), torch=torch.get_rng_state())


def set_rng_state(s):
    random.setstate(s["py"]); np.random.set_state(s["np"]); torch.set_rng_state(s["torch"])
