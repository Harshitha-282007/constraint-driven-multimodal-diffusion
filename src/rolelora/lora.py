"""
Role LoRA: a small trained adapter on the places where SDXL READS its inputs, so that the model itself learns which
input governs which attribute, instead of being handed neutralised inputs.

Where the LoRA sits (and nowhere else):

    text       attn2.to_k, attn2.to_v          cross-attention key / value projections   (up blocks + mid block)
    reference  to_k_ip, to_v_ip                the IP-Adapter's reference key / value projections (same layers)
    sketch     nothing: the ControlNet stays frozen; the sketch role is protected by the data (the reference object never
               has the target's shape) and by the preservation loss in the trainer

Restricted by WHERE (input interfaces of the late layers), not by modality: the whole-UNet LoRA of the earlier run learned
the photo scene and lost the sketch (edge IoU 0.369 -> 0.230). `groups` selects the interfaces: ("text", "ref") is the
full multi-role LoRA, ("ref",) the reference-only ablation.

Wrapping is in place and permanent until `detach`; `set_scale(0)` turns every LoRA off, which gives the exact baseline.
Attach BEFORE RoleGates.install (the gates capture the IP projections when they are installed).
"""
import math
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

SCOPES = ("up_blocks", "mid_block")
GROUPS = ("text", "ref")


class _Ctrl:
    def __init__(self):
        self.scale = 1.0
        self.cond_only = False


class LoRAParams(nn.Module):
    def __init__(self, in_f, out_f, rank):
        super().__init__()
        self.A = nn.Parameter(torch.empty(rank, in_f))
        self.B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))


class LoRALinear(nn.Module):
    """base(x) + scale * (alpha / rank) * B A x.  B starts at zero, so the model is unchanged before training."""

    def __init__(self, base, params, ctrl, rank, alpha):
        super().__init__()
        self.base, self.lora = base, params
        self.mult = alpha / rank
        object.__setattr__(self, "ctrl", ctrl)

    # attributes other code reads from a Linear
    weight = property(lambda self: self.base.weight)
    bias = property(lambda self: self.base.bias)
    in_features = property(lambda self: self.base.in_features)
    out_features = property(lambda self: self.base.out_features)

    def forward(self, x):
        y = self.base(x)
        s = self.ctrl.scale
        if s == 0.0:
            return y
        d = F.linear(F.linear(x.to(self.lora.A.dtype), self.lora.A), self.lora.B) * (self.mult * s)
        if self.ctrl.cond_only and x.shape[0] % 2 == 0:
            # classifier-free batch is [unconditional, conditional]: leave the unconditional half exactly at the baseline
            d = torch.cat([torch.zeros_like(d[: d.shape[0] // 2]), d[d.shape[0] // 2:]])
        return y + d.to(y.dtype)


class RoleLoRA(nn.Module):
    def __init__(self, groups=GROUPS, rank=16, alpha=16):
        super().__init__()
        self.groups, self.rank, self.alpha = tuple(groups), rank, alpha
        self.params = nn.ModuleList()
        self.names = []                                   # one name per params entry, in order (for save / load)
        self.ctrl = _Ctrl()
        self._undo = []

    # ------------------------------------------------------------------ attach / detach
    def _wrap(self, container, key, name):
        base = container[key] if isinstance(container, nn.ModuleList) else getattr(container, key)
        p = LoRAParams(base.in_features, base.out_features, self.rank).to(base.weight.device)
        self.params.append(p); self.names.append(name)
        w = LoRALinear(base, p, self.ctrl, self.rank, self.alpha)
        if isinstance(container, nn.ModuleList):
            container[key] = w
        else:
            setattr(container, key, w)
        self._undo.append((container, key, base))

    @classmethod
    def attach(cls, unet, groups=GROUPS, rank=16, alpha=16, scopes=SCOPES):
        self = cls(groups, rank, alpha)
        for name, proc in unet.attn_processors.items():
            if ".attn2." not in name or not name.startswith(tuple(scopes)):
                continue
            attn = unet.get_submodule(name[: -len(".processor")])
            if "text" in self.groups:
                for attr in ("to_k", "to_v"):
                    self._wrap(attn, attr, f"{name}.{attr}")
            if "ref" in self.groups and hasattr(proc, "to_k_ip"):
                for attr in ("to_k_ip", "to_v_ip"):
                    self._wrap(getattr(proc, attr), 0, f"{name}.{attr}")
        if not self.names:
            raise ValueError("no cross-attention layers found to attach the LoRA to")
        return self

    def detach(self, unet=None):
        for container, key, base in reversed(self._undo):
            if isinstance(container, nn.ModuleList):
                container[key] = base
            else:
                setattr(container, key, base)
        self._undo = []

    # ------------------------------------------------------------------ switch
    def set_scale(self, s, cond_only=False):
        """cond_only=True (inference with classifier-free guidance): the LoRA acts on the conditional half only, so the
        unconditional half stays the baseline and (cond - uncond) x cfg carries just the LoRA's intended change."""
        self.ctrl.scale = float(s)
        self.ctrl.cond_only = bool(cond_only)

    @contextmanager
    def disabled(self):
        old = self.ctrl.scale
        self.ctrl.scale = 0.0
        try:
            yield
        finally:
            self.ctrl.scale = old

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------ io
    def save(self, path, meta=None):
        torch.save(dict(state=self.state_dict(), names=self.names, groups=self.groups, rank=self.rank, alpha=self.alpha, meta=meta or {}), path)

    @classmethod
    def load(cls, path, unet, device=None):
        d = torch.load(path, map_location="cpu", weights_only=False)
        self = cls.attach(unet, d["groups"], d["rank"], d["alpha"])
        assert self.names == d["names"], "checkpoint was made for a different set of layers"
        self.load_state_dict(d["state"])
        return self.to(device or next(unet.parameters()).device).eval()
