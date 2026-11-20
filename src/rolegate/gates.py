"""
Role gates inside SDXL.

Every cross-attention layer (`attn2`) of the UNet gets a per-pixel gate vector over six attribute channels:

    text_other, text_identity, text_colour, text_material   -> scale each caption token group's share of the attention
    ref_colour, ref_texture                                   -> scale each reference channel's own attention output

Every ControlNet residual gets a per-pixel gate; residuals are grouped by resolution into
    sketch_contour (the two highest resolutions) and sketch_placement (lowest resolution + mid block).

All gates read the UNet features at that point plus a conditioning vector c = [conflict flags (object, colour,
material), timestep features], and are initialised so the network starts from the existing fixed-weight pipeline.

    g = 2 * sigmoid(b0 + W_h . LayerNorm(h) + W_c . c)        (W_h, W_c zero at init, so g = 2*sigmoid(b0))

Text colour / material as channels (v2). CLIP's causal text encoder carries a colour word into every later token and
into the pooled embedding, so scaling the colour word's own attention cannot remove the colour. The caption is
therefore also encoded with its colour (material) words replaced by a neutral word at the same token positions, and
the gate acts on the difference:

    out = o(full) + (g_colour - 1) * (o(full) - o(colour-neutral)) + (g_material - 1) * (o(full) - o(material-neutral))

so g = 0 gives the caption without its colour, g = 1 the caption as written. The pooled embedding gets the same
treatment with a global gate that reads only the conditioning vector.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

TEXT_GATES = ["text_other", "text_identity", "text_colour", "text_material"]
REF_GATES = ["ref_colour", "ref_texture"]
CROSS_GATES = TEXT_GATES + REF_GATES
SKETCH_GATES = ["sketch_contour", "sketch_placement"]
ALL_GATES = CROSS_GATES + SKETCH_GATES
COND_DIM = 7
STYLE_PREFIX = "up_blocks.0.attentions.1"            # InstantStyle's style block for SDXL
REF_OFF_INIT = 0.1                                   # initial reference gate outside the style block


def _logit_for(g):                                   # g = 2*sigmoid(b)  ->  b
    p = g / 2
    return math.log(p / (1 - p))


def cond_vector(flags, t, num_train_timesteps=1000):
    """flags (B,3) in [0,1], t (B,) timesteps -> (B, 7)."""
    f = (t.float() / num_train_timesteps).view(-1, 1)
    return torch.cat([flags.float(), torch.sin(2 * math.pi * f), torch.cos(2 * math.pi * f), f, 1 - f], dim=1)


class GateContext:
    """Per-call state the processors read: set by the pipeline / trainer before each UNet call."""

    def __init__(self):
        self.cond = None            # (B, COND_DIM)
        self.text_groups = None     # (B, 77) long, group id per token slot
        self.ip_scale = 0.0         # base reference strength for this step (the existing schedule)
        self.knock = set()          # gate names forced to 0
        self.freeze_text = False    # text gates fixed at 1 (ablation)
        self.collect = None         # list -> (name, gates) with grad, for the prior loss
        self.record = None          # dict -> name: gates (detached), for gate maps
        self.text_neutral = None    # {"colour": (B,77,D) or None, "material": ...}: caption with those words neutralised


class CrossGate(nn.Module):
    def __init__(self, channels, style=False):
        super().__init__()
        self.norm = nn.LayerNorm(channels, elementwise_affine=False)
        self.w_h = nn.Linear(channels, len(CROSS_GATES), bias=False)
        self.w_c = nn.Linear(COND_DIM, len(CROSS_GATES))
        nn.init.zeros_(self.w_h.weight); nn.init.zeros_(self.w_c.weight)
        b0 = [0.0] * len(TEXT_GATES) + [_logit_for(1.0 if style else REF_OFF_INIT)] * len(REF_GATES)
        with torch.no_grad():
            self.w_c.bias.copy_(torch.tensor(b0))

    def forward(self, h, cond):
        """h (B,N,C), cond (B,COND_DIM) -> gates (B,N,6) in (0,2)."""
        return 2 * torch.sigmoid(self.w_h(self.norm(h.float())) + self.w_c(cond.float())[:, None, :])


class SketchGate(nn.Module):
    """Reads the ControlNet residual normalised per pixel over channels (as the cross gates normalise h): raw
    residuals are large, and an unnormalised 1x1 conv saturated the gate to 0 / 2 pixel by pixel (v1 run)."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.w_c = nn.Linear(COND_DIM, 1)
        nn.init.zeros_(self.conv.weight); nn.init.zeros_(self.w_c.weight); nn.init.zeros_(self.w_c.bias)

    def forward(self, r, cond):
        """r (B,C,H,W) -> gate (B,1,H,W) in (0,2)."""
        x = F.layer_norm(r.float().permute(0, 2, 3, 1), (r.shape[1],)).permute(0, 3, 1, 2)
        return 2 * torch.sigmoid(self.conv(x) + self.w_c(cond.float())[:, :, None, None])


class PooledGate(nn.Module):
    """Global colour / material gates for SDXL's pooled text embedding (reads only the conditioning vector)."""

    def __init__(self):
        super().__init__()
        self.w_c = nn.Linear(COND_DIM, 2)
        nn.init.zeros_(self.w_c.weight); nn.init.zeros_(self.w_c.bias)

    def forward(self, cond):
        return 2 * torch.sigmoid(self.w_c(cond.float()))          # (B,2): colour, material


class GatedCrossAttnProcessor(nn.Module):
    """Replaces IPAdapterAttnProcessor2_0 on attn2 layers; reuses its frozen to_k_ip / to_v_ip."""

    def __init__(self, name, gate, to_k_ip, to_v_ip, ctx):
        super().__init__()
        self.name, self.gate, self.ctx = name, gate, ctx
        self.to_k_ip, self.to_v_ip = to_k_ip, to_v_ip

    def _gates(self, h):
        ctx = self.ctx
        g = self.gate(h, ctx.cond.to(h.device))
        if ctx.freeze_text or ctx.knock:
            g = g.clone()
            if ctx.freeze_text:
                g[..., :len(TEXT_GATES)] = 1.0
            for k in ctx.knock:
                if k in CROSS_GATES:
                    g[..., CROSS_GATES.index(k)] = 0.0
        if ctx.collect is not None:
            ctx.collect.append((self.name, g))
        if ctx.record is not None:
            ctx.record[self.name] = g.detach().float().cpu()
        return g

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, **kwargs):
        residual = hidden_states
        ip_hidden_states = None
        if isinstance(encoder_hidden_states, tuple):
            encoder_hidden_states, ip_hidden_states = encoder_hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        batch_size = hidden_states.shape[0]
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        g = self._gates(hidden_states)                                   # (B,N,6)
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key, value = attn.to_k(encoder_hidden_states), attn.to_v(encoder_hidden_states)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        q = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        k = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        v = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # text: softmax over all tokens, then each token group's share scaled by its own gate
        groups = self.ctx.text_groups.to(hidden_states.device)                                      # (B,T)
        neutral = self.ctx.text_neutral or {}
        if any(v_ is not None for v_ in neutral.values()):
            # colour / material act through the neutral-caption channels; their word slots count as 'other'
            groups = torch.where(groups >= TEXT_GATES.index("text_colour"), torch.zeros_like(groups), groups)
        tg = g[..., :len(TEXT_GATES)]                                                               # (B,N,4)
        n_tok = k.shape[2]
        w = tg.gather(2, groups[:, None, :n_tok].expand(-1, tg.shape[1], -1))                       # (B,N,T)

        def text_attn(kk, vv):
            probs = (q.float() @ kk.float().transpose(-1, -2) * attn.scale).softmax(-1)             # (B,H,N,T)
            return ((probs * w[:, None]).to(vv.dtype) @ vv).transpose(1, 2).reshape(batch_size, -1, inner_dim)

        text_full = text_attn(k, v)
        out = text_full
        for gname, key_ in (("text_colour", "colour"), ("text_material", "material")):
            e = neutral.get(key_)
            if e is None:
                continue
            e = e.to(hidden_states.device, encoder_hidden_states.dtype)
            if attn.norm_cross:
                e = attn.norm_encoder_hidden_states(e)
            kn = attn.to_k(e).view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            vn = attn.to_v(e).view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            gi = CROSS_GATES.index(gname)
            out = out + (g[..., gi:gi + 1] - 1).to(v.dtype) * (text_full - text_attn(kn, vn))

        # reference: one attention per channel, each with its own gate
        scale = float(self.ctx.ip_scale)
        if ip_hidden_states is not None and scale != 0.0:
            ip = ip_hidden_states[0]                                                              # (B,C,4,D)
            if ip.ndim == 3:
                ip = ip[:, None]
            for c in range(min(ip.shape[1], len(REF_GATES))):
                ik = self.to_k_ip(ip[:, c]).view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                iv = self.to_v_ip(ip[:, c]).view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                o = F.scaled_dot_product_attention(q, ik, iv, attn_mask=None, dropout_p=0.0, is_causal=False)
                o = o.transpose(1, 2).reshape(batch_size, -1, inner_dim)
                gi = len(TEXT_GATES) + c
                out = out + scale * g[..., gi:gi + 1].to(o.dtype) * o

        out = out.to(query.dtype)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        if input_ndim == 4:
            out = out.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            out = out + residual
        return out / attn.rescale_output_factor


def _key(name):
    return name.replace(".", "__")


class RoleGates(nn.Module):
    def __init__(self, cross_channels, res_channels):
        """cross_channels: {attn2 processor name: hidden size}; res_channels: channels of each ControlNet residual
        (the down residuals in order, then the mid block)."""
        super().__init__()
        self.cfg = dict(cross_channels=dict(cross_channels), res_channels=list(res_channels))
        self.cross = nn.ModuleDict({_key(n): CrossGate(c, style=n.startswith(STYLE_PREFIX)) for n, c in cross_channels.items()})
        self.sketch = nn.ModuleList([SketchGate(c) for c in res_channels])
        self.pooled = PooledGate()
        self.ctx = GateContext()
        self._orig = None

    @classmethod
    def for_unet(cls, unet, controlnet_res_channels):
        cross = {n: p.hidden_size for n, p in unet.attn_processors.items() if n.endswith("attn2.processor") and hasattr(p, "to_k_ip")}
        if not cross:
            raise ValueError("no IP-Adapter attention processors found: load the IP-Adapter before building the gates")
        return cls(cross, controlnet_res_channels)

    # ---------------------------------------------------------------- install / remove
    def install(self, unet):
        procs = unet.attn_processors
        self._orig = dict(procs)
        new = {}
        for n, p in procs.items():
            if _key(n) in self.cross:
                new[n] = GatedCrossAttnProcessor(n, self.cross[_key(n)], p.to_k_ip[0], p.to_v_ip[0], self.ctx)
            else:
                new[n] = p
        unet.set_attn_processor(new)
        return self

    def remove(self, unet):
        if self._orig is not None:
            unet.set_attn_processor(self._orig)
            self._orig = None

    # ---------------------------------------------------------------- context
    def set_context(self, flags, t, text_groups, ip_scale, knock=(), freeze_text=False, text_neutral=None):
        self.ctx.cond = cond_vector(flags, t)
        self.ctx.text_groups = text_groups
        self.ctx.text_neutral = text_neutral
        self.ctx.ip_scale = ip_scale
        self.ctx.knock = set(knock)
        self.ctx.freeze_text = freeze_text

    def mix_pooled(self, pooled, pooled_colour=None, pooled_material=None):
        """Pooled text embedding with the global colour / material gates applied (call after set_context)."""
        g = self.pooled(self.ctx.cond.to(pooled.device))                                  # (B,2)
        if self.ctx.freeze_text:
            g = torch.ones_like(g)
        for i, (name, pn) in enumerate((("text_colour", pooled_colour), ("text_material", pooled_material))):
            if pn is None:
                continue
            gi = torch.zeros_like(g[:, i]) if name in self.ctx.knock else g[:, i]
            if self.ctx.collect is not None:
                self.ctx.collect.append((f"pooled.{name}", gi))
            pooled = pooled + (gi[:, None] - 1).to(pooled.dtype) * (pooled - pn.to(pooled.device, pooled.dtype))
        return pooled

    # ---------------------------------------------------------------- ControlNet residuals
    def sketch_groups(self, shapes):
        """Residual index -> 'sketch_contour' | 'sketch_placement' from the spatial sizes."""
        sizes = sorted({s[-1] for s in shapes}, reverse=True)
        high = set(sizes[:2])
        return ["sketch_contour" if s[-1] in high else "sketch_placement" for s in shapes]

    def gate_controlnet(self, down, mid):
        res = list(down) + [mid]
        names = self.sketch_groups([r.shape for r in res])
        out = []
        for r, gate, name in zip(res, self.sketch, names):
            g = gate(r, self.ctx.cond.to(r.device))
            if name in self.ctx.knock:
                g = torch.zeros_like(g)
            if self.ctx.collect is not None:
                self.ctx.collect.append((f"controlnet.{len(out)}", g))
            if self.ctx.record is not None:
                self.ctx.record[f"controlnet.{len(out)}"] = g.detach().float().cpu()
            out.append(r * g.to(r.dtype))
        return out[:-1], out[-1]

    # ---------------------------------------------------------------- io
    def save(self, path, extra=None):
        torch.save({"cfg": self.cfg, "state": self.state_dict(), **(extra or {})}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = cls(**ck["cfg"])
        missing, unexpected = m.load_state_dict(ck["state"], strict=False)
        if unexpected or any(not k.startswith("pooled.") for k in missing):
            raise RuntimeError(f"role gate checkpoint mismatch: missing {missing}, unexpected {unexpected}")
        return m.to(device)


def gate_maps(record, size=None):
    """Average recorded cross-attention gates per attribute over all layers -> {gate name: (H,W) tensor} for the
    conditional half of a CFG batch (index -1). `size` resamples every layer's map to one resolution."""
    acc = {n: [] for n in CROSS_GATES + SKETCH_GATES}
    target = size
    cn = {k: v for k, v in record.items() if k.startswith("controlnet")}
    for name, g in record.items():
        if name.startswith("controlnet"):
            continue
        n = g.shape[1]
        s = int(round(n ** 0.5))
        if s * s != n:
            continue
        m = g[-1].view(s, s, -1).permute(2, 0, 1)                                     # (6,s,s)
        target = target or s
        m = F.interpolate(m[None], size=(target, target), mode="bilinear", align_corners=False)[0]
        for i, gn in enumerate(CROSS_GATES):
            acc[gn].append(m[i])
    if cn:
        high = set(sorted({v.shape[-1] for v in cn.values()}, reverse=True)[:2])
        target = target or max(v.shape[-1] for v in cn.values())
        for v in cn.values():
            m = F.interpolate(v[-1:].float(), size=(target, target), mode="bilinear", align_corners=False)[0, 0]
            acc["sketch_contour" if v.shape[-1] in high else "sketch_placement"].append(m)
    return {k: torch.stack(v).mean(0) for k, v in acc.items() if v}
