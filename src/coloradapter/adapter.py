"""
ColourAdapter: a trained, colour-only reference channel for SDXL.

Why: the reference's global IP-Adapter vector mixes colour with shape and background, so it cannot be turned up
without damaging the object (follow-up 3 of the Method 1 report), and gates on it did not learn colour. Here the
reference's colour reaches the UNet through its own, newly trained path:

    palette (K x [L, a, b, share])  --MLP-->  T colour tokens (width D)
    every up-block / mid-block cross-attention layer l:
        out_l = out_l + alpha_l * to_out( softmax(q K_l^T) V_l ),   K_l, V_l = linear_l(tokens)   (no biases)

Zero-initialised (alpha_l = 0), so before training the pipeline is unchanged. The tokens are computed inside each
processor from the raw palette, so gradient checkpointing (which re-runs the processors in backward) never walks
a graph twice. The unconditional (classifier-free) half gets no palette and therefore adds exactly zero.

Install it AFTER the role gates (it wraps whatever processor is there) and remove it BEFORE removing them.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from coloradapter.palette import DIM as PAL_DIM, K as PAL_K

TOKENS = 4
WIDTH = 128
SCOPES = ("up_blocks", "mid_block")


class _Ctx:
    def __init__(self):
        self.palette = None       # (B, K, 4) float, zeros where there is none
        self.has = None           # (B,) float, 1 where the row has a palette
        self.scale = 1.0


class ColourProcessor(nn.Module):
    def __init__(self, inner, owner, name):
        super().__init__()
        self.inner, self.name = inner, name
        object.__setattr__(self, "owner", owner)            # not a registered submodule: the adapter must not live inside the UNet

    def __call__(self, attn, hidden_states, *args, **kwargs):
        out = self.inner(attn, hidden_states, *args, **kwargs)
        ctx = self.owner.ctx
        if ctx.palette is None or ctx.scale == 0.0:
            return out
        tok = self.owner.tokens(ctx.palette.to(hidden_states.device), ctx.has.to(hidden_states.device))       # (B,T,W)
        s = self.owner._safe[self.name]
        wk, wv = self.owner._k[s], self.owner._v[s]
        q = attn.to_q(hidden_states)
        B, N, C = q.shape
        h = attn.heads
        k = F.linear(tok, wk.weight.float()).to(q.dtype)
        v = F.linear(tok, wv.weight.float()).to(q.dtype)
        q, k, v = (t.view(B, -1, h, C // h).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, C)
        a = F.linear(a, attn.to_out[0].weight)                                  # output projection without its bias
        alpha = (self.owner._alpha[s] * ctx.scale).to(a.dtype)
        return out + alpha * a


class ColourAdapter(nn.Module):
    def __init__(self, layer_dims, tokens=TOKENS, width=WIDTH):
        """layer_dims: {attn2 processor name: inner_dim of that layer}."""
        super().__init__()
        self.layer_dims, self.n_tokens, self.width = dict(layer_dims), tokens, width
        self.mlp = nn.Sequential(nn.Linear(PAL_K * PAL_DIM, 256), nn.SiLU(), nn.Linear(256, tokens * width))
        self.norm = nn.LayerNorm(width)
        safe = lambda n: n.replace(".", "_")
        self._safe = {n: safe(n) for n in self.layer_dims}
        self._k = nn.ModuleDict({safe(n): nn.Linear(width, d, bias=False) for n, d in self.layer_dims.items()})
        self._v = nn.ModuleDict({safe(n): nn.Linear(width, d, bias=False) for n, d in self.layer_dims.items()})
        self._alpha = nn.ParameterDict({safe(n): nn.Parameter(torch.zeros(1)) for n in self.layer_dims})
        self.ctx = _Ctx()
        self._orig = {}

    # name-keyed views (processor names contain dots)
    @property
    def k(self):
        return {n: self._k[s] for n, s in self._safe.items()}

    @property
    def v(self):
        return {n: self._v[s] for n, s in self._safe.items()}

    @property
    def alpha(self):
        return {n: self._alpha[s] for n, s in self._safe.items()}

    def tokens(self, palette, has):
        t = self.mlp(palette.flatten(1).float()).view(len(palette), self.n_tokens, self.width)
        return self.norm(t) * has.float().view(-1, 1, 1)

    # ------------------------------------------------------------------ construction / (un)installation
    @classmethod
    def for_unet(cls, unet, **kw):
        dims = {}
        for name in unet.attn_processors:
            if ".attn2." in name and name.startswith(SCOPES):
                attn = unet.get_submodule(name[: -len(".processor")])
                dims[name] = attn.to_q.out_features
        return cls(dims, **kw)

    def set_palette(self, palette, has=None, scale=1.0):
        """palette (B,K,4) tensor (zeros for rows without one); has (B,) 0/1 (default: rows with any non-zero entry)."""
        palette = palette.float()
        self.ctx.palette = palette
        self.ctx.has = palette.flatten(1).abs().sum(1).gt(0).float() if has is None else has.float()
        self.ctx.scale = float(scale)

    def clear(self):
        self.ctx.palette = self.ctx.has = None

    def install(self, unet):
        procs = dict(unet.attn_processors)
        self._orig = {n: procs[n] for n in self.layer_dims}
        for n in self.layer_dims:
            procs[n] = ColourProcessor(self._orig[n], self, n)
        unet.set_attn_processor(procs)

    def remove(self, unet):
        if not self._orig:
            return
        procs = dict(unet.attn_processors)
        for n, p in self._orig.items():
            procs[n] = p
        unet.set_attn_processor(procs)
        self._orig = {}

    # ------------------------------------------------------------------ io
    def save(self, path, meta=None):
        torch.save(dict(state=self.state_dict(), dims=self.layer_dims, tokens=self.n_tokens, width=self.width, meta=meta or {}), path)

    @classmethod
    def load(cls, path, device="cpu"):
        d = torch.load(path, map_location=device, weights_only=False)
        m = cls(d["dims"], tokens=d["tokens"], width=d["width"])
        m.load_state_dict(d["state"])
        return m.to(device).eval()
