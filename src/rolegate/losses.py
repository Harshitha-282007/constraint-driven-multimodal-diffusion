"""
Training losses for the role gates. Everything here is differentiable torch (images in [0,1], (B,3,H,W)).

    masked_diffusion   noise loss weighted 1.0 inside the dilated object mask, `bg` outside      (Break-A-Scene)
    colour_loss        CIELAB mean inside the mask vs the target's                               (reference governs colour)
    texture_loss       gradient-energy mean/std inside the mask vs the target's                  (reference governs material)
    contour_loss       sketch-edge recall + penalty for edges off the sketch                     (sketch governs shape)
    identity_loss      CLIP hinge: closer to the caption's object than to the reference's object (text governs identity)
    swap_loss          two references, same text/sketch/noise: edges and identity must agree     (only colour may change)
    prior_loss         gates pulled toward the rules (annealed to a floor)
    keep_loss          sketch gates may open but not close: the sketch always governs shape and placement
"""
import torch
import torch.nn.functional as F

from .gates import CROSS_GATES, REF_OFF_INIT, STYLE_PREFIX

_M = torch.tensor([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
_WHITE = torch.tensor([0.95047, 1.0, 1.08883])


def rgb_to_lab(x):
    """(B,3,H,W) in [0,1] -> (B,3,H,W) CIELAB."""
    c = torch.where(x > 0.04045, ((x + 0.055) / 1.055).clamp_min(1e-8) ** 2.4, x / 12.92)
    xyz = torch.einsum("ij,bjhw->bihw", _M.to(x), c) / _WHITE.to(x).view(1, 3, 1, 1)
    f = torch.where(xyz > 0.008856, xyz.clamp_min(1e-8) ** (1 / 3), 7.787 * xyz + 16 / 116)
    return torch.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], 1)


def _masked_mean(v, m):
    """v (B,C,H,W), m (B,1,H,W) -> (B,C)."""
    return (v * m).sum((2, 3)) / m.sum((2, 3)).clamp_min(1.0)


def masked_diffusion(pred, target, mask_lat, bg=0.1):
    """pred/target (B,4,h,w), mask_lat (B,1,h,w) in {0,1}."""
    w = mask_lat + bg * (1 - mask_lat)
    return ((pred.float() - target.float()) ** 2 * w).sum() / (w.sum() * pred.shape[1])


def colour_loss(img, tgt, mask):
    la, lb = rgb_to_lab(img), rgb_to_lab(tgt)
    d = _masked_mean(la, mask) - _masked_mean(lb, mask)
    return (0.5 * d[:, 0].abs() + d[:, 1:].abs().sum(1)).mean() / 100


def _sobel(x):
    g = x.mean(1, keepdim=True)
    kx = torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    return (F.conv2d(g, kx, padding=1) ** 2 + F.conv2d(g, kx.transpose(2, 3), padding=1) ** 2 + 1e-6).sqrt()


def texture_loss(img, tgt, mask):
    ei, et = _sobel(img), _sobel(tgt)
    mi, mt = _masked_mean(ei, mask), _masked_mean(et, mask)
    si = _masked_mean((ei - mi[..., None, None]) ** 2, mask).sqrt()
    st = _masked_mean((et - mt[..., None, None]) ** 2, mask).sqrt()
    return ((mi - mt).abs() + (si - st).abs()).mean()


def contour_loss(img, sketch_mask):
    """sketch_mask (B,1,H,W) = dilated strokes in {0,1}."""
    e = _sobel(img)
    e = e / e.amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
    rec = (e * sketch_mask).sum((1, 2, 3)) / sketch_mask.sum((1, 2, 3)).clamp_min(1.0)
    out = (e * (1 - sketch_mask)).mean((1, 2, 3))
    return ((1 - rec) + 0.5 * out).mean()


def identity_loss(img_emb, caption_obj_emb, ref_obj_emb, margin=0.05):
    """unit embeddings (B,D): want cos(img, caption object) >= cos(img, reference object) + margin."""
    gap = (img_emb * caption_obj_emb).sum(-1) - (img_emb * ref_obj_emb).sum(-1)
    return F.relu(margin - gap).mean()


def swap_loss(img_a, img_b, emb_a, emb_b, mask):
    ea, eb = _sobel(img_a), _sobel(img_b)
    edges = ((ea - eb).abs() * mask).sum() / (mask.sum() * ea.shape[1]).clamp_min(1.0)
    ident = (1 - (emb_a * emb_b).sum(-1)).mean()
    return edges + ident


def prior_loss(collected, obj_mask, flags):
    """Pull the gates toward the rules.

    collected: [(name, gates)] from GateContext.collect (cross-attention (B,N,6), pooled (B,), ControlNet ignored);
    obj_mask (B,1,H,W) in {0,1}; flags (B,3) rule labels (object, colour, material). Targets: text colour / material
    gates 0 when that attribute is in conflict (else 1); text identity and other 1; reference gates at their layer's
    initial level on the object (1 in the style block, REF_OFF_INIT elsewhere: full IP-Adapter strength in every
    layer leaks the reference's layout and identity) and 0 off it.
    """
    total, n = 0.0, 0
    for name, g in collected:
        if name.startswith("controlnet"):
            continue
        if name.startswith("pooled."):
            col = 1 if name.endswith("colour") else 2
            total = total + F.mse_loss(g.float(), (1 - flags[:, col].float()).to(g.device))
            n += 1
            continue
        b, nq, _ = g.shape
        s = int(round(nq ** 0.5))
        if s * s != nq:
            continue
        m = F.interpolate(obj_mask.float(), size=(s, s), mode="area").view(b, nq, 1).to(g.device)
        lvl = 1.0 if name.startswith(STYLE_PREFIX) else REF_OFF_INIT
        tgt = torch.ones_like(g)
        tgt[..., CROSS_GATES.index("ref_colour")] = lvl * m[..., 0]
        tgt[..., CROSS_GATES.index("ref_texture")] = lvl * m[..., 0]
        tgt[..., CROSS_GATES.index("text_colour")] = (1 - flags[:, 1].float()).view(b, 1).to(g.device)
        tgt[..., CROSS_GATES.index("text_material")] = (1 - flags[:, 2].float()).view(b, 1).to(g.device)
        total = total + F.mse_loss(g.float(), tgt)
        n += 1
    return total / max(n, 1)


def keep_loss(collected, floor=0.9):
    """Penalise ControlNet residual gates below `floor` (squared hinge, mean over pixels and residuals)."""
    vals = [F.relu(floor - g.float()).pow(2).mean() for name, g in collected if name.startswith("controlnet")]
    return torch.stack(vals).mean() if vals else torch.zeros(())


def reg_loss(gates_module, init_state):
    """L1 distance of every gate parameter to its initial value."""
    total, n = 0.0, 0
    for k, p in gates_module.named_parameters():
        total = total + (p - init_state[k].to(p.device)).abs().mean()
        n += 1
    return total / max(n, 1)
