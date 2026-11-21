"""
Colour-only description of a reference object: a small palette in CIELAB.

The reference image carries colour, material, shape and background mixed together (that is why turning the global
IP-Adapter vector up drags the reference's scene into the output). The palette keeps only the colour: K colours with
their share of the object's pixels. Pure numpy; the same function is used for training references (hue-rotated) and
for inference references, so the two distributions match.
"""
import numpy as np

from governance.metrics import rgb_to_lab
from rolegate.augment import reference_mask

K = 4
DIM = 4                      # per colour: L/100, a/128, b/128, share of pixels


def _kmeans(x, k, iters=12):
    """x (N,3) -> centres (k,3), shares (k,). Farthest-point init so that small distinct colours get a cluster."""
    x = np.asarray(x, np.float64)
    c = [x.mean(0)]
    for _ in range(k - 1):
        d = np.min([((x - ci) ** 2).sum(1) for ci in c], axis=0)
        c.append(x[int(np.argmax(d))])
    c = np.stack(c)
    for _ in range(iters):
        lab = np.argmin(((x[:, None] - c[None]) ** 2).sum(-1), axis=1)
        for j in range(k):
            if (lab == j).any():
                c[j] = x[lab == j].mean(0)
    lab = np.argmin(((x[:, None] - c[None]) ** 2).sum(-1), axis=1)
    return c, np.bincount(lab, minlength=k) / len(x)


def palette(rgb, mask=None, k=K, max_pixels=6000, seed=0):
    """rgb (H,W,3) uint8 -> (k, 4) float32 sorted by share, zero-padded when the object has fewer colours.

    mask: object pixels; for a reference on a flat background it is found with rolegate.augment.reference_mask.
    """
    rgb = np.asarray(rgb)
    m = reference_mask(rgb) if mask is None else np.asarray(mask, bool)
    px = rgb[m] if m.any() else rgb.reshape(-1, 3)
    if len(px) > max_pixels:
        px = px[np.random.RandomState(seed).choice(len(px), max_pixels, replace=False)]
    lab = rgb_to_lab(px)
    centres, share = _kmeans(lab, min(k, len(lab)))
    order = np.argsort(-share)
    out = np.zeros((k, DIM), np.float32)
    for i, j in enumerate(order):
        out[i] = [centres[j, 0] / 100, centres[j, 1] / 128, centres[j, 2] / 128, share[j]]
    return out


def palette_distance(a, b):
    """Share-weighted mean CIELAB distance from every colour of `a` to the nearest colour of `b` (both (k,4))."""
    la = np.asarray(a)[:, :3] * np.array([100, 128, 128]); lb = np.asarray(b)[:, :3] * np.array([100, 128, 128])
    wa = np.asarray(a)[:, 3]
    d = np.sqrt(((la[:, None] - lb[None]) ** 2).sum(-1)).min(1)
    return float((wa * d).sum() / max(wa.sum(), 1e-8))
