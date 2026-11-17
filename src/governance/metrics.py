"""
Adherence metrics: did the output follow the modality that governs each attribute?
Pure numpy (no torch), so it is unit-tested locally.

Colour is measured without CLIP: object pixels are assigned to the nearest of 14 named colours in CIELAB.
"""
import numpy as np

NAMED_RGB = {
    "red": (200, 30, 30), "blue": (40, 70, 200), "green": (40, 150, 60), "yellow": (240, 220, 40),
    "orange": (240, 140, 30), "purple": (130, 50, 170), "pink": (245, 150, 190), "black": (20, 20, 20),
    "white": (245, 245, 245), "brown": (120, 75, 40), "gray": (128, 128, 128), "silver": (196, 198, 202),
    "gold": (212, 175, 55), "beige": (225, 205, 170),
}
NAMES = list(NAMED_RGB)
# colours that cannot be told apart reliably from pixels alone are scored as one
EQUIV = {"silver": "gray", "grey": "gray"}


def canon(c):
    return EQUIV.get(c, c)


def rgb_to_lab(rgb):
    """rgb: (..., 3) in 0..255 -> CIELAB (D65)."""
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


_ANCHORS = rgb_to_lab(np.array([NAMED_RGB[n] for n in NAMES]))


def name_hist(pixels):
    """pixels (N,3) RGB -> {colour name: share of pixels}, with equivalent names merged."""
    px = np.asarray(pixels, dtype=np.float64).reshape(-1, 3)
    if len(px) == 0:
        return {}
    if len(px) > 20000:
        px = px[np.random.default_rng(0).choice(len(px), 20000, replace=False)]
    idx = np.linalg.norm(rgb_to_lab(px)[:, None, :] - _ANCHORS[None], axis=-1).argmin(1)
    h = {}
    for i, n in enumerate(NAMES):
        share = float((idx == i).mean())
        if share:
            h[canon(n)] = h.get(canon(n), 0.0) + share
    return h


def dominant(pixels, top=2, min_share=0.15):
    h = name_hist(pixels)
    return [n for n, s in sorted(h.items(), key=lambda kv: -kv[1])[:top] if s >= min_share]


def colour_adherence(gen_pixels, source_colours, other_colours=()):
    """share / hit of the generated object's colour on the governing source; leak = share on the other source."""
    h = name_hist(gen_pixels)
    src = {canon(c) for c in source_colours}
    oth = {canon(c) for c in other_colours} - src
    dom = dominant(gen_pixels, top=1, min_share=0.0)
    return {
        "col_share": float(sum(h.get(c, 0.0) for c in src)) if src else float("nan"),
        "col_hit": float(bool(dom) and dom[0] in src) if src else float("nan"),
        "col_leak": float(sum(h.get(c, 0.0) for c in oth)) if oth else float("nan"),
    }


def object_region(thin, pad=4):
    """Filled sketch region from a boolean stroke mask (falls back to the stroke bounding box)."""
    import scipy.ndimage as ndi
    thin = np.asarray(thin, bool)
    if not thin.any():
        return np.ones_like(thin)
    filled = ndi.binary_fill_holes(ndi.binary_dilation(thin, iterations=pad))
    if filled.sum() < 2.5 * ndi.binary_dilation(thin, iterations=pad).sum():      # open outline: nothing got filled
        ys, xs = np.where(thin)
        filled = np.zeros_like(thin)
        filled[max(ys.min() - pad, 0):ys.max() + pad + 1, max(xs.min() - pad, 0):xs.max() + pad + 1] = True
    return filled


def foreground_pixels(img_rgb, region=None, tol=30):
    """Pixels inside `region` that differ from the border colour (drops a flat background)."""
    a = np.asarray(img_rgb, dtype=np.float64)
    border = np.concatenate([a[:6].reshape(-1, 3), a[-6:].reshape(-1, 3), a[:, :6].reshape(-1, 3), a[:, -6:].reshape(-1, 3)])
    m = np.linalg.norm(a - np.median(border, 0), axis=-1) > tol
    if region is not None:
        m &= np.asarray(region, bool)
        if m.sum() < 50:
            m = np.asarray(region, bool)
    return a[m] if m.sum() >= 50 else a.reshape(-1, 3)


def paired_bootstrap(diffs, groups, n=5000, seed=0):
    """Mean difference with a 95% interval, resampling GROUPS (base photos), not items."""
    d, g = np.asarray(diffs, float), np.asarray(groups)
    keep = ~np.isnan(d)
    d, g = d[keep], g[keep]
    if len(d) == 0:
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"), n=0, groups=0)
    ids = np.unique(g)
    per = [d[g == i] for i in ids]
    rng = np.random.default_rng(seed)
    bs = [np.concatenate([per[j] for j in rng.integers(0, len(ids), len(ids))]).mean() for _ in range(n)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(mean=float(d.mean()), lo=float(lo), hi=float(hi), n=int(len(d)), groups=int(len(ids)))
