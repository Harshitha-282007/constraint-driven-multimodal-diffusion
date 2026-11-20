"""
Colour augmentation with an exact target: rotate the object's hue in CIELAB (a*b* plane) inside its mask, and apply
the same rotation to the reference object. The recoloured photo is then the correct output for the recoloured
reference, whatever colour the caption names (rule: the reference governs colour).

Pure numpy, unit-tested locally.
"""
import numpy as np
import scipy.ndimage as ndi

from governance.metrics import rgb_to_lab

_M = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
_WHITE = np.array([0.95047, 1.0, 1.08883])


def lab_to_rgb(lab):
    lab = np.asarray(lab, np.float64)
    fy = (lab[..., 0] + 16) / 116
    fx, fz = fy + lab[..., 1] / 500, fy - lab[..., 2] / 200
    f = np.stack([fx, fy, fz], -1)
    xyz = np.where(f ** 3 > 0.008856, f ** 3, (f - 16 / 116) / 7.787) * _WHITE
    c = xyz @ np.linalg.inv(_M).T
    c = np.clip(c, 0, 1)
    c = np.where(c > 0.0031308, 1.055 * np.power(c, 1 / 2.4) - 0.055, 12.92 * c)
    return np.clip(c * 255, 0, 255)


def rotate_hue(rgb, mask, degrees, feather=2):
    """rgb (H,W,3) uint8/float, mask (H,W) bool -> uint8 image whose masked region has its hue rotated."""
    rgb = np.asarray(rgb, np.float64)
    lab = rgb_to_lab(rgb)
    th = np.deg2rad(degrees)
    a, b = lab[..., 1], lab[..., 2]
    rot = lab.copy()
    rot[..., 1], rot[..., 2] = a * np.cos(th) - b * np.sin(th), a * np.sin(th) + b * np.cos(th)
    out = lab_to_rgb(rot)
    w = ndi.gaussian_filter(np.asarray(mask, np.float64), feather)[..., None] if feather else np.asarray(mask, float)[..., None]
    return np.clip(w * out + (1 - w) * rgb, 0, 255).astype(np.uint8)


def mean_chroma(rgb, mask):
    lab = rgb_to_lab(np.asarray(rgb, np.float64)[np.asarray(mask, bool)])
    return float(np.hypot(lab[:, 1], lab[:, 2]).mean()) if len(lab) else 0.0


def reference_mask(rgb, tol=30, min_frac=0.02):
    """Foreground of a reference rendered on a flat background: pixels far from the border's median colour."""
    a = np.asarray(rgb, np.float64)
    border = np.concatenate([a[:6].reshape(-1, 3), a[-6:].reshape(-1, 3), a[:, :6].reshape(-1, 3), a[:, -6:].reshape(-1, 3)])
    m = np.linalg.norm(a - np.median(border, 0), axis=-1) > tol
    m = ndi.binary_opening(m, iterations=2)
    lab, n = ndi.label(m)
    if n:
        sizes = ndi.sum(m, lab, range(1, n + 1))
        m = lab == (1 + int(np.argmax(sizes)))
        m = ndi.binary_fill_holes(m)
    return m if m.mean() >= min_frac else np.ones(m.shape, bool)


def hue_pair(photo_rgb, photo_mask, ref_rgb, degrees, min_chroma=12.0):
    """Recoloured (photo, reference) with the same hue rotation, or None if the object is too grey to rotate."""
    if mean_chroma(photo_rgb, photo_mask) < min_chroma:
        return None
    rm = reference_mask(ref_rgb)
    if mean_chroma(ref_rgb, rm) < min_chroma:
        return None
    return rotate_hue(photo_rgb, photo_mask, degrees), rotate_hue(ref_rgb, rm, degrees)
