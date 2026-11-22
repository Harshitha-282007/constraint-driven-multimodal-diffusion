"""Attribute-specific metrics. Each metric targets ONE attribute so modality governance can be tested separately.
Object regions come from the SKETCH (filled outline), not from the generated image, so no segmenter is needed.
CLIP-based metrics need `transformers` + openai/clip-vit-base-patch32 (held-out: not used in training or guidance)."""
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.feature import canny
from skimage.color import rgb2lab, rgb2gray
from cas_data import COLOR_RGB, ref_object_mask


def _lab_mean(rgb, mask):
    lab = rgb2lab(np.asarray(rgb.convert("RGB")).astype(np.float32) / 255.0)
    return lab[mask].mean(0) if mask.sum() > 20 else np.full(3, np.nan)


def sketch_region(sketch_img, dilate=3):
    """Filled sketch outline. sketch_img: white lines on black (as fed to ControlNet)."""
    s = np.asarray(sketch_img.convert("L")) > 127
    s = ndi.binary_dilation(s, iterations=dilate)
    return ndi.binary_fill_holes(s)


def edges(img, sigma=2.0):
    return canny(rgb2gray(np.asarray(img.convert("RGB"))), sigma=sigma)


def edge_recall(curve_mask, out_edges, tol=6):
    """fraction of curve pixels (sketch strokes / reference contour) with an output edge within `tol` px."""
    if curve_mask.sum() == 0: return float("nan")
    d = ndi.distance_transform_edt(~out_edges)
    return float((d[curve_mask] <= tol).mean())


def contour(mask):
    return mask & ~ndi.binary_erosion(mask, iterations=2)


def geometry_scores(out, sketch_img, ref_img):
    """geo_sketch: do output edges lie on the sketch strokes. geo_ref_leak: do they lie on the reference's silhouette
    (reference silhouette is a different object by construction). geo_margin = sketch - leak."""
    e = edges(out)
    sk = np.asarray(sketch_img.convert("L")) > 127
    rc = contour(ref_object_mask(ref_img))
    a, b = edge_recall(sk, e), edge_recall(rc, e)
    return dict(geo_sketch=a, geo_ref_leak=b, geo_margin=a - b)


def colour_scores(out, sketch_img, ref_img, caption_colour=None):
    """Mean Lab chroma distance (ab only; lightness ignored) inside the sketch region between output and reference object."""
    m = sketch_region(sketch_img)
    o = _lab_mean(out, m); r = _lab_mean(ref_img, ref_object_mask(ref_img))
    d_ref = float(np.linalg.norm(o[1:] - r[1:]))
    res = dict(col_dist_ref=d_ref)
    if caption_colour in COLOR_RGB:
        c = rgb2lab(np.array([[COLOR_RGB[caption_colour]]], np.float32) / 255.0)[0, 0]
        d_cap = float(np.linalg.norm(o[1:] - c[1:]))
        res.update(col_dist_caption=d_cap, col_follows_ref=float(d_ref < d_cap))
    return res


class ClipJudge:
    def __init__(self, device="cuda", mid="openai/clip-vit-base-patch32"):
        from transformers import CLIPModel, CLIPProcessor
        self.m = CLIPModel.from_pretrained(mid).to(device).eval(); self.p = CLIPProcessor.from_pretrained(mid); self.dev = device

    def _img(self, ims):
        import torch
        with torch.no_grad():
            x = self.p(images=ims, return_tensors="pt").to(self.dev)
            f = self.m.get_image_features(**x)
        return f / f.norm(dim=-1, keepdim=True)

    def _txt(self, texts):
        import torch
        with torch.no_grad():
            x = self.p(text=texts, return_tensors="pt", padding=True).to(self.dev)
            f = self.m.get_text_features(**x)
        return f / f.norm(dim=-1, keepdim=True)

    def identity(self, out, categories, target_cat, other_cat=None):
        """zero-shot category prediction among all categories; returns top1 correctness for target (and for the competing one)."""
        t = self._txt([f"a photo of a {c}" for c in categories]); s = (self._img([out]) @ t.T)[0]
        pred = categories[int(s.argmax())]
        r = dict(id_top1=float(pred == target_cat), id_logit=float(s[categories.index(target_cat)] * 100))
        if other_cat: r["id_other_top1"] = float(pred == other_cat)
        return r

    def material(self, out, sketch_img, ref_material, other_material):
        """CLIP margin on the object crop (sketch bbox): sim('a {ref material} object') - sim('a {caption material} object')."""
        m = sketch_region(sketch_img); ys, xs = np.where(m)
        crop = out.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)) if len(xs) else out
        t = self._txt([f"a {ref_material} object", f"a {other_material} object"]); s = (self._img([crop]) @ t.T)[0]
        return dict(mat_margin=float((s[0] - s[1]) * 100))


def paired_bootstrap(a, b, n=2000, seed=0):
    """mean(a-b) with 95% CI over items (paired)."""
    d = np.asarray(a, float) - np.asarray(b, float); d = d[~np.isnan(d)]
    if len(d) == 0: return (float("nan"),) * 3
    rng = np.random.RandomState(seed); bs = [rng.choice(d, len(d)).mean() for _ in range(n)]
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
