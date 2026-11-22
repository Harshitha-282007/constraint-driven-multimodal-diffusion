"""Counterfactual Attribute Supervision (CAS) - data side.

Builds training samples whose TARGET IMAGE is known even when modalities conflict:
  real    : original (caption, sketch, reference, photo) quadruple (all modalities agree)
  cf_ref  : reference swapped for another photo's reference (different colour AND different object shape).
            Target = the original photo recoloured (masked Lab chroma transfer) to the swapped reference's colours.
            Geometry/identity unchanged (sketch + caption of the original), colour owned by the reference.
            Caption colour words are removed or replaced by a wrong colour (so text colour must lose).
  cf_cap  : original reference+photo but caption colour replaced by a wrong colour (text colour must lose).
Material is NOT synthesised (no pixel-level material editing exists here); it is supervised only by `real` pairs
and by the shape-invariance consistency loss. This is a stated limitation.
"""
import hashlib, json, os, random, re
import numpy as np
from PIL import Image, ImageOps

COLORS = ["red", "blue", "green", "yellow", "orange", "purple", "pink", "black", "white", "brown", "gray", "silver", "gold", "beige"]
# canonical sRGB values, used both for evaluation (caption-colour distance) and nothing else
COLOR_RGB = dict(red=(200, 30, 30), blue=(30, 60, 200), green=(40, 160, 60), yellow=(240, 220, 40), orange=(240, 140, 30),
                 purple=(130, 50, 170), pink=(240, 150, 190), black=(20, 20, 20), white=(240, 240, 240), brown=(110, 70, 40),
                 gray=(128, 128, 128), silver=(190, 190, 195), gold=(212, 175, 55), beige=(225, 210, 175))


def letterbox(im, size, fill):
    im = ImageOps.contain(im, (size, size), Image.LANCZOS)
    c = Image.new("RGB", (size, size), fill)
    c.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    return c


def prep_sketch(path, res):
    im = Image.open(path).convert("L")
    if np.array(im).mean() > 127:
        im = ImageOps.invert(im)
    return letterbox(im.convert("RGB"), res, (0, 0, 0))


def prep_ref(path, res):
    return letterbox(Image.open(path).convert("RGB"), res, (127, 127, 127))


def ref_object_mask(ref_rgb, thr=24):
    """Reference images sit on a neutral gray background (v2 data): object = pixels far from the border colour."""
    a = np.asarray(ref_rgb).astype(np.float32)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]]).reshape(-1, 3)
    bg = np.median(border, axis=0)
    return (np.abs(a - bg).sum(-1) > thr)


def _lab(rgb):
    from skimage.color import rgb2lab
    return rgb2lab(np.asarray(rgb).astype(np.float32) / 255.0)


def _rgb(lab):
    from skimage.color import lab2rgb
    return (np.clip(lab2rgb(lab), 0, 1) * 255).astype(np.uint8)


def recolor_object(photo, obj_mask, ref_rgb, ref_mask=None, l_blend=0.5):
    """Masked Lab transfer: object chroma (a,b) mean/std -> reference object's; luminance mean shifted by l_blend.
    Keeps the photo's shading/texture structure. Returns PIL image (same size as photo)."""
    P = _lab(photo); R = _lab(ref_rgb)
    rm = ref_object_mask(ref_rgb) if ref_mask is None else ref_mask
    m = np.asarray(obj_mask.resize(photo.size)) > 127 if isinstance(obj_mask, Image.Image) else obj_mask
    if m.sum() < 50 or rm.sum() < 50:
        return photo
    out = P.copy()
    for c in (1, 2):
        mu_p, sd_p = P[..., c][m].mean(), P[..., c][m].std() + 1e-3
        mu_r, sd_r = R[..., c][rm].mean(), R[..., c][rm].std() + 1e-3
        out[..., c] = np.where(m, (P[..., c] - mu_p) * np.clip(sd_r / sd_p, 0.5, 2.0) + mu_r, P[..., c])
    out[..., 0] = np.where(m, np.clip(P[..., 0] + l_blend * (R[..., 0][rm].mean() - P[..., 0][m].mean()), 0, 100), P[..., 0])
    res = _rgb(out)
    # feathered composite so the edge does not show a seam
    from scipy.ndimage import gaussian_filter
    soft = gaussian_filter(m.astype(np.float32), 1.5)[..., None]
    comp = res * soft + np.asarray(photo) * (1 - soft)
    return Image.fromarray(comp.astype(np.uint8))


def swap_caption_colour(caption, rng, remove=False, avoid=()):
    """Replace (or drop) colour words in the caption. Returns (new_caption, new_colour or None)."""
    present = [c for c in COLORS if re.search(rf"\b{c}\b", caption, re.I)]
    options = [c for c in COLORS if c not in avoid and c not in present]
    new = rng.choice(options)
    if remove:
        cap = re.sub(r"\b(" + "|".join(COLORS) + r")\b[ -]*", "", caption, flags=re.I)
        return (cap if cap.strip() else caption), None
    if present:
        cap = re.sub(rf"\b{present[0]}\b", new, caption, count=1, flags=re.I)
    else:
        cap = caption.rstrip(".") + f", {new}."
    return cap, new


def tokens(s):
    return set(re.findall(r"[a-z]+", str(s).lower()))


class CASDataset:
    """Index-addressable, deterministic: sample(i, epoch_seed) always returns the same item (resumable)."""

    def __init__(self, data_dir, meta_file, split="train", res=640, p=(0.4, 0.4, 0.2), seed=0):
        rows = [json.loads(l) for l in open(os.path.join(data_dir, meta_file)) if l.strip()]
        self.base = [r for r in rows if r["split"] == split and r.get("conflict_type", "none") == "none"]
        assert self.base, f"no base rows for split={split}"
        self.dir, self.res, self.p, self.seed = data_dir, res, p, seed

    def __len__(self):
        return len(self.base)

    def _open(self, rel):
        return os.path.join(self.dir, rel)

    def sample(self, step):
        rng = random.Random(int(hashlib.md5(f"{self.seed}|{step}".encode()).hexdigest()[:8], 16))
        a = self.base[rng.randrange(len(self.base))]
        kind = rng.choices(["real", "cf_ref", "cf_cap"], weights=self.p)[0]
        photo = Image.open(self._open(a["target"])).convert("RGB")
        photo = letterbox(photo, self.res, (127, 127, 127))
        mask = letterbox(Image.open(self._open(a["mask"])).convert("L").convert("RGB"), self.res, (0, 0, 0)).convert("L") if a.get("mask") else None
        sketch = prep_sketch(self._open(a["sketch"]), self.res)
        ref = prep_ref(self._open(a["reference"]), self.res)
        cap, target = a["caption"], photo
        if kind == "cf_ref" and mask is not None:
            others = [b for b in self.base if b["category"] != a["category"] and not (tokens(b.get("ref_colors", "")) & tokens(a.get("ref_colors", "")))]
            if others:
                b = rng.choice(others)
                ref = prep_ref(self._open(b["reference"]), self.res)
                target = recolor_object(photo, mask, ref)
                cap, _ = swap_caption_colour(a["caption"], rng, remove=rng.random() < 0.5, avoid=tokens(b.get("ref_colors", "")))
            else:
                kind = "real"
        elif kind == "cf_cap":
            cap, _ = swap_caption_colour(a["caption"], rng, avoid=tokens(a.get("ref_colors", "")))
        elif kind == "cf_ref":
            kind = "real"
        return dict(kind=kind, id=a["id"], caption=cap, target=target, sketch=sketch, ref=ref, mask=mask)


def to_tensor(im):
    import torch
    return torch.from_numpy(np.asarray(im.convert("RGB"))).permute(2, 0, 1).float() / 127.5 - 1.0


def shape_perturb(ref, rng):
    """Same appearance, different silhouette: random flip + rotation + anisotropic rescale, gray padding."""
    im = ref.copy()
    if rng.random() < 0.5:
        im = ImageOps.mirror(im)
    im = im.rotate(rng.uniform(-35, 35), resample=Image.BICUBIC, fillcolor=(127, 127, 127))
    w, h = im.size
    sx, sy = rng.uniform(0.6, 1.0), rng.uniform(0.6, 1.0)
    small = im.resize((max(8, int(w * sx)), max(8, int(h * sy))), Image.LANCZOS)
    c = Image.new("RGB", (w, h), (127, 127, 127))
    c.paste(small, (rng.randint(0, w - small.width), rng.randint(0, h - small.height)))
    return c
