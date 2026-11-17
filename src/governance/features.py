"""
Frozen CLIP ViT-L/14 features for the governance network.

The same functions are used when caching the training set (notebook 05) and inside the generation pipeline
(src/pipeline.py), so the network sees identical features at training and at inference time.

Input slots, in this order:
    cap     caption text
    obj     "a photo of a {object}" text
    sketch  sketch image (dark lines on white)
    ref     reference image
    grey    greyscale copy of the reference          (texture / material without hue)
    col     ref - grey                               (colour part, as in SADis, arXiv 2503.14275)
"""
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

CLIP_ID = "openai/clip-vit-large-patch14"
SLOTS = ["cap", "obj", "sketch", "ref", "grey", "col"]
MEAN_C = (0.48145466, 0.4578275, 0.40821073)
STD_C = (0.26862954, 0.26130258, 0.27577711)


def letterbox(im, size, fill):
    im = ImageOps.contain(im, (size, size), Image.LANCZOS)
    c = Image.new("RGB", (size, size), fill)
    c.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    return c


def on_white(im):
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        im = Image.alpha_composite(Image.new("RGBA", im.size, (255, 255, 255, 255)), im)
    return im.convert("RGB")


def to_grey(im):
    return ImageOps.grayscale(im.convert("RGB")).convert("RGB")


def sketch_for_clip(im):
    """Any sketch -> dark lines on white, letterboxed to 224 (what CLIP sees)."""
    g = on_white(im).convert("L")
    if np.array(g).mean() < 127:
        g = ImageOps.invert(g)
    return letterbox(g.convert("RGB"), 224, (255, 255, 255))


def ref_for_clip(im):
    return letterbox(on_white(im), 512, (127, 127, 127))


def object_phrase(obj):
    return f"a photo of a {str(obj).replace('_', ' ')}"


class ClipL:
    """CLIP ViT-L/14 image + text towers. fp16 on GPU, fp32 on CPU."""

    def __init__(self, device=None):
        from transformers import CLIPTextModelWithProjection, CLIPTokenizer, CLIPVisionModelWithProjection
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dt = torch.float16 if str(self.dev).startswith("cuda") else torch.float32
        self.vis = CLIPVisionModelWithProjection.from_pretrained(CLIP_ID, torch_dtype=self.dt).to(self.dev).eval()
        self.txt = CLIPTextModelWithProjection.from_pretrained(CLIP_ID, torch_dtype=self.dt).to(self.dev).eval()
        self.tok = CLIPTokenizer.from_pretrained(CLIP_ID)
        for m in (self.vis, self.txt):
            m.requires_grad_(False)
        self.mean = torch.tensor(MEAN_C, device=self.dev).view(1, 3, 1, 1)
        self.std = torch.tensor(STD_C, device=self.dev).view(1, 3, 1, 1)

    def image(self, x):
        """x: (B,3,H,W) in [0,1]. Not wrapped in no_grad so guidance can differentiate through it."""
        x = F.interpolate(x.float(), size=(224, 224), mode="bilinear", align_corners=False)
        return F.normalize(self.vis(pixel_values=((x - self.mean) / self.std).to(self.dt)).image_embeds.float(), dim=-1)

    @torch.no_grad()
    def image_pil(self, pil):
        x = torch.from_numpy(np.array(pil.convert("RGB"))).permute(2, 0, 1)[None].float().to(self.dev) / 255
        return self.image(x)[0].cpu()

    @torch.no_grad()
    def text(self, strs):
        tk = self.tok(list(strs), padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(self.dev)
        return F.normalize(self.txt(input_ids=tk.input_ids, attention_mask=tk.attention_mask).text_embeds.float(), dim=-1).cpu()


def build_x(cap, obj, sketch, ref, grey):
    """Five unit embeddings (d,) -> governance input (6, d). `col` is left un-normalised; the net normalises it."""
    return torch.stack([cap, obj, sketch, ref, grey, ref - grey]).float()


def six_cosines(cap, obj, attr, sketch, ref):
    """The detector of the old router (notebook 04, conflict_features). Kept as the baseline."""
    c = lambda a, b: float((a * b).sum())
    return [1 - c(obj, sketch), 1 - c(obj, ref), 1 - c(cap, attr), 1 - c(sketch, ref), c(cap, sketch), c(cap, ref)]


def attr_phrase(r):
    return f"a {r['ref_finish']} {r['ref_material']} object in {r['ref_colors']} colors"


@torch.no_grad()
def cache_rows(rows, data_dir, clip, progress=None):
    """Embed every row. Images and texts are embedded once each and reused across rows.

    Returns dict(X=(N,6,d), C6=(N,6), Y=(N,3), group=[base photo id], split=[...], ids=[...]).
    Rows must carry `gates` (see labels.expand_rows).
    """
    from .labels import GATES
    T, I = {}, {}

    def t(s):
        if s not in T:
            T[s] = clip.text([s])[0]
        return T[s]

    def img(path, kind):
        k = (path, kind)
        if k not in I:
            im = Image.open(os.path.join(data_dir, path))
            if kind == "sketch":
                I[k] = clip.image_pil(sketch_for_clip(im))
            else:
                r = ref_for_clip(im)
                I[k] = clip.image_pil(to_grey(r) if kind == "grey" else r)
        return I[k]

    X, C6, Y = [], [], []
    for r in (progress(rows) if progress else rows):
        cap, obj = t(r["caption"]), t(object_phrase(r.get("caption_category", r["category"])))
        sk, ref, grey = img(r["sketch"], "sketch"), img(r["reference"], "ref"), img(r["reference"], "grey")
        X.append(build_x(cap, obj, sk, ref, grey))
        C6.append(six_cosines(cap, obj, t(attr_phrase(r)), sk, ref))
        Y.append([r["gates"][g] for g in GATES])
    return dict(X=torch.stack(X), C6=torch.tensor(C6, dtype=torch.float32), Y=torch.tensor(Y, dtype=torch.float32),
                group=[r.get("original_id", r["id"]) for r in rows], split=[r["split"] for r in rows],
                ids=[r["id"] for r in rows])
