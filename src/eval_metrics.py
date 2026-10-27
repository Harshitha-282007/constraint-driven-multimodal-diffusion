"""
Metrics from the presentation's evaluation plan, computed on generated images for one split.

per-image : CLIP-T (vs caption), CLIP-I and DINO-I (vs reference image), DreamSim and LPIPS (vs target photo),
            layout_iou (CLIPSeg object mask of output vs of target; our mIoU), sketch_recall (ControlNet++-style
            condition extraction: fraction of sketch strokes that land on an edge of the output; the same number for
            the real target photo is reported as an oracle ceiling).
set-level : FID (output set vs target set). Meaningless below a few hundred images: printed with a warning.

Not here yet: AP, the GPT-4o attribute judge, and conflict-set scoring (next phase).
"""
import os, json
import numpy as np, pandas as pd, cv2
from PIL import Image
import scipy.ndimage as ndi


# ----------------------------------------------------------------------------- pure-numpy pieces (unit-tested)
def sketch_recall(sketch_img, photo_img, tol=6, size=512):
    sk = np.asarray(sketch_img.convert("L").resize((size, size), Image.BICUBIC)) < 128
    g = cv2.cvtColor(np.asarray(photo_img.convert("RGB").resize((size, size), Image.BICUBIC)), cv2.COLOR_RGB2GRAY)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    med = float(np.median(g))
    edges = cv2.Canny(g, int(max(0, 0.66 * med)), int(min(255, 1.33 * med))) > 0
    if sk.sum() == 0:
        return float("nan")
    dist = ndi.distance_transform_edt(~edges)
    return float((dist[sk] <= tol).mean())


def mask_iou(a, b):
    a, b = a.astype(bool), b.astype(bool)
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float("nan")


def _big(m, frac=0.25):
    lab, n = ndi.label(m)
    if n == 0:
        return m
    sizes = np.array(ndi.sum(m, lab, range(1, n + 1)))
    return ndi.binary_fill_holes(np.isin(lab, [i + 1 for i, s in enumerate(sizes) if s >= frac * sizes.max()]))


# ----------------------------------------------------------------------------- model-based pieces
class Evaluator:
    def __init__(self, device="cuda"):
        import torch, lpips
        from transformers import (CLIPModel, CLIPProcessor, AutoModel, AutoImageProcessor,
                                  CLIPSegProcessor, CLIPSegForImageSegmentation)
        from dreamsim import dreamsim
        self.torch, self.dev = torch, device
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
        self.clip_p = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.dino = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()
        self.dino_p = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        self.seg = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device).eval()
        self.seg_p = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.lp = lpips.LPIPS(net="alex").to(device).eval()
        self.ds, self.ds_p = dreamsim(pretrained=True, device=device)

    def _clip_img(self, im):
        t = self.torch
        with t.no_grad():
            x = self.clip_p(images=im, return_tensors="pt").to(self.dev)
            return t.nn.functional.normalize(self.clip.get_image_features(**x), dim=-1)[0]

    def _clip_txt(self, s):
        t = self.torch
        with t.no_grad():
            x = self.clip_p(text=[s], return_tensors="pt", padding=True, truncation=True, max_length=77).to(self.dev)
            return t.nn.functional.normalize(self.clip.get_text_features(**x), dim=-1)[0]

    def _dino(self, im):
        t = self.torch
        with t.no_grad():
            x = self.dino_p(images=im, return_tensors="pt").to(self.dev)
            return t.nn.functional.normalize(self.dino(**x).last_hidden_state[:, 0], dim=-1)[0]

    def _mask(self, im, cat):
        t, F = self.torch, self.torch.nn.functional
        W, H = im.size
        with t.no_grad():
            x = self.seg_p(text=[cat.replace("_", " ")], images=[im], padding=True, return_tensors="pt").to(self.dev)
            lg = self.seg(**x).logits.float()
            p = F.interpolate(t.sigmoid(lg).reshape(1, 1, *lg.shape[-2:]), size=(H, W), mode="bilinear")[0, 0].cpu().numpy()
        return _big(p > max(0.2, 0.5 * float(p.max())))

    def _lpips(self, a, b):
        t = self.torch
        f = lambda im: (t.from_numpy(np.asarray(im.convert("RGB").resize((256, 256)))).permute(2, 0, 1).float() / 127.5 - 1)[None].to(self.dev)
        with t.no_grad():
            return float(self.lp(f(a), f(b)))

    def _dreamsim(self, a, b):
        with self.torch.no_grad():
            return float(self.ds(self.ds_p(a.convert("RGB")).to(self.dev), self.ds_p(b.convert("RGB")).to(self.dev)))

    def score(self, gen, tgt, ref, sketch, caption, category):
        gen, tgt, ref = (im.convert("RGB").resize((512, 512)) for im in (gen, tgt, ref))
        return dict(
            clip_t=float(self._clip_img(gen) @ self._clip_txt(caption)),
            clip_i=float(self._clip_img(gen) @ self._clip_img(ref)),
            dino_i=float(self._dino(gen) @ self._dino(ref)),
            dreamsim=self._dreamsim(gen, tgt),            # lower = more similar
            lpips=self._lpips(gen, tgt),                  # lower = more similar
            layout_iou=mask_iou(self._mask(gen, category), self._mask(tgt, category)),
            sketch_recall=sketch_recall(sketch, gen),
            sketch_recall_oracle=sketch_recall(sketch, tgt))


def fid(gen_imgs, real_imgs, device="cuda"):
    import torch
    from torchmetrics.image.fid import FrechetInceptionDistance
    m = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    to_t = lambda ims: torch.stack([torch.from_numpy(np.asarray(i.convert("RGB").resize((299, 299)))).permute(2, 0, 1) for i in ims]).float().div(255).to(device)
    m.update(to_t(real_imgs), real=True); m.update(to_t(gen_imgs), real=False)
    return float(m.compute())


def evaluate_split(root, gen_dir, split="test", meta_file="metadata.jsonl", out_csv=None, device="cuda"):
    ev = Evaluator(device)
    rows = [json.loads(l) for l in open(os.path.join(root, meta_file))]
    rows = [r for r in rows if r["split"] == split and os.path.exists(os.path.join(gen_dir, f"{r['id']}.png"))]
    recs, gens, tgts = [], [], []
    for r in rows:
        op = lambda k: Image.open(os.path.join(root, r[k]))
        gen = Image.open(os.path.join(gen_dir, f"{r['id']}.png"))
        s = ev.score(gen, op("target"), op("reference"), op("sketch"), r["caption"], r["category"])
        recs.append(dict(id=r["id"], category=r["category"], **s)); gens.append(gen); tgts.append(op("target"))
    df = pd.DataFrame(recs)
    summary = df.drop(columns=["id", "category"]).mean().round(4).to_dict()
    if len(gens) >= 50:
        summary["fid"] = round(fid(gens, tgts, device), 2)
    else:
        print(f"[warn] only {len(gens)} images: FID skipped (needs a few hundred to mean anything)")
    if out_csv:
        df.to_csv(out_csv, index=False)
    return df, summary
