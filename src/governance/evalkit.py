"""
Held-out evaluation for one generated image. Covers the five metric families of the project brief plus
per-attribute adherence. Uses models that are NOT used for generation or for the governance network
(CLIP-B/32, DINOv2-small, LPIPS, Canny, CLIPSeg), as in notebook 04.

    text-image alignment   clip_t, sem_text, sem_sketch, sem_margin, obj_hit
    sketch consistency     edge_recall, edge_precision, edge_iou
    reference similarity   dino_ref, attr_B, col_share / col_hit / col_leak, mat_hit
    perceptual quality     lpips_tgt, dino_tgt                    (conflict-free rows that have a target photo)
    layout consistency     layout_sketch_iou (all rows), layout_tgt_iou (rows with a target)
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from . import metrics as M
from .features import MEAN_C, STD_C
from .labels import MATERIALS, caption_colors, caption_materials, ref_colors, ref_has_material

CLIPB_ID, DINO_ID, SEG_ID = "openai/clip-vit-base-patch32", "facebook/dinov2-small", "CIDAS/clipseg-rd64-refined"
EV = 256


def thin_mask(sketch_white_on_black, size=EV):
    t = torch.from_numpy(np.array(sketch_white_on_black.convert("L"))).float()[None, None] / 255
    t = F.interpolate(t, size=(size, size), mode="area")[0, 0].numpy()
    return t > 0.35 * t.max()


class EvalKit:
    def __init__(self, device=None, use_lpips=True, use_clipseg=True):
        from transformers import (AutoModel, CLIPSegForImageSegmentation, CLIPSegProcessor, CLIPTextModelWithProjection,
                                  CLIPTokenizer, CLIPVisionModelWithProjection)
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dt = torch.float16 if str(self.dev).startswith("cuda") else torch.float32
        self.cbv = CLIPVisionModelWithProjection.from_pretrained(CLIPB_ID, torch_dtype=self.dt).to(self.dev).eval()
        self.cbt = CLIPTextModelWithProjection.from_pretrained(CLIPB_ID, torch_dtype=self.dt).to(self.dev).eval()
        self.tok = CLIPTokenizer.from_pretrained(CLIPB_ID)
        self.dino = AutoModel.from_pretrained(DINO_ID, torch_dtype=self.dt).to(self.dev).eval()
        self.mc = torch.tensor(MEAN_C, device=self.dev).view(1, 3, 1, 1); self.sc = torch.tensor(STD_C, device=self.dev).view(1, 3, 1, 1)
        self.mi = torch.tensor([0.485, 0.456, 0.406], device=self.dev).view(1, 3, 1, 1)
        self.si = torch.tensor([0.229, 0.224, 0.225], device=self.dev).view(1, 3, 1, 1)
        self.lp = self.seg = None
        if use_lpips:
            try:
                import lpips
                self.lp = lpips.LPIPS(net="alex", verbose=False).to(self.dev).eval()
            except Exception as e:
                print("LPIPS unavailable:", e)
        if use_clipseg:
            try:
                self.seg = CLIPSegForImageSegmentation.from_pretrained(SEG_ID).to(self.dev).eval()
                self.seg_p = CLIPSegProcessor.from_pretrained(SEG_ID)
            except Exception as e:
                print("CLIPSeg unavailable:", e)
        self._txt = {}
        self.mat_names = [s for _, s in MATERIALS]
        self.mat_emb = self.text([f"an object made of {m}" for m in self.mat_names])

    def _t(self, pil, size=None):
        if size:
            pil = pil.resize((size, size), Image.BICUBIC)
        return torch.from_numpy(np.array(pil.convert("RGB"))).permute(2, 0, 1)[None].float().to(self.dev) / 255

    @torch.no_grad()
    def clip_img(self, pil):
        x = F.interpolate(self._t(pil), size=(224, 224), mode="bilinear", align_corners=False)
        return F.normalize(self.cbv(pixel_values=((x - self.mc) / self.sc).to(self.dt)).image_embeds.float(), dim=-1)

    @torch.no_grad()
    def text(self, strs):
        tk = self.tok(list(strs), padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(self.dev)
        return F.normalize(self.cbt(input_ids=tk.input_ids, attention_mask=tk.attention_mask).text_embeds.float(), dim=-1)

    def t1(self, s):
        if s not in self._txt:
            self._txt[s] = self.text([s])
        return self._txt[s]

    @torch.no_grad()
    def dino_img(self, pil):
        x = F.interpolate(self._t(pil), size=(224, 224), mode="bilinear", align_corners=False)
        return F.normalize(self.dino(pixel_values=((x - self.mi) / self.si).to(self.dt)).last_hidden_state[:, 0].float(), dim=-1)

    @torch.no_grad()
    def seg_mask(self, pil, name):
        """CLIPSeg object mask at EV x EV (same recipe as src/eval_metrics.py)."""
        import scipy.ndimage as ndi
        pil = pil.convert("RGB").resize((EV, EV))
        x = self.seg_p(text=[str(name).replace("_", " ")], images=[pil], padding=True, return_tensors="pt").to(self.dev)
        lg = self.seg(**x).logits.float()
        p = F.interpolate(torch.sigmoid(lg).reshape(1, 1, *lg.shape[-2:]), size=(EV, EV), mode="bilinear")[0, 0].cpu().numpy()
        m = p > max(0.2, 0.5 * float(p.max()))
        lab, n = ndi.label(m)
        if n:
            sizes = np.array(ndi.sum(m, lab, range(1, n + 1)))
            m = ndi.binary_fill_holes(np.isin(lab, [i + 1 for i, s in enumerate(sizes) if s >= 0.25 * sizes.max()]))
        return m

    @staticmethod
    def iou(a, b):
        u = (a | b).sum()
        return float((a & b).sum() / u) if u else float("nan")

    def score(self, img, row, sketch, ref, target=None):
        """img: generated PIL. row: metadata row. sketch: white-on-black PIL. ref: reference PIL. target: PIL or None."""
        import cv2
        out = {}
        cap_cat, sk_cat = row.get("caption_category", row["category"]), row["category"]
        img256 = img.convert("RGB").resize((EV, EV))
        # --- sketch consistency (Canny)
        E = cv2.Canny(np.array(img256.convert("L")), 80, 160) > 0
        thin = thin_mask(sketch); k = np.ones((7, 7), np.uint8)
        Ed, Td = cv2.dilate(E.astype(np.uint8), k) > 0, cv2.dilate(thin.astype(np.uint8), k) > 0
        out["edge_recall"] = float((thin & Ed).sum() / max(thin.sum(), 1))
        out["edge_precision"] = float((E & Td).sum() / max(E.sum(), 1))
        out["edge_iou"] = float((Ed & Td).sum() / max((Ed | Td).sum(), 1))
        # --- text-image alignment (CLIP-B/32)
        ci = self.clip_img(img)
        out["clip_t"] = float((ci * self.t1(row["caption"])).sum())
        out["sem_text"] = float((ci * self.t1(f"a photo of a {cap_cat.replace('_', ' ')}")).sum())
        out["sem_sketch"] = float((ci * self.t1(f"a photo of a {sk_cat.replace('_', ' ')}")).sum())
        out["sem_margin"] = out["sem_text"] - out["sem_sketch"]
        out["obj_hit"] = float(out["sem_margin"] > 0) if cap_cat != sk_cat else float("nan")     # text governs identity
        # --- reference similarity
        out["dino_ref"] = float((self.dino_img(img) * self.dino_img(ref)).sum())
        if all(k_ in row for k_ in ("ref_finish", "ref_material", "ref_colors")):
            out["attr_B"] = float((ci * self.t1(f"a {row['ref_finish']} {row['ref_material']} object in {row['ref_colors']} colors")).sum())
        # --- colour adherence: the governing source is the text if it states a colour, otherwise the reference
        region = M.object_region(thin)
        gen_px = M.foreground_pixels(np.array(img256), region)
        cc, rc = caption_colors(row["caption"]), ref_colors(row)
        if not rc:
            rc = M.dominant(M.foreground_pixels(np.array(ref.convert("RGB").resize((EV, EV)))))
        out.update(M.colour_adherence(gen_px, cc if cc else rc, rc if cc else ()))
        out["col_governed_by"] = "text" if cc else "reference"
        # --- material adherence (CLIP-B zero-shot over 8 materials; weak, report as secondary)
        pred = self.mat_names[int((ci @ self.mat_emb.T).argmax())]
        cm = caption_materials(row["caption"])
        want = cm if cm else [s for s in self.mat_names if ref_has_material(row, s)]
        out["mat_hit"] = float(pred in want) if want else float("nan")
        out["mat_governed_by"] = "text" if cm else "reference"
        # --- layout consistency
        if self.seg is not None:
            gm = self.seg_mask(img, cap_cat)
            out["layout_sketch_iou"] = self.iou(gm, region)
            if target is not None:
                out["layout_tgt_iou"] = self.iou(gm, self.seg_mask(target, sk_cat))
        # --- perceptual quality / fidelity to the real photo (only meaningful when the row is conflict-free)
        if target is not None:
            out["dino_tgt"] = float((self.dino_img(img) * self.dino_img(target)).sum())
            if self.lp is not None:
                with torch.no_grad():
                    out["lpips_tgt"] = float(self.lp(self._t(img, EV) * 2 - 1, self._t(target, EV) * 2 - 1))
        return out
