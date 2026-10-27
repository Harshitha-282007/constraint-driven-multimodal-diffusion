"""
Dataset classes that plug the frozen Sketchy pilot into UniCombine's trainers.

Frozen layout (made by 01_freeze_pilot_dataset.ipynb):
    pilot_v1/
      metadata.jsonl      one json per sample (see freeze notebook)
      target/  sketch/  reference/

Batch format expected by UniCombine's train loops:
    pixel_values      (B, 3, H, W)      target image, in [-1, 1]
    condition_latents (B, n_cond, 3, H, W)   conditions, in [-1, 1], order == condition_types
    condition_types   list[list[str]]   e.g. [['sketch', 'reference']]
    descriptions      list[str]         caption
"""
import json, os, random
import torch
from PIL import Image
from torch.utils.data import Dataset
from diffusers.image_processor import VaeImageProcessor

# Condition types whose tokens are NOT spatially aligned with the target
# (they get a shifted position id, exactly like UniCombine's "subject").
SHIFTED_TYPES = ("subject", "reference")


class SketchyPilot(Dataset):
    def __init__(self, root, split="train", condition_types=("sketch",), resolution=512,
                 only_conflict_free=True, vae_scale_factor=8, caption_dropout=0.0):
        self.root, self.res = root, resolution
        self.cond_types = list(condition_types)
        self.caption_dropout = caption_dropout
        self.proc = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2, do_resize=True, do_convert_rgb=True)
        self.rows = []
        with open(os.path.join(root, "metadata.jsonl")) as f:
            for line in f:
                r = json.loads(line)
                if r["split"] != split:
                    continue
                if only_conflict_free and r.get("conflict_type", "none") != "none":
                    continue
                self.rows.append(r)
        assert self.rows, f"no rows for split={split}"

    def __len__(self):
        return len(self.rows)

    def _img(self, rel):
        im = Image.open(os.path.join(self.root, rel)).convert("RGB")
        return self.proc.preprocess(im, width=self.res, height=self.res).squeeze(0)

    def __getitem__(self, i):
        r = self.rows[i]
        target = self._img(r["target"])
        conds = []
        for t in self.cond_types:
            key = {"sketch": "sketch", "reference": "reference"}[t]
            conds.append(self._img(r[key]))
        caption = "" if random.random() < self.caption_dropout else r["caption"]
        return {"pixel_values": target,
                "condition_latents": torch.stack(conds),
                "description": caption,
                "condition_types": self.cond_types}


def collate_fn(examples):
    pv = torch.stack([e["pixel_values"] for e in examples]).contiguous().float()
    cl = torch.stack([e["condition_latents"] for e in examples]).contiguous().float()
    return {"pixel_values": pv, "condition_latents": cl,
            "condition_types": [e["condition_types"] for e in examples],
            "descriptions": [e["description"] for e in examples]}
