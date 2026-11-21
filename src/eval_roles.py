"""
Role-compliance evaluation: for every kind of conflict, did the input that SHOULD govern the disputed attribute win?

    conflict row (what disagrees)                     governing input   primary metric (EvalKit, rule="reference")
    obj  caption names another object than the sketch   text            obj_hit   (image reads as the caption's object), id_leak (lower = better)
    col  caption names a wrong colour                   reference       col_hit   (reference colour present), col_leak (caption colour present; lower)
    mat  caption names a wrong material                 reference       mat_hit
    ref  reference swapped for another photo's          reference       col_hit   (the new reference's colour)
    all  sketch is the only source of the outline       sketch          edge_iou  (higher = outline kept)
    base unchanged rows                                 -               edge_iou, dino_tgt (no regression)

Arms are functions (item, row) -> PIL image. Images are written to disk one by one and skipped if present, so a runtime
restart loses nothing.
"""
import os

import numpy as np
from PIL import Image

from governance import labels as L

KINDS = ("obj", "col", "mat", "ref")
GOVERNING = dict(obj="text", col="reference", mat="reference", ref="reference", base="-")
PRIMARY = dict(obj="obj_hit", col="col_hit", mat="mat_hit", ref="col_hit")
LOWER_BETTER = dict(obj="id_leak", col="col_leak")


def kind_of(row):
    """'base' for an unchanged row, else the variant tag ('obj' / 'col' / 'mat' / 'ref' / 'agr')."""
    orig = row.get("original_id", row["id"])
    if row["id"] == orig:
        return "base"
    tag = row["id"][len(orig) + 1:]
    return tag.rstrip("0123456789")


def eval_rows(base_rows, split="test", n_photos=12, kinds=KINDS, seed=1, include_base=True):
    """Held-out rows: the first `n_photos` photos of the split, each as base row + one variant per kind."""
    rows = [r for r in L.expand_rows(base_rows, per_type=1, seed=seed, kinds=kinds) if r["split"] == split]
    photos = sorted({r["original_id"] for r in rows})[:n_photos]
    out = [r for r in rows if r["original_id"] in photos and (include_base or kind_of(r) != "base")]
    return sorted(out, key=lambda r: (r["original_id"], kind_of(r)))


def image_path(out_dir, arm, row):
    return os.path.join(out_dir, arm, f"{row['id']}.png")


def generate(rows, items, arms, out_dir, log=print):
    """items: {row id: item from Pipeline.make_items}. arms: {name: fn(item, row) -> PIL}. Resumable."""
    for arm, fn in arms.items():
        os.makedirs(os.path.join(out_dir, arm), exist_ok=True)
        todo = [r for r in rows if not os.path.exists(image_path(out_dir, arm, r))]
        log(f"arm {arm}: {len(todo)} to generate, {len(rows) - len(todo)} done")
        for i, r in enumerate(todo):
            fn(items[r["id"]], r).save(image_path(out_dir, arm, r))
            if (i + 1) % 10 == 0:
                log(f"  {arm}: {i + 1}/{len(todo)}")


def score(kit, rows, arms, out_dir, data_dir):
    """-> list of dicts (arm, kind, id, metrics...)."""
    import json
    cache = os.path.join(out_dir, "scores.json")
    done = {(d["arm"], d["id"]): d for d in json.load(open(cache))} if os.path.exists(cache) else {}
    for arm in arms:
        for r in rows:
            if (arm, r["id"]) in done:
                continue
            img = Image.open(image_path(out_dir, arm, r)).convert("RGB")
            sk = Image.open(os.path.join(data_dir, r["sketch"])); ref = Image.open(os.path.join(data_dir, r["reference"])).convert("RGB")
            tgt = Image.open(os.path.join(data_dir, r["target"])).convert("RGB") if kind_of(r) == "base" and r.get("target") else None
            m = kit.score(img, r, sk, ref, tgt)
            done[(arm, r["id"])] = dict(arm=arm, id=r["id"], kind=kind_of(r), **{k: v for k, v in m.items() if not isinstance(v, str)})
    out = list(done.values())
    json.dump(out, open(cache, "w"))
    return out


def tables(scores, arms):
    """Returns (compliance, shape): two DataFrames.

    compliance: one row per conflict kind: governing input, then the primary metric per arm (and the 'lower is better'
    metric, if any, in brackets). shape: edge_iou and base-row fidelity per arm."""
    import pandas as pd
    df = pd.DataFrame(scores)
    comp = []
    for kind in KINDS:
        sub = df[df.kind == kind]
        row = {"conflict": kind, "governs": GOVERNING[kind], "n": int(sub.id.nunique())}
        for arm in arms:
            s = sub[sub.arm == arm]
            cell = f"{s[PRIMARY[kind]].mean():.2f}" if len(s) else "-"
            if kind in LOWER_BETTER and len(s):
                cell += f" ({LOWER_BETTER[kind]} {s[LOWER_BETTER[kind]].mean():.2f})"
            row[arm] = cell
        comp.append(row)
    shape = []
    for arm in arms:
        a = df[df.arm == arm]
        base = a[a.kind == "base"]
        shape.append({"arm": arm, "edge_iou (all rows)": round(a.edge_iou.mean(), 3),
                      "edge_iou (conflict rows)": round(a[a.kind != "base"].edge_iou.mean(), 3),
                      "dino_tgt (base rows)": round(base.dino_tgt.mean(), 3) if "dino_tgt" in base and len(base) else float("nan"),
                      "clip_t (all rows)": round(a.clip_t.mean(), 3)})
    return pd.DataFrame(comp), pd.DataFrame(shape)
