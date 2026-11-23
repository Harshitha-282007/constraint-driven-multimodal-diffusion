"""
Pilot: does flag-gated text conditioning move colour / material, and does it compose with the existing router?

One test photo, three conflict rows (object, colour, material), four arms, identical seed / resolution / steps:

    A_fixed             Pipeline.run(shaped=False)                               fixed guidance baseline (CENTER)
    B0_scheduled        Pipeline.run(shaped=True)                                control: schedule with centre values
    B_router            Pipeline.run(p=router params, shaped=True)               existing learned Attribute Routing Layer
    D_flaggate          Pipeline.run(shaped=False, text_flags=(f_col, f_mat))    A + flag-gated text (single change vs A)
    D_hard              Pipeline.run(shaped=False, text_flags=(1, 1))            control: flags forced to 1
    E_router_flaggate   Pipeline.run(p=router params, shaped=True, text_flags)   B + flag-gated text (router and gate together)

Flags come from the governance network (models on Drive); if it is unavailable the rule labels are used and the run is
labelled 'oracle flags'. Nothing is trained. Images, per-image timings, metrics and the config are written under out_dir;
every image that exists is skipped, so a rerun resumes and the set can be extended.
"""
import json
import os
import time

import numpy as np
import pandas as pd
from PIL import Image

import eval_roles as ER
from governance import labels as L
from governance.features import attr_phrase
from rolegate.tokens import neutral_caption

ARMS = ["A_fixed", "B0_scheduled", "B_router", "D_flaggate", "D_hard", "E_router_flaggate"]
KINDS = ("obj", "col", "mat")


def pick_photo(base_rows, tokenizer, split="test", seed=1):
    """First test photo (sorted id) with object, colour and material variants whose rule labels are on and whose
    colour / material words can be neutralised. Fixed before any image is generated."""
    rows = [r for r in L.expand_rows(base_rows, per_type=1, seed=seed, kinds=KINDS) if r["split"] == split]
    by = {}
    for r in rows:
        by.setdefault(r["original_id"], {})[ER.kind_of(r)] = r
    for oid in sorted(by):
        g = by[oid]
        if not set(KINDS) <= set(g):
            continue
        if not (g["obj"]["gates"]["g_obj"] and g["col"]["gates"]["g_col"] and g["mat"]["gates"]["g_mat"]):
            continue
        if neutral_caption(tokenizer, g["col"]["caption"], "colour") is None or neutral_caption(tokenizer, g["mat"]["caption"], "material") is None:
            continue
        return oid, [g[k] for k in KINDS]
    raise RuntimeError("no test photo satisfies the selection rule")


def pick_photos(base_rows, tokenizer, split="test", seed=1):
    """Every photo of the split, with each conflict kind (obj / col / mat) kept when its rule flag is on and, for colour /
    material, the word can be neutralised. Returns (photo ids, rows); unlike pick_photo no photo is dropped for lacking a kind."""
    rows = [r for r in L.expand_rows(base_rows, per_type=1, seed=seed, kinds=KINDS) if r["split"] == split]
    out, oids = [], set()
    for r in sorted(rows, key=lambda r: (r["original_id"], ER.kind_of(r))):
        k = ER.kind_of(r)
        if not r["gates"]["g_" + k]:
            continue
        if k != "obj" and neutral_caption(tokenizer, r["caption"], {"col": "colour", "mat": "material"}[k]) is None:
            continue
        out.append(r); oids.add(r["original_id"])
    return sorted(oids), out


def run_pilot(P, rows, data_dir, out_dir, net=None, router=None, kit=None, arms=ARMS, log=print):
    from governance import model as GM
    from pipeline import seed_for
    os.makedirs(out_dir, exist_ok=True)
    op = lambda r, k: Image.open(f"{data_dir}/{r[k]}")
    P.prepare_object_dirs([r.get("ref_object") or "object" for r in rows])
    specs = [dict(key=r["id"], sketch=op(r, "sketch"), ref=op(r, "reference"), caption=r["caption"], obj=r.get("caption_category", r["category"]),
                  seed=seed_for(r["original_id"]), ref_object=r.get("ref_object"), attr=attr_phrase(r)) for r in rows]
    items = dict(zip([r["id"] for r in rows], P.make_items(specs)))

    info = {}
    for r in rows:
        it = items[r["id"]]
        rule = {k: float(v) for k, v in r["gates"].items()}
        pred = GM.predict_gates(net, it["x"]) if net is not None else None
        use = pred if pred is not None else rule
        info[r["id"]] = dict(kind=ER.kind_of(r), caption=r["caption"], caption_category=r.get("caption_category", r["category"]), sketch_category=r["category"],
                             ref_colors=r.get("ref_colors"), ref_material=r.get("ref_material"), seed=it["seed"], rule_flags=rule, pred_flags=pred,
                             flags_used="predicted" if pred is not None else "oracle(rule labels)", text_flags=[use["g_col"], use["g_mat"]],
                             router=router.params(it) if router is not None else None)
        log(r["id"], json.dumps({k: info[r["id"]][k] for k in ("kind", "caption", "rule_flags", "pred_flags", "text_flags", "router")}))

    def gen(arm, it, i):
        tf, rp = i["text_flags"], i["router"]
        if arm == "A_fixed":
            return P.run(it, shaped=False)
        if arm == "B0_scheduled":                      # control: same schedule as B, centre values (separates router from schedule)
            return P.run(it, shaped=True)
        if arm == "B_router":
            return P.run(it, p=rp, shaped=True)
        if arm == "D_flaggate":
            return P.run(it, shaped=False, text_flags=tf)
        if arm == "D_hard":                            # control: D's code path with both flags forced to 1 (enforcement without the detector)
            return P.run(it, shaped=False, text_flags=(1.0, 1.0))
        if arm == "E_router_flaggate":
            return P.run(it, p=rp, shaped=True, text_flags=tf)
        raise ValueError(arm)

    import torch
    tpath = f"{out_dir}/timings.json"
    times = json.load(open(tpath)) if os.path.exists(tpath) else {}
    for r in rows:
        for arm in arms:
            path = f"{out_dir}/eval/{arm}/{r['id']}.png"
            if os.path.exists(path):
                continue
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.cuda.synchronize(); t0 = time.time()
            gen(arm, items[r["id"]], info[r["id"]]).save(path)
            torch.cuda.synchronize(); times[f"{arm}|{r['id']}"] = round(time.time() - t0, 2)
            json.dump(times, open(tpath, "w"), indent=1)
            log("%-18s %-22s %.1fs" % (arm, r["id"], times[f"{arm}|{r['id']}"]))
    config = dict(res=P.res, steps=P.steps, arms=list(arms), rows=[r["id"] for r in rows], info=info,
                  cfg_center=dict(cs=1.0, hold=0.7, ip=0.5, cfg=7.0), relax=0.3)
    json.dump(config, open(f"{out_dir}/config.json", "w"), indent=1, default=str)

    recs = []
    if kit is not None:
        for r in rows:
            it = items[r["id"]]
            base = np.asarray(Image.open(f"{out_dir}/eval/A_fixed/{r['id']}.png").convert("RGB"), np.int16)
            for arm in arms:
                im = Image.open(f"{out_dir}/eval/{arm}/{r['id']}.png").convert("RGB")
                sc = kit.score(im, r, it["sketch"], it["ref"], None)
                recs.append(dict(id=r["id"], kind=ER.kind_of(r), arm=arm, pix_diff_vs_A=float(np.abs(np.asarray(im, np.int16) - base).mean()),
                                 **{k: v for k, v in sc.items() if not isinstance(v, str)}))
        S = pd.DataFrame(recs)
        S.to_csv(f"{out_dir}/scores.csv", index=False)
        return items, info, S
    return items, info, None


def contact_sheet(items, rows, out_dir, arms=ARMS):
    import matplotlib
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(len(rows), 2 + len(arms), figsize=(2.8 * (2 + len(arms)), 3 * len(rows)), squeeze=False)
    for i, r in enumerate(rows):
        it = items[r["id"]]
        ax[i, 0].imshow(it["sketch"]); ax[i, 0].set_title(f"{ER.kind_of(r)}: {r['caption'][:40]}", fontsize=6)
        ax[i, 1].imshow(it["ref"]); ax[i, 1].set_title(f"ref: {r.get('ref_colors')}, {r.get('ref_material')}", fontsize=6)
        for j, arm in enumerate(arms):
            ax[i, 2 + j].imshow(Image.open(f"{out_dir}/eval/{arm}/{r['id']}.png"))
            if i == 0:
                ax[i, 2 + j].set_title(arm, fontsize=8)
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    plt.tight_layout(); plt.savefig(f"{out_dir}/contact_sheet.png", dpi=110)
    return f"{out_dir}/contact_sheet.png"
