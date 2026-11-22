"""Baseline vs CAS on identical items / seeds, with attribute-specific metrics and intervention tests.
python governance/evaluate.py --data_dir <pilot dir> --weights runs/cas_v1/cas_weights.pt --out runs/cas_v1/eval
Resumable: images already on disk are skipped."""
import argparse, json, os, sys, hashlib, re
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from PIL import Image
from cas_data import COLORS, prep_sketch, prep_ref
import metrics as M

def seed_for(s): return int(hashlib.md5(str(s).encode()).hexdigest()[:8], 16)
def cap_colour(c):
    f = [x for x in COLORS if re.search(rf"\b{x}\b", c, re.I)]; return f[0] if f else None
def cap_material(c):
    f = [x for x in ["wooden", "metal", "plastic", "ceramic", "glass", "leather", "fabric", "stone"] if re.search(rf"\b{x}\b", c, re.I)]; return f[0] if f else None

ARMS = ["baseline", "baseline_ip1", "instantstyle_block", "cas"]

def build_jobs(rows, n_per_kind, split="test"):
    """Each job = (job_id, kind, sketch_row, ref_row, caption, base_row). Intervention jobs reuse a base row with ONE input changed."""
    base = [r for r in rows if r["split"] == split and r.get("conflict_type", "none") == "none"]
    jobs = []
    kinds = {"object_text_sketch": "sem", "color_text_ref": "color", "material_text_ref": "material", "appearance_ref_swap": "app"}
    for r in [r for r in rows if r["split"] == split and r.get("conflict_type", "none") == "none"][:n_per_kind]:
        jobs.append(dict(id=r["id"] + "|normal", kind="normal", item=r, sketch=r["sketch"], ref=r["reference"], caption=r["caption"], seed_key=r["id"]))
    for r in [r for r in rows if r["split"] == split and r.get("conflict_type", "none") != "none"]:
        k = kinds[r["conflict_type"]]
        if sum(j["kind"] == k for j in jobs) < n_per_kind:
            jobs.append(dict(id=r["id"], kind=k, item=r, sketch=r["sketch"], ref=r["reference"], caption=r["caption"], seed_key=r.get("original_id", r["id"])))
    # interventions, one factor changed vs the 'normal' job of the same base row (same seed)
    for i, r in enumerate(base[:n_per_kind]):
        o = base[(i + 5) % len(base)]
        if o["category"] == r["category"]: continue
        sk = r["id"]
        jobs.append(dict(id=f"{sk}|swap_ref", kind="swap_ref", item=r, sketch=r["sketch"], ref=o["reference"], caption=r["caption"], seed_key=sk, other=o))
        jobs.append(dict(id=f"{sk}|swap_sketch", kind="swap_sketch", item=r, sketch=o["sketch"], ref=r["reference"], caption=r["caption"], seed_key=sk, other=o))
        jobs.append(dict(id=f"{sk}|swap_caption", kind="swap_caption", item=r, sketch=r["sketch"], ref=r["reference"], caption=o["caption"], seed_key=sk, other=o))
    return jobs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True); ap.add_argument("--meta", default="metadata_with_conflicts.jsonl")
    ap.add_argument("--weights", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=12); ap.add_argument("--res", type=int, default=640)
    a = ap.parse_args()
    from cas_stack import load_pipe
    import infer
    rows = [json.loads(l) for l in open(os.path.join(a.data_dir, a.meta)) if l.strip()]
    jobs = build_jobs(rows, a.n); os.makedirs(a.out, exist_ok=True)
    P = lambda rel: os.path.join(a.data_dir, rel)
    pipe = load_pipe()
    def gen_arm(arm):
        for j in jobs:
            path = f"{a.out}/{arm}/{j['id'].replace('|','_')}.png"
            if os.path.exists(path): continue
            os.makedirs(os.path.dirname(path), exist_ok=True)
            scale = {"baseline": infer.BASELINE_IP_SCALE, "baseline_ip1": 1.0, "cas": infer.CAS_IP_SCALE}.get(arm)
            if arm == "instantstyle_block": scale = {"up": {"block_0": [0.0, 1.0, 0.0]}}   # training-free reference rule from InstantStyle (comparison only)
            infer.generate(pipe, P(j["sketch"]), P(j["ref"]), j["caption"], seed_for(j["seed_key"]), a.res, scale).save(path)
    for arm in ["baseline", "baseline_ip1", "instantstyle_block"]: gen_arm(arm)
    backup, step = infer.apply_weights(pipe.unet, a.weights); print("CAS weights step", step); gen_arm("cas")
    infer.restore_weights(pipe.unet, backup)
    del pipe
    judge = M.ClipJudge(); cats = sorted({r["category"] for r in rows})
    recs = []
    for j in jobs:
        sk = prep_sketch(P(j["sketch"]), a.res); rf = prep_ref(P(j["ref"]), a.res); it = j["item"]
        for arm in ARMS:
            out = Image.open(f"{a.out}/{arm}/{j['id'].replace('|','_')}.png").convert("RGB")
            rec = dict(job=j["id"], kind=j["kind"], arm=arm)
            rec.update(M.geometry_scores(out, sk, rf)); rec.update(M.colour_scores(out, sk, rf, cap_colour(j["caption"])))
            sketch_cat = (j["other"] if j["kind"] == "swap_sketch" else it)["category"]
            cap_cat = it.get("caption_category", it["category"]) if j["kind"] != "swap_caption" else j["other"]["category"]
            rec.update(M.ClipJudge.identity(judge, out, cats, cap_cat, sketch_cat if sketch_cat != cap_cat else None))
            ref_mat = str(it.get("ref_material", "")).split(",")[0].strip()
            cm = cap_material(j["caption"])
            if ref_mat and cm and cm != ref_mat: rec.update(M.ClipJudge.material(judge, out, sk, ref_mat, cm))
            recs.append(rec)
    df = pd.DataFrame(recs); df.to_csv(f"{a.out}/per_item_metrics.csv", index=False)
    summ = df.groupby(["kind", "arm"]).mean(numeric_only=True).round(3); summ.to_csv(f"{a.out}/summary_by_kind_arm.csv"); print(summ.to_string())
    # intervention tests (paired outputs, same seed): pixel/edge/colour response to ONE changed input
    iv = []
    for arm in ARMS:
        for key in sorted({j["seed_key"] for j in jobs if j["kind"] == "swap_ref"}):
            L = lambda n: Image.open(f"{a.out}/{arm}/{key}_{n}.png").convert("RGB")
            j = next(j for j in jobs if j["id"] == f"{key}|swap_ref"); r0 = j["item"]; o = j["other"]
            sk0 = prep_sketch(P(r0["sketch"]), a.res); sk1 = prep_sketch(P(o["sketch"]), a.res)
            n_, sr, ss, sc = L("normal"), L("swap_ref"), L("swap_sketch"), L("swap_caption")
            R0, R1 = prep_ref(P(r0["reference"]), a.res), prep_ref(P(o["reference"]), a.res)
            e = lambda x: M.edges(x)
            iou = lambda x, y: float((e(x) & e(y)).sum() / max(1, (e(x) | e(y)).sum()))
            lab = lambda x, m: M._lab_mean(x, m)
            m0 = M.sketch_region(sk0)
            iv.append(dict(arm=arm, key=key,
                ref_swap_colour_shift=float(np.linalg.norm(lab(n_, m0)[1:] - lab(sr, m0)[1:])),
                ref_swap_follows_new_ref=float(np.linalg.norm(lab(sr, m0)[1:] - M._lab_mean(R1, M.ref_object_mask(R1))[1:]) < np.linalg.norm(lab(n_, m0)[1:] - M._lab_mean(R1, M.ref_object_mask(R1))[1:])),
                ref_swap_geometry_kept=iou(n_, sr),
                sketch_swap_geo_follows_new=float(M.edge_recall(np.asarray(sk1.convert('L')) > 127, e(ss)) > M.edge_recall(np.asarray(sk1.convert('L')) > 127, e(n_))),
                sketch_swap_colour_kept=float(np.linalg.norm(lab(ss, M.sketch_region(sk1))[1:] - M._lab_mean(R0, M.ref_object_mask(R0))[1:])),
                caption_swap_identity_follows=float(judge.identity(sc, cats, o["category"])["id_top1"]),
                caption_swap_geometry_kept=M.edge_recall(np.asarray(sk0.convert('L')) > 127, e(sc))))
    ivd = pd.DataFrame(iv); ivd.to_csv(f"{a.out}/interventions_per_item.csv", index=False)
    s2 = ivd.groupby("arm").mean(numeric_only=True).round(3); s2.to_csv(f"{a.out}/interventions_summary.csv"); print(s2.T.to_string())
    # paired bootstrap CIs: CAS - baseline per metric
    ci = []
    for kind in sorted(df.kind.unique()):
        for met in [c for c in df.columns if c not in ("job", "kind", "arm")]:
            for ref_arm in ["baseline", "baseline_ip1", "instantstyle_block"]:
                x = df[(df.kind == kind) & (df.arm == "cas")].set_index("job")[met]; y = df[(df.kind == kind) & (df.arm == ref_arm)].set_index("job")[met]
                d, lo, hi = M.paired_bootstrap(x.values, y.loc[x.index].values); ci.append(dict(kind=kind, metric=met, vs=ref_arm, diff=d, lo=lo, hi=hi))
    pd.DataFrame(ci).round(3).to_csv(f"{a.out}/cas_minus_baselines_ci.csv", index=False)
    # qualitative grid
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    show = [j for j in jobs if j["kind"] in ("color", "app", "material", "sem")][:8]
    fig, ax = plt.subplots(len(show), 2 + len(ARMS), figsize=(3 * (2 + len(ARMS)), 3 * len(show)))
    for i, j in enumerate(show):
        ims = [prep_sketch(P(j["sketch"]), a.res), prep_ref(P(j["ref"]), a.res)] + [Image.open(f"{a.out}/{arm}/{j['id'].replace('|','_')}.png") for arm in ARMS]
        for k, im in enumerate(ims): ax[i, k].imshow(im); ax[i, k].axis("off")
        ax[i, 0].set_title(j["caption"][:40], fontsize=7)
    for k, t in enumerate(["sketch", "reference"] + ARMS): ax[0, k].text(0.5, 1.15, t, transform=ax[0, k].transAxes, ha="center")
    plt.tight_layout(); plt.savefig(f"{a.out}/qualitative.png", dpi=90)

if __name__ == "__main__": main()
