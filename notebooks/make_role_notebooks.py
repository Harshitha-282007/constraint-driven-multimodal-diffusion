"""Writes notebooks 08_role_gates_train.ipynb and 09_role_gates_eval.ipynb (run: python notebooks/make_role_notebooks.py)."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

SETUP = '''# ---- CONFIG + SETUP (same in notebooks 05-09) ----
DRIVE_ROOT = "/content/drive/MyDrive/sketchy_dataset_one_sketch"
ZIP_PATH = f"{DRIVE_ROOT}/pilot_v1.zip"
LOCAL_DATA = "/content/pilot_v1"
GOV_DIR = f"{DRIVE_ROOT}/governance"            # governance.pt (notebook 05) lives here
ROLE_DIR = f"{GOV_DIR}/role_gates_v2"            # role_gates.pt (notebook 08) lives here (v1 run kept in role_gates/)
# The code lives in the repo's src/ folder (cloned below). To use a copy on Drive instead, set REPO_DIR to it.
REPO_URL = "https://github.com/Harshitha-282007/IE643_final.git"
REPO_BRANCH = "governed-pipeline"
REPO_DIR = "/content/repo"

import os, sys, glob, json, zipfile
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from google.colab import drive
drive.mount("/content/drive")
if not os.path.isdir(REPO_DIR):
    !git clone -q -b {REPO_BRANCH} {REPO_URL} {REPO_DIR}
assert os.path.exists(f"{REPO_DIR}/src/train_role_gates.py"), "src/train_role_gates.py not found: push the new code or point REPO_DIR at it"
sys.path.insert(0, f"{REPO_DIR}/src")
os.makedirs(ROLE_DIR, exist_ok=True)
if not glob.glob(f"{LOCAL_DATA}/**/metadata.jsonl", recursive=True):
    os.makedirs(LOCAL_DATA, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(LOCAL_DATA)
DATA_DIR = os.path.dirname(glob.glob(f"{LOCAL_DATA}/**/metadata.jsonl", recursive=True)[0])
BASE_ROWS = [json.loads(l) for l in open(f"{DATA_DIR}/metadata.jsonl") if l.strip()]
GOV = f"{GOV_DIR}/governance.pt"
assert os.path.exists(GOV), "governance.pt missing: run notebook 05 first"
print("data:", DATA_DIR, "| base rows:", len(BASE_ROWS))'''

PIP = '''!pip uninstall -y torchao -q
!pip install -q -U diffusers transformers accelerate safetensors peft lpips opencv-python-headless'''

# ============================================================================ 08
NB08 = [
("md", """# 08 - Role-gated SDXL: training

Team One Sheep Army. Trains the role gates (`src/rolegate/`) inside frozen SDXL + scribble ControlNet + IP-Adapter, so that SDXL itself decides, per layer, per pixel and per attribute, how much of each input acts.

| Attribute | Governing input | Gates that can carry it |
|---|---|---|
| Shape, placement | Sketch | `sketch_contour`, `sketch_placement` (ControlNet residuals) |
| Object identity | Text | `text_identity` (token group of the object words) |
| Colour, material | Reference, also when the caption names another | `ref_colour`, `ref_texture` (two reference channels); `text_colour`, `text_material` should learn to close on a conflict |

Training data (all with a known correct output): normal triplets, colour conflicts, material conflicts, and hue pairs (object and reference hue-rotated together). Gates are conditioned on the governance network's predicted conflict flags (notebook 05). Details in `ROLE_GATED_PLAN.md`.

Run order: setup -> memory test (2-3 min) -> full training (about 1.5-2 h on a T4; checkpoints every 150 steps on Drive) -> curves -> before/after check."""),
("code", SETUP),
("code", PIP + '''
import torch, numpy as np, pandas as pd, matplotlib.pyplot as plt
from PIL import Image
from pipeline import Pipeline
import train_role_gates as TR
print(torch.cuda.get_device_name(0), "|", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")'''),
("md", "## Memory test\nOne step of every kind (warm-up, attribute loss, swap pair) on a few photos. Must print `MEMORY TEST OK` with peak memory under ~14 GB. The loaded pipeline is reused below."),
("code", '''RES = 512
P = Pipeline(res=RES)
P, _, _ = TR.train(DATA_DIR, GOV, f"{ROLE_DIR}/memtest", res=RES, memory_test=True, P=P)'''),
("md", "## Full training\n600 optimizer steps x 4 micro-batches. Checkpoints `role_gates_stepXXXX.pt` every 150 steps and `train_log.json` are written to Drive, so a disconnect loses at most 150 steps."),
("code", '''P, role, hist = TR.train(DATA_DIR, GOV, ROLE_DIR, res=RES, P=P)
print(open(f"{ROLE_DIR}/role_gates_config.json").read())'''),
("code", '''# ---- training curves ----
H = pd.DataFrame(json.load(open(f"{ROLE_DIR}/train_log.json")))
cols = [c for c in ["diff", "attr", "swap", "prior", "keep", "a_colour", "a_texture", "a_contour", "a_identity"] if c in H]
fig, ax = plt.subplots(2, (len(cols) + 1) // 2, figsize=(4 * ((len(cols) + 1) // 2), 6))
for a, c in zip(ax.ravel(), cols):
    s = H.groupby("step")[c].mean().dropna()
    a.plot(s.index, s.values, lw=0.6, alpha=0.4); a.plot(s.index, s.rolling(20, min_periods=1).mean().values, lw=1.5)
    for x in (150, 450): a.axvline(x, color="k", lw=0.5, ls="--")
    a.set_title(c)
plt.tight_layout(); plt.savefig(f"{ROLE_DIR}/train_curves.png", dpi=110); plt.show()
print(H.groupby("phase")[cols].mean().round(4))'''),
("md", "## Before / after on held-out colour conflicts\nSame item, gates at initialisation (equivalent to the fixed pipeline with InstantStyle) vs trained gates. The trained model should follow the reference colour although the caption names another one; the gate maps show where each input acts."),
("code", '''from governance import labels as L, model as GM
from rolegate.gates import RoleGates, gate_maps, CROSS_GATES, SKETCH_GATES
MAPS = CROSS_GATES + SKETCH_GATES
from pipeline import seed_for
P.res, P.steps = 640, 20
net = GM.load(GOV)
test = [r for r in L.expand_rows(BASE_ROWS, per_type=1, seed=1, kinds=("col",)) if r["split"] == "test" and r["id"] != r["original_id"]][:3]
op = lambda r, k: Image.open(f"{DATA_DIR}/{r[k]}")
P.prepare_object_dirs([r.get("ref_object") or "object" for r in test])
items = P.make_items([dict(key=r["id"], sketch=op(r, "sketch"), ref=op(r, "reference"), caption=r["caption"], obj=r["category"],
                           seed=seed_for(r["original_id"]), ref_object=r.get("ref_object")) for r in test])
init = RoleGates.for_unet(P.unet, P.controlnet_res_channels())
trained = RoleGates.load(f"{ROLE_DIR}/role_gates.pt", P.dev)
fig, ax = plt.subplots(len(test), 4 + len(MAPS), figsize=(2.6 * (4 + len(MAPS)), 2.8 * len(test)))
for i, (r, it) in enumerate(zip(test, items)):
    gp = GM.predict_gates(net, it["x"]); fl = [gp["g_obj"], gp["g_col"], gp["g_mat"]]
    a, _ = P.run_role(it, init, fl)
    b, rec = P.run_role(it, trained, fl, record_at=10)
    gm = gate_maps(rec)
    for j, (im, t) in enumerate([(it["sketch"], "sketch"), (it["ref"], f"ref: {r['ref_colors']}"), (a, "gates at init"), (b, "trained gates")]):
        ax[i, j].imshow(im); ax[i, j].set_title(t, fontsize=8)
    for j, g in enumerate(MAPS):
        ax[i, 4 + j].imshow(gm[g].numpy(), vmin=0, vmax=2, cmap="magma"); ax[i, 4 + j].set_title(g, fontsize=8)
    ax[i, 0].set_ylabel(r["caption"][:40], fontsize=7)
for x in ax.ravel(): x.set_xticks([]); x.set_yticks([])
plt.tight_layout(); plt.savefig(f"{ROLE_DIR}/before_after.png", dpi=110); plt.show()'''),
]

# ============================================================================ 09
NB09 = [
("md", """# 09 - Role-gated SDXL: evaluation against the instructor's baselines

Every arm runs on the **same items** (test photos x 5 kinds: unedited, object conflict, colour conflict, material conflict, reference swap), 640 px, 20 steps, the same seed per photo.

| Arm | What it is | Question it answers |
|---|---|---|
| `fixed` | SDXL + ControlNet + IP-Adapter, constant weights | instructor's baseline |
| `router` | learned global routing (notebook 04) | does learned global weighting help |
| `router_constraints` | router + sketch/appearance guidance | do constraints help |
| `role` | trained role gates inside SDXL | does learned local, per-attribute routing help |
| `role_noflags` | role gates, conflict flags set to 0 | is conflict detection needed |
| `role_textfrozen` | role gates, text gates fixed at 1 | is damping contradicted words needed |

Governing input per conflict type (fixed rules): object identity -> text (sketch keeps placement); colour and material -> reference, also when the caption names another. Colour and material are measured inside the CLIPSeg object mask against the reference.

Needs `governance.pt` (05), `models/router.pt` (04, in the repo) and `role_gates.pt` (08). Resumable."""),
("code", SETUP),
("code", PIP + '''
import torch, numpy as np, pandas as pd, matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm
from governance import labels as L, model as GM, metrics as MT
from governance.evalkit import EvalKit
from governance.features import attr_phrase
from pipeline import Pipeline, seed_for
from router import RouterEnsemble
from rolegate.gates import RoleGates, gate_maps, CROSS_GATES, ALL_GATES

RUN = "role_gates_v2"
OUT = f"{DRIVE_ROOT}/sdxl_runs/{RUN}"
RES, STEPS, SEED = 640, 20, 0
MAX_PHOTOS = 10                      # first pass on 10 test photos (50 items); set None for all 20 - the run is resumable
ROLE_CKPT = f"{ROLE_DIR}/role_gates.pt"
os.makedirs(OUT, exist_ok=True)

rows = [r for r in L.expand_rows(BASE_ROWS, per_type=1, seed=1, kinds=("obj", "col", "mat", "ref")) if r["split"] == "test"]
if MAX_PHOTOS:
    keep = sorted({r["original_id"] for r in rows})[:MAX_PHOTOS]
    rows = [r for r in rows if r["original_id"] in keep]
kind = lambda r: "base" if r["id"] == r["original_id"] else r["id"].split("_")[-1][:3]
print(len(rows), "items from", len({r["original_id"] for r in rows}), "photos:", pd.Series([kind(r) for r in rows]).value_counts().to_dict())'''),
("code", '''# ---- models and items ----
P = Pipeline(res=RES, steps=STEPS)
net = GM.load(GOV)
router = RouterEnsemble(f"{REPO_DIR}/models/router.pt")
role = RoleGates.load(ROLE_CKPT, P.dev)
op = lambda r, k: Image.open(f"{DATA_DIR}/{r[k]}")
P.prepare_object_dirs([r.get("ref_object") or "object" for r in rows])
ITEMS = {}
for i in tqdm(range(0, len(rows), 16), desc="encode"):
    chunk = rows[i:i + 16]
    specs = [dict(key=r["id"], sketch=op(r, "sketch"), ref=op(r, "reference"), caption=r["caption"], obj=r.get("caption_category", r["category"]),
                  seed=seed_for(r["original_id"]), ref_object=r.get("ref_object"), attr=attr_phrase(r)) for r in chunk]
    for r, it in zip(chunk, P.make_items(specs)):
        gp = GM.predict_gates(net, it["x"])
        it["flags"] = [gp["g_obj"], gp["g_col"], gp["g_mat"]]
        it["router"] = router.params(it)
        ITEMS[r["id"]] = it
print(pd.DataFrame([dict(kind=kind(r), obj=ITEMS[r["id"]]["flags"][0], col=ITEMS[r["id"]]["flags"][1], mat=ITEMS[r["id"]]["flags"][2])
                    for r in rows]).groupby("kind").mean().round(2).rename_axis("predicted conflict flags"))'''),
("code", '''# ---- generate every arm (resumable) ----
ARMS = {
    "fixed":              lambda it: P.run(it, shaped=False),
    "router":             lambda it: P.run(it, p=it["router"], shaped=True),
    "router_constraints": lambda it: P.run(it, p=it["router"], shaped=True, guided=True),
    "role":               lambda it: P.run_role(it, role, it["flags"])[0],
    "role_noflags":       lambda it: P.run_role(it, role, [0.0, 0.0, 0.0])[0],
    "role_textfrozen":    lambda it: P.run_role(it, role, it["flags"], freeze_text=True)[0],
}
manifest = []
for r in tqdm(rows, desc="generate"):
    it = ITEMS[r["id"]]
    for arm, fn in ARMS.items():
        path = f"{OUT}/eval/{arm}/{r['id']}.png"
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fn(it).save(path)
        manifest.append(dict(id=r["id"], base=r["original_id"], kind=kind(r), arm=arm, generated=path, caption=r["caption"],
                             flag_obj=it["flags"][0], flag_col=it["flags"][1], flag_mat=it["flags"][2], **r["gates"]))
    torch.cuda.empty_cache()
pd.DataFrame(manifest).to_csv(f"{OUT}/manifest.csv", index=False)
print(len(manifest), "images | peak GPU GB:", round(torch.cuda.max_memory_allocated() / 1e9, 2))'''),
("code", '''# ---- score (resumable) ----
EK = EvalKit(rule="reference")
HP = f"{OUT}/scores.csv"
done = pd.read_csv(HP) if os.path.exists(HP) else pd.DataFrame()
have = set(zip(done.id, done.arm)) if len(done) else set()
recs = done.to_dict("records") if len(done) else []
by_id = {r["id"]: r for r in rows}
for m in tqdm(manifest, desc="score"):
    if (m["id"], m["arm"]) in have: continue
    r, it = by_id[m["id"]], ITEMS[m["id"]]
    tgt = op(r, "target") if m["kind"] == "base" else None
    recs.append(dict(id=m["id"], base=m["base"], kind=m["kind"], arm=m["arm"],
                     **EK.score(Image.open(m["generated"]).convert("RGB"), r, it["sketch"], it["ref"], tgt)))
S = pd.DataFrame(recs); S.to_csv(HP, index=False)
print("scored", len(S))'''),
("md", """## Governed-attribute metric per conflict type

| Kind | Governing input | Hit metric | Continuous metric (used for bootstrap and figure) | Must not break |
|---|---|---|---|---|
| obj | text (identity), sketch (placement) | `obj_hit` | `sem_margin` | `edge_iou` |
| col | reference | `col_hit` | `col_share` (lower `col_leak` = caption colour absent) | `edge_iou` |
| mat | reference | `mat_hit` | `mat_hit` | `edge_iou` |
| ref | reference (new reference) | `col_hit` | `col_share` | `edge_iou` |
| base | all agree | - | `dino_tgt` | `lpips_tgt` |"""),
("code", '''GOV_HIT = {"obj": "obj_hit", "col": "col_hit", "mat": "mat_hit", "ref": "col_hit"}
GOV_CONT = {"obj": "sem_margin", "col": "col_share", "mat": "mat_hit", "ref": "col_share", "base": "dino_tgt"}
S["gov_hit"] = [s[GOV_HIT[s["kind"]]] if s["kind"] in GOV_HIT else np.nan for _, s in S.iterrows()]
S["gov"] = [s[GOV_CONT[s["kind"]]] for _, s in S.iterrows()]
order = list(ARMS)
print("== governed-attribute hit rate (did the governing input win?) ==")
print(S.pivot_table(index="kind", columns="arm", values="gov_hit").reindex(columns=order).round(2))
print("\\n== governed-attribute continuous score ==")
print(S.pivot_table(index="kind", columns="arm", values="gov").reindex(columns=order).round(3))
for m, note in [("col_leak", "caption colour present (lower is better)"), ("id_leak", "image reads as the reference's object (lower is better)"),
                ("edge_iou", "sketch consistency"), ("layout_sketch_iou", "layout consistency"), ("clip_t", "text-image alignment")]:
    if m in S:
        print(f"\\n== {m}: {note} =="); print(S.pivot_table(index="kind", columns="arm", values=m).reindex(columns=order).round(3))
FAM = {"text-image alignment": ["clip_t", "sem_margin", "obj_hit"], "sketch consistency": ["edge_recall", "edge_precision", "edge_iou"],
       "reference similarity": ["dino_ref", "col_share", "col_hit", "col_leak", "mat_hit", "id_leak"],
       "layout consistency": ["layout_sketch_iou", "layout_tgt_iou"], "perceptual quality (unedited rows)": ["lpips_tgt", "dino_tgt"]}
T = S.groupby("arm")[[c for cs in FAM.values() for c in cs if c in S]].mean().reindex(order)
print("\\n== five metric families of the brief (mean over all items) =="); print(T.round(3))
T.round(4).to_csv(f"{OUT}/results_table.csv"); S.to_csv(f"{OUT}/results_per_item.csv", index=False)'''),
("code", '''# ---- paired comparisons over base photos + win/loss counts per kind ----
PAIRS = [("router", "fixed"), ("router_constraints", "router"), ("role", "fixed"), ("role", "router"), ("role", "router_constraints"),
         ("role", "role_noflags"), ("role", "role_textfrozen")]
out = []
for a, b in PAIRS:
    A, B = S[S.arm == a].set_index("id"), S[S.arm == b].set_index("id")
    for k in ["all"] + sorted(S.kind.unique()):
        ks = A.index.intersection(B.index)
        if k != "all": ks = [i for i in ks if A.loc[i, "kind"] == k]
        if k == "all": ks = [i for i in ks if A.loc[i, "kind"] != "base"]
        if len(ks) < 3: continue
        d = (A.loc[ks, "gov"] - B.loc[ks, "gov"]).values
        bs = MT.paired_bootstrap(d, A.loc[ks, "base"].values, n=5000)
        out.append(dict(comparison=f"{a} - {b}", kind=k, wins=int((d > 1e-6).sum()), losses=int((d < -1e-6).sum()),
                        **{x: round(v, 4) if isinstance(v, float) else v for x, v in bs.items()},
                        significant="yes" if bs["lo"] > 0 or bs["hi"] < 0 else "no"))
PT = pd.DataFrame(out); PT.to_csv(f"{OUT}/paired_comparisons.csv", index=False)
print(PT.to_string(index=False))
print("\\nMetric = governed-attribute continuous score of each item's kind. 'all' pools the four conflict kinds.")'''),
("code", '''# ---- where does the next improvement belong? (instructor's question) ----
def get(a, b, k="all"):
    q = PT[(PT.comparison == f"{a} - {b}") & (PT.kind == k)]
    return q.iloc[0] if len(q) else None
lines = []
for a, b, part in [("router", "fixed", "global guidance weighting"), ("role", "router", "local, per-attribute weighting (the role gates)"),
                   ("role", "role_noflags", "conflict detection"), ("router_constraints", "router", "constraints"),
                   ("role", "role_textfrozen", "damping the contradicted caption words")]:
    q = get(a, b)
    if q is None: continue
    verdict = "helps" if q.lo > 0 else ("hurts" if q.hi < 0 else "no significant effect")
    lines.append(f"{part:48s} {a} - {b}: {q['mean']:+.3f} [{q.lo:+.3f}, {q.hi:+.3f}] -> {verdict}")
print("\\n".join(lines))
print("\\nReading: a part whose ablation costs the most is where the method's gain comes from; a part with no effect, "
      "or the kind with the lowest hit rate in the table above, is where the next improvement belongs.")
open(f"{OUT}/bottleneck.txt", "w").write("\\n".join(lines))'''),
("md", "## Success and failure, side by side\nRule fixed before looking: **success** = the conflict item whose `role` governed score is at the median; **failure** = the item where `role` loses most against `fixed`. Each shown with every arm and the role gates' maps (step 10 of 20, conditional branch)."),
("code", '''R, F0 = S[(S.arm == "role") & (S.kind != "base")].set_index("id"), S[(S.arm == "fixed") & (S.kind != "base")].set_index("id")
ks = R.index.intersection(F0.index)
z = R.loc[ks].groupby("kind")["gov"].transform(lambda v: (v - v.mean()) / (v.std() + 1e-6))     # comparable across kinds
succ = (z - z.median()).abs().idxmin()
fail = (R.loc[ks, "gov"] - F0.loc[ks, "gov"]).groupby(R.loc[ks, "kind"]).transform(lambda v: v / (v.abs().max() + 1e-6)).idxmin()
pick = [("success (median)", succ), ("failure (largest loss vs fixed)", fail)]
fig, ax = plt.subplots(2, 2 + len(ARMS) + len(CROSS_GATES), figsize=(2.4 * (2 + len(ARMS) + len(CROSS_GATES)), 5.6))
for i, (title, rid) in enumerate(pick):
    r, it = by_id[rid], ITEMS[rid]
    _, rec = P.run_role(it, role, it["flags"], record_at=10)
    gm = gate_maps(rec)
    ax[i, 0].imshow(it["sketch"]); ax[i, 0].set_title(f"{title}\\n{kind(r)}: {r['caption'][:34]}", fontsize=7)
    ax[i, 1].imshow(it["ref"]); ax[i, 1].set_title(f"ref: {r.get('ref_colors')}, {r.get('ref_material')}", fontsize=7)
    for j, arm in enumerate(ARMS):
        sc = S[(S.id == rid) & (S.arm == arm)]["gov"].iloc[0]
        ax[i, 2 + j].imshow(Image.open(f"{OUT}/eval/{arm}/{rid}.png")); ax[i, 2 + j].set_title(f"{arm}\\n{GOV_CONT[kind(r)]}={sc:.2f}", fontsize=7)
    for j, g in enumerate(CROSS_GATES):
        ax[i, 2 + len(ARMS) + j].imshow(gm[g].numpy(), vmin=0, vmax=2, cmap="magma"); ax[i, 2 + len(ARMS) + j].set_title(g, fontsize=7)
for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.savefig(f"{OUT}/success_failure.png", dpi=120); plt.show()
print("flags (obj, col, mat):", {t: [round(v, 2) for v in ITEMS[i]["flags"]] for t, i in pick})'''),
("md", "## Gate knockout matrix\nSwitch one gate group off at a time on 8 photos (4 colour conflicts, 4 object conflicts) and measure which attributes move. A clean role separation is a diagonal pattern: reference gates move colour/material, the sketch gates move edges/layout, the identity gate moves the object score."),
("code", '''KO_ITEMS = [r["id"] for r in rows if kind(r) == "col"][:4] + [r["id"] for r in rows if kind(r) == "obj"][:4]
MEAS = ["col_share", "mat_hit", "edge_iou", "layout_sketch_iou", "sem_margin", "dino_ref"]
KP = f"{OUT}/knockout.csv"
ko = pd.read_csv(KP).to_dict("records") if os.path.exists(KP) else []
have = {(k["id"], k["gate"]) for k in ko}
for rid in tqdm(KO_ITEMS, desc="knockout"):
    r, it = by_id[rid], ITEMS[rid]
    for gname in ["none"] + ALL_GATES:
        if (rid, gname) in have: continue
        path = f"{OUT}/knockout/{gname}/{rid}.png"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            P.run_role(it, role, it["flags"], knock=() if gname == "none" else (gname,))[0].save(path)
        sc = EK.score(Image.open(path).convert("RGB"), r, it["sketch"], it["ref"])
        ko.append(dict(id=rid, kind=kind(r), gate=gname, **{m: sc.get(m, np.nan) for m in MEAS}))
    pd.DataFrame(ko).to_csv(KP, index=False)
K = pd.DataFrame(ko)
base = K[K.gate == "none"].set_index("id")[MEAS]
D = K[K.gate != "none"].copy()
for m in MEAS: D[m] = D[m].values - base.loc[D.id, m].values
M = D.groupby("gate")[MEAS].apply(lambda x: x.abs().mean()).reindex(ALL_GATES)
print("mean |change| when the gate is knocked out (rows: gate, columns: attribute metric)"); print(M.round(3))
plt.figure(figsize=(7, 4)); plt.imshow((M / (M.max(0) + 1e-9)).values, cmap="Blues", aspect="auto")
plt.xticks(range(len(MEAS)), MEAS, rotation=30, fontsize=8); plt.yticks(range(len(ALL_GATES)), ALL_GATES, fontsize=8)
plt.title("knockout effect (column-normalised)", fontsize=9); plt.colorbar(); plt.tight_layout()
plt.savefig(f"{OUT}/knockout_matrix.png", dpi=120); plt.show()
M.round(4).to_csv(f"{OUT}/knockout_matrix.csv")'''),
]


def nb(cells):
    out = []
    for kind, src in cells:
        lines = src.split("\n")
        source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
        c = {"cell_type": "markdown" if kind == "md" else "code", "metadata": {}, "source": source}
        if kind == "code":
            c.update(execution_count=None, outputs=[])
        out.append(c)
    return {"cells": out, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}


if __name__ == "__main__":
    for name, cells in [("08_role_gates_train.ipynb", NB08), ("09_role_gates_eval.ipynb", NB09)]:
        json.dump(nb(cells), open(os.path.join(HERE, name), "w", encoding="utf-8"), indent=1)
        print("wrote", name)
