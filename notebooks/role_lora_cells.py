# Colab cells for the multi-role LoRA (paste each "# %%" block into its own cell of a fresh notebook tab, in order).
# Needs the code pushed to the repo branch. Outputs go to the Colab disk first (a mounted Drive folder vanished mid-run once)
# and are copied to Drive at the end.

# %% 0. setup: Drive, repo, data, imports, pipeline (res 512 = training resolution)
import os, sys, glob, json, zipfile, shutil, importlib
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from google.colab import drive
drive.mount("/content/drive")
DRIVE_ROOT = "/content/drive/MyDrive/sketchy_dataset_one_sketch"; GOV_DIR = f"{DRIVE_ROOT}/governance"; GOV = f"{GOV_DIR}/governance.pt"
REPO_DIR, REPO_URL, BRANCH = "/content/repo", "https://github.com/Harshitha-282007/IE643_final.git", "governed-pipeline"
if not os.path.isdir(REPO_DIR):
    !git clone -q -b {BRANCH} {REPO_URL} {REPO_DIR}
else:
    !git -C {REPO_DIR} checkout -- .
    !git -C {REPO_DIR} pull -q
sys.path.insert(0, f"{REPO_DIR}/src")
LOCAL = "/content/pilot_v1"
if not glob.glob(f"{LOCAL}/**/metadata.jsonl", recursive=True):
    os.makedirs(LOCAL, exist_ok=True); zipfile.ZipFile(f"{DRIVE_ROOT}/pilot_v1.zip").extractall(LOCAL)
DATA_DIR = os.path.dirname(glob.glob(f"{LOCAL}/**/metadata.jsonl", recursive=True)[0])
BASE_ROWS = [json.loads(l) for l in open(f"{DATA_DIR}/metadata.jsonl") if l.strip()]
!pip uninstall -y torchao -q
!pip install -q -U diffusers transformers accelerate safetensors peft lpips opencv-python-headless
import numpy as np, pandas as pd, matplotlib.pyplot as plt, torch
from PIL import Image
from pipeline import Pipeline, seed_for
import train_role_lora as TL
P = Pipeline(res=512, steps=20)
print("ready | rows:", len(BASE_ROWS))

# %% 1. memory test (2 optimizer steps). Must print MEMORY TEST OK, peak under ~14 GB
LORA_DIR = "/content/role_lora_v1"; os.makedirs(LORA_DIR, exist_ok=True)
P, _lora, _ = TL.train(DATA_DIR, GOV, "/content/role_lora_memtest", res=512, memory_test=True, P=P)
_lora.detach()                                   # the test LoRA must not stay attached to the UNet

# %% 2. train the multi-role LoRA (text + reference interfaces), 400 optimizer steps x 4 micro-batches, ~50 min
P, lora, hist = TL.train(DATA_DIR, GOV, LORA_DIR, res=512, steps=400, groups=("text", "ref"), P=P)

# %% 3. evaluation: baseline vs rule-based vs LoRA on the same held-out rows, one table per conflict kind
import eval_roles as ER
from governance.evalkit import EvalKit
from rolegate.gates import RoleGates
rows = ER.eval_rows(BASE_ROWS, split="test", n_photos=12)
print(len(rows), "rows:", pd.Series([ER.kind_of(r) for r in rows]).value_counts().to_dict())
op = lambda r, k: Image.open(f"{DATA_DIR}/{r[k]}")
P.prepare_object_dirs([r.get("ref_object") or "object" for r in rows])
items = {}
for i in range(0, len(rows), 8):
    part = rows[i:i + 8]
    for r, it in zip(part, P.make_items([dict(key=r["id"], sketch=op(r, "sketch"), ref=op(r, "reference"), caption=r["caption"], obj=r["category"],
                                              seed=seed_for(r["original_id"]), ref_object=r.get("ref_object")) for r in part])):
        it.pop("S_gh", None); items[r["id"]] = it
gates = RoleGates.for_unet(P.unet, P.controlnet_res_channels())
IP_RULE = 1.0                                     # reference strength of the rule-based arm (boost test: x2 moved colour toward the reference)
def a_fixed(it, r): lora.set_scale(0); return P.run(it, shaped=False)                                    # the baseline: constant strengths
def a_rule(it, r): lora.set_scale(0); return P.run_role(it, gates, [0, 1, 1], knock=("text_colour", "text_material"), cn_text="mix", p=dict(ip=IP_RULE))[0]   # hand-fed rule
def a_lora(it, r): lora.set_scale(1); return P.run_role(it, gates, [0, 0, 0])[0]                         # learned: nothing neutralised, no detector
ARMS = {"fixed": a_fixed, "rule": a_rule, "lora": a_lora}
EVAL_DIR = "/content/role_eval"; os.makedirs(EVAL_DIR, exist_ok=True)
ER.generate(rows, items, ARMS, EVAL_DIR)

# %% 4. score + tables
kit = EvalKit(rule="reference")
scores = ER.score(kit, rows, ARMS, EVAL_DIR, DATA_DIR)
comp, shape = ER.tables(scores, list(ARMS))
print("ROLE COMPLIANCE (higher is better; in brackets lower is better)"); print(comp.to_string(index=False))
print("\nSKETCH / FIDELITY"); print(shape.to_string(index=False))
comp.to_csv(f"{EVAL_DIR}/compliance.csv", index=False); shape.to_csv(f"{EVAL_DIR}/shape.csv", index=False)

# %% 5. success / failure figure: same row, all arms side by side (rows where the LoRA beats fixed on the primary metric, then where it loses)
df = pd.DataFrame(scores); prim = {"obj": "obj_hit", "col": "col_hit", "mat": "mat_hit", "ref": "col_hit"}
rowsd = {r["id"]: r for r in rows}
gain = []
for rid, g in df[df.kind.isin(prim)].groupby("id"):
    k = g.kind.iloc[0]; a = g[g.arm == "lora"][prim[k]].mean(); b = g[g.arm == "fixed"][prim[k]].mean()
    gain.append((rid, a - b))
gain.sort(key=lambda x: -x[1]); pick = [gain[0][0], gain[-1][0]] if gain else []
fig, ax = plt.subplots(len(pick), 2 + len(ARMS), figsize=(2.8 * (2 + len(ARMS)), 3 * max(len(pick), 1)), squeeze=False)
for i, rid in enumerate(pick):
    r = rowsd[rid]; ax[i, 0].imshow(op(r, "sketch")); ax[i, 0].set_title("sketch", fontsize=7); ax[i, 1].imshow(op(r, "reference")); ax[i, 1].set_title("reference: " + str(r.get("ref_colors")), fontsize=7)
    for j, a in enumerate(ARMS):
        ax[i, 2 + j].imshow(Image.open(ER.image_path(EVAL_DIR, a, r))); ax[i, 2 + j].set_title(a, fontsize=7)
    ax[i, 0].set_ylabel(("SUCCESS: " if i == 0 else "FAILURE: ") + r["caption"][:34], fontsize=7)
for x in ax.ravel(): x.set_xticks([]); x.set_yticks([])
plt.tight_layout(); plt.savefig(f"{EVAL_DIR}/success_failure.png", dpi=110); plt.show()

# %% 6. copy everything to Drive, into ONE new folder (create it once)
DEST = f"{GOV_DIR}/role_lora_run1"
os.mkdir(DEST)
for f in os.listdir(LORA_DIR): shutil.copy(f"{LORA_DIR}/{f}", DEST) if os.path.isfile(f"{LORA_DIR}/{f}") else None
for f in ("compliance.csv", "shape.csv", "scores.json", "success_failure.png"): shutil.copy(f"{EVAL_DIR}/{f}", DEST)
print(sorted(os.listdir(DEST)))

# %% 7. ablation: reference-only LoRA (does restricting to one modality hurt?)  ~50 min. Run after 6.
lora.detach(); LORA_REF = "/content/role_lora_refonly"; os.makedirs(LORA_REF, exist_ok=True)
P, lora_ref, _ = TL.train(DATA_DIR, GOV, LORA_REF, res=512, steps=400, groups=("ref",), P=P)
def a_lora_ref(it, r): lora_ref.set_scale(1); return P.run_role(it, gates, [0, 0, 0])[0]
def a_fixed2(it, r): lora_ref.set_scale(0); return P.run(it, shaped=False)
ARMS2 = {"lora_ref_only": a_lora_ref}
ER.generate(rows, items, ARMS2, EVAL_DIR)
ALL = {**{k: None for k in ARMS}, **ARMS2}
scores = ER.score(kit, rows, ALL, EVAL_DIR, DATA_DIR)
comp, shape = ER.tables(scores, list(ALL))
print(comp.to_string(index=False)); print(shape.to_string(index=False))
