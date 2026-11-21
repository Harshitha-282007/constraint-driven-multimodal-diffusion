# Colab cells for Method 2 (ColourAdapter). Paste each "# %%" block into its own cell of the 08 notebook tab, after its
# setup cells (config + imports + the ad-hoc cell that builds P). Needs the code pushed to the repo branch.

# %% 1. update code (discards the runtime patch of pipeline.py, pulls the committed version, reloads)
import importlib, sys
!git -C {REPO_DIR} checkout -- src/pipeline.py
!git -C {REPO_DIR} pull -q
import pipeline; importlib.reload(pipeline); P.__class__ = pipeline.Pipeline
import coloradapter.palette, coloradapter.adapter
for m in (coloradapter.palette, coloradapter.adapter): importlib.reload(m)
import train_colour_adapter as TC; importlib.reload(TC)
COL_DIR = f"{GOV_DIR}/colour_adapter_v1"; os.makedirs(COL_DIR, exist_ok=True)
print("code updated:", hasattr(P, "controlnet_text"), "| out:", COL_DIR)

# %% 2. memory test (2 optimizer steps, ~3 min). Must print MEMORY TEST OK, peak under ~14 GB
P, _, _ = TC.train(DATA_DIR, GOV, f"{COL_DIR}/memtest", res=512, memory_test=True, P=P)

# %% 3. full training: 300 optimizer steps x 4 micro-batches, checkpoint every 100 steps to Drive
P, ca, hist = TC.train(DATA_DIR, GOV, COL_DIR, res=512, steps=300, P=P)

# %% 3b. if the runtime died: resume from the newest checkpoint
# import glob; last = sorted(glob.glob(f"{COL_DIR}/colour_adapter_step*.pt"))[-1]
# P, ca, hist = TC.train(DATA_DIR, GOV, COL_DIR, res=512, steps=300, resume=last, P=P)

# %% 4. before / after on held-out colour conflicts (same three rows as the Method 1 report, plus the caption conflict check)
import numpy as np, matplotlib.pyplot as plt, json
from PIL import Image
from governance import labels as L, model as GM
from rolegate.gates import RoleGates
from coloradapter.adapter import ColourAdapter
from coloradapter.palette import palette, palette_distance
from pipeline import seed_for
P.res, P.steps = 640, 20
net = GM.load(GOV)
ca = ColourAdapter.load(f"{COL_DIR}/colour_adapter.pt", P.dev)
test = [r for r in L.expand_rows(BASE_ROWS, per_type=1, seed=1, kinds=("col",)) if r["split"] == "test" and r["id"] != r["original_id"]][:3]
op = lambda r, k: Image.open(f"{DATA_DIR}/{r[k]}")
P.prepare_object_dirs([r.get("ref_object") or "object" for r in test])
items = P.make_items([dict(key=r["id"], sketch=op(r, "sketch"), ref=op(r, "reference"), caption=r["caption"], obj=r["category"],
                           seed=seed_for(r["original_id"]), ref_object=r.get("ref_object")) for r in test])
gates = RoleGates.for_unet(P.unet, P.controlnet_res_channels())
KN = ("text_colour", "ref_colour")
fig, ax = plt.subplots(len(test), 5, figsize=(14, 2.9 * len(test)))
rows = []
for i, (r, it) in enumerate(zip(test, items)):
    gp = GM.predict_gates(net, it["x"]); fl = [gp["g_obj"], 1.0, 0.0]               # rule: the reference wins colour, always
    pal = palette(np.array(it["ref"].convert("RGB")))
    base, _ = P.run_role(it, gates, fl, knock=("text_colour",), cn_text="mix")               # caption colour off, no adapter
    ours, _ = P.run_role(it, gates, fl, knock=KN, cn_text="mix", adapter=ca, palette=pal)    # + colour adapter
    full, _ = P.run_role(it, gates, [gp["g_obj"], 0.0, 0.0])                                 # nothing changed: stale caption colour
    for j, (im, t) in enumerate([(it["sketch"], "sketch"), (it["ref"], "ref: " + str(r["ref_colors"])), (full, "stale caption (old pipeline)"),
                                 (base, "caption colour off"), (ours, "colour adapter")]):
        ax[i, j].imshow(im); ax[i, j].set_title(t, fontsize=7)
    ax[i, 0].set_ylabel(r["caption"][:40], fontsize=7)
    def centre(im):                                                    # object is roughly central; background of generated images is not flat
        a = np.array(im.convert("RGB")); h, w = a.shape[:2]; m = np.zeros((h, w), bool); m[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = True
        return palette(a, mask=m)
    d = lambda im: palette_distance(pal, centre(im))
    rows.append(dict(id=r["id"], caption=r["caption"], ref=r["ref_colors"], lab_dist_to_ref=dict(stale=round(d(full), 1), off=round(d(base), 1), adapter=round(d(ours), 1))))
    print(rows[-1])
for x in ax.ravel(): x.set_xticks([]); x.set_yticks([])
plt.tight_layout(); plt.savefig(f"{COL_DIR}/before_after.png", dpi=110); plt.show()
json.dump(rows, open(f"{COL_DIR}/before_after.json", "w"), indent=1)
