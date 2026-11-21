# Method 1: ControlNet caption-leak test. Paste into the nb 08 Colab tab AFTER the before/after cell
# (needs P, test, items, net, GM, init, trained, plt, ROLE_DIR from that session). ~10 min on a T4.
# Reload src first so the new cn_text option exists:
#   import importlib, pipeline; importlib.reload(pipeline)   # then rebuild P only if run_role is missing cn_text
import torch, matplotlib.pyplot as plt

ARMS = [("trained, full caption in ControlNet", trained, "full"),
        ("trained, neutral caption in ControlNet", trained, "mix"),
        ("init (InstantStyle-like), neutral ControlNet", init, "mix")]
fig, ax = plt.subplots(len(test), 2 + len(ARMS), figsize=(2.8 * (2 + len(ARMS)), 2.9 * len(test)))
diff = []
for i, (r, it) in enumerate(zip(test, items)):
    gp = GM.predict_gates(net, it["x"]); fl = [gp["g_obj"], gp["g_col"], gp["g_mat"]]
    outs = [P.run_role(it, g, fl, cn_text=m)[0] for _, g, m in ARMS]
    for j, (im, t) in enumerate([(it["sketch"], "sketch"), (it["ref"], f"ref: {r['ref_colors']}")] + [(o, a[0]) for o, a in zip(outs, ARMS)]):
        ax[i, j].imshow(im); ax[i, j].set_title(t, fontsize=7)
    ax[i, 0].set_ylabel(r["caption"][:40], fontsize=7)
    import numpy as np
    diff.append(float(np.abs(np.asarray(outs[0], float) - np.asarray(outs[1], float)).mean()))
    print(r["id"], "flags", [round(v, 2) for v in fl], "mean |full - neutral| pixel diff:", round(diff[-1], 1), "/255")
for x in ax.ravel(): x.set_xticks([]); x.set_yticks([])
plt.tight_layout(); plt.savefig(f"{ROLE_DIR}/leak_test.png", dpi=110); plt.show()
# Reading it: if column 4 shows the reference colour where column 3 shows the caption colour, the ControlNet leak is confirmed.
