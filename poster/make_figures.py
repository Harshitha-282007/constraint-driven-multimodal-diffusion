import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image
import pandas as pd, io

"""Regenerates the poster figures (Intro, Workflow, Architecture, Qualitative, Results_pilot) into poster/figures.
Pilot outputs are read from poster/pilot_images (crops of results/pilot_flaggate_v1/contact_sheet.png).
Run:  python poster/make_figures.py"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
IMG = os.path.join(HERE, "pilot_images")
OUT = os.path.join(HERE, "figures")
for f in ["pala.ttf", "palab.ttf", "palai.ttf", "palabi.ttf"]:       # Palatino if present (Windows); DejaVu Sans otherwise
    try:
        fm.fontManager.addfont(os.path.join(r"C:\Windows\Fonts", f))
    except Exception:
        pass
plt.rcParams["font.family"] = ["Palatino Linotype", "DejaVu Sans"]

NAVY, GREEN, BLUE = "#000080", "#2E8B57", "#0000FF"
TXT, SKE, REF = "#1d4ed8", "#15803d", "#c2410c"      # modality colours
LT = {"t": "#dbe7ff", "s": "#d4f1de", "r": "#fde2d0", "n": "#e8e8f5", "g": "#f3f3f3"}


def box(ax, x, y, w, h, text="", fc="#fff", ec=NAVY, lw=2.5, fs=15, tc="k", bold=False, r=0.02, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(ax, p, q, c=NAVY, lw=3, style="-|>", ms=22, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms, lw=lw, color=c, ls=ls, shrinkA=0, shrinkB=0))


def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    return fig, ax


def im(ax, path, x, y, w, h=None):
    a = np.asarray(Image.open(path).convert("RGB"))
    h = h or w * a.shape[0] / a.shape[1]
    ax.imshow(a, extent=(x, x + w, y, y + h), zorder=3, aspect="auto")
    ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, ec="#333", lw=1.5, zorder=4))


# ---------------------------------------------------------------- Intro: who governs what
fig, ax = canvas(10, 6.2)
ax.text(5, 5.85, "Who governs what?", ha="center", fontsize=24, fontweight="bold", color=NAVY)
# inputs
box(ax, 0.15, 3.5, 2.9, 1.9, "", fc=LT["s"], ec=SKE)
ax.text(1.6, 5.2, "SKETCH", ha="center", fontsize=16, fontweight="bold", color=SKE)
im(ax, IMG + "/q_1_sketch.jpg", 0.9, 3.6, 1.4)
box(ax, 3.55, 3.5, 2.9, 1.9, "", fc=LT["r"], ec=REF)
ax.text(5.0, 5.2, "REFERENCE", ha="center", fontsize=16, fontweight="bold", color=REF)
im(ax, IMG + "/q_1_ref.jpg", 4.3, 3.6, 1.4)
box(ax, 6.95, 3.5, 2.9, 1.9, "", fc=LT["t"], ec=TXT)
ax.text(8.4, 5.2, "TEXT", ha="center", fontsize=16, fontweight="bold", color=TXT)
ax.text(8.4, 4.3, "\u201cA person holds an axe\nabove a pile of logs,\nblack.\u201d", ha="center", va="center", fontsize=13, style="italic", color="k")
# attributes
for x, c, lt, t1, t2 in [(0.15, SKE, LT["s"], "SHAPE", "placement"), (3.55, REF, LT["r"], "COLOUR", "+ MATERIAL"), (6.95, TXT, LT["t"], "IDENTITY", "what the object is")]:
    arrow(ax, (x + 1.45, 3.45), (x + 1.45, 2.65), c=c)
    box(ax, x, 1.35, 2.9, 1.25, "", fc=lt, ec=c)
    ax.text(x + 1.45, 2.1, t1, ha="center", fontsize=19, fontweight="bold", color=c)
    ax.text(x + 1.45, 1.65, t2, ha="center", fontsize=14, color="k")
box(ax, 0.15, 0.1, 9.7, 0.95, "", fc="#fff8d6", ec="#b8860b", lw=2.5)
ax.text(5.0, 0.58, "The caption says \u201cblack\u201d, the reference is brown leather: the reference wins colour.", ha="center", va="center", fontsize=15, fontweight="bold")
fig.savefig(OUT + "/Intro.png", dpi=200); plt.close(fig)

# ---------------------------------------------------------------- Workflow
fig, ax = canvas(10, 8.6)
rows = [
    ("1  Data preparation", [("Sketchy\n125 classes", "n"), ("200 photos\ncaptioned\n(Qwen3-VL-2B)", "n"), ("Neutral-object\nreferences v2\n(colour, material)", "n"), ("pilot_v1\n160 / 20 / 20\n+ 800 conflict rows", "n")]),
    ("2  Baselines", [("SDXL + ControlNet\n+ IP-Adapter\n(nb 02)", "g"), ("Gated fusion\nMLP (nb 03)", "g"), ("Attribute router\n6 cosines \u2192 4\nstrengths (nb 04)", "t"), ("Governance net\n3 conflict flags\n(nb 05)", "t")]),
    ("3  Novelty attempts", [("Fusion LoRA\n(nb 06, 07)", "g"), ("Role gates in\nSDXL (nb 08, 09)", "g"), ("Role LoRA /\ncolour adapter\n(nb 10)", "g"), ("Flag-gated text\n(nb 11)", "t")]),
    ("4  Submitted method", [("Router +\nflag gate", "r"), ("= Attribute\nRouting Layer\n(ARL)", "r"), ("Evaluation:\nobject, colour,\nmaterial, edge IoU", "n"), ("Demo and\nrepository", "n")]),
]
y0 = 6.9
for k, (title, items) in enumerate(rows):
    y = y0 - k * 2.05
    ax.text(0.05, y + 1.5, title, fontsize=17, fontweight="bold", color=NAVY)
    for j, (t, c) in enumerate(items):
        x = 0.1 + j * 2.45
        ec = {"n": NAVY, "g": "#777", "t": TXT, "r": "#b45309"}[c]
        fc = {"n": LT["n"], "g": LT["g"], "t": LT["t"], "r": "#fde7c8"}[c]
        box(ax, x, y, 2.05, 1.35, t, fc=fc, ec=ec, fs=13.5, bold=(c == "r"))
        if j < 3:
            arrow(ax, (x + 2.07, y + 0.67), (x + 2.43, y + 0.67), c=ec, lw=2.5, ms=18)
    if k < 3:
        arrow(ax, (9.0, y - 0.03), (9.0, y - 0.62), c=NAVY, lw=2.5, ms=18) if False else None
ax.text(5, 0.18, "grey = tried and reported as a negative result;  blue = components reused by the ARL;  orange = submitted", ha="center", fontsize=13, style="italic")
fig.savefig(OUT + "/Workflow.png", dpi=200); plt.close(fig)

# ---------------------------------------------------------------- Architecture (ARL)
fig, ax = canvas(10, 11.4)
# inputs
for x, nm, sub, c, lt in [(0.15, "TEXT", "caption", TXT, LT["t"]), (3.55, "SKETCH", "scribble contour", SKE, LT["s"]), (6.95, "REFERENCE", "neutral object, photo palette", REF, LT["r"])]:
    box(ax, x, 10.15, 2.9, 1.05, "", fc=lt, ec=c)
    ax.text(x + 1.45, 10.85, nm, ha="center", fontsize=17, fontweight="bold", color=c)
    ax.text(x + 1.45, 10.4, sub, ha="center", fontsize=12.5)
for x in (1.6, 5.0, 8.4):
    arrow(ax, (x, 10.12), (x, 9.62), lw=2.5, ms=18)
box(ax, 0.15, 8.9, 9.7, 0.7, "Frozen CLIP ViT-L/14 features (caption, object phrase, sketch, reference, grey reference, colour part)", fc=LT["g"], ec="#777", fs=13.5)
arrow(ax, (2.8, 8.87), (2.8, 8.32), lw=2.5, ms=18); arrow(ax, (7.2, 8.87), (7.2, 8.32), lw=2.5, ms=18)
# ARL frame
ax.add_patch(FancyBboxPatch((0.05, 3.55), 9.9, 4.85, boxstyle="round,pad=0,rounding_size=0.1", fc="#fffbea", ec="#b45309", lw=4))
ax.text(5, 8.02, "ATTRIBUTE ROUTING LAYER (ARL)", ha="center", fontsize=20, fontweight="bold", color="#b45309")
box(ax, 0.3, 6.9, 4.5, 0.95, "Governance net (trained)\n3 conflict flags: g_obj, g_col, g_mat", fc=LT["t"], ec=TXT, fs=13)
box(ax, 5.2, 6.9, 4.5, 0.95, "Router MLP (trained)\n6 cosines \u2192 cs, hold, ip, cfg", fc=LT["t"], ec=TXT, fs=13)
arrow(ax, (2.55, 6.87), (2.55, 6.42), lw=2.5, ms=16); arrow(ax, (7.45, 6.87), (7.45, 6.42), lw=2.5, ms=16)
ax.text(5, 6.2, "per-attribute routing decision", ha="center", fontsize=13, style="italic", color="#b45309")
# three route boxes
route = [(0.3, TXT, LT["t"], "Text path", "colour / material words\nblended to the neutral\ncaption by the flag:\n\u0070' = p \u2212 f\u00b7(p \u2212 p_neutral)"),
         (3.55, SKE, LT["s"], "Sketch path", "ControlNet scale cs,\nheld until hold time,\nthen relaxed (\u00d70.3)\n"),
         (6.8, REF, LT["r"], "Reference path", "IP-Adapter scale ip,\nstyle-block schedule:\nreference supplies\ncolour and material")]
for x, c, lt, h, body in route:
    box(ax, x, 3.75, 2.9, 2.2, "", fc=lt, ec=c)
    ax.text(x + 1.45, 5.65, h, ha="center", fontsize=15, fontweight="bold", color=c)
    ax.text(x + 1.45, 4.6, body, ha="center", va="center", fontsize=12, linespacing=1.3)
for x in (1.75, 5.0, 8.25):
    arrow(ax, (x, 3.52), (x, 3.0), lw=3, ms=20)
# backbone
ax.add_patch(FancyBboxPatch((0.05, 0.95), 9.9, 2.0, boxstyle="round,pad=0,rounding_size=0.1", fc="#fff", ec=NAVY, lw=3.5))
ax.text(5, 2.6, "FROZEN SDXL BACKBONE (no weights changed)", ha="center", fontsize=16, fontweight="bold", color=NAVY)
for x, c, lt, t in [(0.3, TXT, LT["t"], "UNet cross-attention\n+ pooled text"), (3.55, SKE, LT["s"], "Scribble ControlNet\nresiduals (+ its text)"), (6.8, REF, LT["r"], "IP-Adapter ViT-H\ndecoupled attention")]:
    box(ax, x, 1.1, 2.9, 1.25, t, fc=lt, ec=c, fs=13)
arrow(ax, (5, 0.92), (5, 0.55), lw=3, ms=20)
box(ax, 3.2, 0.0, 3.6, 0.5, "Generated image", fc=NAVY, ec=NAVY, tc="w", fs=15, bold=True)
fig.savefig(OUT + "/Architecture.png", dpi=200); plt.close(fig)

# ---------------------------------------------------------------- Qualitative grid
arms = [("sketch", "Sketch"), ("ref", "Reference"), ("A_fixed", "Fixed"), ("B_router", "Router"), ("D_flaggate", "Flag gate"), ("E_router_flaggate", "ARL\n(router + gate)")]
rowsq = [(0, "Object conflict\n\u201cBlack ape\u2026\u201d\nsketch is an axe"), (1, "Colour conflict\n\u201c\u2026axe\u2026, black\u201d\nref: brown leather"), (2, "Material conflict\n\u201c\u2026made of metal\u201d\nref: leather")]
fig, axs = plt.subplots(3, 7, figsize=(14, 6.0), gridspec_kw=dict(width_ratios=[1.35] + [1] * 6, wspace=0.03, hspace=0.0))
for i, (k, lab) in enumerate(rowsq):
    axs[i, 0].axis("off"); axs[i, 0].text(0.97, 0.5, lab, ha="right", va="center", fontsize=15, fontweight="bold", color=NAVY, linespacing=1.3)
    for j, (a, t) in enumerate(arms):
        ax = axs[i, j + 1]
        ax.imshow(Image.open(IMG + f"/q_{k}_{a}.jpg")); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_linewidth(3 if a == "E_router_flaggate" else 1); s.set_edgecolor("#b45309" if a == "E_router_flaggate" else "#555")
        if i == 0: ax.set_title(t, fontsize=15, fontweight="bold", color="#b45309" if a == "E_router_flaggate" else "k", pad=6)
fig.savefig(OUT + "/Qualitative.png", dpi=170, bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- Pilot metrics
txt = """id kind arm edge_iou col_hit mat_hit obj_hit dino_ref clip_t
o obj A_fixed 0.044 1 1 0 0.243 0.205
o obj B0_scheduled 0.132 1 0 0 0.110 0.239
o obj B_router 0.064 1 1 0 0.142 0.248
o obj D_flaggate 0.127 1 0 0 0.201 0.204
o obj D_hard 0.103 0 1 0 0.213 0.189
o obj E_router_flaggate 0.118 1 1 0 0.161 0.235
c col A_fixed 0.709 0 1 nan 0.308 0.200
c col B0_scheduled 0.037 1 0 nan 0.129 0.239
c col B_router 0.039 1 0 nan 0.137 0.255
c col D_flaggate 0.288 0 1 nan 0.252 0.210
c col D_hard 0.089 1 1 nan 0.264 0.232
c col E_router_flaggate 0.029 1 1 nan 0.159 0.248
m mat A_fixed 0.380 0 0 nan 0.322 0.210
m mat B0_scheduled 0.028 1 0 nan 0.126 0.221
m mat B_router 0.028 1 0 nan 0.163 0.243
m mat D_flaggate 0.695 0 0 nan 0.302 0.216
m mat D_hard 0.638 1 1 nan 0.270 0.214
m mat E_router_flaggate 0.029 1 0 nan 0.157 0.258"""
S = pd.read_csv(io.StringIO(txt), sep=" ")
order = ["A_fixed", "B0_scheduled", "B_router", "D_flaggate", "D_hard", "E_router_flaggate"]
M = S.groupby("arm")[["col_hit", "mat_hit", "edge_iou", "dino_ref", "clip_t"]].mean().loc[order]
M.round(3).to_csv(os.path.join(REPO, "results", "pilot_flaggate_v1", "scores_mean_per_arm.csv"))
print(M.round(3))
names = ["Fixed", "Schedule\n(control)", "Router", "Flag gate", "Flag=1\n(control)", "ARL"]
cols = ["#8c8c8c", "#b7b7d9", "#4c78d0", "#4aa37a", "#9bd3b4", "#d9731a"]
fig, axs = plt.subplots(1, 3, figsize=(13, 4.4))
for ax, (c, ttl) in zip(axs, [("col_hit", "Reference colour reached"), ("edge_iou", "Sketch fidelity (edge IoU)"), ("clip_t", "Text\u2013image CLIP score")]):
    b = ax.bar(range(6), M[c].values, color=cols, edgecolor="k", lw=1.2)
    ax.set_xticks(range(6)); ax.set_xticklabels(names, fontsize=10.5); ax.set_title(ttl, fontsize=15, fontweight="bold", color=NAVY)
    for r_, v in zip(b, M[c].values): ax.text(r_.get_x() + r_.get_width() / 2, v + (0.02 if c != "clip_t" else 0.004), f"{v:.2f}" if c != "clip_t" else f"{v:.3f}", ha="center", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False); ax.tick_params(axis="y", labelsize=11)
    if c == "col_hit": ax.set_ylim(0, 1.15)
    if c == "edge_iou": ax.set_ylim(0, 0.5)
    if c == "clip_t": ax.set_ylim(0.15, 0.28)
plt.tight_layout(); fig.savefig(OUT + "/Results_pilot.png", dpi=170); plt.close(fig)
