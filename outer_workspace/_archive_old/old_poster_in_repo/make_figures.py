"""Generates every figure of the poster into poster/figures/.

Everything plotted is either (a) drawn from the repo's own functions/constants (schedules from src/pipeline.py) or
(b) copied from the logged Colab outputs of the role-LoRA run (compliance.csv, shape.csv, leak_test.json).
No number is invented. Run:  python poster/make_figures.py   (needs matplotlib; `qrcode` optional for the QR).
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
NAVY, GREEN, ORANGE, GREY, RED = "#000080", "#2e8b2e", "#d9731a", "#6b6b6b", "#c0392b"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15})


def box(ax, x, y, w, h, text, fc="#e8eefc", ec=NAVY, fs=15, ls="-", tc="black", bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=2.2, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, fontweight="bold" if bold else "normal")


def arrow(ax, x0, y0, x1, y1, c=NAVY, ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=c, lw=2.4, ls=ls))


def swatch(ax, x, y, rgb, w=0.9, h=0.9, label=None, fs=13):
    ax.add_patch(Rectangle((x, y), w, h, fc=np.array(rgb) / 255, ec="black", lw=1.5))
    if label:
        ax.text(x + w / 2, y - 0.18, label, ha="center", va="top", fontsize=fs)


# ------------------------------------------------------------------ 1. intro: the conflict (logged example 0036_col0)
def intro():
    fig, ax = plt.subplots(figsize=(9, 9)); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    box(ax, 0.2, 8.0, 3.0, 1.6, "SKETCH\nshape, placement", "#fff4e0", ORANGE, bold=True)
    box(ax, 3.5, 8.0, 3.0, 1.6, "TEXT\nobject identity", "#e6f5e6", GREEN, bold=True)
    box(ax, 6.8, 8.0, 3.0, 1.6, "REFERENCE\ncolour + material", "#e8eefc", NAVY, bold=True)
    ax.text(5, 7.35, "Who governs what?", ha="center", fontsize=19, fontweight="bold")
    ax.text(5, 6.55, 'caption: "Vintage motorcycle with purple and silver finish"', ha="center", fontsize=14, style="italic")
    ax.text(5, 5.95, "reference object: silver, black and white", ha="center", fontsize=14, style="italic")
    ax.text(5, 4.95, "Mean RGB of the object region (logged, test photo 0036)", ha="center", fontsize=14, fontweight="bold")
    # values from leak_test.json: rgb_ref, rgb_1 (baseline arm, caption colour leaks)
    swatch(ax, 1.4, 3.1, (117, 116, 117), 2.2, 1.3, "reference\n(117,116,117)")
    swatch(ax, 4.0, 3.1, (93, 71, 97), 2.2, 1.3, "output, full caption\n(93,71,97): purple")
    ax.text(7.7, 3.75, "rule: the\nreference\nwins colour", ha="center", va="center", fontsize=15, color=RED, fontweight="bold")
    ax.text(5, 0.95, "With the full caption the output takes the caption's colour;\nthe rule says the reference should decide it.", ha="center", fontsize=14)
    fig.savefig(os.path.join(OUT, "Intro.png"), dpi=140, bbox_inches="tight"); plt.close(fig)


# ------------------------------------------------------------------ 2. workflow
def workflow():
    fig, ax = plt.subplots(figsize=(14, 6)); ax.set_xlim(0, 14); ax.set_ylim(0, 6); ax.axis("off")
    steps = [("Sketchy\nphotos + sketches", 0.1), ("Qwen3-VL-2B\ncaptions", 2.45), ("Reference v2\n(neutral object,\nphoto's colour)", 4.8),
             ("pilot_v1 +\nconflict rows\n(01b)", 7.15), ("Frozen SDXL\n+ neutral caption\n+ IP scale", 9.5), ("Per-conflict\nevaluation", 11.85)]
    for i, (t, x) in enumerate(steps):
        c = ORANGE if i == 4 else NAVY
        box(ax, x, 2.2, 2.05, 2.1, t, "#fff4e0" if i == 4 else "#e8eefc", c, fs=13, bold=(i == 4))
        if i:
            arrow(ax, x - 0.28, 3.25, x, 3.25)
    ax.text(7.0, 5.4, "Data preparation (offline)", ha="center", fontsize=15, color=NAVY, fontweight="bold")
    ax.plot([0.1, 9.2], [5.05, 5.05], color=NAVY, lw=2)
    ax.text(10.9, 5.4, "Generation and test", ha="center", fontsize=15, color=ORANGE, fontweight="bold")
    ax.plot([9.5, 13.9], [5.05, 5.05], color=ORANGE, lw=2)
    ax.text(7.0, 1.2, "200 photos: 160 train / 20 val / 20 test (split by photo).  Conflict rows: object, colour, material, swapped reference.",
            ha="center", fontsize=13)
    ax.text(7.0, 0.5, "Colab free T4, 512 px, SDXL base 1.0 + xinsir scribble ControlNet + IP-Adapter ViT-H.", ha="center", fontsize=13)
    fig.savefig(os.path.join(OUT, "Workflow.png"), dpi=140, bbox_inches="tight"); plt.close(fig)


# ------------------------------------------------------------------ 3. architecture: the rule sits OUTSIDE the model
def arch():
    fig, ax = plt.subplots(figsize=(14, 8.4)); ax.set_xlim(0, 14); ax.set_ylim(0, 8.4); ax.axis("off")
    # inputs
    box(ax, 0.1, 6.2, 2.4, 1.3, "caption", "#e6f5e6", GREEN, bold=True)
    box(ax, 0.1, 3.9, 2.4, 1.3, "sketch", "#fff4e0", ORANGE, bold=True)
    box(ax, 0.1, 1.6, 2.4, 1.3, "reference\nimage", "#e8eefc", NAVY, bold=True)
    # the rule, dashed, outside the model
    ax.add_patch(Rectangle((3.0, 5.6), 3.6, 2.5, fc="#fdecea", ec=RED, lw=2.5, ls="--"))
    ax.text(4.8, 7.8, "RULE (outside the model)", ha="center", fontsize=12, color=RED, fontweight="bold")
    box(ax, 3.2, 6.55, 3.2, 0.95, 'neutral caption\npurple -> "colored"', "#ffffff", RED, fs=12)
    box(ax, 3.2, 5.75, 3.2, 0.65, "IP scale = 1.0", "#ffffff", RED, fs=12)
    # frozen model
    ax.add_patch(Rectangle((7.2, 0.9), 6.6, 7.0, fc="#f4f7ff", ec=NAVY, lw=2.5))
    ax.text(10.5, 7.55, "FROZEN SDXL (no weights changed)", ha="center", fontsize=13, color=NAVY, fontweight="bold")
    box(ax, 7.5, 5.9, 2.8, 1.1, "text encoders\n(neutral caption)", fs=12)
    box(ax, 7.5, 3.9, 2.8, 1.1, "scribble ControlNet\n(neutral caption)", fs=12)
    box(ax, 7.5, 1.9, 2.8, 1.1, "IP-Adapter\n(style block, scale 1.0)", fs=12)
    box(ax, 10.9, 3.4, 2.6, 2.2, "SDXL UNet\n+ VAE", "#dfe8ff", NAVY, fs=14, bold=True)
    for y in (6.45, 4.45, 2.45):
        arrow(ax, 10.3, y, 10.9, 4.5)
    arrow(ax, 2.5, 6.85, 3.2, 6.95); arrow(ax, 6.4, 6.95, 7.5, 6.45)
    arrow(ax, 2.5, 4.55, 7.5, 4.45)
    arrow(ax, 2.5, 2.25, 7.5, 2.45)
    arrow(ax, 4.8, 5.75, 4.8, 3.05, c=RED, ls="--"); ax.plot([4.8, 7.5], [3.05, 2.5], color=RED, lw=2, ls="--")
    ax.text(12.2, 1.35, "image", ha="center", fontsize=14, fontweight="bold"); arrow(ax, 12.2, 3.4, 12.2, 1.7)
    fig.savefig(os.path.join(OUT, "Deep-Neural-Network-Model.png"), dpi=140, bbox_inches="tight"); plt.close(fig)


# ------------------------------------------------------------------ 4. schedules (from src/pipeline.py: cs_shape, ip_shape, CENTER)
def schedules():
    f = np.linspace(0, 1, 400)
    cs = np.where(f < 0.7, 1.0, 0.3) * 1.0                                  # CENTER cs=1.0, hold=0.7, RELAX=0.3
    ipshape = np.where(f < 0.15, 0.0, np.minimum(1.0, (f - 0.15) / 0.25))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(f, cs, color=ORANGE, lw=3.5, label="ControlNet (sketch)")
    ax.plot(f, 0.5 * ipshape, color=NAVY, lw=3, ls="--", label="IP-Adapter, baseline (0.5)")
    ax.plot(f, 1.0 * ipshape, color=RED, lw=3.5, label="IP-Adapter, rule (1.0)")
    ax.set_xlabel("denoising progress (0 = noise, 1 = image)"); ax.set_ylabel("conditioning scale")
    ax.set_ylim(-0.05, 1.15); ax.grid(alpha=0.3); ax.legend(fontsize=12, loc="center left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "schedules.png"), dpi=140); plt.close(fig)


# ------------------------------------------------------------------ 5. results (compliance.csv, shape.csv of the role-LoRA run; n = 10 per kind)
def results():
    kinds = ["object\n(text governs)", "colour\n(ref governs)", "material\n(ref governs)", "swapped ref\n(ref governs)"]
    fixed = [0.10, 0.70, 0.67, 0.70]; rule = [0.30, 0.70, 0.67, 0.00]; lora = [0.30, 0.30, 0.00, 0.40]
    x = np.arange(4); w = 0.26
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w, fixed, w, color=GREY, label="fixed baseline")
    ax.bar(x, rule, w, color=RED, label="neutral caption + IP scale (submitted)")
    ax.bar(x + w, lora, w, color="#b9c6ee", hatch="//", ec=NAVY, label="learned LoRA (not submitted)")
    for xs, vs in ((x - w, fixed), (x, rule), (x + w, lora)):
        for a, v in zip(xs, vs):
            ax.text(a, v + 0.015, f"{v:.2f}", ha="center", fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels(kinds, fontsize=12); ax.set_ylim(0, 0.9)
    ax.set_ylabel("share where the governing input won"); ax.legend(fontsize=11, loc="upper right"); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "results_compliance.png"), dpi=140); plt.close(fig)


# ------------------------------------------------------------------ 6. neutral-caption illustration (word-level; the code pads so BPE length is unchanged)
def neutral():
    fig, ax = plt.subplots(figsize=(10, 3.6)); ax.set_xlim(0, 10); ax.set_ylim(0, 3.6); ax.axis("off")
    ax.text(0.1, 2.9, "caption   ", fontsize=15, fontweight="bold", va="center")
    ax.text(1.9, 2.9, "Vintage motorcycle with", fontsize=15, va="center")
    ax.text(5.55, 2.9, "purple", fontsize=15, va="center", color=RED, fontweight="bold")
    ax.text(6.5, 2.9, "and", fontsize=15, va="center"); ax.text(7.15, 2.9, "silver", fontsize=15, va="center", color=RED, fontweight="bold")
    ax.text(8.05, 2.9, "finish", fontsize=15, va="center")
    arrow(ax, 5.9, 2.5, 5.9, 1.45, c=RED); arrow(ax, 7.5, 2.5, 7.5, 1.45, c=RED)
    ax.text(0.1, 1.0, "neutral    ", fontsize=15, fontweight="bold", va="center")
    ax.text(1.9, 1.0, "Vintage motorcycle with", fontsize=15, va="center")
    ax.text(5.55, 1.0, "colored", fontsize=15, va="center", color=GREEN, fontweight="bold")
    ax.text(6.5, 1.0, "and", fontsize=15, va="center"); ax.text(7.15, 1.0, "colored", fontsize=15, va="center", color=GREEN, fontweight="bold")
    ax.text(8.05, 1.0, "finish", fontsize=15, va="center")
    ax.text(5, 0.2, "Used for the UNet text, the pooled embedding and the ControlNet text. Material words -> \"solid\".", ha="center", fontsize=12)
    fig.savefig(os.path.join(OUT, "neutral_caption.png"), dpi=140, bbox_inches="tight"); plt.close(fig)


def qr():
    try:
        sys.path.insert(0, os.environ.get("QR_PKGS", ""))
        import qrcode
    except Exception as e:
        print("qrcode not available, keeping the template QR:", e); return
    qrcode.make("https://github.com/Harshitha-282007/IE643_final").save(os.path.join(OUT, "Repo_QR.png"))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for fn in (intro, workflow, arch, schedules, results, neutral, qr):
        fn(); print("ok", fn.__name__)
