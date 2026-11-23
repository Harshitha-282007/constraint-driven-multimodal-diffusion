"""
M1 signal probe and one-example overfit. See m1/routing.py for what is trained.

Losses (all on the same x_t and noise, so no target and no irreducible noise):

    L_inv  = mean || eps_S(x_t; caption, reference, s=1) - sg[ eps_T(x_t) ] ||^2        teacher T = the neutral-caption rule
    L_keep = mean || eps_S(x_t; caption, no reference, s=0) - sg[ eps_B(x_t; no reference) ] ||^2    B = baseline, caption as written
    L_std  = mean || eps_S - eps ||^2                                                    ordinary denoising loss (comparison only)
"""
import json
import os
import zlib

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from eval_roles import kind_of
from governance import labels as L
from governance.metrics import rgb_to_lab
from m1.core import batch, encode_target, generate, installed, make_ctx, noisy_input, predict
from rolegate.tokens import COLOUR, IDENTITY, MATERIAL, neutral_caption, token_groups


def seed_of(s):
    return zlib.crc32(str(s).encode()) % 100000


# ----------------------------------------------------------------------------- data
def select(base_rows, n=5, split="train", seed=1):
    """n photos, each with its base row and one colour / reference / object variant."""
    rows = L.expand_rows(base_rows, per_type=1, seed=seed, kinds=("obj", "col", "ref"))
    by = {}
    for r in rows:
        if r["split"] == split:
            by.setdefault(r["original_id"], {})[kind_of(r)] = r
    return [by[o] for o in sorted(by) if {"base", "col", "ref", "obj"} <= set(by[o])][:n]


def build_items(P, data_dir, groups):
    from pipeline import seed_for
    rows = [r for g in groups for r in g.values()]
    P.prepare_object_dirs([r.get("ref_object") or "object" for r in rows])
    op = lambda r, k: Image.open(f"{data_dir}/{r[k]}")
    items = {}
    for i in range(0, len(rows), 8):
        part = rows[i:i + 8]
        specs = [dict(key=r["id"], sketch=op(r, "sketch"), ref=op(r, "reference"), caption=r["caption"], obj=r["category"],
                      seed=seed_for(r["original_id"]), ref_object=r.get("ref_object")) for r in part]
        for r, it in zip(part, P.make_items(specs)):
            it.pop("S_gh", None)
            items[r["id"]] = it
    return items


# ----------------------------------------------------------------------------- 1. token grouping
def check_grouping(P, groups, items):
    """The neutral caption must change only colour (material) tokens; report how far the text embedding moves per token group."""
    tok = P.pipe.tokenizer
    out = dict(tested=0, no_neutral=0, violations=[], dist={"identity": [], "colour": [], "material": [], "other": []}, pooled=[])
    for g in groups:
        for r in (g["base"], g["col"]):
            it, cap = items[r["id"]], r["caption"]
            for w in ("colour", "material"):
                neu = neutral_caption(tok, cap, w)
                if neu is None:
                    out["no_neutral"] += 1
                    continue
                out["tested"] += 1
                a, b = tok.tokenize(cap), tok.tokenize(neu)
                grp = token_groups(tok, cap, r["category"])
                want = COLOUR if w == "colour" else MATERIAL
                bad = [(a[i], b[i], grp[i + 1]) for i in range(len(a)) if a[i] != b[i] and grp[i + 1] != want]
                if bad:
                    out["violations"].append(dict(id=r["id"], caption=cap, neutral=neu, which=w, bad=bad))
                pe, pn = it["txt"][0][0].float(), it["txt_neutral"][w][0][0].float()
                dist = 1 - F.cosine_similarity(pe, pn, dim=-1)
                gid = torch.tensor(it["text_groups"])
                for name, k in (("identity", IDENTITY), ("colour", COLOUR), ("material", MATERIAL), ("other", 0)):
                    m = gid == k
                    m[len(a) + 2:] = False
                    if m.any():
                        out["dist"][name].append(float(dist[m].mean()))
                out["pooled"].append(float(1 - F.cosine_similarity(it["txt"][2][0].float(), it["txt_neutral"][w][1][0].float(), dim=-1)))
    out["dist_mean"] = {k: (round(float(np.mean(v)), 4) if v else None) for k, v in out["dist"].items()}
    out["pooled_mean"] = round(float(np.mean(out["pooled"])), 4) if out["pooled"] else None
    return out


# ----------------------------------------------------------------------------- 2. equivalence with the real pipeline
def _diff(a, b):
    d = np.abs(np.asarray(a.convert("RGB"), np.int16) - np.asarray(b.convert("RGB"), np.int16))
    return dict(max=int(d.max()), mean=round(float(d.mean()), 3))


def check_equivalence(P, gates, it, ip=0.5):
    """predict()/generate() against Pipeline.run_role on the same item and seed."""
    run = lambda *a, **k: P.run_role(it, gates, *a, **k)[0]
    ref_teacher = run([0, 1, 1], knock=("text_colour", "text_material"), cn_text="mix", p=dict(ip=ip))
    ref_base = run([0, 0, 0], p=dict(ip=ip))
    my = {m: generate(P, gates, it, m, ip=ip) for m in ("teacher", "base", "student")}
    return dict(teacher_vs_run_role=_diff(my["teacher"], ref_teacher), base_vs_run_role=_diff(my["base"], ref_base),
                student_init_vs_base=_diff(my["student"], my["base"]), teacher_vs_base=_diff(my["teacher"], my["base"])), \
        dict(ref_teacher=ref_teacher, ref_base=ref_base, **my)


# ----------------------------------------------------------------------------- 3. sensitivity (step level)
@torch.no_grad()
def sensitivity(P, gates, groups, items, sched, data_dir, ts=(300, 550, 800), modes=("base", "teacher")):
    """RMS change of the noise prediction when ONE input changes, on the same x_t:
    S_col = caption colour word, S_ref = reference, S_obj = object noun (caption of another object)."""
    rows = []
    rms = lambda a, b: float(((a - b) ** 2).mean().sqrt())
    with installed(P, gates):
        for g in groups:
            z0 = encode_target(P, g["base"], data_dir)
            cb = {k: batch(make_ctx(P, items[g[k]["id"]]), cfg=False) for k in ("base", "col", "ref", "obj")}
            for t in ts:
                x, _ = noisy_input(sched, z0, t, seed_of(g["base"]["id"]) + t)
                for mode in modes:
                    e = {k: predict(P, gates, v, x, torch.tensor(t), 1 - t / 1000, mode, autocast=True).float() for k, v in cb.items()}
                    rows.append(dict(photo=g["base"]["id"], t=t, mode=mode, S_col=rms(e["col"], e["base"]), S_ref=rms(e["ref"], e["base"]),
                                     S_obj=rms(e["obj"], e["base"])))
    return rows


# ----------------------------------------------------------------------------- 4. gradients of the two loss terms
def _mse(a, b):
    return ((a.float() - b.float()) ** 2).mean()


def grad_probe(P, gates, router, groups, items, sched, data_dir, bins=((250, 450), (450, 650), (650, 850)), n=5, seed=0):
    """Gradient of L_inv and of L_std w.r.t. the trainable scalars, per timestep band."""
    rng = np.random.default_rng(seed)
    rows = []
    P.unet.enable_gradient_checkpointing(); P.unet.train()
    try:
        with installed(P, gates):
            for gi, g in enumerate(groups[:n]):
                r = g["col"]
                z0 = encode_target(P, g["base"], data_dir)
                c = batch(make_ctx(P, items[r["id"]], with_ref=True), cfg=False)
                for bi, (lo, hi) in enumerate(bins):
                    t = int(rng.integers(lo, hi))
                    x, noise = noisy_input(sched, z0, t, 1000 * gi + bi)
                    tt, f = torch.tensor(t), 1 - t / 1000
                    with torch.no_grad():
                        eps_T = predict(P, gates, c, x, tt, f, "teacher", autocast=True)
                    eps_S = predict(P, gates, c, x, tt, f, "student", autocast=True)
                    L_inv, L_std = _mse(eps_S, eps_T), _mse(eps_S, noise)
                    g_inv = dict(zip(router.params, torch.autograd.grad(L_inv, router.params, retain_graph=True, allow_unused=True)))
                    g_std = dict(zip(router.params, torch.autograd.grad(L_std, router.params, allow_unused=True)))
                    vi, vs = router.flat(g_inv), router.flat(g_std)
                    ri, rs = router.grad_report(g_inv), router.grad_report(g_std)
                    rows.append(dict(photo=r["original_id"], t=t, band=bi, L_inv=float(L_inv), L_std=float(L_std), norm_inv=float(vi.norm()),
                                     norm_std=float(vs.norm()), cos_inv_std=float(F.cosine_similarity(vi, vs, dim=0)),
                                     mean_grad_w_colour=ri["cross_colour_w"]["mean"], mean_grad_b_colour=ri["cross_colour_b"]["mean"],
                                     inv_groups={k: round(v["norm"], 6) for k, v in ri.items()}, std_groups={k: round(v["norm"], 6) for k, v in rs.items()}))
                    del eps_S, L_inv, L_std
    finally:
        P.unet.disable_gradient_checkpointing(); P.unet.eval()
    return rows


def keep_probe(P, gates, router, groups, items, sched, data_dir, t=450):
    """L_keep (no reference, s=0): gradient at initialisation and after shifting every colour/material bias by +0.3."""
    g = groups[0]
    r = g["col"]
    z0 = encode_target(P, g["base"], data_dir)
    c = batch(make_ctx(P, items[r["id"]], with_ref=False), cfg=False)
    x, _ = noisy_input(sched, z0, t, 77)
    tt, f = torch.tensor(t), 1 - t / 1000
    out = {}
    P.unet.enable_gradient_checkpointing(); P.unet.train()
    try:
        with installed(P, gates):
            with torch.no_grad():
                eps_B = predict(P, gates, c, x, tt, f, "base", autocast=True)

            def measure():
                eps_S = predict(P, gates, c, x, tt, f, "student", autocast=True)
                loss = _mse(eps_S, eps_B)
                grads = dict(zip(router.params, torch.autograd.grad(loss, router.params, allow_unused=True)))
                rep = router.grad_report(grads)
                return dict(L_keep=float(loss), w_colour=rep["cross_colour_w"]["abs"], b_colour_mean=rep["cross_colour_b"]["mean"],
                            w_material=rep["cross_material_w"]["abs"], b_material_mean=rep["cross_material_b"]["mean"])
            out["at_init"] = measure()
            u1, u2 = router.shift("cross_colour_b", 0.3), router.shift("cross_material_b", 0.3)
            out["b_plus_0.3"] = measure()
            u1(); u2()
    finally:
        P.unet.disable_gradient_checkpointing(); P.unet.eval()
    return out


# ----------------------------------------------------------------------------- 5. appearance: does swapping the reference change colour as intended
def _lab_centre(im):
    a = np.asarray(im.convert("RGB"), float)
    h, w = a.shape[:2]
    return rgb_to_lab(a[h // 4: 3 * h // 4, w // 4: 3 * w // 4].reshape(-1, 3)).mean(0)


def appearance(P, gates, groups, items, n=3, ip=0.5, res=640):
    """Same caption and seed, reference of the base row vs reference of another photo. Output colour shift vs reference colour shift."""
    old = P.res
    P.res = res
    rows, imgs = [], {}
    try:
        for g in groups[:n]:
            a, b = items[g["base"]["id"]], items[g["ref"]["id"]]
            ref_shift = _lab_centre(b["ref"])[1:] - _lab_centre(a["ref"])[1:]
            for mode in ("base", "teacher"):
                ia, ib = generate(P, gates, a, mode, ip=ip), generate(P, gates, b, mode, ip=ip)
                out_shift = _lab_centre(ib)[1:] - _lab_centre(ia)[1:]
                cos = float(out_shift @ ref_shift / (np.linalg.norm(out_shift) * np.linalg.norm(ref_shift) + 1e-9))
                rows.append(dict(photo=g["base"]["id"], mode=mode, ref_shift=float(np.linalg.norm(ref_shift)), out_shift=float(np.linalg.norm(out_shift)),
                                 cos=round(cos, 3), pixel_diff=_diff(ia, ib)["mean"]))
                imgs[(g["base"]["id"], mode)] = (ia, ib)
    finally:
        P.res = old
    return rows, imgs


# ----------------------------------------------------------------------------- 6. one-example overfit
def overfit(P, gates, router, group, items, sched, data_dir, steps=80, lr=0.1, lam_keep=4.0, seed=0, log=print):
    r = group["col"]
    it = items[r["id"]]
    z0 = encode_target(P, group["base"], data_dir)
    c_ref = batch(make_ctx(P, it, with_ref=True), cfg=False)
    c_no = batch(make_ctx(P, it, with_ref=False), cfg=False)
    opt = torch.optim.AdamW(router.params, lr=lr, weight_decay=0.0)
    rng = np.random.default_rng(seed)

    def step_losses(t, sd, grad):
        x, _ = noisy_input(sched, z0, t, sd)
        tt, f = torch.tensor(t), 1 - t / 1000
        with torch.no_grad():
            eps_T = predict(P, gates, c_ref, x, tt, f, "teacher", autocast=True)
            eps_B0 = predict(P, gates, c_no, x, tt, f, "base", autocast=True)
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            L_inv = _mse(predict(P, gates, c_ref, x, tt, f, "student", autocast=True), eps_T)
            L_keep = _mse(predict(P, gates, c_no, x, tt, f, "student", autocast=True), eps_B0)
        return L_inv, L_keep

    def grid():
        a = [step_losses(t, 5000 + i, False) for i, t in enumerate((250, 400, 550, 700, 850))]
        return dict(L_inv=float(np.mean([float(x[0]) for x in a])), L_keep=float(np.mean([float(x[1]) for x in a])))

    def gates_now():
        b, w = router.values("cross_colour_b"), router.values("cross_colour_w")
        return dict(g_ref_present=float((2 * torch.sigmoid(b + w)).mean()), g_ref_absent=float((2 * torch.sigmoid(b)).mean()),
                    pooled_present=float(2 * torch.sigmoid(router.values("pooled_colour_b") + router.values("pooled_colour_w"))),
                    pooled_absent=float(2 * torch.sigmoid(router.values("pooled_colour_b"))))

    P.unet.enable_gradient_checkpointing(); P.unet.train()
    hist = []
    try:
        with installed(P, gates):
            before = dict(grid(), **gates_now())
            for s in range(steps):
                t = int(rng.integers(200, 850))
                L_inv, L_keep = step_losses(t, s, True)
                loss = L_inv + lam_keep * L_keep
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                hist.append(dict(step=s, t=t, L_inv=float(L_inv), L_keep=float(L_keep)))
                if (s + 1) % 10 == 0:
                    log(f"step {s + 1:3d} L_inv {np.mean([h['L_inv'] for h in hist[-10:]]):.6f} L_keep {np.mean([h['L_keep'] for h in hist[-10:]]):.6f} {gates_now()}")
            after = dict(grid(), **gates_now())
    finally:
        P.unet.disable_gradient_checkpointing(); P.unet.eval()
    return dict(before=before, after=after, hist=hist, untouched_ok=router.untouched_ok(), trained=router.n_trained())
