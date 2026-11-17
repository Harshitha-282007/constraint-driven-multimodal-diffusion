"""
Gate labels and conflict-row generation. Pure Python (no torch), so it is unit-tested locally.

Agreed rules (single object):
  geometry + layout  <- sketch, always
  object identity    <- text            -> g_obj = 1 when the caption's object differs from the sketch's object
  colour             <- text if the caption states one, otherwise the reference
                                        -> g_col = 1 when the caption states a colour the reference does not have
  material           <- text if the caption states one, otherwise the reference
                                        -> g_mat = 1 when the caption states a material the reference does not have

A gate of 1 means "the reference (or the sketch, for g_obj) must NOT supply this attribute".
The row makers are ported from notebooks/data_prep/01b_add_conflicts.ipynb.
"""
import hashlib
import random
import re

COLORS = ["red", "blue", "green", "yellow", "orange", "purple", "pink", "black", "white",
          "brown", "gray", "silver", "gold", "beige"]
# (adjective used in captions, stem looked for in the reference's material string)
MATERIALS = [("wooden", "wood"), ("metal", "metal"), ("plastic", "plastic"), ("ceramic", "ceramic"),
             ("glass", "glass"), ("leather", "leather"), ("fabric", "fabric"), ("stone", "stone")]
GATES = ["g_obj", "g_col", "g_mat"]
_CANON = {"grey": "gray"}

# Words the label rules do NOT know. Used only to build an out-of-vocabulary test set whose labels are
# still known by construction (see paraphrase_caption): it checks the learned network on wording the
# word lists cannot match.
SYNONYMS = {
    "red": ["crimson", "scarlet"], "blue": ["navy", "azure"], "green": ["emerald", "olive"],
    "yellow": ["lemon-coloured", "mustard"], "orange": ["tangerine", "amber"], "purple": ["violet", "lavender"],
    "pink": ["rose-coloured", "magenta"], "black": ["jet-dark", "charcoal"], "white": ["ivory", "snowy"],
    "brown": ["chestnut", "chocolate-coloured"], "gray": ["slate-coloured", "ash-coloured"],
    "silver": ["chrome", "steel-coloured"], "gold": ["golden", "brass-coloured"], "beige": ["cream", "tan"],
    "wooden": ["timber", "oak"], "wood": ["timber", "oak"], "metal": ["steel", "iron"],
    "plastic": ["acrylic", "PVC"], "ceramic": ["porcelain", "earthenware"], "glass": ["crystal", "glassy"],
    "leather": ["suede", "hide"], "fabric": ["cloth", "woven textile"], "stone": ["granite", "marble"],
}


def _words(s):
    return [_CANON.get(w, w) for w in re.findall(r"[a-z]+", str(s).lower())]


def caption_colors(caption):
    """Colour words stated in the caption, in order, without repeats."""
    seen = []
    for w in _words(caption):
        if w in COLORS and w not in seen:
            seen.append(w)
    return seen


def caption_materials(caption):
    """Material stems stated in the caption ('wooden' and 'wood' both give 'wood')."""
    ws = set(_words(caption))
    return [stem for adj, stem in MATERIALS if adj in ws or stem in ws]


def ref_colors(row):
    return [w for w in dict.fromkeys(_words(row.get("ref_colors", ""))) if w in COLORS]


def ref_has_material(row, stem):
    m = str(row.get("ref_material", "")).lower()
    return stem in m or stem[:4] in m


def gates_for(row):
    """Rule labels for one metadata row -> {'g_obj','g_col','g_mat'} in {0,1}."""
    cap_cat = row.get("caption_category", row["category"])
    cc, rc = caption_colors(row["caption"]), ref_colors(row)
    cm = caption_materials(row["caption"])
    return {
        "g_obj": int(cap_cat != row["category"]),
        "g_col": int(bool(cc) and any(c not in rc for c in cc)),
        "g_mat": int(bool(cm) and any(not ref_has_material(row, s) for s in cm)),
    }


def governing(row):
    """Readable governance map for one row (what the interface shows)."""
    g = gates_for(row)
    return {
        "geometry": "sketch", "layout": "sketch",
        "object": "text",
        "colour": "text" if caption_colors(row["caption"]) else "reference",
        "material": "text" if caption_materials(row["caption"]) else "reference",
        "conflicts": [k for k, v in g.items() if v],
    }


# ----------------------------------------------------------------------------- caption edits
def strip_colors(caption):
    """Remove colour words (used by the LoRA 'colour from the reference' regime)."""
    out = caption
    for c in COLORS + ["grey"]:
        out = re.sub(rf"\b{c}\b[- ]?", "", out, flags=re.I)
    out = re.sub(r"\s{2,}", " ", out).strip(" ,")
    return (out[:1].upper() + out[1:]) if out else caption


def ensure_color(caption, colors):
    """Make sure the caption states a colour (used by the LoRA 'colour from the text' regime)."""
    if caption_colors(caption) or not colors:
        return caption
    return caption.rstrip(". ") + f", {colors[0]}."


def paraphrase_caption(caption, rng):
    """Replace every known colour / material word by a synonym the rules do not know."""
    def sub(m):
        w = m.group(0)
        alts = SYNONYMS.get(_CANON.get(w.lower(), w.lower()))
        if not alts:
            return w
        a = rng.choice(alts)
        return a.capitalize() if w[:1].isupper() else a
    vocab = sorted(set(COLORS + ["grey"] + [a for a, _ in MATERIALS] + [s for _, s in MATERIALS]), key=len, reverse=True)
    return re.sub(r"\b(" + "|".join(vocab) + r")\b", sub, caption, flags=re.I)


# ----------------------------------------------------------------------------- row makers (from 01b)
REF_FIELDS = ["reference", "ref_colors", "ref_material", "ref_finish", "ref_object", "ref_prompt"]


def _rng(*parts):
    return random.Random(int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:8], 16))


def _variant(base, tag):
    r = dict(base)
    r.update(id=f"{base['id']}_{tag}", original_id=base.get("original_id", base["id"]),
             caption_category=base.get("caption_category", base["category"]))
    return r


def make_object(base, pool, rng):
    cands = [p for p in pool if p["category"] != base["category"]]
    if not cands:
        return None
    p = rng.choice(cands)
    r = _variant(base, "obj")
    r.update(caption=p["caption"], caption_category=p["category"])
    return r


def make_color(base, pool, rng):
    avoid = set(ref_colors(base)) | set(caption_colors(base["caption"]))
    options = [c for c in COLORS if c not in avoid]
    if not options:
        return None
    new, cap = rng.choice(options), base["caption"]
    present = caption_colors(cap)
    if present:
        cap2 = re.sub(rf"\b{present[0]}\b", new, cap, count=1, flags=re.I)
        if cap[:1].isupper():
            cap2 = cap2[:1].upper() + cap2[1:]
    else:
        cap2 = cap.rstrip(". ") + f", {new}."
    r = _variant(base, "col")
    r.update(caption=cap2)
    return r


def make_material(base, pool, rng):
    options = [(a, s) for a, s in MATERIALS if not ref_has_material(base, s)]
    if not options:
        return None
    adj, stem = rng.choice(options)
    cap = base["caption"]
    present = [a for a, s in MATERIALS if re.search(rf"\b({a}|{s})\b", cap, re.I)]
    if present:
        cap2 = re.sub(rf"\b{present[0]}\b", adj, cap, count=1, flags=re.I)
    else:
        cap2 = cap.rstrip(". ") + f", made of {stem}."
    r = _variant(base, "mat")
    r.update(caption=cap2)
    return r


def make_refswap(base, pool, rng):
    cands = [p for p in pool if p["id"] != base["id"] and p.get("ref_colors") != base.get("ref_colors")]
    if not cands:
        return None
    p = rng.choice(cands)
    r = _variant(base, "ref")
    for k in REF_FIELDS:
        if k in p:
            r[k] = p[k]
    return r


def make_agree(base, pool, rng):
    """Conflict-free variant: the caption states exactly what the reference shows (all gates 0)."""
    rc = ref_colors(base)
    cap = strip_colors(base["caption"])
    for adj, stem in MATERIALS:
        cap = re.sub(rf"\b({adj}|{stem})\b[- ]?", "", cap, flags=re.I)
    cap = re.sub(r"\s{2,}", " ", cap).strip(" ,").rstrip(".")
    if rc and rng.random() < 0.7:
        cap += f", {rc[0]}"
    r = _variant(base, "agr")
    r.update(caption=cap + ".")
    return r


MAKERS = {"obj": make_object, "col": make_color, "mat": make_material, "ref": make_refswap, "agr": make_agree}


def expand_rows(base_rows, per_type=3, seed=0, kinds=tuple(MAKERS)):
    """Base rows -> base rows + `per_type` variants of each kind, each kept in its base photo's split.

    Every returned row carries `gates` (rule labels) and `original_id` (the base photo, used for group splits).
    """
    by_split = {}
    for r in base_rows:
        by_split.setdefault(r["split"], []).append(r)
    out = []
    for split, pool in by_split.items():
        for base in pool:
            b = dict(base)
            b.setdefault("original_id", base["id"])
            b.setdefault("caption_category", base["category"])
            out.append(b)
            for kind in kinds:
                for k in range(per_type):
                    row = MAKERS[kind](base, pool, _rng(seed, base["id"], kind, k))
                    if row is not None:
                        row["id"] += f"{k}"
                        out.append(row)
    for r in out:
        r["gates"] = gates_for(r)
    return out


def paraphrased(rows, seed=0):
    """Out-of-vocabulary copies of rows: same labels, wording the rules cannot match."""
    out = []
    for r in rows:
        cap = paraphrase_caption(r["caption"], _rng(seed, r["id"], "para"))
        if cap != r["caption"]:
            q = dict(r)
            q.update(id=r["id"] + "_para", caption=cap, canonical_caption=r["caption"])
            out.append(q)
    return out
