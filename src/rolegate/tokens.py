"""
Text attribute channels: which of the 77 caption token slots carry the object's identity, a colour, a material.

SDXL's two text encoders use the same CLIP BPE vocabulary, so token positions match between them and one map
serves the concatenated embedding. Position 0 is the start token; words are runs of BPE pieces ending in `</w>`.
"""
import re

from governance.labels import COLORS, MATERIALS

OTHER, IDENTITY, COLOUR, MATERIAL = 0, 1, 2, 3
GROUP_NAMES = ["other", "identity", "colour", "material"]
_COLOUR_WORDS = set(COLORS) | {"grey"}
_MATERIAL_WORDS = {a for a, _ in MATERIALS} | {s for _, s in MATERIALS}


def _word_group(word, identity_words):
    w = word.lower()
    if w in _COLOUR_WORDS:
        return COLOUR
    if w in _MATERIAL_WORDS:
        return MATERIAL
    if w in identity_words or (w.endswith("s") and w[:-1] in identity_words):
        return IDENTITY
    return OTHER


def identity_words(category):
    """'teddy_bear' -> {'teddy', 'bear'}."""
    return {p for p in re.split(r"[_\s-]+", str(category).lower()) if p}


def token_groups(tokenizer, caption, category, max_len=77):
    """List of `max_len` group ids, one per token slot of `tokenizer(caption)` (padding included, as OTHER)."""
    groups = [OTHER] * max_len
    pieces = tokenizer.tokenize(caption)
    ids = identity_words(category)
    pos, word, start = 1, "", 1                       # slot 0 = start-of-text
    for p in pieces:
        if pos >= max_len - 1:
            break
        word += p.replace("</w>", "")
        if p.endswith("</w>"):
            g = _word_group(word, ids)
            for k in range(start, pos + 1):
                groups[k] = g
            word, start = "", pos + 1
        pos += 1
    return groups


NEUTRAL = {"colour": "colored", "material": "solid"}


def neutral_caption(tokenizer, caption, which):
    """Caption with its colour (which="colour") or material (which="material") words replaced by a neutral word,
    repeated so every word keeps its number of BPE tokens: all token positions stay aligned with `caption`.
    Returns None when the caption has no such word or the alignment cannot be kept."""
    words = _COLOUR_WORDS if which == "colour" else _MATERIAL_WORDS
    neutral = NEUTRAL[which]
    n_neu = len(tokenizer.tokenize(neutral))
    hit = False

    def sub(m):
        nonlocal hit
        w = m.group(0)
        if w.lower() not in words:
            return w
        k = len(tokenizer.tokenize(w))
        if k % n_neu:
            return w
        hit = True
        return " ".join([neutral] * (k // n_neu))

    out = re.sub(r"[A-Za-z]+", sub, caption)
    if not hit or len(tokenizer.tokenize(out)) != len(tokenizer.tokenize(caption)):
        return None
    return out
