import random

from governance import labels as L


def row(**kw):
    base = dict(id="0000", category="hammer", caption="Blue-handled hammer with wooden handle.", split="train",
                reference="reference/0000.png", ref_colors="blue and white", ref_material="plastic", ref_finish="glossy")
    base.update(kw)
    return base


def test_caption_parsing():
    assert L.caption_colors("Blue-handled hammer, grey head") == ["blue", "gray"]
    assert L.caption_materials("a hammer with a wooden handle") == ["wood"]
    assert L.caption_materials("made of wood and stone") == ["wood", "stone"]
    assert L.caption_colors("Tiger walks through dry grass.") == []


def test_gates_normal_row_has_material_conflict():
    # caption says wooden, reference is plastic: a conflict even though the row is "normal" in 01b
    assert L.gates_for(row()) == {"g_obj": 0, "g_col": 0, "g_mat": 1}


def test_gates_colour_and_object():
    assert L.gates_for(row(caption="Pink-handled hammer."))["g_col"] == 1
    assert L.gates_for(row(caption="A hammer."))["g_col"] == 0                      # nothing stated -> reference governs
    assert L.gates_for(row(caption="Blue hammer.", ref_colors="grey", ref_material="solid"))["g_col"] == 1
    g = L.gates_for(row(caption="A shiny, silver French horn.", caption_category="trumpet"))
    assert g["g_obj"] == 1 and g["g_col"] == 1
    assert L.gates_for(row(caption="A plastic hammer."))["g_mat"] == 0


def test_governing_map():
    m = L.governing(row(caption="Pink hammer."))
    assert m["colour"] == "text" and m["material"] == "reference" and m["geometry"] == "sketch"
    assert m["conflicts"] == ["g_col"]


def test_caption_edits():
    assert L.caption_colors(L.strip_colors("Blue-handled hammer with wooden handle.")) == []
    assert L.caption_colors(L.ensure_color("A hammer.", ["red"])) == ["red"]
    assert L.ensure_color("A blue hammer.", ["red"]) == "A blue hammer."


def test_makers_produce_the_intended_gate():
    pool = [row(), row(id="0001", category="racket", caption="Tennis racket.", ref_colors="red", ref_material="metal")]
    rng = random.Random(0)
    assert L.gates_for(L.make_object(pool[0], pool, rng))["g_obj"] == 1
    assert L.gates_for(L.make_color(pool[0], pool, rng))["g_col"] == 1
    assert L.gates_for(L.make_material(pool[0], pool, rng))["g_mat"] == 1
    assert L.gates_for(L.make_agree(pool[0], pool, rng)) == {"g_obj": 0, "g_col": 0, "g_mat": 0}
    swapped = L.make_refswap(pool[0], pool, rng)
    assert swapped["ref_colors"] == "red" and L.gates_for(swapped)["g_col"] == 1


def test_expand_keeps_split_and_group():
    base = [row(id=f"{i:04d}", split="train" if i < 4 else "test", category=["hammer", "racket"][i % 2],
                ref_colors=["blue", "red", "green"][i % 3]) for i in range(6)]
    rows = L.expand_rows(base, per_type=2, seed=0)
    split_of = {b["id"]: b["split"] for b in base}
    assert len(rows) > len(base)
    assert len({r["id"] for r in rows}) == len(rows)
    for r in rows:
        assert r["split"] == split_of[r["original_id"]]
        assert set(r["gates"]) == set(L.GATES)
    assert all(any(r["gates"][g] for r in rows) and any(not r["gates"][g] for r in rows) for g in L.GATES)


def test_paraphrase_hides_words_from_the_rules_but_keeps_labels():
    r = row(caption="Pink-handled hammer with wooden handle.")
    r["gates"] = L.gates_for(r)
    p = L.paraphrased([r], seed=0)[0]
    assert L.caption_colors(p["caption"]) == [] and L.caption_materials(p["caption"]) == []
    assert p["gates"] == r["gates"] and p["canonical_caption"] == r["caption"]
