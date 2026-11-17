import numpy as np

from governance import metrics as M


def solid(rgb, n=400):
    return np.tile(np.array(rgb, float), (n, 1))


def test_named_colours_map_to_themselves():
    for name, rgb in M.NAMED_RGB.items():
        assert M.dominant(solid(rgb), top=1) == [M.canon(name)], name


def test_off_anchor_colours():
    assert M.dominant(solid((180, 20, 40)), top=1) == ["red"]
    assert M.dominant(solid((30, 60, 160)), top=1) == ["blue"]


def test_colour_adherence_and_leak():
    px = np.concatenate([solid((200, 30, 30), 300), solid((40, 70, 200), 100)])
    a = M.colour_adherence(px, ["red"], ["blue"])
    assert abs(a["col_share"] - 0.75) < 1e-6 and a["col_hit"] == 1.0 and abs(a["col_leak"] - 0.25) < 1e-6
    b = M.colour_adherence(px, ["blue"], ["red"])
    assert b["col_hit"] == 0.0
    assert np.isnan(M.colour_adherence(px, [])["col_share"])


def test_object_region_fills_a_closed_outline():
    thin = np.zeros((64, 64), bool)
    thin[10, 10:50] = thin[50, 10:51] = thin[10:50, 10] = thin[10:51, 50] = True
    reg = M.object_region(thin, pad=1)
    assert reg[30, 30] and not reg[2, 2]


def test_foreground_pixels_drop_flat_background():
    img = np.full((64, 64, 3), 127.0)
    img[20:40, 20:40] = (200, 30, 30)
    px = M.foreground_pixels(img)
    assert len(px) == 400 and M.dominant(px, top=1) == ["red"]


def test_bootstrap_resamples_groups():
    r = M.paired_bootstrap([0.2, 0.2, 0.1, 0.1, np.nan], ["a", "a", "b", "b", "c"], n=500)
    assert r["n"] == 4 and r["groups"] == 2 and r["lo"] > 0 and abs(r["mean"] - 0.15) < 1e-9
