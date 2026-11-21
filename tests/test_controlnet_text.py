"""CPU check for the ControlNet caption leak fix (flagged colour/material words are removed in proportion to the flag)."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
from pipeline import Pipeline


def _inputs():
    pr = torch.randn(2, 77, 8)
    tn = {"colour": torch.randn(2, 77, 8), "material": None}
    return pr, tn


def test_flag_zero_keeps_full_caption():
    pr, tn = _inputs()
    assert torch.allclose(Pipeline.controlnet_text(pr, tn, (0, 0, 0)), pr)


def test_flag_one_gives_neutral_colour():
    pr, tn = _inputs()
    assert torch.allclose(Pipeline.controlnet_text(pr, tn, (0, 1, 0)), tn["colour"], atol=1e-6)


def test_missing_neutral_is_ignored():
    pr, tn = _inputs()
    assert torch.allclose(Pipeline.controlnet_text(pr, tn, (0, 0, 1)), pr)   # material has no neutral caption


def test_half_flag_is_midpoint():
    pr, tn = _inputs()
    assert torch.allclose(Pipeline.controlnet_text(pr, tn, (0, 0.5, 0)), 0.5 * (pr + tn["colour"]), atol=1e-6)
