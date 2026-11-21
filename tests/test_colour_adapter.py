"""CPU checks for the colour adapter: palette extraction and the processor wrapper (fake UNet, no SDXL needed)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from coloradapter.adapter import ColourAdapter
from coloradapter.palette import K, palette, palette_distance
from rolegate.augment import rotate_hue


def _ref(rgb=(150, 40, 170), bg=(128, 128, 128), size=96):
    im = np.full((size, size, 3), bg, np.uint8)
    im[28:70, 28:70] = rgb
    return im


# ----------------------------------------------------------------------------- palette
def test_palette_shape_and_shares():
    p = palette(_ref())
    assert p.shape == (K, 4) and p.dtype == np.float32
    assert abs(p[:, 3].sum() - 1.0) < 1e-4 and (np.diff(p[:, 3]) <= 1e-6).all()      # shares sum to 1, sorted descending


def test_palette_finds_object_colour_not_background():
    a = palette(_ref((150, 40, 170)))[0]                 # purple object: a* > 0, b* < 0
    assert a[1] > 0.2 and a[2] < -0.1
    g = palette(_ref((128, 128, 128)))                   # nothing but background: must not crash
    assert g.shape == (K, 4)


def test_palette_follows_hue_rotation():
    ref = _ref((150, 40, 170))
    mask = np.zeros(ref.shape[:2], bool); mask[28:70, 28:70] = True
    d_same = palette_distance(palette(ref), palette(ref))
    d_rot = palette_distance(palette(ref), palette(rotate_hue(ref, mask, 150)))
    assert d_same < 1.0 and d_rot > 20.0


# ----------------------------------------------------------------------------- adapter on a fake UNet
class _Inner:
    def __call__(self, attn, h, *a, **k):
        return h * 2.0


class _Attn(nn.Module):
    def __init__(self, dim=16, heads=2):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, dim)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim), nn.Dropout(0.0)])
        self.processor = _Inner()

    def forward(self, h):
        return self.processor(self, h)


class _UNet(nn.Module):
    NAMES = ["up_blocks.0.attentions.0.transformer_blocks.0.attn2", "mid_block.attentions.0.transformer_blocks.0.attn2",
             "down_blocks.1.attentions.0.transformer_blocks.0.attn2", "up_blocks.0.attentions.0.transformer_blocks.0.attn1"]

    def __init__(self):
        super().__init__()
        self.attns = nn.ModuleList([_Attn() for _ in self.NAMES])

    @property
    def attn_processors(self):
        return {n + ".processor": a.processor for n, a in zip(self.NAMES, self.attns)}

    def set_attn_processor(self, d):
        for n, a in zip(self.NAMES, self.attns):
            a._modules.pop("processor", None)          # diffusers' Attention.set_processor does the same
            a.processor = d[n + ".processor"]

    def get_submodule(self, name):
        return self.attns[self.NAMES.index(name)]


def _setup():
    unet = _UNet()
    ca = ColourAdapter.for_unet(unet, tokens=3, width=8)
    return unet, ca


def test_only_up_and_mid_cross_attention_layers_are_wrapped():
    _, ca = _setup()
    assert sorted(ca.layer_dims) == sorted(n + ".processor" for n in _UNet.NAMES[:2])


def test_zero_init_leaves_output_unchanged_then_palette_changes_it():
    unet, ca = _setup()
    h = torch.randn(2, 5, 16)
    base = unet.attns[0](h)
    ca.install(unet)
    ca.set_palette(torch.rand(2, K, 4))
    assert torch.allclose(unet.attns[0](h), base)                                    # alpha = 0
    with torch.no_grad():
        for p in ca._alpha.values():
            p.fill_(1.0)
    out = unet.attns[0](h)
    assert not torch.allclose(out, base)
    ca.set_palette(torch.rand(2, K, 4) + 0.5)
    assert not torch.allclose(unet.attns[0](h), out)                                  # depends on the palette


def test_unconditional_rows_add_nothing():
    unet, ca = _setup()
    with torch.no_grad():
        for p in ca._alpha.values():
            p.fill_(1.0)
    h = torch.randn(2, 5, 16)
    base = unet.attns[0](h)
    ca.install(unet)
    pal = torch.stack([torch.zeros(K, 4), torch.rand(K, 4)])                          # row 0: classifier-free half
    ca.set_palette(pal)
    out = unet.attns[0](h)
    assert torch.allclose(out[0], base[0], atol=1e-6) and not torch.allclose(out[1], base[1])


def test_install_remove_restores_processors_and_params_train():
    unet, ca = _setup()
    before = dict(unet.attn_processors)
    ca.install(unet)
    assert all(type(unet.attn_processors[n]).__name__ == "ColourProcessor" for n in ca.layer_dims)
    ca.remove(unet)
    assert all(unet.attn_processors[n] is before[n] for n in before)
    ca.install(unet); ca.set_palette(torch.rand(1, K, 4))
    with torch.no_grad():
        for p in ca._alpha.values():
            p.fill_(0.5)
    unet.attns[0](torch.randn(1, 5, 16)).sum().backward()
    assert ca._alpha[ca._safe[_UNet.NAMES[0] + ".processor"]].grad is not None and ca.mlp[0].weight.grad is not None


def test_save_load_roundtrip(tmp_path):
    _, ca = _setup()
    with torch.no_grad():
        for p in ca._alpha.values():
            p.fill_(0.3)
    ca.save(str(tmp_path / "a.pt"), dict(step=7))
    b = ColourAdapter.load(str(tmp_path / "a.pt"))
    assert all(torch.allclose(a, c) for a, c in zip(ca.state_dict().values(), b.state_dict().values()))
