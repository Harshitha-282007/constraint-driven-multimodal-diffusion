import numpy as np
import pytest
import torch

from governance.metrics import rgb_to_lab as np_lab
from rolegate import augment as A
from rolegate import gates as G
from rolegate import losses as LS
from rolegate import tokens as T


class FakeCLIPTokenizer:
    """Mimics CLIP BPE pre-tokenisation: lower-case, words and punctuation as separate pieces ending in </w>;
    long words split into two pieces to exercise multi-piece words."""

    def tokenize(self, text):
        import re
        out = []
        for w in re.findall(r"[a-z]+|[^\sa-z]", text.lower()):
            if len(w) > 6:
                out += [w[:4], w[4:] + "</w>"]
            else:
                out.append(w + "</w>")
        return out


def test_token_groups():
    g = T.token_groups(FakeCLIPTokenizer(), "Pink-handled hammer with wooden handle.", "hammer")
    # slots: 1 pink, 2 -, 3-4 handled (two pieces), 5 hammer, 6 with, 7 wooden, 8-9 handle?, ...
    assert g[0] == T.OTHER and g[1] == T.COLOUR and g[2] == T.OTHER
    assert g[3] == g[4] == T.OTHER and g[5] == T.IDENTITY and g[7] == T.MATERIAL
    assert len(g) == 77 and g[-1] == T.OTHER
    assert T.token_groups(FakeCLIPTokenizer(), "two teddy bears", "teddy_bear")[2:4] == [T.IDENTITY, T.IDENTITY]


def test_lab_roundtrip_and_rotation():
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, (16, 16, 3)).astype(float)
    assert np.abs(A.lab_to_rgb(np_lab(rgb)) - rgb).max() < 1.0
    img = np.zeros((32, 32, 3), np.uint8); img[8:24, 8:24] = (200, 30, 30)
    mask = np.zeros((32, 32), bool); mask[8:24, 8:24] = True
    same = A.rotate_hue(img, mask, 0, feather=0)
    assert np.abs(same.astype(int) - img.astype(int)).max() <= 1
    rot = A.rotate_hue(img, mask, 180, feather=0)
    assert rot[16, 16, 0] < 150 and (rot[0, 0] == img[0, 0]).all()


def test_hue_pair_skips_grey_and_reference_mask():
    grey = np.full((32, 32, 3), 120, np.uint8)
    m = np.zeros((32, 32), bool); m[8:24, 8:24] = True
    assert A.hue_pair(grey, m, grey, 90) is None
    ref = np.full((64, 64, 3), 127, np.uint8); ref[20:44, 20:44] = (30, 60, 200)
    rm = A.reference_mask(ref)
    assert rm[32, 32] and not rm[2, 2]
    photo = np.zeros((32, 32, 3), np.uint8); photo[8:24, 8:24] = (200, 30, 30)
    out = A.hue_pair(photo, m, ref, 120)
    assert out is not None and out[0].shape == photo.shape and out[1].shape == ref.shape


def test_torch_lab_matches_numpy():
    rng = np.random.default_rng(1)
    rgb = rng.integers(0, 256, (8, 8, 3)).astype(float)
    t = torch.tensor(rgb / 255, dtype=torch.float32).permute(2, 0, 1)[None]
    assert np.abs(LS.rgb_to_lab(t)[0].permute(1, 2, 0).numpy() - np_lab(rgb)).max() < 0.05


def test_gate_init_and_cond():
    gs = G.CrossGate(32, style=True, name=G.STYLE_PREFIX + ".transformer_blocks.0.attn2.processor")
    g = gs(torch.randn(2, 5, 32), torch.zeros(2, G.COND_DIM))
    keep = [0, 1, 2, 3, 5]                                                                     # text gates + ref texture
    assert torch.allclose(g[..., keep], torch.ones_like(g[..., keep]), atol=1e-6)
    assert torch.allclose(g[..., 4], torch.full_like(g[..., 4], G.REF_COLOUR_INIT[G.STYLE_PREFIX]), atol=1e-5)
    # a flagged colour conflict starts with the text colour channel nearly closed, the other text gates open
    gf = gs(torch.randn(1, 5, 32), G.cond_vector(torch.tensor([[0., 1, 0]]), torch.tensor([500])))
    assert (gf[..., 2] < 0.15).all() and torch.allclose(gf[..., 0], torch.ones(1, 5), atol=1e-6)
    assert G.ref_init("up_blocks.0.attentions.2.transformer_blocks.3.attn2.processor") == (0.5, G.REF_OFF_INIT)
    assert G.ref_init("down_blocks.2.attentions.0.transformer_blocks.0.attn2.processor") == (G.REF_OFF_INIT, G.REF_OFF_INIT)
    go = G.CrossGate(32, style=False)(torch.randn(1, 5, 32), torch.zeros(1, G.COND_DIM))
    assert torch.allclose(go[..., :4], torch.ones(1, 5, 4)) and torch.allclose(go[..., 4:], torch.full((1, 5, 2), G.REF_OFF_INIT), atol=1e-6)
    sk = G.SketchGate(8)(torch.randn(1, 8, 4, 4), torch.zeros(1, G.COND_DIM))
    assert torch.allclose(sk, torch.ones_like(sk))
    c = G.cond_vector(torch.tensor([[1., 0, 1]]), torch.tensor([500]))
    assert c.shape == (1, G.COND_DIM)


def test_sketch_groups():
    rg = G.RoleGates({}, [4] * 10)
    shapes = [(1, 4, 64, 64)] * 3 + [(1, 4, 32, 32)] * 3 + [(1, 4, 16, 16)] * 3 + [(1, 4, 16, 16)]
    names = rg.sketch_groups(shapes)
    assert names[:6] == ["sketch_contour"] * 6 and names[6:] == ["sketch_placement"] * 4


def test_gated_processor_matches_ip_adapter_at_init():
    """With neutral gates and one reference channel, the gated processor reproduces IP-Adapter exactly."""
    diffusers = pytest.importorskip("diffusers")
    from diffusers.models.attention_processor import Attention, IPAdapterAttnProcessor2_0
    torch.manual_seed(0)
    attn = Attention(query_dim=64, cross_attention_dim=32, heads=2, dim_head=32).eval()
    ip = IPAdapterAttnProcessor2_0(hidden_size=64, cross_attention_dim=32, num_tokens=(4,), scale=0.7).eval()
    h = torch.randn(2, 16, 64)
    text = torch.randn(2, 77, 32)
    ip_tokens = torch.randn(2, 1, 4, 32)
    with torch.no_grad():
        ref = ip(attn, h, encoder_hidden_states=(text, [ip_tokens]))
        gates = G.RoleGates({"up_blocks.0.attentions.1.x.attn2.processor": 64}, [])
        gates.set_context(torch.zeros(2, 3), torch.tensor([10, 10]), torch.zeros(2, 77, dtype=torch.long), 0.7)
        cg = gates.cross[G._key("up_blocks.0.attentions.1.x.attn2.processor")]
        cg.w_c.bias[G.CROSS_GATES.index("ref_colour")] = 0.0                       # neutral gate (v3 starts the colour channel at 1.5)
        proc = G.GatedCrossAttnProcessor("p", cg,
                                         ip.to_k_ip[0], ip.to_v_ip[0], gates.ctx)
        out = proc(attn, h, encoder_hidden_states=(text, [ip_tokens]))
        assert torch.allclose(out, ref, atol=1e-5)
        # knocking out the reference channel removes the IP-Adapter term
        gates.ctx.knock = {"ref_colour"}
        no_ref = proc(attn, h, encoder_hidden_states=(text, [ip_tokens]))
        ip.scale = [0.0]
        assert torch.allclose(no_ref, ip(attn, h, encoder_hidden_states=(text, [ip_tokens])), atol=1e-5)
        # a text group gate of 0 removes exactly that group's share
        gates.ctx.knock = {"text_identity"}
        groups = torch.zeros(2, 77, dtype=torch.long); groups[:, 3] = 1
        gates.ctx.text_groups = groups
        out2 = proc(attn, h, encoder_hidden_states=(text, [ip_tokens]))
        assert not torch.allclose(out2, out, atol=1e-5)


def test_losses_run_and_prior_targets():
    img = torch.rand(2, 3, 32, 32, requires_grad=True)
    tgt = torch.rand(2, 3, 32, 32)
    m = torch.zeros(2, 1, 32, 32); m[..., 8:24, 8:24] = 1
    total = LS.colour_loss(img, tgt, m) + LS.texture_loss(img, tgt, m) + LS.contour_loss(img, m)
    total.backward()
    assert img.grad is not None and torch.isfinite(img.grad).all()
    assert LS.colour_loss(tgt, tgt, m).item() < 1e-6
    e = torch.nn.functional.normalize(torch.randn(2, 8), dim=-1)
    assert LS.identity_loss(e, e, -e).item() == 0.0
    g = torch.ones(2, 16, 6, requires_grad=True)
    p = LS.prior_loss([("a", g)], torch.ones(2, 1, 4, 4), torch.tensor([[0., 1, 0], [0, 0, 0]]))
    p.backward()
    assert p.item() > 0 and g.grad[0, :, G.CROSS_GATES.index("text_colour")].abs().sum() > 0
    assert g.grad[1, :, G.CROSS_GATES.index("text_colour")].abs().sum() == 0
