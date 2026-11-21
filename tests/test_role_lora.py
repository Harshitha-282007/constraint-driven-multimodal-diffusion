"""CPU checks for the Role LoRA (fake UNet, no SDXL) and the evaluation helpers."""
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from eval_roles import kind_of
from rolelora.lora import LoRALinear, RoleLoRA


class _IP(nn.Module):
    def __init__(self, dim_in=12, dim_out=16):
        super().__init__()
        self.to_k_ip = nn.ModuleList([nn.Linear(dim_in, dim_out, bias=False)])
        self.to_v_ip = nn.ModuleList([nn.Linear(dim_in, dim_out, bias=False)])

    def forward(self, x):
        return self.to_k_ip[0](x) + self.to_v_ip[0](x)


class _Attn(nn.Module):
    def __init__(self, dim=16, ctx=20):
        super().__init__()
        self.to_k, self.to_v = nn.Linear(ctx, dim, bias=False), nn.Linear(ctx, dim, bias=False)
        self.processor = _IP()

    def forward(self, text, ref):
        return self.to_k(text) + self.to_v(text), self.processor(ref)


class _UNet(nn.Module):
    NAMES = ["up_blocks.0.attentions.0.transformer_blocks.0.attn2", "mid_block.attentions.0.transformer_blocks.0.attn2",
             "down_blocks.1.attentions.0.transformer_blocks.0.attn2", "up_blocks.0.attentions.0.transformer_blocks.0.attn1"]

    def __init__(self):
        super().__init__()
        self.attns = nn.ModuleList([_Attn() for _ in self.NAMES])

    @property
    def attn_processors(self):
        return {n + ".processor": a.processor for n, a in zip(self.NAMES, self.attns)}

    def get_submodule(self, name):
        return self.attns[self.NAMES.index(name)]


def _inputs():
    return torch.randn(2, 5, 20), torch.randn(2, 4, 12)


def test_attach_wraps_only_up_and_mid_cross_attention():
    u = _UNet()
    lora = RoleLoRA.attach(u, rank=4, alpha=4)
    assert len(lora.names) == 2 * 4                                   # 2 layers x (to_k, to_v, to_k_ip, to_v_ip)
    assert isinstance(u.attns[0].to_k, LoRALinear) and isinstance(u.attns[0].processor.to_k_ip[0], LoRALinear)
    assert not isinstance(u.attns[2].to_k, LoRALinear)                # down block untouched
    assert not isinstance(u.attns[3].to_k, LoRALinear)                # self-attention untouched


def test_groups_select_the_interfaces():
    u = _UNet()
    ref_only = RoleLoRA.attach(u, groups=("ref",), rank=4)
    assert len(ref_only.names) == 2 * 2 and not isinstance(u.attns[0].to_k, LoRALinear)
    ref_only.detach()
    text_only = RoleLoRA.attach(u, groups=("text",), rank=4)
    assert len(text_only.names) == 2 * 2 and not isinstance(u.attns[0].processor.to_k_ip[0], LoRALinear)


def test_zero_init_is_identity_then_trained_changes_output_and_scale_zero_restores():
    u = _UNet(); t, r = _inputs()
    base = u.attns[0](t, r)
    lora = RoleLoRA.attach(u, rank=4)
    out = u.attns[0](t, r)
    assert all(torch.allclose(a, b) for a, b in zip(out, base))                           # B = 0
    with torch.no_grad():
        for p in lora.params:
            p.B.normal_()
    changed = u.attns[0](t, r)
    assert not torch.allclose(changed[0], base[0]) and not torch.allclose(changed[1], base[1])
    lora.set_scale(0.0)
    assert all(torch.allclose(a, b) for a, b in zip(u.attns[0](t, r), base))             # exact baseline
    lora.set_scale(1.0)
    with lora.disabled():
        assert torch.allclose(u.attns[0](t, r)[0], base[0])
    assert torch.allclose(u.attns[0](t, r)[0], changed[0])                               # context restored


def test_detach_restores_original_modules_and_gradients_reach_A_and_B():
    u = _UNet()
    orig = (u.attns[0].to_k, u.attns[0].processor.to_k_ip[0])
    lora = RoleLoRA.attach(u, rank=4)
    with torch.no_grad():
        lora.params[0].B.normal_()
    t, r = _inputs()
    sum(o.sum() for o in u.attns[0](t, r)).backward()
    assert lora.params[0].A.grad is not None and lora.params[0].B.grad is not None
    lora.detach()
    assert u.attns[0].to_k is orig[0] and u.attns[0].processor.to_k_ip[0] is orig[1]


def test_save_load_roundtrip(tmp_path):
    u = _UNet()
    lora = RoleLoRA.attach(u, rank=4)
    with torch.no_grad():
        for p in lora.params:
            p.B.normal_()
    t, r = _inputs()
    want = u.attns[0](t, r)
    lora.save(str(tmp_path / "l.pt"), dict(step=3))
    u2 = _UNet(); u2.load_state_dict(u.state_dict(), strict=False)
    u2.attns.load_state_dict({k.replace("base.", ""): v for k, v in u.attns.state_dict().items() if "lora" not in k}, strict=False)
    lora2 = RoleLoRA.load(str(tmp_path / "l.pt"), u2)
    got = u2.attns[0](t, r)
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(want, got))


def test_kind_of():
    assert kind_of(dict(id="0027", original_id="0027")) == "base"
    assert kind_of(dict(id="0027_col0", original_id="0027")) == "col"
    assert kind_of(dict(id="0027_obj2", original_id="0027")) == "obj"
