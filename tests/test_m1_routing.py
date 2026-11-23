"""CPU checks for M1: what is trained, that init is the baseline, that nothing else can move."""
import pytest

torch = pytest.importorskip("torch")
from m1.routing import COL, FLAG_COL, FLAG_MAT, MAT, M1Router
from rolegate.gates import RoleGates, cond_vector

NAMES = {"up_blocks.0.attentions.1.transformer_blocks.0.attn2.processor": 32, "mid_block.attentions.0.transformer_blocks.0.attn2.processor": 16}


def _role():
    return RoleGates(NAMES, [8, 8])


def _gates_out(role, presence):
    """Sum of every cross-attention gate output (each layer gets features of its own width) plus the pooled gate."""
    c = _cond(presence)
    return sum(gate(torch.randn(1, 5, gate.w_h.in_features), c).sum() for gate in role.cross.values()) + role.pooled(c).sum()


def _cond(presence):
    return cond_vector(torch.tensor([[0.0, presence, presence]]), torch.tensor([500]))


def test_inventory_counts_exactly_the_intended_scalars():
    r = M1Router(_role())
    assert r.n_trained() == 4 * len(NAMES) + 4
    assert r.inventory() == {"cross_colour_w": 2, "cross_colour_b": 2, "cross_material_w": 2, "cross_material_b": 2,
                             "pooled_colour_w": 1, "pooled_colour_b": 1, "pooled_material_w": 1, "pooled_material_b": 1}


def test_init_is_the_baseline_for_both_presence_values():
    role = _role(); M1Router(role)
    h = torch.randn(1, 5, 32)
    for presence in (0.0, 1.0):
        g = role.cross[list(role.cross)[0]](h, _cond(presence))
        assert torch.allclose(g[..., COL], torch.ones_like(g[..., COL]), atol=1e-6)
        assert torch.allclose(g[..., MAT], torch.ones_like(g[..., MAT]), atol=1e-6)
        pg = role.pooled(_cond(presence))
        assert torch.allclose(pg, torch.ones_like(pg), atol=1e-6)


def test_only_the_intended_entries_receive_gradient():
    role = _role(); r = M1Router(role)
    _gates_out(role, 1.0).backward()
    for gate in role.cross.values():
        gw = gate.w_c.weight.grad
        mask = torch.zeros_like(gw); mask[COL, FLAG_COL] = mask[MAT, FLAG_MAT] = 1
        assert (gw * (1 - mask)).abs().sum() == 0 and gw[COL, FLAG_COL] != 0 and gw[MAT, FLAG_MAT] != 0
        assert gate.w_h.weight.grad is None                      # the token-dependent gate weights are frozen
        gb = gate.w_c.bias.grad
        assert gb[COL] != 0 and gb[MAT] != 0 and gb[[0, 1, 4, 5]].abs().sum() == 0


def test_optimizer_cannot_move_anything_else():
    role = _role(); r = M1Router(role)
    opt = torch.optim.AdamW(r.params, lr=0.1, weight_decay=0.0)
    for _ in range(5):
        opt.zero_grad()
        _gates_out(role, 1.0).backward()
        opt.step()
    assert r.untouched_ok()
    assert (r.values("cross_colour_w").abs() > 0).all() and (r.values("pooled_colour_b").abs() > 0).all()


def test_shift_and_undo_and_grad_helpers():
    role = _role(); r = M1Router(role)
    undo = r.shift("cross_colour_b", 0.3)
    assert torch.allclose(r.values("cross_colour_b"), torch.full((2,), 0.3))
    undo()
    assert r.values("cross_colour_b").abs().sum() == 0
    _gates_out(role, 1.0).backward()
    grads = {t: t.grad for t in r.params}
    rep = r.grad_report(grads)
    assert rep["cross_colour_w"]["norm"] > 0 and rep["pooled_colour_w"]["norm"] > 0
    assert r.flat(grads).shape == (r.n_trained(),)
