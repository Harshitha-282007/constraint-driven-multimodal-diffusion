"""
M1: conditional routing of the text colour / material channel, trained by distilling the neutral-caption rule.

What is trained (and nothing else). Using the existing gate machinery (rolegate/gates.py), for every cross-attention layer l
and for the pooled-embedding gate, four scalars decide how much of the caption's colour / material words reaches the image:

    g_colour,l   = 2 * sigmoid( b_colour,l   + w_colour,l   * s )         s = presence of a reference (0 or 1), derived from the input
    g_material,l = 2 * sigmoid( b_material,l + w_material,l * s )

    out_l = o(full) + (g_colour - 1) * (o(full) - o(colour-neutral)) + (g_material - 1) * (o(full) - o(material-neutral))

g = 1 reproduces the caption as written, g = 0 the neutral-caption rule. At initialisation b = w = 0, so g = 1: the model is exactly
the baseline. The ControlNet caption follows the pooled gate (alpha = clamp(1 - g_pooled, 0, 1), no gradient through it).

Trainable scalars: 4 per cross-attention layer (b_col, w_col, b_mat, w_mat) + 4 for the pooled gate. Every other parameter of the
role gates, SDXL, the ControlNet and the IP-Adapter is frozen (requires_grad False) and, for the entries that share a tensor with a
trainable one, masked by a gradient hook so they cannot move.
"""
import torch

from rolegate.gates import CROSS_GATES

COL, MAT = CROSS_GATES.index("text_colour"), CROSS_GATES.index("text_material")     # rows of CrossGate.w_c
FLAG_COL, FLAG_MAT = 1, 2                                                          # columns of the conditioning vector: colour / material "flag"
# In M1 these two columns carry the presence of a reference, not a conflict flag.


class M1Router:
    def __init__(self, role):
        self.role = role
        for p in role.parameters():
            p.requires_grad_(False)
        self.entries = {}                                  # group -> list of (tensor, index) of trainable scalars
        union = {}                                         # id(tensor) -> (tensor, mask of all trainable entries in it)

        def reg(group, tensor, idx):
            self.entries.setdefault(group, []).append((tensor, idx))
            t, m = union.setdefault(id(tensor), (tensor, torch.zeros_like(tensor)))
            m[idx] = 1.0
            with torch.no_grad():
                tensor[idx] = 0.0                          # zero init: g = 2*sigmoid(0) = 1, no dependence on presence

        for gate in role.cross.values():
            w, b = gate.w_c.weight, gate.w_c.bias
            reg("cross_colour_w", w, (COL, FLAG_COL)); reg("cross_colour_b", b, COL)
            reg("cross_material_w", w, (MAT, FLAG_MAT)); reg("cross_material_b", b, MAT)
        pw, pb = role.pooled.w_c.weight, role.pooled.w_c.bias
        reg("pooled_colour_w", pw, (0, FLAG_COL)); reg("pooled_colour_b", pb, 0)
        reg("pooled_material_w", pw, (1, FLAG_MAT)); reg("pooled_material_b", pb, 1)
        for t, m in union.values():
            t.requires_grad_(True)
            t.register_hook(lambda g, m=m: g * m.to(g.device, g.dtype))
        self.params = [t for t, _ in union.values()]
        self._frozen = [(t.detach().clone(), m.clone()) for t, m in union.values()]

    # ---------------------------------------------------------------- bookkeeping
    def n_trained(self):
        return sum(len(v) for v in self.entries.values())

    def inventory(self):
        return {g: len(v) for g, v in self.entries.items()}

    def values(self, group):
        return torch.stack([t[idx].detach().float() for t, idx in self.entries[group]])

    def untouched_ok(self, atol=0.0):
        """True if every entry that is NOT trainable still has its initial value."""
        for (t, m), (t0, _) in zip(self._union_tensors(), self._frozen):
            keep = m == 0
            if not torch.allclose(t.detach()[keep], t0.to(t.device)[keep], atol=atol):
                return False
        return True

    def _union_tensors(self):
        masks = {}
        for tensor, idx in [e for v in self.entries.values() for e in v]:
            m = masks.setdefault(id(tensor), (tensor, torch.zeros_like(tensor)))[1]
            m[idx] = 1.0
        return [masks[id(t)] for t in self.params]

    def grad_report(self, grads):
        """grads: {tensor: grad tensor}. -> {group: (l2 norm, mean, sum of |.|)} over the trainable entries only."""
        out = {}
        for group, ents in self.entries.items():
            vals = torch.stack([grads[t][idx].float() if grads.get(t) is not None else torch.zeros((), device=t.device) for t, idx in ents])
            out[group] = dict(norm=float(vals.norm()), mean=float(vals.mean()), abs=float(vals.abs().sum()))
        return out

    def flat(self, grads):
        """All trainable-entry gradients as one vector (for cosines)."""
        parts = []
        for ents in self.entries.values():
            for t, idx in ents:
                g = grads.get(t)
                parts.append((g[idx].float() if g is not None else torch.zeros((), device=t.device)).reshape(1))
        return torch.cat(parts)

    def shift(self, group, delta):
        """Add `delta` to every entry of a group (used for the finite perturbation check); returns an undo function."""
        with torch.no_grad():
            for t, idx in self.entries[group]:
                t[idx] += delta

        def undo():
            with torch.no_grad():
                for t, idx in self.entries[group]:
                    t[idx] -= delta
        return undo
