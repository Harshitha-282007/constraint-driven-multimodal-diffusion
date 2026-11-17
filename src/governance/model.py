"""
Governance network: frozen CLIP features of (caption, object phrase, sketch, reference, grey reference,
colour part) -> three gates in [0,1]:

    g_obj   the caption's object disagrees with the sketch        -> relax the sketch so the text decides identity
    g_col   the caption states a colour the reference lacks       -> do not take colour from the reference
    g_mat   the caption states a material the reference lacks     -> do not take material from the reference

It replaces the router of notebook 04 (6 cosines -> 4 global strengths). Supervision is the rule labels of
labels.py, which exist for every metadata row, so no image has to be generated to train it.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .labels import GATES

N_SLOTS, N_PAIRS = 6, 15


class GovernanceNet(nn.Module):
    def __init__(self, d=768, h=128, p=0.2):
        super().__init__()
        self.cfg = dict(d=d, h=h, p=p)
        self.proj = nn.Linear(d, h)                                   # shared across the six slots
        self.slot = nn.Parameter(torch.zeros(N_SLOTS, h))             # tells the MLP which slot is which
        self.mlp = nn.Sequential(nn.Linear(N_SLOTS * h + N_PAIRS + 1, 256), nn.ReLU(), nn.Dropout(p),
                                 nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, len(GATES)))
        iu = torch.triu_indices(N_SLOTS, N_SLOTS, offset=1)
        self.register_buffer("iu0", iu[0]); self.register_buffer("iu1", iu[1])

    def forward(self, x):
        """x: (B, 6, d). Returns logits (B, 3)."""
        norm = x.norm(dim=-1, keepdim=True)
        u = x / norm.clamp_min(1e-6)
        z = F.relu(self.proj(u) + self.slot)
        cos = (u[:, self.iu0] * u[:, self.iu1]).sum(-1)               # 15 pairwise cosines
        return self.mlp(torch.cat([z.flatten(1), cos, norm[:, 5]], dim=1))     # + size of the colour part


def fit(X, Y, epochs=60, lr=1e-3, wd=1e-2, bs=64, seed=0, device="cpu", Xv=None, Yv=None, log=None):
    torch.manual_seed(seed)
    m = GovernanceNet(d=X.shape[-1]).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    pos = Y.mean(0).clamp(0.02, 0.98)
    crit = nn.BCEWithLogitsLoss(pos_weight=((1 - pos) / pos).to(device))
    X, Y = X.to(device), Y.to(device)
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(X), device=device)
        tot = 0.0
        for i in range(0, len(X), bs):
            idx = perm[i:i + bs]
            loss = crit(m(X[idx]), Y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        if log and (ep % 10 == 9 or ep == epochs - 1):
            msg = f"epoch {ep + 1:3d} | train loss {tot / len(X):.4f}"
            if Xv is not None:
                msg += f" | val acc {((predict(m, Xv, device) > 0.5).float() == Yv).float().mean(0).round(decimals=3).tolist()}"
            log(msg)
    return m.eval()


@torch.no_grad()
def predict(m, X, device=None):
    """Gate probabilities (N, 3) on CPU."""
    device = device or next(m.parameters()).device
    m.eval()
    return torch.cat([torch.sigmoid(m(X[i:i + 256].to(device))).cpu() for i in range(0, len(X), 256)])


def predict_gates(m, x):
    """One input (6, d) -> {'g_obj': p, 'g_col': p, 'g_mat': p}."""
    p = predict(m, x[None])[0].tolist()
    return dict(zip(GATES, p))


def scores(P, Y):
    """Per-gate accuracy, F1 and AUROC. P: probabilities (N,3); Y: 0/1 (N,3)."""
    from sklearn.metrics import f1_score, roc_auc_score
    P, Y = np.asarray(P), np.asarray(Y)
    out = {}
    for j, g in enumerate(GATES):
        y, p = Y[:, j], P[:, j]
        out[g] = dict(acc=float(((p > 0.5) == y).mean()), f1=float(f1_score(y, p > 0.5, zero_division=0)),
                      auroc=float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else float("nan"), pos_rate=float(y.mean()))
    return out


def cosine_baseline(C_train, Y_train, C_test):
    """Logistic regression on the old router's six cosines: the detector we are replacing."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    C_train, Y_train, C_test = np.asarray(C_train), np.asarray(Y_train), np.asarray(C_test)
    P = np.zeros((len(C_test), Y_train.shape[1]))
    for j in range(Y_train.shape[1]):
        if 0 < Y_train[:, j].mean() < 1:
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
            P[:, j] = clf.fit(C_train, Y_train[:, j]).predict_proba(C_test)[:, 1]
        else:
            P[:, j] = Y_train[:, j].mean()
    return P


def group_kfold(X, Y, C6, groups, k=5, device="cpu", **fit_kw):
    """Cross-validation that never splits a base photo across folds. Returns out-of-fold predictions."""
    from sklearn.model_selection import GroupKFold
    Pn, Pc = torch.zeros_like(Y), np.zeros(tuple(Y.shape))
    for tr, va in GroupKFold(n_splits=k).split(np.zeros(len(Y)), groups=np.asarray(groups)):
        Pn[va] = predict(fit(X[tr], Y[tr], device=device, **fit_kw), X[va], device)
        Pc[va] = cosine_baseline(C6[tr], Y[tr], C6[va])
    return Pn, Pc


def save(m, path, extra=None):
    torch.save({"state": m.state_dict(), "cfg": m.cfg, "gates": GATES, **(extra or {})}, path)


def load(path, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = GovernanceNet(**ck["cfg"])
    m.load_state_dict(ck["state"])
    return m.to(device).eval()
