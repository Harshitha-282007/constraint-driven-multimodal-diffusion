"""
The learned router of notebook 04 (6 conflict cosines -> sketch strength, hold time, reference strength, text guidance),
loaded from models/router.pt. Kept as the instructor's "learned routing" arm.
"""
import torch
import torch.nn as nn

from governance.features import six_cosines

FACTORS = ["cs", "hold", "ip", "cfg"]
RANGES = dict(cs=(0.5, 1.4), hold=(0.4, 1.0), ip=(0.2, 0.8), cfg=(4.0, 10.0))


class Router(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 4), nn.Sigmoid())

    def forward(self, x):
        return self.net(x)


class RouterEnsemble:
    def __init__(self, path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.models = []
        for sd in ck["states"]:
            m = Router(); m.load_state_dict(sd); self.models.append(m.eval())
        self.mu, self.sd = torch.as_tensor(ck["mu"]).float(), torch.as_tensor(ck["sd"]).float()

    @torch.no_grad()
    def params(self, it):
        """Router strengths for an item from Pipeline.make_items (needs `attr_emb`: pass `attr` in the spec)."""
        c = it["clip"]
        feat = torch.tensor([six_cosines(c["cap"], c["obj"], it["attr_emb"], c["sketch"], c["ref"])])
        o = torch.stack([m((feat - self.mu) / self.sd)[0] for m in self.models]).mean(0)
        return {f: RANGES[f][0] + float(o[j]) * (RANGES[f][1] - RANGES[f][0]) for j, f in enumerate(FACTORS)}
