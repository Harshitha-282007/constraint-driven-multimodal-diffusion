"""
Constraint-driven multimodal diffusion (SDXL + scribble ControlNet + IP-Adapter): test interface
Team One Sheep Army

Uses the SAME pipeline (src/pipeline.py), governance network (models/governance.pt) and fusion LoRA
(models/fusion_lora.safetensors) as the experiments in notebooks 05-07.

Colab T4:
    !pip uninstall -y torchao -q
    !pip install -q -U diffusers transformers accelerate safetensors peft gradio psutil
    !cd app && python app.py
Inputs : sketch, reference image, caption (+ optional object name)
Output : the image, plus which modality governed each property and the gates the network predicted.
Methods: governed (proposed), governed + fusion LoRA, and the old arms (fixed / scheduled / router / router + guidance).
"""
import os
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import gradio as gr
import torch
import torch.nn as nn

from governance import model as GM
from governance.features import six_cosines
from governance.labels import caption_colors, caption_materials
from pipeline import CENTER, Pipeline

MODELS = os.path.join(HERE, "..", "models")
GOV_PATH = os.path.join(MODELS, "governance.pt")
LORA_PATH = os.path.join(MODELS, "fusion_lora.safetensors")
ROUTER_PATH = next((p for p in (os.path.join(HERE, "router.pt"), os.path.join(MODELS, "router.pt")) if os.path.exists(p)), None)

P = Pipeline(res=640, steps=20, lora=LORA_PATH)
assert os.path.exists(GOV_PATH), "models/governance.pt not found: run notebooks/05_governance_network.ipynb and copy it here"
GOV = GM.load(GOV_PATH)

# ---- old router (baseline arm): 6 conflict cosines -> 4 global strengths, 5-model ensemble --------------------
FACTORS = ["cs", "hold", "ip", "cfg"]
RANGES = dict(cs=(0.5, 1.4), hold=(0.4, 1.0), ip=(0.2, 0.8), cfg=(4.0, 10.0))


class Router(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 4), nn.Sigmoid())

    def forward(self, x):
        return self.net(x)


ENS = []
if ROUTER_PATH:
    _ck = torch.load(ROUTER_PATH, map_location="cpu", weights_only=False)
    for _sd in _ck["states"]:
        _m = Router(); _m.load_state_dict(_sd); ENS.append(_m.eval())
    MU, SD = torch.as_tensor(_ck["mu"]).float(), torch.as_tensor(_ck["sd"]).float()

UI_COLORS = ["red", "blue", "green", "yellow", "orange", "purple", "pink", "black", "white", "brown", "gray", "silver", "gold", "beige"]
UI_MATERIALS = ["wood", "metal", "plastic", "ceramic", "glass", "leather", "fabric", "stone"]
UI_FINISHES = ["glossy", "matte", "shiny metallic", "rough", "polished", "satin"]


def zero_shot(ref_emb, words, tmpl):
    return words[int((P.clip.text([tmpl.format(w) for w in words]) @ ref_emb).argmax())]


def router_params(it):
    c = it["clip"]
    feat = six_cosines(c["cap"], c["obj"], it["attr_emb"], c["sketch"], c["ref"])
    with torch.no_grad():
        o = torch.stack([m((torch.tensor([feat]) - MU) / SD)[0] for m in ENS]).mean(0)
    return {f: RANGES[f][0] + float(o[j]) * (RANGES[f][1] - RANGES[f][0]) for j, f in enumerate(FACTORS)}


METHODS = {"Governed (proposed)": "governed"}
if P.has_lora:
    METHODS["Governed + fusion LoRA"] = "governed_lora"
METHODS.update({"Fixed weights (old baseline)": "fixed", "Scheduled (old baseline)": "scheduled"})
if ENS:
    METHODS.update({"Router (old)": "router", "Router + constraint guidance (old)": "router_guided"})


def explain(caption, gates):
    """The governance map, in words."""
    cc, cm = caption_colors(caption), caption_materials(caption)
    lines = [
        "geometry and layout  <- sketch",
        f"object identity      <- text      (object conflict gate {gates['g_obj']:.2f}: "
        + ("sketch is relaxed so the text decides)" if gates["g_obj"] > 0.5 else "sketch and text agree, sketch kept at full strength)"),
        f"colour               <- {'text' if gates['g_col'] > 0.5 else 'reference'}      (gate {gates['g_col']:.2f}"
        + (f"; caption states: {', '.join(cc)})" if cc else "; caption states no colour)"),
        f"material / texture   <- {'text' if gates['g_mat'] > 0.5 else 'reference'}      (gate {gates['g_mat']:.2f}"
        + (f"; caption states: {', '.join(cm)})" if cm else "; caption states no material)"),
    ]
    return "\n".join(lines)


def synthesize(sketch_im, ref_im, caption, obj, choice, seed):
    if sketch_im is None or ref_im is None or not caption.strip():
        raise gr.Error("Provide a sketch, a reference image and a caption.")
    caption = caption.strip()
    it = P.make_items([dict(key="ui", sketch=sketch_im, ref=ref_im, caption=caption, obj=obj, seed=int(seed))])[0]
    gates = GM.predict_gates(GOV, it["x"])
    ref_emb = it["clip"]["ref"]
    it["attr"] = (f"a {zero_shot(ref_emb, UI_FINISHES, '{} surface')} {zero_shot(ref_emb, UI_MATERIALS, 'made of {}')} "
                  f"object in {zero_shot(ref_emb, UI_COLORS, '{} colored')} colors")
    it["attr_emb"] = P.clip.text([it["attr"]])[0]
    arms = list(METHODS.values()) if choice == "Compare all" else [METHODS[choice]]
    label = {v: k for k, v in METHODS.items()}
    out, info = [], ["GOVERNANCE MAP (predicted by the governance network)", explain(caption, gates), ""]
    for arm in arms:
        P.use_lora(arm == "governed_lora")
        if arm.startswith("governed"):
            img, note = P.run(it, gates=gates), "gates applied"
        elif arm in ("fixed", "scheduled"):
            img, note = P.run(it, dict(CENTER), shaped=(arm == "scheduled")), "centre weights"
        else:
            p = router_params(it)
            img = P.run(it, p, shaped=True, guided=(arm == "router_guided"))
            note = f"sketch {p['cs']:.2f} hold {p['hold']:.2f} ref {p['ip']:.2f} text {p['cfg']:.1f}"
        out.append((img, f"{label[arm]} | {note}"))
        info.append(f"{label[arm]}: {note}")
        torch.cuda.empty_cache()
    return out, it["sketch"], "\n".join(info)


with gr.Blocks(title="Constraint-driven multimodal diffusion") as demo:
    gr.Markdown("## Constraint-driven multimodal diffusion (SDXL)\n"
                "Sketch → geometry and layout · caption → object identity · reference → appearance, "
                "except attributes the caption states. A trained governance network decides, per input, which modality governs colour and material.")
    with gr.Row():
        with gr.Column():
            i_sk = gr.Image(type="pil", label="Sketch"); i_ref = gr.Image(type="pil", label="Reference image")
        with gr.Column():
            i_cap = gr.Textbox(label="Caption", placeholder="a red sports car")
            i_obj = gr.Textbox(label="Object name (optional, e.g. car)")
            i_arm = gr.Dropdown(["Compare all"] + list(METHODS), value="Governed (proposed)", label="Method")
            i_seed = gr.Number(value=0, precision=0, label="Seed")
            btn = gr.Button("Generate", variant="primary")
    o_gal = gr.Gallery(label="Results", columns=2, height=520)
    with gr.Row():
        o_sk = gr.Image(label="Sketch given to the model"); o_txt = gr.Textbox(label="Which modality governed what", lines=10)
    btn.click(synthesize, [i_sk, i_ref, i_cap, i_obj, i_arm, i_seed], [o_gal, o_sk, o_txt])

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(share=True, show_error=True)
