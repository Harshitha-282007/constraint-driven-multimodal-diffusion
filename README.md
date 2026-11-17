# Constraint-Driven Multimodal Diffusion (Text + Sketch + Reference Image)

Team **One Sheep Army** · IE 643 · IIT Bombay

A diffusion pipeline that takes a **text prompt, a sketch and a reference image** and decides which modality controls what:

| Modality | Controls |
|---|---|
| Sketch | geometry / layout |
| Text | semantics (what the object is) |
| Reference image | appearance (colour, material, style) |

When the three disagree (a conflict), a small learned **attribute router** sets how strongly each modality acts during denoising.

**Current system:** frozen SDXL + scribble ControlNet (sketch) + IP-Adapter (reference) + SDXL's own text encoders, driven by a trained **governance network** that predicts, per input, which modality governs object identity, colour and material, plus a **fusion LoRA** on the UNet trained so the generator obeys those decisions. The earlier router (6 CLIP cosines → 4 global strengths) is kept as a baseline. An earlier FLUX.1-dev / UniCombine route was dropped (see [What we tried](#what-we-tried-and-moved-on-from)).

## Governed pipeline (current method)

> Status: the code below is written and its pure-Python parts are unit-tested (`pytest tests/`). The GPU parts (notebooks 05–07) have **not been run yet**, so there are no results for this method in the repo.

Rules: geometry and layout come from the sketch; object identity from the text; a colour or material the caption states stays with the text, otherwise the reference supplies it.

| Step | Where | What it does |
|---|---|---|
| Gate labels | `src/governance/labels.py` | Rule labels for three gates (`g_obj`, `g_col`, `g_mat`) from metadata; generates thousands of conflict rows without generating any image. |
| Governance network | `src/governance/model.py`, `features.py`, notebook `05` | Frozen CLIP ViT-L/14 embeddings of caption, object phrase, sketch, reference, grey reference and the colour part (reference − grey reference) → three gates. Compared with the old six-cosine detector and on captions reworded with synonyms the rules do not know. Output: `models/governance.pt`. |
| Enforcement | `src/pipeline.py` | The reference embedding is split into a colour part and a texture part and only the governed part is injected, into the style block only; the sketch is relaxed only when the object gate is on; the caption is left as is. |
| Fusion LoRA | `src/train_fusion_lora.py`, notebook `06` | LoRA on the SDXL UNet, trained on (target, sketch, reference, caption) with role dropout: the reference is randomly greyscaled, or the caption's colour words removed, so the model learns both "colour from the text" and "colour from the reference" against the real photo. Output: `models/fusion_lora.safetensors`. |
| Evaluation | `src/governance/evalkit.py`, `metrics.py`, notebook `07` | Text–image alignment, sketch consistency, reference similarity, perceptual quality, layout consistency, plus per-attribute adherence; paired bootstrap over base photos. |
| Interface | `app/app.py` | Shows the predicted gates and which modality governed what; old arms remain selectable. |

Where the mechanisms come from: colour/texture split of the IP-Adapter embedding from [SADis](https://arxiv.org/abs/2503.14275) (used without its SVD rescaling); style-block-only injection from [InstantStyle](https://arxiv.org/abs/2404.02733); relaxing ControlNet under object conflict from [SmartControl](https://arxiv.org/abs/2404.06451) (here a gate-driven scalar, not their learned map); training with all conditions present from [UniCombine](https://arxiv.org/abs/2503.09277)'s denoising LoRA. The governance network, the role dropout and the adherence evaluation are ours. Background and literature notes are in `STAGE1_CURRENT_STATE.md` … `NOVELTY_ASSESSMENT_PLAN.md`.

## Pipeline at a glance

```
Sketchy database
   │  sketchy_make_csv ........ pairs CSV + Qwen3-VL-2B captions (200 photos)
   │  sketchy_reference_v2 .... generated reference images (segmented, filtered)
   │  01_freeze_pilot_dataset . pilot_v1: aligned (target, sketch, reference, caption)
   │  01b_add_conflicts ....... labeled conflict rows
   ▼
02 SDXL baseline  →  03 gated fusion  →  04 attribute router  →  05 evaluate all runs
```

## Repo layout

```
.
├── README.md
├── data/
│   ├── sketchy_pairs.csv              # all 75,481 sketch-photo pairs (125 categories)
│   ├── sketchy_captions_sample.csv    # 200 captioned photos
│   ├── sketchy_with_refs.csv          # + v1 reference paths (superseded)
│   └── sketchy_with_refs_v2.csv       # + v2 reference paths (current)
├── models/
│   └── router.pt                      # trained router MLP weights
├── src/
│   ├── eval_metrics.py                # evaluation metrics
│   └── sketchy_loader.py              # dataset classes for the FLUX / UniCombine trainers
└── notebooks/
    ├── data_prep/                     # dataset generation
    │   ├── sketchy_make_csv.ipynb
    │   ├── sketchy_reference_ipadapter.ipynb   # reference v1 (superseded)
    │   ├── sketchy_reference_v2.ipynb          # reference v2 (current)
    │   └── 01b_add_conflicts.ipynb
    ├── 01_freeze_pilot_dataset.ipynb
    ├── 02_sdxl_baseline_model.ipynb
    ├── 03_gated_fusion_model.ipynb
    ├── 04_attribute_router_model.ipynb
    ├── final_output_eval.ipynb        # notebook "05": scores every run side by side
    └── unicombine_flux_attempt/       # abandoned FLUX route
```

## Interface

`app/app.py` is a Gradio app. Upload a sketch and a reference image, write a caption, and it shows the governed
output together with the gates the governance network predicted and which modality governed each property.
The old arms (fixed / scheduled / router / router_guided) remain selectable for comparison.

    pip install -r app/requirements.txt
    cd app && python app.py

It needs `models/governance.pt` (from notebook 05). `models/fusion_lora.safetensors` (from notebook 06) is optional
and adds a "Governed + fusion LoRA" method when present.

Needs a GPU (SDXL + ControlNet + IP-Adapter use roughly 10 GB). On Colab, run it in a T4 notebook and use
`share=True` to get a public link.

## Dataset generation

Source data is the [Sketchy database](https://sketchy.eye.gatech.edu/) (human-drawn sketches paired with photos): 125 categories × 100 photos = 12,500 photos, 75,481 sketch-photo pairs.

| Notebook | What it does |
|---|---|
| `data_prep/sketchy_make_csv.ipynb` | Builds `sketchy_pairs.csv` (category, image_path, sketch_path), then captions 200 randomly sampled photos with **Qwen3-VL-2B-Instruct** (the category is passed in the prompt as a hint) → `sketchy_captions_sample.csv`. |
| `data_prep/sketchy_reference_ipadapter.ipynb` | Reference images v1: SDXL + style-only IP-Adapter fed with the target photo. Outputs looked wrong; kept for the record. → `sketchy_with_refs.csv` |
| `data_prep/sketchy_reference_v2.ipynb` | Reference images v2: object segmented onto neutral gray, neutral objects (vase / mug / ball / bottle / box), colour / material / finish extracted by a VLM into the prompt, 3-4 candidates per photo filtered on low DINO similarity plus an appearance check. 182 of 200 photos passed; 26 segmentation failures. → `sketchy_with_refs_v2.csv` |
| `01_freeze_pilot_dataset.ipynb` | Freezes the pilot as `pilot_v1`: every sample is an aligned (target photo, sketch, reference, caption) quadruple at one resolution, split **by photo** with a fixed seed, role tags attached (`sketch→geometry, text→semantics, reference→appearance`). Runs on CPU. Output: `pilot_v1.zip` + manifest. |
| `data_prep/01b_add_conflicts.ipynb` | Adds **labeled conflicts** without touching `pilot_v1.zip`. Writes `metadata_conflicts.jsonl` and `metadata_with_conflicts.jsonl`. Conflict kinds: `sem` (caption vs sketch object), `color` and `material` (caption vs reference), `app` (swapped reference). Each conflict row keeps its base photo's split. |

## Model notebooks

All are Colab notebooks written for a free T4 with Google Drive mounted. 03 and 04 are standalone (they reload everything).

| Notebook | What it does |
|---|---|
| `02_sdxl_baseline_model.ipynb` | **Baseline, no training.** Four modes on the `pilot_v1` test split with everything frozen: `text`, `text_sketch`, `text_ref`, `text_sketch_ref`. |
| `03_gated_fusion_model.ipynb` | **Gating MLP.** CLIP similarities between modalities per attribute (object, appearance) → 6 conflict features. Stage A generates a grid of (ControlNet scale, IP-Adapter scale) per train item and scores it to get soft-best targets; the MLP predicts ControlNet and IP-Adapter amplitudes. Arms: `fixed`, `scheduled` (geometry first, appearance later), `gate`, `gate_guided` (latent guidance with a sketch-edge geometry loss and an attribute-level appearance loss). |
| `04_attribute_router_model.ipynb` | **Previous method, now the baseline.** Router MLP with 4 attribute-tied outputs: sketch strength + sketch hold time (geometry), reference strength (appearance), text guidance (semantics). Custom denoising loop with per-step modality weights and constraint guidance. Trained on a one-factor-at-a-time grid (9 images per train item), checked with 5-fold CV, evaluated on held-out items per conflict kind with paired bootstrap confidence intervals. Arms: `fixed`, `scheduled`, `router`, `router_guided`. Ends with an optional Gradio interface (upload sketch + reference, write a caption, see all four arms with the router's chosen weights). Run `01b` first. |
| `final_output_eval.ipynb` | Re-scores every generated image under `sketchy_dataset_one_sketch/sdxl_runs` (`sdxl_baseline_v1`, `gated_v1`, `router_v2`) with the same held-out metrics so runs can be compared. Note that runs differ in resolution and steps, so compare arms within a run first. |

Supporting files:
- `src/eval_metrics.py`: per-image CLIP-T, CLIP-I and DINO-I (vs reference), DreamSim and LPIPS (vs target), layout IoU (CLIPSeg masks), sketch recall (share of sketch strokes landing on an output edge); set-level FID (only meaningful on a few hundred images). AP, the GPT-4o attribute judge and conflict-set scoring are not implemented yet.
- `models/router.pt`: weights of the trained router. The run called v3 in notebook 04 was saved in a folder named `router_v2` on Drive.

## What we tried and moved on from

| Attempt | Where | Why it was dropped |
|---|---|---|
| FLUX.1-dev + UniCombine (cache, condition LoRAs, union LoRA, generate, evaluate) | `notebooks/unicombine_flux_attempt/`, `src/sketchy_loader.py` | The notebooks were written, but training ran out of memory on the free T4 at the first step. Replaced by the SDXL baseline with frozen pretrained adapters. The pilot dataset did not change. |
| Reference images v1 (SDXL + IP-Adapter) | `data_prep/sketchy_reference_ipadapter.ipynb` | Generated references came out wrong; v2 segments the object and filters candidates. |
| Gated fusion (`03`) | `notebooks/03_gated_fusion_model.ipynb` | Kept as the step before the router, which replaces it with attribute-tied outputs and a hold-time control. |

## Not in this repo (too large, kept on Google Drive)

| File | Size | Location (`MyDrive/`) |
|---|---|---|
| `sketchy.zip`, the raw Sketchy database | ~1.1 GB | My Drive root |
| `pilot_v1.zip`, the frozen pilot | ~103 MB | `sketchy_dataset_one_sketch/` |
| `reference_images/`, `ref_v2_backup.zip` | folder / zip | `sketchy_dataset_one_sketch/` |
| `sdxl_runs/`, the generated outputs, scores and manifests for every run | folders | `sketchy_dataset_one_sketch/` |

## Running it

1. Open a notebook in Colab with a T4 GPU (01 and 01b run on CPU) and mount Drive. Notebooks use `ROOT = /content/drive/MyDrive/sketchy_dataset_one_sketch`.
2. Run in this order: `sketchy_make_csv` → `sketchy_reference_v2` → `01_freeze_pilot_dataset` → `01b_add_conflicts` → `02` → `03` → `04` → `final_output_eval`.
3. Notebooks 02 to 04 download SDXL, scribble ControlNet and IP-Adapter weights from Hugging Face on first run.

For the current method, after the data notebooks: push the code, then run `05_governance_network` → `06_fusion_lora` (run its memory test first) → `07_governed_eval`, and copy `governance.pt` and `fusion_lora.safetensors` into `models/`. Notebook 01b's resolution policy text was revised; rerun it if you want the stored `expected_resolution` field to match (the new notebooks do not depend on it).

## Notes

- The router runs reported so far used `RELAX = 0.3` (sketch strength after the hold time) and `GUIDE_W = 0.25` (constraint guidance weight), as in the code. Earlier slides and the report state 0.7 and 0.03; those figures are wrong.
- `01_freeze_pilot_dataset` keeps **all 200** samples (the filter lines are commented out), including 18 that failed the reference filters and 23 with unverified segmentation. The fusion LoRA trains only on rows that passed both.

- Notebook outputs are stripped before committing (`nbstripout`), so figures and tables are regenerated by running them.
- The Sketchy database has its own licence; this repo contains only paths, captions and metadata, not the images.
