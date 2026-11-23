# Constraint-Driven Multimodal Diffusion (Text + Sketch + Reference Image)

**IE 643 course project, IIT Bombay. Team: One Sheep Army** (R. Harshitha, 24B0053)

One object is generated from a **text caption**, a **sketch** and a **reference image** that may disagree.
We fix who governs what and ask the model to enforce it:

| Modality | Governs |
|---|---|
| Sketch | shape and placement |
| Text | object identity |
| Reference image | colour and material (**the reference wins colour** when the caption names a different one) |

## Submission

| Item | Where |
|---|---|
| Poster (A1, 594 x 841 mm), PDF | [`poster/OneSheepArmy_IE643_CourseProject_Poster.pdf`](poster/OneSheepArmy_IE643_CourseProject_Poster.pdf) |
| Poster, LaTeX source (course template) | [`poster/latex/`](poster/latex) |
| Novelty assessment report (2 pages) | [`report/OneSheepArmy_IE643_NoveltyAssessment_report.pdf`](report/OneSheepArmy_IE643_NoveltyAssessment_report.pdf) |
| Submitted method: the **Attribute Routing Layer (ARL)** | this README, [`src/pipeline.py`](src/pipeline.py), [`src/router.py`](src/router.py), [`src/governance/`](src/governance) |
| Run that produced the ARL results | notebook [`notebooks/11_pilot_flag_gate.ipynb`](notebooks/11_pilot_flag_gate.ipynb), outputs in [`results/pilot_flaggate_v1/`](results/pilot_flaggate_v1) |
| Live demo (Gradio, needs a Colab T4) | notebook [`notebooks/12_arl_live_demo.ipynb`](notebooks/12_arl_live_demo.ipynb); recorded replay in [`app/arl_demo_replay.html`](app/arl_demo_replay.html) |
| Results of every approach tried | [Results of all tries](#results-of-all-tries) below, detail in [`docs/`](docs) |
| Demo video | Google Drive folder, linked by the QR code on the poster |

## The submitted method: Attribute Routing Layer (ARL)

The ARL sits between the three inputs and a **frozen** SDXL + scribble ControlNet + IP-Adapter backbone. No diffusion weights are trained; it fits a free Colab T4 (about 10 GB, about 7 s per 512 px image, 20 steps).

```
 text ----------\                               +--> flag-gated text (UNet text, ControlNet text, pooled embedding)
 sketch --------+--> frozen CLIP ViT-L/14 --+   |
 reference -----/    features               +--> governance network --> flags g_obj, g_col, g_mat
                                            +--> router MLP --------> ControlNet scale, hold time, IP-Adapter scale, CFG
                                                                       |
                                  frozen SDXL + scribble ControlNet + IP-Adapter ViT-H --> image
```

| Part | Code | What it does |
|---|---|---|
| Governance network (trained) | `src/governance/model.py`, `features.py`, `labels.py`, notebook `05` | Frozen CLIP embeddings of six slots (caption, object phrase, sketch, reference, grey reference, colour part), shared projection + slot embeddings + 15 pairwise cosines -> MLP -> three conflict flags. Trained with weighted BCE on rule labels generated from metadata (no images generated); Platt-calibrated on validation photos. |
| Router (trained) | `src/router.py`, notebook `04`, `models/router.pt` | Six CLIP cosines -> MLP (32-32-4, sigmoid) -> ControlNet scale, sketch hold time, IP-Adapter scale, CFG. Five-fold ensemble trained on a one-factor-at-a-time grid (9 generated variants per training item). |
| Flag-gated text | `Pipeline.mix_text`, `Pipeline.run(text_flags=...)` in `src/pipeline.py` | Colour words -> "colored", material words -> "solid". Embedding `p' = p - f (p - p_neutral)` with `f` the colour or material flag, applied to the UNet text, the **ControlNet text** and the pooled embedding. Applying it to the ControlNet text closes the leak where the caption's colour still reached the UNet through ControlNet. |
| Schedule | `sketch_schedule`, `ip_shape`, `style_block` in `src/pipeline.py` | Sketch held, then relaxed (x0.3); reference enters through the style block only; geometry first, appearance later. |

### Headline result (pilot, one held-out photo, three conflict rows)

Held-out photo 0027, rows `obj`, `col`, `mat`; identical seed, 512 px, 20 steps; n = 3 images per arm, so **indicative only**.

| Method | Colour hit | Material hit | Edge IoU | DINO-ref | CLIP-T |
|---|---|---|---|---|---|
| Fixed baseline | 0.33 | 0.67 | **0.378** | **0.291** | 0.205 |
| Schedule only (control) | **1.00** | 0.00 | 0.066 | 0.122 | 0.233 |
| Router | **1.00** | 0.33 | 0.044 | 0.147 | **0.249** |
| Flag gate | 0.33 | 0.33 | 0.370 | 0.252 | 0.210 |
| Flags forced to 1 (control) | 0.67 | **1.00** | 0.277 | 0.249 | 0.212 |
| **ARL (router + flag gate)** | **1.00** | 0.67 | 0.059 | 0.159 | 0.247 |

Per-image scores: [`results/pilot_flaggate_v1/scores.csv`](results/pilot_flaggate_v1/scores.csv). Images: [`contact_sheet.png`](results/pilot_flaggate_v1/contact_sheet.png), [`pilot_qualitative.png`](results/figures/pilot_qualitative.png).

What it shows, stated plainly:
- The ARL reaches the reference colour in 3 of 3 rows and the material in 2 of 3 (fixed baseline: 1 of 3 and 2 of 3).
- The schedule-only control reaches the same colour score, so most of the colour gain comes from the schedule, not the learned router.
- **The cost is sketch fidelity** (edge IoU 0.059 vs 0.378). The flag gate alone keeps the sketch (0.370) but rarely changes colour.
- No arm resolves the **object conflict** (the caption's ape is never produced in any row).
- One photo and three rows is a pilot. Evaluation on all 20 test photos and the 800 conflict rows is future work.

## Results of all tries

Everything below was run on the 200-photo Sketchy pilot (`pilot_v1`, 160 / 20 / 20 split by photo). Raw outputs are on Drive (see [Not in this repo](#not-in-this-repo-too-large-kept-on-google-drive)); the numbers here are copied from the run reports. Full histories are in [`docs/`](docs).

| # | Approach | Where | Result | Outcome |
|---|---|---|---|---|
| 1 | **SDXL baseline**: frozen SDXL + scribble ControlNet + IP-Adapter, fixed strengths | `notebooks/02_sdxl_baseline_model.ipynb` | Works roughly, but the reference copies its own object: DINO to reference 0.673 vs to target 0.105. | Baseline |
| 2 | **Gated fusion MLP** | `notebooks/03_gated_fusion_model.ipynb` | Superseded by the router's attribute-tied outputs. | Dropped |
| 3 | **Attribute router**: 6 CLIP cosines -> 4 global strengths | `notebooks/04_attribute_router_model.ipynb`, `models/router.pt` | No better than the hand schedule (beats a constant on 1 of 4 outputs); cosines cannot see colour or material conflicts; guidance caused artefacts. | Kept as a component of the ARL |
| 4 | **Governance network**: frozen CLIP features -> 3 conflict flags | `notebooks/05_governance_network.ipynb`, `src/governance/` | Detection works: colour 0.755 (old detector 0.540), material 0.940 (0.643). But images barely changed: colour hit 0.40, same as the schedule; flipping the colour gate changed pixels by only 7.8/255. | Kept as a component of the ARL |
| 5 | **Fusion LoRA** on the whole UNet | `notebooks/06_fusion_lora.ipynb`, `07_governed_eval.ipynb`, `src/train_fusion_lora.py` | Object conflicts resolved 8 of 8 (vs 2 of 8) and more realistic images, but edge IoU fell 0.369 -> 0.230: it learned to ignore the sketch. | Dropped |
| 6 | **Role gates inside SDXL** v1 to v3 (learned per-layer gates on text, sketch and reference paths) | `notebooks/08_role_gates_train.ipynb`, `09_role_gates_eval.ipynb`, `src/rolegate/`, `src/train_role_gates.py` | v1: sketch gates saturated, sketch ignored. v2: sketch fixed, colour still followed the text. v3: no gate moved more than 0.06 from its start (run stopped at step 450 of 600); effects came from hand-set initial values. Led to finding the ControlNet caption leak. | Dropped |
| 7 | **Colour adapter** (palette -> UNet residuals) and **role LoRA** | `src/coloradapter/`, `src/rolelora/`, `src/train_colour_adapter.py`, `src/train_role_lora.py`, notebook `10` | Palette adapter ignored; role LoRA raised edge IoU to 0.753 on 12 test photos but target similarity (DINO to target) fell 0.359 -> 0.057 and colour / material compliance dropped to 0.30 / 0.00. | Dropped |
| 8 | **Neutral caption + IP-Adapter scale 1.0** (rule applied outside the model), 12 test photos | `src/pipeline.py` | Object compliance 0.10 -> 0.30, edge IoU 0.213 -> 0.376; colour 0.70 and material 0.67 unchanged; swapped reference 0.70 -> 0.00. | Earlier submission candidate |
| 9 | **ARL = router + flag-gated text** (submitted) | `notebooks/11_pilot_flag_gate.ipynb`, `src/pilot_flag_gate.py` | Table above: reference colour in 3 of 3 pilot rows, at a cost in sketch fidelity. | **Submitted** |
| 10 | M1: conditional routing of the text colour / material channel by distillation | `src/m1/`, `tests/test_m1_routing.py` | Experimental code and unit tests only; no results reported. | Not evaluated |
| 11 | FLUX.1-dev + UniCombine | `notebooks/unicombine_flux_attempt/` | Training ran out of memory on the free T4 at the first step. | Dropped |
| 12 | Reference images v1 (SDXL + IP-Adapter) | `notebooks/data_prep/sketchy_reference_ipadapter.ipynb` | Generated references came out wrong; v2 segments the object onto grey and filters candidates. | Replaced by v2 |

**Lesson:** detecting conflicts and rescaling global strengths does not enforce attribute-specific control. What moved the output was routing the **text path, including ControlNet's**, so the reference could supply colour; learned gates inside SDXL did not train.

## Data

Source: the [Sketchy database](https://sketchy.eye.gatech.edu/) (125 classes, 12,500 photos, 75,481 sketch-photo pairs). We use 200 photos, one object each.

| Property | Value |
|---|---|
| Split | 160 train / 20 val / 20 test, by photo (`pilot_v1`) |
| Captions | Qwen3-VL-2B-Instruct prompted with each photo and its Sketchy class as a hint (`data/sketchy_captions_sample.csv`) |
| Reference | neutral object (vase, mug, ball, bottle, box) carrying the photo's colour, material and finish, segmented on grey (`data/sketchy_with_refs_v2.csv`) |
| Conflict rows | +800: object, colour, material, swapped reference (`notebooks/data_prep/01b_add_conflicts.ipynb`); `src/governance/labels.py` generates more from metadata |

| Notebook | What it does |
|---|---|
| `data_prep/sketchy_make_csv.ipynb` | Builds `sketchy_pairs.csv`, then captions 200 sampled photos. |
| `data_prep/sketchy_reference_v2.ipynb` | Reference images v2 (segmentation, neutral objects, VLM-extracted colour / material / finish, DINO and appearance filters). |
| `01_freeze_pilot_dataset.ipynb` | Freezes `pilot_v1`: aligned (target, sketch, reference, caption) with a fixed seed; keeps all 200 samples (18 failed the reference filters, 23 have unverified segmentation). CPU only. |
| `data_prep/01b_add_conflicts.ipynb` | Adds the labelled conflict rows without touching `pilot_v1.zip`. |

## Repo layout

```
.
├── README.md
├── poster/                  final poster: PDF, LaTeX source, HTML source, figure script
│   ├── latex/               course-template LaTeX project (compile with pdflatex)
│   ├── figures/, pilot_images/, make_figures.py, poster.html, build.ps1
├── results/                 outputs of the ARL pilot (scores, contact sheet, notebook with outputs)
├── docs/                    stage reports, literature review, plans for each attempt
├── data/                    captions, pairs, reference tables (images stay on Drive)
├── models/router.pt         trained router weights
├── src/
│   ├── pipeline.py          Pipeline (frozen SDXL + ControlNet + IP-Adapter), mix_text, schedules
│   ├── router.py            RouterEnsemble
│   ├── governance/          labels, features, GovernanceNet, metrics, EvalKit
│   ├── pilot_flag_gate.py   the pilot behind the headline table
│   ├── rolegate/, rolelora/, coloradapter/, m1/    attempts 5-7 and 10
│   └── eval_metrics.py, eval_roles.py, sketchy_loader.py, train_*.py
├── notebooks/               01-11 + data_prep/ + unicombine_flux_attempt/
├── app/app.py               Gradio interface
└── tests/                   unit tests (CPU)
```

## Running it

1. Open a notebook in Colab with a T4 GPU (01 and 01b run on CPU) and mount Drive. Notebooks use `ROOT = /content/drive/MyDrive/sketchy_dataset_one_sketch`.
2. Data: `sketchy_make_csv` -> `sketchy_reference_v2` -> `01_freeze_pilot_dataset` -> `01b_add_conflicts`.
3. Components: `02` (baseline) -> `04` (router, writes `models/router.pt`) -> `05_governance_network` (writes `governance/governance.pt` on Drive).
4. **ARL pilot: `11_pilot_flag_gate`.** It clones the repo, loads `pilot_v1.zip` and `governance.pt` from Drive and writes images, `scores.csv`, `config.json` and `contact_sheet.png`. Without `governance.pt` it falls back to rule-label ("oracle") flags and says so.
5. SDXL, scribble ControlNet and IP-Adapter weights download from Hugging Face on first run.

Unit tests (CPU, no GPU needed):

```
pip install pytest torch numpy pandas pillow
PYTHONPATH=src python -m pytest -q tests      # 43 passed, 3 skipped at submission
```

Poster: `python poster/make_figures.py` regenerates the figures; `poster/build.ps1` builds the PDF from `poster/poster.html` with headless Edge (Windows). The LaTeX source in `poster/latex/` is the course template and was not compiled locally.

## Interface

`app/app.py` is a Gradio app (`pip install -r app/requirements.txt`, then `cd app && python app.py`, `share=True` on Colab). It needs `models/governance.pt` from notebook 05.

## Not in this repo (too large, kept on Google Drive)

| File | Size | Location (`MyDrive/`) |
|---|---|---|
| `sketchy.zip`, the raw Sketchy database | ~1.1 GB | My Drive root |
| `pilot_v1.zip`, the frozen pilot | ~103 MB | `sketchy_dataset_one_sketch/` |
| `reference_images/`, `ref_v2_backup.zip` | folder / zip | `sketchy_dataset_one_sketch/` |
| `sdxl_runs/` (generated outputs, scores, manifests for every run, including `pilot_flaggate_v1`) | folders | `sketchy_dataset_one_sketch/` |
| `governance/governance.pt`, `role_gates*/` | checkpoints | `sketchy_dataset_one_sketch/governance/` |

## Notes

- The router runs reported used `RELAX = 0.3` (sketch strength after the hold time) and `GUIDE_W = 0.25` (constraint guidance weight), as in the code. Earlier slides stated 0.7 and 0.03; those figures were wrong.
- Notebooks 01-04, `final_output_eval` and the data-prep notebooks keep their saved outputs; notebooks 05-11 are stored without outputs (the output copy of notebook 11 is in `results/pilot_flaggate_v1/`).
- The Sketchy database has its own licence; this repo contains only paths, captions and metadata, not the images.
- The poster template is the course template (a0poster class by Kettl and Weiser, CC BY-NC-SA 3.0).
