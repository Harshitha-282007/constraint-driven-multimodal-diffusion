# IE 643 Presentation Study Guide: every model we tried, and why the ARL is the hero

Team One Sheep Army. Project: *Constraint-Driven Multimodal Diffusion using Text, Sketch and Reference Image.*
Built from the repo README, `docs/`, the novelty report, `HANDOFF_CONTEXT.md` and `results/pilot_flaggate_v1/scores.csv`. Every number below comes from those files.

---

## 0. The 60-second version (memorise this)

- **Task.** One object, three inputs that can disagree: a text caption, a sketch, a reference image. The model must decide *automatically* which input controls which attribute.
- **Our rule (fixed per attribute).** Sketch → shape and placement. Text → object identity. Reference → colour and material. The reference **wins colour** even when the caption names another colour.
- **Backbone.** Frozen SDXL + scribble ControlNet + IP-Adapter ViT-H. Runs on a free Colab T4 (about 10 GB, about 7 s per 512 px image, 20 steps).
- **Hero = Attribute Routing Layer (ARL)** = trained **router** + trained **governance network** driving **flag-gated neutral text**. No diffusion weights are trained.
- **Honest result.** On the pilot, ARL hits the reference colour in **3/3** rows (fixed baseline 1/3), but sketch fidelity drops (**edge IoU 0.059 vs 0.378**). The pilot is **one photo, three rows**: indicative only.
- **Real finding.** Detecting conflicts and rescaling global strengths is not enough. What actually moved colour was routing the **text path, including the ControlNet's caption**, so the reference could supply colour. We found that leak ourselves after the learned gates failed.

**Professor's question to be ready for:** "How does the model understand which part of the input governs which part of the output?"
**Our answer:** A trained governance network reads CLIP features of all three inputs and outputs three conflict flags (object, colour, material). A trained router sets four global strengths. The flags then blend a *neutral caption* into the text path, so a stale caption colour cannot compete with the reference. The role assignment is learned detection plus a fixed rule, not learned weights inside SDXL. Say that plainly; we tried learned weights inside SDXL and they did not train (Section 3, models 5–7).

---

## 1. Setup you must be able to explain

| Item | Detail |
|---|---|
| Dataset | Sketchy: 125 classes, 12,500 photos, 75,481 sketch-photo pairs. We use **200 photos**, one object each. |
| Split | 160 train / 20 val / 20 test, **by photo** (`pilot_v1`). |
| Captions | Qwen3-VL-2B-Instruct, prompted with the photo and its Sketchy class. |
| Reference images (v2) | A **neutral object** (vase / mug / ball / bottle / box) that carries the photo's colour, material and finish, segmented on grey. Neutral shape means the reference *can't* leak the target's shape, only its look. |
| Conflict rows | +800 labelled rows: object, colour, material, swapped reference. `governance/labels.py` can make more from metadata. |
| Brief constraints | A **trained** model is required; **one object**; automatic modality control; free T4. |
| Backbone | Frozen SDXL, xinsir scribble ControlNet, IP-Adapter ViT-H (non-plus). |

Reference images v1 (SDXL + IP-Adapter generated) came out wrong, so v2 uses segmentation onto grey with filtering. 18 of 200 failed the reference filters and 23 have unverified segmentation; all 200 were kept.

---

## 2. The hero model: Attribute Routing Layer (ARL)

```
text ----------\                              +--> flag-gated text (UNet text, ControlNet text, pooled embedding)
sketch --------+--> frozen CLIP ViT-L/14 --+  |
reference -----/    features              +--> governance network --> flags g_obj, g_col, g_mat
                                          +--> router MLP ---------> ControlNet scale, hold time, IP scale, CFG
                                                                      |
                                frozen SDXL + scribble ControlNet + IP-Adapter ViT-H --> image
```

### 2.1 The four parts

| Part | Trained? | What it does |
|---|---|---|
| **Governance network** (`src/governance/`, nb 05) | Yes | **Inputs you supply: text, sketch, reference** (plus an optional object name; if omitted, the whole caption is used in its place). Frozen CLIP ViT-L/14 embeds them, and the code builds a 6-row input tensor: caption, object phrase, sketch, reference, **grey reference** and **colour part (`ref − grey`)**, the last two derived automatically from the reference. Shared projection + slot embeddings + 15 pairwise cosines + the size of the colour vector → MLP → **three conflict flags**. Weighted BCE on rule labels generated from metadata (no images generated). Platt-calibrated on validation photos. |
| **Router** (`src/router.py`, nb 04, `models/router.pt`) | Yes | **Six CLIP cosines → MLP (32-32-4, sigmoid)** → ControlNet scale (0.5–1.4), sketch hold time (0.4–1.0), IP-Adapter scale (0.2–0.8), CFG (4–10). **Ensemble of 5 networks (different seeds, all trained on all 28 items)**; 5-fold cross-validation is a separate check. Trained on a one-factor-at-a-time grid, 9 generated images per training item. Details in Section 9. |
| **Flag-gated text** (`Pipeline.mix_text`) | No (rule) | Colour words → "colored", material words → "solid" to make a neutral caption. Then `p' = p − f·(p − p_neutral)`, where `f` is the colour or material flag. Applied to **three** paths: UNet text, **ControlNet text**, pooled embedding. |
| **Schedule** | No (rule) | Sketch held, then relaxed (×0.3). Reference strength ramps in late (zero until 15% of the steps, full by 40%). Geometry first, appearance later. (The README says the style block only, but the ARL arm runs on all blocks; see Section 9.7.) |

### 2.2 Why it works (the mechanism)

1. The caption says "purple motorcycle", the reference is silver. Left alone, the text wins colour (CLIP's causal text encoder also spreads a colour word into later tokens and the pooled embedding).
2. If the governance net raises the colour flag, the colour word is blended out of the text on **every** path, so the caption no longer pushes purple.
3. The IP-Adapter (all blocks in the ARL arm), now unopposed, supplies the reference colour.
4. The sketch holds shape early; the schedule then lets appearance form.

**The key discovery:** closing the colour gate inside the UNet was not enough, because `pipeline.py` fed the **full caption to the ControlNet**, and "purple" re-entered via ControlNet residuals. Neutralising the ControlNet's text too fixed the leak. This is the best "we debugged it" story you have.

### 2.3 Pilot results (held-out photo 0027, rows obj / col / mat, same seed, 512 px, 20 steps, n = 3 images per arm)

| Method | Colour hit | Material hit | Edge IoU | DINO-ref | CLIP-T |
|---|---|---|---|---|---|
| Fixed baseline | 0.33 | 0.67 | **0.378** | **0.291** | 0.205 |
| Schedule only (control) | **1.00** | 0.00 | 0.066 | 0.122 | 0.233 |
| Router | **1.00** | 0.33 | 0.044 | 0.147 | **0.249** |
| Flag gate alone | 0.33 | 0.33 | 0.370 | 0.252 | 0.210 |
| Flags forced to 1 (control) | 0.67 | **1.00** | 0.277 | 0.249 | 0.212 |
| **ARL (router + flag gate)** | **1.00** | 0.67 | 0.059 | 0.159 | 0.247 |

Metric glossary: **Colour/material hit** = output matches the reference's attribute. **Edge IoU** = sketch fidelity (overlap of output edges with the sketch). **DINO-ref** = similarity to the reference (high can mean the reference object leaked). **CLIP-T** = text alignment.

### 2.4 Limitations: say these before they are asked

1. **Pilot only.** One photo, three rows. Not statistically meaningful. Full eval on 20 test photos and 800 conflict rows is future work.
2. **Most of the colour gain comes from the schedule, not the learned parts.** The schedule-only control also scores colour 1.00. The router alone scores 1.00 too. The ARL adds material (0.67 vs 0.00) over the schedule, nothing more on colour.
3. **Sketch fidelity collapses** (0.059 vs 0.378). The cause is the schedule relaxing the sketch (×0.3) and a router that leaves ControlNet weak. The flag gate *alone* keeps the sketch (0.370) but rarely changes colour. This is the central trade-off.
4. **Object conflict is not solved.** The caption's object (an ape) was never produced in any row.
5. **No diffusion weights are trained.** The "trained model" is the router + governance net (small MLPs). Whether that satisfies the brief's "trained/finetuned model" was never confirmed with the TA. Expect this question.
6. **Rule-based flags.** Colour/material conflict is decided by a rule on labels; the network *detects* it, it does not *learn what governs what*.
7. **Governance detection vs image effect.** Detection is good (colour 0.755 vs 0.540, material 0.940 vs 0.643) but, alone, did not change images.

---

## 3. Every model tried (in order), with why it passed or failed

Quick scoreboard:

| # | Model | Verdict |
|---|---|---|
| 1 | SDXL + ControlNet + IP-Adapter baseline | Baseline (works roughly) |
| 2 | Gated fusion MLP | Dropped |
| 3 | Attribute router | **Kept** (part of ARL) |
| 4 | Governance network | **Kept** (part of ARL) |
| 5 | Fusion LoRA (whole UNet) | Dropped: ignored the sketch |
| 6 | Role gates inside SDXL v1–v3 | Dropped: gates didn't learn |
| 7 | Colour adapter + role LoRA | Dropped |
| 8 | Neutral caption + IP scale 1.0 | Earlier candidate |
| 9 | **ARL** | **Submitted (hero)** |
| 10 | M1 distillation routing | Not evaluated |
| 11 | FLUX.1-dev + UniCombine | Dropped: out of memory on T4 |
| 12 | Reference images v1 | Replaced by v2 |

### Model 1. SDXL baseline (nb 02)
- **What:** Frozen SDXL + scribble ControlNet + IP-Adapter, fixed strengths.
- **Result:** Works roughly. Fixed weights gave the best edge precision.
- **Why it falls short:** The reference **copies its own object**: DINO to reference **0.673** vs to target **0.105**. It cannot separate "take the colour" from "take the object".
- **Use it as:** the comparison point and the starting backbone for everything else.

### Model 2. Gated fusion MLP (nb 03)
- **What:** An early MLP fusing the signals.
- **Why dropped:** Superseded by the router, whose outputs are tied to attributes.

### Model 3. Attribute router (nb 04): kept
- **What:** 6 CLIP cosines → MLP → 4 global scalars (sketch strength, hold time, IP strength, CFG), plus time schedule and optional guidance.
- **Failed on its own because:**
  - It beat a constant on **only 1 of 4** outputs; no better than the hand schedule.
  - CLIP cosines **cannot see colour or material conflicts** (conflict scores: colour 0.610, material 0.541, swapped reference 0.597, normal 0.604, so almost no separation). Router outputs barely varied.
  - Constraint guidance improved colour match by 0.042 but cut edge precision by 0.085, and caused artefacts.
- **Trap to avoid:** the code used `RELAX = 0.3` and `GUIDE_W = 0.25`. Early slides said 0.7 and 0.03. **The slides were wrong, the code is right.** Say "0.3 and 0.25".
- **Why it's still in the ARL:** it is the trained part that sets ControlNet scale, hold time, IP scale and CFG, and gives the brief's "learned routing".

### Model 4. Governance network (nb 05): kept
- **What:** Frozen CLIP features of the three inputs (text, sketch, reference; plus grey and colour-only versions derived from the reference, and an optional object phrase) → 3 conflict flags (object, colour, material). Full walk-through in Section 8.
- **Passed (detection):** colour **0.755** (old detector 0.540); material **0.940** (old 0.643).
- **Failed (effect):** images barely changed. Colour hit 0.40, identical to the schedule. Flipping the colour gate changed pixels by only **7.8/255**.
- **Lesson:** a flag is only useful if something downstream *acts* on it. That "something" ended up being the neutral-text blend in the ARL.

### Model 5. Fusion LoRA on the whole UNet (nb 06, 07): dropped
- **What:** LoRA on attention layers of the whole UNet, trained on 160 (target, sketch, reference, caption) quadruples with role dropout (full inputs 40%, grey reference 25%, caption colour words removed 25%, no conditions 10%). Real photo as target.
- **Passed:** object conflicts resolved **8/8** (baseline 2/8), more realistic images.
- **Failed:** **edge IoU fell 0.369 → 0.230.** To reproduce the whole photo it learned the whole scene, and because it sits where ControlNet residuals enter, it learned to **ignore the sketch**. The identity gain was bought with the shape role.
- **Sound bite:** "It solved one role by sacrificing another."

### Model 6. Role gates inside SDXL, v1 → v3 (nb 08, 09, `src/rolegate/`): dropped
This is the one the professor's question points at ("SDXL itself should understand roles"), so know it well.
- **Idea:** learned scalar gates `g = 2·σ(b0 + W_h·LN(h) + W_c·c)`, with `c` = [3 flags, sin, cos, f, 1−f]. Cross-attention gates: text_other, text_identity, text_colour, text_material, ref_colour, ref_texture. Sketch gates on the ControlNet residuals: sketch_contour, sketch_placement.
- **v1:** sketch gates **saturated** to 0 or 2 (they read raw residuals), so the sketch was ignored. Colour followed the text.
- **v2:** SketchGate got LayerNorm, 0.2× lr, a keep-loss. Added **neutral-caption channels** and a pooled gate. Sketch fixed (motorcycle and pistol follow sketch; axe became a log pile), but colour **still followed the text** (purple motorcycle, gold pistol). Gates reached only about 0.5–0.6 (text_colour) and about 0.1 (ref).
- **v3 (commit ab7d57e):** closed text-colour/material gates at start (FLAG_INIT = −3), stronger reference colour in the style block (`up_blocks.0.attentions.1`).
  - Run stopped at step **450 of 600** (Colab disconnect).
  - **No gate moved more than 0.06 from its start.** Colour loss flat at about 0.10; contour about 0.97 flat. Any effect came from **hand-set initial values**, not learning.
  - Axe got worse (log pile, sketch lost).
- **Why it failed (diagnosis):**
  - Training signal too weak: single-step x0 decode at random t; a prior pulling gates back; lr 5e-4 too low for scalar gates in a frozen SDXL.
  - The IP-Adapter's single global vector mixes colour, shape and identity, so turning it up leaks the reference object.
  - **The ControlNet leak** (full caption → ControlNet) meant the colour word reached the UNet through the back door even with the text gate closed.
- **Best honest line:** "We built learned gates; they did not train. But debugging them led us to the real cause."

### Model 7. Colour adapter and role LoRA (nb 10): dropped
- **Colour adapter** (`src/coloradapter/`): reference → k-means palette in CIELAB (4 colours) → MLP → 4 tokens → zero-init cross-attention in 46 UNet layers (~14 M params), trained with hue-rotated rows. **Result:** the palette was **ignored**: identical output for red, green, silver.
- **Role LoRA** (`src/rolelora/`): LoRA only on the *input interfaces*: text attn2 K/V (~4.8 M), IP-Adapter K/V (~3.4 M), late blocks, rank 16, ControlNet frozen. Loss: masked diffusion + 2× colour + 0.5× texture + 0.5× preservation. On 12 test photos:
  - Highest edge IoU **0.753**, **but** DINO to target **0.359 → 0.057**, CLIP-T 0.263 → 0.220.
  - Colour 0.30, material 0.00, swapped reference 0.40 (fixed baseline: 0.70 / 0.67 / 0.70).
  - Outputs followed the sketch but **no longer looked like the photo**; colour and material roles weren't learned.
- **What the images looked like** (from an earlier session's read of `lora_640.png`, Drive folder `governance/role_lora_run1`; I have not seen the image myself): with the LoRA off the vintage-motorcycle outputs are clean but grey or black and don't take the reference's silver/black/white. At LoRA scale 0.5 they turn **sepia or purple with hatched, sketchy textures**; at scale 1.0 they **break into pink and orange artefacts**. So the LoRA damaged image quality without moving colour from the reference. Compliance table (n = 10 per type): colour 0.30, material 0.00 vs 0.70 / 0.67 for fixed.
- **Caveat to admit:** 160 photos, one 400-step run, no repeat seeds, so we can't tell over-fitting from a wrong loss design. Object-vs-sketch conflict had no exact target, so no training rows.

### Model 8. Neutral caption + IP scale 1.0 (rule outside the model): earlier candidate
- **What:** Colour words → "colored", material → "solid" on UNet text, ControlNet text and pooled embedding; IP-Adapter 0.5 → 1.0. 12 test photos.
- **Result:** object compliance 0.10 → 0.30; edge IoU 0.213 → 0.376; colour 0.70 and material 0.67 unchanged; swapped reference 0.70 → **0.00**.
- **Why not submitted:** it is a rule applied from **outside** the model, with no trained component deciding when to apply it. The ARL turns this into a *learned-flag-driven* version.

### Model 9. ARL: submitted. See Section 2.

### Model 10. M1 conditional routing by distillation (`src/m1/`)
- Scalar gates distilled from the neutral-caption rule; only unit tests, **no results**. If asked: "experimental, not evaluated, we don't claim it."

### Model 11. FLUX.1-dev + UniCombine
- The original plan for the intensive assessment. **Out of memory on the free T4 at the first training step.** This is why we moved to SDXL.

### Model 12. Reference images v1
- Generated with SDXL + IP-Adapter; came out wrong. Replaced by v2 (segmentation, neutral objects, VLM-extracted colour/material/finish, DINO and appearance filters).

---

## 4. The storyline (one coherent narrative for the talk)

1. **Intensive assessment:** proposed attribute routing on FLUX + UniCombine → OOM → SDXL pipeline with an MLP router. Router ≈ hand schedule. Reference copies its own object.
2. **Novelty question:** can a trained layer enforce *per-attribute* control rather than rescale global strengths?
3. **Detect:** governance net detects conflicts well but images don't change.
4. **Train inside SDXL:** Fusion LoRA (loses the sketch) → role gates v1–v3 (don't train) → colour adapter / role LoRA (ignore palette / lose the photo).
5. **Diagnose:** found the **ControlNet caption leak**. Colour had been getting in through a path our gates did not cover.
6. **Fix and route:** neutralise text on *all three* paths, driven by the governance flags, with the router setting strengths → **ARL**.
7. **Result and honesty:** reference colour in 3/3 rows; sketch fidelity is the price; pilot-sized.

---

## 5. Likely questions and answers

**Q. Where is the trained model?** The governance network (conflict flags) and a 5-fold router ensemble (`models/router.pt`). SDXL, ControlNet and IP-Adapter are frozen. We tried training inside SDXL (LoRA, gates, adapter) and it failed or hurt other roles. *(Admit the TA question on whether this meets the brief is open.)*

**Q. How does the model know which input governs what?** The governance network's three flags are the decision: object, colour, material conflicts. The mapping from attribute to governing input is a fixed rule (sketch → shape, text → identity, reference → colour/material). The flags control how much of each caption attribute survives.

**Q. Why does the reference win colour?** A team design choice: the reference image is the more direct evidence of appearance. It reverses our earlier rule (text colour stays).

**Q. Your ARL has worse sketch fidelity than the baseline. Isn't that a failure?** It's the measured trade-off on a 3-row pilot: relaxing the sketch lets the reference colour through. Flag-gate-only keeps the sketch (0.370) but rarely changes colour. We report it rather than hide it. Next step: hold the sketch longer or gate only where colour is decided.

**Q. Is the learned part actually necessary?** Honest answer: for **colour**, the schedule alone matches it (1.00). The ARL adds material (0.67 vs 0.00) and gives conflict-aware behaviour, but one photo cannot prove the learned parts matter.

**Q. Why did the learned gates inside SDXL fail?** Weak training signal (single-step decode, prior pulling back, low lr), the IP-Adapter's global vector entangling colour with shape, and the ControlNet leak.

**Q. What is the ControlNet leak?** The ControlNet received the full caption, so the colour word reached the UNet through ControlNet's residuals even when UNet text was neutral. Fix: neutralise the ControlNet text too.

**Q. Why not multi-object?** The brief's example is a single object and the professor did not ask for multi-object.

**Q. Why is the object conflict unsolved?** Text-vs-sketch identity needs the model to override the sketch's shape prior. No ARL arm did it; only the Fusion LoRA did (8/8), at the cost of the sketch.

**Q. What would you do next?** (a) evaluate on all 20 test photos and the 800 conflict rows; (b) keep the sketch while changing colour; (c) a colour-only reference channel that doesn't carry shape; (d) a stronger, per-timestep training signal for gates.

**Q. Can it run on a laptop / free GPU?** Yes: free Colab T4, about 10 GB, about 7 s per 512 px image. FLUX did not fit.

---

## 6. Numbers cheat-sheet

| Fact | Number |
|---|---|
| Photos / split | 200; 160 / 20 / 20 |
| Conflict rows | 800 (object, colour, material, swapped ref) |
| Reference copies its own object | DINO-ref 0.673 vs DINO-target 0.105 |
| Governance, colour detection | 0.755 (old 0.540) |
| Governance, material detection | 0.940 (old 0.643) |
| Governance gate flip effect | 7.8 / 255 pixels |
| Router vs constant | beats it on 1 of 4 outputs |
| `RELAX` / `GUIDE_W` (code) | 0.3 / 0.25 (slides wrongly said 0.7 / 0.03) |
| Fusion LoRA objects / edge IoU | 8/8 vs 2/8 / 0.369 → 0.230 |
| Role gates v3 | max gate movement 0.06; stopped at step 450/600 |
| Role LoRA | edge IoU 0.753; DINO-target 0.359 → 0.057; colour 0.30; material 0.00 |
| Neutral caption + IP 1.0 | object 0.10 → 0.30; edge IoU 0.213 → 0.376; swapped ref 0.70 → 0.00 |
| **ARL pilot** | colour 1.00; material 0.67; edge IoU 0.059; DINO-ref 0.159; CLIP-T 0.247 |
| Fixed baseline pilot | colour 0.33; material 0.67; edge IoU 0.378; DINO-ref 0.291; CLIP-T 0.205 |
| Runtime | T4, about 10 GB, about 7 s per 512 px image, 20 steps |

## 7. Where things are

| Need | Location |
|---|---|
| Poster | `poster_arl/OneSheepArmy_IE643_CourseProject_Poster.pdf` |
| Report | `report/OneSheepArmy_IE643_NoveltyAssessment_report.pdf` |
| ARL pilot notebook | `notebooks/11_pilot_flag_gate.ipynb`, outputs in `results/pilot_flaggate_v1/` |
| Live demo | `notebooks/12_arl_live_demo.ipynb`, replay `app/arl_demo_replay.html` |
| Core code | `src/pipeline.py` (`mix_text`, line 240), `src/router.py`, `src/governance/` |
| Per-image pilot scores | `results/pilot_flaggate_v1/scores.csv` |

---

## 8. How the governance network works, checked against the code

Sources: `src/governance/features.py`, `model.py`, `labels.py`, notebook 05, and `Pipeline.make_items` in `src/pipeline.py`.

### 8.1 Inputs
You supply **text, sketch, reference** (and optionally an **object name**). `make_items` then builds six rows:

| Row | How it is made |
|---|---|
| `cap` | CLIP text embedding of the caption |
| `obj` | CLIP text embedding of `"a photo of a {object}"`. If no object name is given, the code falls back to the whole caption (`obj = ... or s["caption"]`). |
| `sketch` | CLIP image embedding of the sketch (forced to dark lines on white) |
| `ref` | CLIP image embedding of the reference |
| `grey` | The same reference converted to greyscale, then embedded |
| `col` | `ref − grey`, a subtraction in embedding space (idea from the SADis paper). Always available, never supplied. |

CLIP is **frozen** (`ViT-L/14`, 768-d, `requires_grad_(False)`). The same functions are used for training and inference, so the network sees identical features both times.

### 8.2 The network (`GovernanceNet.forward`)
1. Normalise each of the six vectors to unit length (keep the length of `col`).
2. Project each with one shared `Linear(768 → 128)`, add a learned **slot embedding** (so the MLP knows which row is which), then ReLU. Flatten: 6 × 128 = 768 numbers.
3. Compute the **15 pairwise cosines** between all six vectors.
4. Concatenate: 768 + 15 + 1 (the length of the colour vector) = 784 numbers.
5. MLP `784 → 256 → 64 → 3` with dropout 0.2. Output: three logits.

Calibration then maps each logit to a probability (Platt scaling, fitted on validation photos), because the flags are later used as **continuous blend weights**, not just yes/no.

### 8.3 What the three flags mean (`labels.py`)
- `g_obj = 1` when the caption's object category differs from the sketch's category.
- `g_col = 1` when the caption **states a colour** that the reference does not have.
- `g_mat = 1` when the caption **states a material** that the reference does not have.

Importantly these are *conflict detectors*. Who wins is fixed separately by the rule (reference wins colour and material).

### 8.4 Training
- **Labels come from metadata by word-list rules** (`gates_for`): the caption's colour/material words compared with the reference's recorded colours/material. **No images are generated for training.**
- Rows: each photo expanded with object, colour, material, swapped-reference and agreeing variants (`PER_TYPE = 4`).
- Loss: weighted BCE (positive-class weighting for rare gates). 60 epochs, AdamW, lr 1e-3, batch 64.
- The network never sees the word lists. It sees only CLIP vectors, so in principle it can also catch wording the lists miss. Notebook 05 tests this with **synonym-paraphrased captions** ("crimson" for "red").
- Compared against logistic regression on the old router's six cosines, using 5-fold cross-validation grouped by photo. That comparison produced the "colour 0.755 vs 0.540, material 0.940 vs 0.643" numbers.

### 8.5 Worked example: "text says blue, reference is red"
`cap` encodes "blue ...", `ref`/`col` encode red. The `cap` and `col` vectors, and the cosine between them, disagree, and the network learned from labelled rows that this pattern means `g_col` high. The flag then makes `mix_text` blend the colour word out of the caption, so the reference supplies the red.

### 8.6 Honest check on the pilot (read this before presenting)
In the logged ARL pilot, the **predicted** flags did not match the rule labels in two of three rows:

| Row | Rule flags (truth) | Predicted flags | Verdict |
|---|---|---|---|
| obj ("black ape ...") | g_obj 1, g_col 1, g_mat 0 | g_obj **0.01**, g_col 0.70, g_mat 0.01 | Missed the object conflict |
| col ("... axe ..., black") | g_obj 0, g_col 1, g_mat 0 | g_obj **0.92**, g_col **0.30**, g_mat 0.10 | Wrong object flag; weak colour flag |
| mat ("... made of metal") | g_obj 0, g_col 0, g_mat 1 | g_obj 0.06, g_col 0.10, g_mat **0.91** | Correct |

Consequences you should be ready to state:
- The pilot used the **real predicted flags** (`flags_used="predicted"`), not oracle labels.
- This explains why the **flag gate alone** barely changed colour in the colour row (flag 0.30). The ARL's 3/3 colour hit comes mainly from the schedule, which matches Section 2.4.
- Detection accuracy on the held-out test set (0.755 / 0.940) is an average; on one photo it can miss. Say "detection works on average, but the pilot photo shows it can miss".
- The `obj` flag only feeds the sketch relaxation in older pipelines. The ARL's text blend uses only `g_col` and `g_mat`.


---

## 9. How the router works, checked against the code

Sources: `src/router.py`, `src/governance/features.py` (`six_cosines`), `src/pipeline.py` (`run`, `cs_shape`, `ip_shape`), notebook 04 and its saved outputs.

### 9.1 What it does
Per item (one text, sketch, reference), it outputs **four global numbers** that set how strongly each input acts:

| Output | Meaning | Range (low to high) | Centre (default) |
|---|---|---|---|
| `cs` | ControlNet (sketch) strength | 0.5 to 1.4 | 1.0 |
| `hold` | fraction of denoising steps the sketch stays at full strength | 0.4 to 1.0 | 0.7 |
| `ip` | IP-Adapter (reference) strength | 0.2 to 0.8 | 0.5 |
| `cfg` | classifier-free guidance, i.e. text strength | 4 to 10 | 7 |

### 9.2 Input: six CLIP cosines
CLIP ViT-L/14 embeddings, then `six_cosines`:
`[1 - cos(object, sketch), 1 - cos(object, reference), 1 - cos(caption, attribute phrase), 1 - cos(sketch, reference), cos(caption, sketch), cos(caption, reference)]`.
The "attribute phrase" is text like "a glossy metal object in silver colors". In the dataset it comes from Qwen3-VL metadata; in the old app it was a CLIP zero-shot pick from fixed lists.

### 9.3 Network
`Linear(6,32) - ReLU - Linear(32,32) - ReLU - Linear(32,4) - Sigmoid`. Inputs are standardised (mean/std from training). The sigmoid output in 0 to 1 is mapped onto each range above. The final answer is the **mean of 5 networks** trained with seeds 0 to 4.

### 9.4 How it was trained: no labelled "right answer" exists, so targets are built from generated images
1. **Grid:** for each of **28 training items**, generate **9 images**: the centre point plus each of the 4 factors moved to its low and high level, one at a time (252 images).
2. **Score** each image on three things: geometry, appearance, semantics. Each is standardised (z-score) and summed with equal weights into one utility `S`.
3. **Target per factor:** among the images where the *other* three factors sit at centre, take a soft-max over `S` (temperature 0.5) of the factor levels, and average them. The result is a soft-best level in 0 to 1. This assumes each factor acts independently ("main effects").
4. **Fit:** MSE regression, Adam, lr 3e-3, weight decay 1e-3, 500 full-batch epochs.

### 9.5 How the output is used at generation (legacy path, which is what the ARL calls)
In `Pipeline.run` with `p=router`, `shaped=True`:
- ControlNet scale at step fraction `f`: `cs` while `f < hold`, then `cs * 0.3` (`RELAX = 0.3`).
- IP-Adapter scale: `ip * ip_shape(f)`: zero before 15% of the steps, then ramps linearly to full by 40%. So geometry forms first and appearance later.
- Text: `eps = eps_uncond + cfg * (eps_cond - eps_uncond)`.
- Optional constraint guidance (`router_guided`), not used by the ARL.

### 9.6 What the router actually achieved: be honest
Notebook 04's saved cross-validation (28 items, lower MSE is better):

| Output | Router | Constant predictor | Beats constant? |
|---|---|---|---|
| `cs` | 0.083 | 0.122 | Yes |
| `hold` | 0.036 | 0.037 | Yes, barely |
| `ip` | 0.089 | 0.058 | **No** |
| `cfg` | 0.081 | 0.057 | **No** |

- The saved output shows it beating a constant on **2 of 4** outputs (one barely). Your README and handoff say "1 of 4". The saved notebook is a mix of runs (STAGE1 doc, item 8), so the exact count is unclear. Safest wording: "it did not clearly beat a constant predictor; at best on two of four outputs, and not on `ip` or `cfg`."
- **Outputs barely differ by conflict type** (test items: `cs` 0.87 to 1.09, `cfg` 7.0 to 7.9; the 0.5 to 1.4 and 4 to 10 ranges are much wider). The cosines can't see colour or material conflicts, so the router behaves almost like the hand schedule.
- On 18 conflict items the **schedule** (the step-shaping above), versus fixed weights, cut edge IoU by 0.24 (significant) and DINO to reference by 0.07. The router added no clear gain over the schedule.

### 9.7 Two corrections to what the README says about the ARL
1. **"Five-fold ensemble"** is wrong. It is a 5-seed ensemble; the 5-fold part is the cross-validation check.
2. **"Reference enters through the style block only"** does **not** apply to the submitted ARL. `style_only` defaults to true only when `gates` is passed (`style_only = governed if style_only is None`). The ARL arm calls `run(p=router, shaped=True, text_flags=...)` with no `gates`, so the IP-Adapter acts on **all** blocks. The style-block restriction belongs to the governed pipeline (`gates=...`). Likewise `g_obj` is not used by the ARL; only `g_col` and `g_mat` drive the text blend.
