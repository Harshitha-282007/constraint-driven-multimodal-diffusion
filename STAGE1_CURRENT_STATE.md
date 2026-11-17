# Stage 1: current state of the project

Constraint-Driven Multimodal Diffusion using Text, Sketch and Reference Image (IE 643, Team One Sheep Army).

Sources read: the slide deck (18 pages), the project report (15 pages), `README.md`, `app/app.py`, `src/*.py`, `models/router.pt`, and every notebook under `notebooks/` (code cells and saved outputs). Generated images, `pilot_v1.zip`, `metadata_with_conflicts.jsonl` and the run folders live on Google Drive and were **not** available, so nothing below is based on looking at outputs.

Labels used: **[V]** verified in code or saved notebook output, **[D]** stated in the deck/report/README only, **[A]** my inference.

---

## 1. Original proposal versus actual implementation

| | Proposed (deck p.5, report §4.1) | Built (code) |
|---|---|---|
| Backbone | FLUX.1-dev diffusion transformer | Frozen SDXL base 1.0 UNet, fp16-fix VAE **[V]** `app.py:61-64` |
| Sketch path | OminiControl-style latent-token injection + LoRA | Pretrained `xinsir/controlnet-scribble-sdxl-1.0`, frozen **[V]** |
| Reference path | LoRA customisation branch | Pretrained `h94/IP-Adapter` `ip-adapter_sdxl_vit-h`, frozen **[V]** |
| Fusion | UniCombine conditional-MMDiT attention, trained union LoRA | None. The three conditions are summed inside the stock UNet with scalar weights **[V]** |
| "Attribute routing layer" | A layer inside the attention (`attn_forward` in `UniCombineTransformerBlock.py`) that gates each condition stream per block/head (`02_unicombine_pipeline.ipynb`, last cell) | An MLP outside the network: 6 CLIP cosines in, 4 global scalars out **[V]** `app.py:262-279` |
| Trained parameters | Condition LoRAs + denoising LoRA | Only the router: 6→32→32→4, about 1.4k weights, ×5 ensemble **[V]** `models/router.pt` |
| Conflict handling | Conflict subset to test resolution | Labelled conflict rows (01b), 6 CLIP "conflict features", optional latent guidance **[V]** |

The FLUX route exists as notebooks (`notebooks/unicombine_flux_attempt/`, `src/sketchy_loader.py`) but was dropped after out-of-memory at the first training step **[D]**. The meaning of "routing" changed in the move: from *which tokens attend to which condition, inside the network* to *how strongly each whole modality is applied, for the whole image*.

Stated problem and goal (deck p.3): sketch fixes shape and layout, text names the object, reference gives colour/material/style; "detect attribute-level conflicts and set modality strengths per item and per denoising step." Note the goal as written is already about **strengths**, not about spatial or object-level correspondence.

---

## 2. Execution flow (as implemented in `app/app.py` and notebook 04)

`app.py` is extracted from `04_attribute_router_model.ipynb`; the functions below are identical in both.

1. **Preprocess** (`build_item`, `app.py:317-326`)
   - Sketch → greyscale, inverted to white-on-black if bright, letterboxed to 640 on black (`prep_sketch_pil`, `:311-314`).
   - Reference → letterboxed to 640 on grey.
   - Reference attribute phrase "a {finish} {material} object in {colour} colors": in the dataset it comes from Qwen3-VL (`sketchy_reference_v2`); in the app it is a CLIP zero-shot pick from fixed lists of 14 colours, 8 materials, 6 finishes (`zero_shot`, `:301-306`).
2. **Encode once** (`ui_encode`, `encode_text`, `encode_ip`, `:174-181`)
   - Caption → `pipe.encode_prompt` → 77-token sequence from SDXL's two CLIP text encoders + pooled vector, plus the negative prompt.
   - Reference → `pipe.prepare_ip_adapter_image_embeds` → IP-Adapter CLIP ViT-H image embedding.
3. **Conflict features** (`conflict_features`, `:183-190`): six cosines between CLIP ViT-L/14 embeddings of (object phrase, full caption, attribute phrase, sketch image, reference image). All are whole-image / whole-sentence embeddings.
4. **Router** (`predict_params`, `:276-279`): standardise features, average 5 MLPs, de-normalise to `cs ∈ [0.5,1.4]`, `hold ∈ [0.4,1.0]`, `ip ∈ [0.2,0.8]`, `cfg ∈ [4,10]`.
5. **Denoising loop** (`run_item`, `:211-248`), 20 Euler steps, batch of 2 (uncond, cond). At step fraction `f`:
   - `cs_t = cs · (1 if f < hold else RELAX)`; `ip_t = ip · ramp(f)` with ramp 0 until 15 %, linear to 1 at 40 % (`:196-197`).
   - `pipe.set_ip_adapter_scale(ip_t)`: one float for every cross-attention layer.
   - `pipe.controlnet(latents, t, text_embeds, sketch, conditioning_scale=cs_t)` → down/mid residuals.
   - `pipe.unet(latents, t, text_embeds, image_embeds=ipe, residuals)` → noise prediction.
   - `eps = eps_uncond + cfg · (eps_cond − eps_uncond)`.
   - Optional guidance (`guidance_delta`, `:199-209`), steps in 25–75 %, every 2nd step: predict x0, downsample the latent to 40×40, decode with the VAE (320 px), compute `L = λ_geo[(1−rec)+0.5·out] + λ_app[1−cos(CLIP(x), t_attr)]`, take the gradient w.r.t. x0 (through the VAE and CLIP, not through the UNet), normalise, subtract `GUIDE_W ×` it from the latent.
6. **Decode** final latent with the VAE.

Four arms (`ARMS`, `:253-256`): `fixed` (centre values, no schedule), `scheduled` (centre values + schedule), `router` (router values + schedule), `router_guided` (+ guidance).

**Router training** (notebook 04 cells 10-11): for 28 train items, 9 images each (centre + each factor at low/high), scored with Sobel-edge geometry + CLIP-L attribute similarity + CLIP-L object similarity, z-normalised and summed; per-factor soft-argmax target; MSE, Adam, 500 epochs; 5-fold CV.

---

## 3. Architecture and data flow

```
 caption ───────────────┐                sketch ─────────────┐            reference ────────────┐
   │                    │                  │ invert,         │              │ letterbox on grey │
   │                    │                  │ letterbox       │              │                   │
   ▼                    │                  ▼                 │              ▼                   │
 SDXL CLIP-L + bigG     │           scribble ControlNet      │      IP-Adapter CLIP ViT-H       │
 77 tokens + pooled     │           (copy of UNet encoder;   │      ONE global vector           │
   │                    │            also reads text tokens) │        → 4 image tokens          │
   │                    │                  │                 │              │                   │
   │ text cross-attn    │                  │ residuals added │              │ extra cross-attn  │
   │ (every layer)      │                  │ to UNet down+mid│              │ (every layer)     │
   │                    │                  │ × c_t           │              │ × ρ_t             │
   ▼                    │                  ▼                 │              ▼                   │
 ┌──────────────────────┼─ frozen SDXL UNet, 20 Euler steps, 640 px ────────────────────────────┼─┐
 │  eps = eps_uncond + w · (eps_cond − eps_uncond)             (w = guidance scale)             │ │
 └──────────────────────┼───────────────────────────┬──────────┼────────────────────────────────┼─┘
                        │                           │          │                                │
                        ▼                           ▼          ▼                                ▼
            CLIP ViT-L/14 text emb.        (optional guidance)   CLIP ViT-L/14 image emb. of sketch, reference
            of object phrase, caption,      x0 → VAE decode →            │
            attribute phrase                Sobel edges vs sketch mask   │
                        │                   + CLIP vs attribute phrase   │
                        └──────────► 6 cosines ◄─────────────────────────┘
                                         │
                                         ▼
                              Router MLP 6→32→32→4 (×5)      ← the only trained part
                                         │
                                 c, h, ρ, w   (4 scalars per image)
                                         │
                           schedule: c_t = c·[1 | RELAX after h],  ρ_t = ρ·ramp(15%→40%)
```

### Component roles and where they act

| Component | What it receives | Where it enters generation | Spatial? | Granularity of our control |
|---|---|---|---|---|
| SDXL text encoders | whole caption | UNet text cross-attention, every block, every step; pooled vector in time embedding | Tokens are attended per location, but the maps are learned and untouched | 1 scalar `w` (CFG) |
| Scribble ControlNet | sketch pixels + text tokens + noisy latent | Residuals added to UNet down/mid features | Yes, pixel-aligned | 1 scalar `c`, 1 switch time `h` |
| IP-Adapter (vit-h, non-"plus") | reference image | Separate cross-attention next to the text one, every block | No. The image is one pooled vector expanded to 4 tokens **[V code path, A on token count from the public checkpoint]** | 1 scalar `ρ` |
| CLIP ViT-L/14 | text phrases, sketch, reference, decoded x0 | Conflict features; guidance appearance loss; grid targets | No | n/a |
| Router MLP | 6 cosines | Sets `c, h, ρ, w` before denoising starts | No | per image |
| Schedule | step fraction | Scales `c`, `ρ` over time | No (temporal only) | hand-set shapes |
| Guidance | x0 estimate, sketch stroke mask, attribute phrase | Latent nudge in the middle 50 % of steps | Geometry term yes (stroke mask); appearance term no | weights follow `c`, `ρ` |
| DINOv2-small, CLIP-B/32, LPIPS, Canny | outputs | Evaluation only (and DINO for reference filtering) | n/a | n/a |

---

## 4. The professor's question

> How does the model understand which part of the input should govern which part of the generated image?

**Short answer: in the current system it does not, explicitly. Nothing in our code establishes it.**

Taking the example (text "a red car and a tree", sketch with car left and tree right, artistic reference):

1. **Are semantic objects associated with sketch regions?** No. **[V]**
   - The caption is never parsed into objects or attributes for generation. `cap_category` is one dataset class label used only for a CLIP phrase in the conflict features and in metrics (`app.py:185, 323`). There is no noun→token-index mapping anywhere.
   - The sketch is never segmented into regions or components. It is used whole, as (a) the ControlNet input, (b) a dilated stroke mask for the edge loss (`soft_sketch`, `:150-154`), (c) a bounding box of all strokes for the colour-histogram metric (notebook 04 cell 13).
   - No code reads, stores or edits cross-attention maps. A search of notebooks 02-04 and `app.py` for attention/mask/region/box/layout logic finds only the three uses above.
   - So "car ↔ left strokes, tree ↔ right strokes" would be decided entirely inside frozen pretrained weights: the ControlNet sees edges plus the full prompt, and SDXL's cross-attention decides which token paints where. That binding is implicit, unobserved and uncontrolled by us.

2. **What does ControlNet actually control?** Where edges/structure appear, pixel-aligned, class-agnostic. It carries no label saying which strokes belong to which word. With two objects it constrains both outlines but not which noun fills which outline (attribute/identity swapping between regions is possible and we have no mechanism or metric for it).

3. **What does IP-Adapter actually control?** It injects one global image embedding into every cross-attention layer with one scalar. That embedding mixes the reference's object identity, layout, colour and style. It cannot be told "style only" or "apply to the car only" in our pipeline. Evidence it carries content, not just appearance: `text_ref` baseline reaches DINO-to-reference 0.673 while DINO-to-target falls to 0.105 **[D]** (report Table 2). Notably, the *data-prep* notebooks do use a style-only block restriction (`set_ip_adapter_scale({'up': {'block_0': [0.0, s, 0.0]}})` in `sketchy_reference_v2`), but the main pipeline uses a plain float on all blocks **[V]**.

4. **Explicit or implicit?**
   - The role assignment "sketch→geometry, text→semantics, reference→appearance" exists as (a) a naming convention for the four knobs, (b) metadata strings `geometry_src / semantics_src / appearance_src` and `expected_resolution` written in notebooks 01 and 01b that no model code ever reads **[V]**, and (c) the choice of adapter per modality.
   - The router decides **how much** of each modality, globally, per image. It never decides **where** or **to which object/attribute**.
   - The schedule decides **when**. Still not where.
   - The only spatially explicit constraint we add is the geometry guidance term, and it is class-agnostic.

5. **Limitations this exposes**
   - The dataset is single-object (Sketchy: one object per sketch) and the negative prompt literally contains "multiple objects" (`app.py:41`), so the professor's multi-object case cannot be expressed, is actively suppressed, and has never been tested.
   - The knobs are entangled, so even the modality→attribute story is loose: ControlNet takes the text embeddings as input; and because the unconditional branch gets a zero image embedding, the CFG scale `w` amplifies text **and** reference together, not text alone **[V in code that both branches share `ipe`; A on the zero-negative, from diffusers' standard behaviour]**.
   - No part-level control either: "pink handle" versus a blue reference is a part-level attribute conflict, but we can only turn the whole reference up or down.
   - The conflict detector cannot see attribute conflicts (report Table 6: colour 0.610, material 0.541, ref-swap 0.597 vs normal 0.604) and the router's outputs barely vary (report Table 7), so in practice the system behaves like the hand schedule.

**What the concern reveals, precisely:** the project claims modality-to-attribute control, but implements modality-level *gain* control. The correspondence problem has three levels and we address none of them explicitly: (i) text token ↔ image region, (ii) sketch region ↔ text entity, (iii) reference attribute (style/colour/material) ↔ target entity or region, disentangled from the reference's own content. Our four scalars and a time schedule are global over the image, so they cannot represent any of these bindings.

---

## 5. Implemented / partial / proposed only

**Implemented and run [V]**
- Data: Sketchy pairs CSV, 200 Qwen3-VL captions, reference images v2, frozen `pilot_v1` (200 samples; 160/20/20), 800 conflict rows (4 kinds) → 1000-row metadata.
- SDXL + ControlNet + IP-Adapter baseline, 4 modes (notebook 02).
- Custom denoising loop with per-step `c`, `ρ` and per-item `w`.
- 6-feature CLIP conflict detector.
- Gate (2 outputs, notebook 03) and router (4 outputs, notebook 04) with grid targets and 5-fold CV.
- Constraint guidance (edge + CLIP attribute loss).
- Held-out metrics, paired bootstrap, unified re-scoring (`final_output_eval`).
- Gradio app.

**Partially implemented**
- Learned routing: trained, but beats a constant predictor on only 1 of 4 outputs clearly (cs 0.083 vs 0.122; hold tied; ip and cfg worse) **[V saved output, cell 11]**.
- Conflict detection: separates object conflicts only.
- `src/eval_metrics.py`: layout IoU (CLIPSeg), DreamSim, FID are coded but not part of the reported tables; its docstring lists AP, GPT-4o judge and conflict-set scoring as not done.
- Reference quality filtering: computed, then bypassed (see discrepancy 3).
- FLUX/UniCombine: notebooks and loader exist; never trained.

**Proposed only (deck/report, no code)**
- Routing layer inside the attention; per-block/per-head gating.
- VLM attribute-level conflict detector; factorial grid; schedule ablation; guidance re-tuning; larger eval; human study; sketch-vs-reference conflicts.
- Any region-, object- or token-level correspondence.

---

## 6. Discrepancies between documents and code (please confirm which is right)

1. **Sketch relax factor.** Deck p.8 and report eq. 2 say the sketch drops to **0.7c** after the hold time. Code has `RELAX = 0.3` (`app.py:54`, notebook 04 cell 1) and notebook 03 hard-codes `0.3` in `cs_shape`.
2. **Guidance weight.** Deck p.9 and report say **0.03**. Code has `GUIDE_W = 0.25` in notebooks 03, 04 and `app.py:58`.
   **Resolved (confirmed by the team): the reported runs used the code values, `RELAX = 0.3` and `GUIDE_W = 0.25`. The deck (p.8, p.9) and report (eq. 2, §4.2, E5) are wrong and need correcting.** Consequences: the sketch is cut to 30 % (not 70 %) after the hold time, which fits the scheduled arm's large precision drop (−0.23 to −0.29); and guidance is about 8× stronger than documented, which fits the artefacts and precision loss in the guided arms.
3. **Pilot filtering.** README and the freeze notebook's text say only filter-passing, well-segmented samples are kept. The code sets `df['keep'] = True` with the filter lines commented out (notebook 01 cell 9); saved output shows 200/200 kept, including 18 that failed the filters and 23 with unverified segmentation.
4. **What counts as success on object conflicts.** Notebook 01b's `expected_resolution` says "keep the sketch outline; do not turn the object into the one named in the caption." The `sem_margin` metric and report finding 1 treat moving toward the caption's object as the improvement. The intended resolution policy is undefined.
5. **App versus training features.** The app builds the attribute phrase by CLIP zero-shot over fixed word lists and, if no object name is typed, uses the whole caption as the "object" (`app.py:320, 323`); training used VLM attributes and dataset class labels. Router inputs at demo time are off-distribution.
6. **Item selection** takes the first k rows per kind (`make_items`, `[:k]`), so the 30 test items come from 12 base photos and the per-kind examples in saved output are all the same hammer.
7. **Statistics.** Notebook 04 bootstraps over items with 2000 resamples; the report's base-photo bootstrap with 5000 resamples is in `final_output_eval` (I read only its first cells; not fully traced).
8. Notebook 04's saved cell-2 output says profile "fast" while later cells show the full run (252 grid, 240 eval images); the saved notebook is a mix of runs.

---

## 7. Open technical questions for Stage 2 (questions only, no design)

**Correspondence**
1. Which of the three bindings does the project want to claim: token↔region, sketch-region↔entity, reference-attribute↔entity? All three, or one done properly?
2. In the frozen stack, what do SDXL's text cross-attention maps look like under ControlNet for a two-object prompt? Do noun tokens already localise onto the matching sketch strokes, and how often do they swap? (We have never measured this.)
3. Does the scribble ControlNet's own use of the text tokens help or hurt binding when text and sketch disagree?
4. What does the non-plus IP-Adapter embedding actually encode for our references: how much is colour/material and how much is object identity/layout? Does restricting it to particular blocks, or using patch-level variants, change that?
5. Can a reference's influence be restricted spatially (to a sketch region) or semantically (to one entity) without retraining, and what does the literature already do here (regional prompting, attention masking, layout-grounded and multi-subject methods)? Which of those fit a free T4 with frozen SDXL?
6. How would sketch regions be obtained: user-labelled strokes, connected components, a segmentation of the sketch, or inferred from attention?

**Data and evaluation**
7. Sketchy is single-object. What multi-object sketch data (or synthetic composition of Sketchy sketches) is needed to even pose the professor's example?
8. What metric shows correct binding (right object in right region, right attribute on right object)? Our current metrics are global or whole-sketch.
9. What is the intended resolution policy for each conflict type (discrepancy 4)?

**Existing mechanism**
10. (Answered: code values, `RELAX = 0.3`, `GUIDE_W = 0.25`.) Follow-up: were the report's conclusions about the schedule and guidance drawn with 0.7 / 0.03 in mind, and do the planned "lower guidance weight" and "ablate the schedule" items change now that the true settings are known?
11. Is CFG scale really a "text" knob given it also scales the reference? Does that explain the router learning nothing for `cfg` and `ip`?
12. Is a per-image global router worth keeping at all, given it is statistically indistinguishable from the schedule, or is the 4-scalar control space itself the ceiling?
13. Is the guidance appearance term (global CLIP vs a text phrase) meaningful as "reference appearance", when it never looks at the reference image?
