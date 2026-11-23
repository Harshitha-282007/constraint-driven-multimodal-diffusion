# Stage 4: final research direction and project scope

> **Superseded in part.** After the official project brief was reviewed, sections 2–6 were replaced by `NOVELTY_ASSESSMENT_PLAN.md` (single object, a trained governance network instead of a hand-written policy). Section 1 (what the papers showed) still stands.

Consolidates `STAGE1_CURRENT_STATE.md`, `STAGE2_LITERATURE_REVIEW.md` and `STAGE3_NOVELTY_AND_REDESIGN.md`.

**Labels used throughout**
- **[Read]** I read the method section of the original paper for this stage.
- **[Abstract]** I know it from the abstract only.
- **[Ours]** proposed by us; not implemented, not tested.
- **[Hypothesis]** an expectation that an experiment below is designed to test.

Nothing in this document has been run.

---

## 1. What reading the papers changed

Stage 3 recommended "the sketch as a shared spatial index" (recover entity regions from unlabelled strokes, then bind text and reference to them) as the core. After reading the closest papers, I am **demoting that from main contribution to supporting component**, and promoting the conflict side.

| Paper | What the full text shows | Effect on our plan |
|---|---|---|
| [T³-S2S](https://arxiv.org/abs/2412.13486) **[Read]** | Baseline is SDXL + the same xinsir ControlNet we use. Each instance keyword is paired with **its own instance sketch**, supplied as input. Three training-free modules: prompt-embedding energy balance, TopK channel enhancement inside each instance's sketch region, and DenseDiffusion-style tuning. They report that applying DenseDiffusion directly with ControlNet **degrades** results, and fix it by applying it only in the ControlNet branch (`down_blocks.2`, `mid_block`) with the "expand" term only. Ran at 1024 px on one V100. No reference image, no conflicts. | Sketch-region ↔ text binding on our exact backbone is done. It also tells us how to do the attention edit safely. |
| [Subjective Camera 1.0](https://arxiv.org/abs/2506.23711) **[Read]** | Input is text plus a **sequence of sketches, one per concept**. Optimisation-based (gradients on the latent). Its CMC dataset is 142 scenes of 2–4 Sketchy sketches composed on a canvas. | Composing Sketchy sketches into multi-object scenes is not our idea. Per-concept sketches as input is an accepted assumption. |
| [SketchingReality](https://arxiv.org/abs/2602.14648) **[Read]** | Uses a pretrained open-vocabulary sketch segmentation encoder to get per-concept spatial signals from an unlabelled freehand scene sketch, and trains with a loss aligning text cross-attention to them. Trained. | "Unlabelled strokes → entities → attention" exists in trained form. Our Stage 3 headline claim is not safe. |
| [Griffin](https://arxiv.org/abs/2509.23643) **[Read]** | Layout masks/boxes and one source image per component are given. Masked IP-Adapter for initialisation, then masked self-attention sharing to copy identity. States it cannot do text-based stylisation or mix styles. No sketch. | Reference ↔ region binding with masked IP-Adapter is done. It copies identity; it does not select attributes. |
| [SmartControl](https://arxiv.org/abs/2404.06451) **[Read]** | A 3-layer conv predictor per ControlNet decoder block outputs a pixel-wise scale map from UNet and UNet+ControlNet features. Trained on 1,000–2,000 text–condition pairs per condition type, built by sweeping the global scale and hand-filtering. Conflict type: **object class only** (deer depth map, "tiger" prompt). Policy is fixed: text decides the object. | Local relaxation of a spatial condition under object conflict is done, with training. No reference modality, no attribute conflicts, no choice of policy. |
| [SADis](https://arxiv.org/abs/2503.14275) **[Read]** | On SDXL + IP-Adapter, tuning-free. Colour embedding = embedding(image) − embedding(greyscale image); texture embedding = embedding(greyscale texture image) with SVD rescaling. The user supplies one colour reference and one texture reference. | Separating colour from texture in IP-Adapter's embedding space is done and reusable. Roles are assigned by the user; nothing is detected. |
| [FiVA](https://arxiv.org/abs/2412.07674) **[Read: method]** | Trained adapter; the user names which attribute to take from each reference. No conflict handling. | Same: selection by the user, not detection or resolution. |

Not read in full, so not relied on: [MATTE](https://arxiv.org/abs/2311.11919), [IPAdapter-Instruct](https://arxiv.org/abs/2408.03209), [Text-Anchored Score Composition](https://arxiv.org/abs/2306.14408), Cross-ControlNet, DynFusion, AtteConDA. These must be read before the final report; the first two are the most likely to overlap.

**What is established:** region binding of text (given regions), region binding of references (given regions), per-instance sketches on SDXL + ControlNet, local relaxation of a spatial condition under object conflict, and colour/texture separation of a reference.

**What none of the papers I read does:** take text, sketch and reference together, **detect** which properties they disagree on, and **resolve** each disagreement by a stated, switchable rule. In every case the governing modality is either fixed by design (SmartControl: text wins the object) or chosen by the user per input (SADis, FiVA, Griffin).

---

## 2. Final problem and novelty

**Problem statement.** Given a caption, a sketch and a reference image that may disagree, generate an image in which each visual property (shape, object identity, colour, material) of each entity is determined by one declared modality, and show which modality that was.

**Research gap.** Existing work either binds conditions to places, or lets the user pick what a reference contributes. Cross-modal disagreement at the level of a named attribute is not detected, and the choice of winner is not an explicit, inspectable object. (Supported by the papers marked [Read]; pending the six unread ones.)

**Main contribution [Ours].** A **governance map**: a small explicit table, built from the inputs, that states for every (entity, attribute) which modality governs it, with conflicts detected by comparing typed attribute records and resolved by a switchable policy. The map is then compiled into settings of a frozen SDXL + ControlNet + IP-Adapter pipeline, training-free.

This replaces Stage 1's router. The router mapped six global similarity numbers to four global strengths. The map is typed, per entity, and readable.

**Supporting contributions [Ours]**
1. A typed conflict detector (symbolic comparison of attribute records) in place of the six CLIP cosines, which Stage 1 showed cannot separate colour or material conflicts.
2. A **policy-adherence** evaluation: for each attribute, did the output follow the modality the map named. Includes the policy-switch test (same inputs, different policy, output changes accordingly).
3. An extension of our labelled conflict set to two-entity scenes with per-entity conflicts.

**What is reused, not claimed:** regional text attention (DenseDiffusion, applied the T³-S2S way), masked and block-wise IP-Adapter (Griffin, InstantStyle), colour/texture embedding separation (SADis), the idea of locally relaxing ControlNet (SmartControl; ours is rule-based, not learned), per-entity sketch layers as input (T³-S2S, Subjective Camera).

**How it differs**

| | Regions | Reference contributes | Conflict detected | Who wins | Training |
|---|---|---|---|---|---|
| T³-S2S | per-instance sketches | n/a | no | n/a | none |
| Griffin | given masks | identity, copied | no | n/a | none |
| SmartControl | learned map | n/a | object, implicitly | text, fixed | small predictor |
| SADis / FiVA | global | attribute chosen by user | no | user decides per input | none / large |
| Our Stage 1 system | none | everything, one scalar | 6 cosines (object only) | implicit | tiny MLP |
| **Proposed** | per-entity sketch layers | attribute set by the map | **typed, per entity** | **declared policy, switchable** | none |

**What we intend to show**
- The typed detector finds colour and material conflicts that the cosine detector misses.
- Outputs follow the declared policy more often than the fixed, scheduled and router arms do.
- Switching the policy switches the measured attribute.
- In two-entity scenes, an attribute assigned to one entity does not leak to the other.
- No loss on conflict-free items.

**What we are not claiming:** a new binding mechanism, a new adapter, automatic understanding of unlabelled strokes, or better image quality.

---

## 3. Architecture

```
 INPUTS
 caption                     sketch layers s_1..s_K                reference image(s)
   │                         (one per entity; K=1 for             (one global, or one per entity)
   │                          the current dataset)
   ▼                               ▼                                    ▼
 ┌───────────────────────────────────────────────────────────────────────────────────────┐
 │ 1. TYPED PARSER  [Ours]                                                               │
 │   text record  T_k = {object, colour words, material words}   rules + word lists      │
 │   sketch record S_k = {object guess, region mask R_k}         CLIP zero-shot, fill    │
 │   ref record    F_k = {colours, material, finish}             Qwen3-VL (already used) │
 └───────────────────────────────────────────────────────────────────────────────────────┘
                                         ▼
 ┌───────────────────────────────────────────────────────────────────────────────────────┐
 │ 2. CONFLICT TABLE  [Ours]    per entity k, attribute a ∈ {object, colour, material}   │
 │   object:   T_k.object  vs S_k.object                                                 │
 │   colour:   T_k.colour  vs F_k.colour     (only if the text states one)               │
 │   material: T_k.material vs F_k.material  (only if the text states one)               │
 └───────────────────────────────────────────────────────────────────────────────────────┘
                                         ▼
 ┌───────────────────────────────────────────────────────────────────────────────────────┐
 │ 3. POLICY → GOVERNANCE MAP  G[k][a] ∈ {sketch, text, reference}   [Ours]              │
 │   P0 (default): shape←sketch, object←text, colour←reference, material←reference       │
 │   P1:           colour←text, rest as P0                                               │
 │   P2:           object←sketch, rest as P0                                             │
 └───────────────────────────────────────────────────────────────────────────────────────┘
                                         ▼
 ┌───────────────────────────────────────────────────────────────────────────────────────┐
 │ 4. COMPILER: map → pipeline settings  [Ours; each mechanism below is established]     │
 │                                                                                       │
 │  TEXT SIDE              REFERENCE SIDE                       SKETCH SIDE              │
 │  rewrite entity phrase: split IP embedding (SADis):          ControlNet residuals ×   │
 │  drop attribute words   colour = E(ref) − E(grey ref)        spatial map m(x,y):      │
 │  text does not govern;  texture = E(grey ref)                1 where sketch governs   │
 │  keep the ones it does  inject only governed part,           shape; lowered inside    │
 │                         style blocks only (InstantStyle),    R_k after the hold time  │
 │  K>1: bias entity       masked to R_k (ip_adapter_masks)     if object conflicts and  │
 │  tokens to R_k                                               text governs object      │
 │  (DenseDiffusion via                                                                  │
 │   T³-S2S placement)                                                                   │
 └───────────────────────────────────────────────────────────────────────────────────────┘
                                         ▼
        existing custom Euler loop (run_item): frozen SDXL UNet + ControlNet + IP-Adapter
                                         ▼
                        image  +  the governance map shown to the user
```

### Inputs and preprocessing
- **Sketch.** One layer per entity. For the existing single-object data, the whole sketch is the one layer. For multi-entity scenes, layers come from composition (ground truth) or from the user drawing entities separately. The union, white on black and letterboxed, is the ControlNet input, as now. Region mask `R_k` = the layer's strokes closed, filled and dilated.
- **Caption.** Split into entity phrases by noun chunks; colour and material words found with the word lists already in notebook 01b.
- **Reference.** Attributes from Qwen3-VL, as in `sketchy_reference_v2`. For the dataset they are already in the metadata (`ref_colors`, `ref_material`, `ref_finish`).

### Role of each pretrained model (all frozen)
| Model | Role now | Role in the proposal |
|---|---|---|
| SDXL UNet + VAE | generator | unchanged |
| Scribble ControlNet | sketch → residuals × scalar | residuals × spatial map |
| IP-Adapter (vit-h) + CLIP ViT-H | reference → all blocks × scalar | governed component only, style blocks, masked per region |
| SDXL text encoders | caption | rewritten entity phrases |
| CLIP ViT-L/14 | six conflict cosines, guidance loss | sketch object guess only |
| Qwen3-VL-2B | data preparation | reference (and optionally caption) attribute records |
| CLIP-B/32, DINOv2, LPIPS, Canny | evaluation | evaluation, unchanged |

### What is dropped
The router MLP, the six-cosine detector and the gradient-based constraint guidance leave the main method and become baselines. Stage 1 found the router indistinguishable from the schedule, and the guidance traded colour match for precision.

### How this answers the professor
For "a red car and a tree; car sketched left, tree right; artistic reference":
- the car layer and the tree layer give two regions, each tied to its phrase;
- the map says: shape of both ← sketch; object of each ← its phrase; colour of the car ← text ("red" is stated); colour of the tree ← reference; texture/style of both ← reference;
- each line of the map corresponds to one setting in the compiler, and the map is displayed beside the output.

The correspondence is therefore an explicit data structure, not something assumed to happen inside the network.

### Integration with the existing code
`run_item` in `app/app.py` already calls ControlNet and the UNet separately at every step, so:
- scaling residuals by a map is a multiplication of the `down`/`mid` tensors after resizing the map;
- block-wise IP scales are a dict passed to `set_ip_adapter_scale` (already done in the data-prep notebooks);
- region masks go through `cross_attention_kwargs={"ip_adapter_masks": ...}` (documented in diffusers; a wrong key is silently ignored, so it needs a visible test);
- text-side binding needs custom attention processors and is only required for K>1.

Lightweight training: none planned. The one place it could be argued for is learning the ControlNet map as SmartControl does; we do not have the data or memory budget, and a rule-based map is enough to test the idea.

---

## 4. Experimental plan

| # | Experiment | Validates | GPU |
|---|---|---|---|
| E1 | Typed detector vs six-cosine detector on the labelled conflict rows: precision/recall per conflict type | Supporting contribution 1 | none |
| E2 | Single-entity generation on the test split: all arms, per-attribute adherence | Main contribution | T4 |
| E3 | Policy switch: same inputs under P0, P1, P2 | Main contribution (the map controls the output) | T4 |
| E4 | Two-entity composed scenes: per-region adherence and cross-entity leakage | Per-entity governance; the professor's case | T4 |
| E5 | Ablations | Which compiler part matters | T4 |
| E6 | Conflict-free items: fidelity to the target photo | No harm when nothing conflicts | T4 |

**Baselines**
1. Fixed weights (existing arm).
2. Scheduled (existing arm).
3. Router (existing arm; images already on Drive for 12 base photos).
4. Block-wise IP-Adapter only (InstantStyle setting), no map.
5. Prompt rewriting only: the map changes the text, nothing else.
6. Binding only (E4): masked IP-Adapter + regional text attention with no conflict handling.
7. If time allows: T³-S2S from its public code for E4.

**Ablations (E5)**
- Remove one compiler part at a time: prompt rewrite; embedding split; block restriction; ControlNet map.
- Spatial ControlNet map vs the same reduction applied globally.
- Typed detector vs oracle conflict labels (how much is lost to detection errors).
- Region assignment given vs assigned by CLIP on each layer (stretch).

**Data**
- Existing: `pilot_v1` and its 800 conflict rows. Use all 20 test base photos, not 12, and restrict to references that passed the quality filters (Stage 1 found all 200 were kept).
- Detector test set: the conflict rows were made with the same word lists the detector will use, so E1 on them alone is circular. Add a set of captions paraphrased by the VLM or written by hand (about 50) where the conflicting attribute is phrased differently.
- New: about 40 two-entity scenes composed from Sketchy test sketches, with one conflict on one entity. Precedent: Subjective Camera's CMC set (check whether it is public before building ours).
- Qualitative: the professor's car/tree/style example.

**Metrics**
- *Colour adherence:* map the region's pixels to the nearest of the 14 named colours (deterministic, in a perceptual colour space); compare with the governing source's colour. Not CLIP-based.
- *Material adherence:* CLIP-B/32 zero-shot over the 8 materials on the region crop, plus a VLM judgement. This is the weakest metric; report it as secondary.
- *Object adherence:* existing `sem_text` vs `sem_sketch` margin, per region.
- *Shape adherence:* existing edge recall/precision, per region.
- *Policy-adherence rate:* share of (item, attribute) pairs where the output matches the governing modality. *Leak rate:* share where it matches the non-governing one.
- *Fidelity:* existing LPIPS and DINO to the target on conflict-free items.
- *Statistics:* existing paired bootstrap over base photos.

**Expected results [Hypothesis]**
- E1: the typed detector separates colour and material conflicts; the cosine detector does not (already seen in Stage 1 Table 6).
- E2/E3: colour adherence is clearly above the existing arms and flips with the policy. Material adherence improves less and is noisier.
- Object conflicts under "text governs object" improve only with the ControlNet map; prompt rewriting alone will not overcome the sketch.
- E4: masked reference injection reduces leakage; regional text attention is the least certain part.
- E6: no significant drop.

Any of these may fail. A negative result on material, or on regional text attention, is reportable as long as E1–E3 hold.

---

## 5. Implementation roadmap

Ordered so that M0–M3 alone are a complete, presentable result.

| Milestone | Work | Reuse / modify / new | Output | Risk |
|---|---|---|---|---|
| **M0** Housekeeping | Correct RELAX 0.3 and GUIDE_W 0.25 in deck and report; rewrite 01b's `expected_resolution` for object conflicts to "text decides the object"; select filter-passing rows | modify docs | consistent baseline description | none |
| **M1** Parser and conflict table | Attribute records, comparison, governance map; paraphrased test set; run E1 | new (CPU only); reuse 01b word lists and metadata | E1 table | circularity if the paraphrased set is skipped |
| **M2** Compiler, single entity | Prompt rewrite; block-wise IP scales; embedding split; ControlNet residual map | modify `run_item`; reuse data-prep IP code | a `run_item` that accepts a governance map | embedding split is shown for SDXL IP-Adapter in SADis, but our checkpoint variant may behave differently |
| **M3** E2, E3, E6 | Roughly 100 items × 6–7 arms at 20 steps, 640 px; new metrics | reuse eval and bootstrap; new adherence metrics | main results table, policy-switch figure | a few GPU hours on a T4; resumable as now |
| **M4** Two-entity scenes | Compose data; per-entity masks; masked IP-Adapter | new data script; modify loop | E4 without text binding | `ip_adapter_masks` silently ignored if misused |
| **M5** Regional text attention | Custom attention processors, T³-S2S placement | new | full E4 | highest: fragile, and must coexist with IP-Adapter processors |
| **M6** Ablations and interface | E5; Gradio shows the map and lets the user switch policy | modify `app.py` | final figures, demo | time |

Dependencies: M2 needs M1; M3 needs M2; M4 needs M2; M5 needs M4. M5 can be cut without losing the main claim.

Memory: no step adds gradients through any network, so peak memory should stay near the 10.8 GB measured for the router run. This is an estimate until M2 runs.

---

## 6. Presentation outline (novelty assessment)

1. **Recap in one slide.** What we built: frozen SDXL + ControlNet + IP-Adapter with a router setting four global strengths.
2. **The professor's question, and our honest answer.** Nothing in the old system linked parts of the inputs to parts of the output. Evidence from the code.
3. **What the old results already told us.** Router ≈ schedule; the detector cannot see colour or material conflicts.
4. **What exists.** One table: T³-S2S, Griffin, SmartControl, SADis/FiVA. Each solves a piece; who decides the winner in each.
5. **The gap.** Conflicts are not detected by type, and the winner is never a declared, switchable rule across three modalities.
6. **Our contribution.** The governance map. One worked example (the car/tree/style case) shown as a table.
7. **Architecture.** Parser → conflict table → policy → compiler → frozen pipeline. Mark every reused mechanism with its source.
8. **How we will prove it.** E1–E6, with the policy-switch test as the centrepiece.
9. **Plan and risks.** M0–M6, what is cut first, what counts as a negative result.
10. **What we do not claim.** No new adapter, no stroke understanding, no quality gain.

---

## 7. Still to settle

1. Read MATTE and IPAdapter-Instruct in full; if either detects and resolves text–reference conflicts, the gap statement in section 2 must be narrowed.
2. (Settled by the team: **text wins object conflicts by default.**) This matches P0 as written, SmartControl's assumption and the existing `sem_margin` metric. It contradicts notebook 01b's `expected_resolution` for `object_text_sketch` ("keep the sketch outline; do not turn the object into the one named in the caption"), which must be rewritten in M0. "Sketch wins" stays available as policy P2 for the policy-switch experiment.
3. One global reference or one per entity for E4.
4. How much time is left; this decides whether M4–M5 are in scope.
5. Whether Subjective Camera's CMC dataset is available, to avoid rebuilding it.
