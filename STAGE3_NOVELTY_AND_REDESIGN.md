# Stage 3: novelty reassessment and redesign options

Builds on `STAGE1_CURRENT_STATE.md` (what we built) and `STAGE2_LITERATURE_REVIEW.md` (what exists).

**How far to trust this document**
- New arXiv links were checked against arXiv (ID resolves to the stated title). I read abstracts, not full papers. Every "closest work" below must be read in full before a novelty claim goes into a report.
- "Established" means published by others. "Proposed" means our idea, unverified and unimplemented.
- Nothing here has been run. Feasibility statements are engineering estimates, not measurements.

---

## 1. Re-checking the Stage 2 gaps

I ran further searches against each gap. Three of them are weaker than Stage 2 suggested.

| Stage 2 gap | New closest work found | Verdict |
|---|---|---|
| Unlabelled sketch strokes ↔ text entities | [Open Vocabulary Semantic Scene Sketch Understanding](https://arxiv.org/abs/2312.12463) (segments freehand scene sketches by category from captions); [SketchingReality](https://arxiv.org/abs/2602.14648) (2026, freehand scene sketch + text → photo, trained); [Training-Free Sketch-Guided Diffusion with Latent Optimization](https://arxiv.org/abs/2409.00313) (uses cross-attention maps to follow a sketch); [Subjective Camera 1.0](https://arxiv.org/abs/2506.23711) (text + progressively drawn sketches, training-free); [T³-S2S](https://arxiv.org/abs/2412.13486) | **Partly addressed.** Sketch understanding and sketch+text scene generation both exist. I did not find one that also binds a reference image per entity, but that is a narrower claim than "unaddressed". |
| Reference attribute vs text attribute | [FiVA](https://arxiv.org/abs/2412.07674) (adapter that takes a chosen attribute, e.g. colour, lighting, texture, from a reference; trained on ~1M images); [MATTE: multi-attribute inversion](https://arxiv.org/abs/2311.11919) (colour, object, layout, style from one reference, mixed with text); [IPAdapter-Instruct](https://arxiv.org/abs/2408.03209) (an instruction says which aspect of the image to use); [SADis: colour–texture disentanglement](https://arxiv.org/abs/2503.14275) (tuning-free); [ColorWave](https://arxiv.org/abs/2503.09864) (training-free colour control through IP-Adapter's colour binding); [ProSpect](https://arxiv.org/abs/2305.16225); [ZeST](https://arxiv.org/abs/2404.06425) (material transfer) | **Largely addressed** as "take only attribute X from the reference". Stage 2's claim that reference methods do not transfer named attributes was wrong. What I still did not find: an explicit *conflict* treatment (text says pink, reference says blue: detect, decide, enforce). [Conflict-free guidance](https://arxiv.org/abs/2508.02004) addresses a different conflict (inside the guidance signal). |
| Reference ↔ region | [Griffin](https://arxiv.org/abs/2509.23643) (training-free, layout + one image per component, uses IP-Adapter; reports part-level leakage); [ROICtrl](https://arxiv.org/abs/2411.17949) (trained regional control, compatible with ControlNet and IP-Adapter); [LCP-Diffusion](https://arxiv.org/abs/2505.20909); [MS-Diffusion](https://arxiv.org/abs/2406.07209) | **Addressed** for boxes. Not new. |
| Three-modality conflict benchmark | [MMIG-Bench](https://arxiv.org/abs/2505.19415) evaluates multi-modal image generation, apparently without deliberate conflicts. Conflict benchmarks I found are for vision-language QA models, not generation. | **Still open as far as I can tell**, on one search. |
| Conflict as a local quantity | [SmartControl](https://arxiv.org/abs/2404.06451): text vs one spatial condition, learned pixel map (3-layer conv predictor, about 1–2k training samples). | Addressed for text vs spatial. Not found for the reference side. |

Consequence: no single mechanism we could add is new on its own. Region binding, attribute-selective reference transfer, block-wise IP-Adapter and local control scales are all published. A defensible contribution has to be (a) a specific problem formulation others have not evaluated, with (b) a mechanism that is more than switching on existing modules, and (c) evidence.

---

## 2. Candidate directions

### Direction 1: sketch-grounded binding (the sketch as the shared spatial index)

- **Contribution (proposed):** from an unlabelled freehand sketch, recover entity regions and assign each to a caption entity; then use those regions to drive *both* text binding and reference binding inside a frozen SDXL.
- **Closest work:** T³-S2S (sketch + text, multi-instance, SDXL + ControlNet, no reference); Griffin (layout + references, training-free, layout is given); DenseDiffusion and Bounded Attention (masks or boxes given); SketchingReality (trained); Open-vocabulary sketch segmentation (understanding only).
- **What is different:** the regions are not given by the user. They come from the strokes, and one grounding step feeds two binding paths (text cross-attention and image cross-attention).
- **Already addressed?** Each half is. The joint use on unlabelled strokes with a per-entity reference I did not find. That is a narrow gap and could close on reading Griffin, T³-S2S and Subjective Camera.
- **To establish novelty we would need:** (i) stroke→entity assignment accuracy on data with ground truth; (ii) binding accuracy (right object in right region, right appearance on right object) against the plain stack and against "regions given by oracle"; (iii) an ablation showing each binding path matters.

### Direction 2: typed conflict resolution (which modality governs which property, stated and enforced)

- **Contribution (proposed):** detect conflicts at the level of (entity, attribute type) rather than with global CLIP cosines; resolve them by an explicit policy; enforce the decision through the mechanism that actually controls that attribute.
- **Closest work:** SmartControl (text vs spatial, learned); FiVA, MATTE, IPAdapter-Instruct, SADis (choose what the reference contributes, but the user chooses, nothing is detected); Text-Anchored Score Composition; our own router.
- **What is different:** a conflict is a typed object ("colour of the hammer handle: text says pink, reference says blue"), the resolution is a readable rule, and it covers three modalities.
- **Already addressed?** The enforcement tools are. Detection plus policy across text, sketch and reference I did not find. Risk: a reviewer may see it as prompt rewriting plus knob selection. Technical depth depends on showing the enforcement is attribute-specific and measurable.
- **To establish novelty:** (i) detector precision/recall on our labelled conflict set, against the current 6-cosine detector, which Stage 1 showed cannot see attribute conflicts; (ii) per-attribute "policy adherence" rate: did the output follow the modality the policy named; (iii) proof that changing the policy changes the output accordingly.

### Direction 3: a measurement study and benchmark of three-modality conflicts

- **Contribution (proposed):** a labelled benchmark of typed conflicts across text, sketch and reference, binding/adherence metrics, and a study of which modality wins per attribute in the frozen stack and in training-free variants.
- **Closest work:** Uni-ControlNet's conflict analysis; Cross-ControlNet's conflicting-condition evaluation (spatial only); MMIG-Bench; MATTE's evaluation; T2I-CompBench++.
- **What is different:** deliberate, typed, three-way conflicts with a stated expected resolution.
- **Already addressed?** Not that I found. It is an empirical contribution, not an architecture.
- **To establish it:** scale beyond 12 base photos, clean references (Stage 1 found the quality filters were bypassed), and validated metrics.

### Direction 4 (considered, not shortlisted): learned spatial gates for all three modalities

A SmartControl-style predictor that outputs per-pixel maps for ControlNet and IP-Adapter strength. Highest technical depth, but training needs gradients through the SDXL UNet. Our FLUX attempt died of memory at step one, and Stage 1 showed our learned router did not beat a constant. I would not build the project on this.

---

## 3. Architecture alternatives

Shared base in all three: frozen SDXL UNet, frozen scribble ControlNet, frozen IP-Adapter, our existing custom denoising loop (`run_item`). Nothing below trains the backbone.

### Architecture A: region-bound conditioning (implements Direction 1)

```
 caption ──► entity parser (VLM or rules) ──► entities e1..eK with attribute words
                                                    │
 sketch ──► stroke grouping ──► components s1..sM   │
              (connected components / proximity)    │
                         │                          │
                         └──► assignment  ◄─────────┘
                              components ↔ entities          [PROPOSED]
                              (score matrix + matching)
                                     │
                              region masks R1..RK (filled, dilated)
                                     │
      ┌──────────────────────────────┼─────────────────────────────────┐
      ▼                              ▼                                 ▼
 scribble ControlNet          text cross-attention              image cross-attention
 (whole sketch, as now)       modulated so tokens of e_k        reference r_k masked to R_k,
                              attend inside R_k                 style blocks only
 [established]                [established: DenseDiffusion]     [established: IP-Adapter masks,
                                                                 InstantStyle]
      └──────────────────────────────┬─────────────────────────────────┘
                                     ▼
                         frozen SDXL UNet, Euler loop ──► image
```

- **Three modalities:** sketch does two jobs (ControlNet input, and source of regions). Text is split into entity token spans. Each reference is attached to one entity (one reference for all entities is the degenerate case).
- **Correspondence:** explicit, in the assignment step. Two candidate scorers, both proposed and untested:
  1. *Appearance of the strokes:* crop each component, compare with each entity name using CLIP (we already compute sketch–text CLIP similarity; Stage 1 showed it separates object conflicts, 0.749 vs 0.832).
  2. *Read-then-enforce:* run a few early denoising steps with ControlNet, read each noun's cross-attention map, and score components by attention mass. This uses the model's own implicit binding, makes it visible, then locks it in.
- **Interaction:** conditions no longer compete globally; each acts inside its region. Self-attention can also be restricted between regions (DenseDiffusion, Bounded Attention) to reduce leakage.
- **Conflict:** an unmatched component or entity is itself a detected object conflict.
- **Reused:** everything in the current loop, block-wise IP scales from our data-prep notebooks, CLIP ViT-L already loaded.
- **New:** stroke grouping, assignment, custom SDXL attention processors that add the text-attention bias while keeping IP-Adapter behaviour.

### Architecture B: typed conflict policy layer (implements Direction 2; sits on top of A or on the current stack)

```
 caption ──► VLM/rules ──► {entity: {colour, material, ...}}  ┐
 reference ──► VLM ──────► {colour, material, finish}         ├─► typed conflict table      [PROPOSED]
 sketch ──► class guess (CLIP) ──► {object}                   ┘   (entity, attribute, text says, ref says, sketch says)
                                                                        │
                                              policy: attribute → governing modality        [PROPOSED]
                                              e.g. shape: sketch | object: text | colour: text | material: reference
                                                                        │
                    ┌───────────────────────────┬───────────────────────┴─────────────┐
                    ▼                           ▼                                     ▼
            text side                    reference side                         sketch side
            keep/remove/insert           which blocks, which component          ControlNet scale lowered
            attribute words              (colour vs texture), scale, mask       where object conflicts
            [simple]                     [established: InstantStyle, SADis]     [established idea: SmartControl,
                                                                                 here rule-based, not learned]
                    └───────────────────────────┴─────────────────────────────────────┘
                                                ▼
                                      same frozen denoising loop
```

- **Three modalities:** each is reduced to a small attribute record; the comparison is symbolic, not a cosine.
- **Correspondence:** attribute type → the mechanism that controls it. Global unless combined with A's regions.
- **Conflict:** detected by comparing records; resolved by the policy; this replaces the router and its four scalars.
- **Reused:** Qwen3-VL attribute extraction (already used to build references), the labelled conflict set from notebook 01b and its `expected_resolution` field, all held-out metrics.
- **New:** the extractor/comparator, the policy, and the attribute-specific enforcement. The open technical question is whether colour and material can be controlled separately through SDXL IP-Adapter blocks or embeddings. SADis and ColorWave suggest so, but I have not read their mechanisms.

### Architecture C: learned spatial gates (Direction 4)

```
 UNet + ControlNet features ──► small conv predictor ──► per-pixel maps: c(x,y), ρ(x,y)
                                 (trained with the diffusion loss on conflict examples)
```

- Extends SmartControl to the reference path. Needs a conflict training set with targets and backpropagation through the SDXL UNet.
- Included for comparison only.

---

## 4. Feasibility comparison

Scores are my judgement (1 low, 5 high), not measurements.

| | A: region binding | B: typed conflict policy | C: learned gates | Benchmark study (Dir. 3) |
|---|---|---|---|---|
| Research novelty | 3 (narrow, to be verified) | 3 (formulation is new-ish, tools are not) | 3–4 | 3 (empirical) |
| Technical depth | 4 | 2–3 | 5 | 2 |
| Implementation feasibility | 3–4 | 4–5 | 1–2 | 5 |
| Training needed | none | none | yes, through the UNet | none |
| T4 memory | about the same as now, plus a few early steps if read-then-enforce | same as now, plus a VLM call | likely out of memory at 640 px | same as now |
| Expected improvement | large on multi-object binding; little on single-object items | moderate on attribute conflicts, where we are weakest today | unknown | none by itself |
| Ease of validation | needs multi-object data with ground truth | uses the conflict set we already have | hard | easy |
| Main risk | attention editing on SDXL is fragile; needs new data | seen as engineering unless enforcement is shown to be attribute-specific | does not fit, or learns nothing, as the router did | not an architecture |

Data note for A: Sketchy is single-object. Pasting two or three Sketchy sketches onto one canvas gives multi-object sketches with known regions and labels at no cost, which is exactly the ground truth the assignment step needs. Real scene sketches (FS-COCO, used by the sketch-understanding papers) are an option to check later.

---

## 5. Recommendation (not final)

The strongest option is **A as the core, B as its conflict layer, and the benchmark as the evaluation**, built in that order so that each stage is a usable result on its own:

1. **Diagnostic first (a few hours of GPU).** On composed two-object sketches, with the current stack, measure how often the right object lands in the right region and read the cross-attention maps. This tells us whether binding is actually broken and by how much. If the frozen stack already binds well, A shrinks to "make the binding visible and controllable" and B becomes the main contribution.
2. **A with oracle regions.** Use the known regions from composition. This isolates the binding mechanism from the grounding step and gives an upper bound.
3. **A with recovered regions.** Add stroke grouping and assignment; report assignment accuracy and the drop from oracle.
4. **B on top.** Typed detection and policy, evaluated on the existing conflict set plus per-entity conflicts in composed scenes.

Why this over the alternatives:
- It answers the professor's question with a mechanism (an explicit assignment that can be shown and scored), not with a stronger adapter.
- It needs no backbone training, so it avoids both failure modes we have already hit: out-of-memory, and a learned module that cannot beat a constant.
- It reuses what works: the denoising loop, the conflict labels, the metrics and the bootstrap evaluation.
- B alone is the safest fallback if attention editing on SDXL proves too fragile. The benchmark alone is the floor.

What it gives up: no trained network beyond what exists, so the "deep learning" content is in understanding and steering attention in a pretrained model. If the course requires training something, that needs to be raised with the professor before committing.

What would be claimed, at most: *sketch-derived regions as a shared index for binding text and reference in a frozen model, plus typed, policy-driven conflict resolution across three modalities, evaluated on a labelled conflict benchmark.* Combining DenseDiffusion, IP-Adapter masks and InstantStyle is not in itself a contribution; the grounding step, the typed policy and the evidence would be.

---

## 6. Open questions before fixing the scope

**Novelty checks (reading, in this order)**
1. Griffin: does it already bind references to regions with text, training-free, and does it take sketches?
2. T³-S2S and Subjective Camera 1.0: how do they associate sketch parts with prompt entities?
3. SketchingReality: what is its text-to-sketch token guidance?
4. FiVA, IPAdapter-Instruct, SADis, MATTE: does any detect or resolve text–reference conflicts, or only let the user choose?
5. SmartControl: exact inputs and training cost of the predictor, to see what a rule-based local scale loses.

**Technical unknowns (small experiments)**
6. Does the frozen stack mis-bind on two-object sketches, and how often? (Step 1 above.)
7. Do DenseDiffusion-style biases work on SDXL's attention layers at 640 px together with ControlNet residuals? Which layers and steps?
8. Does diffusers' `ip_adapter_masks` work inside our custom loop with several references? The docs describe it; a wrong key is silently ignored, so it must be tested.
9. Can colour and material from one reference be separated with block choice or embedding edits on SDXL?
10. How well does CLIP label a cropped Sketchy component against two or three candidate nouns?

**Scope decisions for the team**
11. One reference for the whole image, or one per entity? The professor's example uses one (an artistic style), which is the simpler case.
12. Who sets the conflict policy: fixed rule, user choice, or both? Stage 1 found the 01b policy and the `sem_margin` metric disagree on object conflicts; that must be settled first.
13. (Answered by the team: a training-free method is acceptable.) This removes the main objection to the recommended order in section 5 and leaves Architecture C as comparison only.
14. How much time remains? Steps 1–2 are small; steps 3–4 are where the risk is.
