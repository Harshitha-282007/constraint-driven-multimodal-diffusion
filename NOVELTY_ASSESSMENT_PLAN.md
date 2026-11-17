# Revised plan for novelty assessment

Supersedes sections 2–6 of `STAGE4_FINAL_SCOPE.md`. Written after reading the official project brief. The literature findings in Stage 4 section 1 still stand.

Nothing here has been implemented or run. Items marked **[Hypothesis]** are expectations an experiment is designed to test.

---

## 1. What the official brief changes

| Brief says | Stage 4 assumed | Change |
|---|---|---|
| The model "**automatically determines** which modality should control geometry, semantics, appearance and spatial layout" | A hand-written, user-switchable policy decides | The decision must come from a model, not a rule. The policy switch becomes an analysis tool, not the method. |
| Submit "all **trained/finetuned models**" | Training-free | A trained component must be at the centre of the method. |
| Example: "a red sports car" + sketch (shape, viewpoint) + reference (material, style) | Multi-object example | **Single object is the intended setting.** The two-object work is dropped from scope. |
| Four properties: geometry, semantics, appearance, spatial layout | Shape, object, colour, material | Rename and add layout. For one object, layout means position, scale and viewpoint, all carried by the sketch. |
| Example has the text giving the colour ("red") while the reference gives material and style | Existing conflict set says colour comes from the reference | Appearance has to be split: a colour the text states stays with the text; the reference supplies the rest. This is what makes the colour/texture split necessary rather than optional. |
| Evaluation: text–image alignment, sketch consistency, reference similarity, perceptual quality, **layout consistency** | Adherence metrics only | Report all five. Layout IoU is already coded in `src/eval_metrics.py` but unused. |

One point to confirm with the TA, as the brief itself suggests: whether a small trained network that controls frozen pretrained generators satisfies "trained/finetuned model", or whether part of the generator itself must be fine-tuned. Section 6 has a fallback for the second case.

---

## 2. Final problem and novelty

**Problem.** Given a caption, a sketch and a reference image of one object, which may agree or conflict, generate an image where geometry and layout follow the sketch, identity follows the text, and appearance follows the reference except where the text explicitly states an attribute. The system must work out by itself, per input, which case it is in.

**Gap (from Stage 4 reading).** In the papers read in full, the governing modality is fixed by design (SmartControl: text wins the object, object conflicts only, text versus one spatial condition) or chosen by the user per input (SADis, FiVA, Griffin). None learns to detect, per attribute, where text, sketch and reference disagree and gate the three accordingly. MATTE and IPAdapter-Instruct are still unread and could narrow this.

**Main contribution [proposed].** A **learned governance network**: a small trained model that reads frozen features of the three inputs and predicts, per attribute, which modality governs it. Its outputs drive attribute-specific injection into a frozen SDXL + ControlNet + IP-Adapter pipeline.

It replaces the current router, and differs from it in three ways:

| | Current router | Governance network |
|---|---|---|
| Input | 6 global CLIP cosines | Full frozen embeddings, including the reference split into colour and texture parts |
| Output | 4 global strengths (regression) | Per-attribute gates: object conflict, colour stated-and-different, material stated-and-different |
| Supervision | Soft targets from scoring 9 generated images per item (noisy; 28 items) | The conflict labels we already have (clean; thousands of rows can be made from metadata without generating any image) |

Stage 1 showed why the router failed: weak targets, 28 training items, and inputs that cannot see attribute conflicts. Each row above addresses one of those causes.

**Supporting contributions [proposed]**
1. Attribute-specific enforcement: the gates select which part of the reference embedding is injected (colour, texture, both, neither), which caption words are kept, and where the sketch is relaxed.
2. A per-attribute adherence evaluation alongside the five metric families the brief asks for.

**Reused, not claimed:** colour/texture split of the IP-Adapter embedding (SADis), style-block-only injection (InstantStyle), the idea of relaxing ControlNet under object conflict (SmartControl).

---

## 3. Architecture

```
 caption                     sketch                         reference
   │                           │                               │
   ├─► SDXL text encoders      ├─► scribble ControlNet         ├─► IP-Adapter CLIP ViT-H ─► E(ref)
   │                           │                               │        and E(grey ref)
   │                           │                               │   colour part  = E(ref) − E(grey ref)
   │                           │                               │   texture part = E(grey ref)
   ▼                           ▼                               ▼
 frozen CLIP features of: caption, object phrase, sketch image, reference colour part, texture part
                                         │
                                         ▼
                 ┌────────────────────────────────────────────────┐
                 │  GOVERNANCE NETWORK  (trained; small MLP/      │
                 │  attention over the feature set)               │
                 │  outputs, each in [0,1]:                       │
                 │   g_obj  text and sketch disagree on identity  │
                 │   g_col  text states a colour ≠ reference      │
                 │   g_mat  text states a material ≠ reference    │
                 └────────────────────────────────────────────────┘
                                         │
        ┌────────────────────────────────┼─────────────────────────────────┐
        ▼                                ▼                                 ▼
  geometry + layout ← sketch       semantics ← text              appearance ← reference
  ControlNet residuals;            caption kept; stated          inject colour part × (1 − g_col)
  relaxed after the hold time      attributes stay in the        and texture part × (1 − g_mat),
  in proportion to g_obj           prompt                        style blocks only
        └────────────────────────────────┼─────────────────────────────────┘
                                         ▼
                  existing Euler loop: frozen SDXL UNet + ControlNet + IP-Adapter
                                         ▼
                          image + the predicted gates shown to the user
```

**Training the governance network**
- Data: from the 160 training photos, generate conflict rows the way notebook 01b does (swap captions across categories, change colour words, change material words, swap references), in much larger numbers. Only embeddings are needed, so no image generation and almost no GPU.
- Labels: the conflict type of each row, which is known by construction.
- Loss: binary cross-entropy per gate.
- Split: by base photo, as now. Test on the 20 test photos, plus about 50 captions paraphrased by the VLM or written by hand, so the network is not graded only on the word lists it was built from.
- Cost **[estimate]**: minutes on a T4 or CPU once embeddings are cached.

**Main risk.** Stage 1 found that CLIP cosines do not separate colour or material conflicts. The bet here is that a trained head on full embeddings, with the reference's colour part isolated, can. This is testable in a day without generating images, and it should be the first thing tested. If it fails, the fallback input is attribute records from Qwen3-VL (which we already run), still feeding a trained network.

---

## 4. How the plan meets the brief

| Brief deliverable | Covered by |
|---|---|
| 1. Multimodal data conditioning | Existing preprocessing, plus the reference colour/texture split and cached features |
| 2. Constraint-aware multimodal diffusion model | Frozen SDXL + ControlNet + IP-Adapter driven by predicted gates |
| 3. Modality-specific control mechanism that automatically determines control | The governance network |
| 4. Conflict resolution and adaptive conditioning | Gates → attribute-specific injection; sketch for shape and viewpoint, text for identity, reference for material and style |
| 5. Evaluation | Section 5 |
| Trained model | Governance network weights (and the old `router.pt` as the baseline) |
| Interface | Existing Gradio app, updated to show the gates |
| Metric display, qualitative results, report | Existing evaluation notebooks and contact sheets, extended |

---

## 5. Experiments

| # | Experiment | Shows | GPU |
|---|---|---|---|
| E1 | Gate accuracy on held-out photos and on paraphrased captions; versus six-cosine + logistic regression, versus rule-based comparison of VLM attributes | The network detects typed conflicts | none |
| E2 | Generation on the test split: fixed, scheduled, router, block-wise IP only, proposed | The gates improve attribute adherence | T4 |
| E3 | Ablations: no embedding split; no style-block restriction; no sketch relaxation; oracle gates instead of predicted | Which part matters; cost of detection errors | T4 |
| E4 | Conflict-free items | No loss when nothing conflicts | T4 |
| E5 | Policy probe: force each gate on and off | The gates causally control the attribute | T4 |

**Metrics, mapped to the brief**
- Text–image alignment: CLIP-T; object margin (`sem_text − sem_sketch`).
- Structural and sketch consistency: edge recall, precision, IoU.
- Reference similarity: DINO to reference; colour adherence (nearest named colour of the object region versus the governing source); material adherence (CLIP zero-shot plus VLM check; weakest, secondary).
- Perceptual quality: LPIPS and DreamSim to the target on conflict-free items. FID only if the test set is large enough to mean anything.
- Layout consistency: CLIPSeg mask IoU between output and target (`layout_iou` in `src/eval_metrics.py`).
- Statistics: paired bootstrap over base photos, as now.

**Expected [Hypothesis]:** E1 clearly beats the cosine baseline on colour and material; colour adherence improves most; material less and with more noise; object conflicts improve only with sketch relaxation. Any of these can fail and still be reported.

---

## 6. Roadmap

| Milestone | Work | Output | Risk |
|---|---|---|---|
| **M0** | Correct RELAX 0.3 / GUIDE_W 0.25 in deck and report; change 01b so a stated text colour or material stays with the text and text decides the object; keep only filter-passing references | Consistent baseline and labels | none |
| **M1** | Cache features; generate metadata-only conflict rows; train governance network; run E1 | Trained model + detection table | CLIP features may not carry attribute conflicts → switch to VLM-record inputs |
| **M2** | Enforcement in `run_item`: embedding split, style-block scales, sketch relaxation by `g_obj` | Pipeline driven by gates | Embedding split may behave differently on our IP-Adapter checkpoint |
| **M3** | E2, E4 on the 20 test photos; all five metric families | Main results table | a few T4 hours |
| **M4** | E3, E5; update the Gradio app to show gates | Ablations, demo | time |
| **M5 (only if the TA requires fine-tuning the generator)** | Train a small LoRA or a SmartControl-style scale predictor at reduced resolution | A fine-tuned generator component | Memory on a T4; needs a one-hour feasibility test before committing |

M0–M3 are the minimum complete result.

---

## 7. Presentation outline

1. **The brief in one line.** Automatically decide which modality controls geometry, semantics, appearance and layout.
2. **What we built.** Frozen SDXL + ControlNet + IP-Adapter with a router predicting four global strengths.
3. **What it could not do.** The router matched a hand schedule; its inputs cannot see colour or material conflicts; the reference enters as one bundle.
4. **What the literature offers.** SmartControl, SADis, InstantStyle, T³-S2S, Griffin: what each solves and who decides the winner.
5. **The gap.** Nobody learns typed, per-attribute governance across text, sketch and reference.
6. **Our contribution.** The governance network, shown on the "red sports car" example from the brief.
7. **Architecture.** Features → gates → attribute-specific injection → frozen pipeline; reused parts labelled with sources.
8. **Why it should work where the router did not.** Clean labels, far more training rows, attribute-level inputs.
9. **Evaluation.** E1–E5 and the brief's five metric families.
10. **Plan, risks, and what we do not claim.**

---

## 8. To settle

1. With the TA: does a trained controller over frozen generators count as the "trained/finetuned model"?
2. Read MATTE and IPAdapter-Instruct in full before stating the gap in the report.
3. Confirm the appearance rule: a colour or material stated in the text stays with the text (as in the brief's "red sports car"), everything else from the reference.
4. Time remaining, which decides whether M4 is done in full.
