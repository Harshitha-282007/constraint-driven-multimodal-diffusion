# Learnable modality governance in SDXL — report (overnight run)

**Status in one paragraph.** The method (Counterfactual Attribute Supervision, CAS) is implemented in `governance/`, and its data builder, loss, trainable-parameter selection and exact resume are verified by a CPU smoke test on a tiny random SDXL-shaped model. **No SDXL training or image generation was run, so there are no baseline-vs-CAS results and no evidence yet that CAS works.** The sandbox has no GPU, and the egress proxy blocks huggingface.co (HTTP 403), so SDXL, ControlNet and IP-Adapter weights could not be downloaded. The Sketchy/pilot_v1 images live on Google Drive and are not in the repo. The code that talks to the real SDXL stack (`cas_stack.py`, `infer.py`, `evaluate.py`) has been compiled but **not executed**.

## Stage 1 — What the repo actually contains (verified)
- Present: notebooks 01–04 and `final_output_eval`, data CSVs, `src/eval_metrics.py`, the abandoned FLUX/UniCombine notebooks, a Gradio app, and `models/router.pt` (37 KB). Notebook outputs are stripped, so **there are no recorded results or logs**.
- **Absent from the repo:** any code, checkpoint or result for the governance network, whole-UNet LoRA, Role Gates V1–V3, the colour-specific reference adapter, or the multi-role attention LoRA. The README only documents the attribute router (04), gated fusion (03) and the FLUX attempt. I therefore **could not verify the LoRA collapse or its inference-scale/guidance diagnostic**, and I do not assert why those methods failed. If their code lives elsewhere (e.g. Drive), point me at it.
- What I can verify about the router (notebook 04): it only predicts four global scalars per item (ControlNet scale, sketch hold time, IP-Adapter scale, CFG) from six CLIP conflict features, with targets from a one-factor-at-a-time grid of 9 images on ~18 train items ("fast" profile). All SDXL weights stay frozen.
  - Why that cannot give attribute-level governance (reasoning from the implementation, not from measured results): the IP-Adapter ViT-H image tokens are a 4-token projection of the global CLIP embedding, which carries object identity and shape as well as colour/material. A single scalar scales all of that together, so it cannot separate "take the colour but not the shape". The network never sees a training signal that penalises copying the reference's shape.
  - Data fact that matters for the new method: in `sketchy_with_refs_v2.csv` the reference is a *different, neutral object* (vase/mug/ball/bottle/box) with the target's colour/material. So every real training pair is already shape-counterfactual (reference shape ≠ target shape, appearance = target's). Nothing in the router exploits this.
  - The `01b` conflict rows have **no ground-truth target image** (the target photo does not match the swapped caption/reference). Training on them with the photo as target would teach the wrong thing.
  - Policy conflict: `01b` `RESOLUTION["object_text_sketch"]` says "keep the sketch outline, don't become the caption object", while this task's ownership says identity → text. I did not train on object conflicts (no target exists); evaluation reports both outcomes.

## Stage 2 — Candidates
References I am fairly sure of: IP-Adapter (Ye et al. 2023), ControlNet (Zhang et al. 2023), InstantStyle (arXiv 2404.02733) and B-LoRA (arXiv 2403.14572) — the last two confirmed by a web search in this run; the rest from memory, not re-checked.

| # | Method | What is learned / where | Attribute separation | Needs | Verdict |
|---|---|---|---|---|---|
| A | Block-restricted reference injection (InstantStyle) | Nothing learned; reference tokens only into one style block. SDXL cross-attn | Layout vs style by *block* (InstantStyle reports the 4th/6th blocks ↔ layout/style) | No training | Use as a comparison arm only; it is a hand-set rule, which the task forbids as the solution |
| B | B-LoRA style/content LoRAs | Per-image LoRA on 2 blocks | Content vs style | Per-image optimisation, no conflict handling | Not suited (per-subject, no multi-modal conflict) |
| C | Learned per-layer/per-modality gates (≈ Role Gates) | Gate scalars in attention | By layer | Only the denoising loss | Reportedly failed; gates get no attribute-specific signal — the missing ingredient is the *data/loss*, not the gate |
| D | Whole-UNet / attention LoRA | Many weights | Implicit | Large data | Reportedly collapsed; high capacity vs ~180 photos |
| E | **CAS: counterfactual-target training of the reference pathway (chosen)** | IP-Adapter image projection + `to_k_ip/to_v_ip` (≈ 20–30 M params), optional rank-4 LoRA on text cross-attn K/V | Supervised by counterfactual targets: reference owns colour; sketch owns geometry; text owns identity | Existing pilot data + masks; known targets | Chosen |
| F | Attribute classifier/regressor losses on x0 (colour histogram, CLIP material) | Same params as E | Explicit per-attribute loss | VAE decode in the loop (memory) | Deferred; latent-space masked loss approximates it |

## Stage 3 — Selection: CAS
Chosen because it (1) targets exactly what is missing — a training signal in which modalities conflict *and the target image is known*; (2) trains a small, low-capacity subset (mitigating the collapse risk of D) with an L2-SP anchor to the pretrained IP-Adapter; (3) keeps ControlNet, text encoders and the rest of SDXL frozen; (4) fits a 16 GB T4 (ControlNet + SDXL fp16 with gradient checkpointing at 640 px, batch 1, accumulation 4 — **memory not measured**).

## Architecture — where learning happens
```
sketch ──► ControlNet (FROZEN, scale 1.0) ─► residuals ─────────────┐
caption ─► SDXL text enc 1+2 (FROZEN) ─► text tokens ─► attn2 K/V ──┤ (optional rank-4 LoRA ✎)
reference ► CLIP ViT-H (FROZEN) ► image_proj ✎ ► 4 tokens ► to_k_ip/to_v_ip ✎ (IP cross-attn, every layer)
                                                                    ▼
                                                    SDXL UNet (FROZEN except ✎)  ─► eps
✎ = trained.  No router, no conflict flag, no per-modality scale at inference (fixed constants).
```

## Training process in plain terms
Each step draws one item (deterministically from seed+step) of three kinds:
1. **real (40 %)** — the original aligned quadruple.
2. **cf_ref (40 %)** — swap in another photo's reference (different object *and* colour). The target is the original photo with its object **recoloured to the swapped reference's colours** (masked Lab chroma transfer). The caption's colour word is removed or replaced by a wrong one. The only way to lower the loss is: colour from the reference, geometry from the sketch, identity from the caption's noun, and *not* the reference's silhouette.
3. **cf_cap (20 %)** — the true reference/photo, but the caption names a wrong colour (text colour must lose to the reference).

Losses: (i) ε-prediction MSE with ×(1+2·mask) weight on the object region; (ii) **shape-invariance** — the noise prediction with the reference randomly flipped/rotated/rescaled (same appearance, different silhouette) must match the unperturbed one (weight 0.1, every 4th step); (iii) L2-SP anchor to the starting weights. 10 % dropout per modality keeps CFG valid.
**Limitation by construction:** material/texture cannot be synthesised, so it gets supervision only from `real` pairs and (ii). Colour is the only attribute with true counterfactual targets. The recolour pseudo-targets are edited photos (hue-shifted), not natural images.

## Files
All new, under `governance/`; nothing existing was modified.
`cas_data.py` (counterfactual builder) · `cas_model.py` (trainable selection, checkpoint helpers) · `cas_core.py` (one training step) · `cas_stack.py` (SDXL+ControlNet+IP loading) · `train.py` · `infer.py` · `metrics.py` · `evaluate.py` · `smoke_test.py` · this report. Branch `claude/modest-ride-vkaas0`; baseline notebooks and `router.pt` untouched.

## Commands (on a T4/A10 with Drive data extracted from `pilot_v1.zip`)
```
pip install -U torch diffusers transformers accelerate safetensors peft scipy scikit-image matplotlib pandas
python governance/smoke_test.py                       # CPU, ~1 min, no weights
python governance/train.py --data_dir /content/pilot_v1/<dir> --out runs/cas_v1 --steps 3000   # re-run same command to resume
python governance/infer.py --weights runs/cas_v1/cas_weights.pt --sketch s.png --ref r.png --caption "a red bicycle" --out o.png
python governance/evaluate.py --data_dir /content/pilot_v1/<dir> --weights runs/cas_v1/cas_weights.pt --out runs/cas_v1/eval
```
Compute estimate (unmeasured guess): 3000 optimiser-steps ×4 accumulation ≈ 12 k forward/backward passes; expect many hours on a T4, so start with `--steps 600` and check the loss and qualitative grid.

## Evaluation design (implemented, not run)
Same items, same per-item seeds, four arms: `baseline` (IP 0.6), `baseline_ip1` (IP 1.0, isolates the scale change), `instantstyle_block` (training-free rule, comparison only), `cas`. Separate metrics: geometry = edge recall on sketch strokes; shape leak = edge recall on the reference silhouette (`geo_margin` = difference); colour = Lab-ab distance in the sketch-filled region to reference vs caption colour; material = CLIP-B/32 margin on the object crop; identity = CLIP zero-shot over the 125 categories. Interventions with a fixed seed: swap reference / swap sketch / swap caption, each checking the intended attribute moves and the others stay. Paired-bootstrap 95 % CIs for CAS − each baseline. Caveats: the evaluation set is ~12 items per kind, the object region comes from the sketch fill (rough), and CLIP-B/32 is a weak material judge — treat differences as indicative only.

## Smoke-test result (the only thing actually run)
Tiny random model: only the 12 IP-adapter tensors received gradient/moved, 0 frozen tensors changed, kinds sampled 21/9/10 cf_ref/cf_cap/real, recoloured object mean RGB (239,220,39) vs reference (240,220,40), and an interrupted-then-resumed run matched an uninterrupted run exactly (max diff 0.0). This validates plumbing only, not image quality.

## Known risks / open questions
- CLIP global-embedding IP-Adapter (4 tokens) may not carry enough fine colour/material information; the `plus` variant (patch tokens) is the next step if CAS underfits.
- Pseudo-target recolouring can leave artefacts; the model may learn them.
- CUDA path: VAE latent sampling is not seeded per step, so resume is exact on CPU but only statistically equivalent on GPU.
- CAS does not teach text→identity beyond what SDXL already does; identity governance is only tested, not trained.
- Does the method demonstrate attribute-level governance? **Unknown — not yet trained or evaluated.**
