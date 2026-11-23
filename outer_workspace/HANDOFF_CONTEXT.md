# IE 643 project: full context handoff (as of 2026-11-21)

Paste or attach this file at the start of the next conversation.

## 1. Project

- **Course:** IE 643. **Title:** "Constraint Driven Multimodal Diffusion using Text, Sketch and Reference Image". **Team:** One Sheep Army.
- **Repo:** github.com/Harshitha-282007/constraint-driven-multimodal-diffusion (old URL Harshitha-282007/IE643_final redirects). Working branch: `governed-pipeline`. Local copy: `C:\Users\nikhi\Desktop\IE643_final\IE643_final`.
- **Brief constraints:** a trained model is required (not only hand settings); one object per image; the model must decide automatically which input controls what; must run on a free Colab T4 (~15 GB).
- **Governing rules (final, decided by the team):**
  - sketch → shape and placement;
  - text → object identity;
  - reference → colour AND material. The reference wins colour even when the caption names a different colour.
- **Professor's core question:** "How does the model understand which part of the input governs which part of the output?" The professor wants SDXL itself to understand the input roles and act on them, not tricks around it. TA: "come up with a method to incorporate sketch and image into the SDXL model."

## 2. Data

- Sketchy dataset. 200 photos captioned with Qwen3-VL-2B. Reference images v2: a neutral object (vase / mug / ball / bottle / box) carrying the photo's colour, material and finish, segmented on grey.
- `pilot_v1`: 200 samples, split 160 train / 20 val / 20 test by photo. Includes object masks (`mask/`).
- Notebook 01b adds 800 conflict rows (object, colour, material, swapped reference). `governance/labels.py` can generate more conflict rows from metadata.
- Drive root: `MyDrive/sketchy_dataset_one_sketch/`, containing `pilot_v1.zip`, `governance/governance.pt`, and `governance/role_gates_v3/` (plus `role_gates/` = v1 and `role_gates_v2/`).

## 3. History: what was tried, what worked, what failed

1. **Baseline (nb 02).** Frozen SDXL + xinsir scribble ControlNet + IP-Adapter ViT-H (non-plus), fixed strengths.
   - Works roughly.
   - The reference copies its own object (DINO to reference 0.673 vs to target 0.105).
2. **Router / gated fusion (nb 03, 04).** 6 CLIP cosines → MLP → 4 global scalars (sketch strength, hold time, IP strength, CFG), plus a time schedule and optional guidance.
   - The router was no better than the hand schedule (beat a constant on 1 of 4 outputs).
   - The cosines can't see colour or material conflicts.
   - Guidance caused artefacts.
   - The code used RELAX=0.3 and GUIDE_W=0.25; the slides wrongly said 0.7 and 0.03.
   - FLUX/UniCombine was tried earlier and dropped (out of memory on the T4).
3. **Governance network (nb 05).** Frozen CLIP features of caption, object phrase, sketch, reference, grey reference and the colour part → 3 conflict flags (g_obj, g_col, g_mat). **Works:**
   - colour-conflict detection 0.755 vs 0.540 for the old detector;
   - material 0.940 vs 0.643.

   **But the images didn't change:**
   - colour hit 0.40, the same as the schedule;
   - flipping the colour gate changed pixels by only 7.8/255.
4. **Fusion LoRA (nb 06, 07)** on the whole UNet.
   - Object conflicts resolved 8 of 8 (vs 2 of 8), and images were more realistic.
   - But edge IoU fell from 0.369 to 0.230. It learned the whole photo scene and sat where ControlNet enters, so it learned to ignore the sketch.
   - Results report artifact: https://claude.ai/artifact/4bqmB6RUzKZjnXSx36MwBm
5. **Rule change:** the reference wins colour.
6. **Role gates inside SDXL (nb 08, `src/rolegate/`), v1.**
   - Learned gates `g = 2·σ(b0 + W_h·LN(h) + W_c·c)`, where c = [3 conflict flags, sin, cos, f, 1−f] (COND_DIM 7).
   - Cross-attention gates: text_other, text_identity, text_colour, text_material, ref_colour, ref_texture. Sketch gates on the ControlNet residuals: sketch_contour, sketch_placement.
   - Result: the sketch gates saturated to 0 or 2 (they read raw residuals), so the output ignored the sketch. Colour followed the text, because CLIP's causal text encoder spreads a colour word into later tokens and the pooled embedding.
7. **v2.** Changes:
   - SketchGate layer-norms its input, trains at 0.2× lr, and has a keep_loss.
   - Neutral-caption channels: the caption is re-encoded with colour words → "colored" and material words → "solid", then `out = o_full + (g_col−1)(o_full − o_neutral_colour) + (g_mat−1)(…)`.
   - PooledGate applies the same mix to the pooled embedding.
   - Annealed prior; contour weight 0.1.

   Result:
   - The sketch is fixed (motorcycle and pistol follow their sketches; the axe became a log pile).
   - Colour still followed the text (purple motorcycle, gold pistol). text_colour only reached ~0.5–0.6 and ref gates ~0.1, because zero-initialised flag weights move the logit only ~0.3 in 600 steps.
8. **v3 (commit ab7d57e, pushed).** Changes:
   - FLAG_INIT = −3, so text_colour, text_material and the pooled gates start closed when flagged.
   - REF_COLOUR_INIT: 1.5 in up_blocks.0.attentions.1 (the style block), 0.5 in up_blocks.0.attentions.0 and .2, 0.1 elsewhere. Texture keeps the InstantStyle placement.
   - The prior targets the same starting levels.

   Results:
   - The run stopped at step 450 of 600 (Colab disconnected).
   - **No gate moved more than 0.06 from its start.** The colour loss was flat at ~0.10, as in v2; contour ~0.97, identity ~0.015 also flat. Any effect comes from the hand-set starting values, not learning.
   - Before/after figure (`role_gates_v3/before_after.png`). Left column = an InstantStyle baseline (flag weights zeroed, ref_colour bias = texture bias); right column = step-450 gates, copied to `role_gates.pt`:
     - motorcycle still purple (the caption says purple; the reference is silver, black and white);
     - pistol gold, not silver;
     - axe got worse: a log pile that loses the sketch (the sketch_placement gate is low on the object).
   - Gate maps (magma colourmap, 0–2; pink ≈ 1, dark ≈ 0): flat, no spatial pattern; text_colour ≈ 0.1; ref maps dark (averaged over layers).

## 4. Key diagnosis now

- **Leak found:** `src/pipeline.py:266`. The ControlNet gets `encoder_hidden_states=pr`, the FULL caption, ungated and not neutralised. So "purple" reaches the UNet through ControlNet residuals even with text_colour closed. Not yet fixed or tested.
- The reference signal (IP-Adapter, one global vector → 4 tokens) mixes colour, shape and identity, so it can't be turned up without leaking the reference object.
- The training signal is too weak:
  - the attribute losses use a single-step x0 decode at random t;
  - the prior pulls gates back;
  - lr 5e-4 is too low for scalar gates inside a frozen SDXL.

## 5. Options for next steps (user to choose)

- **A. Quick test (recommended first, ~15 min on Colab, no retraining):** when the colour flag is on, give the ControlNet the colour-neutral caption embeddings (mix by the gate, as for UNet text), then rerun the before/after. If the purple disappears, the leak is confirmed.
- **B. Measure:** run notebook 09 (eval, RUN="role_gates_v3") on baseline vs v3-init vs v3-trained.
- **C. v4 architecture (proposed):**
  - a trained colour adapter (T2I-Adapter style), whose input is the sketch region filled with the reference palette; zero-init residuals into the UNet encoder;
  - trained self-supervised on ~12k Sketchy photos, with random wrong colour words in captions;
  - IP-Adapter fed a tiled interior patch of the reference (texture/material without shape);
  - keep the sketch gates, the neutral-caption trick and the governance flags.

  This gives a trained component with a real loss signal.
- **D. Cheaper gate fixes:** lr ~10×, prior ~0.05, colour loss via latent→RGB at colour-deciding timesteps, and a rebuilt contour loss (it's broken, flat at 0.97).

## 6. Code map

- `src/rolegate/gates.py`: CrossGate, SketchGate, PooledGate, RoleGates (`for_unet`, `load`, `install`, `remove`, `mix_pooled`, `gate_maps`), FLAG_INIT, REF_COLOUR_INIT, `ref_init`.
- `src/rolegate/tokens.py`: token groups; `NEUTRAL = {"colour": "colored", "material": "solid"}`; `neutral_caption`.
- `src/rolegate/losses.py`: prior_loss, keep_loss, swap_loss, masked_diffusion.
- `src/train_role_gates.py`: training (W weights, ATTR_W, SKETCH_LR = 0.2, VERSION = 3, checkpoints every 150 steps; no resume).
- `src/pipeline.py`: Pipeline, make_items (neutral captions), run_role (CFG-batched neutral tensors).
- `src/governance/`: labels, model, features, evalkit (rule="reference"), metrics.
- `notebooks/make_role_notebooks.py`: generates nb 08 (train) and 09 (eval); `ROLE_DIR = governance/role_gates_v3`.
- Tests: `PYTHONPATH="src;<scratchpad>/pkgs/site" python -m pytest -q tests`. 28 pass on CPU.
- Plans and notes: STAGE1–4 .md, NOVELTY_ASSESSMENT_PLAN.md, ROLE_GATED_PLAN.md, README.md.

## 7. Working notes

- Commit and push only when asked. Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- The user's email is private; never send it to outside services.
- Colab runs in the user's Chrome (Claude in Chrome tools).
  - "Connect to Google Drive" needs the user's OK each time.
  - Google may open a separate sign-in popup window that only the user can complete.
  - Don't grant HF_TOKEN / secrets dialogs.
- Read Colab results from Drive with the Google Drive connector; Colab output is often unreadable from the page.
- **Colab gotchas:**
  - Run all also re-runs the memory test, which loads a second SDXL and runs out of memory. Run cells selectively.
  - A monkey-patch cell run twice recursed; reload `rolegate.gates` before re-patching.
- **Current Colab state:**
  - The nb 08 tab has two extra ad-hoc cells: one copies step0450 → role_gates.pt and sets the InstantStyle baseline patch; one is the re-patch.
  - The runtime may have disconnected since.
- **Docs:**
  - Project history doc: https://claude.ai/artifact/QoQh1F7ZxNagUQfwujrjUw ("IE 643 Project Journey: What We Tried", updated with the v3 result).
  - Earlier results report: https://claude.ai/artifact/4bqmB6RUzKZjnXSx36MwBm
