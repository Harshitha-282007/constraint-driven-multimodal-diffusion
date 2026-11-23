# Stage 2: literature review

Starting point: Stage 1 found that our system sets four global strengths (sketch `c`, hold `h`, reference `ρ`, text `w`) and has no explicit binding between text tokens, sketch regions and reference attributes. This review looks for work on exactly those bindings, on multi-condition fusion, and on conflicting conditions.

**How far to trust this document**
- Every arXiv link below was checked against the arXiv API: the ID resolves to the stated title. Two 2026 conference papers have no arXiv ID that I could find and are linked to their venue pages.
- Mechanism descriptions for 2022–2024 papers are from my knowledge of the papers, not from re-reading them now. Treat details as "confirm when you read it".
- Papers marked **(abstract-level)** I know only from the abstract or a search summary. Do not cite their mechanisms without reading them.
- The search was targeted, not exhaustive. "I did not find" means a few searches turned up nothing, not that nothing exists.

---

## 1. Prioritised paper list

### Tier A: bear directly on the professor's question, and fit a frozen SDXL on a T4

| # | Paper | Why it matters to us |
|---|---|---|
| A1 | [SmartControl: Enhancing ControlNet for Handling Rough Visual Conditions](https://arxiv.org/abs/2404.06451) (2024) | Text-versus-spatial-condition conflict, solved with a **spatial map** of ControlNet strength instead of one scalar. The closest published relative of our router. |
| A2 | [Text-Anchored Score Composition: Tackling Condition Misalignment](https://arxiv.org/abs/2306.14408) (2023; earlier title "Decompose and Realign") | Training-free handling of conditions that disagree with the text. |
| A3 | [Dense Text-to-Image Generation with Attention Modulation (DenseDiffusion)](https://arxiv.org/abs/2308.12964) (2023) | Training-free: binds each text segment to a region by editing attention scores. |
| A4 | [Be Yourself: Bounded Attention for Multi-Subject Generation](https://arxiv.org/abs/2403.16990) (2024) | Explains and prevents leakage between subjects in cross- and self-attention; works on SDXL. |
| A5 | [InstantStyle](https://arxiv.org/abs/2404.02733) and [InstantStyle-Plus](https://arxiv.org/abs/2407.00788) (2024) | What IP-Adapter injects in which SDXL block; style without content leakage. We already use this trick in data prep but not in the main pipeline. |
| A6 | [MS-Diffusion: Multi-subject Personalization with Layout Guidance](https://arxiv.org/abs/2406.07209) (2024) | Binds each reference image to a box via masked image cross-attention. Reference ↔ region. |
| A7 | [T³-S2S: Training-free Triplet Tuning for Sketch to Scene](https://arxiv.org/abs/2412.13486) (2024) | Multi-instance sketch + text on **SDXL + ControlNet**, training-free. Same backbone as ours. |
| A8 | [Ctrl-X: Controlling Structure and Appearance Without Guidance](https://arxiv.org/abs/2406.07540) (2024) | Training-free structure image + appearance image + text on SDXL. Our three modalities with a different mechanism. |
| A9 | [InstanceDiffusion: Instance-level Control](https://arxiv.org/abs/2402.03290) (2024) | Per-instance text plus location given as point, box, mask or **scribble**. |
| A10 | [Uni-ControlNet](https://arxiv.org/abs/2305.16322) (2023) | Local (sketch-like) + global (reference image) + text in one model; includes a condition-conflict analysis. |

### Tier B: mechanisms worth understanding

**Region and layout grounding**
- [GLIGEN](https://arxiv.org/abs/2301.07093): gated self-attention over (phrase, box) grounding tokens.
- [MIGC](https://arxiv.org/abs/2402.05408): divide-and-conquer per-instance shading, then aggregation.
- [Training-Free Layout Control with Cross-Attention Guidance](https://arxiv.org/abs/2304.03373), [BoxDiff](https://arxiv.org/abs/2307.10816), [R&B](https://arxiv.org/abs/2310.08872): energy on cross-attention maps, gradient on the latent.
- [MultiDiffusion](https://arxiv.org/abs/2302.08113): separate denoising path per region, fused each step.
- [SpaText](https://arxiv.org/abs/2211.14305), [eDiff-I](https://arxiv.org/abs/2211.01324) ("paint with words").
- LLM planners that turn a prompt into a layout: [LLM-grounded Diffusion](https://arxiv.org/abs/2305.13655), [RPG](https://arxiv.org/abs/2401.11708), [Ranni](https://arxiv.org/abs/2311.17002).

**Text-side binding (which attribute goes with which noun)**
- [Prompt-to-Prompt](https://arxiv.org/abs/2208.01626): cross-attention maps carry token-to-pixel layout.
- [Attend-and-Excite](https://arxiv.org/abs/2301.13826), [Divide & Bind](https://arxiv.org/abs/2307.10864).
- [Linguistic Binding (SynGen)](https://arxiv.org/abs/2306.08877), [Structured Diffusion Guidance](https://arxiv.org/abs/2212.05032).
- [TokenCompose](https://arxiv.org/abs/2312.03626): token-level mask supervision during fine-tuning.

**Reference image: style, appearance, subject**
- [IP-Adapter](https://arxiv.org/abs/2308.06721): decoupled cross-attention.
- [DEADiff](https://arxiv.org/abs/2403.06951), [B-LoRA](https://arxiv.org/abs/2403.14572), [CSGO](https://arxiv.org/abs/2408.16766): style/content disentangling.
- [Style Aligned](https://arxiv.org/abs/2312.02133), [Cross-Image Attention](https://arxiv.org/abs/2311.03335), [MasaCtrl](https://arxiv.org/abs/2304.08465), [Plug-and-Play](https://arxiv.org/abs/2211.12572): appearance through shared or injected self-attention.
- [FastComposer](https://arxiv.org/abs/2305.10431), [Character-Adapter](https://arxiv.org/abs/2406.16537) (abstract-level), [Break-A-Scene](https://arxiv.org/abs/2305.16311), [Cones 2](https://arxiv.org/abs/2305.19327), [Mix-of-Show](https://arxiv.org/abs/2305.18292), [AnyDoor](https://arxiv.org/abs/2307.09481): localising a reference to a subject or region.
- [ColorizeDiffusion](https://arxiv.org/abs/2401.01456) (abstract-level): sketch + reference + text for colourisation.

**Multi-condition fusion**
- [ControlNet](https://arxiv.org/abs/2302.05543), [T2I-Adapter](https://arxiv.org/abs/2302.08453), [ControlNet++](https://arxiv.org/abs/2404.07987).
- [Composer](https://arxiv.org/abs/2302.09778), [UniControl](https://arxiv.org/abs/2305.11147), [Cocktail](https://arxiv.org/abs/2306.00964), [AnyControl](https://arxiv.org/abs/2406.18958).
- [MaxFusion](https://arxiv.org/abs/2404.09977): training-free, per-location fusion by feature variance.
- [DynamicControl](https://arxiv.org/abs/2412.03255): an MLLM ranks conditions, adapter uses them in order.
- [OminiControl](https://arxiv.org/abs/2411.15098), [UniCombine](https://arxiv.org/abs/2503.09277), [EasyControl](https://arxiv.org/abs/2503.07027), [UNO](https://arxiv.org/abs/2504.02160), [OmniGen](https://arxiv.org/abs/2409.11340), [FLUX.1 Kontext](https://arxiv.org/abs/2506.15742): condition images as tokens in a diffusion transformer.

**Recent work on conflicting conditions (all abstract-level)**
- [Cross-ControlNet](https://iclr.cc/virtual/2026/poster/10011222) (ICLR 2026): training-free pixel-wise and channel-wise fusion of conflicting spatial conditions.
- [DynFusion](https://cvpr.thecvf.com/virtual/2026/poster/37846) (CVPR 2026): learned gates per DiT block choose which conditions are active per timestep.
- [PixelPonder](https://arxiv.org/abs/2503.06684) (2025): patch-level selection among visual conditions over time.
- [AtteConDA](https://arxiv.org/abs/2605.09425) (2026): attention-based conflict suppression among segmentation, depth and edge.
- [MixDiffusion](https://arxiv.org/abs/2607.17634) (2026): mixes single-condition models; notes conflicting conditions give ambiguous results.

**Guidance and schedules** (the family our current guidance belongs to)
- [Universal Guidance](https://arxiv.org/abs/2302.07121), [FreeDoM](https://arxiv.org/abs/2303.09833), [FreeControl](https://arxiv.org/abs/2312.07536), [Composable Diffusion](https://arxiv.org/abs/2206.01714).
- [Applying Guidance in a Limited Interval](https://arxiv.org/abs/2404.07724), [Analysis of CFG Weight Schedulers](https://arxiv.org/abs/2404.13040).

**Sketch-specific**
- [Sketch-Guided Text-to-Image Diffusion Models](https://arxiv.org/abs/2211.13752), [Sketch-Guided Scene Image Generation](https://arxiv.org/abs/2407.06469) (abstract-level), [Step-Layer-Wise Tuning of Attention Specialty](https://arxiv.org/abs/2504.10148) (abstract-level; instance sketches on DiTs).

**Evaluation**
- [T2I-CompBench++](https://arxiv.org/abs/2307.06350), [GenEval](https://arxiv.org/abs/2310.11513): attribute binding, spatial relations, object counts.

---

## 2. Paper notes (Tier A, six questions each)

**A1 SmartControl**
- Problem: ControlNet forces the output onto the condition even when the text asks for something the condition contradicts (depth map of one animal, prompt naming another).
- Idea: a small control-scale predictor outputs a **pixel-wise** ControlNet scale map, so control is relaxed only where text and condition conflict.
- Inputs: text through the usual path; condition through ControlNet; the predictor reads features and emits a map that multiplies the ControlNet residuals.
- Control: strict where the two agree, loose where they do not.
- Useful because: it is our `c` made spatial, and it treats conflict as a local property.
- Limits: the predictor is trained (on a small conflict dataset); one spatial condition versus text only; no reference image.

**A2 Text-Anchored Score Composition**
- Problem: several conditions that are not aligned with each other or the text produce blended or broken images.
- Idea: compute a separate score for each text–condition pair and compose them, using the text as the common anchor, with a cross-attention step to keep the composed layout consistent.
- Control: training-free; each condition can govern its own part.
- Useful because: it is a principled alternative to "one forward pass with everything switched on".
- Limits: several UNet passes per step (cost on a T4); details to confirm on reading.

**A3 DenseDiffusion**
- Problem: dense captions where each phrase should land in a given region.
- Idea: add or subtract from attention scores before softmax so that a phrase's tokens attend inside its mask (cross-attention) and pixels attend within their own region (self-attention). Modulation strength adapts to mask area and score range.
- Useful because: no training, no gradients, cheap; the most direct way to make token ↔ region explicit.
- Limits: needs a mask per phrase; rough masks give soft boundaries; text only.

**A4 Bounded Attention**
- Problem: semantically similar subjects leak features into each other; the paper shows leakage happens in both cross- and self-attention.
- Idea: during denoising, bound each subject's attention to its own box/mask, plus a guidance term early on; masks are refined by clustering self-attention.
- Useful because: it diagnoses *why* binding fails in SD/SDXL, which is the analysis Stage 1 said we lack.
- Limits: needs boxes; gradient guidance through the UNet is heavy.

**A5 InstantStyle / InstantStyle-Plus**
- Problem: IP-Adapter copies the reference's content along with its style.
- Idea: (i) inject the image embedding only into the SDXL attention blocks that carry style, leaving the layout-carrying block untouched; (ii) optionally subtract the content's text embedding from the image embedding.
- Useful because: zero training, one line in diffusers, and it explains our baseline result (text+reference copies the reference).
- Limits: "style" is global; nothing ties it to an object or region; block roles are empirical.

**A6 MS-Diffusion**
- Problem: several reference subjects, each should appear in its own place without mixing.
- Idea: a grounding resampler combines each subject's image features with its phrase and box; image cross-attention is masked so each subject's tokens act only in its box.
- Useful because: this is reference ↔ region binding done explicitly.
- Limits: trained; subject identity, not attribute-only transfer; boxes, not sketches.

**A7 T³-S2S** (confirm details on reading)
- Problem: SDXL + ControlNet loses or merges instances in multi-instance sketches.
- Idea: three training-free adjustments: balance prompt-token embeddings across instances, emphasise instance-specific feature channels, and refine attention maps with the sketch contours.
- Useful because: same backbone, same adapter, training-free, and it is about sketch ↔ instance.
- Limits: no reference image; concept-art scenes.

**A8 Ctrl-X**
- Problem: structure from one image, appearance from another, text for semantics, without training or guidance gradients.
- Idea: run the structure and appearance images through the same diffusion model, inject structure features and self-attention, and transfer appearance statistics using self-attention correspondence between output and appearance image.
- Useful because: the correspondence between reference parts and output parts is computed, not assumed.
- Limits: correspondence is semantic similarity, so it needs the reference to contain something comparable; our neutral-object references (vase, mug) may not correspond well. Extra forward passes.

**A9 InstanceDiffusion**
- Problem: each instance has its own caption and its own location.
- Idea: per-instance (text, location) tokens fused into the UNet; location may be a point, box, mask or scribble; a multi-instance sampler reduces leakage.
- Useful because: it is the trained answer to "this phrase governs this place".
- Limits: SD 1.5-based and trained at scale; "scribble" means points along a stroke marking where an instance is, not a drawing of its shape; no reference image.

**A10 Uni-ControlNet**
- Problem: many condition types without one adapter each.
- Idea: one local adapter for all spatial conditions (multi-scale injection) and one global adapter that turns a CLIP image embedding into extra tokens next to the text tokens.
- Useful because: it is the trained version of our stack, and it reports which condition wins under conflict.
- Limits: global image condition is still global; conflicts are observed, not resolved.

---

## 3. Comparison table

| Approach | Backbone | Training | Text ↔ region | Sketch region ↔ entity | Reference ↔ region/entity | Handles conflict | T4-feasible |
|---|---|---|---|---|---|---|---|
| **Ours (Stage 1)** | SDXL | tiny MLP | implicit | no | no (global) | global strengths only | yes |
| ControlNet / T2I-Adapter | SD/SDXL | adapter | implicit | no | n/a | no | yes (inference) |
| IP-Adapter | SD/SDXL | adapter | implicit | n/a | no (global tokens) | no | yes |
| InstantStyle | SDXL | none | implicit | n/a | no, but separates style from layout by block | partly (content leak) | yes |
| Uni-ControlNet | SD 1.5 | 2 adapters | implicit | no | no | analysed only | inference |
| SmartControl | SD 1.5 | small predictor | implicit | spatial scale map | n/a | yes: text vs spatial, local | likely |
| Text-Anchored Score Composition | SD | none | via composition | partial | partial | yes | slow |
| DenseDiffusion | SD | none | explicit (masks) | if masks come from sketch | no | no | yes |
| Bounded Attention | SD/SDXL | none | explicit (boxes) | no | no | leakage only | heavy |
| Layout guidance / BoxDiff | SD | none | explicit (boxes) | no | no | no | heavy (UNet gradients) |
| GLIGEN / InstanceDiffusion / MIGC | SD 1.5 | large | explicit | InstanceDiffusion: scribble as location | GLIGEN: image per box | no | inference only |
| MS-Diffusion | SDXL | large | explicit | no | explicit (masked attention) | no | inference only |
| Ctrl-X | SDXL | none | implicit | structure image | computed correspondence | no | slow |
| T³-S2S | SDXL + ControlNet | none | improved | yes (instances) | no | no | probably |
| MaxFusion / Cross-ControlNet | SD (+FLUX) | none | implicit | no | no | yes: spatial vs spatial, per pixel | probably |
| DynamicControl / DynFusion / PixelPonder | SD / DiT | yes | implicit | no | no | selection of conditions | no |
| OminiControl / UniCombine / UNO | FLUX | LoRA | joint attention | no | token-level, implicit | no | no (our OOM) |

Main differences in one line each:
- **Where the binding lives**: in edited attention scores (DenseDiffusion, Bounded Attention), in trained grounding tokens (GLIGEN, InstanceDiffusion, MS-Diffusion), in a guidance energy (layout guidance), or in a computed feature correspondence (Ctrl-X).
- **How conflict is handled**: by a local strength map (SmartControl), by composing separate scores (TASC), by per-pixel feature fusion (MaxFusion, Cross-ControlNet), or by choosing which conditions to use (DynamicControl, DynFusion).
- **What the reference carries**: global style (InstantStyle), subject identity in a box (MS-Diffusion), or appearance matched part-to-part (Ctrl-X, Cross-Image Attention).

---

## 4. Mechanisms worth exploring for our project

**Could be reused almost as-is (training-free, SDXL, already in diffusers or a few lines)**
1. Block-specific IP-Adapter scales (InstantStyle). Our main loop passes a float; data prep already passes a per-block dict.
2. IP-Adapter attention masks, which restrict a reference to a region. I believe diffusers supports this (`ip_adapter_masks`); check the docs before relying on it.
3. Reading cross-attention maps (Prompt-to-Prompt style) as a **diagnostic**: where does each noun attend under our ControlNet? Stage 1 question 2.
4. Guidance interval for CFG, which overlaps with our hand schedule.
5. T2I-CompBench++ / GenEval-style binding metrics, and the CLIPSeg layout IoU already coded in `src/eval_metrics.py`.

**Could be adapted**
6. DenseDiffusion-style attention modulation with masks derived from the sketch.
7. A spatial map for ControlNet strength (SmartControl's idea), in place of our scalar.
8. Per-location fusion by feature statistics (MaxFusion) between ControlNet residuals and other signals.
9. An energy on cross-attention maps (layout guidance) added to our existing `guidance_delta`. Caveat: it needs gradients through the UNet, which our current guidance deliberately avoids.
10. An LLM/VLM that parses the caption into entities and attributes (LMD, RPG). We already run Qwen3-VL for captions and attributes.
11. Delayed reference conditioning, which FastComposer also uses, as support for our ramp.

**Which of these address the professor's concern**
- Text ↔ region: 3 (diagnose), 6, 9, 10.
- Sketch region ↔ entity: 6 with sketch-derived masks, and A7.
- Reference ↔ region or entity: 2, and the ideas in A6 and A8.
- Reference attribute without identity: 1, and DEADiff/B-LoRA as background.

---

## 5. Limitations of existing approaches for our three-modality setting

1. **Almost everything that binds explicitly needs boxes or masks.** A hand-drawn sketch gives strokes, not labelled regions. Someone has to turn strokes into regions and regions into entity labels, and the papers assume that is given.
2. **Conflict work is about spatial conditions.** SmartControl and TASC handle text versus one spatial map; MaxFusion, Cross-ControlNet, AtteConDA handle spatial versus spatial. None of the ones I found treats a reference image's colour or material contradicting the text.
3. **Reference methods transfer style or identity, not named attributes.** "Take the material from the reference but the colour from the text" is not what InstantStyle, IP-Adapter or MS-Diffusion do.
4. **Fusion papers mostly assume the conditions agree** and report conflict as a failure case (Uni-ControlNet, MixDiffusion).
5. **The strongest multi-condition models are diffusion transformers trained on large GPUs**, which is where our FLUX attempt failed.
6. **Guidance through the UNet is expensive.** Attention-energy methods that work on SD 1.5 at 512 px may not fit SDXL at 640 px on a T4.
7. **Training-free attention editing is fragile**: results depend on layer choice, step range and strength, and are usually demonstrated on SD 1.5.
8. **Evaluation**: binding benchmarks are text-only. I found no benchmark with labelled text/sketch/reference conflicts.

---

## 6. Research opportunities (not decisions)

Each is stated with what would have to be checked before claiming anything.

1. **Sketch strokes as the carrier of binding.** Derive regions from the sketch and bind caption entities and reference(s) to them. Check: T³-S2S, InstanceDiffusion's scribble mode, the DiT instance-sketch paper, and "Sketch-Guided Scene Image Generation" to see how much of this is done.
2. **Conflict as a local, attribute-typed quantity.** SmartControl makes text-vs-sketch conflict spatial. Extending "local conflict" to the reference (appearance) side is something I did not find. Check Cross-ControlNet and AtteConDA first.
3. **An explicit resolution policy.** Existing methods pick a winner implicitly. A system where the user or a rule states which modality wins for which attribute, and where that is measured, seems uncommon. Our 01b metadata already stores such a policy but nothing uses it.
4. **Attribute-level reference transfer.** Use only colour, or only material, from the reference, per entity. Check DEADiff, CSGO, ColorizeDiffusion and Composer's colour-palette condition.
5. **A measurement study on the frozen stack.** How often does SDXL + scribble ControlNet + IP-Adapter bind correctly for two-object sketches, and which component breaks it? Bounded Attention does this for text-only; I did not find it for the three-modality stack.
6. **The benchmark itself.** Our labelled conflict set (object, colour, material, reference swap) is small, but a three-modality conflict benchmark with a binding metric appears to be missing.
7. **Replacing the global router.** The literature's adaptive methods act per pixel, per block or per condition, never as four scalars per image. That supports the Stage 1 suspicion that the control space, not the MLP, is our ceiling.

Our earlier ideas, checked against the literature:
- "Per-step modality schedules" are well covered (guidance intervals, CFG schedulers, delayed conditioning, PixelPonder, DynFusion). Not new.
- "A learned network that sets condition strengths from the inputs" is close to SmartControl and DynamicControl, which do it with finer granularity. Not new in general.
- "Constraint guidance with edge and CLIP losses" is an instance of Universal Guidance / FreeDoM. Not new.

---

## 7. Shortlist to read in detail (in this order)

1. [SmartControl](https://arxiv.org/abs/2404.06451): closest to our router; conflict made spatial.
2. [Be Yourself: Bounded Attention](https://arxiv.org/abs/2403.16990): why binding fails, in both attention types.
3. [DenseDiffusion](https://arxiv.org/abs/2308.12964): the simplest explicit token ↔ region mechanism.
4. [InstantStyle](https://arxiv.org/abs/2404.02733): what our reference path is really injecting.
5. [T³-S2S](https://arxiv.org/abs/2412.13486): multi-instance sketches on our exact backbone.
6. [MS-Diffusion](https://arxiv.org/abs/2406.07209): reference ↔ region done explicitly.
7. [Text-Anchored Score Composition](https://arxiv.org/abs/2306.14408): misaligned conditions, training-free.
8. [Ctrl-X](https://arxiv.org/abs/2406.07540): structure + appearance + text with computed correspondence.
9. [Cross-ControlNet](https://iclr.cc/virtual/2026/poster/10011222) with [MaxFusion](https://arxiv.org/abs/2404.09977): per-pixel fusion under conflict; needed to judge opportunity 2.
10. [Uni-ControlNet](https://arxiv.org/abs/2305.16322): the trained baseline for our setting and its conflict analysis.

Background if time allows: [Prompt-to-Prompt](https://arxiv.org/abs/2208.01626), [GLIGEN](https://arxiv.org/abs/2301.07093), [InstanceDiffusion](https://arxiv.org/abs/2402.03290), [Universal Guidance](https://arxiv.org/abs/2302.07121).
