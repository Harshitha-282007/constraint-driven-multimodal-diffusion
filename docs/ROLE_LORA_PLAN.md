# Multi-role LoRA: the model learns which input governs which attribute

## Question (professor)

When several image and text inputs enter SDXL, how does the model understand, by itself, which input controls which attribute?

## Why the earlier attempts do not answer it

| Attempt | What it did | Why it fell short |
| --- | --- | --- |
| Router / gated fusion | 4 global scalars from CLIP cosines | no better than a hand schedule; cannot see colour conflicts |
| Role gates v1 to v3 | per-layer gates on text groups, reference channels, ControlNet residuals | gates moved <= 0.06; the effect was the hand-set start values |
| Colour adapter | palette tokens in 46 cross-attention layers | ignored the palette (identical output for red, green, silver) |
| Neutral caption + IP scale | strips caption colour words, tunes reference strength | works, but it is a rule applied from OUTSIDE the model |

## Method

A LoRA on the places where SDXL reads its inputs, trained on conflict examples with exact targets. Nothing is neutralised and no
detector is used, at training or at inference.

```
text       attn2.to_k, attn2.to_v        up blocks + mid block   (about 4.8 M parameters at rank 16)
reference  to_k_ip, to_v_ip              same layers             (about 3.4 M)
sketch     ControlNet frozen; role protected by the data and the preservation loss
```

Restricted by WHERE (input interfaces of the late layers), not by modality. The earlier whole-UNet LoRA learned the photo scene and lost the
sketch (edge IoU 0.369 to 0.230). `groups=("ref",)` is the reference-only ablation.

## Training signal (src/train_role_lora.py)

| Role | Where the supervision is |
| --- | --- |
| sketch to shape | every row: the reference object has another shape than the target; preservation loss on normal rows |
| text to identity | every row; preservation loss keeps identity |
| reference to colour, material | col / mat rows (caption names a wrong colour / material, target = photo) and hue rows (photo and reference hue-rotated together, caption stale, target = recoloured photo) |

No exact target exists for "the caption names another object than the sketch", so that conflict has no training rows. It is evaluated, not trained.

Loss: masked diffusion + 2 x colour (decoded x0, object mask) + 0.5 x texture + 0.5 x preservation (MSE to the same model with the LoRA off, same noise, normal rows only).

## Evaluation (src/eval_roles.py, notebooks/role_lora_cells.py)

For every conflict kind, did the input that should govern the attribute win? EvalKit with rule = "reference".

| Conflict | Governs | Metric |
| --- | --- | --- |
| obj | text | obj_hit, id_leak |
| col | reference | col_hit, col_leak |
| mat | reference | mat_hit |
| ref (swapped reference) | reference | col_hit |
| all rows | sketch | edge_iou |
| base rows | - | edge_iou, dino_tgt (no regression) |

Arms: `fixed` (baseline, constant strengths), `rule` (hand-fed neutralisation + IP scale 1.0), `lora` (learned), `lora_ref_only` (ablation).

## Known limits

- One training run (400 steps) is what fits tonight; two seeds are not possible.
- The object-vs-sketch conflict has no training rows.
- 12 held-out photos x 5 row kinds: indicative, not significant.
