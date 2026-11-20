# Role-gated SDXL: implementation plan

Goal: SDXL itself learns which input governs which attribute, and where. Answers the instructor's feedback by comparing fixed guidance, the learned router and router + constraints against a model whose routing is learned *inside* the network, on the same items, per conflict type, with a stated governing input.

Status: code written and CPU-tested only. Nothing has been trained or generated with it yet.

## Rules (agreed, fixed per conflict type)

| Attribute | Governing input |
|---|---|
| Shape, placement | Sketch |
| Object identity | Text |
| Colour, material | Reference, also when the caption names another colour or material |

## Architecture

Frozen: SDXL UNet, VAE, text encoders, scribble ControlNet, IP-Adapter (ViT-H). Trained: the gates only (well under 1 M parameters).

**1. Attribute channels per input** (a gate can only control what it can isolate)

| Input | Channels | How |
|---|---|---|
| Text | identity · colour words · material words · other | Word positions from the caption (`rolegate/tokens.py`) mapped to the 77 token slots |
| Reference | colour · texture/material | colour = E(flat grey) + E(ref) − E(grey ref) (SADis); texture = E(grey ref) with the reference object's text direction removed (InstantStyle subtraction, ViT-H text tower). Each channel is projected by IP-Adapter to its own 4 tokens |
| Sketch | contour (high-resolution residuals) · placement (low-resolution residuals and mid block) | ControlNet residuals, grouped by resolution |

**2. Gates** (`rolegate/gates.py`)

- Cross-attention (every `attn2` of the UNet): per-pixel gates `g = 2·σ(b0 + W_h·LN(h) + W_c·c)` for the 4 text groups and 2 reference channels. Text gates scale each token group's share of the attention output (softmax kept); each reference channel gets its own softmax attention and gate.
- ControlNet: per-pixel gate per residual, `g = 2·σ(b0 + conv1x1(r) + W_c·c)`.
- Conditioning `c` = governance-network conflict flags (object, colour, material) + timestep features.
- Init: weights zero; text and sketch gates start at 1 (current behaviour); reference gates start at 1 on the InstantStyle style block (`up_blocks.0.attentions.1`) and 0.1 elsewhere.

## Training data (`train_role_gates.py`)

From the 125 clean training photos, target always available:

| Kind | Construction | Target |
|---|---|---|
| normal | original triplet | photo |
| colour conflict | caption names a wrong colour (01b `make_color`) | photo (reference carries its colour) |
| material conflict | caption names a wrong material (01b `make_material`) | photo |
| hue pair | object hue rotated in CIELAB a*b*, same rotation applied to the reference; caption unchanged | recoloured photo |

Gate conditioning uses the governance network's *predicted* flags, so training sees detector errors.

## Losses and schedule

`L = L_diff(masked) + 0.5·L_attr + 0.2·L_swap + 0.1·L_prior (annealed to 0) + 0.01·L_reg`

- `L_diff`: noise prediction loss weighted 1.0 inside the dilated object mask, 0.1 outside (Break-A-Scene).
- `L_attr` (every 2nd step, t ∈ [200, 600], decode at 256 px): colour = CIELAB mean distance inside the mask to the target; texture = gradient-energy mean/std inside the mask; contour = sketch-edge recall plus outside-edge penalty; identity = CLIP-L hinge, caption object vs reference object.
- `L_swap` (every 4th step, hue pairs only): two forwards with the same noise and timestep but the two references; edges and identity embedding must agree.
- `L_prior`: reference gates high inside the object, low outside; text colour/material gates low when that flag is on; text identity gate high.
- `L_reg`: L1 to the initial gates.

| Phase | Steps | Data | Losses |
|---|---|---|---|
| warm-up | 0–150 | normal | diff + prior |
| conflict | 150–450 | all | + attr |
| exclusivity | 450–600 | all | + swap |

512 px, batch 1, accumulation 4, AdamW lr 5e-4, checkpoints every 150 steps. Memory test first.

## Evaluation (`09_role_gates_eval.ipynb`)

Same items for every arm: 20 test photos × 5 kinds = 100 items, 640 px, 20 steps, seed 0.

| Arm | Answers |
|---|---|
| fixed | instructor's baseline |
| router | learned global routing |
| router + constraints | do constraints help |
| role-gated | learned routing inside SDXL |
| role-gated, flags zeroed | is conflict detection needed |
| role-gated, text gates frozen at 1 | is damping contradicted words needed |

- Governed-attribute metric per conflict type; colour and material measured inside the CLIPSeg object mask against the reference; identity leak = image closer to the reference's object than the caption's.
- Paired bootstrap over base photos; win/loss counts per type.
- Figure rule: success = item at the median of the governed metric, failure = largest loss vs fixed, each with its gate maps.
- Gate knockout matrix on 8 photos: switch one gate group off, measure which attributes move.
- Bottleneck reading: router vs fixed → global weighting; role-gated vs router → local weighting; flags-zeroed → detection; router + constraints → constraints.

## Risks

- Object conflicts (text vs sketch) have no target image: identity and contour gates learn only from the attribute and swap losses.
- The reference colour/texture split is approximate; the knockout test measures the leak.
- Material has the weakest signal.
- Training time with decodes is an estimate (1.5–2 h on a T4).
