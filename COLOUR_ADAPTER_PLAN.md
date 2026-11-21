# Method 2: ColourAdapter (colour-only reference channel)

## Why

The Method 1 report (leak test, three follow-ups) separated two problems:

- **Removal** of the caption's colour works when every caption colour path is neutral (ControlNet caption, UNet text gates, pooled embedding).
- **Injection** of the reference's colour does not: the only lever is the global IP-Adapter strength, and turning it up drags the reference's shape
  and background into the output (motorcycle and pistol fall apart). Gates on that vector did not learn colour (they moved <= 0.06 in v3).

So the reference needs a channel that carries colour and nothing else, and that channel has to be trained.

## What

```
reference --reference_mask--> object pixels --k-means in CIELAB--> palette (4 x [L, a, b, share])      coloradapter/palette.py
palette --MLP--> 4 colour tokens (width 128)
each up-block / mid-block cross-attention layer l (46 layers):                                         coloradapter/adapter.py
    out_l += alpha_l * W_o softmax(q K_l^T) V_l ,   K_l, V_l = linear_l(tokens) (no bias),  alpha_l = 0 at init
```

- Zero-init: before training the pipeline is unchanged. The unconditional (classifier-free) half gets no palette and adds exactly zero.
- Tokens are computed inside each processor from the raw palette, so gradient checkpointing never walks a graph twice.
- About 14 M trainable parameters. SDXL, ControlNet, IP-Adapter and the role gates are frozen. The IP-Adapter colour channel is switched off
  (`knock=("ref_colour",)`); its texture channel stays (material is not part of this method).

## Training (src/train_colour_adapter.py)

Every row has an exact target:

| kind | caption | reference palette | target |
| --- | --- | --- | --- |
| normal | original | reference | photo |
| hue (4 random rotations per photo, 40 to 320 degrees) | original (now names the WRONG colour) | hue-rotated reference | hue-rotated photo |

Each micro-step uses the stale caption or the colour-neutral caption at random. Loss: masked diffusion (object weighted) + 0.5 x CIELAB colour loss
on the decoded x0 inside the object mask (t in [200, 600], every second micro-step). 300 optimizer steps x 4 micro-batches, lr 5e-4
(alpha: 1e-2), checkpoint every 100 steps.

Inference rule (matches the project rule "the reference wins colour"): colour flag forced to 1 (neutral caption in ControlNet, UNet and pooled
embedding), `ref_colour` off, palette of the reference into the adapter. No conflict detector is needed for colour.

## Evaluation

`notebooks/colour_adapter_cells.py` cell 4: same three held-out rows as the Method 1 report. Metrics: CIELAB distance between the reference palette
and the output's palette (centre crop, lower is better), plus the figure. Compare: stale caption, caption colour off, colour adapter.

## Known limits

- Colour only. Material still comes from the IP-Adapter texture channel.
- The output palette is read from the central half of the image (the generated background is not flat), so background colour still counts a little; use the object mask for the final numbers.
- 125 training photos x (1 + up to 4) rows; hue rotations of grey objects are skipped (no colour to rotate).
