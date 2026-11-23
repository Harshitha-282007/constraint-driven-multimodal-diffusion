# IE 643 presentation: likely questions and answers

Every answer is grounded in the repo, the notebooks' saved outputs, or the checks made while preparing this guide. Where a number is shaky it is marked **(careful)**. Pair this with `PRESENTATION_STUDY_GUIDE.md`.

**Three rules for answering**
1. Name the limitation before they do (pilot size, sketch loss, no diffusion training).
2. Use "pilot" and "indicative", never "proves" or "outperforms".
3. If you don't know, say what you did measure, then what you would test next.

---

## A. The problem and the idea

**1. What problem are you solving?**
Three inputs can disagree: a caption, a sketch and a reference image. We want one object whose shape comes from the sketch, identity from the text, and colour and material from the reference, and the model must decide that automatically.

**2. What is your governing rule?**
Sketch → shape and placement. Text → object identity. Reference → colour and material, and the reference wins colour even when the caption names a different colour.

**3. Why does the reference win colour?**
It is a team design choice: the reference is direct visual evidence of appearance, while a caption colour word may be stale or contradictory. It reverses our earlier rule (stated text colour stays).

**4. Why one object only?**
The brief's example is a single object ("a red sports car"), and the professor did not ask for multi-object.

**5. What is novel in your work?**
The honest answer: (a) an explicit per-attribute governing rule enforced through the text path, (b) a trained conflict detector that decides when to apply it, and (c) the diagnosis that a caption colour re-enters through the ControlNet's text input even when the UNet text is neutralised. We don't claim a new generator.

---

## B. The submitted method (ARL)

**6. Explain your method in two sentences.**
A governance network reads CLIP features of the text, sketch and reference and outputs three conflict flags. The flags blend a colour- and material-neutral caption into every text path of a frozen SDXL + ControlNet + IP-Adapter, while a small router sets the sketch, reference and text strengths.

**7. What exactly is trained?**
Two small networks: the governance network (about 784 → 256 → 64 → 3 MLP over frozen CLIP features) and a router MLP (6 → 32 → 32 → 4, ensemble of 5). No diffusion weights are trained.

**8. Does that satisfy "submit a trained model"?**
It has trained components, but the generator is frozen. Whether that meets the brief was never confirmed with the TA. We also report the trained-inside-SDXL attempts and why they failed.

**9. How does the router relate to the governance network?**
They are parallel and independent. Both read the same CLIP features. The router outputs four global strengths; the governance network outputs the three flags. They meet only inside `Pipeline.run`. The router does not take the flags as input.

**10. What does the router output control?**
ControlNet scale (0.5–1.4), how long the sketch is held at full strength (0.4–1.0 of the steps), IP-Adapter strength (0.2–0.8) and classifier-free guidance (4–10).

**11. How do you "neutralise" the caption?**
Colour words become "colored" and material words become "solid", keeping the same token count. The embedding is then blended: `p' = p − f·(p − p_neutral)`, where `f` is the colour or material flag. It is applied to the UNet text, the ControlNet text and the pooled embedding.

**12. Why the ControlNet text too?**
The ControlNet received the full caption, so "purple" reached the UNet through its residuals even when the UNet text was neutral. Neutralising the ControlNet's text closes that path.

**13. Why rewrite the caption instead of deleting the colour token?**
CLIP's text encoder is causal. A colour word leaks into every later token and the pooled embedding, so removing one token does not remove the colour. Re-encoding a rewritten sentence changes all of them.

**14. Does your model neutralise the object?**
No. Only colour and material. Text should govern identity, so the object conflict is a sketch-versus-text problem, and the ARL has no mechanism for it. The `g_obj` flag exists but the ARL does not use it.

**15. What happens if I give no object name?**
The object-phrase slot falls back to the whole caption, so the pipeline still runs. The network was trained with a clean "a photo of a {object}" phrase, so the input is slightly off from training. We have not measured how much accuracy drops.

---

## C. The governance network

**16. How does it detect a conflict?**
Frozen CLIP ViT-L/14 embeds the caption, sketch and reference. The code also derives a greyscale reference and a "colour part" (`ref − grey`), plus an optional object phrase. A shared projection with slot embeddings, 15 pairwise cosines and the colour part's length go through an MLP to three flags.

**17. Is it "six inputs"?**
No. The user supplies three inputs (plus an optional object name). The other three rows are derived from the reference or the caption.

**18. How was it trained, and where did the labels come from?**
From metadata by word-list rules: the caption's colour or material words are compared with the reference's recorded colours or material. No images are generated for training. The network sees only CLIP vectors, never the word lists.

**19. How did you get 800 conflict rows from 200 photos?**
By editing the caption text or swapping which reference is paired with a photo (200 photos × 4 conflict types). The sketch and target images are reused. Notebook 05 expands this to roughly 4,000 rows with a fifth "agree" type.

**20. Is that a real evaluation of detection?**
It is evaluated on held-out test photos split by photo, so no photo's variants cross splits. But the variants are synthetic and correlated, and only 160 photos train it.

**21. How good is it?**
On the held-out test set: colour 0.755 (versus 0.540 for the old cosine detector), material 0.940 (versus 0.643). A caption-synonym test ("crimson" for "red") checks it beyond the word lists.

**22. But on your pilot photo the flags were wrong. Why?**
They were. The predicted flags matched the rule on only the material row: the object row got `g_obj` 0.01 (truth 1), and the colour row got `g_col` 0.30 (truth 1) with a spurious `g_obj` 0.92. Test-set averages hide per-photo misses, and one photo can be an unlucky case.

**23. What are the flags used for?**
`g_col` and `g_mat` drive the text blend. `g_obj` is unused in the ARL.

---

## D. Results and evaluation

**24. What are your results?**
On a pilot of one held-out photo and three conflict rows: the ARL hit the reference colour in 3 of 3 rows (fixed baseline 1 of 3) and the material in 2 of 3 (fixed 2 of 3). Edge IoU dropped from 0.378 to 0.059.

**25. Why only one photo?**
It was a pilot to test whether flag-gated text moves colour at all, with the selection rule fixed before any image was generated. A full evaluation on the 20 test photos and the 800 conflict rows is the next step.

**26. Is 3 of 3 meaningful?**
No, not statistically. It is indicative. We report it as a pilot.

**27. Your ARL lost the sketch. Isn't that a failure?**
It is the measured trade-off. The schedule relaxes the sketch to 30% after the hold time so the reference can set appearance. The flag gate alone keeps the sketch (edge IoU 0.370) but rarely changes colour. We report it rather than hide it.

**28. Is the learned part actually necessary?**
Honest answer: for colour, no. The schedule-only control also scores 3 of 3. The ARL adds material (0.67 versus 0.00 for the schedule) and conflict-aware behaviour, but one photo cannot show that the learned parts matter.

**29. Did the router help?**
Not clearly. In cross-validation on 28 items it beat a constant predictor on `cs` and barely on `hold`, and did not on `ip` or `cfg` **(careful: the README says 1 of 4 and the saved output shows 2 of 4)**. Its outputs vary little by conflict type, so it behaves close to the hand schedule.

**30. Which metrics did you use and what do they mean?**
Colour and material hit (does the output match the reference's attribute), edge IoU (sketch fidelity), DINO-ref (similarity to the reference, high may mean leakage), CLIP-T (text alignment). The brief also asks for perceptual quality and layout consistency; we did not report those for the ARL.

**31. Your ARL images look washed out.**
Fair. The reference-matching brown dominates the images and the object is small. We over-push toward the reference. A quality metric would show it; we did not run one on the ARL.

**32. Why didn't you evaluate on all test photos?**
Time, and the Colab T4 budget. The pilot code resumes and extends, so the full run is the next step. *(If you manage to run it before the talk, replace this with the new table.)*

---

## E. What failed and why

**33. Why did the fusion LoRA fail?**
It resolved object conflicts 8 of 8 (versus 2 of 8) but edge IoU fell 0.369 → 0.230. Its only target was the whole photo, and it sits in the layers where the ControlNet residuals enter, so it learned to ignore the sketch. That mechanism is our explanation, not a proven one.

**34. Why did the learned role gates fail?**
They barely trained. In v3 no gate moved more than 0.06 from its start, and the colour loss stayed flat. Our diagnosis: weak single-step decoded losses, a prior pulling gates back, a low learning rate, and the ControlNet caption leak. The v3 run was also cut at step 450 of 600.

**35. Did anything work in the gates?**
v2 fixed the sketch gates (they had saturated in v1). Colour still followed the text.

**36. What went wrong with the colour adapter?**
The palette was ignored: red, green and silver palettes gave identical output. With a zero-initialised adapter and a short, weak signal, it never learned to matter.

**37. What went wrong with the role LoRA?**
It had the highest edge IoU (0.753) but lost the photo (DINO to target 0.359 → 0.057) and the roles (colour 0.30, material 0.00). Images turned sepia or purple with hatched textures at scale 0.5 and pink and orange artefacts at 1.0 *(from an earlier session's reading of the output image)*. With 160 photos and one run we cannot separate over-fitting from a wrong loss.

**38. Are these failures definitive?**
No. Each was one short run on a small dataset with one seed. They show these designs did not work in our budget, not that they cannot work.

**39. Why not fix and keep one of them?**
Time. The ARL was the one approach that fit the constraints and gave a result.

---

## F. Choice and positioning

**40. Why did you pick the ARL over the others?**
It fits the brief's constraints (trained components, single object, free T4), doesn't sacrifice the sketch the way the LoRAs did, and rests on a diagnosed mechanism. It isn't the best on every metric.

**41. Why not the simpler neutral caption + IP scale 1.0?**
It had better edge IoU (0.376 on 12 photos) but is a hand-written rule with no trained decision. The ARL makes that decision learned (flags and router). We lost sketch fidelity in exchange. The two were not evaluated on the same items.

**42. Isn't this just prompt engineering?**
Partly: the mechanism is a text-path edit. The contribution is deciding automatically when to apply it and showing that the ControlNet text path must be covered too.

**43. How is this different from IP-Adapter or InstantStyle?**
Those inject a reference. We keep them frozen and add conflict detection plus text neutralisation, so the reference isn't competing with a contradicting caption.

**44. How does it compare with state of the art?**
We did not run a comparison against published multimodal methods. The baseline is our own fixed-weight SDXL + ControlNet + IP-Adapter.

---

## G. Data, setup and reproducibility

**45. What data do you use?**
200 Sketchy photos (160/20/20 split by photo), one object each. Captions from Qwen3-VL-2B. References are generated neutral objects carrying the photo's colour, material and finish.

**46. What went wrong with your reference images?**
v1 fed the whole photo into the IP-Adapter, so backgrounds leaked in, and random object swaps made hybrids. v2 segments the object onto grey (CLIPSeg + SAM), uses neutral objects, extracts colour/material/finish with Qwen3-VL, and filters candidates with DINO and Lab-histogram checks.

**47. Did you keep bad references?**
Yes, all 200 were kept, including 18 that failed the filters and 23 with unverified segmentation. The flags are stored in the metadata.

**48. Why not FLUX?**
FLUX.1-dev + UniCombine ran out of memory on the free T4 at the first training step.

**49. Can it run on a free GPU?**
Yes: about 10 GB, about 7 s per 512 px image at 20 steps on a Colab T4.

**50. Is there a demo?**
Yes: `notebooks/12_arl_live_demo.ipynb` (Gradio, needs a T4) and a recorded replay in `app/arl_demo_replay.html`. *(Test it before the talk.)*

**51. Anything wrong in your earlier slides?**
Two parameter values: the code used `RELAX = 0.3` and `GUIDE_W = 0.25`, while earlier slides said 0.7 and 0.03.

---

## H. Hard follow-ups

**52. "How does the model understand which part governs which part?"**
A trained network reads CLIP features of all three inputs and flags where they conflict. A fixed rule says who wins each attribute. The flags decide how much of the caption's colour and material survives, and a router sets the strengths. The model "knows" via learned detection plus a fixed rule, not learned weights inside SDXL, which we tried and could not train.

**53. "If the schedule alone gets the colour, why do you need the flags?"**
On colour, we couldn't show that on one photo. The flags add material control (0.67 versus 0.00) and stop stale caption colours when the reference should win. The full test would settle it.

**54. "What would you do with another month?"**
Run the full 20-photo evaluation with confidence intervals; add a perceptual-quality metric; fix the sketch loss (hold the sketch longer, or restrict relaxation to colour-deciding steps); handle the object conflict with a trained sketch-relaxation using `g_obj`; and train gates with a stronger per-timestep signal.

**55. "What is the biggest weakness?"**
Sample size and the sketch trade-off. The pilot is one photo, and the ARL's gain comes with an 84% drop in edge IoU (0.378 → 0.059).

**56. "Did you tune on the test photo?"**
The pilot photo selection rule was fixed before any image was generated (the first test photo, sorted by id, with all three conflict types). The router and governance network were trained on train photos only.

---

## Quick numbers to keep in your head

| Fact | Value |
|---|---|
| Photos / split | 200; 160/20/20 |
| Conflict rows | 800 (4 types × 200) |
| Governance detection | colour 0.755 (old 0.540), material 0.940 (old 0.643) |
| ARL pilot | colour 3/3, material 2/3, edge IoU 0.059 |
| Fixed baseline pilot | colour 1/3, material 2/3, edge IoU 0.378 |
| Fusion LoRA | objects 8/8 vs 2/8; edge IoU 0.369 → 0.230 |
| Role gates v3 | max gate movement 0.06; stopped at step 450/600 |
| Role LoRA | edge IoU 0.753; DINO-target 0.359 → 0.057; colour 0.30; material 0.00 |
| Neutral caption + IP 1.0 | object 0.10 → 0.30; edge IoU 0.213 → 0.376 |
| Runtime | T4, about 10 GB, about 7 s per image |
