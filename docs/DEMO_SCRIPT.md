# Demo script: Attribute Routing Layer (about 3 minutes)

Team One Sheep Army, IE 643. Say the bold lines; the italics are what to do on screen.

## Before you start (do this 10 minutes early)
- Colab tab open and running, the gradio.live link loaded in another tab.
- Click one built-in Example once so the model is warm (first run is slower).
- Have `q_1_sketch.jpg` and `q_1_ref.jpg` ready in case you upload by hand.
- Keep the replay page `app/arl_demo_replay.html` open as a backup if the link fails.

## 1. Opening (20 s)
**"Our task is to generate one object from three inputs: a sketch, a text caption and a reference image. The problem is that these inputs often disagree. The caption may say black while the reference is brown, or say metal while the reference is leather."**

**"A fixed pipeline just mixes everything with constant weights, so one input silently wins. Our method, the Attribute Routing Layer, decides per input which source to trust."**

## 2. The rule (15 s)
*Point to the line "Rule: sketch -> shape..." on the page.*

**"The rule is fixed: the sketch gives the shape, the text gives the object identity, and the reference gives colour and material. Where they conflict, the reference wins colour."**

## 3. Colour conflict run (60 s)
*Click the first Example (caption ends in "black", reference is brown leather). Press Generate.*

**"The sketch is an axe. The caption says black. The reference is a brown leather box. While it runs: a frozen SDXL model with a scribble ControlNet and an IP-Adapter does the drawing. On top of it, a trained governance network looks at the six inputs and predicts three conflict flags: object, colour, material."**

*When it finishes, point to the two images, then to the text box.*

**"Left is the fixed baseline, right is the ARL. The colour flag fired, so the caption's colour word was blended toward a neutral word, and the router raised the reference strength. The right image takes the reference colour, brown, where the baseline stays closer to the text."**

*Read the flag values and router settings from the box. Do not quote numbers you have not read on screen.*

## 4. Material conflict run (40 s)
*Click the second Example (caption says "made of metal"). Generate.*

**"Same sketch, now the conflict is material. The caption says metal, the reference is leather. The material flag handles this one the same way, by neutralising the word 'metal' in the text path."**

## 5. What the router does (20 s)
**"The router is a small trained model. From six CLIP similarities between the inputs it outputs four settings: ControlNet strength, how long the sketch is held, IP-Adapter strength and guidance scale. So the weights change per example instead of being constant."**

## 6. Honest results (30 s)
**"Our pilot was three conflict rows on one photo, six arms. Colour hit rate: fixed 0.33, ARL 1.00. Material hit rate: fixed 0.67, ARL 0.67, and the flags-only arm reaches 1.00. The cost is sketch fidelity: edge IoU drops from 0.378 to 0.059. And no arm solved the object conflict. So this is a small pilot, not a general claim."**

## 7. Close (15 s)
**"The demo, code and poster are linked by the QR codes. Future work is attribute-specific routing, disentangled reference conditioning and spatially aware conditioning. Thank you."**

## Likely questions
- **Is the model trained?** Yes: the governance network and router are trained; SDXL, ControlNet and IP-Adapter are frozen.
- **Why does the sketch get lost?** Raising the reference strength and neutralising text pulls the image away from the sketch; the ARL trades shape fidelity for colour. We report that in the pilot.
- **Why only 3 rows?** Compute and time on a T4; we state n = 3 openly.
- **Did the fixed baseline get tuned?** It uses constant settings; the schedule control (B0) also reaches colour 1.00, so the colour gain is not unique to the router.
- **What if the link is slow?** Generation takes about 20-30 s per run on a T4; the replay page shows the recorded outputs.
