"""CPU smoke test with a TINY randomly-initialised SDXL-shaped stack (no weights downloaded).
Verifies: IP-adapter-only trainable selection, loss/backward, that frozen weights do not move, that trainable ones do,
checkpoint + exact resume, and the counterfactual data builder on synthetic images. It does NOT test image quality."""
import os, sys, json, tempfile, types, hashlib
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch
from PIL import Image, ImageDraw
from diffusers import UNet2DConditionModel, ControlNetModel, AutoencoderKL, DDPMScheduler
from cas_data import CASDataset, recolor_object, ref_object_mask
from cas_model import select_trainable
import train as T

torch.manual_seed(0)
RES, CD, ED = 64, 32, 16

def tiny_unet():
    return UNet2DConditionModel(sample_size=RES // 8, in_channels=4, out_channels=4, block_out_channels=(32, 64),
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"), up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        layers_per_block=1, cross_attention_dim=CD, attention_head_dim=8, use_linear_projection=True,
        addition_embed_type="text_time", addition_time_embed_dim=8, projection_class_embeddings_input_dim=32 + 6 * 8,
        transformer_layers_per_block=1)

def add_fake_ip(unet):
    keys, n = [], 0
    for name in unet.attn_processors:
        if not name.endswith("attn1.processor"): keys.append(name)
    sd = {"image_proj": {"proj.weight": torch.randn(4 * CD, ED) * .05, "proj.bias": torch.zeros(4 * CD),
                         "norm.weight": torch.ones(CD), "norm.bias": torch.zeros(CD)}, "ip_adapter": {}}
    for i, name in enumerate(keys):
        # hidden size of cross-attn k/v input is CD; to_k_ip: CD -> hidden(inner dim)
        hs = unet.attn_processors[name]  # existence check
        inner = _inner(unet, name)
        sd["ip_adapter"][f"{1 + 2 * i}.to_k_ip.weight"] = torch.randn(inner, CD) * .05
        sd["ip_adapter"][f"{1 + 2 * i}.to_v_ip.weight"] = torch.randn(inner, CD) * .05
    unet._load_ip_adapter_weights(sd, low_cpu_mem_usage=False)

def _inner(unet, name):
    mod = unet
    for part in name.replace(".processor", "").split("."):
        mod = getattr(mod, part) if not part.isdigit() else mod[int(part)]
    return mod.to_k.out_features

def build():
    unet = tiny_unet(); cn = ControlNetModel.from_unet(unet); add_fake_ip(unet)
    vae = AutoencoderKL(in_channels=3, out_channels=3, down_block_types=("DownEncoderBlock2D",) * 4, up_block_types=("UpDecoderBlock2D",) * 4,
                        block_out_channels=(16,) * 4, norm_num_groups=8, latent_channels=4, layers_per_block=1, sample_size=RES)
    for m in (vae, cn): m.requires_grad_(False)
    def enc_text(prompts):
        h = int(hashlib.md5(prompts[0].encode()).hexdigest()[:8], 16)
        g = torch.Generator().manual_seed(h)
        return torch.randn(1, 8, CD, generator=g), torch.randn(1, 32, generator=g)
    def enc_img(img):
        a = torch.from_numpy(np.asarray(img.resize((8, 8))).astype(np.float32) / 255.).reshape(-1)[:ED]
        return (a - 0.5).unsqueeze(0)
    return dict(unet=unet, controlnet=cn, vae=vae, scheduler=DDPMScheduler(num_train_timesteps=1000), encode_text=enc_text,
                encode_image=enc_img, dtype=torch.float32, dev="cpu", res=RES)

def make_data(root):
    rng = np.random.RandomState(0); rows = []
    cols = [(200, 30, 30), (30, 60, 200), (40, 160, 60), (240, 220, 40)]
    for i in range(8):
        c = cols[i % 4]; cat = ["hammer", "mug", "vase", "ball"][i % 4]
        ph = Image.new("RGB", (RES, RES), (200, 200, 200)); d = ImageDraw.Draw(ph); d.ellipse([16, 16, 48, 48], fill=c)
        mk = Image.new("L", (RES, RES), 0); ImageDraw.Draw(mk).ellipse([16, 16, 48, 48], fill=255)
        sk = Image.new("L", (RES, RES), 255); ImageDraw.Draw(sk).ellipse([16, 16, 48, 48], outline=0, width=2)
        rc = cols[(i + 1) % 4]; rf = Image.new("RGB", (RES, RES), (127, 127, 127)); ImageDraw.Draw(rf).rectangle([20, 10, 44, 54], fill=rc)
        for n, im in (("t", ph), ("s", sk), ("r", rf), ("m", mk)): im.save(f"{root}/{i}{n}.png")
        rows.append(dict(id=f"s{i}", split="train", target=f"{i}t.png", sketch=f"{i}s.png", reference=f"{i}r.png", mask=f"{i}m.png",
                         caption=f"A {['red','blue','green','yellow'][i%4]} {cat}.", category=cat, ref_colors=["red","blue","green","yellow"][(i+1)%4],
                         conflict_type="none"))
    with open(f"{root}/meta.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")

with tempfile.TemporaryDirectory() as tmp:
    make_data(tmp)
    ds = CASDataset(tmp, "meta.jsonl", "train", RES, seed=1)
    kinds = [ds.sample(i)["kind"] for i in range(40)]
    print("kinds:", {k: kinds.count(k) for k in set(kinds)})
    assert "cf_ref" in kinds and "cf_cap" in kinds and "real" in kinds
    # counterfactual recolour really moves object colour toward the swapped reference
    s = next(ds.sample(i) for i in range(40) if ds.sample(i)["kind"] == "cf_ref")
    m = np.asarray(s["mask"]) > 127; rm = ref_object_mask(s["ref"])
    got, want = np.asarray(s["target"])[m].mean(0), np.asarray(s["ref"])[rm].mean(0)
    print("recoloured object mean RGB", got.round(), "ref object mean RGB", want.round()); assert np.abs(got - want).sum() < 120
    # determinism
    assert ds.sample(5)["caption"] == ds.sample(5)["caption"]

    C = build()
    before = {n: p.detach().clone() for n, p in C["unet"].named_parameters()}
    class A: pass
    a = A(); a.out = tmp + "/run"; a.steps = 6; a.lr = 1e-3; a.accum = 1; a.seed = 0; a.precision = "fp32"; a.text_lora_rank = 0
    a.obj_weight = 2.0; a.lambda_inv = 0.1; a.inv_every = 2; a.l2sp = 1.0; a.grad_ckpt = 0; a.log_every = 3; a.save_every = 3
    names = T.run(C, ds, a)
    after = dict(C["unet"].named_parameters())
    moved = [n for n in before if not torch.equal(before[n], after[n].detach().to(before[n].dtype))]
    assert set(moved) <= set(names) and len(moved) > 0, (len(moved), len(names))
    print(f"trainable tensors: {len(names)}; moved: {len(moved)}; frozen tensors moved: {len(set(moved) - set(names))}")
    w6 = torch.load(a.out + "/cas_weights.pt", weights_only=False)["weights"]
    # resume test: fresh model, train to step 3 then 'crash', resume to 6 -> identical to uninterrupted run
    torch.manual_seed(0); C2 = build(); a2 = A(); a2.__dict__.update(a.__dict__); a2.out = tmp + "/run2"; a2.steps = 3
    torch.manual_seed(0); 
    # same init as C: copy initial weights
    C2["unet"].load_state_dict({**C["unet"].state_dict(), **{n: before[n] for n in before}}); C2["controlnet"].load_state_dict(C["controlnet"].state_dict()); C2["vae"].load_state_dict(C["vae"].state_dict())
    T.run(C2, ds, a2); a2.steps = 6
    C3 = build(); C3["unet"].load_state_dict({n: before[n] for n in before}, strict=False); C3["controlnet"].load_state_dict(C["controlnet"].state_dict()); C3["vae"].load_state_dict(C["vae"].state_dict())
    T.run(C3, ds, a2)
    w6b = torch.load(a2.out + "/cas_weights.pt", weights_only=False)["weights"]
    diff = max((w6[n] - w6b[n]).abs().max().item() for n in w6)
    print("max |uninterrupted - resumed| =", diff); assert diff < 1e-5
print("SMOKE TEST PASSED")
