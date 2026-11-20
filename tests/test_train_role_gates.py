"""CPU checks for the role-gate trainer's data side (no SDXL needed)."""
import json
import os

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("diffusers")
import train_role_gates as TR


def _row(i, split="train", colors="red", material="plastic"):
    return dict(id=f"p{i}", split=split, category="hammer", caption=f"A {colors} hammer with a wooden handle.",
                ref_colors=colors, ref_material=material, ref_finish="glossy", ref_object="mug",
                target=f"target/p{i}.png", sketch=f"sketch/p{i}.png", reference=f"reference/p{i}.png", mask=f"mask/p{i}.png")


@pytest.fixture
def data_dir(tmp_path):
    for d in ("target", "sketch", "reference", "mask"):
        os.makedirs(tmp_path / d)
    rows = [_row(i) for i in range(3)]
    for r in rows:
        t = np.full((64, 64, 3), 230, np.uint8); t[16:48, 16:48] = (200, 30, 30)
        m = np.zeros((64, 64), np.uint8); m[16:48, 16:48] = 255
        ref = np.full((64, 64, 3), 127, np.uint8); ref[20:44, 20:44] = (210, 40, 40)
        sk = np.full((64, 64), 255, np.uint8); sk[16, 16:48] = 0
        Image.fromarray(t).save(tmp_path / r["target"]); Image.fromarray(m).save(tmp_path / r["mask"])
        Image.fromarray(ref).save(tmp_path / r["reference"]); Image.fromarray(sk).save(tmp_path / r["sketch"])
    with open(tmp_path / "metadata.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return str(tmp_path), rows


def test_build_rows_kinds_targets_and_rules(data_dir):
    d, rows = data_dir
    specs = TR.build_rows(rows, d, res=64)
    kinds = [s["kind"] for s in specs]
    assert kinds.count("normal") == 3 and kinds.count("col") == 3 and kinds.count("mat") == 3 and kinds.count("hue") == 3
    for s in specs:
        if s["kind"] == "col":
            assert s["rule"]["g_col"] == 1 and s["caption"] != rows[0]["caption"]
        if s["kind"] == "mat":
            assert s["rule"]["g_mat"] == 1
        if s["kind"] == "hue":
            base = next(b for b in specs if b["key"] == s["pair"])
            assert s["caption"] == base["caption"] and s["rule"]["g_col"] == 1          # stated colour now contradicts the reference
            assert np.abs(np.array(s["target"], int) - np.array(base["target"], int)).max() > 30     # recoloured target
            assert (np.array(s["target"])[0, 0] == np.array(base["target"])[0, 0]).all()           # background untouched
        assert s["mask"].shape == (64, 64)


def test_mask_helper():
    m = np.zeros((64, 64), bool); m[30:34, 30:34] = True
    t = TR._mask_t(m, 8, dilate=1)
    assert t.shape == (1, 1, 8, 8) and t.sum() > TR._mask_t(m, 8).sum()


def test_tiny_forward_backward_reaches_every_gate_group():
    """Tiny SDXL-shaped UNet + ControlNet + IP-Adapter: one training forward/backward with gradient checkpointing,
    attribute and swap losses; every gate group must receive gradient."""
    import torch
    from types import SimpleNamespace
    from diffusers import AutoencoderKL, ControlNetModel, DDPMScheduler, UNet2DConditionModel
    from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0
    from diffusers.models.embeddings import ImageProjection, MultiIPAdapterImageProjection
    from rolegate import losses as LS
    from rolegate.gates import RoleGates
    torch.manual_seed(0)
    D, E = 32, 16
    unet = UNet2DConditionModel(sample_size=16, block_out_channels=(32, 64), layers_per_block=1, norm_num_groups=8,
                                down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"), up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
                                cross_attention_dim=D, attention_head_dim=(2, 4), addition_embed_type="text_time",
                                addition_time_embed_dim=8, projection_class_embeddings_input_dim=6 * 8 + D)
    cn = ControlNetModel.from_unet(unet)
    with torch.no_grad():                                   # a fresh ControlNet's zero convs give all-zero residuals
        for m in list(cn.controlnet_down_blocks) + [cn.controlnet_mid_block]:
            m.weight.normal_(0, 0.1)
    unet.encoder_hid_proj = MultiIPAdapterImageProjection([ImageProjection(image_embed_dim=E, cross_attention_dim=D, num_image_text_embeds=4)])
    unet.register_to_config(encoder_hid_dim_type="ip_image_proj")
    procs = {}
    for n, p in unet.attn_processors.items():
        if n.endswith("attn2.processor"):
            hs = unet.config.block_out_channels[-1] if n.startswith("mid") else (
                unet.config.block_out_channels[int(n.split(".")[1])] if n.startswith("down") else
                list(reversed(unet.config.block_out_channels))[int(n.split(".")[1])])
            procs[n] = IPAdapterAttnProcessor2_0(hidden_size=hs, cross_attention_dim=D, num_tokens=(4,))
        else:
            procs[n] = p
    unet.set_attn_processor(procs)
    vae = AutoencoderKL(block_out_channels=(8, 8, 8, 8), norm_num_groups=8, latent_channels=4,
                        down_block_types=("DownEncoderBlock2D",) * 4, up_block_types=("UpDecoderBlock2D",) * 4)
    for m in (unet, cn, vae):
        m.requires_grad_(False)
    proj = torch.nn.Linear(3, 8)
    clip = SimpleNamespace(image=lambda x: torch.nn.functional.normalize(proj(x.mean((2, 3))), dim=-1))
    P = SimpleNamespace(dev="cpu", dt=torch.float32, res=128, unet=unet, controlnet=cn, vae=vae, clip=clip)
    role = RoleGates.for_unet(unet, [m.out_channels for m in cn.controlnet_down_blocks] + [cn.controlnet_mid_block.out_channels])
    # make the gates non-trivial so every path carries signal
    with torch.no_grad():
        for p in role.parameters():
            p.add_(0.01 * torch.randn_like(p))

    def sample(seed):
        g = torch.Generator().manual_seed(seed)
        groups = torch.zeros(77, dtype=torch.long); groups[1:4] = torch.tensor([1, 2, 3])
        m = torch.zeros(1, 1, 256, 256); m[..., 64:192, 64:192] = 1
        return dict(kind="normal", lat=torch.randn(1, 4, 16, 16, generator=g), ctrl=torch.rand(1, 3, 128, 128, generator=g),
                    txt=(torch.randn(1, 77, D, generator=g), torch.randn(1, 77, D, generator=g), torch.randn(1, D, generator=g),
                         torch.randn(1, D, generator=g)), groups=groups, emb=torch.randn(1, 2, E, generator=g),
                    flags_pred=torch.tensor([0.1, 0.9, 0.2]), flags_rule=torch.tensor([0., 1, 0]),
                    mask_lat=torch.ones(1, 1, 16, 16), mask256=m, tgt256=torch.rand(1, 3, 256, 256, generator=g), sk256=m,
                    obj_emb=torch.nn.functional.normalize(torch.randn(8, generator=g), dim=0),
                    ref_obj_emb=torch.nn.functional.normalize(torch.randn(8, generator=g), dim=0),
                    txt_neu={"colour": (torch.randn(1, 77, D, generator=g), torch.randn(1, D, generator=g)), "material": None})

    batch = [sample(0), sample(1)]
    sched = DDPMScheduler(num_train_timesteps=1000)
    unet.enable_gradient_checkpointing(); unet.train(); vae.enable_gradient_checkpointing()
    role.install(unet)
    try:
        t = torch.tensor([300])
        pred, noise, noisy, coll = TR.forward(P, role, batch, sched, t)
        assert pred.shape == (2, 4, 16, 16) and any(n.startswith("controlnet") for n, _ in coll)
        loss = LS.masked_diffusion(pred, noise, torch.cat([d["mask_lat"] for d in batch]))
        loss = loss + LS.prior_loss(coll, torch.cat([d["mask256"] for d in batch]), torch.stack([d["flags_rule"] for d in batch]))
        img = TR.decode_x0(P, sched, noisy, pred, t.repeat(2))
        assert img.shape == (2, 3, 256, 256)
        la, parts, emb = TR.attr_losses(P, img, batch)
        assert any(n == "pooled.text_colour" for n, _ in coll)
        loss = loss + la + LS.swap_loss(img[:1], img[1:], emb[:1], emb[1:], batch[0]["mask256"]) + LS.keep_loss(coll)
        role.ctx.collect = None
        loss.backward()
        # drop rows must also run (negative prompt, zero reference)
        with torch.no_grad():
            TR.forward(P, role, batch[:1], sched, t, drop=True)
    finally:
        role.remove(unet)
    cross = [m for m in role.cross.values()]
    gb = torch.stack([m.w_c.bias.grad for m in cross]).abs().sum(0)          # (6,) per gate group
    assert (gb > 0).all(), gb
    assert all(g.conv.weight.grad is not None and g.conv.weight.grad.abs().sum() > 0 for g in role.sketch)
    assert role.pooled.w_c.bias.grad is not None and role.pooled.w_c.bias.grad[0].abs() > 0      # colour pooled gate
    # material has no neutral caption here, so its pooled gate gets nothing
    assert all(type(p).__name__ != "GatedCrossAttnProcessor" for p in unet.attn_processors.values())


class _Tok:
    """Stand-in for the CLIP BPE tokenizer: lower-cases, splits words, 'beige' is two pieces."""
    def tokenize(self, text):
        import re
        out = []
        for w in re.findall(r"[a-z]+|[^a-z\s]", text.lower()):
            out += ["bei", "ge</w>"] if w == "beige" else [w + "</w>"]
        return out


def test_neutral_caption_keeps_token_positions():
    from rolegate.tokens import neutral_caption, token_groups, COLOUR
    tok = _Tok()
    cap = "A red mug with a beige wooden handle."
    nc = neutral_caption(tok, cap, "colour")
    assert nc == "A colored mug with a colored colored wooden handle."
    assert len(tok.tokenize(nc)) == len(tok.tokenize(cap))
    assert neutral_caption(tok, cap, "material") == "A red mug with a beige solid handle."
    assert neutral_caption(tok, "A mug.", "colour") is None
    g = token_groups(tok, cap, "mug")
    assert g[2] == COLOUR and g[6] == COLOUR and g[7] == COLOUR


def test_gate_maps_include_sketch_groups():
    import torch
    from rolegate.gates import gate_maps
    rec = {"down.attn2": torch.rand(2, 16, 6), "controlnet.0": torch.rand(2, 1, 8, 8), "controlnet.1": torch.rand(2, 1, 8, 8),
           "controlnet.2": torch.rand(2, 1, 4, 4), "controlnet.3": torch.rand(2, 1, 2, 2)}
    gm = gate_maps(rec)
    assert gm["sketch_contour"].shape == (4, 4) and "sketch_placement" in gm and "text_colour" in gm


def test_keep_loss_only_penalises_closing():
    import torch
    from rolegate import losses as LS
    assert LS.keep_loss([("controlnet.0", torch.full((1, 1, 4, 4), 1.5))]) == 0
    assert LS.keep_loss([("controlnet.0", torch.full((1, 1, 4, 4), 0.2))]) > 0
