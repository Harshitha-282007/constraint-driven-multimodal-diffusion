"""Train Counterfactual Attribute Supervision on SDXL. Resumable: re-run the same command after any interruption.
python governance/train.py --data_dir /content/pilot_v1/<dir> --meta metadata_with_conflicts.jsonl --out runs/cas_v1
"""
import argparse, json, os, random, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch
from cas_data import CASDataset
from cas_model import select_trainable, trainable_state, atomic_save, rng_state, set_rng_state, l2sp
from cas_core import train_step


def run(C, ds, args, log=print):
    unet = C["unet"]
    names, params = select_trainable(unet, args.text_lora_rank)
    anchors = [p.detach().clone() for p in params]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    use_scaler = args.precision == "fp16" and C["dev"] == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    os.makedirs(args.out, exist_ok=True)
    state_path = os.path.join(args.out, "state.pt")
    step = 0
    if os.path.exists(state_path):
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        sd = unet.state_dict()
        for n, v in st["weights"].items(): sd[n].copy_(v)
        opt.load_state_dict(st["opt"]); step = st["step"]; anchors = [a for a in st["anchors"]]
        log(f"resumed from step {step}")
    cfg = dict(seed=args.seed, p_drop_text=0.1, p_drop_ref=0.1, p_drop_sketch=0.1, obj_weight=args.obj_weight,
               lambda_inv=args.lambda_inv, inv_every=args.inv_every)
    json.dump(vars(args), open(os.path.join(args.out, "args.json"), "w"), indent=1)
    unet.train(); unet.enable_gradient_checkpointing() if args.grad_ckpt else None
    t0 = time.time(); logf = open(os.path.join(args.out, "train_log.jsonl"), "a")
    amp_dt = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": None}[args.precision]
    while step < args.steps:
        rng = random.Random(args.seed * 7919 + step)
        sample = ds.sample(step)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(C["dev"], dtype=amp_dt, enabled=amp_dt is not None):
            out = train_step(C, sample, step, cfg, rng)
        loss = out.pop("loss")
        if args.l2sp > 0:
            loss = loss + args.l2sp * l2sp(params, anchors)
        scaler.scale(loss / args.accum).backward()
        if (step + 1) % args.accum == 0:
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update()
        step += 1
        rec = dict(step=step, kind=sample["kind"], loss=float(loss.detach()), **out, sec=round(time.time() - t0, 1))
        logf.write(json.dumps(rec) + "\n"); logf.flush()
        if step % args.log_every == 0: log(rec)
        if not np.isfinite(float(loss)):
            raise RuntimeError(f"non-finite loss at step {step}; restart from last checkpoint with lower --lr")
        if step % args.save_every == 0 or step == args.steps:
            w = trainable_state(unet, names)
            atomic_save(dict(weights=w, opt=opt.state_dict(), step=step, anchors=anchors, names=names), state_path)
            atomic_save(dict(weights=w, names=names, step=step), os.path.join(args.out, "cas_weights.pt"))
    return names


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True); ap.add_argument("--meta", default="metadata_with_conflicts.jsonl")
    ap.add_argument("--out", required=True); ap.add_argument("--res", type=int, default=640)
    ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--accum", type=int, default=4); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--precision", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--text_lora_rank", type=int, default=0)
    ap.add_argument("--obj_weight", type=float, default=2.0)
    ap.add_argument("--lambda_inv", type=float, default=0.1); ap.add_argument("--inv_every", type=int, default=4)
    ap.add_argument("--l2sp", type=float, default=10.0)
    ap.add_argument("--grad_ckpt", type=int, default=1)
    ap.add_argument("--log_every", type=int, default=20); ap.add_argument("--save_every", type=int, default=100)
    args = ap.parse_args()
    from cas_stack import load_pipe, components_from_pipe
    dtype = torch.float16
    pipe = load_pipe(dtype, "cuda")
    C = components_from_pipe(pipe, args.res, dtype, "cuda")
    ds = CASDataset(args.data_dir, args.meta, "train", args.res, seed=args.seed)
    run(C, ds, args)
