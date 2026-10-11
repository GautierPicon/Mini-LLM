import argparse
import math
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch

from mini_llm.checkpoints import cfg_from, rank, read_checkpoint, resolve_checkpoint, save_checkpoint
from mini_llm.config import DATA_DIR, MODELS_DIR, Cfg, get_device
from mini_llm.model import LLM

SAVE_EVERY = 500


def pretrain(steps=5000, resume=False, init=None):
    cfg = Cfg()
    dev = get_device()
    print("device:", dev)
    batch, accum = (4, 8) if dev == "mps" else (32, 4)
    lr_max, lr_cont, lr_min, warmup = 5e-4, 3e-4, 5e-5, 200

    data = {
        s: np.memmap(str(DATA_DIR / f"{s}.bin"), dtype=np.uint16, mode="r")
        for s in ("train", "val")
    }

    def get_batch(split):
        d = data[split]
        ix = np.random.randint(0, len(d) - cfg.ctx - 1, batch)
        x = np.stack([d[i : i + cfg.ctx] for i in ix]).astype(np.int64)
        y = np.stack([d[i + 1 : i + 1 + cfg.ctx] for i in ix]).astype(np.int64)
        return torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev)

    model = LLM(cfg).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"mini-llm-base-{n_params / 1e6:.0f}M-{steps}steps"
    weights_path = MODELS_DIR / f"{stem}.pt"
    state_path = MODELS_DIR / f"{stem}.state.pt"
    print(f"{n_params / 1e6:.1f}M parameters -> {weights_path.name}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr_max, betas=(0.9, 0.95), weight_decay=0.1)

    start, phase_start = 0, 0
    if resume and init:
        sys.exit("Use either --resume or --init, not both")
    if resume:
        if not state_path.exists():
            sys.exit(f"Nothing to resume: {state_path.name} not found")
        state = read_checkpoint(state_path, dev)
        if state["total_steps"] != steps:
            sys.exit(f"The saved run targets {state['total_steps']} steps, not {steps}")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        start = state["step"]
        phase_start = state.get("phase_start", 0)
        print(f"resuming from step {start}")
    else:
        if weights_path.exists() or state_path.exists():
            sys.exit(f"{stem} already exists. Use --resume to continue it, or delete its files to start over.")
        if init:
            init_path = resolve_checkpoint(init)
            ck = read_checkpoint(init_path, dev)
            if cfg_from(ck["cfg"]) != cfg:
                sys.exit(f"{init_path.name} was trained with a different configuration")
            phase_start = rank(init_path)[0]
            if phase_start >= steps:
                sys.exit(f"{init_path.name} already has {phase_start} steps, ask for more than that")
            model.load_state_dict(ck["model"])
            start = phase_start
            print(f"continuing from {init_path.name} (step {phase_start})")

    lr_peak = lr_max if phase_start == 0 else lr_cont

    cmodel = torch.compile(model) if dev == "cuda" else model
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if dev == "cuda" else nullcontext()

    def lr_at(s):
        i = s - phase_start
        if i < warmup:
            return lr_peak * (i + 1) / warmup
        p = (i - warmup) / max(1, steps - phase_start - warmup)
        return lr_min + 0.5 * (lr_peak - lr_min) * (1 + math.cos(math.pi * p))

    @torch.no_grad()
    def evaluate():
        model.eval()
        ls = []
        for _ in range(20):
            x, y = get_batch("val")
            with amp:
                ls.append(cmodel(x, y)[1].item())
        model.train()
        return sum(ls) / len(ls)

    val_loss = None
    t0 = time.time()
    for step in range(start, steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        for _ in range(accum):
            x, y = get_batch("train")
            with amp:
                loss = cmodel(x, y)[1] / accum
            loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not math.isfinite(loss.item()) or not math.isfinite(gnorm.item()):
            print(f"!! non-finite value at step {step} (loss={loss.item()}, gnorm={gnorm.item()}). Stopping.")
            if state_path.exists():
                print(f"Last saved state: {state_path.name} (continue with --resume).")
            return
        opt.step()
        opt.zero_grad(set_to_none=True)

        if step % 50 == 0:
            print(f"step {step:5d} | loss {loss.item() * accum:.3f} | gnorm {gnorm.item():.2f} | {time.time() - t0:.0f}s")
        done = step + 1
        if done % SAVE_EVERY == 0 or done == steps:
            val_loss = evaluate()
            print(f"  >> val loss {val_loss:.3f}")
            save_checkpoint(
                state_path, model, cfg,
                kind="base", step=done, total_steps=steps, val_loss=val_loss,
                phase_start=phase_start, optimizer=opt.state_dict(),
            )

    save_checkpoint(
        weights_path, model, cfg,
        kind="base", step=steps, total_steps=steps, val_loss=val_loss, phase_start=phase_start,
    )
    print(f"done -> {weights_path.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", type=int, nargs="?", default=5000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init")
    args = ap.parse_args()
    pretrain(args.steps, args.resume, args.init)


if __name__ == "__main__":
    main()