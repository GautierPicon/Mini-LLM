import argparse
import math
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch

from mini_llm.checkpoints import (
    cfg_from,
    rank,
    read_checkpoint,
    resolve_checkpoint,
    save_checkpoint,
    sha256_file,
)
from mini_llm.config import DATA_DIR, MODELS_DIR, TOKENIZER_PATH, get_preset, resolve_device, size_millions
from mini_llm.model import LLM

DATA_SEED = 1234
EVAL_WINDOWS = 1024

HPARAMS = {
    "34M": {"lr_max": 5e-4, "lr_cont": 3e-4, "lr_min": 5e-5, "warmup": 200},
    "70M": {"lr_max": 6e-4, "lr_cont": 3e-4, "lr_min": 6e-5, "warmup": 500},
    "150M": {"lr_max": 5e-4, "lr_cont": 3e-4, "lr_min": 5e-5, "warmup": 1000},
}


def default_batch(dev, size="34M"):
    if dev == "cuda":
        try:
            total = torch.cuda.get_device_properties(0).total_memory
            if total >= 40 * 1024**3:
                return (64, 4) if size == "150M" else (64, 2)
        except Exception:
            pass
        # Smaller CUDA GPUs, and documented OOM fallback for 150M on 80 GB:
        # batch=32, accum=8 keeps the same effective batch as (64, 4).
        return (32, 8) if size == "150M" else (32, 4)
    return 32, 4


def check_data_matches(meta_path, cfg, tokenizer_path=None):
    """Abort if tokenized data was made for another vocab/tokenizer.

    Old data directories have no meta.json: fall back to scanning the
    maximum token id, and only warn when everything fits.
    """
    import json

    if tokenizer_path is None:
        tokenizer_path = TOKENIZER_PATH
    if not meta_path.exists():
        print("warning: no data/meta.json (data predates vocab tracking)")
        return
    meta = json.loads(meta_path.read_text())
    if meta.get("vocab") != cfg.vocab:
        raise SystemExit(
            f"data/meta.json has vocab={meta.get('vocab')} but the {cfg.vocab} preset needs vocab={cfg.vocab}. "
            "Regenerate data/ with prep, or pick the matching --size."
        )
    if not tokenizer_path.exists():
        raise SystemExit(f"Tokenizer missing: {tokenizer_path}")
    actual = sha256_file(tokenizer_path)
    if meta.get("tokenizer_sha256") != actual:
        raise SystemExit(
            f"data/meta.json tokenizer {meta.get('tokenizer_sha256')} does not match {tokenizer_path} ({actual}). "
            "Regenerate data/ with prep, or restore the matching tok.json."
        )


def check_ids_fit(train, val, cfg):
    top = max(int(train.max()), int(val.max())) if len(train) and len(val) else -1
    if top >= cfg.vocab:
        raise SystemExit(
            f"data contains token id {top} but the preset needs vocab={cfg.vocab}. "
            "Regenerate data/ with prep, or pick the matching --size."
        )


def pretrain(steps=5000, resume=False, init=None, size="34M", batch=None, accum=None, lr=None,
             save_every=None, milestone_every=None, device="auto"):
    cfg = get_preset(size)
    hp = HPARAMS[size]
    lr_max, lr_cont, lr_min, warmup = hp["lr_max"], hp["lr_cont"], hp["lr_min"], hp["warmup"]
    dev = resolve_device(device)
    if dev == "cuda":
        torch.set_float32_matmul_precision("high")
    if save_every is None:
        save_every = 1000
    if milestone_every is None:
        milestone_every = 10000 if size == "150M" else 5000
    db, da = default_batch(dev, size)
    batch = db if batch is None else batch
    accum = da if accum is None else accum

    data = {
        s: np.memmap(str(DATA_DIR / f"{s}.bin"), dtype=np.uint16, mode="r")
        for s in ("train", "val")
    }
    train, val = data["train"], data["val"]
    check_data_matches(DATA_DIR / "meta.json", cfg)
    if not (DATA_DIR / "meta.json").exists():
        check_ids_fit(train, val, cfg)
    n_win = (len(train) - 1) // cfg.ctx
    if n_win <= 0:
        sys.exit(f"train.bin too small ({len(train)} tokens) for ctx {cfg.ctx}")

    model = LLM(cfg).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"mini-llm-base-{size_millions(n_params)}M-{steps}steps"
    weights_path = MODELS_DIR / f"{stem}.pt"
    state_path = MODELS_DIR / f"{stem}.state.pt"
    print(f"device: {dev}")
    print(f"{n_params / 1e6:.1f}M parameters -> {weights_path.name}")

    decay, nodecay = [], []
    for p in model.parameters():
        (decay if p.dim() >= 2 else nodecay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": nodecay, "weight_decay": 0.0}],
        lr=lr_max, betas=(0.9, 0.95), fused=(dev == "cuda"),
    )

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
    if lr is not None and phase_start == 0:
        lr_peak = lr
    # Warmup is at most 5% of the run, so short runs (smoke test) still reach the peak LR.
    warmup = min(warmup, max(1, (steps - phase_start) // 20))

    toks_step = batch * accum * cfg.ctx
    total_toks = toks_step * steps
    print(f"batch {batch}x{accum} | {toks_step} tokens/step")
    print(f"steps {steps} | total {total_toks / 1e9:.2f}B tokens | {total_toks / n_params:.1f} tokens/param")
    print(f"train {len(train) / 1e6:.1f}M tokens | {total_toks / max(1, len(train)):.1f} passes")
    if total_toks > 1.1 * len(train):
        print(f"warning: total tokens exceed 1.1x train.bin size ({len(train) / 1e6:.1f}M)")
    print(f"lr peak {lr_peak:g} | warmup {warmup}")

    n_seen = start * batch * accum
    epoch, pos = divmod(n_seen, n_win)
    perm = np.random.default_rng(DATA_SEED + epoch).permutation(n_win)

    def get_train_batch():
        nonlocal epoch, pos, perm
        starts = []
        for _ in range(batch):
            if pos >= n_win:
                epoch += 1
                perm = np.random.default_rng(DATA_SEED + epoch).permutation(n_win)
                pos = 0
            starts.append(int(perm[pos]) * cfg.ctx)
            pos += 1
        x = np.stack([train[s : s + cfg.ctx] for s in starts]).astype(np.int64)
        y = np.stack([train[s + 1 : s + 1 + cfg.ctx] for s in starts]).astype(np.int64)
        return torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev)

    n_vwin = (len(val) - 1) // cfg.ctx
    if n_vwin <= 0:
        eval_starts = np.array([0], dtype=np.int64)
    elif n_vwin <= EVAL_WINDOWS:
        eval_starts = np.arange(n_vwin, dtype=np.int64) * cfg.ctx
    else:
        eval_starts = np.sort(np.random.default_rng(0).choice(n_vwin, size=EVAL_WINDOWS, replace=False)).astype(np.int64) * cfg.ctx

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
        for k in range(0, len(eval_starts), batch):
            ss = eval_starts[k : k + batch]
            x = np.stack([val[s : s + cfg.ctx] for s in ss]).astype(np.int64)
            y = np.stack([val[s + 1 : s + 1 + cfg.ctx] for s in ss]).astype(np.int64)
            x, y = torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev)
            with amp:
                ls.append(cmodel(x, y)[1].item())
        model.train()
        return sum(ls) / len(ls)

    val_loss = None
    t0 = time.time()
    t_last, done_last = t0, start
    for step in range(start, steps):
        cur_lr = lr_at(step)
        for g in opt.param_groups:
            g["lr"] = cur_lr
        step_loss = torch.zeros((), device=dev)
        for _ in range(accum):
            x, y = get_train_batch()
            with amp:
                loss = cmodel(x, y)[1] / accum
            loss.backward()
            step_loss += loss.detach()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        loss_val = step_loss.item()
        gnorm_val = gnorm.item()
        if not math.isfinite(loss_val) or not math.isfinite(gnorm_val):
            print(f"!! non-finite value at step {step} (loss={loss_val}, gnorm={gnorm_val}). Stopping.")
            if state_path.exists():
                print(f"Last saved state: {state_path.name} (continue with --resume).")
            return
        opt.step()
        opt.zero_grad(set_to_none=True)

        # The first step includes torch.compile: start timing after it so that the
        # throughput and the ETA only reflect steady-state speed.
        if step == start:
            t_last, done_last = time.time(), step + 1
        if step % 50 == 0:
            now = time.time()
            head = f"step {step:5d}/{steps} | loss {loss_val:.3f} | gnorm {gnorm_val:.2f} | lr {cur_lr:.2e}"
            if step == start:
                print(f"{head} | warm-up step (compile), throughput from the next log | {now - t0:.0f}s", flush=True)
            else:
                rate = (step + 1 - done_last) * toks_step / max(1e-6, now - t_last)
                eta_h = (steps - step - 1) * toks_step / max(1.0, rate) / 3600
                print(f"{head} | {rate / 1000:.1f}k tok/s | {now - t0:.0f}s | ETA {eta_h:.1f}h", flush=True)
                t_last, done_last = now, step + 1
        done = step + 1
        if done % save_every == 0 or done == steps:
            val_loss = evaluate()
            print(f"  >> val loss {val_loss:.3f}", flush=True)
            save_checkpoint(
                state_path, model, cfg,
                kind="base", step=done, total_steps=steps, val_loss=val_loss,
                phase_start=phase_start, optimizer=opt.state_dict(),
            )
        if done % milestone_every == 0 and done != steps:
            mile_dir = MODELS_DIR / "milestones"
            mile_dir.mkdir(parents=True, exist_ok=True)
            save_checkpoint(
                mile_dir / f"{stem}.at{done}.pt", model, cfg,
                kind="base", step=done, total_steps=steps, val_loss=val_loss,
                phase_start=phase_start,
            )

    save_checkpoint(
        weights_path, model, cfg,
        kind="base", step=steps, total_steps=steps, val_loss=val_loss, phase_start=phase_start,
    )
    state_path.unlink(missing_ok=True)
    print(f"done -> {weights_path.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", type=int, nargs="?", default=5000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init")
    ap.add_argument("--size", default="34M")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--accum", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--save-every", type=int, default=None)
    ap.add_argument("--milestone-every", type=int, default=None)
    args = ap.parse_args()
    pretrain(args.steps, args.resume, args.init, args.size, args.batch, args.accum, args.lr,
             args.save_every, args.milestone_every, args.device)


if __name__ == "__main__":
    main()
    