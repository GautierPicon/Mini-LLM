import argparse
import math
import random
import sys
import time

import torch
from datasets import load_dataset

from mini_llm.checkpoints import (
    load_model,
    load_tokenizer,
    read_checkpoint,
    resolve_checkpoint,
    save_checkpoint,
)
from mini_llm.config import get_device
from mini_llm.identity import arithmetic_pairs, identity_pairs
from mini_llm.prompt import format_prompt

VAL_SIZE = 500
AI_MAX_SHARE = 0.25
IDENTITY_REPEAT = 3


def dolly_pairs():
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    return [(d["instruction"], d["response"]) for d in ds if not d["context"].strip()]


def no_robots_pairs():
    ds = load_dataset("HuggingFaceH4/no_robots")
    split = "train_sft" if "train_sft" in ds else "train"
    pairs = []
    for d in ds[split]:
        msgs = [m for m in d["messages"] if m["role"] != "system"]
        if len(msgs) >= 2 and msgs[0]["role"] == "user" and msgs[1]["role"] == "assistant":
            pairs.append((msgs[0]["content"], msgs[1]["content"]))
    return pairs


def oasst_pairs():
    ds = load_dataset("OpenAssistant/oasst2", split="train")
    ds = [d for d in ds if d["lang"] == "en" and not d["deleted"] and d["review_result"]]
    prompts = {d["message_id"]: d["text"] for d in ds if d["role"] == "prompter" and d["parent_id"] is None}
    return [
        (prompts[d["parent_id"]], d["text"])
        for d in ds
        if d["role"] == "assistant" and d["rank"] == 0 and d["parent_id"] in prompts
    ]


def alpaca_pairs():
    ds = load_dataset("yahma/alpaca-cleaned", split="train")
    return [(d["instruction"], d["output"]) for d in ds if not d["input"].strip()]


HUMAN_SOURCES = {"dolly": dolly_pairs, "no_robots": no_robots_pairs, "oasst2": oasst_pairs}
AI_SOURCES = {"alpaca": alpaca_pairs}


def encode_pairs(name, pairs, tok, eos, ctx):
    pairs = [(q, a) for q, a in pairs if q.strip() and a.strip()]
    prompts = tok.encode_batch([format_prompt(q) for q, _ in pairs])
    answers = tok.encode_batch([" " + a.strip() for _, a in pairs])
    kept = [
        (p.ids, a.ids + [eos])
        for p, a in zip(prompts, answers)
        if len(p.ids) + len(a.ids) + 1 <= ctx
    ]
    print(f"{name}: {len(kept)}/{len(pairs)} examples kept")
    return kept


def load_examples(tok, eos, ctx):
    human = []
    for name, fn in HUMAN_SOURCES.items():
        human += encode_pairs(name, fn(), tok, eos, ctx)
    ai = []
    for name, fn in AI_SOURCES.items():
        ai += encode_pairs(name, fn(), tok, eos, ctx)
    cap = int(len(human) * AI_MAX_SHARE / (1 - AI_MAX_SHARE))
    random.Random(0).shuffle(ai)
    ai = ai[:cap]
    print(f"AI-written examples used: {len(ai)} ({len(ai) / (len(human) + len(ai)):.0%} of the mix)")
    return human + ai


def sft(epochs=3, base=None, resume=False):
    dev = get_device()
    base = resolve_checkpoint(base)
    print(f"base: {base.name} | device: {dev}")
    model, cfg = load_model(base, dev)
    tok = load_tokenizer(base)
    eos = tok.token_to_id("<eos>")

    n_params = sum(p.numel() for p in model.parameters())

    data = load_examples(tok, eos, cfg.ctx)
    random.Random(0).shuffle(data)
    val, train = data[:VAL_SIZE], data[VAL_SIZE:]
    extra = encode_pairs(
        "identity and arithmetic",
        identity_pairs(round(n_params / 1e6), repeat=IDENTITY_REPEAT) + arithmetic_pairs(),
        tok, eos, cfg.ctx,
    )
    train = train + extra
    print(f"{len(train)} train examples, {len(val)} validation examples")

    def make_batch(items):
        xs, ys = [], []
        for p, a in items:
            ids = p + a
            labels = [-100] * len(p) + a
            xs.append(ids[:-1])
            ys.append(labels[1:])
        n = max(len(x) for x in xs)
        x = torch.tensor([x + [eos] * (n - len(x)) for x in xs])
        y = torch.tensor([y + [-100] * (n - len(y)) for y in ys])
        return x.to(dev), y.to(dev)

    bs, accum = 2, 8
    per_epoch = len(train) // (bs * accum)
    total = per_epoch * epochs
    lr_max, lr_min, warmup = 2e-4, 2e-5, 50
    opt = torch.optim.AdamW(model.parameters(), lr=lr_max, betas=(0.9, 0.95), weight_decay=0.01)

    def lr_at(s):
        if s < warmup:
            return lr_max * (s + 1) / warmup
        p = (s - warmup) / max(1, total - warmup)
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * p))

    @torch.no_grad()
    def evaluate():
        model.eval()
        losses = []
        for k in range(0, len(val), bs):
            x, y = make_batch(val[k : k + bs])
            losses.append(model(x, y)[1].item())
        model.train()
        if dev == "mps":
            torch.mps.empty_cache()
        return sum(losses) / len(losses)

    out_path = base.parent / f"mini-llm-assistant-{n_params / 1e6:.0f}M.pt"
    state_path = base.parent / f"mini-llm-assistant-{n_params / 1e6:.0f}M.state.pt"

    best = float("inf")
    step, start_epoch = 0, 0
    if resume:
        if not state_path.exists():
            sys.exit(f"Nothing to resume: {state_path.name} not found")
        state = read_checkpoint(state_path, dev)
        if state["total_steps"] != total:
            sys.exit(f"The saved run targets {state['total_steps']} steps, not {total}")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        step, start_epoch, best = state["step"], state["epoch"], state["best_val"]
        print(f"resuming after epoch {start_epoch} (step {step})")
    elif out_path.exists():
        print(f"note: {out_path.name} will be overwritten")

    model.train()
    t0 = time.time()
    for ep in range(start_epoch, epochs):
        random.shuffle(train)
        for i in range(per_epoch):
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            step_loss = 0.0
            for j in range(accum):
                k = (i * accum + j) * bs
                x, y = make_batch(train[k : k + bs])
                loss = model(x, y)[1] / accum
                loss.backward()
                step_loss += loss.item()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not math.isfinite(step_loss) or not math.isfinite(gnorm.item()):
                print(f"!! non-finite value at step {step}. Stopping.")
                return
            opt.step()
            opt.zero_grad(set_to_none=True)
            if dev == "mps":
                torch.mps.empty_cache()
            if step % 25 == 0:
                print(f"epoch {ep + 1}/{epochs} | step {step:4d}/{total} | loss {step_loss:.3f} | {time.time() - t0:.0f}s")
            step += 1
        val_loss = evaluate()
        print(f">> epoch {ep + 1} done | val loss {val_loss:.3f}")
        if val_loss < best:
            best = val_loss
            save_checkpoint(
                out_path, model, cfg,
                kind="assistant", step=step, epoch=ep + 1, total_steps=total, val_loss=val_loss,
            )
            print(f"   best so far, saved -> {out_path.name}")
        save_checkpoint(
            state_path, model, cfg,
            kind="assistant", step=step, epoch=ep + 1, total_steps=total,
            val_loss=val_loss, best_val=best, optimizer=opt.state_dict(),
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("epochs", type=int, nargs="?", default=3)
    ap.add_argument("base", nargs="?")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    sft(args.epochs, args.base, args.resume)


if __name__ == "__main__":
    main()