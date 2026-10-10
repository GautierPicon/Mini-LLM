import math
import random
import sys
import time
from dataclasses import asdict

import torch
from datasets import load_dataset

from mini_llm.chat import format_prompt
from mini_llm.checkpoints import load_model, load_tokenizer, resolve_checkpoint
from mini_llm.config import get_device


def load_examples(tok, eos, ctx):
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    rows = [d for d in ds if not d["context"].strip()]
    prompts = tok.encode_batch([format_prompt(d["instruction"]) for d in rows])
    answers = tok.encode_batch([" " + d["response"].strip() for d in rows])
    data = []
    for p, a in zip(prompts, answers):
        p_ids, a_ids = p.ids, a.ids + [eos]
        if len(p_ids) + len(a_ids) <= ctx:
            data.append((p_ids, a_ids))
    return data


def sft(epochs=3, base=None):
    dev = get_device()
    base = resolve_checkpoint(base)
    print(f"base: {base.name} | device: {dev}")
    model, cfg = load_model(base, dev)
    tok = load_tokenizer(base)
    eos = tok.token_to_id("<eos>")

    data = load_examples(tok, eos, cfg.ctx)
    print(f"{len(data)} exemples")

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
    per_epoch = len(data) // (bs * accum)
    total = per_epoch * epochs
    lr_max, lr_min, warmup = 2e-4, 2e-5, 50
    opt = torch.optim.AdamW(model.parameters(), lr=lr_max, betas=(0.9, 0.95), weight_decay=0.01)

    def lr_at(s):
        if s < warmup:
            return lr_max * (s + 1) / warmup
        p = (s - warmup) / max(1, total - warmup)
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * p))

    n_params = sum(p.numel() for p in model.parameters())
    out_path = base.parent / f"mini-llm-assistant-{n_params / 1e6:.0f}M.pt"

    model.train()
    step, t0 = 0, time.time()
    for ep in range(epochs):
        random.shuffle(data)
        for i in range(per_epoch):
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            step_loss = 0.0
            for j in range(accum):
                k = (i * accum + j) * bs
                x, y = make_batch(data[k : k + bs])
                loss = model(x, y)[1] / accum
                loss.backward()
                step_loss += loss.item()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not math.isfinite(step_loss) or not math.isfinite(gnorm.item()):
                print(f"!! valeur non finie au step {step}. Arrêt.")
                return
            opt.step()
            opt.zero_grad(set_to_none=True)
            if dev == "mps":
                torch.mps.empty_cache()
            if step % 25 == 0:
                print(f"epoch {ep + 1}/{epochs} | step {step:4d}/{total} | loss {step_loss:.3f} | {time.time() - t0:.0f}s")
            step += 1
        torch.save({"model": model.state_dict(), "cfg": asdict(cfg)}, out_path)
        print(f">> epoch {ep + 1} terminé -> {out_path.name}")


def main():
    sft(
        int(sys.argv[1]) if len(sys.argv) > 1 else 3,
        sys.argv[2] if len(sys.argv) > 2 else None,
    )


if __name__ == "__main__":
    main()