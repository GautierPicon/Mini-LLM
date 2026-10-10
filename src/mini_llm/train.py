import math
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict

import numpy as np
import torch

from mini_llm.config import DATA_DIR, MODELS_DIR, Cfg, get_device
from mini_llm.model import LLM


def train(steps=5000):
    cfg = Cfg()
    dev = get_device()
    print("device:", dev)
    batch, accum = (8, 8) if dev == "mps" else (32, 4)
    lr_max, lr_min, warmup = 5e-4, 5e-5, 200

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
    ckpt_path = MODELS_DIR / f"mini-llm-{n_params / 1e6:.0f}M-{steps}steps.pt"
    print(f"{n_params / 1e6:.1f}M paramètres -> {ckpt_path}")

    cmodel = torch.compile(model) if dev == "cuda" else model
    opt = torch.optim.AdamW(model.parameters(), lr=lr_max, betas=(0.9, 0.95), weight_decay=0.1)
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if dev == "cuda" else nullcontext()

    def lr_at(s):
        if s < warmup:
            return lr_max * (s + 1) / warmup
        p = (s - warmup) / max(1, steps - warmup)
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * p))

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

    t0 = time.time()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        for _ in range(accum):
            x, y = get_batch("train")
            with amp:
                loss = cmodel(x, y)[1] / accum
            loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not math.isfinite(loss.item()) or not math.isfinite(gnorm.item()):
            print(f"!! valeur non finie au step {step} (loss={loss.item()}, gnorm={gnorm.item()}). Arrêt.")
            return
        opt.step()
        opt.zero_grad(set_to_none=True)

        if step % 50 == 0:
            print(f"step {step:5d} | loss {loss.item() * accum:.3f} | gnorm {gnorm.item():.2f} | {time.time() - t0:.0f}s")
        if (step > 0 and step % 500 == 0) or step == steps - 1:
            print(f"  >> val loss {evaluate():.3f}")
            torch.save({"model": model.state_dict(), "cfg": asdict(cfg)}, ckpt_path)


def main():
    train(int(sys.argv[1]) if len(sys.argv) > 1 else 5000)


if __name__ == "__main__":
    main()