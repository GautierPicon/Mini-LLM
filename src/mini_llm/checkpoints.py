import hashlib
import pickle
import re
import sys
from dataclasses import asdict, fields
from pathlib import Path

import torch
from tokenizers import Tokenizer

from mini_llm.config import MODELS_DIR, Cfg
from mini_llm.model import LLM

STEPS_RE = re.compile(r"-(\d+)steps")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rank(path):
    m = STEPS_RE.search(path.name)
    return (int(m.group(1)) if m else 0, path.stat().st_mtime)


def resolve_checkpoint(path=None, pattern="mini-llm-base-*.pt"):
    if path is None:
        files = [f for f in MODELS_DIR.rglob(pattern) if not f.name.endswith(".state.pt")]
        if not files:
            sys.exit(f"No {pattern} file under {MODELS_DIR}")
        return max(files, key=rank)
    p = Path(path)
    return p if p.exists() else MODELS_DIR / path


def read_checkpoint(path, dev):
    try:
        return torch.load(path, map_location=dev, weights_only=True)
    except pickle.UnpicklingError:
        import __main__

        __main__.Cfg = Cfg
        return torch.load(path, map_location=dev, weights_only=False)


def cfg_from(raw):
    if not isinstance(raw, dict):
        return raw
    known = {f.name for f in fields(Cfg)}
    return Cfg(**{k: v for k, v in raw.items() if k in known})


def load_model(path, dev):
    ck = read_checkpoint(path, dev)
    cfg = cfg_from(ck["cfg"])
    model = LLM(cfg).to(dev)
    model.load_state_dict(ck["model"])
    return model, cfg


def save_checkpoint(path, model, cfg, **meta):
    tmp = path.with_name(path.name + ".tmp")
    torch.save({"model": model.state_dict(), "cfg": asdict(cfg), **meta}, tmp)
    tmp.replace(path)


def load_tokenizer(ckpt_path):
    return Tokenizer.from_file(str(Path(ckpt_path).parent / "tok.json"))