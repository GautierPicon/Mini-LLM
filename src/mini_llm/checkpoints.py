import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

from mini_llm.config import MODELS_DIR, Cfg
from mini_llm.model import LLM


def resolve_checkpoint(path=None, pattern="mini-llm-base-*.pt"):
    if path is None:
        files = list(MODELS_DIR.glob(pattern))
        if not files:
            sys.exit(f"Aucun fichier {pattern} dans {MODELS_DIR}")
        return max(files, key=lambda f: f.stat().st_mtime)
    p = Path(path)
    return p if p.exists() else MODELS_DIR / path


def load_model(path, dev):
    import __main__

    __main__.Cfg = Cfg
    ck = torch.load(path, map_location=dev, weights_only=False)
    cfg = Cfg(**ck["cfg"]) if isinstance(ck["cfg"], dict) else ck["cfg"]
    model = LLM(cfg).to(dev)
    model.load_state_dict(ck["model"])
    return model, cfg


def load_tokenizer(ckpt_path):
    return Tokenizer.from_file(str(Path(ckpt_path).parent / "tok.json"))