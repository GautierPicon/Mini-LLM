import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

from mini_llm.config import MODELS_DIR, TOKENIZER_PATH, Cfg, get_device
from mini_llm.model import LLM


def resolve_checkpoint(path=None):
    if path is None:
        files = list(MODELS_DIR.glob("mini-llm-*.pt"))
        if not files:
            sys.exit(f"Aucun fichier mini-llm-*.pt dans {MODELS_DIR}. Lance d'abord train.")
        return max(files, key=lambda f: f.stat().st_mtime)
    p = Path(path)
    return p if p.exists() else MODELS_DIR / path


def gen(prompt, path=None):
    dev = get_device()
    path = resolve_checkpoint(path)
    print(f"[{path.name}]")
    ck = torch.load(path, map_location=dev, weights_only=False)
    cfg = Cfg(**ck["cfg"]) if isinstance(ck["cfg"], dict) else ck["cfg"]
    model = LLM(cfg).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    ids = torch.tensor([tok.encode(prompt).ids], device=dev)
    out = model.generate(ids, n=200)[0].tolist()
    print(tok.decode(out))


def main():
    gen(
        sys.argv[1] if len(sys.argv) > 1 else "Once upon a time",
        sys.argv[2] if len(sys.argv) > 2 else None,
    )


if __name__ == "__main__":
    main()