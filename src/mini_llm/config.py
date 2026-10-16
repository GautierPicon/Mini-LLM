from dataclasses import dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
TOKENIZER_PATH = MODELS_DIR / "tok.json"


@dataclass
class Cfg:
    vocab: int = 16384
    dim: int = 512
    layers: int = 8
    heads: int = 8
    ffn: int = 1408
    ctx: int = 512


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_device(name):
    if name is None or name == "auto":
        return get_device()
    if name not in ("cuda", "cpu"):
        raise SystemExit(f"Unknown device {name!r}, expected one of: auto, cuda, cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested (--device cuda) but no GPU is available")
    return name


def size_millions(n_params):
    """Model size tag in millions: total params rounded to the nearest 10M.

    69,975,680 -> 70, 152,596,224 -> 150. The older 34M checkpoints predate
    this rule (34,087,424 would give 30) and keep their historical name.
    """
    return round(n_params / 1e7) * 10


PRESETS = {
    "34M": Cfg(),
    "70M": Cfg(dim=640, layers=12, heads=10, ffn=1728),
    "150M": Cfg(dim=768, layers=18, heads=12, ffn=2048, vocab=32768),
}


def get_preset(name):
    try:
        return PRESETS[name]
    except KeyError:
        raise SystemExit(f"Unknown size {name!r}, expected one of: {', '.join(sorted(PRESETS))}")