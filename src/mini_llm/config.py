from dataclasses import dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
TOKENIZER_PATH = MODELS_DIR / "tok.json"


@dataclass
class Cfg:
    vocab: int = 8192
    dim: int = 512
    layers: int = 8
    heads: int = 8
    ffn: int = 1408
    ctx: int = 256


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"