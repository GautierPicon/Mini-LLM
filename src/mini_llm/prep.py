import os
import sys

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from mini_llm.config import DATA_DIR, MODELS_DIR, TOKENIZER_PATH, Cfg

DATASET = "HuggingFaceFW/fineweb-edu"
SUBSET = "sample-10BT"
TOKENIZER_DOCS = 30_000
VAL_TOKENS = 1_000_000
BATCH = 1_000


def stream():
    return load_dataset(DATASET, name=SUBSET, split="train", streaming=True)


def batches(rows, size):
    buf = []
    for row in rows:
        buf.append(row["text"])
        if len(buf) == size:
            yield buf
            buf = []
    if buf:
        yield buf


def train_tokenizer(cfg):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=cfg.vocab,
        special_tokens=["<eos>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    docs = (row["text"] for _, row in zip(range(TOKENIZER_DOCS), stream()))
    tok.train_from_iterator(docs, trainer)
    tok.save(str(TOKENIZER_PATH))


def prep(target_millions=120):
    cfg = Cfg()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not TOKENIZER_PATH.exists():
        print("Entraînement du tokenizer BPE...")
        train_tokenizer(cfg)
    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    eos = tok.token_to_id("<eos>")

    target = VAL_TOKENS + int(target_millions * 1_000_000)
    written = 0
    print(f"Tokenisation de {DATASET} (objectif : {target / 1e6:.0f}M tokens)...")
    with open(DATA_DIR / "val.bin", "wb") as val_f, open(DATA_DIR / "train.bin", "wb") as train_f:
        for texts in batches(stream(), BATCH):
            for enc in tok.encode_batch(texts):
                arr = np.array(enc.ids + [eos], dtype=np.uint16)
                arr.tofile(val_f if written < VAL_TOKENS else train_f)
                written += len(arr)
            print(f"  {written / 1e6:.1f}M tokens", flush=True)
            if written >= target:
                break
    print("Terminé.")


def main():
    prep(float(sys.argv[1]) if len(sys.argv) > 1 else 120)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()