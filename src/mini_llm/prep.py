import os
import random
import sys

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from mini_llm.config import DATA_DIR, MODELS_DIR, TOKENIZER_PATH, Cfg

SOURCES = [
    ("HuggingFaceFW/fineweb-edu", "sample-10BT", 0.7),
    ("wikimedia/wikipedia", "20231101.en", 0.3),
]
SEED = 1
TOKENIZER_DOCS = 30_000
VAL_TOKENS = 1_000_000
BATCH = 1_000


def texts():
    rng = random.Random(SEED)
    streams = [
        iter(
            load_dataset(name, subset, split="train", streaming=True).shuffle(
                seed=SEED, buffer_size=10_000
            )
        )
        for name, subset, _ in SOURCES
    ]
    weights = [w for _, _, w in SOURCES]
    while streams:
        i = rng.choices(range(len(streams)), weights=weights)[0]
        try:
            yield next(streams[i])["text"]
        except StopIteration:
            del streams[i]
            del weights[i]


def batches(items, size):
    buf = []
    for item in items:
        buf.append(item)
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
    docs = (text for _, text in zip(range(TOKENIZER_DOCS), texts()))
    tok.train_from_iterator(docs, trainer)
    tok.save(str(TOKENIZER_PATH))


def prep(target_millions=120):
    cfg = Cfg()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not TOKENIZER_PATH.exists():
        print("Training the BPE tokenizer...")
        train_tokenizer(cfg)
    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    eos = tok.token_to_id("<eos>")

    target = VAL_TOKENS + int(target_millions * 1_000_000)
    written = 0
    names = ", ".join(f"{name} ({w:.0%})" for name, _, w in SOURCES)
    print(f"Tokenizing {names} (target: {target / 1e6:.0f}M tokens)...")
    with open(DATA_DIR / "val.bin", "wb") as val_f, open(DATA_DIR / "train.bin", "wb") as train_f:
        for batch in batches(texts(), BATCH):
            for enc in tok.encode_batch(batch):
                arr = np.array(enc.ids + [eos], dtype=np.uint16)
                arr.tofile(val_f if written < VAL_TOKENS else train_f)
                written += len(arr)
            print(f"  {written / 1e6:.1f}M tokens", flush=True)
            if written >= target:
                break
    print("Done.")


def main():
    prep(float(sys.argv[1]) if len(sys.argv) > 1 else 120)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()