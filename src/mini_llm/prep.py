import argparse
import json
import os
import queue
import random
import sys
import threading
import time

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from mini_llm.checkpoints import sha256_file
from mini_llm.config import DATA_DIR, MODELS_DIR, TOKENIZER_PATH, Cfg, get_preset

SOURCES = [
    ("HuggingFaceFW/fineweb-edu", "sample-10BT", 0.7),
    ("wikimedia/wikipedia", "20231101.en", 0.3),
]
SEED = 1
TOKENIZER_DOCS = 30_000
TOKENIZER_DOCS_LARGE_VOCAB = 300_000
VAL_TOKENS = 1_000_000
BATCH = 1_000


def texts(exhausted=None):
    rng = random.Random(SEED)
    streams = [
        [
            name,
            iter(
                load_dataset(name, subset, split="train", streaming=True).shuffle(
                    seed=SEED, buffer_size=10_000
                )
            ),
        ]
        for name, subset, _ in SOURCES
    ]
    weights = [w for _, _, w in SOURCES]
    while streams:
        i = rng.choices(range(len(streams)), weights=weights)[0]
        try:
            yield streams[i][0], next(streams[i][1])["text"]
        except StopIteration:
            if exhausted is not None:
                exhausted.append(streams[i][0])
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


def prefetch(items, size=8):
    q = queue.Queue(maxsize=size)
    done = object()

    def run():
        try:
            for it in items:
                q.put(it)
        except BaseException as e:
            q.put(e)
        finally:
            q.put(done)

    threading.Thread(target=run, daemon=True).start()
    while True:
        it = q.get()
        if it is done:
            return
        if isinstance(it, BaseException):
            raise it
        yield it


def train_tokenizer(vocab, n_docs):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab,
        special_tokens=["<eos>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    t0 = time.time()
    docs = (text for _, text in zip(range(n_docs), (t for _, t in texts())))
    tok.train_from_iterator(docs, trainer)
    print(f"tokenizer: vocab={vocab} from {n_docs} docs in {time.time() - t0:.0f}s")
    tok.save(str(TOKENIZER_PATH))


def prep(target_millions=120, size=None, vocab=None, tokenizer_docs=None):
    if vocab is None:
        vocab = get_preset(size).vocab if size else Cfg().vocab
    if tokenizer_docs is None:
        tokenizer_docs = TOKENIZER_DOCS_LARGE_VOCAB if vocab > Cfg().vocab else TOKENIZER_DOCS
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if TOKENIZER_PATH.exists():
        existing = Tokenizer.from_file(str(TOKENIZER_PATH)).get_vocab_size()
        if existing != vocab:
            raise SystemExit(
                f"{TOKENIZER_PATH} has vocab={existing} but --vocab={vocab} was requested. "
                "Move it aside (e.g. models/16k/tok.json) or drop --vocab."
            )
    else:
        print("Training the BPE tokenizer...")
        train_tokenizer(vocab, tokenizer_docs)
    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    eos = tok.token_to_id("<eos>")

    target = VAL_TOKENS + int(target_millions * 1_000_000)
    written = 0
    val_tokens, train_tokens = 0, 0
    docs_per_source = {}
    names = ", ".join(f"{name} ({w:.0%})" for name, _, w in SOURCES)
    print(f"Tokenizing {names} (target: {target / 1e6:.0f}M tokens)...")
    t0 = time.time()
    nb = 0
    exhausted = []
    with open(DATA_DIR / "val.bin", "wb") as val_f, open(DATA_DIR / "train.bin", "wb") as train_f:
        for batch in batches(prefetch(texts(exhausted)), BATCH):
            nb += 1
            val_ids, train_ids = [], []
            for src, text in batch:
                docs_per_source[src] = docs_per_source.get(src, 0) + 1
                for enc in tok.encode_batch([text]):
                    ids = enc.ids + [eos]
                    if written < VAL_TOKENS:
                        val_ids.extend(ids)
                        val_tokens += len(ids)
                    else:
                        train_ids.extend(ids)
                        train_tokens += len(ids)
                    written += len(ids)
            if val_ids:
                np.asarray(val_ids, dtype=np.uint16).tofile(val_f)
            if train_ids:
                np.asarray(train_ids, dtype=np.uint16).tofile(train_f)
            if nb % 20 == 0:
                el = max(1e-6, time.time() - t0)
                rate = written / el / 1e6
                eta = max(0, target - written) / max(1e-6, rate * 1e6) / 60
                print(f"  {written / 1e6:.1f}M / {target / 1e6:.0f}M tokens | {rate:.2f}M tok/s | ETA {eta:.1f}min", flush=True)
            if written >= target:
                break
    for src in exhausted:
        print(f"warning: source exhausted before target: {src}")
    meta = {
        "vocab": vocab,
        "tokenizer_sha256": sha256_file(TOKENIZER_PATH),
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "docs_per_source": docs_per_source,
    }
    (DATA_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    if written < target:
        print(f"STOPPED SHORT: {written / 1e6:.1f}M / {target / 1e6:.0f}M tokens (a source ran dry)")
        sys.stdout.flush()
        os._exit(1)
    print("Done.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("millions", type=float, nargs="?", default=120)
    ap.add_argument("--size", default=None)
    ap.add_argument("--vocab", type=int, default=None)
    ap.add_argument("--tokenizer-docs", type=int, default=None)
    args = ap.parse_args()
    prep(args.millions, args.size, args.vocab, args.tokenizer_docs)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()