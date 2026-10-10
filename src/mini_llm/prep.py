import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from mini_llm.config import DATA_DIR, MODELS_DIR, TOKENIZER_PATH, Cfg


def prep():
    cfg = Cfg()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    train = load_dataset("roneneldan/TinyStories", split="train")
    val = load_dataset("roneneldan/TinyStories", split="validation")

    if not TOKENIZER_PATH.exists():
        print("Entraînement du tokenizer BPE...")
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=cfg.vocab,
            special_tokens=["<eos>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        it = (train[i : i + 1000]["text"] for i in range(0, 200_000, 1000))
        tok.train_from_iterator(it, trainer)
        tok.save(str(TOKENIZER_PATH))
    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    eos = tok.token_to_id("<eos>")

    for name, ds in [("val", val), ("train", train)]:
        print(f"Tokenisation {name}...")
        chunks = []
        for i in range(0, len(ds), 10_000):
            texts = ds[i : i + 10_000]["text"]
            for enc in tok.encode_batch(texts):
                chunks.append(np.array(enc.ids + [eos], dtype=np.uint16))
        arr = np.concatenate(chunks)
        arr.tofile(str(DATA_DIR / f"{name}.bin"))
        print(f"  {name}: {len(arr)/1e6:.1f}M tokens")


def main():
    prep()


if __name__ == "__main__":
    main()