import sys

import torch

from mini_llm.checkpoints import load_model, load_tokenizer, resolve_checkpoint
from mini_llm.config import get_device
from mini_llm.prompt import format_prompt


TEMPERATURE = 0.5
TOP_K = 40
REPETITION_PENALTY = 1.15
MAX_NEW_TOKENS = 256


def reply(model, tok, eos, dev, question):
    ids = torch.tensor([tok.encode(format_prompt(question)).ids], device=dev)
    out = model.generate(
        ids, n=MAX_NEW_TOKENS, temp=TEMPERATURE, top_k=TOP_K, eos=eos, rep_penalty=REPETITION_PENALTY
    )
    return tok.decode(out[0, ids.size(1):].tolist()).strip()


def chat(path=None):
    dev = get_device()
    path = resolve_checkpoint(path, pattern="mini-llm-assistant-*.pt")
    print(f"[{path.name}] Ctrl+C to quit")
    model, _ = load_model(path, dev)
    model.eval()
    tok = load_tokenizer(path)
    eos = tok.token_to_id("<eos>")
    while True:
        try:
            question = input("\nYou > ")
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if question.strip():
            print("AI  >", reply(model, tok, eos, dev, question))


def main():
    chat(sys.argv[1] if len(sys.argv) > 1 else None)


if __name__ == "__main__":
    main()