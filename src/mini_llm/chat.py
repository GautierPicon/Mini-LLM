import sys

import torch

from mini_llm.checkpoints import load_model, load_tokenizer, resolve_checkpoint
from mini_llm.config import get_device
from mini_llm.prompt import format_prompt


def reply(model, tok, eos, dev, question):
    ids = torch.tensor([tok.encode(format_prompt(question)).ids], device=dev)
    out = model.generate(ids, n=200, temp=0.7, top_k=40, eos=eos)
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