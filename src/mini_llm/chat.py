import sys

import torch

from mini_llm.checkpoints import load_model, load_tokenizer, resolve_checkpoint
from mini_llm.config import get_device
from mini_llm.prompt import format_prompt


TEMPERATURE = 0.5
TOP_K = 40
REPETITION_PENALTY = 1.15
MAX_NEW_TOKENS = 256


def reply(model, tok, eos, dev, question, ctx=None):
    """Answer one question on its own, with no conversation memory.

    Earlier turns make these small models anchor on their own answers, so the
    question alone is encoded (long questions keep the last ctx - 1 tokens).
    """
    if ctx is None:
        ctx = getattr(getattr(model, "c", None), "ctx", 512)
    ids = tok.encode(format_prompt(question)).ids[-(ctx - 1):]
    x = torch.tensor([ids], device=dev)
    out = model.generate(
        x, n=MAX_NEW_TOKENS, temp=TEMPERATURE, top_k=TOP_K, eos=eos, rep_penalty=REPETITION_PENALTY
    )
    return tok.decode(out[0, x.size(1):].tolist()).strip()


def chat(path=None):
    dev = get_device()
    path = resolve_checkpoint(path, pattern="mini-llm-assistant-*.pt")
    print(f"[{path.name}] Ctrl+C to quit, /quit to exit (each question is answered on its own)")
    model, cfg = load_model(path, dev)
    model.eval()
    ctx = cfg.ctx
    tok = load_tokenizer(path)
    eos = tok.token_to_id("<eos>")
    while True:
        try:
            question = input("\nYou > ")
        except (KeyboardInterrupt, EOFError):
            print()
            break
        cmd = question.strip().lower()
        if cmd in ("/quit", "/exit"):
            break
        if cmd == "/help":
            print("Commands: /quit exit, Ctrl+C quit. No conversation memory; each question is answered on its own.")
            continue
        if question.strip():
            answer = reply(model, tok, eos, dev, question, ctx)
            print("AI  >", answer)


def main():
    chat(sys.argv[1] if len(sys.argv) > 1 else None)


if __name__ == "__main__":
    main()
