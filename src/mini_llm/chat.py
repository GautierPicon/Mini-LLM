import sys

import torch

from mini_llm.checkpoints import load_model, load_tokenizer, resolve_checkpoint
from mini_llm.config import get_device
from mini_llm.prompt import format_conversation, format_prompt


TEMPERATURE = 0.5
TOP_K = 40
REPETITION_PENALTY = 1.15
MAX_NEW_TOKENS = 256


def build_prompt_ids(tok, history, question, ctx):
    """Encode the full conversation, dropping oldest turns if it exceeds ctx.

    Keeps at least the current question. history is a list of (q, a) tuples.
    """
    hist = list(history)
    while True:
        text = format_conversation(hist, question) if hist else format_prompt(question)
        ids = tok.encode(text).ids
        if len(ids) <= ctx - 1 or not hist:
            return ids, len(hist)
        hist.pop(0)


def reply(model, tok, eos, dev, question, history=None, ctx=None):
    history = history or []
    if ctx is None:
        ctx = getattr(getattr(model, "c", None), "ctx", 512)
    prompt_ids, _ = build_prompt_ids(tok, history, question, ctx)
    ids = torch.tensor([prompt_ids], device=dev)
    out = model.generate(
        ids, n=MAX_NEW_TOKENS, temp=TEMPERATURE, top_k=TOP_K, eos=eos, rep_penalty=REPETITION_PENALTY
    )
    return tok.decode(out[0, ids.size(1):].tolist()).strip()


def chat(path=None):
    dev = get_device()
    path = resolve_checkpoint(path, pattern="mini-llm-assistant-*.pt")
    print(f"[{path.name}] Ctrl+C to quit, /reset to clear history, /quit to exit")
    model, cfg = load_model(path, dev)
    model.eval()
    ctx = cfg.ctx
    tok = load_tokenizer(path)
    eos = tok.token_to_id("<eos>")
    history = []
    while True:
        try:
            question = input("\nYou > ")
        except (KeyboardInterrupt, EOFError):
            print()
            break
        cmd = question.strip().lower()
        if cmd in ("/quit", "/exit"):
            break
        if cmd == "/reset":
            history = []
            print("(history cleared)")
            continue
        if cmd == "/help":
            print("Commands: /reset clear history, /quit exit, Ctrl+C quit")
            continue
        if question.strip():
            answer = reply(model, tok, eos, dev, question, history, ctx)
            print("AI  >", answer)
            history.append((question.strip(), answer))


def main():
    chat(sys.argv[1] if len(sys.argv) > 1 else None)


if __name__ == "__main__":
    main()