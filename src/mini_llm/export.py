import argparse
import shutil
from pathlib import Path

import torch

from mini_llm.checkpoints import load_model, load_tokenizer, resolve_checkpoint
from mini_llm.configuration_mini_llm import MiniLlmConfig
from mini_llm.modeling_mini_llm import MiniLlmForCausalLM
from mini_llm.prompt import format_prompt

CHAT_TEMPLATE = (
    "{% for m in messages %}"
    "{{ 'User: ' + m['content'].strip() + '\\n' if m['role'] == 'user' "
    "else 'Assistant: ' + m['content'].strip() + '\\n' }}"
    "{% endfor %}{% if add_generation_prompt %}Assistant:{% endif %}"
)


def build_tokenizer(tok_path):
    from transformers import PreTrainedTokenizerFast

    tok = PreTrainedTokenizerFast(
        tokenizer_file=str(tok_path),
        eos_token="<eos>",
        pad_token="<eos>",
        unk_token=None,
        bos_token=None,
    )
    tok.chat_template = CHAT_TEMPLATE
    return tok


def export(checkpoint=None, out=None, verify=True):
    ckpt = resolve_checkpoint(checkpoint)
    print(f"checkpoint: {ckpt.name}")
    dev = "cpu"
    native, cfg = load_model(str(ckpt), dev)
    native.eval()
    tok_path = ckpt.parent / "tok.json"
    raw_tok = load_tokenizer(str(ckpt))
    eos = raw_tok.token_to_id("<eos>")

    hf_config = MiniLlmConfig.from_cfg(cfg)
    hf_config.auto_map = {
        "AutoConfig": "configuration_mini_llm.MiniLlmConfig",
        "AutoModelForCausalLM": "modeling_mini_llm.MiniLlmForCausalLM",
    }
    hf = MiniLlmForCausalLM(hf_config)
    missing, unexpected = hf.load_state_dict(native.state_dict(), strict=False)
    hf.tie_weights()
    if missing or unexpected:
        print(f"load_state_dict: missing={missing} unexpected={unexpected}")
        allowed_missing = {"head.weight", "cos", "sin"}
        leftover_missing = [k for k in missing if k not in allowed_missing]
        if leftover_missing or unexpected:
            raise SystemExit(f"weight mismatch: missing={missing} unexpected={unexpected}")

    if out is None:
        out = ckpt.parent.parent / "exports" / f"{ckpt.stem}-hf"
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    if verify:
        with torch.no_grad():
            sample = torch.tensor(
                [raw_tok.encode(format_prompt("Hello world")).ids[:32] or [eos]]
            )
            ref, _ = native(sample)
            got = hf(input_ids=sample).logits
            max_diff = (ref.float() - got.float()).abs().max().item()
            print(f"logits max abs diff: {max_diff:.3e}")
            if max_diff > 1e-5:
                raise SystemExit(f"export verification failed (diff {max_diff})")

    hf.save_pretrained(str(out), safe_serialization=True)

    tok = build_tokenizer(tok_path)
    tok.save_pretrained(str(out))

    from transformers import GenerationConfig

    gen = GenerationConfig(
        do_sample=True,
        temperature=0.5,
        top_k=40,
        repetition_penalty=1.15,
        max_new_tokens=256,
        eos_token_id=eos,
        pad_token_id=eos,
    )
    gen.save_pretrained(str(out))

    src_dir = Path(__file__).resolve().parent
    for name in ("configuration_mini_llm.py", "modeling_mini_llm.py"):
        shutil.copy(src_dir / name, out / name)

    if verify:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        rt_tok = AutoTokenizer.from_pretrained(str(out), trust_remote_code=True)
        rt = AutoModelForCausalLM.from_pretrained(
            str(out), trust_remote_code=True, dtype=torch.float32
        ).eval()
        with torch.no_grad():
            sample = torch.tensor([rt_tok.encode(format_prompt("Hello world"))[:32] or [eos]])
            a = hf(input_ids=sample).logits.float()
            b = rt(input_ids=sample).logits.float()
            diff = (a - b).abs().max().item()
            print(f"reload max abs diff: {diff:.3e}")
            if diff > 1e-5:
                raise SystemExit("reload verification failed")

    files = sorted(p.name for p in out.iterdir())
    print(f"done -> {out}\n  {', '.join(files)}")
    print("upload: huggingface-cli upload <org>/<repo> " + str(out) + " .")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", nargs="?")
    ap.add_argument("--out")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()
    export(args.checkpoint, args.out, verify=not args.no_verify)


if __name__ == "__main__":
    main()
