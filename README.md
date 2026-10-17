# Mini-LLM

Small assistants (~34M and ~70M parameters) trained **entirely from scratch** with
PyTorch: no pretrained weights, no existing model.

Training runs on NVIDIA GPUs (CUDA).

> With 34M parameters and ~100M training tokens, the assistant writes coherent
> sentences in the right format, but it often gets facts wrong. The 70M model trained
> on ~1.44B tokens (cloud GPU) lowers base val loss from 3.618 to 3.030: answers are
> more fluent and better formatted, but factual errors and hallucinations remain.

**Full documentation: [../Documentation/Mini-LLM](../Documentation/Mini-LLM/README.md)**

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```
git clone https://github.com/GautierPicon/Mini-LLM
cd Mini-LLM
uv sync

uv run prep          # tokenizer + data
uv run pretrain 300  # short smoke run
uv run sft 1
uv run chat
```

To speed up Hugging Face downloads, create a free token and export it before running
`prep` or `sft`: `export HF_TOKEN=...`.

Full training runs go through the cloud GPU workflow — see
[Cloud GPU](../Documentation/Mini-LLM/cloud-gpu.md). Everything else — architecture,
checkpoints, data sources, Hugging Face export, design notes — lives in the
[documentation](../Documentation/Mini-LLM/README.md).
