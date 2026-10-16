# Mini-LLM

Small assistants (~34M and ~70M parameters) trained **entirely from scratch** with PyTorch: no pretrained weights, no existing model. The project covers the whole chain: tokenizer, data, architecture, pretraining, supervised fine-tuning into an assistant, and chat.

Training runs on NVIDIA GPUs (CUDA).

> With 34M parameters and ~100M training tokens, the assistant writes coherent sentences in the right format, but it often gets facts wrong. The 70M model trained on ~1.44B tokens (cloud GPU) lowers base val loss from 3.618 to 3.030: answers are more fluent and better formatted, but factual errors and hallucinations remain.

## Model

Llama-style decoder: RMSNorm, RoPE (rotary position embeddings), causal attention, SwiGLU feed-forward, and an embedding matrix shared with the output head.

| Setting                   | 34M        | 70M        | 150M       |
| ------------------------- | ---------- | ---------- | ---------- |
| Parameters                | 34,087,424 | 69,975,680 | 152,596,224 |
| `dim` (width)             | 512        | 640        | 768        |
| `layers` (depth)          | 8          | 12         | 18         |
| `heads` (attention heads) | 8          | 10         | 12         |
| `ffn` (feed-forward size) | 1408       | 1728       | 2048       |
| Vocabulary (BPE)          | 16,384     | 16,384     | 32,768     |
| Context length            | 512 tokens | 512 tokens | 512 tokens |

Everything lives in `Cfg` (`PRESETS` in `src/mini_llm/config.py`).

Trained checkpoints in `models/`:

| Checkpoint                              | Steps | Tokens seen | Val loss |
| --------------------------------------- | ----- | ----------- | -------- |
| `mini-llm-base-34M-12000steps.pt`       | 12000 | ~0.20B      | 3.618    |
| `mini-llm-base-70M-22000steps.pt`       | 22000 | ~1.44B      | 3.030    |
| `mini-llm-assistant-34M.pt` (epoch 3)   | 5598  | SFT         | 3.194    |
| `mini-llm-assistant-70M.pt` (epoch 2/3) | 3800  | SFT         | 2.718    |

The 70M base was trained on the cloud (`cloud/run.sh`: `SIZE=70M`, `STEPS=22000`, `PREP_MILLIONS=1500`, CUDA batch `32x4x512`, so `65536` tokens/step). The 34M-12000steps model is a continued run (`--init` from 6000 steps, `phase_start=6000`), so its `val.bin` differs: base val losses are not directly comparable across data mixes.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```
git clone <repo-url>
cd mini-llm
uv sync
```

To speed up Hugging Face downloads, create a free token and export it before running `prep` or `sft`: `export HF_TOKEN=...`.

## Usage

The pipeline has four steps, run in this order on a cloud GPU
(see `Training on a cloud GPU` below).

```
uv run prep
uv run pretrain 300
uv run sft 1
uv run chat
```

| Command                                                  | What it does                                                                  | Notes                                   |
| -------------------------------------------------------- | ----------------------------------------------------------------------------- | --------------------------------------- |
| `uv run prep [millions]`                                 | Trains the tokenizer (if missing) and tokenizes a FineWeb-Edu + Wikipedia mix | Full data via `cloud/run.sh prep`       |
| `uv run pretrain [steps] [--resume] [--init checkpoint]` | Pretraining: learn to predict the next token                                  | Full run via `cloud/run.sh train`       |
| `uv run sft [epochs] [base] [--resume]`                  | Supervised fine-tuning on question/answer pairs                               | Full run via `cloud/run.sh sft`         |
| `uv run chat [checkpoint]`                               | Chat with the assistant (Ctrl+C to quit)                                      |                                       |
| `uv run export [checkpoint] [--out dir]`               | Exports a checkpoint to Hugging Face format (`config.json` + `model.safetensors` + tokenizer) | seconds |

To check that everything works before a long run: `uv run pretrain 300`, then `uv run sft 1`.

### Exporting to Hugging Face format

Training checkpoints (`.pt`, pickle) stay as-is: they carry optimizer state for `--resume`. `export` converts the finished weights to the deployable convention, with zero logit difference (checked automatically, `max abs diff` must be `< 1e-5` before and after a `from_pretrained` reload):

```
uv run export mini-llm-base-34M-12000steps.pt
uv run export mini-llm-assistant-34M.pt
uv run export mini-llm-base-70M-22000steps.pt
uv run export mini-llm-assistant-70M.pt
```

Each output directory is self-contained and Hub-ready: `config.json`, `model.safetensors` (tied `head` deduplicated), `tokenizer.json` + `tokenizer_config.json` (+ `chat_template.jinja` matching the `User:/Assistant:` training format), `generation_config.json` (same sampling as `chat.py`), plus `configuration_mini_llm.py` / `modeling_mini_llm.py` for `trust_remote_code`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("exports/mini-llm-assistant-34M-hf", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("exports/mini-llm-assistant-34M-hf", trust_remote_code=True)
```

Upload with `huggingface-cli upload <org>/<repo> exports/<name>-hf .`. `exports/` and `*.safetensors` are ignored by Git; `uv run pytest` covers the native/HF parity.

### Resuming a run

`pretrain` saves its full state (weights, optimizer, step) every 500 steps, and `sft` does it after every epoch. If a run stops, continue it with the same arguments plus `--resume`:

```
uv run pretrain 6000 --resume
uv run sft 3 --resume
```

The number of steps (or epochs) must match the original run, since the learning-rate schedule depends on it. Starting a `pretrain` run whose output already exists is refused, so a finished model is never overwritten by accident.

### Training on more data

A finished base model can be trained further on fresh data instead of starting over. `prep` keeps the existing `tok.json` and rewrites `data/` with a new random sample of the sources, then `--init` loads the weights of a finished model and runs more steps on it:

```
uv run prep 250
uv run pretrain 12000 --init mini-llm-base-34M-6000steps.pt
uv run sft 3
```

`12000` is the final step count, so this adds 6000 steps. The continued phase uses its own short warmup and a lower peak learning rate (3e-4), and the optimizer starts fresh. The result is saved as `mini-llm-base-34M-12000steps.pt`, which `sft` and `chat` then prefer over the 6000-step model. `--init` and `--resume` cannot be combined, and `--resume` works on a continued run like on any other.

Because the data mix changes, `val.bin` changes too: validation losses from before and after are not directly comparable.

### Training on a cloud GPU

This is the only supported path for training runs. On a RunPod pod with a persistent volume on `/workspace` (defaults: 150M preset, 34,332 steps, 4600M tokens):

```
cloud/run.sh setup
cloud/run.sh prep    # 4600M tokens, background
cloud/run.sh smoke   # 100-step check, note the tok/s
cloud/run.sh train   # 34,332 steps, background, adds --resume automatically
cloud/run.sh sft     # 2 epochs
cloud/run.sh status
cloud/run.sh pack
```

`--resume` from an old 34M `.state.pt` is no longer compatible (the optimizer now uses two parameter groups with different weight decay).

### Training a 150M model

Preset `150M` (`Cfg(dim=768, layers=18, heads=12, ffn=2048, vocab=32768)`, 152,596,224 parameters, embeddings shared): the 34M/70M presets are unchanged and their checkpoints still load with the 16k tokenizer.

- **Tokenizer (32k).** `models/tok.json` at the repo root is the 32k BPE, trained by `prep` (`--vocab 32768`, 300k documents, same normalizer/pre-tokenizer/`<eos>` as 16k). The old 16k lineage lives in `models/16k/` (`tok.json` + 34M/70M checkpoints); every checkpoint reads the `tok.json` next to it, so old models keep working.
- **Data budget.** 34,332 steps × 131,072 tokens/step ≈ 4.5B tokens (~29 tokens/param, ≤1 pass over `train.bin`). `prep` writes `data/meta.json` (`vocab`, tokenizer hash, train/val token counts); keep it next to the `.bin` files. `pretrain` aborts if the vocab or tokenizer hash does not match.
- **Run.** Everything goes through `cloud/run.sh` above. `sft` does 2 epochs (the 70M overfit at epoch 3). `pack` archives `models/tok.json`, the final 150M base + assistant and `data/meta.json` (milestones excluded, `.state.pt` excluded).
- **Comparing across tokenizers.** Val losses of the 16k and 32k lineages are **not comparable** (predicting among 32k tokens is a different task). Compare with bits per byte instead — see `mini-eval`, `wikitext` column.

## Project layout

```
mini-llm/
├── pyproject.toml
├── data/                  train.bin, val.bin, meta.json (tokens, generated by prep)
├── models/                tok.json (current lineage) + checkpoints (.pt)
├── models/16k/            tok.json + 34M/70M checkpoints (previous lineage)
└── src/
    └── mini_llm/
        ├── config.py      model configuration, paths, device selection
        ├── model.py       architecture and text generation
        ├── prompt.py      conversation format shared by sft and chat
        ├── identity.py    identity, small-talk and arithmetic examples for sft
        ├── prep.py        tokenizer and data preparation
        ├── pretrain.py    pretraining
        ├── sft.py         fine-tuning into an assistant
        ├── chat.py        chat interface
        ├── checkpoints.py saving, loading and selecting checkpoints
        ├── configuration_mini_llm.py  HF config (deploys as config.json)
        ├── modeling_mini_llm.py       HF model, same math as model.py (deploys as model.safetensors)
        └── export.py      .pt -> HF directory (config + safetensors + tokenizer)
```

## Checkpoints

| File                                             | Content                                                                                       |
| ------------------------------------------------ | --------------------------------------------------------------------------------------------- |
| `models/tok.json`                                | The tokenizer of the current lineage (32,768 tokens for 150M)                                 |
| `models/16k/tok.json`                            | The previous 16,384-token tokenizer, next to the 34M/70M checkpoints                           |
| `models/mini-llm-base-<size>-<steps>steps.pt`    | Base model, written once pretraining is complete (`34M-12000steps`, `70M-22000steps`)         |
| `models/mini-llm-base-<size>-<steps>steps.state.pt` | Resume state (includes optimizer), updated every 500 steps and deleted when the run completes |
| `models/mini-llm-assistant-<size>.pt`            | Best assistant so far, by validation loss (`assistant-34M`, `assistant-70M`)                  |
| `models/mini-llm-assistant-<size>.state.pt`      | Resume state of the SFT run, deleted when the run completes                                   |
| `data/train.bin`, `data/val.bin`, `data/meta.json` | Training data, already converted to tokens, plus vocab/tokenizer provenance                  |

Every checkpoint stores its weights, its configuration and metadata (`step`, `total_steps`, `val_loss`).

How checkpoints are chosen when no path is given:

- `sft` picks the base model with the highest step count in its file name, then the most recent file. A quick 300-step test therefore never takes priority over a 6000-step model. `.state.pt` files are never selected automatically.
- `chat` does the same among the assistant files.
- A path can always be passed explicitly, either absolute or relative to `models/`.

The tokenizer is always read from the folder that contains the checkpoint, so older models can be archived in a subfolder together with their own `tok.json`.

## Data

**Pretraining** (streamed and shuffled, ~70/30 by document count):

- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (`sample-10BT`): educational web pages written by humans.
- [Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) (`20231101.en`): encyclopedia articles written by humans, for factual knowledge.

**SFT**, mostly written by humans:

- [databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k): questions and answers written by Databricks employees (examples without extra context only).
- [No Robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots): 10,000 instructions written by professional annotators, including chat-style prompts.
- [OpenAssistant oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2): volunteer-written conversations. Only English, first turns, and the top-ranked answer to each prompt are kept.
- [alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned): **AI-written** (generated with an OpenAI model). It adds short, simple answers, but it is capped so that it never exceeds 25% of the mix (`AI_MAX_SHARE` in `sft.py`); human data always dominates.
- `identity.py`: examples written for this project (who the assistant is, greetings, thanks, how it handles "test", its limits) repeated three times, plus 200 generated arithmetic examples (sums, differences and times tables with correct answers). They are only added to the training split.

Examples that do not fit in the context window are dropped, and 500 examples are held out for validation. The conversation format is:

```
User: <question>
Assistant: <answer>
```

The SFT loss is computed on the answer only, not on the question.

## Generation settings

`chat.py` samples with a low temperature (0.5), `top_k` 40 and a repetition penalty of 1.15, and stops at the end-of-answer token or after 256 new tokens. The constants at the top of the file can be tuned without retraining anything: a higher temperature makes answers more varied and less reliable, a higher penalty reduces loops but can distort names and numbers.

## Design notes

Things to keep in mind when changing the code:

- Token ids are stored as `uint16` in the `.bin` files, so `Cfg.vocab` must stay at or below 65,536. Changing `vocab` invalidates `tok.json` and the data: delete them and run `prep` again.
- The prompt format is defined once, in `prompt.py`. Changing it changes the behaviour of every model already trained with it.
- Checkpoints store the configuration as a plain dictionary. Adding a field with a default value to `Cfg` keeps old checkpoints loadable; renaming or removing a field does not.
- Checkpoints are loaded with `weights_only=True`. Only legacy checkpoints that pickled the `Cfg` object itself fall back to unrestricted loading. Never load a `.pt` file from an untrusted source.
- `prep.py` ends with `os._exit(0)` on purpose: the Hugging Face streaming workers can keep the interpreter from exiting after the data is written. The files are already closed at that point.
- SFT masks the prompt (labels set to `-100`) and pads with the `<eos>` id, because the tokenizer only defines `<eos>` as a special token. Padded positions carry no loss.
- `*.pt`, `*.safetensors`, `*.bin`, `data/` and `exports/` are ignored by Git, and `models/tok.json` is tracked. Do not `git add -f` model files.
- `model.py` (training) and `modeling_mini_llm.py` (HF deploy) implement the same math; `tests/test_hf_export.py` checks logit parity. Change both together. The native RoPE buffers are non-persistent, the HF ones persistent (transformers' loader needs them in `model.safetensors`).
