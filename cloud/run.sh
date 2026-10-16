#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

SIZE=150M
STEPS=34332
PREP_MILLIONS=4600
# 34332 x 131072 tokens/step = ~4.5B tokens, ~29 tokens/param.

# Expected tokenizer vocab for the current SIZE (16k lineage vs 32k lineage).
case "$SIZE" in
  150M) EXPECT_VOCAB=32768 ;;
  *) EXPECT_VOCAB=16384 ;;
esac

export UV_CACHE_DIR=/workspace/.uv-cache
export UV_LINK_MODE=copy
export HF_HOME=/workspace/.hf
export PYTHONUNBUFFERED=1

LOGDIR=/workspace/logs
mkdir -p "$LOGDIR" "$UV_CACHE_DIR" "$HF_HOME"

cmd="${1:-help}"
case "$cmd" in
setup)
  if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
  uv sync
  nvidia-smi
  uv run python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo "CUDA not available" >&2; exit 1; }
  ;;
prep)
  if [ -f data/meta.json ]; then
    have=$(uv run python -c "import json; print(json.load(open('data/meta.json')).get('vocab'))")
    if [ "$have" != "$EXPECT_VOCAB" ] && [ "${PREP_FORCE:-0}" != "1" ]; then
      echo "data/ holds vocab=$have, but SIZE=$SIZE needs vocab=$EXPECT_VOCAB." >&2
      echo "Move data/ aside, or set PREP_FORCE=1 to overwrite it." >&2
      exit 1
    fi
  fi
  if [ "$EXPECT_VOCAB" = "32768" ]; then
    vocab_args="--vocab 32768 --tokenizer-docs 300000"
  else
    vocab_args=""
  fi
  # shellcheck disable=SC2086
  setsid nohup uv run prep "$PREP_MILLIONS" $vocab_args >"$LOGDIR/prep.log" 2>&1 < /dev/null &
  echo "prep started -> $LOGDIR/prep.log"
  ;;
smoke)
  uv run pretrain 100 --size "$SIZE" --device cuda --save-every 50 --milestone-every 1000000
  rm -f models/mini-llm-base-*-100steps*.pt models/milestones/mini-llm-base-*-100steps*.pt
  echo "smoke done, note the tok/s above"
  ;;
train)
  args=(uv run pretrain "$STEPS" --size "$SIZE" --device cuda)
  if compgen -G "models/mini-llm-base-*-${STEPS}steps.state.pt" > /dev/null; then
    args+=(--resume)
  fi
  setsid nohup "${args[@]}" >"$LOGDIR/train.log" 2>&1 < /dev/null &
  echo "train started (${args[*]}) -> $LOGDIR/train.log"
  ;;
sft)
  setsid nohup uv run sft 2 --device cuda >"$LOGDIR/sft.log" 2>&1 < /dev/null &
  echo "sft started -> $LOGDIR/sft.log"
  ;;
status)
  nvidia-smi || true
  for f in "$LOGDIR"/*.log; do
    [ -e "$f" ] || continue
    echo "== $f =="
    tail -5 "$f"
  done
  echo "== models/ =="; ls -lh models/ || true
  echo "== models/milestones/ =="; ls -lh models/milestones/ || true
  echo "== data/ =="; ls -lh data/ || true
  df -h /workspace
  ;;
pack)
  files=(models/tok.json data/meta.json)
  while IFS= read -r f; do files+=("$f"); done < <(find models -maxdepth 2 -name 'mini-llm-base-150M-*.pt' ! -name '*.state.pt' | sort)
  while IFS= read -r f; do files+=("$f"); done < <(find models -maxdepth 2 -name 'mini-llm-assistant-150M.pt' | sort)
  # Milestones (~0.6 GB each) are excluded by default to keep the archive small.
  tar -cf /workspace/export.tar "${files[@]}"
  ls -lh /workspace/export.tar
  echo "runpodctl send /workspace/export.tar"
  ;;
*)
  echo "usage: $0 {setup|prep|smoke|train|sft|status|pack}"
  exit 1
  ;;
esac
