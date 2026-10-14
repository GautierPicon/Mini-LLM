#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

SIZE=70M
STEPS=22000
PREP_MILLIONS=1500

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
  test -f models/tok.json || { echo "models/tok.json missing" >&2; exit 1; }
  setsid nohup uv run prep "$PREP_MILLIONS" >"$LOGDIR/prep.log" 2>&1 < /dev/null &
  echo "prep started -> $LOGDIR/prep.log"
  ;;
smoke)
  uv run pretrain 100 --size "$SIZE" --save-every 50 --milestone-every 1000000
  rm -f models/mini-llm-base-*-100steps*.pt models/milestones/mini-llm-base-*-100steps*.pt
  echo "smoke done, note the tok/s above"
  ;;
train)
  args=(uv run pretrain "$STEPS" --size "$SIZE")
  if compgen -G "models/mini-llm-base-*-${STEPS}steps.state.pt" > /dev/null; then
    args+=(--resume)
  fi
  setsid nohup "${args[@]}" >"$LOGDIR/train.log" 2>&1 < /dev/null &
  echo "train started (${args[*]}) -> $LOGDIR/train.log"
  ;;
sft)
  setsid nohup uv run sft 3 >"$LOGDIR/sft.log" 2>&1 < /dev/null &
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
  files=(models/tok.json)
  while IFS= read -r f; do files+=("$f"); done < <(find models -maxdepth 2 -name '*.pt' ! -name '*.state.pt' | sort)
  tar -cf /workspace/export.tar "${files[@]}"
  ls -lh /workspace/export.tar
  echo "runpodctl send /workspace/export.tar"
  ;;
*)
  echo "usage: $0 {setup|prep|smoke|train|sft|status|pack}"
  exit 1
  ;;
esac
