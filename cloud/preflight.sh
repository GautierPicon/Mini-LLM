#!/usr/bin/env bash
# Preflight check before `cloud/run.sh prep` (150M default).
# Verifies the exact guards that would make prep refuse to run:
#   1. data/meta.json vocab vs SIZE (run.sh)
#   2. models/tok.json vocab vs --vocab (prep.py)
# Read-only: never modifies data/ or models/.
set -euo pipefail
cd "$(dirname "$0")/.."

SIZE="${SIZE:-150M}"
case "$SIZE" in
  150M) EXPECT_VOCAB=32768 ;;
  *) EXPECT_VOCAB=16384 ;;
esac

fail=0
say() { echo "$1"; }
warn() { echo "!! $1" >&2; fail=1; }

say "SIZE=$SIZE EXPECT_VOCAB=$EXPECT_VOCAB"

# 1. data/meta.json (guard in run.sh prep)
if [ -f data/meta.json ]; then
  have=$(uv run python -c "import json; print(json.load(open('data/meta.json')).get('vocab'))")
  say "data/meta.json vocab=$have"
  if [ "$have" != "$EXPECT_VOCAB" ]; then
    warn "data/ holds vocab=$have but SIZE=$SIZE needs vocab=$EXPECT_VOCAB."
    warn "Move data/ aside, or run with PREP_FORCE=1."
  fi
else
  say "data/meta.json: absent (fresh volume, prep will create vocab=$EXPECT_VOCAB)"
fi

# 2. models/tok.json (guard in prep.py)
if [ -f models/tok.json ]; then
  existing=$(uv run python -c "from tokenizers import Tokenizer; print(Tokenizer.from_file('models/tok.json').get_vocab_size())")
  say "models/tok.json vocab=$existing"
  if [ "$existing" != "$EXPECT_VOCAB" ]; then
    warn "models/tok.json has vocab=$existing but prep needs vocab=$EXPECT_VOCAB."
    warn "Move it aside (e.g. models/16k/tok.json) or drop --vocab."
  fi
else
  say "models/tok.json: absent (prep will train vocab=$EXPECT_VOCAB)"
fi

# 3. Disk (info, never fails the check on its own unless /workspace is full)
if [ -d /workspace ]; then
  df -h /workspace
else
  df -h .
fi
ls -lh data/ 2>/dev/null || say "data/: absent"
ls -lh models/ models/16k/ 2>/dev/null || true

if [ "$fail" -ne 0 ]; then
  echo "preflight: FAIL - prep would refuse. Fix above, then re-run $0." >&2
  exit 1
fi
echo "preflight: OK - cloud/run.sh prep can run (SIZE=$SIZE)."
