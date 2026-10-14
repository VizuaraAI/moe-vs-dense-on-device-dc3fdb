#!/usr/bin/env bash
# The first real experiment: one active-matched pair at concurrency 1.
#
# This is the whole paper in miniature. Two models with nearly identical active
# parameter counts (so identical FLOPs per token) measured on your machine. If
# the MoE arm is slower and larger, the thesis has its first data point.
set -euo pipefail

cd "$(dirname "$0")/.."

PAIR="${PAIR:-olmoe-vs-qwen15}"
TIER="${TIER:-laptop}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
REPEATS="${REPEATS:-5}"

echo "== Device fingerprint =="
python -m src.cli device

echo
echo "== Prediction before measurement =="
echo "Write this down. Then try to falsify it."
python -m src.cli predict --pair "$PAIR"

echo
echo "== Measuring $PAIR at concurrency 1 =="
python -m src.cli bench \
  --pair "$PAIR" \
  --concurrency 1 \
  --prompt-tokens 1024 \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --repeats "$REPEATS" \
  --tier "$TIER"

echo
echo "== Report =="
python -m src.cli analyze

echo
echo "Next: paste the table above into experiments/log_template.md,"
echo "then run scripts/run_sweep.sh to find the crossover."
