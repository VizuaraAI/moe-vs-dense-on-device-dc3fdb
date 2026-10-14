#!/usr/bin/env bash
# The concurrency sweep: locates the measured crossover C*.
#
# Concurrency above 1 is not a device workload. It is swept for two reasons:
# it locates the crossover, and it verifies the harness reproduces the known
# server-regime result where MoE does win. A reviewer will ask for exactly this.
#
# Expect this to take a while and to hit out-of-memory at high concurrency on a
# small machine. Those failures are recorded as results, not errors.
set -euo pipefail

cd "$(dirname "$0")/.."

PAIR="${PAIR:-olmoe-vs-qwen15}"
TIER="${TIER:-laptop}"
CONCURRENCIES="${CONCURRENCIES:-1,2,4,8,16,32}"
REPEATS="${REPEATS:-3}"

echo "== Sweeping $PAIR over C = $CONCURRENCIES =="
python -m src.cli sweep \
  --pair "$PAIR" \
  --concurrencies "$CONCURRENCIES" \
  --prompt-tokens 1024 \
  --max-new-tokens 128 \
  --repeats "$REPEATS" \
  --tier "$TIER"

echo
echo "== Report =="
python -m src.cli analyze

echo
echo "Compare the measured C* above against the prediction from"
echo "  python -m src.cli predict --pair $PAIR"
echo "The gap between them is the most interesting number in the paper."
