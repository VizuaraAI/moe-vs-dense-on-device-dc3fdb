#!/usr/bin/env bash
# Create the virtual environment and install dependencies.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"

if [ ! -d .venv ]; then
  echo "Creating virtual environment in .venv"
  "$PYTHON" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip

echo
echo "Do you have an NVIDIA GPU? If not, the CPU build of torch is much smaller."
echo "  CPU only : pip install torch --index-url https://download.pytorch.org/whl/cpu"
echo "  CUDA     : pip install torch"
echo

pip install -r requirements.txt

mkdir -p results

echo
echo "Done. Now run:"
echo "  source .venv/bin/activate"
echo "  python -m src.cli device"
echo "  python -m src.cli predict --pair olmoe-vs-qwen15"
