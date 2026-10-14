# moe-vs-dense-on-device

Measurement harness for the paper **"Mixture of Experts Is Probably Wrong On Device"**.

Thesis: Mixture-of-Experts (MoE) models are sold on FLOPs per token, but
single-user on-device decoding is bound by **bytes moved per token**. At batch
size one a dense model with the same active parameter count should win on
latency, peak memory and energy at the same time. The MoE advantage should only
appear above a crossover concurrency `C*` that a phone or a laptop never
reaches.

This repo gives you everything needed to turn that claim into measurements:
matched model pairs, a per-token timing loop, a memory and I/O sampler, a
concurrency sweep, an analytical roofline model that predicts `C*`, and an
analysis step that emits the exact tables in the paper draft.

## What is in here

```
src/
  registry.py     matched dense/MoE model pairs (active-matched and quality-matched)
  device_info.py  probes RAM, accelerator, and the bandwidth numbers the model needs
  backends.py     HuggingFace and llama.cpp backends behind one interface
  metrics.py      result records plus a background peak-memory sampler
  bench.py        the measurement loop: TTFT, TPOT, peak memory, concurrency sweep
  roofline.py     analytical crossover model, predicts C* with no measurement
  quality.py      small accuracy check so no arm wins by being worse
  analyze.py      turns results.jsonl into the paper's tables and figures
  cli.py          one entry point for everything
configs/
  default.yaml    the default experiment grid
scripts/
  setup.sh        create the venv and install
  run_baseline.sh the first real experiment: one matched pair at C=1
  run_sweep.sh    the full concurrency sweep
experiments/
  log_template.md one row per run, columns map to Table 1 of the paper
```

No model weights or datasets are committed. Everything downloads from the
Hugging Face Hub on first use.

## Setup

```bash
git clone <your-fork-url> moe-vs-dense-on-device
cd moe-vs-dense-on-device
bash scripts/setup.sh
source .venv/bin/activate
```

If you are on a phone or a machine without a CUDA GPU, install the CPU wheel of
PyTorch instead (the setup script asks). The llama.cpp backend is optional and
only needed once you move to quantized GGUF weights.

## First three things to run

**1. See what your device actually is.** Every number in this project is
relative to your memory bandwidth and capacity, so record them first.

```bash
python -m src.cli device
```

**2. Predict the answer before you measure it.** The roofline model needs no
model downloads and takes a second. Write the predicted `C*` in your lab log,
then try to falsify it.

```bash
python -m src.cli predict --pair olmoe-vs-qwen15
```

**3. Measure one matched pair at concurrency one.** This is the whole paper in
miniature. Start with the smallest pair so a full run takes minutes, not hours.

```bash
bash scripts/run_baseline.sh
python -m src.cli analyze --results results/results.jsonl
```

That prints a markdown table with the same columns as Table 1 in the paper
draft. Paste it into `experiments/log_template.md` and you have your first row
of real results.

## Then

```bash
bash scripts/run_sweep.sh                       # the concurrency sweep, finds measured C*
python -m src.cli quality --pair olmoe-vs-qwen15  # attach a quality number to every cost number
```

## House rules for this project

- Never report a latency number without the memory number next to it. The whole
  argument is that they trade off.
- A configuration that runs out of memory is a **result**. Record it as OOM, do
  not drop the row.
- Report bytes moved per token, not just wall clock. Wall clock can be blamed on
  immature MoE kernels; bytes moved cannot.
- Always run the best available MoE baseline, not a naive one. If a reviewer can
  say "you used a bad MoE implementation", the paper is dead.
