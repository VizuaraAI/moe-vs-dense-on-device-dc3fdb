# CLAUDE.md

Instructions for the coding agent working on this repository with Suyash.

## Project goal

Produce the measurements for a paper titled **"Mixture of Experts Is Probably
Wrong On Device"**. The claim under test:

> At concurrency one, which is the only concurrency a single-user device ever
> sees, a dense model beats an active-parameter-matched Mixture-of-Experts model
> on latency, on peak memory and on energy per token at the same time. The MoE
> advantage appears only above a crossover concurrency `C*` that is one to two
> orders of magnitude above anything a phone or laptop experiences.

This is an **empirical measurement paper**, not a modelling paper. We train
nothing. The deliverable is credible numbers plus an analytical model that
predicts them. Target venue is an efficiency or on-device workshop at a major
ML conference, with roughly eight weeks of work.

## Repo layout

| Path | Role |
| --- | --- |
| `src/registry.py` | The matched model pairs. Single source of truth for which models are compared and why. |
| `src/device_info.py` | Probes the host: RAM, accelerator, measured bandwidth. Feeds `roofline.py`. |
| `src/backends.py` | Backend adapters (`hf`, `llamacpp`) behind one `Backend` interface. Add new runtimes here only. |
| `src/metrics.py` | `RunResult` record and the peak-memory sampler. Change the schema here and nowhere else. |
| `src/bench.py` | The measurement loop. Timing-critical: read the comments before editing. |
| `src/roofline.py` | Pure-math crossover model. No I/O, no torch. Must stay testable without downloads. |
| `src/quality.py` | Small accuracy harness so every cost number has a quality number attached. |
| `src/analyze.py` | `results.jsonl` to paper tables and figures. The only place that formats output. |
| `src/cli.py` | Subcommands: `device`, `predict`, `bench`, `sweep`, `quality`, `analyze`. |
| `configs/default.yaml` | The experiment grid. Prefer editing config over editing code. |
| `experiments/log_template.md` | The lab log. Its columns match Table 1 of the paper. |

## Conventions

- Python 3.10 or newer, type hints on public functions, standard library first.
- Every measurement writes one JSON line to `results/results.jsonl`. Never
  overwrite that file, always append. It is the experimental record.
- Every `RunResult` carries the full configuration that produced it (model,
  backend, quantization, concurrency, prompt length, device fingerprint, git
  commit). A result that cannot be traced to a configuration is worthless.
- Timing: discard warmup iterations, report median and interquartile range over
  at least five repeats, never a single run. Helpers for this are in `bench.py`,
  use them.
- Keep `roofline.py` free of torch and of any import that could fail on a phone.
- Failures are data. An out-of-memory or a load failure must be recorded as a
  `RunResult` with `status="oom"` or `status="error"`, not swallowed.
- No new dependency without a line in `requirements.txt` and a sentence in the
  README saying why.

## How to help

When Suyash asks for a change, prefer this order:

1. Can it be a config change in `configs/default.yaml`?
2. Can it be a new entry in `src/registry.py`?
3. Only then write code, and keep it in the module that owns that concern.

Before saying a change works, actually run it:

```bash
python -m src.cli device
python -m src.cli predict --pair olmoe-vs-qwen15
python -m src.cli bench --pair olmoe-vs-qwen15 --concurrency 1 --max-new-tokens 32 --repeats 2
python -m src.cli analyze --results results/results.jsonl
```

The `predict` and `device` paths must work on any machine with no model
downloads. Keep them that way, they are what makes the project debuggable on a
laptop at midnight.

## Next steps, in order

1. **Make the baseline real.** Run one active-matched pair at `C=1` on the
   machine Suyash actually has. Get a number for TTFT, TPOT and peak memory.
   Everything else is built on this working.
2. **Add the quantized path.** Wire up the llama.cpp backend with GGUF Q4\_K\_M
   weights for both arms. This is where the memory story becomes visible,
   because it decides which models fit in device RAM.
3. **Instrument bytes moved per token.** Currently estimated analytically in
   `roofline.py`. Make it measured where the platform allows it. This is the
   single most reviewer-proof number in the paper.
4. **Run the concurrency sweep** and locate the measured `C*`. Compare against
   the prediction. The paper's sharpest claim is that the prediction holds.
5. **Add the strong MoE baselines.** A naive MoE baseline invites the reviewer
   response "your implementation was bad". Add an expert-offloading backend so
   the sparse arm is represented by real engineering.
6. **Energy and thermals.** Joules per token and a sustained ten-minute run.
   Platform specific, so keep it behind a capability check.

## Things to push back on

If Suyash proposes any of these, say so plainly:

- Comparing a dense model against an MoE model at different quantization levels
  or different quality operating points. The comparison dies without controls.
- Reporting only the best run. Report distributions.
- Dropping the high-concurrency arm of the sweep. It is what makes the paper a
  characterization rather than a complaint, and it is the part that shows we
  reproduce the known server-regime result.
- Adding a new model family before the first pair produces a clean, repeatable
  number.
