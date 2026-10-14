"""One entry point for the whole project.

    python -m src.cli device                      what machine am I on
    python -m src.cli predict --pair <key>        predict before measuring
    python -m src.cli bench   --pair <key>        measure at one concurrency
    python -m src.cli sweep   --pair <key>        the concurrency sweep
    python -m src.cli quality --pair <key>        accuracy guard rail
    python -m src.cli analyze                     tables and figures

``device`` and ``predict`` must keep working with no model downloads and
without torch installed. They are what makes this project debuggable anywhere.
"""

from __future__ import annotations

import argparse
import json
import sys

from .registry import PAIRS, get_pair
from .roofline import TIERS, format_prediction, predict_pair


def _add_pair_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--pair",
        default="olmoe-vs-qwen15",
        choices=sorted(PAIRS),
        help="matched model pair to work on",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="src.cli", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("device", help="print the device fingerprint")
    sub.add_parser("pairs", help="list the matched model pairs")

    p_predict = sub.add_parser("predict", help="analytical crossover prediction")
    _add_pair_arg(p_predict)
    p_predict.add_argument("--json", action="store_true", help="emit JSON")
    p_predict.add_argument(
        "--expert-reuse", type=float, default=0.0,
        help="measured expert cache hit rate in [0,1)",
    )
    p_predict.add_argument(
        "--moe-efficiency", type=float, default=None,
        help="fraction of peak bandwidth a scattered expert gather achieves "
             "(the key assumption; measure it and pass it here)",
    )

    p_bench = sub.add_parser("bench", help="measure both arms at one concurrency")
    _add_pair_arg(p_bench)
    p_bench.add_argument("--backend", default="hf", choices=["hf", "llamacpp"])
    p_bench.add_argument("--quantization", default="none")
    p_bench.add_argument("--concurrency", type=int, default=1)
    p_bench.add_argument("--prompt-tokens", type=int, default=1024)
    p_bench.add_argument("--max-new-tokens", type=int, default=128)
    p_bench.add_argument("--repeats", type=int, default=5)
    p_bench.add_argument("--tier", default="laptop", choices=sorted(TIERS))
    p_bench.add_argument("--results", default="results/results.jsonl")

    p_sweep = sub.add_parser("sweep", help="concurrency sweep over both arms")
    _add_pair_arg(p_sweep)
    p_sweep.add_argument("--backend", default="hf", choices=["hf", "llamacpp"])
    p_sweep.add_argument("--quantization", default="none")
    p_sweep.add_argument("--concurrencies", default="1,2,4,8,16,32")
    p_sweep.add_argument("--prompt-tokens", type=int, default=1024)
    p_sweep.add_argument("--max-new-tokens", type=int, default=128)
    p_sweep.add_argument("--repeats", type=int, default=3)
    p_sweep.add_argument("--tier", default="laptop", choices=sorted(TIERS))
    p_sweep.add_argument("--results", default="results/results.jsonl")

    p_quality = sub.add_parser("quality", help="accuracy guard rail for both arms")
    _add_pair_arg(p_quality)
    p_quality.add_argument("--n-examples", type=int, default=200)
    p_quality.add_argument("--quantization", default="none")

    p_analyze = sub.add_parser("analyze", help="tables and figures from results")
    p_analyze.add_argument("--results", default="results/results.jsonl")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "device":
        from .device_info import collect, describe

        print(describe(collect()))
        return 0

    if args.command == "pairs":
        for key, pair in sorted(PAIRS.items()):
            print(f"{key}  (matched on {pair.axis})")
            print(f"    dense : {pair.dense.hf_id}")
            print(f"    moe   : {pair.moe.hf_id}")
            print(f"    why   : {pair.rationale}\n")
        return 0

    if args.command == "predict":
        extra = {}
        if args.moe_efficiency is not None:
            extra["moe_bandwidth_efficiency"] = args.moe_efficiency
        prediction = predict_pair(
            get_pair(args.pair), expert_reuse=args.expert_reuse, **extra
        )
        print(json.dumps(prediction, indent=2) if args.json else format_prediction(prediction))
        return 0

    # Everything below needs the heavy dependencies.
    if args.command in ("bench", "sweep"):
        from .bench import measure_pair, sweep_pair
        from .metrics import append_result

        shared = dict(
            backend_name=args.backend,
            quantization=args.quantization,
            prompt_tokens=args.prompt_tokens,
            max_new_tokens=args.max_new_tokens,
            repeats=args.repeats,
            tier=args.tier,
        )
        if args.command == "bench":
            results = measure_pair(args.pair, concurrency=args.concurrency, **shared)
        else:
            concurrencies = tuple(int(c) for c in args.concurrencies.split(","))
            results = sweep_pair(args.pair, concurrencies=concurrencies, **shared)

        for result in results:
            append_result(result, args.results)
            status = "" if result.status == "ok" else f"  [{result.status}: {result.error}]"
            print(
                f"{result.model_key:<16} C={result.concurrency:<3} "
                f"TPOT={result.tpot_ms_median:8.2f} ms  "
                f"TTFT={result.ttft_ms_median:8.1f} ms  "
                f"peak={result.peak_rss_gb:6.2f} GB{status}"
            )
        print(f"\nAppended {len(results)} rows to {args.results}")
        return 0

    if args.command == "quality":
        from .quality import evaluate_hellaswag

        pair = get_pair(args.pair)
        for spec in pair.arms():
            outcome = evaluate_hellaswag(
                spec, n_examples=args.n_examples, quantization=args.quantization
            )
            print(
                f"{outcome.model_key:<16} {outcome.task}: "
                f"{outcome.accuracy:.3f} over {outcome.n_examples} examples"
            )
        return 0

    if args.command == "analyze":
        from .analyze import report

        print(report(args.results))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
