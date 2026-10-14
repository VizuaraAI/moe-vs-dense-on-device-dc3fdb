"""Turn results.jsonl into the paper's tables and figures.

The only module that formats output. The markdown table produced by
``main_table`` has the same columns as Table 1 of the paper draft, so a result
can go from the device to the manuscript without being retyped (retyping is how
numbers drift).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .metrics import load_results


def _fmt(value, digits: int = 1) -> str:
    if value is None:
        return "--"
    if isinstance(value, (int, float)):
        if value == 0:
            return "--"
        return f"{value:.{digits}f}"
    return str(value)


def main_table(rows: list[dict], concurrency: int = 1) -> str:
    """Markdown table matching Table 1 of the paper (concurrency one)."""
    selected = [r for r in rows if r.get("concurrency") == concurrency]
    if not selected:
        return f"_No results at concurrency {concurrency} yet._"

    header = (
        "| Pair | Model | Arch | Backend | Quant | TTFT (ms) | TPOT (ms) "
        "| Tokens/s | Peak RAM (GB) | Quality | Status |\n"
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"
    )
    lines = [header]
    for r in sorted(selected, key=lambda r: (r.get("pair", ""), r.get("arch", ""))):
        lines.append(
            "| {pair} | {model} | {arch} | {backend} | {quant} | {ttft} | {tpot} "
            "| {tps} | {mem} | {q} | {status} |".format(
                pair=r.get("pair") or "--",
                model=r.get("model_key", "--"),
                arch=r.get("arch", "--"),
                backend=r.get("backend", "--"),
                quant=r.get("quantization", "--"),
                ttft=_fmt(r.get("ttft_ms_median")),
                tpot=_fmt(r.get("tpot_ms_median"), 2),
                tps=_fmt(r.get("tokens_per_second")),
                mem=_fmt(r.get("peak_rss_gb"), 2),
                q=_fmt(r.get("quality_score"), 3),
                status=r.get("status", "--"),
            )
        )
    return "\n".join(lines)


def crossover_table(rows: list[dict]) -> str:
    """Measured crossover per pair, matching Table 2 of the paper."""
    by_pair: dict[str, dict[int, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        if r.get("status") != "ok" or not r.get("tokens_per_second"):
            continue
        by_pair[r.get("pair", "")][r["concurrency"]][r["arch"]] = r["tokens_per_second"]

    lines = [
        "| Pair | Measured C* | Dense tok/s at C=1 | MoE tok/s at C=1 | Dense advantage at C=1 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for pair, per_c in sorted(by_pair.items()):
        crossover = None
        for concurrency in sorted(per_c):
            arms = per_c[concurrency]
            if "dense" in arms and "moe" in arms and arms["moe"] > arms["dense"]:
                crossover = concurrency
                break
        base = per_c.get(1, {})
        dense, moe = base.get("dense"), base.get("moe")
        advantage = f"{dense / moe:.2f}x" if dense and moe else "--"
        lines.append(
            f"| {pair} | {crossover if crossover else 'none in sweep'} "
            f"| {_fmt(dense)} | {_fmt(moe)} | {advantage} |"
        )
    return "\n".join(lines)


def plot_sweep(rows: list[dict], out_path: str | Path = "results/sweep.png") -> Path | None:
    """Reproduce panel (c) of Figure 2 from measured data."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return None

    series: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for r in rows:
        if r.get("status") == "ok" and r.get("tokens_per_second"):
            series[(r.get("pair", ""), r["arch"])][r["concurrency"]] = r["tokens_per_second"]
    if not series:
        print("no successful runs to plot")
        return None

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    colors = {"dense": "#2F6DB5", "moe": "#D97A20"}
    for (pair, arch), points in sorted(series.items()):
        xs = sorted(points)
        ax.plot(xs, [points[x] for x in xs], "-o", ms=4,
                color=colors.get(arch, "#5C6670"), label=f"{pair} ({arch})")

    ax.set_xscale("log", base=2)
    ax.set_xlabel("concurrency")
    ax.set_ylabel("tokens per second per device")
    ax.set_title("Measured throughput versus concurrency")
    ax.grid(axis="y", alpha=0.4)
    ax.legend(fontsize=8, frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def report(results_path: str | Path = "results/results.jsonl") -> str:
    rows = load_results(results_path)
    if not rows:
        return (
            f"No results found at {results_path}.\n"
            "Run `bash scripts/run_baseline.sh` first."
        )

    ok = sum(1 for r in rows if r.get("status") == "ok")
    failed = len(rows) - ok
    plot = plot_sweep(rows)

    parts = [
        f"# Results report\n",
        f"{len(rows)} runs recorded, {ok} succeeded, {failed} failed or out of memory.\n",
        "## Table 1: comparison at concurrency 1\n",
        main_table(rows, concurrency=1),
        "\n## Table 2: measured crossover\n",
        crossover_table(rows),
    ]
    if failed:
        parts.append(
            "\n## Failed configurations\n\n"
            "These are results, not gaps. A model that cannot run on a device is "
            "the strongest version of the memory argument.\n"
        )
        for r in rows:
            if r.get("status") != "ok":
                parts.append(
                    f"- `{r.get('model_key')}` at C={r.get('concurrency')} "
                    f"({r.get('backend')}, {r.get('quantization')}): "
                    f"**{r.get('status')}** {r.get('error') or ''}"
                )
    if plot:
        parts.append(f"\nSweep plot written to `{plot}`.")
    return "\n".join(parts)
