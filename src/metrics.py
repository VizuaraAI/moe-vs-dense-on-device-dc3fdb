"""Result records and the peak-memory sampler.

The schema lives here and nowhere else. If you add a metric, add it to
``RunResult`` and to ``experiments/log_template.md`` in the same commit, so the
lab log never drifts from the data.
"""

from __future__ import annotations

import json
import os
import statistics
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class RunResult:
    """One measured configuration. Written as one JSON line, append only."""

    # Identity
    pair: str
    model_key: str
    arch: str  # dense | moe
    backend: str
    quantization: str

    # Configuration
    concurrency: int
    prompt_tokens: int
    max_new_tokens: int
    repeats: int

    # Latency, milliseconds
    ttft_ms_median: float = 0.0
    ttft_ms_iqr: float = 0.0
    tpot_ms_median: float = 0.0
    tpot_ms_iqr: float = 0.0
    tokens_per_second: float = 0.0

    # Memory, gigabytes
    peak_rss_gb: float = 0.0
    peak_accel_gb: float = 0.0
    model_resident_gb: float = 0.0

    # Data movement and energy, estimated unless the platform allows measuring
    bytes_moved_per_token: float = 0.0
    bytes_moved_source: str = "estimated"
    joules_per_token: float | None = None

    # Quality, filled in by quality.py
    quality_score: float | None = None
    quality_task: str | None = None

    # Provenance
    status: str = "ok"  # ok | oom | error | skipped
    error: str | None = None
    device: dict = field(default_factory=dict)
    timestamp: float = 0.0
    notes: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def summarize(samples: list[float]) -> tuple[float, float]:
    """Median and interquartile range. Never report a single run."""
    if not samples:
        return 0.0, 0.0
    if len(samples) < 4:
        return statistics.median(samples), max(samples) - min(samples)
    quartiles = statistics.quantiles(samples, n=4, method="inclusive")
    return statistics.median(samples), quartiles[2] - quartiles[0]


def append_result(result: RunResult, path: str | os.PathLike = "results/results.jsonl") -> Path:
    """Append one result. Appends only, because this file is the record."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not result.timestamp:
        result.timestamp = time.time()
    with target.open("a", encoding="utf-8") as handle:
        handle.write(result.to_json() + "\n")
    return target


def load_results(path: str | os.PathLike = "results/results.jsonl") -> list[dict]:
    target = Path(path)
    if not target.exists():
        return []
    rows = []
    with target.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class PeakMemorySampler:
    """Samples process RSS on a background thread while a run is in flight.

    Peak memory is the headline number on the memory axis, and it is easy to
    under-report by reading it only after the run finishes, since allocators
    release memory. Sample continuously instead.
    """

    def __init__(self, interval_s: float = 0.05) -> None:
        self.interval_s = interval_s
        self.peak_rss_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        try:
            import psutil

            self._process = psutil.Process()
        except ImportError:
            self._process = None

    def _sample_once(self) -> int:
        if self._process is not None:
            return int(self._process.memory_info().rss)
        try:  # fall back to /proc on Linux
            with open("/proc/self/statm", encoding="utf-8") as handle:
                pages = int(handle.read().split()[1])
            return pages * os.sysconf("SC_PAGE_SIZE")
        except (OSError, IndexError, ValueError):
            return 0

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.peak_rss_bytes = max(self.peak_rss_bytes, self._sample_once())
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "PeakMemorySampler":
        self.peak_rss_bytes = self._sample_once()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.peak_rss_bytes = max(self.peak_rss_bytes, self._sample_once())

    @property
    def peak_rss_gb(self) -> float:
        return self.peak_rss_bytes / 1e9


def peak_accelerator_gb() -> float:
    """Peak accelerator memory since the last reset, in GB. 0.0 if no GPU."""
    try:
        import torch
    except ImportError:
        return 0.0
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e9
    return 0.0


def reset_accelerator_peak() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
