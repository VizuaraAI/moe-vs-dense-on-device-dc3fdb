"""Device fingerprinting.

Every number this project produces is relative to the memory capacity and the
memory bandwidth of the machine that produced it, so every result record
carries a device fingerprint.

Deliberately dependency-light: torch is imported lazily so that ``cli device``
works on a machine where torch is not installed yet.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass


@dataclass
class DeviceInfo:
    os_name: str
    machine: str
    processor: str
    cpu_count: int
    ram_total_gb: float
    ram_available_gb: float
    disk_free_gb: float
    accelerator: str
    accelerator_memory_gb: float
    measured_mem_bandwidth_gbps: float | None = None
    git_commit: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _ram_gb() -> tuple[float, float]:
    """Total and available RAM in GB. Uses psutil if present, else /proc."""
    try:
        import psutil

        vm = psutil.virtual_memory()
        return vm.total / 1e9, vm.available / 1e9
    except ImportError:
        pass

    try:
        total = available = 0.0
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    total = float(line.split()[1]) * 1024 / 1e9
                elif line.startswith("MemAvailable:"):
                    available = float(line.split()[1]) * 1024 / 1e9
        return total, available
    except OSError:
        return 0.0, 0.0


def _accelerator() -> tuple[str, float]:
    """Accelerator name and its memory in GB. ``cpu`` if there is none."""
    try:
        import torch
    except ImportError:
        return "cpu (torch not installed)", 0.0

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return f"cuda:{props.name}", props.total_memory / 1e9
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        # Apple silicon shares system memory with the GPU.
        total, _ = _ram_gb()
        return "mps:apple-silicon (unified memory)", total
    return f"cpu:{platform.processor() or 'unknown'}", 0.0


def _git_commit() -> str | None:
    if shutil.which("git") is None:
        return None
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def measure_memory_bandwidth(size_mb: int = 256, repeats: int = 5) -> float | None:
    """Crude sequential-read bandwidth probe, in GB/s.

    Streams a large array repeatedly and takes the best time. This is a floor
    on achievable bandwidth, not a peak spec number, which is exactly what the
    roofline model wants: the bandwidth you can actually get. Returns None if
    numpy is unavailable.
    """
    try:
        import numpy as np
    except ImportError:
        return None

    elements = size_mb * 1024 * 1024 // 8
    buffer = np.ones(elements, dtype=np.float64)
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        total = float(buffer.sum())
        elapsed = time.perf_counter() - start
        if total == 0.0:  # keeps the sum from being optimized away
            raise RuntimeError("unexpected zero sum")
        best = min(best, elapsed)
    return (buffer.nbytes / best) / 1e9


def collect(measure_bandwidth: bool = True) -> DeviceInfo:
    total_ram, available_ram = _ram_gb()
    accel, accel_mem = _accelerator()
    disk_free = shutil.disk_usage(os.getcwd()).free / 1e9
    return DeviceInfo(
        os_name=f"{platform.system()} {platform.release()}",
        machine=platform.machine(),
        processor=platform.processor() or "unknown",
        cpu_count=os.cpu_count() or 1,
        ram_total_gb=round(total_ram, 2),
        ram_available_gb=round(available_ram, 2),
        disk_free_gb=round(disk_free, 2),
        accelerator=accel,
        accelerator_memory_gb=round(accel_mem, 2),
        measured_mem_bandwidth_gbps=(
            round(measure_memory_bandwidth() or 0.0, 1) if measure_bandwidth else None
        ),
        git_commit=_git_commit(),
    )


def describe(info: DeviceInfo) -> str:
    lines = [
        "Device fingerprint",
        "==================",
        f"  OS             : {info.os_name} ({info.machine})",
        f"  CPU            : {info.processor} x{info.cpu_count}",
        f"  RAM            : {info.ram_total_gb:.1f} GB total, "
        f"{info.ram_available_gb:.1f} GB available",
        f"  Disk free      : {info.disk_free_gb:.1f} GB",
        f"  Accelerator    : {info.accelerator}",
        f"  Accel memory   : {info.accelerator_memory_gb:.1f} GB",
    ]
    if info.measured_mem_bandwidth_gbps:
        lines.append(
            f"  Read bandwidth : {info.measured_mem_bandwidth_gbps:.1f} GB/s (measured floor)"
        )
    if info.git_commit:
        lines.append(f"  Git commit     : {info.git_commit}")
    lines.append("")
    lines.append(
        "  Note: available RAM is your real model budget. A model whose weights\n"
        "  exceed it pays flash bandwidth, which is where the MoE penalty lives."
    )
    return "\n".join(lines)


if __name__ == "__main__":
    print(json.dumps(collect().to_dict(), indent=2))
