"""Shared CSV schema and probes for the SuperNeuroABM / NEST GPU comparison.

Every benchmark writes one row per *phase* (long format), so a simulator that
reports five phases and one that reports twenty share a schema without either
padding columns or losing detail.
"""

import csv
import os
import resource
import subprocess
import time
from pathlib import Path

COLUMNS = [
    "simulator",   # sna | nestgpu | brian2
    "variant",     # free-form tag: columnar, record, hpc_benchmark, ...
    "neurons",
    "in_degree",
    "synapses",
    "repeat",
    "phase",       # dotted path, e.g. setup.jit or first_tick.prop_tensors.hyperparameters
    "seconds",
    "host_rss_mb",
    "gpu_mb",
]


def host_rss_mb() -> float:
    """Peak resident set size of this process, in MiB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def gpu_mb() -> float:
    """GPU memory in use, in MiB, read from nvidia-smi (whole device, not just us)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True,
        )
        return float(out.stdout.strip().splitlines()[0])
    except Exception:
        return float("nan")


class PhaseLog:
    """Collects (phase, seconds, rss, gpu) rows and writes them as CSV."""

    def __init__(self, simulator, variant, neurons, in_degree, synapses=0, repeat=0):
        self.meta = dict(
            simulator=simulator, variant=variant, neurons=neurons,
            in_degree=in_degree, synapses=synapses, repeat=repeat,
        )
        self.rows = []

    def set_synapses(self, n):
        self.meta["synapses"] = int(n)

    def add(self, phase, seconds, *, rss=None, gpu=None):
        self.rows.append({
            **self.meta,
            "phase": phase,
            "seconds": float(seconds),
            "host_rss_mb": round(host_rss_mb() if rss is None else rss, 1),
            "gpu_mb": round(gpu_mb() if gpu is None else gpu, 1),
        })

    def add_all(self, prefix, mapping, *, rss=None, gpu=None):
        for key, value in mapping.items():
            if isinstance(value, (int, float)):
                self.add(f"{prefix}.{key}", value, rss=rss, gpu=gpu)

    def timer(self, phase):
        return _Timer(self, phase)

    def write(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        exists = path.exists()
        # Synapse count is usually known only after the network is built, so stamp
        # every row with the final value rather than whatever was set at add() time.
        for row in self.rows:
            row["synapses"] = self.meta["synapses"]
        with open(path, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=COLUMNS)
            if not exists:
                writer.writeheader()
            writer.writerows(self.rows)
        return path


class _Timer:
    def __init__(self, log, phase):
        self.log, self.phase = log, phase

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.log.add(self.phase, time.time() - self.t0)
        return False


def print_table(rows, title):
    """Human-readable summary so a run is legible without opening the CSV."""
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    width = max((len(r["phase"]) for r in rows), default=10)
    for r in rows:
        print(f"  {r['phase']:<{width}}  {r['seconds']:>10.4f}s  "
              f"rss={r['host_rss_mb']:>8.1f}MB  gpu={r['gpu_mb']:>7.1f}MB")
