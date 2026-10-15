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

import numpy as np

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


# ---------------------------------------------------------------------------
# External drive for the runtime comparison.
#
# Deliberately NOT shared with scaling_analysis/scaling_diagnostics.py, whose
# inject_poisson_drive the weak and strong campaigns call. The two have opposite
# purposes: the campaigns want a SUB-THRESHOLD drive so the network stays silent and
# ghost-exchange cost is provably activity-independent, while the runtime comparison
# needs the network to actually fire, or it times an idle scan against NEST's empty
# event loop. Sharing one function would mean one of the two callers is always wrong.
# ---------------------------------------------------------------------------

# Cap on the (synapses x ticks) count matrix drawn at once, in entries, so a long run
# cannot allocate an arbitrarily large int64 block.
_DRIVE_CHUNK_ENTRIES = 4_000_000


def drive_lambda(rate_hz, dt_ms, inputs_per_soma=1):
    """Mean external spike count per tick that inject_poisson_drive will deliver."""
    return float(inputs_per_soma) * float(rate_hz) * float(dt_ms) / 1000.0


def inject_poisson_drive(model, input_synapses, rate_hz, ticks, dt_ms, seed, rank,
                         inputs_per_soma=1):
    """Schedule a compound-Poisson external drive on each (pre == -1) input synapse.

    One external synapse stands in for ``inputs_per_soma`` independent Poisson inputs of
    ``rate_hz`` each -- the Brunel convention, where a soma's C_E external afferents are
    collapsed onto a single modelled synapse. The number of external spikes arriving in
    one tick is therefore

        k_t ~ Poisson(inputs_per_soma * rate_hz * dt_ms / 1000)

    and the event VALUE carries k_t rather than k_t separate events: the tick-major event
    list sums duplicate ``(tick, synapse)`` pairs and ``get_soma_spike`` returns that sum
    as the spike magnitude, which the synapse kernel multiplies by ``scale * weight``. So
    ``value = k`` is exactly k coincident spikes, preserving the compound-Poisson mean AND
    variance at one event per synapse per tick.

    Ticks are 0-indexed, matching the kernel's ``thread_local_tick`` (which runs
    ``0..ticks-1``).

    Uses ``add_local_spike_list``. Note that the ``add_local_*`` family is an
    error-checking policy, not a different mechanism: nothing in the injection path
    communicates (``_append_input_events`` only validates and appends to a host list),
    so the plain ``add_spike_list`` would behave identically and cannot deadlock. What
    ``local=True`` buys is an eager ``KeyError`` for an id this rank does not own instead
    of silently dropping it when the event list is compiled.
    """
    syn = np.asarray(input_synapses, dtype=np.int64).ravel()
    if syn.size == 0 or ticks <= 0:
        return
    lam = drive_lambda(rate_hz, dt_ms, inputs_per_soma)
    if lam <= 0.0:
        return

    rng = np.random.default_rng(np.random.SeedSequence([seed, rank, 7]))
    block = max(1, _DRIVE_CHUNK_ENTRIES // int(ticks))
    for start in range(0, syn.size, block):
        chunk = syn[start:start + block]
        counts = rng.poisson(lam, size=(chunk.size, ticks))
        for i, syn_id in enumerate(chunk.tolist()):
            row = counts[i]
            spike_ticks = np.nonzero(row)[0]
            if spike_ticks.size:
                model.add_local_spike_list(
                    syn_id, np.column_stack((spike_ticks, row[spike_ticks])))
