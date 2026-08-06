"""Shared measurement path for the weak and strong scaling campaigns.

This module exists so both campaigns measure **identically**. The two drivers differ only in
sizing and CLI -- weak holds neurons-per-rank constant, strong holds the total constant -- but
if their measurement code were two copies it could drift, and the campaigns would silently stop
being comparable. That is exactly the failure this module was created to prevent: the weak
driver's own diagnostics once averaged over all ticks while the strong driver's excluded the
first, so their headline numbers were not measuring the same thing.

**No warm-up window is applied here.** Every tick is recorded truthfully, including the
expensive leading ones, and which ticks count as warm-up is decided later by the analysis
scripts. A window baked in at collection time cannot be revisited without re-running -- and
that is how a second warm-up tick went unnoticed for a whole campaign after the first was
already being excluded. Collection reports; analysis decides.

Outputs per run:

* a **per-tick CSV** (``outputs/ticks/...``): one row per tick, across-rank mean/max (and min
  for the step total) of every SAGESim timer. This is the source of truth.
* **topology facts** for the run summary: peer count, ghost volume, bytes -- properties of the
  partition rather than of any particular tick.
"""

import csv
import os
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# What SAGESim records per tick, and how it nests.
#
# The timers are NESTED, so they are grouped by level and never summed flat. On a steady-state
# tick SAGESim's `data_prep` window contains exactly one thing -- `exchange_ghost_data()`
# (SAGESim/sagesim/model.py, the `else:` branch of the first-tick test) -- so `data_prep` IS the
# ghost exchange and is recorded as `comm`. On tick 1 the same window also holds ghost discovery,
# the GPU buffer build and comm_init, which is why tick 1 is enormous.
#
#   level 1 (disjoint, sums to `total` apart from a small unattributed remainder):
#       comm | compute | gpu_sync | write_back
#   level 2 (inside `comm`):
#       pack | exchange | unpack
#   level 3 (inside `exchange`, for the latency-vs-imbalance split; never summed with the above):
#       wait
# ---------------------------------------------------------------------------
TIMER_COLUMNS = (
    ("total", ("total",)),
    ("comm", ("data_prep",)),
    ("compute", ("gpu_compute",)),
    ("gpu_sync", ("gpu_sync",)),
    ("write_back", ("write_back",)),
    ("pack", ("mpi_gpu_pack", "mpi_gpu_sync_pack")),
    ("exchange", ("mpi_exchange",)),
    ("unpack", ("mpi_gpu_unpack",)),
    ("wait", ("mpi_wait_time",)),
    ("peers", ("mpi_num_peers",)),
    ("ghost_somas", ("num_neighbors",)),
    ("send_bytes", ("mpi_send_bytes",)),
)

# Level-1 names, for analysis scripts that need the disjoint decomposition.
LEVEL1 = ("comm", "compute", "gpu_sync", "write_back")
LEVEL2 = ("pack", "exchange", "unpack")
LEVEL3 = ("wait",)


def inject_poisson_drive(model, input_synapses, rate_hz, ticks, dt_ms, seed, rank):
    """Schedule a Poisson spike train on each external (pre == -1) input synapse.

    Each tick is ``dt_ms`` milliseconds; the per-tick spike probability is
    ``rate_hz * dt_ms / 1000``. Vectorized draw per synapse.

    Uses ``add_local_spike_list`` (non-collective, batched): every synapse in ``input_synapses``
    is owned by this rank, so no MPI is needed and each synapse's whole spike train is written in
    one round-trip. The collective ``add_spike``/``add_spike_list`` would DEADLOCK here, because
    each rank injects a different set of local synapses and a different number of spikes, while
    collective calls require every rank to call in lockstep with the same id.
    """
    rng = np.random.default_rng(np.random.SeedSequence([seed, rank, 7]))
    p_spike = rate_hz * dt_ms / 1000.0
    for syn_id in input_synapses:
        fires = rng.random(ticks) < p_spike
        spike_ticks = np.nonzero(fires)[0] + 1  # ticks are 1-indexed
        if spike_ticks.size:
            model.add_local_spike_list(syn_id, [[int(t), 1.0] for t in spike_ticks])


def _local_tick_matrix(model, n_ticks):
    """This rank's ``(n_ticks, n_timers)`` matrix of per-tick timer values."""
    ticks = getattr(model, "_tick_timings", []) or []
    out = np.zeros((n_ticks, len(TIMER_COLUMNS)), dtype=np.float64)
    for i in range(min(n_ticks, len(ticks))):
        entry = ticks[i]
        for j, (_, keys) in enumerate(TIMER_COLUMNS):
            out[i, j] = sum(float(entry.get(k, 0.0)) for k in keys)
    return out


def collect_tick_records(model, comm, rank):
    """Across-rank mean/max/min of every timer, per tick. Returns rows on rank 0, else None.

    Reduced rather than gathered: three ``Reduce`` calls on a ``(n_ticks, n_timers)`` array cost
    the same regardless of rank count, whereas gathering every rank's matrix would move
    ``ranks x ticks x timers`` floats (3 M at 2048 ranks) for data that is then immediately
    summarised. Use ``dump_per_rank_ticks`` if the full per-rank distribution is wanted.

    The tick count is min-reduced first: if a rank somehow recorded fewer ticks, the shared
    prefix is still comparable across ranks rather than silently mixing ragged rows.
    """
    n_local = len(getattr(model, "_tick_timings", []) or [])
    if comm is None:
        n_ticks = n_local
    else:
        from mpi4py import MPI
        n_ticks = comm.allreduce(n_local, op=MPI.MIN)
    if n_ticks == 0:
        return None

    local = _local_tick_matrix(model, n_ticks)
    if comm is None:
        total, mx, mn, nranks = local.copy(), local.copy(), local.copy(), 1
    else:
        nranks = comm.Get_size()
        total = np.zeros_like(local)
        mx = np.zeros_like(local)
        mn = np.zeros_like(local)
        comm.Reduce(local, total, op=MPI.SUM, root=0)
        comm.Reduce(local, mx, op=MPI.MAX, root=0)
        comm.Reduce(local, mn, op=MPI.MIN, root=0)
        if rank != 0:
            return None

    mean = total / nranks
    rows = []
    for i in range(n_ticks):
        row = {"tick": i + 1, "total_min": mn[i, 0]}
        for j, (name, _) in enumerate(TIMER_COLUMNS):
            row[f"{name}_mean"] = mean[i, j]
            row[f"{name}_max"] = mx[i, j]
        rows.append(row)
    return rows


def tick_csv_fieldnames():
    """Column order of the per-tick CSV. Identical for both campaigns, by construction."""
    names = ["tick", "total_min"]
    for name, _ in TIMER_COLUMNS:
        names += [f"{name}_mean", f"{name}_max"]
    return names


def write_tick_csv(path, rows):
    """Write the per-tick records. One row per tick; nothing filtered, nothing folded."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=tick_csv_fieldnames(), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.9g}" if isinstance(v, float) else v)
                             for k, v in row.items()})
    return str(path)


def dump_per_rank_ticks(path, model, comm, rank, n_ticks):
    """Optional ``.npz`` of the full ``(ranks, ticks, timers)`` array.

    Cheap insurance -- ~12 MB compressed at 2048 ranks x 100 ticks -- against wanting the
    per-rank distribution later, when a re-run costs hundreds of node-hours.
    """
    local = _local_tick_matrix(model, n_ticks)
    if comm is None:
        stacked = local[None, ...]
    else:
        gathered = comm.gather(local, root=0)
        if rank != 0:
            return None
        stacked = np.stack(gathered, axis=0)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, ticks=stacked,
                        timers=np.array([n for n, _ in TIMER_COLUMNS], dtype=object))
    return str(path)


def topology_facts(model, comm, rank, neurons_per_worker):
    """Partition properties, not tick properties: peers, ghost volume, message size.

    These are constant for a run (the partition does not change between ticks), so they belong
    in the run summary rather than the per-tick file. Peer and ghost counts are taken as the max
    over ticks because SAGESim records them on the tick that establishes the topology.
    Returns a dict on rank 0, None elsewhere.
    """
    ticks = getattr(model, "_tick_timings", []) or []
    peers = max((t.get("mpi_num_peers", 0) for t in ticks), default=0)
    ghost = max((t.get("num_neighbors", 0) for t in ticks), default=0)
    send = max((t.get("mpi_send_bytes", 0) for t in ticks), default=0)
    local = {"peers": int(peers), "ghost": int(ghost), "send_bytes": float(send)}

    gathered = comm.gather(local, root=0) if comm is not None else [local]
    if rank != 0:
        return None
    gathered = gathered or [local]
    peers_all = [g["peers"] for g in gathered]
    ghost_all = [g["ghost"] for g in gathered]
    send_all = [g["send_bytes"] for g in gathered]
    peers_mean = float(np.mean(peers_all))
    send_mean = float(np.mean(send_all))
    return {
        "peers_min": min(peers_all), "peers_mean": peers_mean, "peers_max": max(peers_all),
        "ghost_somas_mean": float(np.mean(ghost_all)), "ghost_somas_max": max(ghost_all),
        # Surface-to-volume: the explanatory variable for a neighbour-exchange code.
        "ghost_local_ratio": float(np.mean(ghost_all)) / neurons_per_worker,
        "send_bytes_mean": send_mean,
        # Message size decides whether the exchange is latency- or bandwidth-bound.
        "bytes_per_peer": (send_mean / peers_mean) if peers_mean > 0 else 0.0,
    }


def tick_csv_path(outputs_dir, campaign, tag):
    """``outputs/ticks/{campaign}_{tag}_{jobid}.csv`` -- job id keeps re-runs from colliding."""
    job = os.environ.get("SLURM_JOB_ID", "local")
    return Path(outputs_dir) / "ticks" / f"{campaign}_{tag}_{job}.csv"
