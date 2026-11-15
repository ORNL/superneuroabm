"""SuperNeuroABM run of a partitioned network: each rank loads only its own partition file.

The counterpart of run_partitioned_nestgpu.py, on the same files and the same drive:
<root>/np<N>/partition_<rank>.npz through model.load_network, the tutorial's alpha neuron
(NEST's iaf_psc_alpha with a 1.5 ms delay, verified against NEST GPU in
tutorials/brunel_simulation.py alpha), and partitioned_common.drive replayed on each soma's
external input synapse.

usage: srun -n N python run_partitioned_sna.py <root> [--ms 1500] [--out spikes.npz]
"""

import argparse
import resource
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np
from mpi4py import MPI
from sagesim.agent_index import rows_of

import partitioned_common as pc

sys.path.insert(0, str(pc.STEP_FUNC_DIR))
from superneuroabm.model import NeuromorphicModel                       # noqa: E402
from user_customized_brunel_alpha_soma import user_customized_brunel_alpha_soma  # noqa: E402
from user_customized_brunel_synapse import user_customized_brunel_synapse        # noqa: E402


def progress(rank, what):
    """Rank 0's progress line with its current and peak host memory."""
    if rank == 0:
        with open("/proc/self/statm") as f:
            rss = int(f.read().split()[1]) * 4096 / 2**30
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
        print(f"  [{time.strftime('%H:%M:%S')}] {what:34s} RSS {rss:6.2f} GiB  peak {peak:6.2f} GiB",
              flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root")
    ap.add_argument("--ms", type=float, default=1500.0, help="simulated time (ms)")
    ap.add_argument("--burn-in-ms", type=float, default=500.0)
    ap.add_argument("--eta", type=float, default=2.0, help="drive / threshold rate")
    ap.add_argument("--seed", type=int, default=1234, help="drive seed")
    ap.add_argument("--out", default=None, help="spikes .npz (default <root>/spikes_sna_np<N>.npz)")
    args = ap.parse_args()

    comm = MPI.COMM_WORLD
    rank, P = comm.Get_rank(), comm.Get_size()
    timers = {}
    t_all = time.time()
    par = pc.alpha_parameters()
    extra = par["delay_ticks"]

    # ---- load this rank's part -----------------------------------------------------------
    t0 = time.time()
    path = Path(args.root) / f"np{P}" / f"partition_{rank}.npz"
    model = NeuromorphicModel(user_config=pc.ALPHA_YAML, enable_internal_states_tracking=False,
                              learning_rule_info={})          # static synapses: no STDP kernels
    model.set_global_property_value("dt", pc.DT_MS / 1000.0)
    model.register_soma_type(name=pc.SOMA_BREED, step_func=user_customized_brunel_alpha_soma,
                             step_func_path=pc.STEP_FUNC_DIR / f"{pc.SOMA_BREED}.py")
    model.register_synapse_type(name=pc.SYNAPSE_BREED, step_func=user_customized_brunel_synapse,
                                step_func_path=pc.STEP_FUNC_DIR / f"{pc.SYNAPSE_BREED}.py",
                                event_capable=True)
    model.load_network(str(path))
    progress(rank, "load_network")
    part = pc.read_partition(path)
    timers["load"] = time.time() - t0

    t0 = time.time()
    model.setup()
    progress(rank, "setup")
    timers["setup"] = time.time() - t0

    # ---- external drive: each soma's events on its external input synapse, delayed by the
    # whole delay (injected input does not pass through the soma's queue) -------------------
    t0 = time.time()
    steps = int(round(args.ms / pc.DT_MS))
    rate_hz = args.eta * pc.threshold_rate_hz(par)
    soma_ids = np.sort(np.asarray(part["soma_ids"], dtype=np.int64))
    pre = np.asarray(part["pre"], dtype=np.int64)
    ext_syn = np.asarray(part["synapse_ids"], dtype=np.int64)[pre < 0]
    ext_post = np.asarray(part["post"], dtype=np.int64)[pre < 0]
    syn_of_soma = np.empty(soma_ids.size, dtype=np.int64)
    k = np.searchsorted(soma_ids, ext_post)
    if np.unique(k).size != soma_ids.size or ext_syn.size != soma_ids.size:
        raise ValueError(f"{path}: expected exactly one external input per soma")
    syn_of_soma[k] = ext_syn
    idx, tick, count = pc.drive(soma_ids, steps, rate_hz, args.seed)
    at = tick + par["delay_steps"]
    keep = at < steps + extra
    idx, at, count = idx[keep], at[keep], count[keep]
    if at.size and at.min() < 1:
        raise ValueError("the drive must start after tick 0: tick 0 runs before it is set")
    timers["drive_draw"] = time.time() - t0
    progress(rank, "drive drawn")

    # ---- run: one tick first (it builds the device buffers; the delayed drive has nothing
    # for it), then hand the drive to the device, then the rest plus the delay queue's flush
    # ticks. The drive goes in with set_input_events, compiled on the GPU from the synapses'
    # buffer rows: through add_spikes, the host compile of its ~10^8 events alone exceeds a
    # login node's 30 GiB at 12,500 neurons ------------------------------------------------
    t0 = time.time()
    model.simulate(ticks=1)
    progress(rank, "first tick")
    timers["first_tick"] = time.time() - t0
    recorded = model.get_all_spike_times()          # collective: every rank's spikes
    t0 = time.time()
    rows = np.asarray(rows_of(model._gpu_buffers.agent_id_to_index, syn_of_soma), dtype=np.int64)
    if (rows < 0).any():
        raise RuntimeError("an external input synapse has no device row")
    model.set_input_events(cp.asarray(rows)[cp.asarray(idx)], cp.asarray(at),
                           cp.asarray(count, dtype=cp.float64))
    del idx, at, count
    timers["drive_to_device"] = time.time() - t0
    progress(rank, "set_input_events")
    t0 = time.time()
    model.simulate(ticks=steps - 1 + extra)
    progress(rank, "simulate")
    timers["simulate"] = time.time() - t0
    for soma, ticks in model.get_all_spike_times().items():
        recorded.setdefault(soma, []).extend(ticks)
    timers["total"] = time.time() - t_all

    gathered = comm.gather((timers, soma_ids.size, int(pre.size)), root=0)
    if rank == 0:
        # emission ticks -> the ticks the somas fired at
        soma = np.concatenate([np.full(len(v), s, np.int64) for s, v in recorded.items()]
                              or [np.zeros(0, np.int64)])
        tick = np.concatenate([np.asarray(v, np.int64) for v in recorded.values()]
                              or [np.zeros(0, np.int64)]) - extra
        ok = (tick >= 0) & (tick < steps)
        soma, tick = soma[ok], tick[ok]
        n_somas = sum(g[1] for g in gathered)
        n_syn = sum(g[2] for g in gathered)
        print(f"\nSuperNeuroABM, {P} rank(s): {n_somas:,} somas, {n_syn:,} synapses, "
              f"{soma.size:,} spikes")
        for key in timers:
            vals = [g[0][key] for g in gathered]
            print(f"  {key:18s} max {max(vals):8.3f} s   min {min(vals):8.3f} s")
        burn = int(round(args.burn_in_ms / pc.DT_MS))
        late = tick >= burn
        print(f"  rate after {args.burn_in_ms:.0f} ms: "
              f"{late.sum() / n_somas / ((steps - burn) * pc.DT_MS / 1000):.3f} Hz")
        out = args.out or str(Path(args.root) / f"spikes_sna_np{P}.npz")
        pc.save_spikes(out, soma, tick, {"n_somas": n_somas, "steps": steps, "ranks": P,
                                         "simulator": "superneuroabm"})
        print(f"  spikes saved to {out}")


if __name__ == "__main__":
    main()
