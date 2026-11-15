"""SuperNeuroABM run of a partitioned network: each rank loads only its own partition file.

The counterpart of run_partitioned_nestgpu.py, on the same files and the same drive:
<root>/np<N>/partition_<rank>.npz through model.load_network, the tutorial's alpha neuron
(NEST's iaf_psc_alpha with a 1.5 ms delay, verified against NEST GPU in
tutorials/brunel_simulation.py alpha), and partitioned_common.drive replayed on each soma's
external input synapse.

usage: srun -n N python run_partitioned_sna.py <root> [--ms 1500] [--out spikes.npz]
       ... <root> --config hpc_alpha_config.yaml --eta 1.685 --presim-ms 50 --ms 300
           (NEST GPU's hpc_benchmark: 50 ms untimed warm-up, then the timed 250 ms)
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
    ap.add_argument("--ms", type=float, default=1500.0, help="simulated time (ms), warm-up included")
    ap.add_argument("--presim-ms", type=float, default=0.0,
                    help="warm-up at the start of --ms, timed apart (presimulate)")
    ap.add_argument("--burn-in-ms", type=float, default=None,
                    help="the rate is taken after this (default: --presim-ms, or 500)")
    ap.add_argument("--eta", type=float, default=2.0, help="drive / threshold rate")
    ap.add_argument("--seed", type=int, default=1234, help="drive seed")
    ap.add_argument("--config", default=str(pc.ALPHA_YAML), help="the alpha neuron's YAML")
    ap.add_argument("--vm-sigma", type=float, default=0.0,
                    help="start every soma at normal(--vm-mean, this) mV (hpc_benchmark: 7.2)")
    ap.add_argument("--vm-mean", type=float, default=5.7)
    ap.add_argument("--tick-timers", action="store_true",
                    help="SAGESim's per-tick timers on for the timed simulate, and their means "
                         "printed (launch, GPU, pack, MPI, unpack, reverse); they slow the tick")
    ap.add_argument("--out", default=None, help="spikes .npz (default <root>/spikes_sna_np<N>.npz)")
    args = ap.parse_args()
    if args.burn_in_ms is None:
        args.burn_in_ms = args.presim_ms if args.presim_ms > 0 else 500.0

    comm = MPI.COMM_WORLD
    rank, P = comm.Get_rank(), comm.Get_size()
    timers = {}
    t_all = time.time()
    par = pc.alpha_parameters(args.config)
    extra = par["delay_ticks"]

    # ---- load this rank's part -----------------------------------------------------------
    t0 = time.time()
    path = Path(args.root) / f"np{P}" / f"partition_{rank}.npz"
    model = NeuromorphicModel(user_config=args.config, enable_internal_states_tracking=False,
                              learning_rule_info={})          # static synapses: no STDP kernels
    model.set_global_property_value("dt", pc.DT_MS / 1000.0)
    model.register_soma_type(name=pc.SOMA_BREED, step_func=user_customized_brunel_alpha_soma,
                             step_func_path=pc.STEP_FUNC_DIR / f"{pc.SOMA_BREED}.py")
    model.register_synapse_type(name=pc.SYNAPSE_BREED, step_func=user_customized_brunel_synapse,
                                step_func_path=pc.STEP_FUNC_DIR / f"{pc.SYNAPSE_BREED}.py",
                                event_capable=True)
    model.load_network(str(path))
    progress(rank, "load_network")
    soma_ids, syn_of_soma, n_synapses = pc.read_external_inputs(path)
    timers["load"] = time.time() - t0

    if args.vm_sigma > 0:                           # the same starting potentials as NEST GPU's
        t0 = time.time()
        at = pc.soma_state_index(args.config, "V_m")
        vm = pc.initial_vm(soma_ids, args.seed, args.vm_mean, args.vm_sigma)
        for soma, v in zip(soma_ids.tolist(), vm.tolist()):
            state = list(model.get_agent_property_value(id=soma, property_name="internal_states"))
            state[at] = v * 1e-3                    # mV -> V
            model.set_agent_property_value(soma, "internal_states", state)
        timers["initial_vm"] = time.time() - t0

    t0 = time.time()
    model.setup()
    progress(rank, "setup")
    timers["setup"] = time.time() - t0

    # ---- external drive: each soma's events on its external input synapse, delayed by the
    # whole delay (injected input does not pass through the soma's queue) -------------------
    t0 = time.time()
    steps = int(round(args.ms / pc.DT_MS))
    presim = int(round(args.presim_ms / pc.DT_MS))
    if not 0 <= presim < steps:
        raise ValueError("--presim-ms must be shorter than --ms")
    rate_hz = args.eta * pc.threshold_rate_hz(par)
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
    done = 1
    if presim > 1:                                  # the warm-up, timed apart
        t0 = time.time()
        model.simulate(ticks=presim - 1)
        timers["presimulate"] = time.time() - t0
        progress(rank, "presimulate")
        for soma, ticks in model.get_all_spike_times().items():
            recorded.setdefault(soma, []).extend(ticks)
        done = presim
    if args.tick_timers:
        model.verbose_timing = True
    t0 = time.time()
    model.simulate(ticks=steps - done + extra)
    timers["simulate"] = time.time() - t0
    progress(rank, "simulate")
    per_tick = {}
    if args.tick_timers:                            # ms per tick, mean over the timed ticks
        ticks_timed = list(model._tick_timings)[-(steps - done + extra):]
        for key in sorted({k for t in ticks_timed for k in t}):
            per_tick[key] = float(np.mean([t.get(key, 0.0) for t in ticks_timed]))
    for soma, ticks in model.get_all_spike_times().items():
        recorded.setdefault(soma, []).extend(ticks)
    timers["total"] = time.time() - t_all
    free, total = cp.cuda.Device().mem_info
    memory = (pc.host_peak_gib(), (total - free) / 2**30)

    gathered = comm.gather((timers, soma_ids.size, n_synapses, memory, per_tick), root=0)
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
        print(f"  timed simulate: {(steps - done) * pc.DT_MS:.0f} ms of network time, "
              f"{max(g[0]['simulate'] for g in gathered) / (steps - done + extra) * 1e3:.3f} ms per tick")
        print(f"  peak host memory per rank: max {max(g[3][0] for g in gathered):.2f} GiB;  "
              f"GPU memory in use at the end: max {max(g[3][1] for g in gathered):.2f} GiB")
        if gathered[0][4]:
            print("  per tick, SAGESim's timers (mean over the timed ticks; time keys in ms):")
            for key in gathered[0][4]:
                vals = [g[4].get(key, 0.0) for g in gathered]
                scale = 1.0 if ("bytes" in key or "peers" in key) else 1e3
                print(f"    {key:26s} max {max(vals) * scale:12.3f}   mean {np.mean(vals) * scale:12.3f}")
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
