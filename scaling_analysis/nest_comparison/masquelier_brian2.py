"""Masquelier 2008 on Brian2, for a wall-time comparison.

Uses the model equations of the reference implementation verbatim
(`/home/xxz/Hathway-Goodman-2018/code/run_simulation.py:77-135`, Hathway &
Goodman 2018) and the same spike trains as `masquelier_sna.py` and
`masquelier_nestgpu.py`, so all three run the same 10 s experiment.

Unlike the reference this does not tile input, sweep parameters or plot; it is
one 10 s run of the same network, timed by phase.

Run with an interpreter that has Brian2, e.g.
    /home/xxz/miniforge3/envs/diehlcook/bin/python masquelier_brian2.py
"""

import argparse
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MASQ_DIR = Path("/home/xxz/ns-applications/duplicates/masquelier_2008")
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(MASQ_DIR))

import numpy as np

import bench_common as bc


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--dt-s", type=float, default=1e-4)
    p.add_argument("--afferents", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-pattern", type=int, default=1000)
    p.add_argument("--refract-ms", type=float, default=1.0,
                   help="Refractory period; the reference passes refractory=refract*ms "
                        "and SuperNeuroABM's config uses tref=1e-3 s.")
    p.add_argument("--standalone", action="store_true",
                   help="Use the cpp_standalone device, as the reference does.")
    p.add_argument("--csv", default=str(SCRIPT_DIR / "results" / "masquelier.csv"))
    p.add_argument("--dump-dir", default=None,
                   help="Save output spike ticks (at --dt-s) and all weights as .npy.")
    return p


def run(args):
    from brian2 import (NeuronGroup, Synapses, SpikeGeneratorGroup, SpikeMonitor,
                        Network, defaultclock, ms, second, set_device, prefs)
    import experiment_utils as eu

    if args.standalone:
        set_device("cpp_standalone", build_on_run=True,
                   directory=str(SCRIPT_DIR / "brian2_standalone"))
    else:
        prefs.codegen.target = "cython"

    log = bc.PhaseLog("brian2", "masquelier-standalone" if args.standalone
                      else "masquelier", args.afferents, 1,
                      synapses=args.afferents)
    t_pipeline = time.time()

    # ---- spike trains: identical generator, identical seed ----
    n_ticks_gen = int(args.seconds / eu.DT)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    rates = eu.generate_rate_profiles(args.afferents, n_ticks_gen, rng)
    trains = eu.generate_poisson_spikes(rates, rng)
    pattern = eu.generate_pattern(eu.N_PATTERN, eu.PATTERN_DURATION, rng)
    trains, _ = eu.embed_pattern(trains, pattern, n_ticks_gen, rng)
    log.add("generate_spikes", time.time() - t0)

    indices = np.concatenate([np.full(len(t), i, dtype=np.int64)
                              for i, t in enumerate(trains)])
    times_s = np.concatenate([np.asarray(t, dtype=np.float64) * eu.DT
                              for t in trains])
    order = np.argsort(times_s, kind="stable")
    indices, times_s = indices[order], times_s[order]
    n_spikes = len(times_s)

    defaultclock.dt = args.dt_s * second

    # Reference constants (run_simulation.py:77-95)
    T = 0.5 * args.n_pattern
    taum, taus, tausyn = 10 * ms, 2.5 * ms, 2.5 * ms
    X = float((2.5 / 10.0) ** (10.0 / (2.5 - 10.0)))
    deltax, deltaa, K2 = 1, 1, 3
    A = -K2 * T
    tauplus, tauminus = 16.8 * ms, 33.7 * ms
    aplus = 2.0 ** -5
    aminus = 0.85 * aplus
    wmin, wmax = 0, 1
    win = 1.9 * T / args.afferents

    # ---- build ----
    t0 = time.time()
    gen = SpikeGeneratorGroup(args.afferents, indices, times_s * second)
    soma = NeuronGroup(
        1,
        '''du/dt = (A*a)/taus + (X*x-u)/taum : 1
           dx/dt = -x/tausyn : 1
           da/dt = -a/taus : 1''',
        threshold="u > T", reset='''x = 0
                                    u = 2*T
                                    a = deltaa''',
        refractory=args.refract_ms * ms,
        method="linear",
    )
    syn = Synapses(
        gen, soma,
        '''wi : 1
           dLTPtrace/dt = -LTPtrace / tauplus  : 1 (event-driven)
           dLTDtrace/dt = -LTDtrace / tauminus : 1 (event-driven)''',
        # The reference's default "RNN" rule (run_simulation.py:110-119): nearest
        # neighbour AND restricted -- each trace is zeroed once consumed. That is
        # what superneuroabm's exp_pair_wise_stdp_bounded_nn implements despite its
        # name. The un-restricted `_nn` variant depresses on every pre spike after a
        # post spike and silences the neuron after ~20 discharges.
        on_pre='''x_post += deltax*wi
                  LTPtrace = aplus
                  wi = clip(wi + LTDtrace, wmin, wmax)
                  LTDtrace = 0''',
        on_post='''LTDtrace = -aminus
                   wi = clip(wi + LTPtrace, wmin, wmax)
                   LTPtrace = 0''',
    )
    syn.connect()
    syn.wi = win
    mon = SpikeMonitor(soma)
    net = Network(gen, soma, syn, mon)
    log.add("create", time.time() - t0)

    # ---- run ----
    t0 = time.time()
    net.run(args.seconds * second)
    log.add("simulate", time.time() - t0)
    log.add("total", time.time() - t_pipeline)

    bc.print_table(log.rows, f"Brian2 Masquelier  {args.afferents} afferents, "
                             f"{args.seconds:g}s @ dt={args.dt_s*1000:g}ms")
    print(f"\n  input spikes delivered: {n_spikes:,}")
    print(f"  output spikes: {mon.num_spikes}")
    print(f"  first 5 weights: {[round(float(w), 4) for w in syn.wi[:5]]}")
    if args.dump_dir:
        import os
        os.makedirs(args.dump_dir, exist_ok=True)
        ticks = np.rint(np.asarray(mon.t / second) / args.dt_s).astype(np.int64)
        np.save(os.path.join(args.dump_dir, "spike_times.npy"), ticks)
        np.save(os.path.join(args.dump_dir, "weights_all.npy"),
                np.asarray(syn.wi[:], dtype=np.float64))
        print(f"  dumped {len(ticks)} spike times and {len(syn.wi[:])} weights to {args.dump_dir}")
    log.write(args.csv)
    print(f"  wrote {args.csv}")


if __name__ == "__main__":
    run(build_parser().parse_args())
