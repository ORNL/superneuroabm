"""NEST GPU run of the network in tutorials/brunel_simulation.py, single GPU.

The same problem as the tutorial, read from the tutorial's own YAML so the two cannot drift:
4 * order neurons (80 % excitatory), each receiving 0.1 * N_E excitatory and 0.1 * N_I
inhibitory synapses, one Poisson input per neuron at eta times the threshold rate, 0.1 ms
steps, 1.5 s. At the default order 2500: 10,000 neurons, 10 M recurrent synapses.

What is the same: sizes, in-degrees, weights (pA), time constants, threshold, reset,
refractory period, Poisson rate, step, duration, spike times recorded for every neuron.

The neuron is the same in both: the tutorial's soma takes the exact membrane step iaf_psc_exp
takes (since 2026-11-06; before that it was forward Euler and this script had a
`--match-charge` option), so the weights are the YAML's.

What differs, and cannot be removed:
  * refractory period: iaf_psc_exp counts round(t_ref / dt) steps; the tutorial's soma
    compares times, which float rounding makes 19 or 20 steps.
  * input: NEST GPU draws the Poisson input on the device while it simulates; the tutorial
    loads its drive from a file and replays it as an event list.
  * wiring: fixed_indegree draws sources with replacement (multapses possible); the tutorial
    draws without replacement.
  * delay: `--delay-ms` (default one step). The tutorial's spike reaches the post neuron's
    current two ticks after the spike's tick, which is NEST's one-step delay: a spike of step
    t is added to the current after step t + 1's update and first moves V in step t + 2.

By default NEST GPU builds this network the way its own Brunel examples do: the connections
from a rule (fixed_indegree, drawn on the device) and the input from a poisson_generator (drawn
on the device at every step). Neither exists for a network that comes as data. `--as-data`
treats the same network as any other SNN with input, which is what the tutorial does on the
SuperNeuroABM side: the tutorial's own source draws handed over as an explicit edge list, and
the tutorial's saved drive (tutorials/data/brunel_poisson_drive_*.npz, written by running the
tutorial once) replayed through one spike_generator per neuron.

usage (one simulator at a time):
    source ~/software/nest-gpu/install/bin/nestgpu_vars.sh
    python brunel_tutorial_nestgpu.py
    python brunel_tutorial_nestgpu.py --as-data
"""

import argparse
import time
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
CONFIG = HERE.parents[1] / "tutorials" / "customized_configs" / "brunel_config.yaml"
DRIVE_DIR = HERE.parents[1] / "tutorials" / "data"
SOMA_BREED = "user_customized_brunel_soma"
SYNAPSE_BREED = "user_customized_brunel_synapse"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--order", type=int, default=2500)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--eta", type=float, default=2.0)
    p.add_argument("--dt-ms", type=float, default=0.1)
    p.add_argument("--sim-ms", type=float, default=1500.0)
    p.add_argument("--burn-in-ms", type=float, default=500.0)
    p.add_argument("--delay-ms", type=float, default=None, help="Default: one step.")
    p.add_argument("--as-data", action="store_true",
                   help="Explicit edge list (the tutorial's draws) and the tutorial's saved "
                        "drive through spike_generators, instead of a rule and a generator.")
    p.add_argument("--explicit-edges", action="store_true", help="Only that half of --as-data.")
    p.add_argument("--saved-drive", action="store_true", help="Only that half of --as-data.")
    p.add_argument("--no-record", action="store_true",
                   help="Count spikes instead of recording every spike time.")
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


def tutorial_parameters():
    """The tutorial's YAML (SI units) as NEST units: pF, ms, mV, pA."""
    cfg = yaml.safe_load(CONFIG.read_text())
    soma = cfg["soma"][SOMA_BREED]["brunel_lif_neuron"]["hyperparameters"]
    syn = cfg["synapse"][SYNAPSE_BREED]
    return {
        "C_m": float(soma["C_m"]) * 1e12,
        "tau_m": float(soma["tau_m"]) * 1e3,
        "tau_ex": float(soma["tau_syn_0"]) * 1e3,
        "tau_in": float(soma["tau_syn_1"]) * 1e3,
        "V_th": float(soma["V_th"]) * 1e3,
        "V_reset": float(soma["V_reset"]) * 1e3,
        "E_L": float(soma["E_L"]) * 1e3,
        "t_ref": float(soma["t_ref"]) * 1e3,
        "J_E": float(syn["brunel_excitatory_synapse"]["hyperparameters"]["weight"]) * 1e12,
        "J_I": float(syn["brunel_inhibitory_synapse"]["hyperparameters"]["weight"]) * 1e12,
    }


def main():
    args = parse_args()
    explicit_edges = args.as_data or args.explicit_edges
    saved_drive = args.as_data or args.saved_drive
    par = tutorial_parameters()
    dt = args.dt_ms
    delay = dt if args.delay_ms is None else args.delay_ms

    NE, NI = int(0.8 * 4 * args.order), int(0.2 * 4 * args.order)
    N = NE + NI
    CE, CI = int(args.epsilon * NE), int(args.epsilon * NI)

    # The tutorial's threshold_rate_hz: the rate at which the soma's mean potential reaches
    # threshold (one event carries the charge weight * tau_syn). Both simulators get this rate.
    rate_hz = args.eta * ((par["V_th"] - par["E_L"]) * par["C_m"]
                          / (par["J_E"] * par["tau_ex"] * par["tau_m"])) * 1e3
    J_E, J_I = par["J_E"], par["J_I"]

    import nestgpu as ngpu

    T = {}
    t_start = time.time()
    ngpu.SetKernelStatus("time_resolution", dt)
    ngpu.SetKernelStatus("rnd_seed", args.seed)
    ngpu.SetKernelStatus("verbosity_level", 1)

    t0 = time.time()
    neurons = ngpu.Create("iaf_psc_exp", N)
    ngpu.SetStatus(neurons, {
        "tau_m": par["tau_m"], "C_m": par["C_m"], "E_L": par["E_L"], "I_e": 0.0,
        "Theta_rel": par["V_th"] - par["E_L"], "V_reset_rel": par["V_reset"] - par["E_L"],
        "tau_ex": par["tau_ex"], "tau_in": par["tau_in"], "t_ref": par["t_ref"],
    })
    T["create"] = time.time() - t0

    steps = round(args.sim_ms / dt)
    if not explicit_edges:
        t0 = time.time()
        exc, inh = neurons[0:NE], neurons[NE:N]
        ngpu.Connect(exc, neurons, {"rule": "fixed_indegree", "indegree": CE},
                     {"weight": J_E, "delay": delay})
        ngpu.Connect(inh, neurons, {"rule": "fixed_indegree", "indegree": CI},
                     {"weight": J_I, "delay": delay})
        T["connect (rule)"] = time.time() - t0
    else:
        # The tutorial's build_brunel_network draws, as arrays (not timed: the tutorial's
        # own 0.3 s for the same loop is inside its build time, noted in the write-up).
        rng = np.random.default_rng(args.seed)
        sources = np.empty((N, CE + CI), dtype=np.int64)
        for target in range(N):
            sources[target, :CE] = rng.choice(NE, size=CE, replace=False)
            sources[target, CE:] = NE + rng.choice(NI, size=CI, replace=False)
        targets = np.repeat(np.arange(N, dtype=np.int64), CE + CI).reshape(N, CE + CI)
        first = neurons.i0

        t0 = time.time()
        edges = [((first + sources[:, :CE]).ravel().tolist(),
                  (first + targets[:, :CE]).ravel().tolist(), J_E),
                 ((first + sources[:, CE:]).ravel().tolist(),
                  (first + targets[:, CE:]).ravel().tolist(), J_I)]
        T["edge list to Python lists"] = time.time() - t0
        t0 = time.time()
        for src, tgt, weight in edges:
            ngpu.Connect(src, tgt, {"rule": "one_to_one"}, {"weight": weight, "delay": delay})
        T["connect (explicit edges)"] = time.time() - t0
        del edges

    if not saved_drive:
        t0 = time.time()
        poisson = ngpu.Create("poisson_generator")
        ngpu.SetStatus(poisson, "rate", rate_hz)
        ngpu.Connect(poisson, neurons, {"rule": "all_to_all"}, {"weight": J_E, "delay": delay})
        T["drive: poisson_generator"] = time.time() - t0
    else:
        drive_file = DRIVE_DIR / f"brunel_poisson_drive_{N}x{steps}.npz"
        if not drive_file.exists():
            raise SystemExit(f"{drive_file} not found: run tutorials/brunel_simulation.py "
                             f"{args.order} once, it draws and saves the drive.")
        t0 = time.time()
        counts = np.load(drive_file)["counts"]                  # [tick, input], uint8
        T["drive: load file"] = time.time() - t0
        t0 = time.time()
        generators = ngpu.Create("spike_generator", N)
        n_events = 0
        for i in range(N):
            column = counts[:, i]
            tick = np.nonzero(column)[0]
            n_events += tick.size
            # a spike of multiplicity k is k events in that step, as in the tutorial
            ngpu.SetStatus(generators[i:i + 1], {
                "spike_times": ((tick + 1) * dt).tolist(),
                "spike_gen_mul": column[tick].astype(float).tolist()})
        ngpu.Connect(generators, neurons, {"rule": "one_to_one"},
                     {"weight": J_E, "delay": delay})
        T[f"drive: {n_events:,} spikes to generators"] = time.time() - t0
        del counts

    if args.no_record:
        ngpu.ActivateSpikeCount(neurons)
    else:
        ngpu.ActivateRecSpikeTimes(neurons, 1000)

    t0 = time.time()
    ngpu.Calibrate()
    T["calibrate"] = time.time() - t0

    burn_in_counts = None
    if saved_drive:
        # One Simulate() call: a spike_generator loses its place when Simulate() is called
        # again (measured: a second call at 500 ms delivers nothing for its first 500 ms and
        # then resumes the train from where it stopped), so no separate first step here.
        if args.no_record:
            raise SystemExit("--no-record reads the counts at the burn-in, which needs a second "
                             "Simulate() call; not available with the saved drive.")
        timed_steps = steps
        key = f"all {steps} steps"
        t0 = time.time()
        ngpu.Simulate(args.sim_ms)
        T[key] = time.time() - t0
    else:
        t0 = time.time()
        ngpu.Simulate(dt)
        T["first step"] = time.time() - t0

        timed_steps = steps - 1
        key = f"remaining {steps - 1} steps"
        t0 = time.time()
        ngpu.Simulate(args.burn_in_ms - dt)
        if args.no_record:
            t1 = time.time()
            burn_in_counts = np.asarray(ngpu.GetStatus(neurons, "spike_count"), dtype=float).ravel()
            t0 += time.time() - t1                  # the read-out is not simulation time
        ngpu.Simulate(args.sim_ms - args.burn_in_ms)
        T[key] = time.time() - t0

    t0 = time.time()
    measured_s = (args.sim_ms - args.burn_in_ms) / 1000.0
    cv_e = cv_i = float("nan")
    if args.no_record:
        counts = np.asarray(ngpu.GetStatus(neurons, "spike_count"), dtype=float).ravel()
        total = int(counts.sum())
        late = counts - burn_in_counts
        rate_e, rate_i = late[:NE].sum() / NE / measured_s, late[NE:].sum() / NI / measured_s
    else:
        spike_times = [np.asarray(t, dtype=float) for t in ngpu.GetRecSpikeTimes(neurons)]
        total = sum(t.size for t in spike_times)
        late = [t[t >= args.burn_in_ms] for t in spike_times]

        def summary(trains):
            n = sum(t.size for t in trains)
            isi = np.concatenate([np.diff(t) for t in trains if t.size > 1] or [np.zeros(0)])
            cv = float(isi.std() / isi.mean()) if isi.size else 0.0
            return n / len(trains) / measured_s, cv

        (rate_e, cv_e), (rate_i, cv_i) = summary(late[:NE]), summary(late[NE:])
        if max(t.size for t in spike_times) >= 1000:
            print("  WARNING: a neuron filled its 1000-spike record; rates are undercounted.")
    T["read spikes"] = time.time() - t0
    T["total"] = time.time() - t_start

    print(f"\nNEST GPU, tutorial Brunel network: {N} neurons, {N * (CE + CI):,} recurrent "
          f"synapses, {steps} steps of {dt} ms")
    print(f"  J_E {J_E:.2f} pA, J_I {J_I:.2f} pA, delay {delay} ms, "
          f"Poisson {rate_hz:.0f} Hz per neuron")
    print("  connections: " + ("explicit edge list" if explicit_edges else "fixed_indegree rule")
          + "; input: " + ("saved drive through spike_generators" if saved_drive
                           else "poisson_generator"))
    width = max(len(name) for name in T) + 2
    for name, seconds in T.items():
        print(f"  {name:{width}s} {seconds:9.3f} s")
    print(f"  {'per step':{width}s} {T[key] / timed_steps * 1e3:9.4f} ms")
    print(f"  spikes {total:,} in the whole run; after {args.burn_in_ms:.0f} ms: "
          f"E {rate_e:.2f} Hz (ISI CV {cv_e:.3f}), I {rate_i:.2f} Hz (ISI CV {cv_i:.3f})")


if __name__ == "__main__":
    main()
