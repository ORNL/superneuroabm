"""
brunel_simulation_nestgpu.py -- the network of brunel_simulation.py, run in NEST GPU without
NEST GPU's shortcuts.

It is the same problem, given to NEST GPU the way brunel_simulation.py gives it to
SuperNeuroABM: as data. The neurons' parameters come from the same YAML, the connections are
the same draws handed over as an explicit list of (source, target) pairs, and the external input
is the same saved Poisson drive (data/brunel_poisson_drive_*.npz), replayed spike by spike.

    python brunel_simulation.py              # once: draws and saves the drive
    source ~/software/nest-gpu/install/bin/nestgpu_vars.sh
    python brunel_simulation_nestgpu.py      # order 2500: 10,000 neurons, about 10 M synapses
    python brunel_simulation_nestgpu.py 250  # order 250 (needs brunel_simulation.py 250 first)

Run one simulator at a time: they share the GPU.

Where NEST GPU would normally take a shortcut, and this script does not (marked SHORTCUT below):

  1. Connections. A Brunel script writes
         Connect(excitatory, neurons, {"rule": "fixed_indegree", "indegree": CE}, ...)
     and NEST GPU draws the sources on the GPU; no list of pairs ever exists on the host. Here
     the pairs are drawn in numpy (the same draws as brunel_simulation.py) and passed with the
     "one_to_one" rule, which is how NEST GPU takes any network that comes as data.
  2. Input. A Brunel script creates one poisson_generator and NEST GPU draws every neuron's
     input on the GPU at every step. Here the saved drive is loaded and given to one
     spike_generator per neuron, which is how NEST GPU takes recorded input.

The neuron is the same in both: the tutorial's soma takes the exact membrane step that NEST GPU's
iaf_psc_exp takes, so the weights are the YAML's, unchanged. What still differs:

  * Refractory period. iaf_psc_exp counts round(t_ref / dt) = 20 refractory steps; the tutorial's
    soma compares times, which float rounding makes 19 or 20 (tutorial 07, section 6). The spike
    trains therefore agree in rate, not spike for spike.
  * A spike_generator loses its place when Simulate() is called a second time, so the whole run
    is one Simulate() call; NEST GPU's one-time preparation is its Calibrate() call.
"""

import sys
import time
from pathlib import Path

import numpy as np
import yaml

TUTORIALS_DIR = Path(__file__).resolve().parent
CONFIG = TUTORIALS_DIR / "customized_configs" / "brunel_config.yaml"
DATA_DIR = TUTORIALS_DIR / "data"
SOMA_BREED = "user_customized_brunel_soma"
SYNAPSE_BREED = "user_customized_brunel_synapse"
DT_MS = 0.1                      # step: 0.1 ms, as in brunel_simulation.py


def tutorial_parameters():
    """The tutorial's YAML (SI units) in NEST units: pF, ms, mV, pA."""
    config = yaml.safe_load(CONFIG.read_text())
    soma = config["soma"][SOMA_BREED]["brunel_lif_neuron"]["hyperparameters"]
    synapse = config["synapse"][SYNAPSE_BREED]
    return {
        "C_m": float(soma["C_m"]) * 1e12,
        "tau_m": float(soma["tau_m"]) * 1e3,
        "tau_ex": float(soma["tau_syn_0"]) * 1e3,
        "tau_in": float(soma["tau_syn_1"]) * 1e3,
        "V_th": float(soma["V_th"]) * 1e3,
        "V_reset": float(soma["V_reset"]) * 1e3,
        "E_L": float(soma["E_L"]) * 1e3,
        "t_ref": float(soma["t_ref"]) * 1e3,
        "J_E": float(synapse["brunel_excitatory_synapse"]["hyperparameters"]["weight"]) * 1e12,
        "J_I": float(synapse["brunel_inhibitory_synapse"]["hyperparameters"]["weight"]) * 1e12,
    }


def run_simulation(order=2500, duration_ms=1500.0, burn_in_ms=500.0, eta=2.0,
                   epsilon=0.1, seed=1234):
    import nestgpu as ngpu

    t_start = time.time()
    par = tutorial_parameters()
    steps = round(duration_ms / DT_MS)
    delay = DT_MS                # one step: what the tutorial's two-tick arrival amounts to

    NE = int(0.8 * (4 * order))
    NI = int(0.2 * (4 * order))
    N = NE + NI
    CE = int(epsilon * NE)
    CI = int(epsilon * NI)

    J_E, J_I = par["J_E"], par["J_I"]

    ngpu.SetKernelStatus("time_resolution", DT_MS)
    ngpu.SetKernelStatus("rnd_seed", seed)
    ngpu.SetKernelStatus("verbosity_level", 1)

    # ---- Neurons -----------------------------------------------------------------------
    print(f"Building network: {NE} E + {NI} I = {N} neurons")
    print(f"  Excitatory in-degree: {CE}")
    print(f"  Inhibitory in-degree: {CI}")
    t_build = time.time()
    neurons = ngpu.Create("iaf_psc_exp", N)
    ngpu.SetStatus(neurons, {
        "tau_m": par["tau_m"], "C_m": par["C_m"], "E_L": par["E_L"], "I_e": 0.0,
        "Theta_rel": par["V_th"] - par["E_L"], "V_reset_rel": par["V_reset"] - par["E_L"],
        "tau_ex": par["tau_ex"], "tau_in": par["tau_in"], "t_ref": par["t_ref"],
    })

    # ---- Connections: an explicit list of (source, target) pairs -------------------------
    # The same draws as build_brunel_network() in brunel_simulation.py: every neuron draws
    # CE excitatory and CI inhibitory sources without replacement.
    #
    # SHORTCUT NOT TAKEN: Connect(neurons[0:NE], neurons, {"rule": "fixed_indegree",
    # "indegree": CE}, ...) would draw these on the GPU and never hold a list on the host.
    rng = np.random.default_rng(seed)
    sources = np.empty((N, CE + CI), dtype=np.int64)
    for target in range(N):
        sources[target, :CE] = rng.choice(NE, size=CE, replace=False)
        sources[target, CE:] = NE + rng.choice(NI, size=CI, replace=False)
    targets = np.repeat(np.arange(N, dtype=np.int64), CE + CI).reshape(N, CE + CI)

    first = neurons.i0           # NEST GPU's index of neuron 0
    for columns, weight in ((slice(0, CE), J_E), (slice(CE, CE + CI), J_I)):
        # Connect takes Python lists of node indices, one pair per connection
        source_list = (first + sources[:, columns]).ravel().tolist()
        target_list = (first + targets[:, columns]).ravel().tolist()
        ngpu.Connect(source_list, target_list, {"rule": "one_to_one"},
                     {"weight": weight, "delay": delay})
    print(f"  Created {N * (CE + CI)} recurrent synapses")
    build_sec = time.time() - t_build
    print(f"  built in {build_sec:.1f} s")

    # ---- External input: the saved Poisson drive, replayed ------------------------------
    # SHORTCUT NOT TAKEN: one poisson_generator connected to every neuron would draw the
    # input on the GPU at every step; nothing would be loaded or stored.
    drive_file = DATA_DIR / f"brunel_poisson_drive_{N}x{steps}.npz"
    if not drive_file.exists():
        raise SystemExit(f"{drive_file} not found: run brunel_simulation.py {order} once, "
                         "it draws and saves the drive.")
    t_drive = time.time()
    with np.load(drive_file) as saved:
        counts = saved["counts"]                     # counts[tick, input], uint8
        rate_hz = float(saved["rate_hz"])
    load_sec = time.time() - t_drive
    print(f"External drive: {rate_hz:.0f} Hz per neuron, {rate_hz * DT_MS / 1000:.2f} events per tick")
    print(f"  loaded from {drive_file} ({np.count_nonzero(counts):,} spikes) in {load_sec:.1f} s")

    t_generators = time.time()
    generators = ngpu.Create("spike_generator", N)
    for i in range(N):
        column = counts[:, i]
        tick = np.nonzero(column)[0]
        # the events of one tick are one spike with that multiplicity, as in the tutorial
        ngpu.SetStatus(generators[i:i + 1], {
            "spike_times": ((tick + 1) * DT_MS).tolist(),
            "spike_gen_mul": column[tick].astype(float).tolist()})
    ngpu.Connect(generators, neurons, {"rule": "one_to_one"}, {"weight": J_E, "delay": delay})
    generators_sec = time.time() - t_generators
    print(f"  handed to {N} spike generators in {generators_sec:.1f} s")
    del counts

    # ---- Run -----------------------------------------------------------------------------
    ngpu.ActivateRecSpikeTimes(neurons, 1000)        # up to 1000 spike times per neuron
    t_calibrate = time.time()
    ngpu.Calibrate()
    calibrate_sec = time.time() - t_calibrate
    print(f"Calibrate() in {calibrate_sec:.1f} s")

    print(f"Simulating {duration_ms:.0f} ms ({steps} steps)...")
    t_sim = time.time()
    ngpu.Simulate(duration_ms)
    steps_sec = time.time() - t_sim
    print(f"  all {steps} steps: {steps_sec:.2f} s ({steps_sec / steps * 1e3:.3f} ms per step)")

    # ---- Rates after the burn-in ---------------------------------------------------------
    spike_times = [np.asarray(times, dtype=float) for times in ngpu.GetRecSpikeTimes(neurons)]
    if max(times.size for times in spike_times) >= 1000:
        print("  WARNING: a neuron filled its 1000-spike record; rates are undercounted.")
    measured_s = (duration_ms - burn_in_ms) / 1000.0

    def firing_summary(trains):
        late = [times[times >= burn_in_ms] for times in trains]
        intervals = np.concatenate([np.diff(times) for times in late if times.size > 1]
                                   or [np.zeros(0)])
        cv = float(intervals.std() / intervals.mean()) if intervals.size else 0.0
        return sum(times.size for times in late) / len(trains) / measured_s, cv

    rate_e, cv_e = firing_summary(spike_times[:NE])
    rate_i, cv_i = firing_summary(spike_times[NE:])
    total_spikes = sum(times.size for times in spike_times)
    total_sec = time.time() - t_start
    print(f"After {burn_in_ms:.0f} ms of burn-in:")
    print(f"  E rate {rate_e:.2f} Hz, ISI CV {cv_e:.3f}")
    print(f"  I rate {rate_i:.2f} Hz, ISI CV {cv_i:.3f}")
    print(f"  {total_spikes:,} spikes in the whole run")
    print(f"Total {total_sec:.1f} s")

    return {
        "build_sec": build_sec, "drive_load_sec": load_sec, "generators_sec": generators_sec,
        "calibrate_sec": calibrate_sec, "steps_sec": steps_sec,
        "ms_per_step": steps_sec / steps * 1e3, "total_sec": total_sec,
        "rate_e_hz": rate_e, "rate_i_hz": rate_i, "total_spikes": total_spikes,
    }


if __name__ == "__main__":
    run_simulation(order=int(sys.argv[1]) if len(sys.argv) > 1 else 2500)
