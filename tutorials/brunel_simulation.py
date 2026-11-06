"""
brunel_simulation.py -- excitatory and inhibitory LIF neurons with sparse random connections and
an external Poisson drive, in the style of Brunel (2000).

It runs the kernels user_customized_brunel_soma and user_customized_brunel_synapse in
customized_step_functions/ with the parameters in customized_configs/brunel_config.yaml. It is not
a calibrated reproduction of a published result, and the rates it prints have no target.

    python brunel_simulation.py          # order 2500: 10,000 neurons, about 10 M synapses
    python brunel_simulation.py 250      # order 250: 1,000 neurons, about 100,000 synapses

Three things keep the run short and its timing readable:

  * The network is handed to the model as arrays (create_somas, create_synapses), not built with
    one create_synapse call per synapse.
  * The Poisson drive is drawn once and saved under data/; later runs load that file.
  * The first simulate() call, which also builds the device buffers and the input event list and
    compiles the kernels, is timed apart from the ticks.

Run it from any directory with the SuperNeuroABM environment active. Generated
step_func_code_*.py files land in the working directory.
"""

import sys
import time
from pathlib import Path

import numpy as np

from superneuroabm.model import NeuromorphicModel

TUTORIALS_DIR = Path(__file__).resolve().parent
CONFIG = TUTORIALS_DIR / "customized_configs" / "brunel_config.yaml"
STEP_FUNC_DIR = TUTORIALS_DIR / "customized_step_functions"
DATA_DIR = TUTORIALS_DIR / "data"
sys.path.insert(0, str(STEP_FUNC_DIR))   # the tutorial-owned kernels are imported by module name

from user_customized_brunel_soma import user_customized_brunel_soma
from user_customized_brunel_synapse import user_customized_brunel_synapse

SOMA_BREED = "user_customized_brunel_soma"
SYNAPSE_BREED = "user_customized_brunel_synapse"
DT = 1e-4   # seconds per tick: 0.1 ms (the default is 1 ms)


def create_model():
    """A model with the two kernels registered, ready for neurons and synapses."""
    model = NeuromorphicModel(user_config=CONFIG, enable_internal_states_tracking=False)
    model.set_global_property_value("dt", DT)
    model.register_soma_type(
        name=SOMA_BREED,
        step_func=user_customized_brunel_soma,
        step_func_path=STEP_FUNC_DIR / f"{SOMA_BREED}.py",
    )
    model.register_synapse_type(
        name=SYNAPSE_BREED,
        step_func=user_customized_brunel_synapse,
        step_func_path=STEP_FUNC_DIR / f"{SYNAPSE_BREED}.py",
        event_capable=True,
    )
    return model


def build_brunel_network(model, order=2500, epsilon=0.1, seed=1234):
    """Create the neurons, the recurrent synapses and the external input synapses.

    Neurons and synapses go to the model as arrays: one create_somas call, and one
    create_synapses call per kind of synapse (external input, excitatory, inhibitory), each
    naming its block of the YAML. A synapse is its pre and post neuron ids; pre = -1 marks an
    external input, which delivers nothing by itself: its spikes are injected, see
    inject_poisson_drive. The calls return the ids of what they made.

    :param order: scaling parameter: 4 * order neurons, 80 % excitatory
    :param epsilon: fraction of each population a neuron receives input from
    :return: (network_info, connectivity_info)
    """
    NE = int(0.8 * (4 * order))
    NI = int(0.2 * (4 * order))
    N_total = NE + NI
    CE = int(epsilon * NE)
    CI = int(epsilon * NI)

    print(f"Building network: {NE} E + {NI} I = {N_total} neurons")
    print(f"  Excitatory in-degree: {CE}")
    print(f"  Inhibitory in-degree: {CI}")

    # Each neuron draws CE excitatory sources (neurons 0 .. NE-1) and CI inhibitory sources
    # (neurons NE .. N-1). The first NE neurons are the excitatory ones only because their
    # outgoing synapses are the excitatory ones; all neurons use one config.
    rng = np.random.default_rng(seed)
    sources = np.empty((N_total, CE + CI), dtype=np.int64)
    for target in range(N_total):
        sources[target, :CE] = rng.choice(NE, size=CE, replace=False)
        sources[target, CE:] = NE + rng.choice(NI, size=CI, replace=False)

    somas = model.create_somas(N_total, breed=SOMA_BREED, config_name="brunel_lif_neuron")

    # One external input synapse per neuron, then every neuron's CE excitatory and CI
    # inhibitory incoming synapses.
    external_synapses = model.create_synapses(
        -1, somas, breed=SYNAPSE_BREED, config_name="brunel_excitatory_synapse")
    model.create_synapses(
        sources[:, :CE].ravel(), np.repeat(somas, CE),
        breed=SYNAPSE_BREED, config_name="brunel_excitatory_synapse")
    model.create_synapses(
        sources[:, CE:].ravel(), np.repeat(somas, CI),
        breed=SYNAPSE_BREED, config_name="brunel_inhibitory_synapse")
    synapse_count = N_total * (CE + CI)

    print(f"  Created {synapse_count} recurrent and {N_total} external synapses")

    return {
        "num_excitatory": NE,
        "num_inhibitory": NI,
        "num_total": N_total,
        "num_synapses": synapse_count,
        "connectivity": epsilon,
    }, {
        "CE": CE,
        "CI": CI,
        "somas": somas.tolist(),
        "external_synapses": external_synapses.tolist(),
    }


def threshold_rate_hz(model, soma, synapse):
    """The Poisson rate at which the mean membrane potential just reaches threshold.

    One event adds ``weight`` to the current, which decays by P = exp(-dt/tau_syn) per tick, so
    the mean current at rate r is r * dt * weight / (1 - P), and the soma's mean V - E_L is that
    times tau_m / C_m. Solved for r at V - E_L = V_th - E_L.
    """
    soma_hp = model.get_hyperparameters(soma)
    weight = model.get_hyperparameters(synapse)["weight"]
    P = np.exp(-DT / soma_hp["tau_syn_0"])
    return ((soma_hp["V_th"] - soma_hp["E_L"]) * soma_hp["C_m"] * (1 - P)
            / (weight * DT * soma_hp["tau_m"]))


def poisson_drive(n_inputs, ticks, rate_hz, seed=1234):
    """Independent Poisson input for each external synapse, as the number of events per tick.

    Drawn once and saved in data/; later runs with the same size, rate and seed load the file,
    so every run sees the same input and spends no time drawing it.

    :return: (counts, path, loaded): counts[tick, input] as uint8, the file, and whether it
        was loaded (True) or drawn and saved by this call (False)
    """
    path = DATA_DIR / f"brunel_poisson_drive_{n_inputs}x{ticks}.npz"
    if path.exists():
        with np.load(path) as saved:
            if (np.isclose(float(saved["rate_hz"]), rate_hz, rtol=1e-6)
                    and float(saved["dt"]) == DT and int(saved["seed"]) == seed):
                return saved["counts"], path, True

    rng = np.random.default_rng(seed)
    counts = np.empty((ticks, n_inputs), dtype=np.uint8)
    for start in range(0, n_inputs, 1000):
        drawn = rng.poisson(rate_hz * DT, size=(min(1000, n_inputs - start), ticks))
        if drawn.max() > 255:
            raise ValueError(f"{drawn.max()} events in one tick do not fit the saved uint8 counts")
        counts[:, start:start + drawn.shape[0]] = drawn.T
    DATA_DIR.mkdir(exist_ok=True)
    np.savez(path, counts=counts, rate_hz=rate_hz, dt=DT, seed=seed)
    return counts, path, False


def inject_poisson_drive(model, external_synapses, counts):
    """Schedule the drive: the events of an input in a tick are delivered as one spike whose
    value is their count (the synapse multiplies it by its weight)."""
    ids = np.asarray(external_synapses, dtype=np.int64)
    for start in range(0, counts.shape[0], 1000):
        window = counts[start:start + 1000]
        tick, row = np.nonzero(window)
        model.add_spikes(ids[row], start + tick, window[tick, row])


def firing_summary(spike_times, somas, first_tick, last_tick):
    """Firing rate (Hz) and the CV of the pooled interspike intervals of some neurons, counting
    spikes in [first_tick, last_tick). get_all_spike_times() gives spike times in ticks."""
    count = 0
    intervals = []
    for soma in somas:
        ticks = np.asarray(spike_times.get(soma, []), dtype=np.int64)
        ticks = ticks[(ticks >= first_tick) & (ticks < last_tick)]
        count += ticks.size
        if ticks.size > 1:
            intervals.append(np.diff(ticks) * DT)
    intervals = np.concatenate(intervals) if intervals else np.zeros(0)
    cv = float(intervals.std() / intervals.mean()) if intervals.size else 0.0
    return count / (len(somas) * (last_tick - first_tick) * DT), cv


def run_simulation(order=2500, duration_s=1.5, burn_in_s=0.5, eta=2.0):
    """Build the network, load the drive, simulate and summarise.

    :param eta: external Poisson rate as a multiple of threshold_rate_hz
    :return: (summary, spike_times), spike_times being {soma_id: [tick, ...]}
    """
    t_start = time.time()
    model = create_model()
    net_info, conn_info = build_brunel_network(model, order=order)
    somas = conn_info["somas"]
    external_synapses = conn_info["external_synapses"]
    NE = net_info["num_excitatory"]
    build_sec = time.time() - t_start
    print(f"  built in {build_sec:.1f} s")

    t0 = time.time()
    model.setup()
    setup_sec = time.time() - t0
    print(f"setup() in {setup_sec:.1f} s")

    ticks = round(duration_s / DT)   # simulate() counts ticks, not milliseconds
    rate_hz = eta * threshold_rate_hz(model, somas[0], external_synapses[0])
    print(f"External drive: {rate_hz:.0f} Hz per neuron, {rate_hz * DT:.2f} events per tick")
    t0 = time.time()
    counts, drive_file, loaded = poisson_drive(len(external_synapses), ticks, rate_hz)
    inject_poisson_drive(model, external_synapses, counts)
    drive_sec = time.time() - t0
    print(f"  {'loaded from' if loaded else 'drawn and saved to'} {drive_file} "
          f"({np.count_nonzero(counts):,} spikes) in {drive_sec:.1f} s")
    del counts

    # The first simulate() call does one-time work before its ticks: it builds the device
    # buffers, compiles the injected spikes into the event list and compiles the kernels. One
    # tick is run on its own so that this is not counted as simulation time. simulate() keeps
    # the spikes of the latest call only, so the two calls' spikes are put together.
    print(f"Simulating {duration_s * 1e3:.0f} ms ({ticks} ticks)...")
    t0 = time.time()
    model.simulate(ticks=1)
    first_call_sec = time.time() - t0
    spike_times = model.get_all_spike_times()
    print(f"  first simulate() call (1 tick, one-time device build): {first_call_sec:.1f} s")

    t0 = time.time()
    model.simulate(ticks=ticks - 1)
    ticks_sec = time.time() - t0
    for soma, soma_ticks in model.get_all_spike_times().items():
        spike_times.setdefault(soma, []).extend(soma_ticks)
    ms_per_tick = ticks_sec / (ticks - 1) * 1e3
    print(f"  remaining {ticks - 1} ticks: {ticks_sec:.2f} s ({ms_per_tick:.3f} ms per tick)")

    first_tick = round(burn_in_s / DT)
    e_rate, e_cv = firing_summary(spike_times, somas[:NE], first_tick, ticks)
    i_rate, i_cv = firing_summary(spike_times, somas[NE:], first_tick, ticks)

    print(f"After {burn_in_s * 1e3:.0f} ms of burn-in:")
    print(f"  E rate {e_rate:.2f} Hz, ISI CV {e_cv:.3f}")
    print(f"  I rate {i_rate:.2f} Hz, ISI CV {i_cv:.3f}")
    if not spike_times:
        print("  WARNING: no neuron fired.")
    print(f"Total {time.time() - t_start:.1f} s")

    return {
        "firing_rate_e_hz": e_rate,
        "firing_rate_i_hz": i_rate,
        "isi_cv_e": e_cv,
        "isi_cv_i": i_cv,
        "build_sec": build_sec,
        "setup_sec": setup_sec,
        "drive_sec": drive_sec,
        "first_call_sec": first_call_sec,
        "ticks_sec": ticks_sec,
        "ms_per_tick": ms_per_tick,
        "ticks": ticks,
        "network": net_info,
    }, spike_times


if __name__ == "__main__":
    run_simulation(order=int(sys.argv[1]) if len(sys.argv) > 1 else 2500)
