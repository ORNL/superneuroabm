"""
brunel_simulation.py -- excitatory and inhibitory LIF neurons with sparse random connections and
an external Poisson drive, in the style of Brunel (2000).

It runs the kernels user_customized_brunel_soma and user_customized_brunel_synapse in
customized_step_functions/ with the parameters in customized_configs/brunel_config.yaml. It is not
a calibrated reproduction of a published result, and the rates it prints have no target.

    python brunel_simulation.py          # order 2500: 10,000 neurons, about 10 M synapses
    python brunel_simulation.py 250      # order 250: 1,000 neurons, about 100,000 synapses

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


def add_poisson_input_synapses(model, somas):
    """One external input synapse per neuron (pre_soma_id=-1 means no presynaptic soma).

    Creating the synapse delivers nothing by itself: its spikes have to be injected, see
    inject_poisson_drive.
    """
    return [
        model.create_synapse(
            breed=SYNAPSE_BREED,
            pre_soma_id=-1,
            post_soma_id=soma,
            config_name="brunel_excitatory_synapse",   # the external input is excitatory
        )
        for soma in somas
    ]


def build_brunel_network(model, order=2500, epsilon=0.1, seed=1234):
    """Create the neurons, the recurrent synapses and the external input synapses.

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

    # All neurons use one config; the first NE are the excitatory ones only because their
    # outgoing synapses below are the excitatory ones.
    somas = [
        model.create_soma(breed=SOMA_BREED, config_name="brunel_lif_neuron")
        for _ in range(N_total)
    ]

    external_synapses = add_poisson_input_synapses(model, somas)

    rng = np.random.default_rng(seed)
    synapse_count = 0

    for target_idx in range(N_total):
        # Draw CE random excitatory sources (from population 0..NE-1)
        e_sources = rng.choice(NE, size=CE, replace=False)

        for source_idx in e_sources:
            model.create_synapse(
                breed=SYNAPSE_BREED,
                pre_soma_id=somas[source_idx],
                post_soma_id=somas[target_idx],
                config_name="brunel_excitatory_synapse",
            )
            synapse_count += 1

        # Draw CI random inhibitory sources (from population NE..NE+NI-1)
        i_sources = NE + rng.choice(NI, size=CI, replace=False)

        for source_idx in i_sources:
            model.create_synapse(
                breed=SYNAPSE_BREED,
                pre_soma_id=somas[source_idx],
                post_soma_id=somas[target_idx],
                config_name="brunel_inhibitory_synapse",
            )
            synapse_count += 1

        if (target_idx + 1) % 1000 == 0:
            print(f"  Connected {target_idx + 1}/{N_total} neurons ({synapse_count} synapses so far)")

    print(f"  Created {synapse_count} recurrent and {len(external_synapses)} external synapses")

    return {
        "num_excitatory": NE,
        "num_inhibitory": NI,
        "num_total": N_total,
        "num_synapses": synapse_count,
        "connectivity": epsilon,
    }, {
        "CE": CE,
        "CI": CI,
        "somas": somas,
        "external_synapses": external_synapses,
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


def inject_poisson_drive(model, external_synapses, rate_hz, ticks, seed=1234):
    """Independent Poisson input on each external synapse: the number of events in a tick is
    delivered as one spike whose value is that count (the synapse multiplies it by its weight)."""
    rng = np.random.default_rng(seed)
    ids = np.asarray(external_synapses, dtype=np.int64)
    for start in range(0, ids.size, 1000):
        chunk = ids[start:start + 1000]
        counts = rng.poisson(rate_hz * DT, size=(chunk.size, ticks))
        row, tick = np.nonzero(counts)
        model.add_spikes(chunk[row], tick, counts[row, tick])


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
    """Build the network, inject the drive, simulate and summarise.

    :param eta: external Poisson rate as a multiple of threshold_rate_hz
    :return: (summary, spike_times), spike_times being {soma_id: [tick, ...]}
    """
    model = create_model()
    net_info, conn_info = build_brunel_network(model, order=order)
    somas = conn_info["somas"]
    NE = net_info["num_excitatory"]

    model.setup()

    ticks = round(duration_s / DT)   # simulate() counts ticks, not milliseconds
    rate_hz = eta * threshold_rate_hz(model, somas[0], conn_info["external_synapses"][0])
    print(f"External drive: {rate_hz:.0f} Hz per neuron, {rate_hz * DT:.2f} events per tick")
    t0 = time.time()
    inject_poisson_drive(model, conn_info["external_synapses"], rate_hz, ticks)
    print(f"  drawn in {time.time() - t0:.1f} s")

    print(f"Simulating {duration_s * 1e3:.0f} ms ({ticks} ticks)...")
    t0 = time.time()
    model.simulate(ticks=ticks)      # returns None; the kernels compile on the first call
    wall_clock_sec = time.time() - t0
    print(f"  done in {wall_clock_sec:.1f} s")

    spike_times = model.get_all_spike_times()

    first_tick = round(burn_in_s / DT)
    e_rate, e_cv = firing_summary(spike_times, somas[:NE], first_tick, ticks)
    i_rate, i_cv = firing_summary(spike_times, somas[NE:], first_tick, ticks)

    print(f"After {burn_in_s * 1e3:.0f} ms of burn-in:")
    print(f"  E rate {e_rate:.2f} Hz, ISI CV {e_cv:.3f}")
    print(f"  I rate {i_rate:.2f} Hz, ISI CV {i_cv:.3f}")
    if not spike_times:
        print("  WARNING: no neuron fired.")

    return {
        "firing_rate_e_hz": e_rate,
        "firing_rate_i_hz": i_rate,
        "isi_cv_e": e_cv,
        "isi_cv_i": i_cv,
        "wall_clock_sec": wall_clock_sec,
        "ticks": ticks,
        "network": net_info,
    }, spike_times


if __name__ == "__main__":
    run_simulation(order=int(sys.argv[1]) if len(sys.argv) > 1 else 2500)
