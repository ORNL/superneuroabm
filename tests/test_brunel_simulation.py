import numpy as np
from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR.parent))

from superneuroabm.brunel import build_brunel_network
from superneuroabm.model import NeuromorphicModel

import matplotlib.pyplot as plt

import time

# Build the network in memory.
somas, synapses = build_brunel_network(
    somas_per_rank=1125,
    excitatory_fraction=0.8,       # 800 excitatory, 200 inhibitory
    excitatory_in_degree=9000,       # recurrent E inputs per soma
    inhibitory_in_degree=2250,       # recurrent I inputs per soma
    inhibitory_weight_ratio=5.0,   # inhibitory weight = -5 × excitatory weight
    external_synapses_per_soma=1,
    # soma_overrides={"hyperparameters": {
    #         "vthr": -40.0,
    #         "tref": 1e-3,
    #     },},
    # soma_config_jitter={
    #     "vthr": (-50.0, 1.0),  # mean -50, standard deviation 1
    #     "tref": (2.5, 0.2),
    # },
    # synapse_overrides={"hyperparameters": {
    #         "tau_fall": 5E-3,
    #     },},
    # synapse_config_jitter={
    #     "tau_fall": (3.0, 0.25),
    #     "weight": (0.1, 0.01),
    # },
    seed=42,
)

# Create and initialize the model.
model = NeuromorphicModel(enable_internal_states_tracking=False)
model.create_from_lists(somas, synapses)
model.set_global_property_value("dt", 1E-3)
model.setup()

# Find external input synapses. Each has pre == -1.
input_synapse_ids = [
    syn["id"] for syn in synapses
    if syn["pre"] == -1
]

# Schedule independent Poisson input spikes.
ticks = 300       # model timestep is 1 ms
dt_ms = 1.0
rate_hz = 10.0
p_spike_per_tick = rate_hz * dt_ms / 1000.0

rng = np.random.default_rng(123)
for syn_id in input_synapse_ids:
    spike_ticks = np.flatnonzero(
        rng.random(ticks) < p_spike_per_tick
    ) + 1  # input ticks are 1-indexed

    if spike_ticks.size:
        model.add_spike_list(
            synapse_id=syn_id,
            spike_list=[[int(t), 1.0] for t in spike_ticks],
        )

# Run the simulation and retrieve soma spike times.
t_start = time.time()
model.simulate(ticks=ticks, update_data_ticks=1)
t_end = time.time()
spikes_by_soma = model.get_all_spike_times()

print(f"Input synapses: {len(input_synapse_ids)}")
print(f"Somas with recorded spikes: {len(spikes_by_soma)}")
print("Example:", list(spikes_by_soma.items())[:5])

# Get the spike frequencies for each soma.
spike_frequencies = {
    soma_id: len(spike_ticks) / (ticks * dt_ms / 1000.0)
    for soma_id, spike_ticks in spikes_by_soma.items()
}   
# print("Spike frequencies (Hz):", spike_frequencies)
print("Mean spike frequency (Hz):", np.mean(list(spike_frequencies.values())))
print("Simulation time (s):", t_end - t_start)
# Plot the spike raster.
spikes_by_soma = model.get_all_spike_times()

exc_ids, exc_times = [], []
inh_ids, inh_times = [], []
n_exc = round(0.8 * 1125)  # matches somas_per_rank and excitatory_fraction above

for soma_id, spike_ticks in spikes_by_soma.items():
    if soma_id < n_exc:
        exc_ids.extend([soma_id] * len(spike_ticks))
        exc_times.extend(spike_ticks)
    else:
        inh_ids.extend([soma_id] * len(spike_ticks))
        inh_times.extend(spike_ticks)

plt.figure(figsize=(12, 5))
plt.scatter(np.array(exc_times) * dt_ms, exc_ids,
            s=2, color="tab:blue", label="Excitatory")
plt.scatter(np.array(inh_times) * dt_ms, inh_ids,
            s=2, color="tab:red", label="Inhibitory")

plt.xlabel("Time (ms)")
plt.ylabel("Neuron ID")
plt.title("Brunel network spike raster")
plt.legend(markerscale=3)
plt.tight_layout()
plt.savefig("brunel_spike_raster_sna.png", dpi=300)