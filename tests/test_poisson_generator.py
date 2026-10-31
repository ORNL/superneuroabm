import numpy as np
from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR.parent))


from superneuroabm.model import NeuromorphicModel
from superneuroabm.poisson_generator import PoissonInput

# Example: one target soma with an external input synapse
model = NeuromorphicModel()
target = model.create_soma(breed="lif_soma", config_name="config_0",
                overrides={"hyperparameters": {
                    "vthr": 20.0, "tref": 1e-3,
                    "vreset": 0.0, "vrest": 0.0, "C": 250e-12, "R": 4E7,
                    "tref_allows_integration": 0, "scaling_factor": 1E-4,
                    },
                "internal_states": {
                    "v": 5.7,
                    },
                },)
input_synapse = model.create_synapse(
    breed="single_exp_synapse",
    pre_soma_id=-1,          # external input
    post_soma_id=target,
    config_name="config_0",
    overrides={"hyperparameters": {
        "tau_fall": 0.325E-3, "weight": 3E-3,
        },
    },
)

dt_ms = 0.1
duration_ms = 1000.0
rate_hz = 20 #20_856.0           # set this to your desired input frequency
n_ticks = round(duration_ms / dt_ms)
print(f"Simulating {n_ticks} ticks at dt={dt_ms:.1f} ms, duration={duration_ms:.0f} ms, target input rate={rate_hz:.0f} Hz")
model.setup()

generator = PoissonInput(rate_hz=rate_hz, dt_ms=dt_ms, seed=42)
counts = generator.add_to_model(model, input_synapse, n_ticks)

print("Generated input events:", counts)

model.simulate(ticks=n_ticks)

print("Generated input events:", int(counts.sum()))
print("Target output spikes:", model.get_spike_times(soma_id=target))

# get the membrane potential trace:
vm = np.asarray(model.get_internal_states_history(agent_id=target))[:, 0]

# plot the results:
import matplotlib.pyplot as plt
time_ms = np.arange(len(vm)) * dt_ms
plt.figure(figsize=(9, 4))
plt.plot(time_ms, vm, label="Membrane potential")
plt.axhline(20.0, color="tab:red", linestyle="--", label="Spike threshold")
spike_times_ms = np.asarray(model.get_spike_times(soma_id=target)) * dt_ms
if spike_times_ms.size:
    plt.scatter(spike_times_ms, np.full(spike_times_ms.size, 20.0),
                color="black", marker="|", label="Output spikes")
plt.xlabel("Time (ms)")
plt.ylabel("Membrane potential (mV)")
plt.title(f"Poisson input at {rate_hz:.0f} Hz, dt={dt_ms:.1f} ms, duration={duration_ms:.0f} ms")
plt.legend()
plt.tight_layout()
plt.savefig(CURRENT_DIR / "poisson_input_test.png", dpi=300)