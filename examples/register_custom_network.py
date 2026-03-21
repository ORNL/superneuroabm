"""
Example: Register custom components and run a network.

Demonstrates the SuperNeuroABM registration API with user-defined step
functions that live OUTSIDE the library (in user_step_functions/).
Configurations are loaded from a user YAML file (custom_components_config.yaml).

Network topology:
  external input --> [exp_lif_soma pre] --alpha_synapse+STDP--> [exp_lif_soma post]

Custom components used:
  - exp_lif_soma       (custom soma with exponential decay)
  - alpha_synapse      (second-order alpha-kernel synapse)
  - bounded_stdp       (weight-bounded multiplicative STDP)
  - symmetric_stdp     (correlation-based symmetric STDP)

Both STDP rules are registered to show multi-rule registration;
bounded_stdp is used on the internal synapse.

Requirements:
  - CUDA-capable GPU
  - superneuroabm installed (pip install -e .)
"""

from pathlib import Path

from superneuroabm.model import NeuromorphicModel

from user_step_functions.exp_lif_soma import exp_lif_soma_step_func
from user_step_functions.alpha_synapse import alpha_synapse_step_func
from user_step_functions.bounded_stdp import bounded_stdp
from user_step_functions.symmetric_stdp import symmetric_stdp

# Paths to the source files (needed by the JIT compiler)
SCRIPT_DIR = Path(__file__).resolve().parent
USER_FUNCS_DIR = SCRIPT_DIR / "user_step_functions"
SOMA_PATH = USER_FUNCS_DIR / "exp_lif_soma.py"
SYNAPSE_PATH = USER_FUNCS_DIR / "alpha_synapse.py"
BOUNDED_STDP_PATH = USER_FUNCS_DIR / "bounded_stdp.py"
SYMMETRIC_STDP_PATH = USER_FUNCS_DIR / "symmetric_stdp.py"

# User's own config file (lives alongside the script, not inside the library)
USER_CONFIG_PATH = SCRIPT_DIR / "custom_components_config.yaml"


def main():
    model = NeuromorphicModel(user_config=USER_CONFIG_PATH, enable_internal_state_tracking=True)

    # ── Register custom soma ──────────────────────────────────────────
    model.register_soma_type(
        name="exp_lif_soma",
        step_func=exp_lif_soma_step_func,
        step_func_path=SOMA_PATH,
    )

    # ── Register custom synapse ───────────────────────────────────────
    model.register_synapse_type(
        name="alpha_synapse",
        step_func=alpha_synapse_step_func,
        step_func_path=SYNAPSE_PATH,
    )

    # ── Register TWO custom learning rules ────────────────────────────
    bounded_id = model.register_learning_rule(
        step_func=bounded_stdp,
        step_func_path=BOUNDED_STDP_PATH,
    )
    symmetric_id = model.register_learning_rule(
        step_func=symmetric_stdp,
        step_func_path=SYMMETRIC_STDP_PATH,
    )
    print(f"Registered bounded_stdp  (ID={bounded_id})")
    print(f"Registered symmetric_stdp (ID={symmetric_id})")

    # ── Build network ─────────────────────────────────────────────────
    # Two custom exp-LIF somas
    soma_pre = model.create_soma(breed="exp_lif_soma", config_name="config_0")
    soma_post = model.create_soma(breed="exp_lif_soma", config_name="config_0")

    # External input via built-in single_exp_synapse (no learning)
    syn_input = model.create_synapse(
        breed="single_exp_synapse",
        pre_soma_id=-1,
        post_soma_id=soma_pre,
        config_name="config_0",
    )

    # Internal connection via custom alpha synapse with bounded STDP
    syn_learn = model.create_synapse(
        breed="alpha_synapse",
        pre_soma_id=soma_pre,
        post_soma_id=soma_post,
        config_name="config_0",
        learning_rule="exp_pair_wise_stdp",
        overrides={"learning_hyperparameters": {"stdp_type": float(bounded_id)}},
    )

    model.setup(use_gpu=True)

    # ── Read initial weight ───────────────────────────────────────────
    initial_weight = model.get_agent_property_value(
        id=syn_learn, property_name="hyperparameters"
    )[0]

    # ── Inject input spikes (bursts of 3 to ensure firing) ───────────
    for burst_start in [3, 20, 40, 60, 80]:
        for t in range(burst_start, burst_start + 3):
            model.add_spike(synapse_id=syn_input, tick=t, value=1.0)

    # ── Run simulation ────────────────────────────────────────────────
    model.simulate(ticks=120)

    # ── Report results ────────────────────────────────────────────────
    learned_weight = model.get_agent_property_value(
        id=syn_learn, property_name="hyperparameters"
    )[0]
    spikes_pre = model.get_spike_times(soma_id=soma_pre)
    spikes_post = model.get_spike_times(soma_id=soma_post)

    print(f"\nPre-soma spike times:  {spikes_pre}")
    print(f"Post-soma spike times: {spikes_post}")
    print(f"\nInitial weight: {initial_weight}")
    print(f"Learned weight: {learned_weight}")
    print(f"Weight changed: {initial_weight != learned_weight}")


if __name__ == "__main__":
    main()
