from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
from superneuroabm.model import NeuromorphicModel


def vizualize_responses(model: NeuromorphicModel, vthr: int, fig_name: str) -> None:

    soma_ids = model.soma2synapse_map.keys()
    synapse_ids = model.synapse2soma_map.keys()

    # Generate visualization comparing membrane potential and synaptic currents
    plt.figure(figsize=(12, 8))

    total_plot_count = len(soma_ids) + len(synapse_ids) * 3

    for i, soma_id in enumerate(soma_ids):
        # Get internal states history for the soma
        internal_states_history_soma = np.array(
            model.get_internal_states_history(agent_id=soma_id)
        )

        # Plot membrane potential
        plt.subplot(total_plot_count, 1, i + 1)
        plt.plot(internal_states_history_soma[:, 0], "b-", label=f"Soma {soma_id}")
        plt.axhline(y=vthr, color="r", linestyle="--", label="Threshold")
        plt.ylabel("Membrane Pot. (mV)")
        plt.title(f"Soma {soma_id}")
        plt.legend()

    for i, synapse_id in enumerate(synapse_ids):
        # Get internal states history for the synapse
        internal_states_history_synapse = np.array(
            model.get_internal_states_history(agent_id=synapse_id)
        )
        # Get the internal learning states for synapses
        internal_learning_state_synapse = np.array(
            model.get_internal_learning_states_history(agent_id=synapse_id)
        )
        num_plots = 1 if internal_learning_state_synapse.size == 0 else 3
        # Plot synaptic current from synapse A
        plt.subplot(total_plot_count, 1, len(soma_ids) + (i * num_plots) + 1)
        plt.plot(
            internal_states_history_synapse[:, 0],
            "g-",
            label=f"Synapse {synapse_id} Current",
        )
        plt.ylabel(f"Synapse {synapse_id} Current")
        plt.legend()

        print(internal_learning_state_synapse)

        if num_plots > 1:
            # Plot pre and post traces for synapse A
            plt.subplot(total_plot_count, 1, len(soma_ids) + (i * num_plots) + 2)
            plt.plot(
                internal_learning_state_synapse[:, 0],
                "r-",
                label=f"Synapse {synapse_id} pre_trace",
            )
            plt.ylabel("Synaptic A pre-trace ")
            plt.legend()

            plt.subplot(total_plot_count, 1, len(soma_ids) + (i * num_plots) + 3)
            plt.plot(
                internal_learning_state_synapse[:, 1],
                "r-",
                label=f"Synapse {synapse_id} post-trace",
            )
            plt.ylabel(f"Synaptic {synapse_id} post-trace ")
            plt.legend()

    plt.tight_layout()
    dir_path = Path(__file__).resolve().parent / "output"
    dir_path.mkdir(parents=True, exist_ok=True)
    fig_path = dir_path / fig_name
    plt.savefig(
        fig_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
