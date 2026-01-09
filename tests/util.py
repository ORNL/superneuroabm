from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
from superneuroabm.model import NeuromorphicModel


def vizualize_responses(model: NeuromorphicModel, vthr: int, fig_name: str, figsize=None) -> None:

    # Filter out invalid soma IDs (negative values like -1 or NaN used for external inputs)
    all_soma_ids = model.soma2synapse_map.keys()
    soma_ids = [sid for sid in all_soma_ids if sid >= 0 and not np.isnan(sid)]
    synapse_ids = model.synapse2soma_map.keys()

    # Always print spike times first
    print("\n=== Spike Times ===")
    for soma_id in soma_ids:
        spike_times = model.get_spike_times(soma_id=soma_id)
        print(f"Soma {soma_id} spike times: {spike_times}")
    print("===================\n")

    # Check if internal state tracking is enabled
    if not model.enable_internal_state_tracking:
        # When tracking is disabled, only print spike times (already done above)
        return

    # Calculate total number of plots needed
    total_plot_count = len(soma_ids)
    for synapse_id in synapse_ids:
        internal_learning_state_synapse = np.array(
            model.get_internal_learning_states_history(agent_id=synapse_id)
        )
        num_plots = 1 if internal_learning_state_synapse.size == 0 else 3
        total_plot_count += num_plots

    # Auto-calculate figure size if not provided
    if figsize is None:
        width = 12  # Fixed width
        height_per_subplot = 2  # Height per subplot in inches
        min_height = 6  # Minimum height
        calculated_height = max(min_height, total_plot_count * height_per_subplot)
        figsize = (width, calculated_height)

    # Generate visualization comparing membrane potential and synaptic currents
    plt.figure(figsize=figsize)

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

    current_subplot = len(soma_ids) + 1
    
    for synapse_id in synapse_ids:
        # Get internal states history for the synapse
        internal_states_history_synapse = np.array(
            model.get_internal_states_history(agent_id=synapse_id)
        )
        # Get the internal learning states for synapses
        internal_learning_state_synapse = np.array(
            model.get_internal_learning_states_history(agent_id=synapse_id)
        )
        num_plots = 1 if internal_learning_state_synapse.size == 0 else 3
        
        # Plot synaptic current
        plt.subplot(total_plot_count, 1, current_subplot)
        plt.plot(
            internal_states_history_synapse[:, 0],
            "g-",
            label=f"Synapse {synapse_id} Current",
        )
        plt.ylabel(f"Synapse {synapse_id} Current")
        plt.legend()
        current_subplot += 1

        # print(internal_learning_state_synapse)

        if num_plots > 1:
            # Plot pre trace
            plt.subplot(total_plot_count, 1, current_subplot)
            plt.plot(
                internal_learning_state_synapse[:, 0],
                "r-",
                label=f"Synapse {synapse_id} pre_trace",
            )
            plt.ylabel(f"Synapse {synapse_id} pre-trace")
            plt.legend()
            current_subplot += 1

            # Plot post trace
            plt.subplot(total_plot_count, 1, current_subplot)
            plt.plot(
                internal_learning_state_synapse[:, 1],
                "r-",
                label=f"Synapse {synapse_id} post-trace",
            )
            plt.ylabel(f"Synapse {synapse_id} post-trace")
            plt.legend()
            current_subplot += 1

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
