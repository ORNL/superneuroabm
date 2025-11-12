"""
Standalone test for LIF soma with mixed synapses for MPI execution.

This test can be run with mpirun for parallel execution:
    mpirun -n 4 python test_lif_mixed_synapses_mpi.py
"""

import sys
from pathlib import Path

# Add parent directory to path to allow imports
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR.parent))

import numpy as np

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments

from superneuroabm.model import NeuromorphicModel
from tests.util import vizualize_responses


def test_lif_soma_mixed_synapses(enable_internal_state_tracking=True, use_gpu=True):
    """
    Tests multi-synapse integration with a two-soma network.

    This test creates a neural circuit:
    - External input (synapse_2) -> soma_0 -> synapse_3 -> soma_1
    - External input (synapse_4) -> soma_1
    - soma_1 -> synapse_5 -> soma_0 (to test bidirectional connections)

    This verifies that soma_1 can integrate inputs from both an internal synapse
    (from soma_0) and an external synapse (synapse_4) simultaneously.
    """

    # Create NeuromorphicModel instance for testing
    model = NeuromorphicModel(enable_internal_state_tracking=enable_internal_state_tracking)

    # Define input spike patterns for synapses
    spike_times = [
        # Synapse 0 receives early, strong spikes
        [(2, 1)],  # (time_tick, spike_value)
        # Additional spike pattern for future multi-synapse tests
        [(100, 1)],  # (time_tick, spike_value)
    ]
    # Define simulation duration
    simulation_duration = 200  # Total simulation time in ticks
    sync_every_n_ticks = 1  # Synchronization interval for updates

    # Create soma_0 (LIF)
    soma_0 = model.create_soma(
        breed="lif_soma",
        config_name="config_0",
    )

    # Create soma_1 (LIF)
    soma_1 = model.create_soma(
        breed="lif_soma",
        config_name="config_0",
    )

    # Create synapse_2: external input -> soma_0
    synapse_2 = model.create_synapse(
        breed="single_exp_synapse",
        pre_soma_id=-1,  # External input
        post_soma_id=soma_0,
        config_name="no_learning_config_0",
    )

    # Create synapse_3: soma_0 -> soma_1
    synapse_3 = model.create_synapse(
        breed="single_exp_synapse",
        pre_soma_id=soma_0,
        post_soma_id=soma_1,
        config_name="no_learning_config_0",
    )

    # Create synapse_4: external input -> soma_1
    synapse_4 = model.create_synapse(
        breed="single_exp_synapse",
        pre_soma_id=-1,  # External input
        post_soma_id=soma_1,
        config_name="no_learning_config_0",
    )

    # Create synapse_5: soma_1 -> soma_0 (to test bidirectional connections)
    synapse_5 = model.create_synapse(
        breed="single_exp_synapse",
        pre_soma_id=soma_1,
        post_soma_id=soma_0,
        config_name="no_learning_config_0",
    )

    # Initialize the simulation environment
    model.setup(use_gpu=use_gpu)

    # Inject spikes into synapse_2 (external -> soma_0)
    for spike in spike_times[0]:
        model.add_spike(synapse_id=synapse_2, tick=spike[0], value=spike[1])

    # Inject spikes into synapse_4 (external -> soma_1) using second spike pattern
    for spike in spike_times[1]:
        model.add_spike(synapse_id=synapse_4, tick=spike[0], value=spike[1])

    # Run simulation
    model.simulate(
        ticks=simulation_duration, update_data_ticks=sync_every_n_ticks
    )

    # Generate visualization
    caller_name = "test_lif_soma_mixed_synapses"
    vizualize_responses(model, vthr=-45, fig_name=f"{caller_name}_mpi.png")

    print(f"Test completed successfully!")
    return model


if __name__ == "__main__":
    # Run the test
    # This can be executed with mpirun:
    #   mpirun -n 4 python test_lif_mixed_synapses_mpi.py

    model = test_lif_soma_mixed_synapses(enable_internal_state_tracking=True, use_gpu=True)
    print("Simulation finished. Check output visualization.")
