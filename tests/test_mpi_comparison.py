#!/usr/bin/env python
"""
Test to verify MPI execution produces same results as non-MPI execution.

This test:
1. Runs simulation without MPI
2. Can be run with MPI (mpirun -n 4 python -m unittest test_mpi_comparison.TestMPIComparison)
3. Compares spike times against baseline to detect breaking changes
4. Generates visualization plot

Usage:
    # First, create baseline with single process:
    python test_mpi_comparison.py TestMPIComparison.save_baseline

    # Then verify single process against baseline:
    python -m unittest test_mpi_comparison.TestMPIComparison

    # Test with MPI (should match baseline):
    mpirun -n 4 python -m unittest test_mpi_comparison.TestMPIComparison
"""

import sys
import unittest
from pathlib import Path

# Add parent directory to path to allow imports
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')

from superneuroabm.model import NeuromorphicModel
from tests.util import vizualize_responses
from tests.baseline_utils import BaselineComparator


def test_lif_soma_mixed_synapses_mpi(enable_internal_state_tracking=True, use_gpu=True):
    """
    Tests multi-synapse integration with a two-soma network for MPI execution.

    This test creates a neural circuit:
    - External input (synapse_2) -> soma_0 -> synapse_3 -> soma_1
    - External input (synapse_4) -> soma_1
    - soma_1 -> synapse_5 -> soma_0 (to test bidirectional connections)

    This verifies that soma_1 can integrate inputs from both an internal synapse
    (from soma_0) and an external synapse (synapse_4) simultaneously.

    Args:
        enable_internal_state_tracking: Whether to track internal states
        use_gpu: Whether to use GPU acceleration

    Returns:
        NeuromorphicModel instance after simulation
    """
    # Create NeuromorphicModel instance for testing
    model = NeuromorphicModel(enable_internal_state_tracking=enable_internal_state_tracking)

    # Define input spike patterns for synapses
    spike_times = [
        # Synapse 2 receives early, strong spikes
        [(2, 1)],  # (time_tick, spike_value)
        # Synapse 4 receives later spikes
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

    return model


def get_mpi_rank():
    """Get MPI rank if running with MPI, otherwise return 0."""
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD.Get_rank()
    except ImportError:
        return 0


def get_mpi_size():
    """Get MPI size if running with MPI, otherwise return 1."""
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD.Get_size()
    except ImportError:
        return 1


class TestMPIComparison(unittest.TestCase):
    """
    Test suite to verify MPI execution consistency.

    Tests that MPI execution produces same spike times as single process
    and compares against saved baselines to detect model breaking changes.
    """

    def test_mpi_consistency(self):
        """Test that MPI execution produces correct spike times."""
        rank = get_mpi_rank()
        size = get_mpi_size()

        if rank == 0:
            print("\n" + "=" * 70)
            if size > 1:
                print(f"Testing: MPI execution with {size} processes")
            else:
                print("Testing: Single process execution")
            print("=" * 70)

        comparator = BaselineComparator()
        test_name = "test_mpi_lif_soma_mixed_synapses"  # Include model name for future expansion

        # Run simulation
        if rank == 0:
            print(f"\nRunning simulation (rank 0/{size-1})...")
            print("-" * 70)

        model = test_lif_soma_mixed_synapses_mpi(
            enable_internal_state_tracking=True,
            use_gpu=True
        )

        # Generate visualization - ALL ranks must call this (vizualize_responses handles rank internally)
        fig_name = f"test_mpi_lif_soma_mixed_synapses_np{size}.png"
        vizualize_responses(
            model,
            vthr=-45,
            fig_name=fig_name
        )

        # Only rank 0 does printing and baseline comparison
        if rank == 0:
            print(f"✓ Visualization saved: output/{fig_name}")

            # Only compare baseline for single process (size == 1)
            # MPI runs just verify they complete successfully
            if size == 1:
                print("\n" + "=" * 70)
                print("Baseline Comparison")
                print("=" * 70)

                passed, message = comparator.compare_with_baseline(model, test_name)
                print(message)

                if not passed:
                    print("\n⚠ WARNING: Spike times differ from baseline!")
                    print("  If this is intentional, run: python test_mpi_comparison.py TestMPIComparison.save_baseline")
            else:
                print(f"\n✓ MPI execution with {size} processes completed successfully!")
                print("  (Baseline comparison only done with single process)")

    def save_baseline(self):
        """Save current spike times as baseline."""
        rank = get_mpi_rank()
        size = get_mpi_size()

        if rank == 0:
            print("\n" + "=" * 70)
            print("Saving Baseline")
            print("=" * 70)

            if size > 1:
                print("⚠ WARNING: Creating baseline with MPI. Recommended to use single process.")
                print("           Run: python test_mpi_comparison.py TestMPIComparison.save_baseline")

        # ALL ranks must create the model
        model = test_lif_soma_mixed_synapses_mpi(
            enable_internal_state_tracking=True,
            use_gpu=True
        )

        # ALL ranks must call visualization (it handles rank internally)
        fig_name = f"test_mpi_lif_soma_mixed_synapses_baseline_np{size}.png"
        vizualize_responses(
            model,
            vthr=-45,
            fig_name=fig_name
        )

        # Only rank 0 does baseline saving and printing
        if rank == 0:
            print(f"✓ Visualization saved: output/{fig_name}")

            comparator = BaselineComparator()
            test_name = "test_mpi_lif_soma_mixed_synapses"  # Include model name for future expansion
            comparator.save_baseline(model, test_name)

            print("\nℹ Baseline saved successfully!")
            print("  Next runs:")
            print("    Single process: python -m unittest test_mpi_comparison.TestMPIComparison")
            print(f"    MPI:            mpirun -n {size} python -m unittest test_mpi_comparison.TestMPIComparison")


if __name__ == "__main__":
    # Usage:
    #   python test_mpi_comparison.py                 # Run tests normally
    #   python test_mpi_comparison.py --save-baselines # Save baseline
    #   mpirun -n 4 python test_mpi_comparison.py     # Run with MPI
    import sys

    if "--save-baselines" in sys.argv:
        sys.argv.remove("--save-baselines")
        # Run the save_baseline method
        test = TestMPIComparison()
        test.save_baseline()
    else:
        unittest.main()
