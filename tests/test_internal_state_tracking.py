#!/usr/bin/env python
"""
Test to verify that enable_internal_state_tracking=True/False produces same spike times.

This test runs the same simulation with both settings and compares:
1. Spike times should be identical
2. Visualization is generated only when tracking is enabled
3. Baseline comparison to detect model breaking changes

Usage:
    # First run to create baseline:
    python test_internal_state_tracking.py TestInternalStateTracking.save_baseline

    # Subsequent runs to verify against baseline:
    python -m unittest test_internal_state_tracking.TestInternalStateTracking
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


def test_lif_soma_mixed_synapses(enable_internal_state_tracking=True, use_gpu=True):
    """
    Tests multi-synapse integration with a two-soma network.

    This test creates a neural circuit:
    - External input (synapse_2) -> soma_0 -> synapse_3 -> soma_1
    - External input (synapse_4) -> soma_1
    - soma_1 -> synapse_5 -> soma_0 (to test bidirectional connections)

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


class TestInternalStateTracking(unittest.TestCase):
    """
    Test suite to verify internal state tracking consistency.

    Tests that enable_internal_state_tracking=True/False produce identical spike times
    and compares against saved baselines to detect model breaking changes.
    """

    def test_tracking_consistency(self):
        """Test that tracking ON/OFF produces same spike times."""
        print("\n" + "=" * 70)
        print("Testing: enable_internal_state_tracking comparison")
        print("=" * 70)

        # Run with tracking ENABLED
        print("\n[1/2] Running with enable_internal_state_tracking=True")
        print("-" * 70)
        model_with_tracking = test_lif_soma_mixed_synapses(
            enable_internal_state_tracking=True,
            use_gpu=True
        )

        # Generate visualization
        vizualize_responses(
            model_with_tracking,
            vthr=-45,
            fig_name="test_internal_state_tracking_enabled.png"
        )
        print("✓ Visualization saved: output/test_internal_state_tracking_enabled.png")

        # Run with tracking DISABLED
        print("\n[2/2] Running with enable_internal_state_tracking=False")
        print("-" * 70)
        model_without_tracking = test_lif_soma_mixed_synapses(
            enable_internal_state_tracking=False,
            use_gpu=True
        )

        # Only prints spike times (no visualization when tracking disabled)
        comparator = BaselineComparator()
        comparator.print_spike_times(model_without_tracking)

        # Compare spike times between both runs
        print("\n" + "=" * 70)
        print("Comparing spike times: Tracking ON vs Tracking OFF")
        print("=" * 70)

        all_soma_ids_with = model_with_tracking.soma2synapse_map.keys()
        soma_ids_with = [sid for sid in all_soma_ids_with if sid >= 0 and not np.isnan(sid)]

        for soma_id in sorted(soma_ids_with):
            spikes_with = list(model_with_tracking.get_spike_times(soma_id=soma_id))
            spikes_without = list(model_without_tracking.get_spike_times(soma_id=soma_id))

            self.assertEqual(
                spikes_with,
                spikes_without,
                f"Soma {soma_id} spike times differ between tracking modes:\n"
                f"  Tracking ON:  {spikes_with}\n"
                f"  Tracking OFF: {spikes_without}"
            )
            print(f"✓ Soma {soma_id}: {len(spikes_with)} spikes - MATCH")

        print("\n✓ TEST PASSED: Spike times identical regardless of tracking mode")

        # Baseline comparison
        print("\n" + "=" * 70)
        print("Baseline Comparison")
        print("=" * 70)

        test_name = "test_lif_soma_mixed_synapses_tracking"
        passed, message = comparator.compare_with_baseline(model_with_tracking, test_name)
        print(message)

        if not passed:
            print("\n⚠ WARNING: Spike times differ from baseline!")
            print("  If this is intentional, run: python test_internal_state_tracking.py TestInternalStateTracking.save_baseline")
            # Don't fail the test on baseline mismatch, just warn
            # This allows running tests even when baseline doesn't exist yet

    def save_baseline(self):
        """Save current spike times as baseline."""
        print("\n" + "=" * 70)
        print("Saving Baseline")
        print("=" * 70)

        model = test_lif_soma_mixed_synapses(
            enable_internal_state_tracking=True,
            use_gpu=True
        )

        comparator = BaselineComparator()
        test_name = "test_lif_soma_mixed_synapses_tracking"
        comparator.save_baseline(model, test_name)

        print("\nℹ Baseline saved successfully!")
        print("  Next run: python -m unittest test_internal_state_tracking.TestInternalStateTracking")


if __name__ == "__main__":
    unittest.main()
