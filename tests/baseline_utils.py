#!/usr/bin/env python
"""
Baseline comparison utilities for SuperNeuroABM tests.

This module provides utilities for saving and comparing spike times against
baseline files to detect breaking changes in model behavior.
"""

import json
from pathlib import Path
from typing import Tuple, Dict, List


class BaselineComparator:
    """
    Utility class for comparing test results against saved baselines.

    Baselines are saved as JSON files in tests/baselines/ directory.
    Each baseline contains spike times for all somas in the model.
    """

    def __init__(self, baseline_dir: str = None):
        """
        Initialize baseline comparator.

        Args:
            baseline_dir: Directory to store baseline files.
                         Defaults to tests/baselines/
        """
        if baseline_dir is None:
            self.baseline_dir = Path(__file__).parent / "baselines"
        else:
            self.baseline_dir = Path(baseline_dir)

        # Create baseline directory if it doesn't exist
        self.baseline_dir.mkdir(exist_ok=True)

    def save_baseline(self, model, test_name: str) -> None:
        """
        Save spike times from model as baseline.

        Args:
            model: NeuromorphicModel instance
            test_name: Name of the test (used as filename)
        """
        baseline_file = self.baseline_dir / f"{test_name}.json"

        # Extract spike times for all somas
        spike_data = {}

        # Get all soma IDs from model
        all_soma_ids = sorted([
            sid for sid in model.soma2synapse_map.keys()
            if sid >= 0  # Filter out external inputs (-1)
        ])

        for soma_id in all_soma_ids:
            spike_times = list(model.get_spike_times(soma_id=soma_id))
            spike_data[str(soma_id)] = spike_times

        # Save to JSON
        with open(baseline_file, 'w') as f:
            json.dump(spike_data, f, indent=2)

        print(f"✓ Baseline saved: {baseline_file}")
        print(f"  Somas: {len(spike_data)}")
        total_spikes = sum(len(spikes) for spikes in spike_data.values())
        print(f"  Total spikes: {total_spikes}")

    def compare_with_baseline(self, model, test_name: str) -> Tuple[bool, str]:
        """
        Compare model spike times against saved baseline.

        Args:
            model: NeuromorphicModel instance
            test_name: Name of the test (used to find baseline file)

        Returns:
            Tuple of (passed: bool, message: str)
        """
        baseline_file = self.baseline_dir / f"{test_name}.json"

        # Check if baseline exists
        if not baseline_file.exists():
            return False, f"❌ Baseline not found: {baseline_file}\n" \
                         f"   Run save_all_baselines() to create it."

        # Load baseline
        with open(baseline_file, 'r') as f:
            baseline_data = json.load(f)

        # Extract current spike times
        all_soma_ids = sorted([
            sid for sid in model.soma2synapse_map.keys()
            if sid >= 0
        ])

        current_data = {}
        for soma_id in all_soma_ids:
            spike_times = list(model.get_spike_times(soma_id=soma_id))
            current_data[str(soma_id)] = spike_times

        # Compare
        passed = True
        differences = []

        # Check for missing/extra somas
        baseline_somas = set(baseline_data.keys())
        current_somas = set(current_data.keys())

        if baseline_somas != current_somas:
            passed = False
            missing = baseline_somas - current_somas
            extra = current_somas - baseline_somas
            if missing:
                differences.append(f"  Missing somas: {missing}")
            if extra:
                differences.append(f"  Extra somas: {extra}")

        # Compare spike times for common somas
        for soma_id in baseline_somas & current_somas:
            baseline_spikes = baseline_data[soma_id]
            current_spikes = current_data[soma_id]

            if baseline_spikes != current_spikes:
                passed = False
                differences.append(
                    f"  Soma {soma_id}:\n"
                    f"    Baseline: {baseline_spikes}\n"
                    f"    Current:  {current_spikes}"
                )

        # Build message
        if passed:
            total_spikes = sum(len(spikes) for spikes in current_data.values())
            message = f"✓ PASSED: Spike times match baseline\n" \
                     f"  Baseline: {baseline_file}\n" \
                     f"  Somas: {len(current_data)}\n" \
                     f"  Total spikes: {total_spikes}"
        else:
            message = f"❌ FAILED: Spike times differ from baseline\n" \
                     f"  Baseline: {baseline_file}\n" \
                     f"  Differences:\n" + "\n".join(differences)

        return passed, message

    def print_spike_times(self, model) -> None:
        """
        Print spike times for all somas in the model.

        Args:
            model: NeuromorphicModel instance
        """
        print("\n" + "=" * 70)
        print("SPIKE TIMES")
        print("=" * 70)

        all_soma_ids = sorted([
            sid for sid in model.soma2synapse_map.keys()
            if sid >= 0
        ])

        for soma_id in all_soma_ids:
            spike_times = list(model.get_spike_times(soma_id=soma_id))
            print(f"Soma {soma_id}: {spike_times}")

        print("=" * 70 + "\n")
