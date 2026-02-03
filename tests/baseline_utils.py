"""
Baseline utilities for saving and comparing spike times.

This module provides functions to:
1. Save spike times from a model run as a baseline
2. Compare current spike times against saved baseline
3. Report differences to detect model breaking changes
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from superneuroabm.model import NeuromorphicModel


class BaselineComparator:
    """Handles saving and comparing spike time baselines."""

    def __init__(self, baseline_dir: Path = None):
        """
        Initialize baseline comparator.

        Args:
            baseline_dir: Directory to store baseline files.
                         Defaults to tests/baselines/
        """
        if baseline_dir is None:
            self.baseline_dir = Path(__file__).resolve().parent / "baselines"
        else:
            self.baseline_dir = Path(baseline_dir)

        self.baseline_dir.mkdir(parents=True, exist_ok=True)

    def save_baseline(self, model: NeuromorphicModel, test_name: str) -> Dict:
        """
        Save spike times from model as baseline.

        Args:
            model: NeuromorphicModel instance after simulation
            test_name: Name of the test (used as filename)

        Returns:
            Dictionary of spike times saved
        """
        # Get all valid soma IDs
        all_soma_ids = model.soma2synapse_map.keys()
        soma_ids = [sid for sid in all_soma_ids if sid >= 0 and not np.isnan(sid)]

        # Collect spike times
        baseline_data = {}
        for soma_id in sorted(soma_ids):
            spike_times = model.get_spike_times(soma_id=soma_id)
            # Convert numpy arrays to lists for JSON serialization
            baseline_data[str(soma_id)] = [int(t) for t in spike_times]

        # Save to file
        baseline_file = self.baseline_dir / f"{test_name}.json"
        with open(baseline_file, 'w') as f:
            json.dump(baseline_data, f, indent=2)

        print(f"\n✓ Baseline saved: {baseline_file}")
        print(f"  Somas tracked: {len(baseline_data)}")
        total_spikes = sum(len(times) for times in baseline_data.values())
        print(f"  Total spikes: {total_spikes}")

        return baseline_data

    def load_baseline(self, test_name: str) -> Dict:
        """
        Load baseline spike times from file.

        Args:
            test_name: Name of the test

        Returns:
            Dictionary of spike times, or None if baseline doesn't exist
        """
        baseline_file = self.baseline_dir / f"{test_name}.json"

        if not baseline_file.exists():
            return None

        with open(baseline_file, 'r') as f:
            return json.load(f)

    def compare_with_baseline(self, model: NeuromorphicModel, test_name: str,
                             tolerance: float = 0) -> Tuple[bool, str]:
        """
        Compare current model spike times with saved baseline.

        Args:
            model: NeuromorphicModel instance after simulation
            test_name: Name of the test
            tolerance: Allowed difference in spike times (default: 0 for exact match)

        Returns:
            Tuple of (passed: bool, message: str)
        """
        # Load baseline
        baseline = self.load_baseline(test_name)

        if baseline is None:
            return False, f"❌ No baseline found for '{test_name}'. Run with --save-baseline first."

        # Get current spike times
        all_soma_ids = model.soma2synapse_map.keys()
        soma_ids = [sid for sid in all_soma_ids if sid >= 0 and not np.isnan(sid)]

        current_data = {}
        for soma_id in sorted(soma_ids):
            spike_times = model.get_spike_times(soma_id=soma_id)
            current_data[str(soma_id)] = [int(t) for t in spike_times]

        # Compare
        differences = []

        # Check for missing/extra somas
        baseline_somas = set(baseline.keys())
        current_somas = set(current_data.keys())

        if baseline_somas != current_somas:
            missing = baseline_somas - current_somas
            extra = current_somas - baseline_somas
            if missing:
                differences.append(f"  Missing somas: {sorted(missing)}")
            if extra:
                differences.append(f"  Extra somas: {sorted(extra)}")

        # Compare spike times for each soma
        for soma_id in baseline_somas & current_somas:
            baseline_spikes = baseline[soma_id]
            current_spikes = current_data[soma_id]

            if len(baseline_spikes) != len(current_spikes):
                differences.append(
                    f"  Soma {soma_id}: spike count mismatch "
                    f"(baseline: {len(baseline_spikes)}, current: {len(current_spikes)})"
                )
            else:
                # Check individual spike times
                for i, (b_time, c_time) in enumerate(zip(baseline_spikes, current_spikes)):
                    if abs(b_time - c_time) > tolerance:
                        differences.append(
                            f"  Soma {soma_id}, spike #{i}: time mismatch "
                            f"(baseline: {b_time}, current: {c_time})"
                        )

        # Generate report
        if differences:
            message = f"❌ FAILED: Spike times differ from baseline\n"
            message += "\n".join(differences)
            return False, message
        else:
            total_spikes = sum(len(times) for times in current_data.values())
            message = f"✓ PASSED: All spike times match baseline\n"
            message += f"  Somas checked: {len(current_data)}\n"
            message += f"  Total spikes verified: {total_spikes}"
            return True, message

    def print_spike_times(self, model: NeuromorphicModel) -> None:
        """
        Print spike times for all somas in the model.

        Args:
            model: NeuromorphicModel instance after simulation
        """
        all_soma_ids = model.soma2synapse_map.keys()
        soma_ids = [sid for sid in all_soma_ids if sid >= 0 and not np.isnan(sid)]

        print("\n=== Spike Times ===")
        for soma_id in sorted(soma_ids):
            spike_times = model.get_spike_times(soma_id=soma_id)
            print(f"Soma {soma_id} spike times: {list(spike_times)}")
        print("===================\n")
