import unittest

import numpy as np
import csv
import yaml
import inspect
from pathlib import Path


from matplotlib import pyplot as plt

from superneuroabm.model import NeuromorphicModel
from tests.util import vizualize_responses

CURRENT_DIR = Path(__file__).resolve().parent
COMPONENT_CONFIG_FPATH = CURRENT_DIR / "component_base_hyperparameter_config.yaml"


class TestSynapses(unittest.TestCase):
    """
    Tests SNN (Spiking Neural Network) semantics by ensuring that basic logic
    gate functionality can be replicated using LIF (Leaky Integrate-and-Fire) neurons.

    This test suite validates the core functionality of neuromorphic simulations
    by creating simple neural circuits and verifying their spike behavior.
    """

    def __init__(self, methodName: str = ...) -> None:
        """
        Initialize the test case with a NeuromorphicModel instance.

        Args:
            methodName: Name of the test method to run
        """
        super().__init__(methodName)
        # Create NeuromorphicModel instance for testing
        self._model = NeuromorphicModel()
        # Set global simulation parameters
        self._model.register_global_property("dt", 1e-1)  # Time step (100 μs)
        self._model.register_global_property("I_bias", 0)  # No bias current
        # Set to use CPU for base test (GPU variant in separate class)
        self._use_gpu = False

        with open(COMPONENT_CONFIG_FPATH, "r") as f:
            self._component_configurations = yaml.safe_load(f)

        # Define input spike patterns for synapses
        self._spike_times = [
            # Synapse 0 receives early, strong spikes
            [(2, 1), (10, 1), (20, 1)],  # (time_tick, spike_value)
            # Synapse 1 receives delayed, weaker spikes that overlap with Synapse 0
            [(5, 1), (12, 1), (25, 1)],  # (time_tick, spike_value)
        ]

        # Define simulation duration
        self._simulation_duration = 100  # Total simulation time in ticks
        self._sync_every_n_ticks = 100  # Synchronization interval for updates

    def test_exp_synapse(self) -> None:
        """
        Tests the basic functionality of two connected LIF neurons (somas).

        This test creates a simple two-neuron circuit:
        1. External input -> Soma 0 (via synapse)
        2. Soma 0 -> Soma 1 (via internal synapse)

        The test verifies that input spikes propagate through the network
        and generate expected output spikes.
        """

        soma_breed = "lif_soma"
        soma_parameters = self._component_configurations["soma"][soma_breed][
            "hyperparameters"
        ]
        soma_parameters = [float(val) for val in soma_parameters.values()]
        soma_internal_state = self._component_configurations["soma"][soma_breed][
            "default_internal_state"
        ]
        soma_internal_state = [float(val) for val in soma_internal_state.values()]

        # Create first LIF soma (receives external input)
        soma_0 = self._model.create_soma(
            breed=soma_breed,
            parameters=soma_parameters,
            default_internal_state=soma_internal_state,
        )

        # Package parameters for soma creation
        synapse_breed = "single_exp_synapse"
        synapse_parameters = self._component_configurations["synapse"][synapse_breed][
            "hyperparameters"
        ]
        synapse_parameters = [float(val) for val in synapse_parameters.values()]
        synapse_internal_state = self._component_configurations["synapse"][
            synapse_breed
        ]["default_internal_state"]
        synapse_internal_state = [float(val) for val in synapse_internal_state.values()]
        print(
            f"Soma parameters: {soma_parameters}"
            f"\nSynapse parameters: {synapse_parameters}"
        )

        # Create first LIF neuron (receives external input)
        synapse_0 = self._model.create_synapse(
            breed=synapse_breed,
            pre_soma_id=np.nan,  # External input
            post_soma_id=soma_0,  # Not connected to a soma yet
            parameters=synapse_parameters,
            default_internal_state=synapse_internal_state,
        )

        # Initialize the simulation environment
        self._model.setup(use_gpu=self._use_gpu)

        # Inject spikes into the external synapse
        for spike in self._spike_times[0]:
            self._model.add_spike(synapse_id=synapse_0, tick=spike[0], value=spike[1])

        # Run simulation
        self._model.simulate(
            ticks=self._simulation_duration, update_data_ticks=self._sync_every_n_ticks
        )

        # Verify results: expect at least 2 spikes from soma_0
        minimum_expected_spikes = 2

        # Extract the synaptic current history for soma_0:
        internal_states_history_syn0 = np.array(
            self._model.get_internal_states_history(agent_id=synapse_0)
        )

        print(f"Internal states history from synapse 0: {internal_states_history_syn0}")

        # Generate visualization of membrane potential over time
        vizualize_responses(self._model, vthr=0)

        # Save detailed simulation data to CSV for further analysis
        '''with open("output_LIF.csv", "w", newline="") as file:
            writer = csv.writer(file)
            # Write header
            writer.writerow(
                [
                    "Membrane_Potential_mV",
                    "Time_Count",
                    "Last_Spike_Time",
                    "Synapse_Current",
                ]
            )
            # Write combined data
            for i in range(len(internal_states_history_soma0)):
                row = list(internal_states_history_soma0[i]) + [
                    internal_states_history_syn0[i][0]
                ]
                writer.writerow(row)

        # Assert that the neuron generated expected number of spikes
        actual_spikes = len(self._model.get_spike_times(soma_id=soma_0))
        assert (
            actual_spikes >= minimum_expected_spikes
        ), f"Total number of spikes are {actual_spikes} but should be at least {minimum_expected_spikes}"'''


if __name__ == "__main__":
    # Run all test cases when script is executed directly
    unittest.main()
