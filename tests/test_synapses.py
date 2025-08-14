import unittest

import numpy as np
import csv
import yaml

from superneuroabm.model import NeuromorphicModel
from matplotlib import pyplot as plt

COMPONENT_CONFIG_FPATH = "./component_base_hyperparameter_config.yaml"


class LogicGatesTestLIF(unittest.TestCase):
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

        # Package parameters for soma creation
        synapse_breed = "Single_Exp_Synapse"
        synapse_parameters = self._component_configurations[synapse_breed][
            "hyperparameters"
        ]
        synapse_internal_state = self._component_configurations[synapse_breed][
            "default_internal_state"
        ]

        # Create first LIF neuron (receives external input)
        synapse_0 = self._model.create_synapse(
            breed=synapse_breed,
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
        plt.figure(figsize=(5, 5))
        plt.plot(
            internal_states_history_soma0[:, 0],
            label="Membrane Potential of Soma 0",
        )
        plt.ylabel("Mem. Pot. (mV)")
        plt.xlabel("Time (ms)")
        plt.legend()
        plt.title("LIF Neuron Response to Input Spikes")
        plt.savefig(
            "logic_gates_test_lif_test_synapse.png", dpi=150, bbox_inches="tight"
        )
        plt.close()

        # Save detailed simulation data to CSV for further analysis
        with open("output_LIF.csv", "w", newline="") as file:
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
        ), f"Total number of spikes are {actual_spikes} but should be at least {minimum_expected_spikes}"


def vizualize_responses(model: NeuromorphicModel) -> None:
    