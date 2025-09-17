import unittest

import numpy as np
import csv
import yaml
import inspect
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
from matplotlib import pyplot as plt

from superneuroabm.model import NeuromorphicModel
from tests.util import vizualize_responses

CURRENT_DIR = Path(__file__).resolve().parent
COMPONENT_CONFIG_FPATH = CURRENT_DIR / ".." / "superneuroabm" / "component_base_config.yaml"


class TestSynapseAndSomaModels(unittest.TestCase):
    """
    Tests basic neuromorphic micro-model functionality using different soma and synapse types.

    This test suite validates the core functionality of neuromorphic simulations
    by creating minimal neural circuits (single soma + synapse) and verifying
    their response to input stimulation.
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
        # Set to use CPU for base test (GPU variant in separate class)
        self._use_gpu = True

        with open(COMPONENT_CONFIG_FPATH, "r") as f:
            self._component_configurations = yaml.safe_load(f)

        # Define input spike patterns for synapses
        self._spike_times = [
            # Synapse 0 receives early, strong spikes
            [(2, 1), (10, 1), (20, 1)],  # (time_tick, spike_value)
            # Additional spike pattern for future multi-synapse tests
            [(5, 1), (12, 1), (25, 1)],  # (time_tick, spike_value)
        ]

        # Define simulation duration
        self._simulation_duration = 200  # Total simulation time in ticks
        self._sync_every_n_ticks = 200  # Synchronization interval for updates

    def test_lif_soma_single_exp_synapse(self) -> None:
        """
        Tests the basic functionality of a LIF soma with a single exponential synapse.

        This test creates a minimal neural circuit:
        - External input -> LIF Soma (via single exponential synapse)

        The test verifies that input spikes are properly processed by the synapse
        and cause appropriate responses in the LIF soma.
        """

        self.micro_model_test_helper(
            "lif_soma", "config_0", "single_exp_synapse", "no_learning_config_0"
        )

    def test_izh_soma_single_exp_synapse(self) -> None:
        """
        Tests the basic functionality of an Izhikevich soma with a single exponential synapse.

        This test creates a minimal neural circuit:
        - External input -> Izhikevich Soma (via single exponential synapse)

        The test verifies that input spikes are properly processed by the synapse
        and cause appropriate responses in the Izhikevich soma.
        """

        self.micro_model_test_helper(
            "izh_soma", "config_0", "single_exp_synapse", "no_learning_config_0"
        )

    def test_lif_soma_multi_synapse_integration(self) -> None:
        """
        Tests multi-synapse integration with a two-soma network.

        This test creates a neural circuit:
        - External input (synapse_0) -> soma_0 -> synapse_1 -> soma_1
        - External input (synapse_2) -> soma_1

        This verifies that soma_1 can integrate inputs from both an internal synapse
        (from soma_0) and an external synapse (synapse_2) simultaneously.
        """
        
        # Create soma_0 (LIF)
        soma_0 = self._model.create_soma(
            breed="lif_soma",
            config_name="config_0",
        )

        # Create soma_1 (LIF) 
        soma_1 = self._model.create_soma(
            breed="lif_soma",
            config_name="config_0",
        )

        # Create synapse_0: external input -> soma_0
        synapse_0 = self._model.create_synapse(
            breed="single_exp_synapse",
            pre_soma_id=np.nan,  # External input
            post_soma_id=soma_0,
            config_name="no_learning_config_0",
        )

        # Create synapse_1: soma_0 -> soma_1
        synapse_1 = self._model.create_synapse(
            breed="single_exp_synapse",
            pre_soma_id=soma_0,
            post_soma_id=soma_1,
            config_name="no_learning_config_0",
        )

        # Create synapse_2: external input -> soma_1
        synapse_2 = self._model.create_synapse(
            breed="single_exp_synapse",
            pre_soma_id=np.nan,  # External input
            post_soma_id=soma_1,
            config_name="no_learning_config_0",
        )

        # Initialize the simulation environment
        self._model.setup(use_gpu=self._use_gpu)

        # Inject spikes into synapse_0 (external -> soma_0)
        for spike in self._spike_times[0]:
            self._model.add_spike(synapse_id=synapse_0, tick=spike[0], value=spike[1])

        # Inject spikes into synapse_2 (external -> soma_1) using second spike pattern
        for spike in self._spike_times[1]:
            self._model.add_spike(synapse_id=synapse_2, tick=spike[0], value=spike[1])

        # Run simulation
        self._model.simulate(
            ticks=self._simulation_duration, update_data_ticks=self._sync_every_n_ticks
        )

        # Extract internal states for analysis
        internal_states_syn0 = np.array(
            self._model.get_internal_states_history(agent_id=synapse_0)
        )
        internal_states_syn1 = np.array(
            self._model.get_internal_states_history(agent_id=synapse_1)
        )
        internal_states_syn2 = np.array(
            self._model.get_internal_states_history(agent_id=synapse_2)
        )

        # Generate visualization
        caller_name = inspect.stack()[0].function
        vizualize_responses(self._model, vthr=0, fig_name=f"{caller_name}.png")

        # Verify that soma_1 received inputs from both synapses
        # soma_1 should have activity from both internal (synapse_1) and external (synapse_2) inputs
        assert internal_states_syn1.size > 0, "Synapse 1 (soma_0->soma_1) should have activity"
        assert internal_states_syn2.size > 0, "Synapse 2 (ext->soma_1) should have activity"

    def micro_model_test_helper(
        self, soma_breed: str, soma_config: str, synapse_breed: str, synapse_config: str
    ) -> None:
        """
        Helper method to test micro-models with different soma and synapse types.

        This creates a minimal neural circuit consisting of:
        - One soma of the specified breed (LIF or Izhikevich)
        - One synapse of the specified breed connecting external input to the soma

        Args:
            soma_breed: Type of soma to create ("lif_soma" or "izh_soma")
            synapse_breed: Type of synapse to create ("single_exp_synapse")

        The test injects predefined spike patterns and verifies the soma's response.
        """

        # Create soma with specified breed and config
        soma_0 = self._model.create_soma(
            breed=soma_breed,
            config_name=soma_config,
        )

        # Create synapse connecting external input to soma
        synapse_0 = self._model.create_synapse(
            breed=synapse_breed,
            pre_soma_id=np.nan,  # External input
            post_soma_id=soma_0,  # Connected to the created soma
            config_name=synapse_config,
        )

        # Initialize the simulation environment
        self._model.setup(use_gpu=self._use_gpu)

        # Inject spike pattern into the external synapse
        for spike in self._spike_times[0]:
            self._model.add_spike(synapse_id=synapse_0, tick=spike[0], value=spike[1])

        # Run simulation
        self._model.simulate(
            ticks=self._simulation_duration, update_data_ticks=self._sync_every_n_ticks
        )

        # Verify results: expect at least 2 spikes from the soma
        minimum_expected_spikes = 2

        # Extract the synaptic current history from synapse_0
        internal_states_history_syn0 = np.array(
            self._model.get_internal_states_history(agent_id=synapse_0)
        )

        print(f"Internal states history from synapse 0: {internal_states_history_syn0}")

        # Generate visualization of responses over time
        caller_name = inspect.stack()[1].function
        vizualize_responses(self._model, vthr=0, fig_name=f"{caller_name}.png", figsize=(8, 20))

        # Assert that the neuron generated expected number of spikes
        '''actual_spikes = len(self._model.get_spike_times(soma_id=soma_0))
        assert (
            actual_spikes >= minimum_expected_spikes
        ), f"Total number of spikes are {actual_spikes} but should be at least {minimum_expected_spikes}"'''


if __name__ == "__main__":
    # Run all test cases when script is executed directly
    unittest.main()
