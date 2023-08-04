import unittest

from superneuroabm.model import NeuromorphicModel
from superneuroabm.neuron import (
    neuron_step_func,
    synapse_step_func,
    synapse_with_stdp_step_func,
)


class STDPTest(unittest.TestCase):
    """
    Tests Spike Time Dependent Plasticity learning

    """

    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        # Create NeuromorphicModel
        neuron_breed_info = {
            "Neuron": [neuron_step_func, synapse_step_func],
            "NeuronSTDPForward": [
                neuron_step_func,
                synapse_with_stdp_step_func,
            ],
        }
        self._model = NeuromorphicModel(
            use_cuda=True, neuron_breed_info=neuron_breed_info
        )

    def test_stdp(self):
        """Tests STDP"""

        # Create neurons
        input_0 = self._model.create_neuron(
            breed="NeuronSTDPForward", threshold=0.0
        )
        input_1 = self._model.create_neuron(breed="Neuron", threshold=0.0)
        output_2 = self._model.create_neuron(threshold=2.0)

        # Create synapses
        self._model.create_synapse(
            pre_neuron_id=input_0,
            post_neuron_id=output_2,
            weight=1.0,
            synapse_learning_params=[3, 0.6, 0.3, 8, 5, 0.8, 1000],
            # stdp_timesteps, A_pos, A_neg, tau_pos, tau_neg, sigma, w_max
        )

        self._model.create_synapse(
            pre_neuron_id=input_1, post_neuron_id=output_2, weight=1.0
        )

        spikes = {
            1: [
                (input_0, 0),
                (input_1, 0),
            ],  # Input: (0, 0); Expected output: No spike
            3: [
                (input_0, 0),
                (input_1, 1),
            ],  # Input: (0, 1); Expected output: Spike at time 4
            5: [
                (input_0, 1),
                (input_1, 0),
            ],  # Input: (1, 0); Expected output: Spike at time 6
            7: [
                (input_0, 1),
                (input_1, 1),
            ],  # Input: (1, 1); Expected output: Spike at time 8
        }
        expected_times = [8]
        for time in spikes:
            for neuron, value in spikes[time]:
                self._model.spike(neuron_id=neuron, tick=time, value=value)

        # Setup and simulate
        self._model.setup(output_buffer_len=10)
        self._model.simulate(ticks=10)
        print(self._model.summary())
        assert (
            round(self._model.get_synapse_weight(0, 2)) == 499
        ), "STDP weight update failed"
        assert (
            self._model.get_synapse_weight(1, 2) == 1
        ), "STDP weight update failed"
