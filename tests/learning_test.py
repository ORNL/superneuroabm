import unittest

from superneuroabm.model import NeuromorphicModel
from superneuroabm.neuron import (
    neuron_step_func,
    synapse_step_func,
    synapse_with_stdp_step_func,
)


class LearningTest(unittest.TestCase):
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
        self._model = NeuromorphicModel(neuron_breed_info=neuron_breed_info)

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
        # Setup and simulate
        self._model.setup(output_buffer_len=10, use_cuda=True)
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
                self._model.add_spike(neuron_id=neuron, tick=time, value=value)

        self._model.simulate(ticks=10)
        print(self._model.summary())
        assert (
            round(self._model.get_synapse_weight(0, 2)) == 499
        ), "STDP weight update failed"
        assert (
            self._model.get_synapse_weight(1, 2) == 1
        ), "STDP weight update failed"

    def test_learning_per_synapse(self):
        """Tests if learning can be turned on or off per synapse"""

        # Create neurons
        input_0 = self._model.create_neuron(
            breed="NeuronSTDPForward", threshold=0.0
        )
        input_1 = self._model.create_neuron(breed="Neuron", threshold=0.0)
        input_2 = self._model.create_neuron(
            breed="NeuronSTDPForward", threshold=0.0
        )
        output = self._model.create_neuron(threshold=2.0)

        # Create synapses
        self._model.create_synapse(
            pre_neuron_id=input_0,
            post_neuron_id=output,
            weight=1.0,
            synapse_learning_params=[3, 0.6, 0.3, 8, 5, 0.8, 1000],
            # stdp_timesteps, A_pos, A_neg, tau_pos, tau_neg, sigma, w_max
        )

        self._model.create_synapse(
            pre_neuron_id=input_1, post_neuron_id=output, weight=1.0
        )

        self._model.create_synapse(
            pre_neuron_id=input_2,
            post_neuron_id=output,
            weight=1.0,
            synapse_learning_params=[0, 0.6, 0.3, 8, 5, 0.8, 1000],
            # stdp_timesteps, A_pos, A_neg, tau_pos, tau_neg, sigma, w_max
        )
        # Setup and simulate
        self._model.setup(output_buffer_len=10, use_cuda=True)
        spikes = {
            1: [
                (input_0, 0),
                (input_1, 0),
                (input_2, 0),
            ],  # Input: (0, 0); Expected output: No spike
            3: [
                (input_0, 0),
                (input_1, 1),
            ],  # Input: (0, 1); Expected output: Spike at time 4
            5: [
                (input_0, 1),
                (input_1, 0),
                (input_2, 1),
            ],  # Input: (1, 0); Expected output: Spike at time 6
            7: [
                (input_0, 1),
                (input_1, 1),
                (input_2, 1),
            ],  # Input: (1, 1); Expected output: Spike at time 8
        }
        for time in spikes:
            for neuron, value in spikes[time]:
                self._model.add_spike(neuron_id=neuron, tick=time, value=value)

        self._model.simulate(ticks=10)
        print(self._model.summary())
        assert (
            round(self._model.get_synapse_weight(input_0, output)) == 499
        ), "STDP weight update failed"
        assert (
            self._model.get_synapse_weight(input_1, output) == 1
        ), "STDP weight updated despite not specified"
        assert (
            self._model.get_synapse_weight(input_2, output) == 1
        ), "STDP suppression failed"

    def test_learning_on_off_consecutively(self):
        """Tests if learning can be turned on and off consecutively
        across multiple simulation calls for single run.
        """

        # Create neurons
        input_0 = self._model.create_neuron(
            breed="NeuronSTDPForward", threshold=0.0
        )
        output = self._model.create_neuron(threshold=2.0)

        # Create synapses
        self._model.create_synapse(
            pre_neuron_id=input_0,
            post_neuron_id=output,
            weight=5.0,
            synapse_learning_params=[3, 0.6, 0.3, 8, 5, 0.8, 1000],
            # stdp_timesteps, A_pos, A_neg, tau_pos, tau_neg, sigma, w_max
        )
        # Setup and simulate
        self._model.setup(output_buffer_len=30, use_cuda=True)
        spikes = {
            7: [
                (input_0, 1),
            ],  # Expected output: Spike at time 8
            17: [
                (input_0, 1),
            ],  # Expected output: Spike at time 18
            27: [
                (input_0, 1),
            ],  # Expected output: Spike at time 28
        }
        for time in spikes:
            for neuron, value in spikes[time]:
                self._model.add_spike(neuron_id=neuron, tick=time, value=value)

        self._model.simulate(ticks=10)
        print(self._model.summary())
        assert (
            round(self._model.get_synapse_weight(input_0, output)) == 244
        ), "STDP weight update failed"
        print(self._model.get_synapse_weight(input_0, output))

        # Now update synapse to turn off STDP
        # Create synapses
        self._model.update_synapse(
            input_0,
            output,
            synapse_learning_params=[0, 0.6, 0.3, 8, 5, 0.8, 1000],
            # stdp_timesteps, A_pos, A_neg, tau_pos, tau_neg, sigma, w_max
        )
        self._model.simulate(ticks=10)
        print(self._model.summary())
        print(self._model.get_synapse_weight(input_0, output))
        assert (
            round(self._model.get_synapse_weight(input_0, output)) == 244
        ), "STDP off failed"

        # Finally, try enabling STDP for the rest of the sim again
        # Create synapses
        self._model.update_synapse(
            input_0,
            output,
            synapse_learning_params=[3, 0.6, 0.3, 8, 5, 0.8, 1000],
            # stdp_timesteps, A_pos, A_neg, tau_pos, tau_neg, sigma, w_max
        )
        self._model.simulate(ticks=10)
        print(self._model.summary())
        print(self._model.get_synapse_weight(input_0, output))
        assert (
            round(self._model.get_synapse_weight(input_0, output)) == 425
        ), "STDP off failed"

    def test_learning_on_setup_then_off(self):
        """Tests if learning can be turned on and off
        across simulation runs and if updated weights are
        retained
        """

        # Create neurons
        input = self._model.create_neuron(
            breed="NeuronSTDPForward", threshold=0.0
        )
        output_0 = self._model.create_neuron(
            breed="NeuronSTDPForward", threshold=0.0
        )
        output_1 = self._model.create_neuron(breed="Neuron", threshold=2.0)

        # Create synapses
        self._model.create_synapse(
            pre_neuron_id=input,
            post_neuron_id=output_0,
            weight=5.0,
            synapse_learning_params=[0, 0.6, 0.3, 8, 5, 0.8, 1000],
            # stdp_timesteps, A_pos, A_neg, tau_pos, tau_neg, sigma, w_max
        )
        self._model.create_synapse(
            pre_neuron_id=input,
            post_neuron_id=output_1,
            weight=5.0,
            synapse_learning_params=[3, 0.6, 0.3, 8, 5, 0.8, 1000],
            # stdp_timesteps, A_pos, A_neg, tau_pos, tau_neg, sigma, w_max
        )
        # Setup and simulate
        self._model.setup(output_buffer_len=20, use_cuda=True)
        spikes = {
            7: [
                (input, 1),
            ],  # Expected output: Spike at time 8
            17: [
                (input, 1),
            ],  # Expected output: Spike at time 18
        }
        for time in spikes:
            for neuron, value in spikes[time]:
                self._model.add_spike(neuron_id=neuron, tick=time, value=value)

        self._model.simulate(ticks=20)
        print(self._model.summary())
        assert (
            round(self._model.get_synapse_weight(input, output_0)) == 5
        ), "STDP weight update failed"
        assert (
            round(self._model.get_synapse_weight(input, output_1)) == 425
        ), "STDP weight update failed"

        # Now setup and turn off STDP
        # Create synapses
        self._model.setup(
            output_buffer_len=10, use_cuda=False, retain_weights=True
        )

        for time in spikes:
            for neuron, value in spikes[time]:
                self._model.add_spike(neuron_id=neuron, tick=time, value=value)

        self._model.update_synapse(
            input,
            output_1,
            synapse_learning_params=[0, 0.6, 0.3, 8, 5, 0.8, 1000],
            # stdp_timesteps, A_pos, A_neg, tau_pos, tau_neg, sigma, w_max
        )
        self._model.simulate(ticks=10)
        print(self._model.summary())
        assert (
            round(self._model.get_synapse_weight(input, output_0)) == 5
        ), "STDP weight update failed"
        assert (
            round(self._model.get_synapse_weight(input, output_1)) == 425
        ), "STDP weight update failed"
