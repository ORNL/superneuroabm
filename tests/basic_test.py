import unittest

from superneuroabm.model import NeuromorphicModel
from superneuroabm.neuron import (
    neuron_step_func,
    synapse_step_func,
    synapse_with_stdp_step_func,
)


class BasicTest(unittest.TestCase):
    """
    Tests Spike Time Dependent Plasticity learning

    """

    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)

    def test_consecutive_simulate_calls(self):
        """Tests simulate calls"""

        model = NeuromorphicModel(use_cuda=True)
        # Create neurons
        input = model.create_neuron(threshold=0.0)
        output = model.create_neuron(threshold=0.0)

        # Create synapses
        model.create_synapse(
            pre_neuron_id=input, post_neuron_id=output, weight=1.0
        )

        spikes = {1: [(input, 1)], 6: [(input, 100)]}
        for time in spikes:
            for neuron, value in spikes[time]:
                model.spike(neuron_id=neuron, tick=time, value=value)

        # Setup and simulate
        model.setup(output_buffer_len=10)
        model.simulate(ticks=6)
        print(model.summary())
        assert model._global_data_vector[0] == 6, "Ticks are off"
        assert model.get_spikes(output) == [
            2
        ], "Simulation did not continue properly"

        print(model._agent_factory._property_name_2_agent_data_tensor)
        model.simulate(ticks=4)
        assert (
            model._global_data_vector[0] == 10
        ), "Ticks didn't continue correctly on second simulate call"
        assert model.get_spikes(output) == [
            2,
            7,
        ], "Simulation did not continue properly"
        print(model._agent_factory._property_name_2_agent_data_tensor)
        print(model.summary())

    def test_consecutive_setup_and_simulate_calls(self):
        """Tests setup and simulate calls"""

        model = NeuromorphicModel(use_cuda=True)
        # Create neurons
        input = model.create_neuron(threshold=0.0)
        output = model.create_neuron(threshold=0.0)

        # Create synapses
        model.create_synapse(
            pre_neuron_id=input, post_neuron_id=output, weight=1.0
        )

        spikes = {1: [(input, 1)], 6: [(input, 100)]}
        for time in spikes:
            for neuron, value in spikes[time]:
                model.spike(neuron_id=neuron, tick=time, value=value)

        # Setup and simulate
        model.setup(output_buffer_len=10)
        model.simulate(ticks=6)
        print(model.summary())
        assert model._global_data_vector[0] == 6, "Ticks are off"
        assert model.get_spikes(output) == [
            2
        ], "Simulation did not execute properly"

        # model.setup(output_buffer_len=10)
        print(model._global_data_vector)
        print(model._agent_factory._property_name_2_agent_data_tensor)

        model.setup(output_buffer_len=10)
        model.simulate(ticks=4)
        assert (
            model._global_data_vector[0] == 4
        ), "Ticks didn't continue correctly on second simulate call"
        assert model.get_spikes(output) == [
            2,
        ], "Simulation did not reset properly"
        print(model._global_data_vector)
        print(model._agent_factory._property_name_2_agent_data_tensor)
        print(model.summary())

    def test_breeds(self):
        """Tests multi-breed"""

        neuron_breed_info = {
            "Neuron": [neuron_step_func, synapse_step_func],
            "NeuronSTDPForward": [
                neuron_step_func,
                synapse_with_stdp_step_func,
            ],
        }
        model = NeuromorphicModel(
            use_cuda=True, neuron_breed_info=neuron_breed_info
        )
        # Create neurons
        input_0 = model.create_neuron(breed="NeuronSTDPForward", threshold=0.0)
        input_1 = model.create_neuron(breed="Neuron", threshold=0.0)
        output_2 = model.create_neuron(threshold=2.0)

        # Create synapses
        model.create_synapse(
            pre_neuron_id=input_0, post_neuron_id=output_2, weight=1.0
        )
        model.create_synapse(
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
                model.spike(neuron_id=neuron, tick=time, value=value)

        # Setup and simulate
        model.setup(output_buffer_len=10)
        model.simulate(ticks=10)
        print(model.summary())
        assert (
            round(model.get_synapse_weight(0, 2)) != 1
        ), "Breed had no effect"
        assert model.get_synapse_weight(1, 2) == 1, "Breed had no effect"
