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

        model = NeuromorphicModel()
        # Create neurons
        input = model.create_neuron(threshold=0.0)
        output = model.create_neuron(threshold=0.0)

        # Create synapses
        model.create_synapse(
            pre_neuron_id=input, post_neuron_id=output, weight=1.0
        )

        # Setup and simulate
        model.setup(output_buffer_len=10, use_cuda=True)
        spikes = {1: [(input, 1)], 6: [(input, 100)]}
        for time in spikes:
            for neuron, value in spikes[time]:
                model.add_spike(neuron_id=neuron, tick=time, value=value)

        model.simulate(ticks=6)
        print(model.summary())
        assert model._global_data_vector[0] == 6, "Ticks are off"
        assert model.get_spike_times(output) == [
            2
        ], "Simulation did not continue properly"

        model.simulate(ticks=4)
        print(model.summary())
        assert (
            model._global_data_vector[0] == 10
        ), "Ticks didn't continue correctly on second simulate call"
        assert model.get_spike_times(output) == [
            2,
            7,
        ], "Simulation did not continue properly"

    def test_consecutive_setup_and_simulate_calls(self):
        """Tests setup and simulate calls"""

        model = NeuromorphicModel()
        # Create neurons
        input = model.create_neuron(threshold=0.0)
        output = model.create_neuron(threshold=0.0)

        # Create synapses
        model.create_synapse(
            pre_neuron_id=input, post_neuron_id=output, weight=1.0
        )
        # Setup and simulate
        model.setup(output_buffer_len=10, use_cuda=True)
        spikes = {1: [(input, 1)], 6: [(input, 100)]}
        for time in spikes:
            for neuron, value in spikes[time]:
                model.add_spike(neuron_id=neuron, tick=time, value=value)

        model.simulate(ticks=6)
        print(model.summary())
        assert model._global_data_vector[0] == 6, "Ticks are off"
        assert model.get_spike_times(output) == [
            2
        ], "Simulation did not execute properly"

        model.setup(output_buffer_len=10, use_cuda=True)
        spikes = {1: [(input, 1)], 6: [(input, 100)]}
        for time in spikes:
            for neuron, value in spikes[time]:
                model.add_spike(neuron_id=neuron, tick=time, value=value)
        model.simulate(ticks=4)
        print(model.summary())
        assert (
            model._global_data_vector[0] == 4
        ), "Ticks didn't continue correctly on second simulate call"

        assert model.get_spike_times(output) == [
            2,
        ], "Simulation did not reset properly"

    def test_breeds(self):
        """Tests multi-breed"""

        neuron_breed_info = {
            "Neuron": [neuron_step_func, synapse_step_func],
            "NeuronSTDPForward": [
                neuron_step_func,
                synapse_with_stdp_step_func,
            ],
        }
        model = NeuromorphicModel(neuron_breed_info=neuron_breed_info)
        # Create neurons
        input_0 = model.create_neuron(breed="NeuronSTDPForward", threshold=0.0)
        input_1 = model.create_neuron(breed="Neuron", threshold=0.0)
        output_2 = model.create_neuron(breed="Neuron", threshold=2.0)

        # Create synapses
        model.create_synapse(
            pre_neuron_id=input_0,
            post_neuron_id=output_2,
            weight=1.0,
            synapse_learning_params=[3, 0.6, 0.3, 8, 5, 0.8, 1000],
        )
        model.create_synapse(
            pre_neuron_id=input_1, post_neuron_id=output_2, weight=1.0
        )

        # Setup and simulate
        model.setup(output_buffer_len=10, use_cuda=True)

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
        for time in spikes:
            for neuron, value in spikes[time]:
                model.add_spike(neuron_id=neuron, tick=time, value=value)

        model.simulate(ticks=10)
        print(model.summary())
        assert (
            round(model.get_synapse_weight(0, 2)) != 1
        ), "Breed had no effect"
        assert model.get_synapse_weight(1, 2) == 1, "Breed had no effect"

    def test_save_load(self):
        neuron_breed_info = {
            "Neuron": [neuron_step_func, synapse_step_func],
            "NeuronSTDPForward": [
                neuron_step_func,
                synapse_with_stdp_step_func,
            ],
        }
        model = NeuromorphicModel(neuron_breed_info=neuron_breed_info)
        # Create neurons
        input_0 = model.create_neuron(breed="NeuronSTDPForward", threshold=0.0)
        input_1 = model.create_neuron(breed="Neuron", threshold=0.0)
        output_2 = model.create_neuron(breed="Neuron", threshold=2.0)

        # Create synapses
        model.create_synapse(
            pre_neuron_id=input_0,
            post_neuron_id=output_2,
            weight=1.0,
            synapse_learning_params=[3, 0.6, 0.3, 8, 5, 0.8, 1000],
        )
        model.create_synapse(
            pre_neuron_id=input_1, post_neuron_id=output_2, weight=5.0
        )
        model.setup(use_cuda=True, output_buffer_len=10)

        spikes = {
            7: [
                (input_0, 1),
                (input_1, 1),
            ],  # Expected output: Spike at time 8
        }
        for time in spikes:
            for neuron, value in spikes[time]:
                model.add_spike(neuron_id=neuron, tick=time, value=value)

        # Setup and simulate
        model.simulate(ticks=10)
        print(model.summary())
        model.save("model.pickle")
        model.load("model.pickle")
        print(model.summary())
        assert (
            round(model.get_synapse_weight(input_0, output_2)) == 241
        ), "STDP enabled synapse failed"
        assert (
            model.get_synapse_weight(input_1, output_2) == 5
        ), "Weight change on STDP disabled synapse"

        model.save("model.pickle")
        model.load("model.pickle")
        model.update_synapse(
            input_0, output_2, synapse_learning_params=[0, 0, 0, 0, 0, 0, 0]
        )
        # Setup and simulate
        model.setup(use_cuda=True, output_buffer_len=10, retain_weights=True)
        spikes = {
            3: [
                (input_0, 1),
            ],  # Expected output: Spike at time 8
        }
        for time in spikes:
            for neuron, value in spikes[time]:
                model.add_spike(neuron_id=neuron, tick=time, value=value)

        model.simulate(ticks=10)
        print(model.summary())
        assert model.get_spike_times(output_2) == [
            4
        ], "Trained and loaded model did not spike correctly"

    def test_leak_to_reset(self):
        """Tests if leak does not cause neuron internal state to drop beyond reset state"""

        model = NeuromorphicModel()
        # Create neurons
        input = model.create_neuron(threshold=0.0)
        output1_reset_state = -1
        output1 = model.create_neuron(
            threshold=0.0, leak=1, reset_state=output1_reset_state
        )
        output2 = model.create_neuron(threshold=0.0, leak=1)

        # Create synapses
        model.create_synapse(
            pre_neuron_id=input, post_neuron_id=output1, weight=10.0
        )
        model.create_synapse(
            pre_neuron_id=input, post_neuron_id=output2, weight=10.0
        )
        # Setup and simulate
        model.setup(output_buffer_len=10, use_cuda=True)
        spikes = {1: [(input, 1)]}
        for time in spikes:
            for neuron, value in spikes[time]:
                model.add_spike(neuron_id=neuron, tick=time, value=value)

        model.simulate(ticks=10)

        assert (
            model.get_agent_property_value(output1, "internal_state")
            == output1_reset_state
        )
        assert model.get_agent_property_value(
            output2, "internal_state"
        ) == model.get_agent_property_value(output2, "reset_state")

    def test_refactory_period(self):
        """Tests if refactory periods greater than 1 work"""

        model = NeuromorphicModel()
        # Create neurons
        input = model.create_neuron(threshold=0.0)
        output1 = model.create_neuron(threshold=0.0)
        output2 = model.create_neuron(threshold=0.0, refractory_period=5)

        # Create synapses
        model.create_synapse(
            pre_neuron_id=input, post_neuron_id=output1, weight=10.0
        )
        model.create_synapse(
            pre_neuron_id=input, post_neuron_id=output2, weight=10.0
        )
        # Setup and simulate
        model.setup(output_buffer_len=10, use_cuda=True)
        spikes = {1: [(input, 1)], 3: [(input, 1)]}
        for time in spikes:
            for neuron, value in spikes[time]:
                model.add_spike(neuron_id=neuron, tick=time, value=value)

        model.simulate(ticks=10)

        assert model.get_spike_times(output1) == [2, 4]
        assert model.get_spike_times(output2) == [2]
