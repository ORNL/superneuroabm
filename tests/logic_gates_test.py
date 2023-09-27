import unittest

from superneuroabm.model import NeuromorphicModel


class LogicGatesTest(unittest.TestCase):
    """
    Tests SNN semantics by ensuring that basic logic
    gate functionality can be replicated.

    """

    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        # Create NeuromorphicModel
        self._model = NeuromorphicModel()
        self._use_cuda = False

    def test_two_neurons(self):
        """Tests working of two neurons"""

        # Create neuron
        neuron_0 = self._model.create_neuron(threshold=0.0)
        neuron_1 = self._model.create_neuron(threshold=0.0)

        # Create synapse
        self._model.create_synapse(
            pre_neuron_id=neuron_0, post_neuron_id=neuron_1
        )

        # Setup and simulate
        self._model.setup(output_buffer_len=10, use_cuda=self._use_cuda)

        # Add spikes
        spikes = [(1, 1), (2, 1)]
        for spike in spikes:
            self._model.add_spike(
                neuron_id=neuron_0, tick=spike[0], value=spike[1]
            )

        spikes = [(9, 1), (5, 1), (4, 1)]
        for spike in spikes:
            self._model.add_spike(
                neuron_id=neuron_1, tick=spike[0], value=spike[1]
            )

        self._model.simulate(ticks=10)

        expected_times = [1, 2]
        print(self._model.get_spike_times(neuron_id=neuron_0))
        print(self._model.get_spike_times(neuron_id=neuron_1))
        assert (
            self._model.get_spike_times(neuron_id=neuron_0) == expected_times
        ), f"Spike times are {self._model.get_spike_times(neuron_id=neuron_0)} but should be {expected_times}"
        expected_times = [2, 3, 4, 5, 9]
        assert (
            self._model.get_spike_times(neuron_id=neuron_1) == expected_times
        ), f"Spike times are {self._model.get_spike_times(neuron_id=neuron_1)} but should be {expected_times}"

    def test_or_gate(self):
        """Builds a neuromorphic circuit for OR gate and simulates it"""

        # Create neurons
        input_0 = self._model.create_neuron(threshold=0.0)
        input_1 = self._model.create_neuron(threshold=0.0)
        output_2 = self._model.create_neuron(threshold=0.0)

        # Create synapses
        self._model.create_synapse(
            pre_neuron_id=input_0, post_neuron_id=output_2, weight=1.0
        )
        self._model.create_synapse(
            pre_neuron_id=input_1, post_neuron_id=output_2, weight=1.0
        )

        # Setup and simulate
        self._model.setup(output_buffer_len=10, use_cuda=self._use_cuda)

        # test_cases in format time -> ([(Neuron, value), (Neuron, value)]
        test_cases = {
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
        # Overall expected spike times for output_2 neuron: [4, 6, 8]
        expected_times = [4, 6, 8]
        for time in test_cases:
            for neuron, value in test_cases[time]:
                self._model.add_spike(neuron_id=neuron, tick=time, value=value)

        self._model.simulate(ticks=10)

        print(self._model.get_spike_times(neuron_id=output_2))
        assert (
            self._model.get_spike_times(neuron_id=output_2) == expected_times
        ), f"Spike times are {self._model.get_spike_times(neuron_id=output_2)} but should be {expected_times}"

    def test_and_gate(self):
        """Builds a neuromorphic circuit for AND gate and simulates it"""

        # Create neurons
        input_0 = self._model.create_neuron(threshold=0.0)
        input_1 = self._model.create_neuron(threshold=0.0)
        output_2 = self._model.create_neuron(threshold=2.0)

        # Create synapses
        self._model.create_synapse(
            pre_neuron_id=input_0, post_neuron_id=output_2, weight=1.0
        )
        self._model.create_synapse(
            pre_neuron_id=input_1, post_neuron_id=output_2, weight=1.0
        )

        # Setup and simulate
        self._model.setup(output_buffer_len=10, use_cuda=self._use_cuda)

        # test_cases in format time -> ([(Neuron, value), (Neuron, value)]
        test_cases = {
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
        # Overall expected spike times for output_2 neuron: [8]
        expected_times = [8]
        for time in test_cases:
            for neuron, value in test_cases[time]:
                self._model.add_spike(neuron_id=neuron, tick=time, value=value)

        self._model.simulate(ticks=10)
        print(self._model.get_spike_times(neuron_id=output_2))
        assert (
            self._model.get_spike_times(neuron_id=output_2) == expected_times
        ), f"Spike times are {self._model.get_spike_times(neuron_id=output_2)} but should be {expected_times}"

        print(self._model.summary())


class LogicGatesTestGPU(LogicGatesTest):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self._model = NeuromorphicModel()
        self._use_cuda = True


if __name__ == "__main__":
    unittest.main()
