import unittest

from superneuro.neuro.model import NeuromorphicModel


class LogicGatesTest(unittest.TestCase):
    """
    Tests SNN semantics by ensuring that basic logic
    gate functionality can be replicated.

    """

    def test_one_neuron(self):
        """Tests the working of a single neuron"""

        # Create NeuromorphicModel
        model = NeuromorphicModel()

        # Create neuron
        neuron_0 = model.create_neuron(threshold=0.0, monitor=True)

        # Add spikes
        neuron_0.spike(1, 1)
        neuron_0.spike(2, 2)
        neuron_0.spike(3, 0)

        # Setup and simulate
        model.setup()
        model.simulate(ticks=10)

        expected_times = [1, 2]
        assert (
            neuron_0.get_spike_times() == expected_times
        ), f"Spike times are {neuron_0.get_spike_times()} but should be {expected_times}"

    def test_two_neurons(self):
        """Tests working of two neurons"""

        # Create NeuromorphicModel
        model = NeuromorphicModel()

        # Create neuron
        neuron_0 = model.create_neuron(threshold=0.0, monitor=True)
        neuron_1 = model.create_neuron(threshold=0.0)
        neuron_1.monitor = True

        # Create synapse
        _ = model.create_synapse(neuron_0, neuron_1)

        # Add spikes
        neuron_0.spike(1, 1)
        neuron_0.spike(2, 1)
        neuron_1.spike(9, 1)
        neuron_1.spike(5, 1)
        neuron_1.spike(4, 1)

        # Setup and simulate
        model.setup()
        model.simulate(ticks=10)

        expected_times = [1, 2]
        assert (
            neuron_0.get_spike_times() == expected_times
        ), f"Spike times are {neuron_0.get_spike_times()} but should be {expected_times}"
        expected_times = [2, 3, 4, 5, 9]
        assert (
            neuron_1.get_spike_times() == expected_times
        ), f"Spike times are {neuron_1.get_spike_times()} but should be {expected_times}"

    def test_or_gate(self):
        """Builds a neuromorphic circuit for OR gate and simulates it"""

        # Create NeuromorphicModel
        model = NeuromorphicModel()

        # Create neurons
        input_0 = model.create_neuron(threshold=0.0)
        input_1 = model.create_neuron(threshold=0.0)
        output_2 = model.create_neuron(threshold=0.0, monitor=True)

        # Create synapses
        _ = model.create_synapse(input_0, output_2, weight=1.0)
        _ = model.create_synapse(input_1, output_2, weight=1.0)

        # Input: (0, 0); Expected output: No spike
        input_0.spike(1, 0)
        input_1.spike(1, 0)

        # Input: (0, 1); Expected output: Spike at time 4
        input_0.spike(3, 0)
        input_1.spike(3, 1)

        # Input: (1, 0); Expected output: Spike at time 6
        input_0.spike(5, 1)
        input_1.spike(5, 0)

        # Input: (1, 1); Expected output: Spike at time 8
        input_0.spike(7, 1)
        input_1.spike(7, 1)

        # Overall expected output (spike times for output_2 neuron): [4, 6, 8]

        # Setup and simulate
        model.setup()
        model.simulate(ticks=10)

        expected_times = [4, 6, 8]
        assert (
            output_2.get_spike_times() == expected_times
        ), f"Spike times are {output_2.get_spike_times()} but should be {expected_times}"

    def test_and_gate(self):
        """Builds a neuromorphic circuit for AND gate and simulates it"""

        # Create NeuromorphicModel
        model = NeuromorphicModel()

        # Create neurons
        input_0 = model.create_neuron(threshold=0.0)
        input_1 = model.create_neuron(threshold=0.0)
        output_2 = model.create_neuron(threshold=2.0, monitor=True)

        # Create synapses
        _ = model.create_synapse(input_0, output_2, weight=1.0)
        _ = model.create_synapse(input_1, output_2, weight=1.0)

        # Input: (0, 0); Expected output: No spike
        input_0.spike(1, 0)
        input_1.spike(1, 0)

        # Input: (0, 1); Expected output: No spike
        input_0.spike(3, 0)
        input_1.spike(3, 1)

        # Input: (1, 0); Expected output: No spike
        input_0.spike(5, 1)
        input_1.spike(5, 0)

        # Input: (1, 1); Expected output: Spike at time 8
        input_0.spike(7, 1)
        input_1.spike(7, 1)

        # Overall expected output (spike times for output_2 neuron): [8]

        # Setup and simulate
        model.setup()
        model.simulate(ticks=10)

        expected_times = [8]
        assert (
            output_2.get_spike_times() == expected_times
        ), f"Spike times are {output_2.get_spike_times()} but should be {expected_times}"


if __name__ == "__main__":
    unittest.main()
