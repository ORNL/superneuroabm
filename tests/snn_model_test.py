"""
 Test creation of a basic SNN model instance and several neuron and synapse agents.
"""

import unittest
from superneuro.neuro.model import NeuromorphicModel

from superneuro.neuro.neuron import Neuron
from superneuro.neuro.synapse import Synapse


class SNNTest(unittest.TestCase):
    def test_init_snn_agents(self):
        """
        Simple Synapse and Neuron init test.

        """
        synapse = Synapse()
        _ = Neuron(
            threshold=1,
            reset_state=0,
            leak=0,
            refactory_period=0,
            axonal_delay=1,
            in_synapses=[],
            out_synapses=[synapse],
        )
        _ = Neuron()

    def test_init_NeuromorphicModel(self):
        """
        Inits SNN with neuron 0 connected to synapse 2
            connected to neuron 1

        """
        self._model = NeuromorphicModel()

        self._neurons = []
        self._neurons.append(self._model.create_neuron())
        self._neurons.append(self._model.create_neuron())

        first_syn = self._model.create_synapse(
            self._neurons[0],
            self._neurons[1],
            weight=2,
            delay=1,
        )
        _ = self._model.get_synapse(first_syn._label)

        self._neurons.append(self._model.create_neuron())
        syn = self._model.create_synapse(
            self._neurons[1],
            self._neurons[2],
            weight=4,
            delay=2,
        )
        _ = self._model.get_synapse(syn._label)

        self._model.setup()

    def test_simulate_NeuromorphicModel(self):
        """
        Test simulation of initialized model with a single spike

        """
        self.test_init_NeuromorphicModel()
        self._neurons[0].spike(3, 2)
        self._model.simulate(ticks=10)

    def test_snn_monitor(self):
        """
        Test monitoring functionality and test whether
            spike propagated correctly.

        """
        self.test_init_NeuromorphicModel()

        # Monitor all neurons
        for neuron in self._neurons:
            neuron.monitor = True

        self._neurons[0].spike(3, 2)
        self._model.simulate(ticks=10)

        # Check output spike times
        expected_spike_times = [[3], [4], [6]]
        for idx, neuron in enumerate(self._neurons):
            spike_times = neuron.get_spike_times()
            assert spike_times == expected_spike_times[idx], (
                f"Neuron {idx} spiked at {spike_times} but was "
                f"expected to spike at {expected_spike_times[idx]}"
            )

    def test_model_summary(self):
        self.test_init_NeuromorphicModel()
        self._model.summary()

    def test_reset(self):
        self.test_init_NeuromorphicModel()
        self._model.network_reset()


if __name__ == "__main__":
    unittest.main()
