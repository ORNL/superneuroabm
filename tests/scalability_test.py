"""
 Tests to check the scalability of different SNN models as a function of 
 # neurons and synaptic density in the network.
"""

import unittest
import numpy as np
import time

from superneuro.neuro.model import NeuromorphicModel
from superneuro.neuro.neuron import Neuron


class ScaleTest(unittest.TestCase):
    """
    This class contains stress tests for SuperNeuro by
    building and simulating large networks, both sparse and dense.

    """

    def test_create_random_snn(
        self, n_neurons: int = 1000, n_synapses: int = 1000, ticks: int = 100
    ):
        np.random.seed(10)
        t_init_start = time.time()
        self._model = NeuromorphicModel()
        t_init_end = time.time()
        print(
            f"Time to initialize the model: {t_init_end - t_init_start} seconds."
        )
        t_create_start = time.time()
        for _ in range(n_neurons):
            # set all neurons to monitor as well
            _ = self._model.create_neuron(monitor=True)

        for _ in range(n_synapses):
            source = np.random.randint(0, n_neurons - 1)
            dest = np.random.randint(0, n_neurons - 1)
            # Avoid self connections
            if source == dest:
                continue
            wt_val = np.random.randint(1, 5)
            _ = self._model.create_synapse(
                self._model.get_agent(source),
                self._model.get_agent(dest),
                weight=wt_val,
            )

        self._model.setup()
        t_create_end = time.time()
        print(
            f"Time to create the random network: {t_create_end - t_create_start} seconds"
        )
        # Choose a single random neuron to stimulate:
        input_id = np.random.randint(0, n_neurons - 1)
        print("Input applied to neuron:", input_id)
        self._model.get_agent(input_id).spike(1, 4)

        t_sim_start = time.time()
        # Simulate for 20 steps:
        self._model.simulate(ticks=ticks)
        t_sim_end = time.time()
        print(
            f"Time to simulate the network: {t_sim_end - t_sim_start} seconds"
        )

        t_mon_start = time.time()
        # Get spike times
        neurons = self._model.get_agents_with(lambda a: type(a) == Neuron)
        for neuron in neurons:
            _ = neuron.get_spike_times()
        t_mon_end = time.time()
        print(
            f"Time to get spikes for all neurons: {t_mon_end - t_mon_start} seconds"
        )
