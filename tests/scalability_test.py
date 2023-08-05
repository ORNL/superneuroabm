"""
 Tests to check the scalability of different SNN models as a function of 
 # neurons and synaptic density in the network.
"""

import unittest
import numpy as np
import time
from random import random, randint
from tqdm import tqdm

from superneuroabm.model import NeuromorphicModel


class ScaleTest(unittest.TestCase):
    """
    This class contains stress tests for SuperNeuro by
    building and simulating large networks, both sparse and dense.

    """

    def test_create_random_snn(
        self,
        n_neurons: int = 100,
        connection_prob: float = 0.5,
        ticks: int = 1000,
    ):
        np.random.seed(10)
        t_init_start = time.time()
        model = NeuromorphicModel()
        t_init_end = time.time()
        print(
            f"Time to initialize the model: {t_init_end - t_init_start} seconds."
        )
        t_create_start = time.time()
        for _ in tqdm(range(n_neurons)):
            # set all neurons to monitor as well
            model.create_neuron()

        for n_i in tqdm(range(n_neurons)):
            for n_j in range(n_neurons):
                if random() < connection_prob:
                    wt_val = np.random.randint(1, 5)
                    model.create_synapse(
                        pre_neuron_id=n_i, post_neuron_id=n_j, weight=wt_val
                    )

        model.setup(output_buffer_len=ticks, use_cuda=True)
        # Choose a single random neuron to stimulate:
        # input_id = np.random.randint(0, n_neurons - 1)
        spike_neuron = randint(a=0, b=n_neurons - 1)
        print("Input applied to neuron:", spike_neuron)
        model.spike(spike_neuron, 1, 400)

        t_create_end = time.time()
        print(
            f"Time to create the random network: {t_create_end - t_create_start} seconds"
        )

        t_sim_start = time.time()
        # Simulate for 20 steps:
        model.simulate(ticks=ticks, update_data_ticks=1)
        t_sim_end = time.time()
        print(
            f"Time to simulate the network: {t_sim_end - t_sim_start} seconds"
        )

        t_mon_start = time.time()
        # Get spike times
        for neuron_id in range(n_neurons):
            spike_times = model.get_spikes(neuron_id=neuron_id)
        t_mon_end = time.time()
        print(
            f"Time to get spikes for all neurons: {t_mon_end - t_mon_start} seconds"
        )
