"""
Model class for building an SNN
"""

from collections import namedtuple
import multiprocessing
from typing import List, Optional, Union
import math
from copy import copy

import numpy as np
from numba import cuda
from multiprocessing import Pool

from sagesim.agent import Agent
from superneuro.model import Model
from sagesim.datacollector import AgentDataCollector
from superneuro.neuron import Neuron, neuron_step_func
from superneuro.synapse import Synapse, synaptic_step_func


class NeuromorphicModel(Model):

    THREADSPERBLOCK = 16

    def __init__(self, use_cuda: bool = False) -> None:
        """
        Creates an SNN Model and provides methods to create, simulate,
        and monitor neurons and synapses.

        :param use_cuda: True if the system supports CUDA GPU acceleration
        """
        super().__init__(name="NeuromorphicModel", use_cuda=use_cuda)
        self._syn_labels = set([])
        # dictionary for datacollectors for each neuron to be tracked
        self._neuron_spike_collectors = {}

    def simulate(
        self, ticks: int, update_data_ticks: int = 1, num_cpu_proc: int = 4
    ) -> None:
        """
        Override of superneuro.core.model mainly to register an
        AgentDataCollector to monitor marked output Neurons.

        """
        self._monitored_neurons = self.get_agents_with(
            lambda a: type(a) == Neuron and a.monitor
        )
        self._spike_monitor = AgentDataCollector(
            lambda agent: agent.get_property_value("delay_reg")[-1],
            self._monitored_neurons,
        )
        self.register_agent_data_collector(self._spike_monitor)
        for mn in self._monitored_neurons:
            mn._spike_monitor = self._spike_monitor
        super().simulate(ticks, update_data_ticks, num_cpu_proc)

    def create_neuron(self, **kwargs) -> Neuron:
        """
        Creates and adds Neuron agent to NeuromorphicModel instance.

        :param kwargs: see superneuro.snn.neuron.Neuron
        :return: Neuron agent object
        """
        neuron = Neuron(**kwargs)
        super().add_agent(neuron)
        return neuron

    def create_synapse(self, pre_neuron, post_neuron, **kwargs) -> Synapse:
        """
        Creates a synapse object, adds it to the model.

        :param pre_neuron: Neuron at start of synapse
        :param post_neuron: Neuron at end of synapse
        :param kwargs: see superneuro.snn.Synapse
        :return: created Synapse agent object
        """
        # Assign the weights and delays to the synpase instance
        syn = Synapse(**kwargs)
        syn._label = f"{pre_neuron._agent_id},{ post_neuron._agent_id}"
        if syn._label in self._syn_labels:
            raise ValueError(
                f"Synapse {syn._label} already exists and cannot be created."
            )
        self.add_agent(syn)
        # Update the neuron's in_synapse and out_synapse
        post_synapse = pre_neuron.get_property_value("out_synapses")
        if syn._agent_id not in post_synapse:
            post_synapse.append(syn._agent_id)

        # Add the pre-synapses:
        pre_synapse = post_neuron.get_property_value("in_synapses")
        pre_synapse.append(syn._agent_id)

        post_synapse = []
        pre_synapse = []

        return syn

    def get_synapse(self, syn_label) -> Synapse:
        """
        Returns synapse by synapse label. see superneuro.snn.Synapse

        :param syn_label: Label of Synapse object to be returned
        :return: Synapse agent object that matches label
        """
        syn_label = syn_label.split(",")
        # Function to get the synpatic weights/delays during training
        # NOTE: syn_id - (pre_neuron id, post_neuron id)
        synapse = list(
            self.get_agents_with(
                lambda agent: (type(agent) == Synapse)
                and (agent._label == f"{syn_label[0]},{syn_label[1]}")
            )
        )[0]
        return synapse

    def summary(self) -> str:
        """
        Verbose summary of the network structure.

        :return: str information of netowkr struture
        """
        # Function to print
        # TODO: Have a better way to represent the network structure(?)
        summary = []
        for k in range(len(self._agents)):
            summary.append(f"Agent: {k} Type:{type(self.get_agent(k))}")
        return "\n".join(summary)

    # TODO: Have a reset function to clear the agent_data_tensor from the created network
    # But keep the network intact, before the simulation

    def network_reset(self) -> None:
        # Neuron: Vm, t_elapse, delay_reg to be cleared
        # Synapse: delay_reg to be cleared
        # for k in range(len(self._agents)):
        #    if type(self.get_agent(k))==Neuron:
        #        print("Agent:",k,' Data:',self.get_agent(k).get_property_value("Vm"))

        #        self.get_agent(k).set_property_value("Vm",0)

        #        print('After reset:',self.get_agent(k).get_property_value("Vm"))
        #
        #    elif type(self.get_agent(k))==Synapse:
        #        print("Agent:",k,' Data:',self.get_agent(k).get_property_value("synapse_input"))

        #        self.get_agent(k).set_property_value("synapse_input",0)

        #        print('After reset:',self.get_agent(k).get_property_value("synapse_input"))

        pass
