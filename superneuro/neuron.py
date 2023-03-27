"""
Neuron class for spiking neural networks

"""
from typing import List, Optional, Union
from multiprocessing.managers import ListProxy


import numpy as np
from copy import copy
from sagesim.agent import Agent
from sagesim.datacollector import AgentDataCollector


def neuron_step_func(
    global_data_vector: List[Union[int, float, Agent]],
    agent_data_tensor: ListProxy,  # List[List[Union[int, float, Agent]]],
    my_idx: int,
    rng_state,
):
    # extract the parameters:
    neuron_vector = agent_data_tensor[my_idx]
    threshold = neuron_vector[1]
    reset = neuron_vector[2]
    leak = neuron_vector[3]
    t_ref = neuron_vector[4]
    # agent_data_tensor[my_idx][5] is axonal_delay
    # list of incoming synaspes:
    in_synapses = neuron_vector[6]
    out_synapses = neuron_vector[7]
    # elapsed time since last spike
    Vm = neuron_vector[8]
    t_elapse = neuron_vector[9]
    delay_reg = neuron_vector[10]
    external_spikes = neuron_vector[11]
    t_current = int(global_data_vector[0])

    # update t_elapse from refactory period time
    neuron_vector[9] = 0 if t_elapse == 0 else t_elapse - 1
    # If still in refactory period, return
    if t_elapse > 0:
        agent_data_tensor[my_idx] = neuron_vector
        return
    # Otherwise update Vm, etc
    # Update Vm with the input spikes if any:
    for spike in external_spikes:
        if spike[0] == t_current:
            Vm += spike[1]
    # Access the weighted inputs from each synapse
    Xin_tot = sum(
        [agent_data_tensor[syn_idx][5][-1] for syn_idx in in_synapses]
    )
    # Update Vm with in synapse signals
    Vm = Vm + Xin_tot - leak

    # Apply axonal delay to the output of the spike:
    delay_reg = np.roll(delay_reg, 1, 0)
    delay_reg[0] = 1 if Vm > threshold else 0
    # Shift the register each time step

    # Check if delayed Vm was over threshold, if so spike
    if delay_reg[-1]:
        # Spike
        # Send out the spike to all out_synapses
        for out_synapse in out_synapses:
            # Access the synpase
            out_synapse_vector = agent_data_tensor[out_synapse]
            out_synapse_vector[3] = 1
            agent_data_tensor[out_synapse] = out_synapse_vector
        # reset Vm
        Vm = reset
        # start refactory period
        t_elapse = t_ref

    neuron_vector[8] = Vm
    neuron_vector[9] = t_elapse
    neuron_vector[10] = delay_reg
    agent_data_tensor[my_idx] = neuron_vector


class Neuron(Agent):
    def __init__(
        self,
        threshold: float = 1,
        reset_state: float = 0,
        leak: float = 0,
        refactory_period: int = 0,
        axonal_delay: int = 1,
        in_synapses: List[Agent] = None,
        out_synapses: List[Agent] = None,
        monitor: bool = False,
    ) -> None:
        """
        Initializes Neuron agent.

        :param threshold: Threshold at which to spike
        :param reset_state: State to reset to after spiking
        :param leak: Amount by which current state will fall towards reset_state
             for each tick
        :param refactory_period: Number of ticks to remain dormant after spiking
        :param axonal_delay: Number of ticks to wait before spike is transmitted
            to output synapses.
        :param in_synapses: List of input synapses
        :param out_synapses: List of output synapses.
        :param monitor: If True, this node will be monitored.
            False by default.
        """
        super().__init__()

        self._reset_state = reset_state
        self._threshold = threshold
        self._refactory_period = refactory_period
        self._axonal_delay = axonal_delay
        self._leak = leak
        self._monitor = monitor
        self._spike_monitor = None

        if axonal_delay < 1.0:
            # error check
            raise ValueError("Minimum value of axonal delay is 1.0.")

        super().register_property("threshold", threshold, 1)  # 1
        super().register_property("reset_state", reset_state, 2)  # 2
        super().register_property("leak", leak, 3)  # 3
        super().register_property("refactory_period", refactory_period, 4)  # 4
        super().register_property("axonal_delay", axonal_delay, 5)  # 5
        super().register_property(
            "in_synapses", [] if in_synapses == None else in_synapses, 6
        )  # 6
        super().register_property(
            "out_synapses", [] if out_synapses == None else out_synapses, 7
        )  # 7

        super().register_agent_step_function(neuron_step_func, 0)

        # internal state:
        super().register_property(
            "Vm", reset_state, 8
        )  # current value of state #8
        super().register_property(
            "t_elapse", 0, 9
        )  # time elasped since last spike #9

        super().register_property(
            "delay_reg", [0] * axonal_delay, 10
        )  # time elasped since last spike #10

        # External input spikes to the neuron:
        super().register_property("spikes", [], 11)

    def spike(self, tick: int, value: float) -> None:
        """
        Schedules an external input spike to this Neuron.

        :param tick: tick at which spike should be triggered
        :param value: spike value
        """
        input_spikes = super().get_property_value("spikes")
        input_spikes.append([tick, value])
        super().set_property_value("spikes", input_spikes)

    @property
    def monitor(self) -> bool:
        """
        Returns whether his Neuron's spike state is being monitored
            or not.

        :return: monitored or not
        """
        return self._monitor

    @monitor.setter
    def monitor(self, monitor: bool) -> None:
        """
        Setter for whether this Neuron's spike state is monitored
            or not. NeuromorphicModel will pick this up and initialize
            self._spike_monitor with an AgentDataCollector

        :param monitor: Whether to monitor Neuron or not.
        """
        self._monitor = monitor

    def get_spike_times(self) -> Optional[List[int]]:
        """
        Returns the spike times (and spike values) for the neuron

        :param neurons: List of neuron agents to be tracked (int)
        :return: List of ticks at which Neuron spiked, if monitored.
            None if not monitored.

        """
        if not self._monitor:
            return None
        # Use AgentDataCollector provided by NeuromorphicModel
        spike_times = []
        spike_results = self._spike_monitor.data
        for tick, t in enumerate(spike_results):
            if t[self.agent_id] != 0:
                spike_times.append(tick)
        return spike_times
