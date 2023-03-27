"""
Synapse class for spiking neural networks

"""
from typing import Union
from typing import List

from sagesim.agent import Agent
import numpy as np


def synaptic_step_func(
    global_data_vector: List[Union[int, float, Agent]],
    agent_data_tensor: List[List[Union[int, float, Agent]]],
    my_idx: int,
    rng_state,
):
    synapse_vector = agent_data_tensor[my_idx]
    weight = synapse_vector[1]
    # agent_data_tensor[my_idx][2] is delay
    synapse_input = synapse_vector[3]
    # agent_data_tensor[my_idx][4] is synapse_output
    delay_reg = synapse_vector[5]  # holds the weighted output
    # reset input
    synapse_vector[3] = 0

    delay_reg = np.roll(delay_reg, 1, 0)
    delay_reg[0] = weight if synapse_input else 0

    synapse_vector[5] = delay_reg
    agent_data_tensor[my_idx] = synapse_vector


class Synapse(Agent):
    def __init__(self, weight: float = 1, delay: int = 1) -> None:
        super().__init__()

        # Check values of input parameters
        if delay < 1.0:
            raise ValueError("Minimum value of synaptic delay is 1.")

        # Assign values to object variables
        self._delay = delay
        self._weight = weight

        # Register properties
        super().register_property("weight", weight, 1)  # 1
        super().register_property("delay", delay, 2)  # 2
        super().register_property("synapse_input", 0, 3)  # 3
        super().register_property("synapse_output", 0, 4)  # 4
        super().register_property("delay_reg", [0] * delay, 5)  # 5
        super().register_agent_step_function(synaptic_step_func, 1)

        self._label = ","
