"""
Weighted synapse step functions for spiking neural networks

"""

# import numpy as np
import cupy as cp
from cupyx import jit

from superneuroabm.step_functions.synapse.util import get_soma_spike


@jit.rawkernel(device="cuda")
def weighted_synapse_step_func(
    tick,
    agent_index,
    globals,
    agent_ids,
    breeds,
    locations,
    synapse_params,  # weight, synaptic delay
    learning_params,
    internal_state,  #
    internal_learning_state,
    synapse_history,  # delay
    input_spikes_tensor,  # input spikes
    output_spikes_tensor,
    internal_states_buffer,
    internal_learning_states_buffer,
):
    t_current = int(tick)

    dt = globals[0]  # time step size

    weight = synapse_params[agent_index][0]
    synaptic_delay = synapse_params[agent_index][1]
    

    location_data = locations[agent_index]
    
    pre_soma_id = -1 if cp.isnan(location_data[1]) else location_data[0]
        
    spike = get_soma_spike(
        tick,
        agent_index,
        globals,
        agent_ids,
        pre_soma_id,
        t_current,
        input_spikes_tensor,
        output_spikes_tensor,
    )

    I_synapse = internal_state[agent_index][0]

    I_synapse = spike * weight

    internal_state[agent_index][0] = I_synapse

    internal_states_buffer[agent_index][t_current][0] = I_synapse
    internal_states_buffer[agent_index][t_current][1] = spike
    internal_states_buffer[agent_index][t_current][2] = t_current
    internal_states_buffer[agent_index][t_current][3] = pre_soma_id
