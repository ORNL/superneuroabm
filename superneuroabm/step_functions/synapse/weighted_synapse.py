"""
Weighted synapse step functions for spiking neural networks

"""

# import numpy as np
import cupy as cp
from cupyx import jit

from superneuroabm.step_functions.synapse.util import get_soma_spike
from superneuroabm.step_functions.synaptic_input import deliver_synaptic_current


@jit.rawkernel(device="cuda")
def weighted_synapse_step_func(
    tick,
    agent_index,
    dt,
    I_bias,
    agent_ids,
    breeds,
    locations,
    synapse_params,  # weight, synaptic delay
    learning_params,
    internal_states,  #
    learning_internal_states,
    synapse_history,  # delay
    input_spikes_tensor,  # input spikes
    output_spikes_tensor,
    internal_states_buffer,
    learning_internal_states_buffer,
    synaptic_input,
    synaptic_input_idx,
):
    t_current = int(tick)

    weight = synapse_params[agent_index][0]
    synaptic_delay = synapse_params[agent_index][1]


    location_data = locations[agent_index]

    pre_soma_id = -1 if cp.isnan(location_data[1]) else location_data[0]
    # post slot is optional (an adjacency file may list only the pre); the CSR
    # transform rewrites the one alias above, so test it directly
    post_soma_index = -1
    if len(location_data) > 1:
        post_soma_index = location_data[1]

    spike = get_soma_spike(
        tick,
        agent_index,
        dt,
        I_bias,
        agent_ids,
        pre_soma_id,
        t_current,
        input_spikes_tensor,
        output_spikes_tensor,
    )

    I_synapse = internal_states[agent_index][0]

    I_synapse = spike * weight

    internal_states[agent_index][0] = I_synapse
    # push mode: the post soma reads this from its inbox next tick instead of pulling it
    deliver_synaptic_current(synaptic_input, synaptic_input_idx, post_soma_index, 0, I_synapse)

    internal_states_buffer[agent_index][t_current][0] = I_synapse
    internal_states_buffer[agent_index][t_current][1] = spike
    internal_states_buffer[agent_index][t_current][2] = t_current
    internal_states_buffer[agent_index][t_current][3] = pre_soma_id
