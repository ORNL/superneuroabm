"""
Exponential STDP (Spike-Timing Dependent Plasticity) step function for spiking neural networks

"""

import numpy as np
from cupyx import jit

from superneuroabm.step_functions.synapse.util import get_pre_soma_spike


@jit.rawkernel(device="cuda")
def exp_stdp_all_to_all(
    tick,
    agent_index,
    globals,
    agent_ids,
    breeds,
    locations,
    synapse_params,  # scale, time constant (tau_rise and tau_fall)
    internal_state,  #
    synapse_history,  # delay
    input_spikes_tensor,  # input spikes
    output_spikes_tensor,
    internal_states_buffer,
):
    t_current = int(tick)

    dt = globals[0]  # time step size

    weight = synapse_params[agent_index][0]
    synaptic_delay = synapse_params[agent_index][1]
    scale = synapse_params[agent_index][2]
    tau_fall = synapse_params[agent_index][3]
    tau_rise = synapse_params[agent_index][4]
    tau_pre_stdp = synapse_params[agent_index][5]
    tau_post_stdp = synapse_params[agent_index][6]
    a_exp_pre = synapse_params[agent_index][7]
    a_exp_post = synapse_params[agent_index][8]
    stdp_history_length = synapse_params[agent_index][9]

    location_data = locations[agent_index]
    if len(location_data) == 1:
        pre_soma_id = np.nan
        post_soma_id = location_data[0]
    else:
        pre_soma_id = location_data[0]
        post_soma_id = location_data[1]

    pre_soma_spike = get_pre_soma_spike(
        agent_index,
        globals,
        agent_ids,
        pre_soma_id,
        t_current,
        input_spikes_tensor,
        output_spikes_tensor,
    )
    post_soma_spike = get_pre_soma_spike(
        agent_index,
        globals,
        agent_ids,
        post_soma_id,
        t_current,
        input_spikes_tensor,
        output_spikes_tensor,
    )

    pre_trace = pre_trace * (1 - dt / tau_pre_stdp) + pre_soma_spike * aplus_exp_pre
    post_trace = (
        post_trace * (1 - dt / tau_post_stdp) + post_soma_spike * aplus_exp_post
    )

    internal_state[agent_index][2] = pre_trace
    internal_state[agent_index][3] = post_trace
    internal_states_buffer[agent_index][t_current][2] = pre_trace
    internal_states_buffer[agent_index][t_current][3] = post_trace
