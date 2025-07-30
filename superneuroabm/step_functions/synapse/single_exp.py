"""
Single exponential synapse step functions for spiking neural networks

"""

import numpy as np
from cupyx import jit

from superneuroabm.step_functions.synapse.util import get_pre_soma_spike


@jit.rawkernel(device="cuda")
def synapse_single_exp_step_func(
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

    location_data = locations[agent_index]
    if len(location_data) == 0:
        pre_soma_id = np.nan
    else:
        pre_soma_id = location_data[0]
    spike = get_pre_soma_spike(
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

    I_synapse = I_synapse * (1 - dt / tau_fall) + spike * scale * weight

    internal_state[agent_index][0] = I_synapse
    internal_states_buffer[agent_index][t_current][0] = I_synapse
    internal_states_buffer[agent_index][t_current][1] = spike
    internal_states_buffer[agent_index][t_current][2] = t_current
