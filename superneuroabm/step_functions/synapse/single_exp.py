"""
Single exponential synapse step functions for spiking neural networks

"""

import cupy as cp
import numpy as np
from cupyx import jit

from superneuroabm.step_functions.synapse.util import get_soma_spike
from sagesim.utils import get_neighbor_data_from_tensor


@jit.rawkernel(device="cuda")
def synapse_single_exp_step_func(
    tick,
    agent_index,
    globals,
    agent_ids,
    breeds,
    locations,
    synapse_params,  # scale, time constant (tau_rise and tau_fall)
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
    scale = synapse_params[agent_index][2]
    tau_fall = synapse_params[agent_index][3]
    tau_rise = synapse_params[agent_index][4]

    # locations[agent_index] = [pre_soma_index, post_soma_index]
    # SAGESim has already converted agent IDs to local indices
    pre_soma_index, post_soma_index = (
        locations[agent_index][0],
        locations[agent_index][1],
    )

    spike = get_soma_spike(
        tick,
        agent_index,
        globals,
        agent_ids,
        pre_soma_index,
        t_current,
        input_spikes_tensor,
        output_spikes_tensor,
    )

    I_synapse = internal_state[agent_index][0]

    I_synapse = I_synapse * (1 - dt / tau_fall) + spike * scale * weight

    internal_state[agent_index][0] = I_synapse

    # Safe buffer indexing: use modulo to prevent out-of-bounds access
    # When tracking is disabled, buffer length is 1, so t_current % 1 = 0 always
    buffer_idx = t_current % len(internal_states_buffer[agent_index])
    internal_states_buffer[agent_index][buffer_idx][0] = I_synapse
    internal_states_buffer[agent_index][buffer_idx][1] = spike
    internal_states_buffer[agent_index][buffer_idx][2] = t_current
