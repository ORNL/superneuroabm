"""
Bounded exponential STDP for digits classification tutorial.

Same logic as exp_pair_wise_stdp_bounded: accumulating traces with hard
clipping to [wmin, wmax]. Registered as a custom learning rule so it
coexists with built-in rules.
"""

import cupy as cp
from cupyx import jit

from superneuroabm.step_functions.synapse.util import get_soma_spike


@jit.rawkernel(device="cuda")
def digits_bounded_stdp(
    tick,
    agent_index,
    globals,
    agent_ids,
    breeds,
    locations,
    synapse_params,
    learning_params,
    internal_state,
    internal_learning_state,
    synapse_history,
    input_spikes_tensor,
    output_spikes_tensor,
    internal_states_buffer,
    internal_learning_states_buffer,
):
    t_current = int(tick)

    dt = globals[0]

    weight = synapse_params[agent_index][0]

    tau_pre_stdp = learning_params[agent_index][1]
    tau_post_stdp = learning_params[agent_index][2]
    a_exp_pre = learning_params[agent_index][3]
    a_exp_post = learning_params[agent_index][4]
    wmin = learning_params[agent_index][6]
    wmax = learning_params[agent_index][7]

    pre_trace = internal_learning_state[agent_index][0]
    post_trace = internal_learning_state[agent_index][1]
    dW = internal_learning_state[agent_index][2]

    pre_soma_index = locations[agent_index][0]
    post_soma_index = locations[agent_index][1]

    pre_soma_spike = get_soma_spike(
        tick, agent_index, globals, agent_ids,
        pre_soma_index, t_current,
        input_spikes_tensor, output_spikes_tensor,
    )
    post_soma_spike = get_soma_spike(
        tick, agent_index, globals, agent_ids,
        post_soma_index, t_current,
        input_spikes_tensor, output_spikes_tensor,
    )

    pre_trace = pre_trace * (1 - dt / tau_pre_stdp) + pre_soma_spike * a_exp_pre
    post_trace = post_trace * (1 - dt / tau_post_stdp) + post_soma_spike * a_exp_post
    dW = pre_trace * post_soma_spike - post_trace * pre_soma_spike

    weight += dW
    weight = weight if weight <= wmax else wmax
    weight = weight if weight >= wmin else wmin

    synapse_params[agent_index][0] = weight

    internal_learning_state[agent_index][0] = pre_trace
    internal_learning_state[agent_index][1] = post_trace
    internal_learning_state[agent_index][2] = dW

    buffer_idx = t_current % len(internal_learning_states_buffer[agent_index])
    internal_learning_states_buffer[agent_index][buffer_idx][0] = pre_trace
    internal_learning_states_buffer[agent_index][buffer_idx][1] = post_trace
    internal_learning_states_buffer[agent_index][buffer_idx][2] = dW

    internal_state[agent_index][2] = pre_trace
    internal_state[agent_index][3] = post_trace

    state_buffer_idx = t_current % len(internal_states_buffer[agent_index])
    internal_states_buffer[agent_index][state_buffer_idx][2] = post_soma_spike
    internal_states_buffer[agent_index][state_buffer_idx][3] = post_trace
