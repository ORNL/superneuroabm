"""
Bounded exponential STDP (Spike-Timing Dependent Plasticity) step function.

Same as exp_pair_wise_stdp but clips weights to [wmin, wmax] after each update.
Used for Masquelier et al. (2008) pattern detection replication.
"""

import cupy as cp
from cupyx import jit

from superneuroabm.step_functions.synapse.util import get_soma_spike


@jit.rawkernel(device="cuda")
def exp_pair_wise_stdp_bounded(
    tick,
    agent_index,
    globals,
    agent_ids,
    breeds,
    locations,
    synapse_params,  # scale, time constant (tau_rise and tau_fall)
    learning_params,
    internal_state,  #
    internal_learning_state,  # learning state variables
    synapse_history,  # delay
    input_spikes_tensor,  # input spikes
    output_spikes_tensor,
    internal_states_buffer,
    internal_learning_states_buffer,
):
    t_current = int(tick)

    dt = globals[0]  # time step size

    # Get the synapse parameters:
    weight = synapse_params[agent_index][0]
    synaptic_delay = synapse_params[agent_index][1]

    # Get the learning parameters:
    # stdpType = 2 # Parsed in the learning rule selector
    tau_pre_stdp = learning_params[agent_index][1]
    tau_post_stdp = learning_params[agent_index][2]
    a_exp_pre = learning_params[agent_index][3]
    a_exp_post = learning_params[agent_index][4]
    stdp_history_length = learning_params[agent_index][5]
    wmin = learning_params[agent_index][6]
    wmax = learning_params[agent_index][7]

    pre_trace = internal_learning_state[agent_index][0]
    post_trace = internal_learning_state[agent_index][1]
    dW = internal_learning_state[agent_index][2]

    # locations[agent_index] = [pre_soma_index, post_soma_index]
    # SAGESim has already converted agent IDs to local indices
    pre_soma_index = locations[agent_index][0]
    post_soma_index = locations[agent_index][1]

    # Get the pre-soma spike
    pre_soma_spike = get_soma_spike(
        tick,
        agent_index,
        globals,
        agent_ids,
        pre_soma_index,
        t_current,
        input_spikes_tensor,
        output_spikes_tensor,
    )

    post_soma_spike = get_soma_spike(
        tick,
        agent_index,
        globals,
        agent_ids,
        post_soma_index,
        t_current,
        input_spikes_tensor,
        output_spikes_tensor,
    )

    pre_trace = pre_trace * (1 - dt / tau_pre_stdp) + pre_soma_spike * a_exp_pre
    post_trace = post_trace * (1 - dt / tau_post_stdp) + post_soma_spike * a_exp_post
    dW = pre_trace * post_soma_spike - post_trace * pre_soma_spike

    weight += dW  # Update the weight

    # Clip weight to [wmin, wmax] (no cp.clip in raw kernels)
    weight = weight if weight <= wmax else wmax
    weight = weight if weight >= wmin else wmin

    synapse_params[agent_index][0] = weight  # Update the clipped weight

    internal_learning_state[agent_index][0] = pre_trace
    internal_learning_state[agent_index][1] = post_trace
    internal_learning_state[agent_index][2] = dW

    # Safe buffer indexing: use modulo to prevent out-of-bounds access
    # When tracking is disabled, buffer length is 1, so t_current % 1 = 0 always
    buffer_idx = t_current % len(internal_learning_states_buffer[agent_index])
    internal_learning_states_buffer[agent_index][buffer_idx][0] = pre_trace
    internal_learning_states_buffer[agent_index][buffer_idx][1] = post_trace
    internal_learning_states_buffer[agent_index][buffer_idx][2] = dW

    internal_state[agent_index][2] = pre_trace
    internal_state[agent_index][3] = post_trace

    # Safe buffer indexing for internal_states_buffer (reuse buffer_idx from above)
    state_buffer_idx = t_current % len(internal_states_buffer[agent_index])
    internal_states_buffer[agent_index][state_buffer_idx][2] = post_soma_spike
    internal_states_buffer[agent_index][state_buffer_idx][3] = post_trace
