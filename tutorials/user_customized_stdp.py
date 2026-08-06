"""
user_customized_stdp -- SuperNeuroMAT-style bounded STDP with constant depression.

Tutorial-owned learning rule for the digits notebook. Registered at runtime with
``model.register_learning_rule(step_func=user_customized_stdp, step_func_path=...)``,
which must be called before ``setup()``. The function name must not collide with a
built-in rule (``exp_pair_wise_stdp_bounded`` etc.), or registration is rejected.

Learning rule
-------------
1. When the postsynaptic neuron spikes:
      - If the presynaptic neuron fired recently, potentiate using pre_trace.
      - Otherwise, depress by a_exp_post.

2. When the presynaptic neuron spikes but the postsynaptic neuron does not,
   depress the synapse by a_exp_post.

3. Clip the updated weight to [wmin, wmax].
"""

import cupy as cp
from cupyx import jit

from superneuroabm.step_functions.synapse.util import get_soma_spike
from sagesim.math_utils import clamp



@jit.rawkernel(device="cuda")
def user_customized_stdp(
    tick,
    agent_index,
    dt,
    I_bias,
    agent_ids,
    breeds,
    locations,
    synapse_params,
    learning_params,
    internal_states,
    learning_internal_states,
    synapse_history,
    input_spikes_tensor,
    output_spikes_tensor,
    internal_states_buffer,
    learning_internal_states_buffer,
):
    t_current = int(tick)

    # ------------------------------------------------------------
    # Synapse parameters
    # ------------------------------------------------------------
    weight = synapse_params[agent_index][0]
    synaptic_delay = synapse_params[agent_index][1]

    # ------------------------------------------------------------
    # Learning parameters
    # ------------------------------------------------------------
    tau_pre_stdp = learning_params[agent_index][1]
    tau_post_stdp = learning_params[agent_index][2]

    # Amount added to the presynaptic trace on a pre spike.
    a_exp_pre = learning_params[agent_index][3]

    # Constant depression magnitude.
    a_exp_post = learning_params[agent_index][4]

    stdp_history_length = learning_params[agent_index][5]
    wmin = learning_params[agent_index][6]
    wmax = learning_params[agent_index][7]

    # ------------------------------------------------------------
    # Learning internal states
    # ------------------------------------------------------------
    pre_trace = learning_internal_states[agent_index][0]
    post_trace = learning_internal_states[agent_index][1]
    dW = learning_internal_states[agent_index][2]

    # ------------------------------------------------------------
    # Pre- and postsynaptic neuron indices
    # ------------------------------------------------------------
    pre_soma_index = locations[agent_index][0]
    post_soma_index = locations[agent_index][1]

    # ------------------------------------------------------------
    # Read pre- and postsynaptic spikes
    # ------------------------------------------------------------
    pre_soma_spike = get_soma_spike(
        tick,
        agent_index,
        dt,
        I_bias,
        agent_ids,
        pre_soma_index,
        t_current,
        input_spikes_tensor,
        output_spikes_tensor,
    )

    post_soma_spike = get_soma_spike(
        tick,
        agent_index,
        dt,
        I_bias,
        agent_ids,
        post_soma_index,
        t_current,
        input_spikes_tensor,
        output_spikes_tensor,
    )

    # ------------------------------------------------------------
    # Update traces using current spikes
    # ------------------------------------------------------------
    pre_trace = (
        pre_trace * (1.0 - dt / tau_pre_stdp)
        + pre_soma_spike * a_exp_pre
    )

    post_trace = (
        post_trace * (1.0 - dt / tau_post_stdp)
        + post_soma_spike * a_exp_post
    )

    # ------------------------------------------------------------
    # Remove traces that have decayed below their active windows
    # ------------------------------------------------------------
    trace_threshold_pre = a_exp_pre / 5.0
    trace_threshold_post = a_exp_post / 5.0

    if pre_trace <= trace_threshold_pre:
        pre_trace = 0.0

    if post_trace <= trace_threshold_post:
        post_trace = 0.0

    # ------------------------------------------------------------
    # Weight update
    # ------------------------------------------------------------
    dW = 0.0

    if post_soma_spike > 0.0:
        # The postsynaptic neuron fired.
        if pre_trace > trace_threshold_pre:
            # The presynaptic neuron fired recently:
            # reinforce the synapse.
            dW = pre_trace
        else:
            # The postsynaptic neuron fired without recent activity
            # from this presynaptic neuron.
            dW = -a_exp_post

    elif pre_soma_spike > 0.0:
        # This presynaptic neuron fired, but this postsynaptic neuron
        # did not respond.
        dW = -a_exp_post

    # Clip the updated weight.
    weight = clamp(weight + dW, wmin, wmax)

    # ------------------------------------------------------------
    # Store updated values
    # ------------------------------------------------------------
    synapse_params[agent_index][0] = weight

    learning_internal_states[agent_index][0] = pre_trace
    learning_internal_states[agent_index][1] = post_trace
    learning_internal_states[agent_index][2] = dW

    # ------------------------------------------------------------
    # State tracking buffer
    # ------------------------------------------------------------
    buffer_idx = (
        t_current
        % len(learning_internal_states_buffer[agent_index])
    )

    learning_internal_states_buffer[agent_index][buffer_idx][0] = pre_trace
    learning_internal_states_buffer[agent_index][buffer_idx][1] = post_trace
    learning_internal_states_buffer[agent_index][buffer_idx][2] = dW