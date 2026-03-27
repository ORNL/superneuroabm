"""
Custom Symmetric STDP learning rule.

Unlike standard STDP where pre-before-post causes LTP and post-before-pre
causes LTD, symmetric STDP potentiates for BOTH spike orderings:

    dW = pre_trace * post_spike + post_trace * pre_spike

Both terms are positive, so any correlated pre/post activity strengthens
the synapse. Useful for correlation-based / Hebbian unsupervised learning.

Learning hyperparameters (in order):
    [0] stdp_type         - Rule ID (set by register_learning_rule)
    [1] tau_pre_stdp      - Pre-synaptic trace time constant
    [2] tau_post_stdp     - Post-synaptic trace time constant
    [3] a_exp_pre         - Pre-trace increment on pre-spike
    [4] a_exp_post        - Post-trace increment on post-spike
    [5] stdp_history_length - (unused, kept for compatibility)

Internal learning state (in order):
    [0] pre_trace         - Pre-synaptic eligibility trace
    [1] post_trace        - Post-synaptic eligibility trace
    [2] dW                - Weight change this tick
"""

import cupy as cp
from cupyx import jit

from superneuroabm.step_functions.synapse.util import get_soma_spike


@jit.rawkernel(device="cuda")
def symmetric_stdp(
    tick,
    agent_index,
    dt,
    I_bias,
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

    # Synapse parameters
    weight = synapse_params[agent_index][0]

    # Learning parameters
    tau_pre_stdp = learning_params[agent_index][1]
    tau_post_stdp = learning_params[agent_index][2]
    a_exp_pre = learning_params[agent_index][3]
    a_exp_post = learning_params[agent_index][4]

    # Read traces
    pre_trace = internal_learning_state[agent_index][0]
    post_trace = internal_learning_state[agent_index][1]

    # Pre/post soma indices
    pre_soma_index = locations[agent_index][0]
    post_soma_index = locations[agent_index][1]

    # Get spikes
    pre_soma_spike = get_soma_spike(
        tick, agent_index, dt, I_bias, agent_ids,
        pre_soma_index, t_current,
        input_spikes_tensor, output_spikes_tensor,
    )
    post_soma_spike = get_soma_spike(
        tick, agent_index, dt, I_bias, agent_ids,
        post_soma_index, t_current,
        input_spikes_tensor, output_spikes_tensor,
    )

    # Update traces (exponential decay + spike increment)
    pre_trace = pre_trace * (1 - dt / tau_pre_stdp) + pre_soma_spike * a_exp_pre
    post_trace = post_trace * (1 - dt / tau_post_stdp) + post_soma_spike * a_exp_post

    # Symmetric STDP: both orderings cause potentiation
    dW = pre_trace * post_soma_spike + post_trace * pre_soma_spike

    weight += dW
    synapse_params[agent_index][0] = weight

    # Write learning state
    internal_learning_state[agent_index][0] = pre_trace
    internal_learning_state[agent_index][1] = post_trace
    internal_learning_state[agent_index][2] = dW

    buffer_idx = t_current % len(internal_learning_states_buffer[agent_index])
    internal_learning_states_buffer[agent_index][buffer_idx][0] = pre_trace
    internal_learning_states_buffer[agent_index][buffer_idx][1] = post_trace
    internal_learning_states_buffer[agent_index][buffer_idx][2] = dW
