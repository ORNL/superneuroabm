"""
Custom Bounded STDP (Spike-Timing Dependent Plasticity) learning rule.

Weight-bounded (multiplicative) variant of pair-wise exponential STDP.
Weight changes scale by distance to the boundary:

    if post fires: dW = +pre_trace  * (w_max - w)   (LTP, bounded above)
    if pre fires:  dW = -post_trace * (w - w_min)    (LTD, bounded below)

This keeps weights within [w_min, w_max] and produces a stable equilibrium.

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
def bounded_stdp(
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

    # Weight bounds
    w_min = 0.0
    w_max = 28.0

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

    # Bounded STDP weight update
    dW = (
        pre_trace * post_soma_spike * (w_max - weight)
        - post_trace * pre_soma_spike * (weight - w_min)
    )

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
