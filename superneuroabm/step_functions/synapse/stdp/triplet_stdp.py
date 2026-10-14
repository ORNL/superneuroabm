"""Pfister-Gerstner triplet STDP step function.

State layout in learning_internal_states[agent_index]:
  [r1_pre_fast, r2_pre_slow, o1_post_fast, o2_post_slow, dW]

Learning parameter layout:
  [0]=stdp_type (selector metadata)
  [1]=tau_plus_s, [2]=tau_minus_s
  [3]=A2_plus, [4]=A2_minus, [5]=stdp_history_length (metadata)
  [6]=tau_x_s, [7]=tau_y_s, [8]=A3_plus, [9]=A3_minus

The four traces are updated before the event-gated weight change, following the
trace-first convention of the supplied pairwise kernel. The rule is additive;
unbounded and unquantized.
"""

from cupyx import jit

from superneuroabm.step_functions.synapse.util import get_soma_spike


@jit.rawkernel(device="cuda")
def triplet_stdp(
    tick,
    agent_index,
    dt,
    I_bias,
    agent_ids,
    breeds,
    locations,
    synapse_params,
    learning_params,  # [type,tau+,tau-,A2+,A2-,history,tau_x,tau_y,A3+,A3-]
    internal_states,
    learning_internal_states,  # [r1,r2,o1,o2,dW]
    synapse_history,
    input_spikes_tensor,
    output_spikes_tensor,
    internal_states_buffer,
    learning_internal_states_buffer,
):
    t_current = int(tick)
    r1 = learning_internal_states[agent_index][0]
    r2 = learning_internal_states[agent_index][1]
    o1 = learning_internal_states[agent_index][2]
    o2 = learning_internal_states[agent_index][3]
    dW = learning_internal_states[agent_index][4]

    pre_soma_index = locations[agent_index][0]
    post_soma_index = locations[agent_index][1]
    pre_spike = get_soma_spike(
        tick, agent_index, dt, I_bias, agent_ids, pre_soma_index,
        t_current, input_spikes_tensor, output_spikes_tensor,
    )
    post_spike = get_soma_spike(
        tick, agent_index, dt, I_bias, agent_ids, post_soma_index,
        t_current, input_spikes_tensor, output_spikes_tensor,
    )

    tracking = len(learning_internal_states_buffer[agent_index]) > 1
    if (pre_spike == 0.0 and post_spike == 0.0 and r1 == 0.0 and r2 == 0.0
            and o1 == 0.0 and o2 == 0.0 and dW == 0.0 and not tracking):
        return

    p = learning_params[agent_index]
    tau_plus = p[1]
    tau_minus = p[2]
    A2_plus = p[3]
    A2_minus = p[4]
    tau_x = p[6]
    tau_y = p[7]
    A3_plus = p[8]
    A3_minus = p[9]

    # Forward-Euler trace decay, consistent with the supplied pairwise kernel.
    r1 = r1 * (1.0 - dt / tau_plus) + pre_spike
    r2 = r2 * (1.0 - dt / tau_x) + pre_spike
    o1 = o1 * (1.0 - dt / tau_minus) + post_spike
    o2 = o2 * (1.0 - dt / tau_y) + post_spike

    # At a post spike: pair LTP + post-post-pre triplet term.
    # At a pre spike: pair LTD + pre-pre-post triplet term.
    dW = post_spike * r1 * (A2_plus + A3_plus * o2)
    dW -= pre_spike * o1 * (A2_minus + A3_minus * r2)

    weight = synapse_params[agent_index][0] + dW
    synapse_params[agent_index][0] = weight
    learning_internal_states[agent_index][0] = r1
    learning_internal_states[agent_index][1] = r2
    learning_internal_states[agent_index][2] = o1
    learning_internal_states[agent_index][3] = o2
    learning_internal_states[agent_index][4] = dW

    buffer_idx = t_current % len(learning_internal_states_buffer[agent_index])
    learning_internal_states_buffer[agent_index][buffer_idx][0] = r1
    learning_internal_states_buffer[agent_index][buffer_idx][1] = r2
    learning_internal_states_buffer[agent_index][buffer_idx][2] = o1
    learning_internal_states_buffer[agent_index][buffer_idx][3] = o2
    learning_internal_states_buffer[agent_index][buffer_idx][4] = dW
