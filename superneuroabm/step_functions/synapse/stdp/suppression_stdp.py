"""Classical all-to-all Froemke-Dan suppression STDP.

This is the continuous-valued suppression model, not the simplified one-bit
suppression rule from Gautam et al. (ICONS 2024).

State layout in learning_internal_states[agent_index]:
  [pre_eligibility, post_eligibility, last_pre_tick, last_post_tick,
   has_pre, has_post, dW, last_tick]

The rule is event-woken (run on the ticks its pre or post fires): the eligibility
traces decay over the ticks since it last ran, `(1 - dt/tau)^n` (trace_decay); the
suppression factors use the stored spike ticks, so they are unaffected. last_tick
keeps the tick the rule last ran.

Learning parameter layout:
  [0]=stdp_type, [1]=tau_plus_s, [2]=tau_minus_s,
  [3]=A_plus, [4]=A_minus, [5]=history length (metadata),
  [6]=tau_supp_pre_s, [7]=tau_supp_post_s.

The eligibility traces accumulate the contributions of all earlier spikes,
weighted by their per-spike suppression factors. The rule is additive; unbounded and unquantized.
"""

import cupy as cp
from cupyx import jit

from superneuroabm.step_functions.synapse.util import get_soma_spike, trace_decay


@jit.rawkernel(device="cuda")
def suppression_stdp(
    tick,
    agent_index,
    dt,
    I_bias,
    agent_ids,
    breeds,
    locations,
    synapse_params,
    learning_params,  # [type,tau+,tau-,A+,A-,history,tau_s_pre,tau_s_post]
    internal_states,
    learning_internal_states,  # [pre_elig,post_elig,last_pre,last_post,has_pre,has_post,dW,last_tick]
    synapse_history,
    input_spikes_tensor,
    output_spikes_tensor,
    internal_states_buffer,
    learning_internal_states_buffer,
):
    t_current = int(tick)
    pre_elig = learning_internal_states[agent_index][0]
    post_elig = learning_internal_states[agent_index][1]
    last_pre_tick = int(learning_internal_states[agent_index][2])
    last_post_tick = int(learning_internal_states[agent_index][3])
    has_pre = learning_internal_states[agent_index][4]
    has_post = learning_internal_states[agent_index][5]
    dW = learning_internal_states[agent_index][6]

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
    if (pre_spike == 0.0 and post_spike == 0.0 and pre_elig == 0.0
            and post_elig == 0.0 and dW == 0.0 and not tracking):
        return

    p = learning_params[agent_index]
    tau_plus = p[1]
    tau_minus = p[2]
    A_plus = p[3]
    A_minus = p[4]
    tau_supp_pre = p[6]
    tau_supp_post = p[7]

    # Decay all-to-all pair eligibility traces to the present time.
    # ... over the ticks since the rule last ran (1 when stepped every tick)
    _n = t_current - int(learning_internal_states[agent_index][7])
    if _n < 1:
        _n = 1
    pre_elig *= trace_decay(dt, tau_plus, _n)
    post_elig *= trace_decay(dt, tau_minus, _n)

    # A spike's efficacy is suppressed by close preceding spikes of the same
    # neuron: epsilon = 1 - exp(-preceding_ISI / tau_suppression).
    eps_pre = 1.0
    eps_post = 1.0
    if pre_spike != 0.0 and has_pre != 0.0:
        isi_pre_s = (t_current - last_pre_tick) * dt
        eps_pre = 1.0 - cp.exp(-isi_pre_s / tau_supp_pre)
    if post_spike != 0.0 and has_post != 0.0:
        isi_post_s = (t_current - last_post_tick) * dt
        eps_post = 1.0 - cp.exp(-isi_post_s / tau_supp_post)

    # Pair every new event with prior opposite-side events. For simultaneous
    # pre/post spikes, use the conventional Δt=0 causal (potentiating) branch.
    dW = post_spike * eps_post * pre_elig
    dW -= pre_spike * eps_pre * post_elig
    if pre_spike != 0.0 and post_spike != 0.0:
        dW += A_plus * eps_pre * eps_post

    # Add each current event to its eligibility trace after pairing, so it will
    # interact with future opposite-side events rather than be counted twice.
    pre_elig += pre_spike * eps_pre * A_plus
    post_elig += post_spike * eps_post * A_minus

    weight = synapse_params[agent_index][0] + dW
    synapse_params[agent_index][0] = weight
    if pre_spike != 0.0:
        last_pre_tick = t_current
        has_pre = 1.0
    if post_spike != 0.0:
        last_post_tick = t_current
        has_post = 1.0

    learning_internal_states[agent_index][0] = pre_elig
    learning_internal_states[agent_index][1] = post_elig
    learning_internal_states[agent_index][2] = last_pre_tick
    learning_internal_states[agent_index][3] = last_post_tick
    learning_internal_states[agent_index][4] = has_pre
    learning_internal_states[agent_index][5] = has_post
    learning_internal_states[agent_index][6] = dW
    learning_internal_states[agent_index][7] = 1.0 * t_current

    buffer_idx = t_current % len(learning_internal_states_buffer[agent_index])
    learning_internal_states_buffer[agent_index][buffer_idx][0] = pre_elig
    learning_internal_states_buffer[agent_index][buffer_idx][1] = post_elig
    learning_internal_states_buffer[agent_index][buffer_idx][2] = last_pre_tick
    learning_internal_states_buffer[agent_index][buffer_idx][3] = last_post_tick
    learning_internal_states_buffer[agent_index][buffer_idx][4] = has_pre
    learning_internal_states_buffer[agent_index][buffer_idx][5] = has_post
    learning_internal_states_buffer[agent_index][buffer_idx][6] = dW
