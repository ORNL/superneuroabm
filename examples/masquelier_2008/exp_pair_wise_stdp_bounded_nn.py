"""
Additive Restricted Nearest-Neighbor (RNN) STDP (Masquelier et al. 2008).

Matches the default STDP rule from Hathway & Goodman (2018) reimplementation:
  - Purely ADDITIVE weight updates with hard clipping to [wmin, wmax]
  - Nearest-neighbor pairing: traces RESET (not accumulate) on each spike
  - RESTRICTED: traces are ZEROED after use (each spike pair produces at
    most one weight change)

Brian2 equivalent (from the author's reimplementation):
  on_pre:
    LTPtrace = aplus           # reset pre-trace
    wi = clip(wi + LTDtrace)   # apply LTD (LTDtrace <= 0)
    LTDtrace = 0               # consume post-trace
  on_post:
    LTDtrace = -aminus         # reset post-trace
    wi = clip(wi + LTPtrace)   # apply LTP (LTPtrace >= 0)
    LTPtrace = 0               # consume pre-trace

The "restricted" zeroing is critical for balance: it reduces STDP events
by ~50% and makes LTP/LTD nearly balanced, preventing runaway depression.
"""

import cupy as cp
from cupyx import jit

from superneuroabm.step_functions.synapse.util import get_soma_spike


@jit.rawkernel(device="cuda")
def exp_pair_wise_stdp_bounded_nn(
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

    # --- Decay traces (exact exponential, matching Brian2) ---
    pre_trace = pre_trace * cp.exp(-dt / tau_pre_stdp)
    post_trace = post_trace * cp.exp(-dt / tau_post_stdp)

    # --- Process pre-spike (Brian2 on_pre order) ---
    # 1. Reset pre_trace to a_exp_pre (nearest-neighbor)
    pre_trace = pre_trace * (1.0 - pre_soma_spike) + pre_soma_spike * a_exp_pre
    # 2. Apply LTD: weight -= post_trace (post_trace holds decayed aminus from last post-spike)
    ltd = post_trace * pre_soma_spike
    # 3. Zero post_trace (restricted: consume the trace)
    post_trace = post_trace * (1.0 - pre_soma_spike)

    # --- Process post-spike (Brian2 on_post order) ---
    # 1. Reset post_trace to a_exp_post (nearest-neighbor)
    post_trace = post_trace * (1.0 - post_soma_spike) + post_soma_spike * a_exp_post
    # 2. Apply LTP: weight += pre_trace (pre_trace holds decayed aplus from last pre-spike)
    ltp = pre_trace * post_soma_spike
    # 3. Zero pre_trace (restricted: consume the trace)
    pre_trace = pre_trace * (1.0 - post_soma_spike)

    # --- Additive STDP with hard clipping (NO weight dependence) ---
    dW = ltp - ltd
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

    # Zero synaptic currents on post-spike (Brian2's x=0 reset).
    # After the output neuron fires, all accumulated PSP is consumed.
    # This runs at priority 101 (after synapse step at 100), so it
    # overwrites the synapse step's I_fast/I_slow with zeroed values.
    # Next tick, the soma sees zero accumulated input and must re-accumulate.
    internal_state[agent_index][0] *= (1.0 - post_soma_spike)
    internal_state[agent_index][1] *= (1.0 - post_soma_spike)

    # Learning state (pre_trace, post_trace, dW) is tracked exclusively
    # in internal_learning_states_buffer above — no need to duplicate
    # into internal_states_buffer or internal_state.
