"""
Double exponential synapse step function for spiking neural networks.

Maintains two exponential filters (fast/slow) so that the SRM soma can
compute the double-exponential PSP: K * (I_slow - I_fast).

  internal_states[0] = I_fast  (filtered with tau_fall, i.e. tau_s)
  internal_states[1] = I_slow  (filtered with tau_rise, repurposed as tau_m)
"""

import cupy as cp
import numpy as np
from cupyx import jit

from superneuroabm.step_functions.synapse.util import get_soma_spike

# Flush-to-zero floor for the synaptic currents; 0.0 = exact (see single_exp.I_FLOOR).
I_FLOOR = 0.0


@jit.rawkernel(device="cuda")
def synapse_double_exp_step_func(
    tick,
    agent_index,
    dt,
    I_bias,
    agent_ids,
    breeds,
    locations,
    synapse_params,  # weight, delay, scale, tau_fast (tau_s), tau_slow (tau_m)
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

    pre_soma_index = locations[agent_index][0]

    spike = get_soma_spike(
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

    I_fast = internal_states[agent_index][0]
    I_slow = internal_states[agent_index][1]

    # Idle fast path (see single_exp.py): nothing arrived and both currents are zero.
    tracking = len(internal_states_buffer[agent_index]) > 1
    if spike == 0.0 and I_fast == 0.0 and I_slow == 0.0 and not tracking:
        return

    weight = synapse_params[agent_index][0]
    scale = synapse_params[agent_index][2]
    tau_fast = synapse_params[agent_index][3]   # tau_s (fast synaptic, e.g. 2.5ms)
    tau_slow = synapse_params[agent_index][4]   # tau_m (slow membrane, e.g. 10ms)

    weighted_spike = spike * scale * weight

    I_fast = I_fast * cp.exp(-dt / tau_fast) + weighted_spike
    I_slow = I_slow * cp.exp(-dt / tau_slow) + weighted_spike
    if I_fast < I_FLOOR and I_fast > -I_FLOOR:
        I_fast = 0.0
    if I_slow < I_FLOOR and I_slow > -I_FLOOR:
        I_slow = 0.0

    internal_states[agent_index][0] = I_fast
    internal_states[agent_index][1] = I_slow

    # Record synapse state to history buffer. Only [0] (I_fast) and [1]
    # (I_slow) are computational state — buffer width is shared across
    # all agent types via SAGESim's padded tensor.
    buffer_idx = t_current % len(internal_states_buffer[agent_index])
    internal_states_buffer[agent_index][buffer_idx][0] = I_fast
    internal_states_buffer[agent_index][buffer_idx][1] = I_slow
