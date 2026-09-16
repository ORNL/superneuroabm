"""
Single exponential synapse step functions for spiking neural networks

"""

import cupy as cp
import numpy as np
from cupyx import jit

from superneuroabm.step_functions.synapse.util import get_soma_spike
from sagesim.utils import get_neighbor_data_from_tensor

# Flush-to-zero floor for the synaptic current. 0.0 (default) = exact dynamics. A
# positive value (e.g. 1e-7 * typical weight) snaps a decaying tail to exactly 0 once
# |I| is below it, so an idle synapse reaches the fast path below in a few time
# constants instead of the ~150-900 ticks float32 needs to underflow. This is a
# modelling choice with a bounded effect; set it BEFORE model.setup():
#     superneuroabm.step_functions.synapse.single_exp.I_FLOOR = 1e-7
I_FLOOR = 0.0


@jit.rawkernel(device="cuda")
def synapse_single_exp_step_func(
    tick,
    agent_index,
    dt,
    I_bias,
    agent_ids,
    breeds,
    locations,
    synapse_params,  # scale, time constant (tau_rise and tau_fall)
    learning_params,
    internal_states,  #
    learning_internal_states,
    synapse_history,  # delay
    input_spikes_tensor,  # input spikes
    output_spikes_tensor,
    internal_states_buffer,
    learning_internal_states_buffer,
):
    t_current = int(tick)

    # locations[agent_index] = [pre_soma_index, post_soma_index]
    # SAGESim has already converted agent IDs to local indices
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

    I_synapse = internal_states[agent_index][0]

    # Idle fast path. With no incoming spike and no current, every value this synapse
    # owns would be rewritten unchanged, so return before reading the parameter row.
    # The history buffer is skipped only while tracking is off (its single slot is
    # write-only); with tracking on the buffer must record this tick.
    tracking = len(internal_states_buffer[agent_index]) > 1
    if spike == 0.0 and I_synapse == 0.0 and not tracking:
        return

    weight = synapse_params[agent_index][0]
    scale = synapse_params[agent_index][2]
    tau_fall = synapse_params[agent_index][3]

    I_synapse = I_synapse * (1 - dt / tau_fall) + spike * scale * weight
    if I_synapse < I_FLOOR and I_synapse > -I_FLOOR:
        I_synapse = 0.0

    internal_states[agent_index][0] = I_synapse

    # Safe buffer indexing: use modulo to prevent out-of-bounds access
    # When tracking is disabled, buffer length is 1, so t_current % 1 = 0 always
    buffer_idx = t_current % len(internal_states_buffer[agent_index])
    internal_states_buffer[agent_index][buffer_idx][0] = I_synapse
    internal_states_buffer[agent_index][buffer_idx][1] = spike
    internal_states_buffer[agent_index][buffer_idx][2] = t_current
