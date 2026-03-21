"""
Custom Alpha Synapse step function.

Implements a second-order (alpha-function) synapse kernel:

    dh/dt = -h/tau + spike * weight * scale / tau
    dI/dt = -I/tau + h

Discretized (Euler):
    h(t+dt) = h(t) * (1 - dt/tau) + spike * weight * scale / tau
    I(t+dt) = I(t) * (1 - dt/tau) + h(t) * dt

Hyperparameters (in order):
    [0] weight           - Synaptic weight
    [1] synaptic_delay   - Transmission delay (unused in this simple version)
    [2] scale            - Scaling factor
    [3] tau              - Time constant for the alpha kernel (s)

Internal state (in order):
    [0] I_synapse        - Output synaptic current (read by post-soma)
    [1] h                - Hidden state variable
"""

import cupy as cp
from cupyx import jit

from superneuroabm.step_functions.synapse.util import get_soma_spike


@jit.rawkernel(device="cuda")
def alpha_synapse_step_func(
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

    # Read hyperparameters
    weight = synapse_params[agent_index][0]
    synaptic_delay = synapse_params[agent_index][1]
    scale = synapse_params[agent_index][2]
    tau = synapse_params[agent_index][3]

    # Pre/post soma indices
    pre_soma_index = locations[agent_index][0]
    post_soma_index = locations[agent_index][1]

    # Get pre-synaptic spike
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

    # Read internal state
    I_synapse = internal_state[agent_index][0]
    h = internal_state[agent_index][1]

    # Alpha synapse dynamics
    h = h * (1 - dt / tau) + spike * weight * scale / tau
    I_synapse = I_synapse * (1 - dt / tau) + h * dt

    # Write back
    internal_state[agent_index][0] = I_synapse
    internal_state[agent_index][1] = h

    buffer_idx = t_current % len(internal_states_buffer[agent_index])
    internal_states_buffer[agent_index][buffer_idx][0] = I_synapse
    internal_states_buffer[agent_index][buffer_idx][1] = h
