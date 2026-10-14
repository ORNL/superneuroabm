"""
Izhikevich Neuron and weighted synapse step functions for spiking neural networks

"""

import cupy as cp
from cupyx import jit


@jit.rawkernel(device="cuda")
def izh_soma_step_func(
    tick,
    agent_index,
    dt,
    I_bias,
    agent_ids,
    breeds,
    locations,
    neuron_params,  # [a, b, c, d, v_peak, I_in]
    learning_params,
    internal_states,  # [v, u]
    learning_internal_states,
    synapse_history,
    input_spikes_tensor,
    output_spikes_tensor,
    internal_states_buffer,
    learning_internal_states_buffer,
):
    synapse_indices = locations[agent_index]
    I_synapse = 0.0

    # synapse_indices now contains pre-computed local indices (converted in SAGESim)
    # No linear search needed!
    for i in range(len(synapse_indices)):
        synapse_index = int(synapse_indices[i])
        if synapse_index >= 0:
            I_synapse += internal_states[synapse_index][0]

    p = neuron_params[agent_index]
    a = p[0]
    b = p[1]
    c = p[2]
    d = p[3]
    v_peak = p[4]
    I_in = p[5]

    v_old = internal_states[agent_index][0]
    u_old = internal_states[agent_index][1]
    dt_ms = dt * 1.0e3
    I_total = I_synapse + I_bias + I_in

    # Canonical Izhikevich model in mV/ms and canonical current units:
    # dv/dt = 0.04*v^2 + 5*v + 140 - u + I
    # du/dt = a*(b*v - u)
    v_new = v_old + dt_ms * (
        0.04 * v_old * v_old + 5.0 * v_old + 140.0 - u_old + I_total
    )
    # Figure1.m updates u after v and uses the new v in this Euler step.
    u_new = u_old + dt_ms * a * (b * v_new - u_old)

    s = 0
    # The published figure1.m source uses the strict test V > 30.
    if v_new > v_peak:
        s = 1
        v_new = c
        u_new = u_new + d

    internal_states[agent_index][0] = v_new
    internal_states[agent_index][1] = u_new

    t_current = int(tick)
    output_spikes_tensor[agent_index][t_current % 2] = s

    buffer_idx = t_current % len(internal_states_buffer[agent_index])
    internal_states_buffer[agent_index][buffer_idx][0] = v_new
    internal_states_buffer[agent_index][buffer_idx][1] = u_new
