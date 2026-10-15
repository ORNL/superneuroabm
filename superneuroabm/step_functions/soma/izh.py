"""
Izhikevich soma, canonical form (Izhikevich 2003/2004, figure1.m):

    dv/dt = 0.04 v^2 + 5 v + 140 - u + I        (mV/ms, I in the paper's current units)
    du/dt = a (b v - u)
    if v > v_peak: v = c, u = u + d, spike

Forward Euler with the paper's step (dt in ms inside; `dt` arrives in seconds), u
updated after v with the new v, strict `v > v_peak` test. The 20 configs A-T in
component_base_config.yaml are the paper's figure 1 panels; the commented `stimulus`
line of each is the drive to pass as I_bias (or I_in) to reproduce the panel.

Synaptic input comes from the soma inbox through one exponential filter per channel
(step_functions/synaptic_input.synaptic_current): a delivery synapse's weight is added
to the current I in the paper's units; `tau_syn_c == dt` makes it a one-tick pulse.

Hyperparameters (neuron_params): [0] a, [1] b, [2] c, [3] d, [4] v_peak, [5] I_in,
[6] dt_s (the panel's original Euler step, informational; the model's dt is the global
`dt`), [7 + k] tau_syn_k.
Internal state: [0] v, [1] u, [2 + k] I_syn_k. The history buffer records the summed
synaptic current at [2].
"""

import cupy as cp
from cupyx import jit

from superneuroabm.step_functions.synaptic_input import synaptic_current


@jit.rawkernel(device="cuda")
def izh_soma_step_func(
    tick,
    agent_index,
    dt,
    I_bias,
    agent_ids,
    breeds,
    locations,
    neuron_params,  # [a, b, c, d, v_peak, I_in, dt_s, tau_syn_0, tau_syn_1, ...]
    learning_params,
    internal_states,  # [v, u, I_syn_0, I_syn_1, ...]
    learning_internal_states,
    synapse_history,
    input_spikes_tensor,
    output_spikes_tensor,
    internal_states_buffer,
    learning_internal_states_buffer,
    synaptic_input,
    synaptic_input_idx,
):
    t_current = int(tick)

    I_synapse = synaptic_current(agent_index, dt, neuron_params, 7, internal_states, 2,
                                 synaptic_input, synaptic_input_idx)

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

    output_spikes_tensor[agent_index][t_current % 2] = s

    buffer_idx = t_current % len(internal_states_buffer[agent_index])
    internal_states_buffer[agent_index][buffer_idx][0] = v_new
    internal_states_buffer[agent_index][buffer_idx][1] = u_new
    internal_states_buffer[agent_index][buffer_idx][2] = I_synapse
