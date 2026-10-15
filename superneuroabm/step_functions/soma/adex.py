"""
Adaptive exponential integrate-and-fire soma (AdEx; Neuronal Dynamics, Ch. 6, Table 6.1):

    C dV/dt = -gL (V - EL) + gL DeltaT exp((V - VT) / DeltaT) - w + I
    tau_w dw/dt = a (V - EL) - w
    if V >= V_peak: V = V_reset, w = w + b, spike

Units mV, pA, ms (conductance nS, capacitance pF); forward Euler with both derivatives
taken at the beginning of the step; `dt` arrives in seconds. The configs in
component_base_config.yaml are the table's firing patterns; each commented `stimulus`
line is the step current (pA) that reproduces the pattern when passed as I_bias/I_in.

Synaptic input comes from the soma inbox through one exponential filter per channel
(step_functions/synaptic_input.synaptic_current), in pA: a delivery synapse's weight is
the current one presynaptic spike injects; `tau_syn_c == dt` makes it a one-tick pulse.

Hyperparameters (neuron_params): [0] C, [1] gL, [2] E_L, [3] V_T, [4] Delta_T, [5] a,
[6] tau_w, [7] b, [8] V_reset, [9] V_peak, [10] I_in, [11 + k] tau_syn_k.
Internal state: [0] v (mV), [1] w (pA), [2 + k] I_syn_k. The history buffer records the
summed synaptic current at [2].
"""

import cupy as cp
from cupyx import jit

from superneuroabm.step_functions.synaptic_input import synaptic_current


@jit.rawkernel(device="cuda")
def adex_soma_step_func(
    tick,
    agent_index,
    dt,
    I_bias,
    agent_ids,
    breeds,
    locations,
    neuron_params,  # [C, gL, EL, VT, DeltaT, a, tau_w, b, Vreset, Vpeak, I_in, tau_syn_0, tau_syn_1, ...]
    learning_params,
    internal_states,  # [v_mV, w_pA, I_syn_0, I_syn_1, ...]
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

    I_synapse = synaptic_current(agent_index, dt, neuron_params, 11, internal_states, 2,
                                 synaptic_input, synaptic_input_idx)

    p = neuron_params[agent_index]
    C = p[0]
    gL = p[1]
    EL = p[2]
    VT = p[3]
    DeltaT = p[4]
    a = p[5]
    tau_w = p[6]
    b = p[7]
    V_reset = p[8]
    V_peak = p[9]
    I_in = p[10]

    v_old = internal_states[agent_index][0]
    w_old = internal_states[agent_index][1]
    dt_ms = dt * 1.0e3
    I_total = I_synapse + I_bias + I_in

    # AdEx equations (Euler; both derivatives use the beginning-of-step state):
    # C*dV/dt = -gL*(V-EL) + gL*DeltaT*exp((V-VT)/DeltaT) - w + I
    # tau_w*dw/dt = a*(V-EL) - w
    exp_term = cp.exp((v_old - VT) / DeltaT)
    dVdt = (
        -gL * (v_old - EL)
        + gL * DeltaT * exp_term
        - w_old
        + I_total
    ) / C
    dwdt = (a * (v_old - EL) - w_old) / tau_w

    v_new = v_old + dt_ms * dVdt
    w_new = w_old + dt_ms * dwdt

    s = 0
    if v_new >= V_peak:
        s = 1
        v_new = V_reset
        w_new = w_new + b

    internal_states[agent_index][0] = v_new
    internal_states[agent_index][1] = w_new

    output_spikes_tensor[agent_index][t_current % 2] = s

    buffer_idx = t_current % len(internal_states_buffer[agent_index])
    internal_states_buffer[agent_index][buffer_idx][0] = v_new
    internal_states_buffer[agent_index][buffer_idx][1] = w_new
    internal_states_buffer[agent_index][buffer_idx][2] = I_synapse
