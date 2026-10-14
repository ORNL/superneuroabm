import cupy as cp
from cupyx import jit


@jit.rawkernel(device="cuda")
def adex_soma_step_func(
    tick,
    agent_index,
    dt,
    I_bias,
    agent_ids,
    breeds,
    locations,
    neuron_params,  # [C, gL, EL, VT, DeltaT, a, tau_w, b, Vreset, Vpeak, I_in]
    learning_params,
    internal_states,  # neuron rows: [v_mV, w_pA]; synapse rows [0] = I_syn_pA
    learning_internal_states,
    synapse_history,
    input_spikes_tensor,
    output_spikes_tensor,
    internal_states_buffer,
    learning_internal_states_buffer,
):
    # Same adjacency/current convention as izh_soma_step_func:
    # locations list synapse-agent indices, not presynaptic neuron indices.
    synapse_indices = locations[agent_index]
    I_synapse = 0.0
    for i in range(len(synapse_indices)):
        synapse_index = int(synapse_indices[i])
        if synapse_index >= 0:
            I_synapse += internal_states[synapse_index][0]

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

    t_current = int(tick)
    output_spikes_tensor[agent_index][t_current % 2] = s

    buffer_idx = t_current % len(internal_states_buffer[agent_index])
    internal_states_buffer[agent_index][buffer_idx][0] = v_new
    internal_states_buffer[agent_index][buffer_idx][1] = w_new