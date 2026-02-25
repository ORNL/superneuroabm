import cupy as cp
from cupyx import jit
from superneuroabm.step_functions.synapse.util import get_soma_spike


@jit.rawkernel(device="cuda")
def exp_pair_wise_stdp_memristive(
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

    # =========================
    # ---- Synapse Params -----
    # =========================
    weight = synapse_params[agent_index][0]
    synaptic_delay = synapse_params[agent_index][1]

    # =========================
    # ---- Learning Params ----
    # =========================
    tau_pre_stdp  = learning_params[agent_index][1]
    tau_post_stdp = learning_params[agent_index][2]
    a_exp_pre     = learning_params[agent_index][3]
    a_exp_post    = learning_params[agent_index][4]

    wmin = learning_params[agent_index][6]
    wmax = learning_params[agent_index][7]

    # =========================
    # ---- Memristor Params ---
    # =========================
    Gmin = 166.0
    Gmax = 466.0

    mu_write    = 3.56
    sigma_write = 8.92
    sigma_read  = 4.06

    step_fraction = 1.0
    alpha = 1.0

    pulse_V = 1.0
    pulse_t = 100e-9
    read_V  = 0.2
    read_t  = 50e-9

    # =========================
    # ---- Internal States ----
    # =========================
    pre_trace  = internal_learning_state[agent_index][0]
    post_trace = internal_learning_state[agent_index][1]
    dW         = internal_learning_state[agent_index][2]

    # =========================
    # ---- Connectivity -------
    # =========================
    pre_soma_index  = locations[agent_index][0]
    post_soma_index = locations[agent_index][1]

    pre_spike = get_soma_spike(
        tick, agent_index, globals,
        agent_ids, pre_soma_index,
        t_current, input_spikes_tensor,
        output_spikes_tensor
    )

    post_spike = get_soma_spike(
        tick, agent_index, globals,
        agent_ids, post_soma_index,
        t_current, input_spikes_tensor,
        output_spikes_tensor
    )

    # =========================
    # ---- STDP ---------------
    # =========================
    pre_trace  = pre_trace  * (1.0 - dt / tau_pre_stdp)  + pre_spike  * a_exp_pre
    post_trace = post_trace * (1.0 - dt / tau_post_stdp) + post_spike * a_exp_post

    dW = pre_trace * post_spike - post_trace * pre_spike

    # =========================
    # ---- Weight → Conductance
    # =========================
    # Linear mapping
    G = Gmin + (weight - wmin) * (Gmax - Gmin) / (wmax - wmin)

    # Target weight
    weight_target = weight + dW
    if weight_target > wmax:
        weight_target = wmax
    if weight_target < wmin:
        weight_target = wmin

    G_target = Gmin + (weight_target - wmin) * (Gmax - Gmin) / (wmax - wmin)

    # =========================
    # ---- Deterministic Update
    # =========================
    delta_det = step_fraction * (G_target - G)

    if delta_det > 0.0:
        gamma = (1.0 - (G - Gmin) / (Gmax - Gmin)) ** alpha
    else:
        gamma = ((G - Gmin) / (Gmax - Gmin)) ** alpha

    delta_det = delta_det * gamma

    # =========================
    # ---- Gaussian Noise -----
    # Box-Muller using simple LCG RNG
    # =========================
    seed = t_current * 1103515245 + agent_index * 12345
    seed = (seed ^ 1234567) & 0xFFFFFFFF

    u1 = (seed & 0xFFFF) / 65535.0
    u2 = ((seed >> 16) & 0xFFFF) / 65535.0

    if u1 < 1e-6:
        u1 = 1e-6

    randn = (-2.0 * cp.log(u1)) ** 0.5 * cp.cos(6.2831853 * u2)

    eps_write = mu_write + sigma_write * randn

    delta_G = delta_det + eps_write

    # =========================
    # ---- Apply + Clip -------
    # =========================
    G_new = G + delta_G

    if G_new > Gmax:
        G_new = Gmax
    if G_new < Gmin:
        G_new = Gmin

    # =========================
    # ---- Read Noise ---------
    # =========================
    eps_read = sigma_read * randn
    G_observed = G_new + eps_read

    # =========================
    # ---- Conductance → Weight
    # =========================
    weight_new = wmin + (G_observed - Gmin) * (wmax - wmin) / (Gmax - Gmin)

    if weight_new > wmax:
        weight_new = wmax
    if weight_new < wmin:
        weight_new = wmin

    synapse_params[agent_index][0] = weight_new

    # =========================
    # ---- Energy Calculation --
    # =========================
    G_avg = 0.5 * (G + G_new) * 1e-6
    I_write = pulse_V * G_avg
    E_write = pulse_V * I_write * pulse_t

    I_read = read_V * (G_new * 1e-6)
    E_read = read_V * I_read * read_t

    internal_state[agent_index][4] += E_write
    internal_state[agent_index][5] += E_read

    # =========================
    # ---- Update Learning ----
    # =========================
    internal_learning_state[agent_index][0] = pre_trace
    internal_learning_state[agent_index][1] = post_trace
    internal_learning_state[agent_index][2] = dW

    buffer_idx = t_current % len(internal_learning_states_buffer[agent_index])
    internal_learning_states_buffer[agent_index][buffer_idx][0] = pre_trace
    internal_learning_states_buffer[agent_index][buffer_idx][1] = post_trace
    internal_learning_states_buffer[agent_index][buffer_idx][2] = dW

    internal_state[agent_index][2] = pre_trace
    internal_state[agent_index][3] = post_trace

    state_buffer_idx = t_current % len(internal_states_buffer[agent_index])
    internal_states_buffer[agent_index][state_buffer_idx][2] = post_spike
    internal_states_buffer[agent_index][state_buffer_idx][3] = post_trace