import numpy as np
from cupyx import jit

from superneuroabm.step_functions.synapse.util import get_pre_soma_spike

@jit.rawkernel(device="cuda")
def synapse_single_exp_step_func(
    tick,
    agent_index,
    globals,
    agent_ids,
    breeds,
    locations,
    synapse_params,          # Layout modified to add learning sigmas
    learning_params,         # Adding sigmas for learning noise
    internal_state,          # I_synapse
    internal_learning_state, # Added w_eff, flags, rng state
    synapse_history,         
    input_spikes_tensor,
    output_spikes_tensor,
    internal_states_buffer,
    internal_learning_states_buffer,
):
    # --- tiny device helpers (inline for rawkernel styles) -------------------
    def _xorshift32(state_u32):
        # returns (new_state, uint32)
        x = state_u32 & 0xFFFFFFFF
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= (x >> 17) & 0xFFFFFFFF
        x ^= (x << 5)  & 0xFFFFFFFF
        return x, x

    def _u01_from_u32(u32):
        # map to (0,1); avoid exact 0
        return (u32 + 1.0) * (1.0 / 4294967297.0)

    def _randn(state_u32):
        # Box–Muller using two uniforms
        s, u1_i = _xorshift32(state_u32)
        s, u2_i = _xorshift32(s)
        u1 = _u01_from_u32(u1_i)
        u2 = _u01_from_u32(u2_i)
        # sqrt(-2 ln u1) * cos(2 pi u2)
        r = math.sqrt(-2.0 * math.log(u1))
        z = r * math.cos(6.283185307179586 * u2)
        return s, z  # new_state, standard normal
    # -------------------------------------------------------------------------

    t_current = int(tick)
    dt = globals[0]

    # ---- read params ---------------------------------------------------------
    w_nom          = synapse_params[agent_index][0]
    synaptic_delay = synapse_params[agent_index][1]
    scale          = synapse_params[agent_index][2]
    tau_fall       = synapse_params[agent_index][3]
    tau_rise       = synapse_params[agent_index][4]

    # per-synapse standard deviations for weight noise
    sigma_prog     = synapse_params[agent_index][5] # standard deviation for initial programming
    sigma_stdp     = synapse_params[agent_index][6] # standard deviation for STDP updates

    # ---- persistent learning state ------------------------------------------
    w_eff          = internal_learning_state[agent_index][0]
    is_programmed  = int(internal_learning_state[agent_index][1]) # bool as int
    rng_state_u32  = int(internal_learning_state[agent_index][2])
    needs_reprog   = int(internal_learning_state[agent_index][3])

    # seed rng at first use (deterministic per synapse)
    if rng_state_u32 == 0:
        # simple seed from agent_index (offset avoids zero)
        rng_state_u32 = (1664525 * (agent_index + 1) + 1013904223) & 0xFFFFFFFF
        if rng_state_u32 == 0:
            rng_state_u32 = 123456789

    # initial programming (once)
    if is_programmed == 0:
        rng_state_u32, z = _randn(rng_state_u32)
        # choose additive or multiplicative model;
        # (a) additive: w_eff = w_nom + sigma_prog * z
        # (b) multiplicative: w_eff = w_nom * (1.0 + sigma_prog * z)
        w_eff = w_nom + sigma_prog * z   # Additive, Katie's STDP is multiplicative in the orignal code
        is_programmed = 1
        needs_reprog  = 0

    # reprogram after STDP changed nominal weight
    if needs_reprog == 1:
        rng_state_u32, z = _randn(rng_state_u32)
        # apply STDP programming noise
        w_eff = w_nom + sigma_stdp * z   # or multiplicative form
        needs_reprog = 0

    # store back persistent learning state
    internal_learning_state[agent_index][0] = w_eff
    internal_learning_state[agent_index][1] = is_programmed
    internal_learning_state[agent_index][2] = rng_state_u32
    internal_learning_state[agent_index][3] = needs_reprog

    # ---- spikes and current update (unchanged except weight source) ----------
    location_data = locations[agent_index]
    pre_soma_id = -1 if cp.isnan(location_data[1]) else location_data[0]
    post_soma_id = location_data[0] if cp.isnan(location_data[1]) else location_data[1]

    spike = get_pre_soma_spike(
        tick, agent_index, globals, agent_ids,
        pre_soma_id, t_current, input_spikes_tensor, output_spikes_tensor
    )

    I_synapse = internal_state[agent_index][0]
    # use w_eff (constant between programming events)
    I_synapse = I_synapse * (1 - dt / tau_fall) + spike * scale * w_eff

    internal_state[agent_index][0] = I_synapse
    internal_states_buffer[agent_index][t_current][0] = I_synapse
    internal_states_buffer[agent_index][t_current][1] = spike
    internal_states_buffer[agent_index][t_current][2] = t_current
    internal_states_buffer[agent_index][t_current][3] = pre_soma_id