"""
Fused LIF neuron and heterogeneous synapse step function.

Only neurons are agents. Synapses are stored as CSR edge data.

Supported synapse types
-----------------------
0: Weighted instantaneous synapse
1: Single-exponential synapse
2: Bi-exponential synapse
"""

from cupyx import jit
import cupy as cp


SYNAPSE_WEIGHTED = 0
SYNAPSE_SINGLE_EXP = 1
SYNAPSE_BI_EXP = 2


@jit.rawkernel(device="cuda")
def lif_fused_synapse_step_func(
    tick,
    agent_index,
    dt,
    I_bias,

    # Existing agent arrays
    agent_ids,
    breeds,
    locations, #We don't need this

    # Neuron data
    neuron_params,
    learning_params,
    internal_states, #Only for neuron agents
    learning_internal_states,

    # CSR synapse connectivity
    incoming_row_ptr, #This is an array of lenths num_neurons + 1, where each entry is the index of the first synapse for that neuron in the pre_neuron_indices and synapse_types arrays
    pre_neuron_indices, #Flat array storing the presynaptic neuron for every edge
    synapse_types, #Flat array storing one type code for each edge

    # Synapse data
    synapse_params, #weigth, delay, scale, decay_fall, decay_rise, normalization one in each column for each synapse
    synapse_states, #Dynamic state of each synapse, one row per synapse, columns depend on synapse type

    # Spike history
    input_spikes_tensor, #Spikes from external inputs
    output_spikes_tensor,

    # Output buffers
    internal_states_buffer,
    learning_internal_states_buffer,
):
    """
    neuron_params[neuron]
    ---------------------
    0: C
    1: R
    2: v_threshold
    3: refractory_period
    4: v_rest
    5: v_reset
    6: integrate_during_refractory
    7: constant_input_current
    8: synaptic_current_scaling

    internal_states[neuron]
    -----------------------
    0: membrane potential
    1: timestep count
    2: last spike time

    synapse_params[synapse]
    ------------------------
    0: weight
    1: delay in integer timesteps
    2: scale
    3: fall decay factor = exp(-dt / tau_fall)
    4: rise decay factor = exp(-dt / tau_rise)
    5: bi-exponential normalization factor

    synapse_states[synapse]
    ------------------------
    0: fall/current state
    1: rise state

    output_spikes_tensor[neuron, history_slot]
    -------------------------------------------
    Ring buffer containing neuron spike history.
    """

    t_current = int(tick)

    # ---------------------------------------------------------
    # Update incoming synapses
    # ---------------------------------------------------------

    I_synapse = 0.0

    edge_start = int(incoming_row_ptr[agent_index])
    edge_end = int(incoming_row_ptr[agent_index + 1])
    #The number of incoming connections is: {edge_end - edge_start}
    spike_history_length = len(output_spikes_tensor[agent_index]) #This is to include the delay in the spike history, which is a ring buffer of length delay + 1

    for edge_index in range(edge_start, edge_end):
        pre_index = int(pre_neuron_indices[edge_index])
        synapse_type = int(synapse_types[edge_index])

        weight = synapse_params[edge_index][0]
        delay_steps = int(synapse_params[edge_index][1])
        scale = synapse_params[edge_index][2]

        # A fused single-kernel implementation must use spikes from a
        # previous timestep. Enforce a minimum delay of one timestep.
        if delay_steps < 1:
            delay_steps = 1

        delayed_tick = t_current - delay_steps

        pre_spike = 0.0

        if delayed_tick >= 0:
            history_index = delayed_tick % spike_history_length
            pre_spike = output_spikes_tensor[pre_index][history_index]

        # -----------------------------------------------------
        # Type 0: weighted instantaneous synapse
        # -----------------------------------------------------
        if synapse_type == SYNAPSE_WEIGHTED:
            I_synapse += scale * weight * pre_spike

        # -----------------------------------------------------
        # Type 1: single-exponential synapse
        #
        # state(t+dt) = decay * state(t) + weight * spike
        # I_syn = scale * state
        # -----------------------------------------------------
        elif synapse_type == SYNAPSE_SINGLE_EXP:
            decay_fall = synapse_params[edge_index][3]

            synaptic_state = synapse_states[edge_index][0]

            synaptic_state = (
                synaptic_state * decay_fall
                + weight * pre_spike
            )

            synapse_states[edge_index][0] = synaptic_state

            I_synapse += scale * synaptic_state

        # -----------------------------------------------------
        # Type 2: bi-exponential synapse
        #
        # fall(t+dt) = decay_fall * fall(t) + weight * spike
        # rise(t+dt) = decay_rise * rise(t) + weight * spike
        #
        # I_syn = scale * normalization * (fall - rise)
        # -----------------------------------------------------
        elif synapse_type == SYNAPSE_BI_EXP:
            decay_fall = synapse_params[edge_index][3]
            decay_rise = synapse_params[edge_index][4]
            normalization = synapse_params[edge_index][5]

            fall_state = synapse_states[edge_index][0]
            rise_state = synapse_states[edge_index][1]

            event = weight * pre_spike

            fall_state = fall_state * decay_fall + event
            rise_state = rise_state * decay_rise + event

            synapse_states[edge_index][0] = fall_state
            synapse_states[edge_index][1] = rise_state

            I_synapse += (
                scale
                * normalization
                * (fall_state - rise_state)
            )

    # ---------------------------------------------------------
    # Read neuron parameters
    # ---------------------------------------------------------

    C = neuron_params[agent_index][0]
    R = neuron_params[agent_index][1]
    vthr = neuron_params[agent_index][2]
    tref = neuron_params[agent_index][3]
    vrest = neuron_params[agent_index][4]
    vreset = neuron_params[agent_index][5]

    tref_allows_integration = (
        neuron_params[agent_index][6] != 0.0
    )

    I_in = neuron_params[agent_index][7]
    synaptic_scaling = neuron_params[agent_index][8]

    # ---------------------------------------------------------
    # Read neuron state
    # ---------------------------------------------------------

    v = internal_states[agent_index][0]
    timestep_count = internal_states[agent_index][1]
    tlast = internal_states[agent_index][2]

    current_time = dt * timestep_count

    outside_refractory = (
        tlast < 0.0
        or current_time > tlast + tref
    )

    # ---------------------------------------------------------
    # LIF integration
    # ---------------------------------------------------------

    if outside_refractory or tref_allows_integration:
        total_current = (
            synaptic_scaling * I_synapse
            + I_bias
            + I_in
        )

        dv = (
            (vrest - v) / (R * C)
            + total_current / C
        )

        v += dt * dv

    # A spike is allowed only outside the refractory period.
    spike = 0.0

    if outside_refractory and v >= vthr:
        spike = 1.0
        v = vreset
        tlast = current_time

    # ---------------------------------------------------------
    # Store neuron state
    # ---------------------------------------------------------

    timestep_count += 1.0

    internal_states[agent_index][0] = v
    internal_states[agent_index][1] = timestep_count
    internal_states[agent_index][2] = tlast

    # output_spikes_tensor is now a ring buffer rather than a
    # two-entry ping-pong buffer.
    output_slot = t_current % spike_history_length
    output_spikes_tensor[agent_index][output_slot] = spike

    # ---------------------------------------------------------
    # Optional state tracking
    # ---------------------------------------------------------

    neuron_buffer_length = len(
        internal_states_buffer[agent_index]
    )

    buffer_index = t_current % neuron_buffer_length

    internal_states_buffer[agent_index][buffer_index][0] = v
    internal_states_buffer[agent_index][buffer_index][1] = timestep_count
    internal_states_buffer[agent_index][buffer_index][2] = tlast

    # Store total synaptic current if a fourth buffer field exists.
    if len(internal_states_buffer[agent_index][buffer_index]) > 3:
        internal_states_buffer[agent_index][buffer_index][3] = I_synapse