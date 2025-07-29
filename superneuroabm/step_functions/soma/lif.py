"""
LIF Neuron and weighted synapse step functions for spiking neural networks

"""

from cupyx import jit


@jit.rawkernel(device="cuda")
def lif_soma_step_func(  # NOTE: update the name to soma_step_func from neuron_step_func
    tick,
    agent_index,
    globals,
    agent_ids,
    breeds,
    locations,
    neuron_params,  # k, vth, C, a, b,
    internal_state,  # v, u
    synapse_history,  # Synapse delay
    input_spikes_tensor,  # input spikes
    output_spikes_tensor,
    internal_states_buffer,
):
    synapse_ids = locations[agent_index]  # network location is defined by neighbors

    I_synapse = 0.0
    for i in range(len(synapse_ids)):

        synapse_index = -1  # synapse index local to the current mpi rank
        i = 0
        while i < len(agent_ids) and agent_ids[i] != synapse_ids[0]:
            i += 1
        if i < len(agent_ids):
            synapse_index = i
            I_synapse += internal_state[synapse_index][0]

    # Get the current time step value:
    t_current = int(tick)  # Check if tcount is needed or if we ca use this directly.
    dt = globals[0]  # time step size
    I_bias = globals[1]  # bias current

    # NOTE: neuron_params would need to as long as the max number of params in any spiking neuron model
    # Neuron Parameter
    C = neuron_params[agent_index][0]  # membrane capacitance
    R = neuron_params[agent_index][1]  # Leak resistance
    vthr = neuron_params[agent_index][2]  # spike threshold
    tref = neuron_params[agent_index][3]  # refractory period
    vrest = neuron_params[agent_index][4]  # resting potential
    vreset = neuron_params[agent_index][5]  # reset potential
    tref_allows_integration = neuron_params[agent_index][
        6
    ]  # whether to allow integration during refractory period
    I_in = neuron_params[agent_index][7]  # input current
    # vreset = neuron_params[agent_index][8]
    # I_in = neuron_params[agent_index][9]

    # NOTE: size of internal_state would need to be set as the maximum possible state varaibles of any spiking neuron
    # Internal state variables
    v = internal_state[agent_index][0]  # membrane potential
    tcount = internal_state[agent_index][1]  # time count from the start of the simulation
    tlast = internal_state[agent_index][2]  # last spike time

    # Calculate the membrane potential update
    dv = (vrest - v) / (R * C) + (I_synapse*1e-5 + I_bias + I_in) / C

    v +=  (dv*dt) if ((dt * tcount) > (tlast + tref)) or tref_allows_integration else 0.0

    s = 1 * (v >= vthr) and (
        dt * tcount > tlast + tref
    )  # output spike only happens if the membrane potential exceeds the threshold and the neuron is not in refractory period.
    tlast = tlast * (1 - s) + dt * tcount * s
    v = v * (1 - s) + vreset * s  # If spiked, reset membrane potential

    internal_state[agent_index][0] = v
    internal_state[agent_index][1] += 1
    internal_state[agent_index][2] = tlast
    internal_state[agent_index][3] = I_synapse  # Update time count

    output_spikes_tensor[agent_index][t_current] = s
    internal_states_buffer[agent_index][t_current][0] = internal_state[agent_index][0]
    internal_states_buffer[agent_index][t_current][1] = internal_state[agent_index][1]
    internal_states_buffer[agent_index][t_current][2] = internal_state[agent_index][2]
    internal_states_buffer[agent_index][t_current][3] = internal_state[agent_index][3]
