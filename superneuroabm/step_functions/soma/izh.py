"""
Izhikevich Neuron and weighted synapse step functions for spiking neural networks

"""

from cupyx import jit


@jit.rawkernel(device="cuda")
def izh_soma_step_func(
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
    t_current = int(tick)

    dt = globals[0]  # time step size

    # NOTE: neuron_params would need to be as long as the max number of params in any spiking neuron model
    k = neuron_params[agent_index][0]
    vthr = neuron_params[agent_index][1]
    C = neuron_params[agent_index][2]
    a = neuron_params[agent_index][3]
    b = neuron_params[agent_index][4]
    vpeak = neuron_params[agent_index][5]
    vrest = neuron_params[agent_index][6]
    d = neuron_params[agent_index][7]
    vreset = neuron_params[agent_index][8]
    I_in = neuron_params[agent_index][9]

    # From https://www.izhikevich.org/publications/spikes.htm
    # v' = 0.04v^2 + 5v + 140 -u + I
    # u' = a(bv - u)
    # if v=30mV: v = c, u = u + d, spike

    # dv = (k*(internal_state[my_idx]-vrest)*(internal_state[my_idx]-vthr)-u[my_idx]+I) / C
    # internal_state: [0] - v, [1] - u
    # NOTE: size of internal_state would need to be set as the maximum possible state varaibles of any spiking neuron

    v = internal_state[agent_index][0]
    u = internal_state[agent_index][1]

    I_bias = globals[1]  # bias current

    dv = (k * (v - vrest) * (v - vthr) - u + I_synapse + I_bias + I_in) / C
    v = v + dt * dv *1e3

    u += dt * 1e3 * (a * (b * (v - vrest) - u))
    s = 1 * (v >= vthr)  # output spike
    u = u + d * s   # If spiked, update recovery variable
    v = v * (1 - s) + vreset * s  # If spiked, reset membrane potential

    internal_state[agent_index][0] = v
    internal_state[agent_index][1] = u

    output_spikes_tensor[agent_index][t_current] = s
    internal_states_buffer[agent_index][t_current][0] = v
    internal_states_buffer[agent_index][t_current][1] = u
