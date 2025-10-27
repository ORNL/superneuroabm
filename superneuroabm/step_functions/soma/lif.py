"""
LIF Neuron and weighted synapse step functions for spiking neural networks

"""

from cupyx import jit
import cupy as cp


@jit.rawkernel(device="cuda")
def lif_soma_step_func(  # NOTE: update the name to soma_step_func from neuron_step_func
    tick,
    agent_index,
    globals,
    agent_ids,
    breeds,
    locations,
    connectivity,
    neuron_params,  # k, vth, C, a, b,
    learning_params,
    internal_state,  # v, u
    internal_learning_state,
    synapse_history,  # Synapse delay
    input_spikes_tensor,  # input spikes
    output_spikes_tensor,
    internal_states_buffer,
    internal_learning_states_buffer,
):
    synapse_indices = locations[agent_index]  # Now contains local indices instead of IDs

    I_synapse = 0.0

    # synapse_indices now contains pre-computed local indices (converted in SAGESim)
    # No linear search needed!
    for i in range(len(synapse_indices)):
        synapse_index = int(synapse_indices[i])
        if synapse_index >= 0 and not cp.isnan(synapse_indices[i]):
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
    scaling_factor = neuron_params[agent_index][
        8
    ]  # scaling factor for synaptic current
    # vreset = neuron_params[agent_index][8]
    # I_in = neuron_params[agent_index][9]

    # NOTE: size of internal_state would need to be set as the maximum possible state varaibles of any spiking neuron
    # Internal state variables
    v = internal_state[agent_index][0]  # membrane potential
    tcount = internal_state[agent_index][
        1
    ]  # time count from the start of the simulation
    tlast = internal_state[agent_index][2]  # last spike time

    # Calculate the membrane potential update
    dv = (vrest - v) / (R * C) + (I_synapse * scaling_factor + I_bias + I_in) / C

    v += (
        (dv * dt)
        if ((dt * tcount) > (tlast + tref)) or tref_allows_integration
        else 0.0
    )

    #if tlast > 0 else 1 # output spike only happens if the membrane potential exceeds the threshold and the neuron is not in refractory period.
    s = 1.0 * ((v >= vthr) and (( dt * tcount > tlast + tref) if tlast > 0 else True))


    tlast = tlast * (1 - s) + dt * tcount * s
    v = v * (1 - s) + vreset * s  # If spiked, reset membrane potential

    internal_state[agent_index][0] = v
    internal_state[agent_index][1] += 1
    internal_state[agent_index][2] = tlast

    output_spikes_tensor[agent_index][t_current] = s

    # Safe buffer indexing: use modulo to prevent out-of-bounds access
    # When tracking is disabled, buffer length is 1, so t_current % 1 = 0 always
    buffer_idx = t_current % len(internal_states_buffer[agent_index])
    internal_states_buffer[agent_index][buffer_idx][0] = v
    internal_states_buffer[agent_index][buffer_idx][1] = internal_state[agent_index][1] + 1
    internal_states_buffer[agent_index][buffer_idx][2] = tlast
 