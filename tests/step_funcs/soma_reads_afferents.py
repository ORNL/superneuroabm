"""A soma kernel that reads its incoming synapses: the case that keeps a soma's neighbour list.

It records how many synapses it has and the sum of their weights (hyperparameter 0 of each
synapse row), which is what a soma that normalises its incoming weights would start from.
A soma whose kernel does not use ``locations`` gets no list at all
(NeuromorphicModel._soma_keeps_neighbour_list).
"""

from cupyx import jit


@jit.rawkernel(device="cuda")
def soma_reads_afferents(
    tick,
    agent_index,
    dt,
    I_bias,
    agent_ids,
    breeds,
    locations,
    neuron_params,
    learning_params,
    internal_states,
    learning_internal_states,
    synapse_history,
    input_spikes_tensor,
    output_spikes_tensor,
    internal_states_buffer,
    learning_internal_states_buffer,
    synaptic_input,
    synaptic_input_idx,
):
    _nbrs = locations[agent_index]
    total = 0.0
    for i in range(len(_nbrs)):
        total += neuron_params[int(_nbrs[i])][0]
    internal_states[agent_index][0] = float(len(_nbrs))
    internal_states[agent_index][1] = total
    output_spikes_tensor[agent_index][int(tick) % 2] = 0.0
