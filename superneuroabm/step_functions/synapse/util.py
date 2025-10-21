import math
import cupy as cp
from cupyx import jit


@jit.rawkernel(device="cuda")
def get_soma_spike(
    tick,
    agent_index,
    globals,
    agent_ids,
    pre_soma_id,
    t_current,
    input_spikes_tensor,  # input spikes
    output_spikes_tensor,
):
    """
    Get spike from pre-soma using its AGENT ID.
    Converts agent ID to local index by searching through agent_ids array.

    NOTE: Due to double buffering, this reads from the PREVIOUS tick's spikes.
    Somas write spikes at priority 0, synapses read at priority 1, but the
    write buffer isn't copied to read buffer until after all priorities complete.
    This introduces a 1-tick synaptic delay, which is actually realistic.
    """
    t_current = int(tick)

    if not cp.isnan(pre_soma_id):
        # Convert agent ID to local index
        i = 0
        while i < len(agent_ids) and agent_ids[i] != pre_soma_id:
            i += 1

        # Read from previous tick due to double buffering
        # At tick 0, there are no previous spikes, so spike will be 0
        if t_current > 0:
            spike = output_spikes_tensor[i][t_current - 1]
        else:
            spike = 0.0
    else:
        spike = 0.0
        spike_buffer_max_len = len(input_spikes_tensor[agent_index])
        i = 0
        while i < spike_buffer_max_len and not cp.isnan(
            input_spikes_tensor[agent_index][i][0]
        ):
            if input_spikes_tensor[agent_index][i][0] == t_current:
                spike += input_spikes_tensor[agent_index][i][
                    1
                ]  # TODO: check if we need analog values for spikes
            i += 1

    return spike
