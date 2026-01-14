import math
import cupy as cp
from cupyx import jit


@jit.rawkernel(device="cuda")
def get_soma_spike(
    tick,
    agent_index,
    globals,
    agent_ids,
    pre_soma_index,  # Used to be pre_soma_id, now it's already an index
    t_current,
    input_spikes_tensor,  # input spikes
    output_spikes_tensor,
):
    """
    Get spike from pre-soma using its local index (already converted by SAGESim).

    Args:
        pre_soma_index: Local index of the pre-synaptic soma (-1 if NaN/external input)
        globals, agent_ids: Kept for signature compatibility, not used anymore

    NOTE: Due to double buffering, this reads from the PREVIOUS tick's spikes.
    Somas write spikes at priority 0, synapses read at priority 1, but the
    write buffer isn't copied to read buffer until after all priorities complete.
    This introduces a 1-tick synaptic delay, which is actually realistic.
    """
    t_current = int(tick)

    if pre_soma_index >= 0:
        # pre_soma_index is already a local index (no search needed!)
        # Read from previous tick due to double buffering
        # At tick 0, there are no previous spikes, so spike will be 0
        if t_current > 0:
            spike = output_spikes_tensor[pre_soma_index][t_current - 1]
        else:
            spike = 0.0
    else:
        spike = 0.0
        spike_buffer_max_len = len(input_spikes_tensor[agent_index])
        i = 0

        # input_spikes_tensor is now flattened as [tick, value, tick, value, ...]
        # Check i+1 < max_len to avoid reading past array bounds!
        while i + 1 < spike_buffer_max_len and not cp.isnan(input_spikes_tensor[agent_index][i]):
            if input_spikes_tensor[agent_index][i] == t_current:      # tick at even index
                spike += input_spikes_tensor[agent_index][i+1]        # value at odd index
            i += 2  # Skip by 2 to move to next tick (not i+1!)
    return spike
