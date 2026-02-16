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
        pre_soma_index: Local index of the pre-synaptic soma (-1 for external input)
        globals, agent_ids: Kept for signature compatibility, not used anymore

    NOTE: output_spikes_tensor is NOT double-buffered. Reading t_current-1
    introduces a 1-tick synaptic delay (0.1ms at dt=1e-4), which is
    biologically realistic and negligible at fine dt.
    """
    t_current = int(tick)

    if pre_soma_index >= 0:
        # pre_soma_index is already a local index (no search needed!)
        if t_current > 0:
            spike = output_spikes_tensor[pre_soma_index][t_current - 1]
        else:
            spike = 0.0
    else:
        spike = 0.0
        spike_buffer_max_len = len(input_spikes_tensor[agent_index])

        # Running pointer: offset stored in sentinel value slot (index 1).
        # input_spikes_tensor layout: [-1, offset, tick, value, tick, value, ...]
        # The sentinel tick=-1 never matches t_current; its value slot stores
        # the scan offset so we skip already-processed spikes each tick.
        i = int(input_spikes_tensor[agent_index][1])

        # Phase 1: Advance past stale entries (tick < t_current)
        while i + 1 < spike_buffer_max_len and not cp.isnan(input_spikes_tensor[agent_index][i]) and int(input_spikes_tensor[agent_index][i]) < t_current:
            i += 2
        input_spikes_tensor[agent_index][1] = float(i)  # persist for next call/tick

        # Phase 2: Sum entries at t_current (don't advance offset)
        j = i
        while j + 1 < spike_buffer_max_len and not cp.isnan(input_spikes_tensor[agent_index][j]) and int(input_spikes_tensor[agent_index][j]) == t_current:
            spike += input_spikes_tensor[agent_index][j + 1]
            j += 2
    return spike
