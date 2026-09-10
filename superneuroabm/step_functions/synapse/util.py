import math
import cupy as cp
from cupyx import jit


@jit.rawkernel(device="cuda")
def get_soma_spike(
    tick,
    agent_index,
    dt,
    I_bias,
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
        dt, I_bias: Scalar globals auto-extracted by SAGESim framework
        agent_ids: Kept for signature compatibility, not used anymore

    NOTE: output_spikes_tensor is NOT double-buffered. Reading t_current-1
    introduces a 1-tick synaptic delay (0.1ms at dt=1e-4), which is
    biologically realistic and negligible at fine dt.

    External input (pre_soma_index == -1): input_spikes_tensor[agent_index] is
    [last_delivered_tick, value]. The generated kernel scatters this tick's
    injected events into these rows before priority 0 (see
    NeuromorphicModel._get_extra_kernel_config), so the read is one comparison
    and costs the same at tick 10 and tick 10 million. Spikes injected on the
    same synapse and tick were summed when the event list was built.
    """
    t_current = int(tick)

    if pre_soma_index >= 0:
        # pre_soma_index is already a local index (no search needed!)
        if t_current > 0:
            spike = output_spikes_tensor[pre_soma_index][(t_current - 1) % 2]
        else:
            spike = 0.0
    else:
        if input_spikes_tensor[agent_index][0] == t_current:
            spike = input_spikes_tensor[agent_index][1]
        else:
            spike = 0.0
    return spike
