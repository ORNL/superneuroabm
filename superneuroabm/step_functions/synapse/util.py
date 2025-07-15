import math
import cupy as cp
from cupyx import jit

@jit.rawkernel(device="cuda")
def get_pre_soma_spike(
    tick,
    agent_index,
    globals,
    agent_ids,
    pre_soma_id,
    t_current,
    input_spikes_tensor,  # input spikes
    output_spikes_tensor,
):
    t_current = int(tick)

    if not cp.isnan(pre_soma_id):
        i = 0
        while i < len(agent_ids) and agent_ids[i] != pre_soma_id:
            i += 1
        spike = output_spikes_tensor[i][t_current]
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
    
    # print(f"Pre-soma spike for agent {agent_index} at time {t_current}: {spike}")
    return spike