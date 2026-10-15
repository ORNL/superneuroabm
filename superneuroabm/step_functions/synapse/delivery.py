"""
Delivery-only synapse: no per-tick state.

Reads its pre-soma's spike (or its injected input event); if there is none it is done.
Otherwise it deposits ``spike * scale * weight`` into one channel of the post soma's
inbox. The synaptic kinetics live on the soma (see soma/lif_psc.py), which is NEST's
``iaf_psc_exp`` layout and what makes a synapse cheap enough to be event-woken: with
``NeuromorphicModel(synaptic_input="push", synapse_stepping="event")`` this kernel runs
only for synapses whose pre fired (or received an input event) on the previous tick.

``channel`` picks the inbox slot, i.e. which of the soma's time constants this synapse's
input decays with (NEST's ``receptor_type``): afferents with different time constants
go to different channels of the same soma. 0 <= channel < synaptic_channels; checked
at setup().

Works in clocked mode too (every synapse stepped, idle ones return at once), which is
how the two modes are checked against each other bitwise. Push mode only: in pull mode
there is no current for the soma to read.

Hyperparameters (synapse_params): [weight, synaptic_delay (unused), scale, channel].
"""

from cupyx import jit

from superneuroabm.step_functions.synapse.util import get_soma_spike
from superneuroabm.step_functions.synaptic_input import deliver_synaptic_current


@jit.rawkernel(device="cuda")
def delivery_synapse_step_func(
    tick,
    agent_index,
    dt,
    I_bias,
    agent_ids,
    breeds,
    locations,
    synapse_params,  # weight, synaptic delay, scale, channel
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
    t_current = int(tick)
    # one alias of the neighbour list (the CSR transform rewrites exactly one);
    # the post slot is optional (an adjacency file may list only the pre), and
    # deposit_to_agent ignores a negative target
    _nbrs = locations[agent_index]
    pre_soma_index = _nbrs[0]
    post_soma_index = -1
    if len(_nbrs) > 1:
        post_soma_index = _nbrs[1]

    spike = get_soma_spike(
        tick, agent_index, dt, I_bias, agent_ids, pre_soma_index,
        t_current, input_spikes_tensor, output_spikes_tensor,
    )
    if spike == 0.0:
        return

    weight = synapse_params[agent_index][0]
    scale = synapse_params[agent_index][2]
    channel = int(synapse_params[agent_index][3])
    deliver_synaptic_current(synaptic_input, synaptic_input_idx, post_soma_index, channel,
                             spike * scale * weight)
