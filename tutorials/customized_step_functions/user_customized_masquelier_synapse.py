"""
user_customized_masquelier_synapse -- a delivery-only afferent synapse, tutorial-owned.

What a synapse does in this network is small: when its afferent fires, add the weight
to the output neuron's summed EPSP ``x``. The EPSP's time course is the soma's business
(``tausyn`` in user_customized_masquelier_soma.py), so the synapse keeps **no per-tick
state** and can be *event-woken*: with ``synapse_stepping="event"`` (the default) this
kernel runs only on the ticks on which its afferent spiked. The 2,000 synapses of the
network therefore cost O(input spikes) per tick, not O(synapses).

The kernel reads its input event, and if there is one deposits ``spike * scale * weight``
into channel 0 of the post soma's inbox. The soma reads the inbox on the next tick.
``get_soma_spike`` handles both cases of a presynaptic source: a soma on the same model
(``pre_soma_index >= 0``) and an external input (``pre_soma_id=-1`` at creation, which is
what every afferent here is): then ``input_spikes_tensor[agent_index]`` holds the
``[tick, value]`` stamp of the spike injected with ``model.add_spikes`` for this tick.

This is a copy of the built-in ``delivery_synapse`` kernel minus its ``channel``
hyperparameter (one channel is enough here), kept in ``customized_step_functions/`` so
the synapse can be changed without editing the package: a transmission delay, a
short-term depression, a device model of the weight all go here.

Registration
------------
Before ``setup()``::

    model.register_synapse_type(
        name="user_customized_masquelier_synapse",
        step_func=user_customized_masquelier_synapse,
        step_func_path=Path("customized_step_functions/user_customized_masquelier_synapse.py").resolve(),
        event_capable=True,
    )

``event_capable=True`` is the promise that the kernel keeps no per-tick state; the model
refuses a synapse type without it in event stepping. A synapse kernel must end its
parameter list with ``synaptic_input, synaptic_input_idx``: that is how the model knows
it delivers into the soma inbox.

Hyperparameters -- the YAML key order *is* the index order read here:

    [0] weight          synaptic weight (pinned: the learning rule writes index 0)
    [1] synaptic_delay  pinned slot, unused (the inbox gives a fixed one-tick delay)
    [2] scale           multiplier on the delivered weight (deltax = 1 in the paper)

Internal states: none are used; the config declares one placeholder slot.
"""

from cupyx import jit

from superneuroabm.step_functions.synapse.util import get_soma_spike
from superneuroabm.step_functions.synaptic_input import deliver_synaptic_current


@jit.rawkernel(device="cuda")
def user_customized_masquelier_synapse(
    tick,
    agent_index,
    dt,
    I_bias,
    agent_ids,
    breeds,
    locations,
    synapse_params,
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

    # locations[agent_index] = [pre_soma_index, post_soma_index]; the pre slot is -1
    # for an external afferent. The post slot is optional in general (an adjacency
    # file may list only the pre), and deliver_synaptic_current ignores a -1 target.
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
    deliver_synaptic_current(synaptic_input, synaptic_input_idx, post_soma_index, 0,
                             spike * scale * weight)
