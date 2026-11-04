"""
user_customized_brunel_synapse -- a delivery-only synapse, tutorial-owned.

What it shows (tutorial 07, section 3): everything a synapse step function has to do. When its
presynaptic soma fired on the previous tick, or an input spike was injected for this tick, it
adds ``spike * weight`` into one channel of its post soma's inbox. The soma reads the inbox on
the next tick and owns everything that happens to the value afterwards.

It keeps no per-tick state, so it can be *event-woken*: with ``synapse_stepping="event"`` (the
default) this kernel runs only on the ticks when it has a spike to deliver.

An inhibitory synapse is one with a negative weight. The kernel has no sign logic, and the
channel has nothing to do with the sign: a channel selects a time constant on the soma.

This is the built-in ``delivery_synapse`` minus its ``scale`` hyperparameter. A real model that
needs nothing more should use the built-in; it is copied here as the smallest complete example.

Registration
------------
Before ``setup()``::

    model.register_synapse_type(
        name="user_customized_brunel_synapse",
        step_func=user_customized_brunel_synapse,
        step_func_path=Path("customized_step_functions/user_customized_brunel_synapse.py").resolve(),
        event_capable=True,
    )

``event_capable=True`` is the promise that the kernel keeps no per-tick state; the model refuses
a synapse type without it in event stepping. A synapse kernel must end its parameter list with
``synaptic_input, synaptic_input_idx``.

Hyperparameters -- the YAML key order *is* the index order read here:

    [0] weight          synaptic weight (A); pinned, must be the first key
    [1] synaptic_delay  pinned, must be the second key, and not read by any kernel
    [2] channel         inbox slot this synapse delivers into

The delay between a spike and its arrival is fixed by the framework at two ticks, whatever
``synaptic_delay`` says (tutorial 07, section 3).

Internal states: none are used; the config declares one placeholder slot.

``locations[agent_index]`` is ``[pre_soma_index, post_soma_index]``, already local indices; the
pre slot is -1 for an external input (``pre_soma_id=-1`` at creation).
"""

from cupyx import jit

from superneuroabm.step_functions.synapse.util import get_soma_spike
from superneuroabm.step_functions.synaptic_input import deliver_synaptic_current


@jit.rawkernel(device="cuda")
def user_customized_brunel_synapse(
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

    # === Read synapse parameters, by YAML position ===
    weight = synapse_params[agent_index][0]  # A (positive for excitatory, negative for inhibitory)
    channel = int(synapse_params[agent_index][2])  # index 1 is synaptic_delay

    # === Read locations ===
    _nbrs = locations[agent_index]
    pre_soma_index = int(_nbrs[0])
    post_soma_index = -1
    if len(_nbrs) > 1:
        post_soma_index = int(_nbrs[1])

    # === Read presynaptic spike ===
    # get_soma_spike handles both a soma on the model (pre_soma_index >= 0, its spike of the
    # previous tick) and an external input (pre_soma_index == -1, the spike injected for this tick)
    spike = get_soma_spike(
        tick, agent_index, dt, I_bias, agent_ids, pre_soma_index,
        t_current, input_spikes_tensor, output_spikes_tensor,
    )

    # === Deliver synaptic current if there is a spike ===
    if spike == 0.0:
        return

    synaptic_current = spike * weight

    # Adds the current into the post soma's inbox row, in slot `channel`
    deliver_synaptic_current(
        synaptic_input, synaptic_input_idx, post_soma_index, channel,
        synaptic_current
    )
