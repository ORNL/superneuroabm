"""
user_customized_double_exp_synapse -- a double-exponential (rise and decay) synapse,
tutorial-owned.

The former ``double_exp_synapse`` kept two currents per synapse, a fast one and a slow
one, and the post soma used their difference. Here the two currents live on the post
soma as two inbox channels, and the synapse only says *how much* goes into each: when
its pre fires it deposits ``+q`` into channel ``channel_decay`` and ``-q`` into channel
``channel_rise``, with ``q = spike * scale * weight``. The soma filters channel c with
its own ``tau_syn_c`` and sums the channels, so with ``tau_syn[channel_decay] = tau_decay``
and ``tau_syn[channel_rise] = tau_rise`` (``tau_rise < tau_decay``) the current it
integrates after one spike is, n ticks after the spike reached the soma,

    I(n) = q * ((1 - dt/tau_decay)**n - (1 - dt/tau_rise)**n)

which is 0 on the first tick, rises, and decays with ``tau_decay``. Its peak is below
``q``; set ``scale`` to ``1 / peak`` for a peak of ``weight`` (tutorial 02 computes it).

The time constants belong to the post soma, so every double-exp synapse onto one soma
shares them, and a single-exp synapse with time constant ``tau_decay`` can deliver into
the same ``channel_decay``. A soma that also receives a third time constant needs
``NeuromorphicModel(synaptic_channels=3)``.

The kernel keeps **no per-tick state**, so it is event-capable: with
``synapse_stepping="event"`` (the default) it runs only on the ticks after its pre
fired or an input event arrived, like the built-in ``delivery_synapse``, of which it is
a copy with one more deposit.

Registration
------------
Before ``setup()``::

    model.register_synapse_type(
        name="user_customized_double_exp_synapse",
        step_func=user_customized_double_exp_synapse,
        step_func_path=Path("customized_step_functions/user_customized_double_exp_synapse.py").resolve(),
        event_capable=True,
    )

A synapse kernel must end its parameter list with ``synaptic_input, synaptic_input_idx``:
that is how the model knows it delivers into the soma inbox.

Hyperparameters -- the YAML key order *is* the index order read here:

    [0] weight          synaptic weight (pinned: the learning rule writes index 0)
    [1] synaptic_delay  pinned slot, unused (the inbox gives a fixed one-tick delay)
    [2] scale           multiplier on the delivered weight
    [3] channel_decay   inbox slot whose tau_syn is the decay time constant (receives +q)
    [4] channel_rise    inbox slot whose tau_syn is the rise time constant (receives -q)

Both channels must be in 0..synaptic_channels-1. The model checks a hyperparameter named
``channel`` at setup(), not these two: an out-of-range value would index past the inbox row.

Internal states: none are used; the config declares one placeholder slot.
"""

from cupyx import jit

from superneuroabm.step_functions.synapse.util import get_soma_spike
from superneuroabm.step_functions.synaptic_input import deliver_synaptic_current


@jit.rawkernel(device="cuda")
def user_customized_double_exp_synapse(
    tick,
    agent_index,
    dt,
    I_bias,
    agent_ids,
    breeds,
    locations,
    synapse_params,  # weight, synaptic_delay, scale, channel_decay, channel_rise
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
    # for an external input, and deliver_synaptic_current ignores a -1 target.
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
    channel_decay = int(synapse_params[agent_index][3])
    channel_rise = int(synapse_params[agent_index][4])
    q = spike * scale * weight
    deliver_synaptic_current(synaptic_input, synaptic_input_idx, post_soma_index, channel_decay, q)
    deliver_synaptic_current(synaptic_input, synaptic_input_idx, post_soma_index, channel_rise, -q)
