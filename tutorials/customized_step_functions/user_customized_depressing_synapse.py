"""
user_customized_depressing_synapse -- a synapse with short-term depression, tutorial-owned.

What it shows (tutorial 06): a synapse step function decides *what value* it delivers into
its post soma's channel. The built-in ``delivery_synapse`` always delivers
``spike * scale * weight``; this one delivers less on each spike of a fast train and
recovers during a pause, as in the depression part of the Tsodyks-Markram model:

* ``x`` is the fraction of resources available (1 at rest);
* each spike delivers ``spike * scale * weight * U * x`` and then uses ``U * x`` of them;
* between spikes ``x`` recovers towards 1 with time constant ``tau_rec``.

The soma side is unchanged: the value goes into channel ``channel`` of the post soma's
inbox, and the soma decays that channel with its own ``tau_syn_<channel>``.

Event-capable although ``x`` changes between spikes
---------------------------------------------------
``x`` is only *needed* when a spike arrives, so the kernel does not update it every tick:
when woken it catches up the recovery over the ``n`` ticks since its last spike in closed
form, ``1 - x <- (1 - x) * (1 - dt/tau_rec)**n`` (``trace_decay``, the same forward-Euler
factor the soma and the learning rules use), and keeps the tick of that spike in
``internal_states[1]``. That is what lets it run only on the tick after its pre fired
(``synapse_stepping="event"``) and still give the same values as when stepped every tick:
in clocked mode it returns at once on every tick without a spike, before touching ``x``.

Registration
------------
Before ``setup()``::

    model.register_synapse_type(
        name="user_customized_depressing_synapse",
        step_func=user_customized_depressing_synapse,
        step_func_path=Path("customized_step_functions/user_customized_depressing_synapse.py").resolve(),
        event_capable=True,
    )

A synapse kernel must end its parameter list with ``synaptic_input, synaptic_input_idx``:
that is how the model knows it delivers into the soma inbox.

Hyperparameters -- the YAML key order *is* the index order read here:

    [0] weight          synaptic weight (pinned: a learning rule writes index 0)
    [1] synaptic_delay  pinned slot, unused (the inbox gives a fixed one-tick delay)
    [2] scale           multiplier on the delivered value
    [3] channel         inbox slot of the post soma (checked against synaptic_channels)
    [4] U               fraction of the available resources one spike uses (0..1)
    [5] tau_rec         recovery time constant of the resources (s)

Internal states:

    [0] x               available resources, 1.0 at rest
    [1] last_tick       tick of the last delivered spike, -1 before the first
"""

from cupyx import jit

from superneuroabm.step_functions.synapse.util import get_soma_spike, trace_decay
from superneuroabm.step_functions.synaptic_input import deliver_synaptic_current


@jit.rawkernel(device="cuda")
def user_customized_depressing_synapse(
    tick,
    agent_index,
    dt,
    I_bias,
    agent_ids,
    breeds,
    locations,
    synapse_params,  # weight, synaptic_delay, scale, channel, U, tau_rec
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
    channel = int(synapse_params[agent_index][3])
    U = synapse_params[agent_index][4]
    tau_rec = synapse_params[agent_index][5]

    # Catch up the recovery since the last spike, in closed form.
    x = internal_states[agent_index][0]
    last_tick = internal_states[agent_index][1]
    if last_tick >= 0.0:
        n = t_current - int(last_tick)
        x = 1.0 - (1.0 - x) * trace_decay(dt, tau_rec, n)

    # The value this synapse delivers: the only thing a synapse decides.
    deliver_synaptic_current(synaptic_input, synaptic_input_idx, post_soma_index, channel,
                             spike * scale * weight * U * x)

    internal_states[agent_index][0] = x - U * x
    internal_states[agent_index][1] = 1.0 * t_current
