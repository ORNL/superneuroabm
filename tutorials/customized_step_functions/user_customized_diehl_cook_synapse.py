"""
user_customized_diehl_cook_synapse -- the synapse of the Diehl & Cook (2015) network,
tutorial-owned: delivers its weight into one conductance channel of its post soma.

Registered at runtime by tutorial 04. The conductance itself lives on the soma
(user_customized_diehl_cook_soma.py); the synapse only says *when* and *how much*:

    channel 0.0 -> excitatory conductance ge     (input -> exc, exc -> inh)
    channel 1.0 -> inhibitory conductance gi     (inh -> exc, the lateral inhibition)

An inhibitory synapse has a POSITIVE weight: the soma's driving force ``e_inh - v``
gives it its sign.

Where the spike comes from
--------------------------
* ``pre_soma_index >= 0`` (exc -> inh, inh -> exc): the pre soma's spike of THIS tick,
  slot ``t % 2`` of its width-2 spike ring. The somas run before the synapses within a
  tick, behind a grid barrier, so the slot is complete when it is read. A spike at t is
  therefore deposited at t and reaches the post membrane at t+1, which makes the
  exc -> inh -> exc loop take 2 ticks (1 ms), as in the authors' Brian 1 code.
  The built-in ``get_soma_spike`` reads the PREVIOUS tick's slot and would add a tick
  per hop; and an event-woken synapse runs the tick after its pre fired, which has the
  same effect. That is why the model uses ``synapse_stepping="clocked"``: every synapse
  is stepped every tick.
* ``pre_soma_index == -1`` (the 78,400 input synapses): an injected spike. The notebook
  injects each pixel's Poisson spikes on every synapse of that pixel, shifted by the
  synapse's own fixed delay (0-10 ms), and the framework stamps this tick's event into
  the synapse's ``input_spikes_tensor`` row as ``[tick, value]``.

The spike that arrived is kept in ``internal_states[0]`` (``pre_arrived``): the learning
rule runs right after this kernel on the same thread and reads it there, so the STDP
pre trace sees the spike when it *arrives* at the synapse, which is what Brian 1's
delayed STDP does.

Registration
------------
Before ``setup()``::

    model.register_synapse_type(
        name="user_customized_diehl_cook_synapse",
        step_func=user_customized_diehl_cook_synapse,
        step_func_path=Path("customized_step_functions/user_customized_diehl_cook_synapse.py").resolve(),
    )

with ``NeuromorphicModel(synapse_stepping="clocked")``: the kernel reads the spike of
the current tick, so it is not ``event_capable``. A synapse kernel must end its parameter
list with ``synaptic_input, synaptic_input_idx`` to deliver into the soma inbox.

Hyperparameters -- the YAML key order *is* the index order read here:

    [0] weight          synaptic weight (pinned: the learning rule writes index 0)
    [1] synaptic_delay  pinned slot, unused (the input delays are in the injected ticks)
    [2] channel         0.0 -> excitatory (inbox slot 0), 1.0 -> inhibitory (slot 1)

Internal states:

    [0] pre_arrived     the spike delivered this tick (0 or the injected value)
"""

from cupyx import jit

from superneuroabm.step_functions.synaptic_input import deliver_synaptic_current


@jit.rawkernel(device="cuda")
def user_customized_diehl_cook_synapse(
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
    weight = synapse_params[agent_index][0]
    channel = int(synapse_params[agent_index][2])

    # locations[agent_index] = [pre_soma_index, post_soma_index]; pre is -1 for an input
    _nbrs = locations[agent_index]
    pre_soma_index = _nbrs[0]
    post_soma_index = _nbrs[1]

    spike = 0.0
    if pre_soma_index >= 0:
        spike = output_spikes_tensor[pre_soma_index][t_current % 2]
    elif input_spikes_tensor[agent_index][0] == t_current:
        spike = input_spikes_tensor[agent_index][1]

    internal_states[agent_index][0] = spike
    if spike != 0.0:
        deliver_synaptic_current(synaptic_input, synaptic_input_idx, post_soma_index, channel,
                                 spike * weight)

    buffer_idx = t_current % len(internal_states_buffer[agent_index])
    internal_states_buffer[agent_index][buffer_idx][0] = spike
