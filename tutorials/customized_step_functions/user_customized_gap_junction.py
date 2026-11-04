"""
user_customized_gap_junction -- an electrical synapse (gap junction), tutorial-owned.

A gap junction is a channel between two cells' membranes: current flows from the more
depolarised cell to the other on every tick, in proportion to the voltage difference,
whether or not either cell spikes:

    I = g * (v_pre - v_post)        delivered into the post soma's inbox, every tick

Nothing in that depends on a spike, so the synapse has no event to be woken by: it must
be stepped on every tick, ``NeuromorphicModel(synapse_stepping="clocked")``, and it is
registered without ``event_capable`` (the model refuses it in event stepping). A network
with gap junctions therefore runs clocked as a whole; tutorial 05 shows the cost.

The coupling is symmetric in biology; here one synapse agent carries current one way, so
a junction between A and B is two of them, A -> B and B -> A, with the same ``weight``.

The synapse reads ``internal_states[...][0]`` of both somas, which is the membrane
potential of every built-in soma (``v`` for lif_soma and izh_soma). Somas run before
synapses within a tick, so it reads this tick's potentials; the soma integrates the
current on the next tick. ``channel`` should be a channel whose ``tau_syn`` equals ``dt``
on the post soma, so the current lasts exactly the tick it was computed for.

On more than one GPU a synapse whose pre soma lives on another rank reads that soma's
ghost copy, and only the spike slot of a soma is sent to its ghosts by default: such a
model calls ``model.set_property_neighbor_visible("internal_states", True)`` before
``setup()`` (one GPU, as in the tutorial, needs nothing).

Registration
------------
Before ``setup()``, with ``NeuromorphicModel(synapse_stepping="clocked")``::

    model.register_synapse_type(
        name="user_customized_gap_junction",
        step_func=user_customized_gap_junction,
        step_func_path=Path("customized_step_functions/user_customized_gap_junction.py").resolve(),
    )

Hyperparameters -- the YAML key order *is* the index order read here:

    [0] weight          the coupling conductance g (current per mV of difference)
    [1] synaptic_delay  pinned slot, unused
    [2] channel         inbox slot of the post soma (one with tau_syn == dt)

Internal states: none are used; the config declares one placeholder slot.
"""

from cupyx import jit

from superneuroabm.step_functions.synaptic_input import deliver_synaptic_current


@jit.rawkernel(device="cuda")
def user_customized_gap_junction(
    tick,
    agent_index,
    dt,
    I_bias,
    agent_ids,
    breeds,
    locations,
    synapse_params,  # weight (g), synaptic_delay, channel
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
    _nbrs = locations[agent_index]
    pre_soma_index = _nbrs[0]
    post_soma_index = -1
    if len(_nbrs) > 1:
        post_soma_index = _nbrs[1]
    if pre_soma_index < 0 or post_soma_index < 0:
        return   # a gap junction joins two somas; an external input has no membrane

    g = synapse_params[agent_index][0]
    channel = int(synapse_params[agent_index][2])
    current = g * (internal_states[pre_soma_index][0] - internal_states[post_soma_index][0])
    deliver_synaptic_current(synaptic_input, synaptic_input_idx, post_soma_index, channel, current)
