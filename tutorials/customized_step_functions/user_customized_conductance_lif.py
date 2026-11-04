"""
user_customized_conductance_lif -- a LIF soma whose inbox channels are conductances,
tutorial-owned.

What it shows (tutorial 06): a soma step function decides *what its channels mean*. The
built-in ``lif_soma`` gathers its inbox, decays each channel with ``tau_syn_c`` and adds the
channels up as a *current* (``synaptic_current``). This soma gathers and decays its two
channels the same way, but treats each as a *conductance* with its own reversal potential::

    g_c <- g_c * (1 - dt / tau_syn_c) + (what the synapses delivered into channel c)
    I    = g_0 * (E_rev_0 - v) + g_1 * (E_rev_1 - v)

so the effect of an input depends on the membrane potential at every tick. A synapse
cannot do that in event mode (it runs only on the tick after its pre fired), which is why
this belongs in the soma. The synapses are the unchanged built-in ``delivery_synapse``: an
inhibitory synapse here has a *positive* weight and gets its sign from ``E_rev_1 - v``.

The membrane update, refractory period and reset are the built-in ``lif_soma``'s
(``superneuroabm/step_functions/soma/lif.py``); only the synaptic term differs. The full-size
example of the same idea is ``user_customized_diehl_cook_soma.py`` (tutorial 04).

Registration
------------
Before ``setup()``::

    model.register_soma_type(
        name="user_customized_conductance_lif",
        step_func=user_customized_conductance_lif,
        step_func_path=Path("customized_step_functions/user_customized_conductance_lif.py").resolve(),
    )

A soma kernel that ends its parameter list with ``synaptic_input, synaptic_input_idx`` gets
an inbox. This one needs ``NeuromorphicModel(synaptic_channels=2)`` (the default).

Hyperparameters -- the YAML key order *is* the index order read here:

    [0] C                        membrane capacitance (F)
    [1] R                        leak resistance (Ohm)
    [2] vthr                     spike threshold (mV)
    [3] tref                     refractory period (s)
    [4] vrest                    resting potential (mV)
    [5] vreset                   post-spike reset potential (mV)
    [6] tref_allows_integration  non-zero: keep integrating while refractory
    [7] I_in                     constant input current
    [8] scaling_factor           multiplier on the synaptic current
    [9] tau_syn_0                decay time constant of conductance channel 0 (s)
    [10] tau_syn_1               decay time constant of conductance channel 1 (s)
    [11] E_rev_0                 reversal potential of channel 0 (mV)
    [12] E_rev_1                 reversal potential of channel 1 (mV)

Internal states:

    [0] v        membrane potential (mV)
    [1] tcount   ticks elapsed
    [2] tlast    time of the last spike (s)
    [3] I_syn_0  g_0, the conductance of channel 0 (the name the channel check expects)
    [4] I_syn_1  g_1, the conductance of channel 1

History (``get_internal_states_history``): ``[v, tcount, tlast, g_0, g_1]``, the internal
states after each tick (a history row is only as wide as the internal states).
"""

from cupyx import jit

from superneuroabm.step_functions.synaptic_input import read_synaptic_input


@jit.rawkernel(device="cuda")
def user_customized_conductance_lif(
    tick,
    agent_index,
    dt,
    I_bias,
    agent_ids,
    breeds,
    locations,
    neuron_params,
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

    C = neuron_params[agent_index][0]
    R = neuron_params[agent_index][1]
    vthr = neuron_params[agent_index][2]
    tref = neuron_params[agent_index][3]
    vrest = neuron_params[agent_index][4]
    vreset = neuron_params[agent_index][5]
    tref_allows_integration = neuron_params[agent_index][6]
    I_in = neuron_params[agent_index][7]
    scaling_factor = neuron_params[agent_index][8]
    tau_syn_0 = neuron_params[agent_index][9]
    tau_syn_1 = neuron_params[agent_index][10]
    E_rev_0 = neuron_params[agent_index][11]
    E_rev_1 = neuron_params[agent_index][12]

    # Gather and decay, per channel: last tick's deliveries into channel c, added to the
    # decayed conductance of channel c. (The built-in synaptic_current helper does exactly
    # this and returns the sum; here the channels are needed one by one.)
    g_0 = (internal_states[agent_index][3] * (1 - dt / tau_syn_0)
           + read_synaptic_input(agent_index, synaptic_input, synaptic_input_idx, 0))
    g_1 = (internal_states[agent_index][4] * (1 - dt / tau_syn_1)
           + read_synaptic_input(agent_index, synaptic_input, synaptic_input_idx, 1))
    internal_states[agent_index][3] = g_0
    internal_states[agent_index][4] = g_1

    v = internal_states[agent_index][0]
    tcount = internal_states[agent_index][1]
    tlast = internal_states[agent_index][2]

    # What the channels mean to this soma: conductances with reversal potentials.
    I_synapse = g_0 * (E_rev_0 - v) + g_1 * (E_rev_1 - v)

    # From here on, the built-in lif_soma.
    dv = (vrest - v) / (R * C) + (I_synapse * scaling_factor + I_bias + I_in) / C
    v += (
        (dv * dt)
        if ((dt * tcount) > (tlast + tref)) or tref_allows_integration
        else 0.0
    )
    s = 1.0 * ((v >= vthr) and ((dt * tcount > tlast + tref) if tlast > 0 else True))
    tlast = tlast * (1 - s) + dt * tcount * s
    v = v * (1 - s) + vreset * s

    internal_states[agent_index][0] = v
    internal_states[agent_index][1] += 1
    internal_states[agent_index][2] = tlast

    output_spikes_tensor[agent_index][t_current % 2] = s

    # The history row is as wide as the internal states (five here), so record them as
    # they are; the current I can be recomputed from v and the conductances.
    buffer_idx = t_current % len(internal_states_buffer[agent_index])
    internal_states_buffer[agent_index][buffer_idx][0] = v
    internal_states_buffer[agent_index][buffer_idx][1] = internal_states[agent_index][1]
    internal_states_buffer[agent_index][buffer_idx][2] = tlast
    internal_states_buffer[agent_index][buffer_idx][3] = g_0
    internal_states_buffer[agent_index][buffer_idx][4] = g_1
