"""
user_customized_diehl_cook_soma -- the conductance-based LIF neuron of Diehl & Cook
(2015), tutorial-owned, following the authors' Brian 1 code step for step.

Registered at runtime by tutorial 04; the installed ``superneuroabm`` package is not
touched. One kernel serves both populations of the network: the config block decides
whether a soma is an excitatory cell (adaptive threshold) or an inhibitory one.

The model
---------
Unlike every built-in soma, which adds its synaptic input as a *current*, this neuron is
driven by two *conductances* through their driving forces (Euler, ``dt = 0.5 ms``)::

    dv/dt = ((v_rest - v) + ge * (0 - v) + gi * (e_inh - v)) / tau_v

    spike when v > v_thresh - offset + theta   and the spike lockout has expired
    on a spike: v <- v_reset, theta <- theta + theta_plus

``theta`` is the homeostatic adaptive threshold: every spike raises it by ``theta_plus``
and it decays with a time constant of 1e7 ms. The decay factor per tick, ``1 - 5e-8``,
is not representable in float32 (it rounds to ``1 - 5.96e-8``, 19 % too fast), so the
notebook applies the decay once per digit on the device and this kernel only adds
``theta_plus``. Inhibition has a positive conductance and gets its sign from
``e_inh - v``: an inhibitory synapse of this network has a positive weight.

Where the conductances live
---------------------------
Each afferent synapse deposits its weight into channel 0 (excitatory) or channel 1
(inhibitory) of this soma's inbox on the tick its pre fires
(user_customized_diehl_cook_synapse.py). The kernel decays each channel with its own
time constant and adds last tick's deposits::

    ge <- ge * (1 - dt / tau_syn_0) + deposits_0     factor 0.5 per tick (tau_ge = 1 ms)
    gi <- gi * (1 - dt / tau_syn_1) + deposits_1     factor 0.75 per tick (tau_gi = 2 ms)

which is the authors' per-synapse recursion summed over the afferents. A pre spike at
tick t is deposited at t and felt by this membrane at t+1, as in Brian.

Refractoriness, as Brian 1 runs it
----------------------------------
Brian 1 treats the two populations differently because their resets differ:

* the excitatory reset is a *string* (``'v = v_reset_e; theta += theta_plus_e; ...'``),
  for which Brian 1 installs no membrane clamp: ``refractory = 5 ms`` only forbids
  spikes for 9 ticks while ``v`` keeps integrating from ``v_reset``.
  -> ``lockout_ticks = 9, hold_ticks = 0``
* the inhibitory reset is a *scalar*, so Brian 1 clamps ``v`` at ``v_reset`` at the end
  of the spike tick and of the 4 following ones, and allows the next spike at +4.
  -> ``lockout_ticks = 3, hold_ticks = 4``

Per tick ("check, then decrement")::

    s = (v > threshold) and (lockout == 0)
    if s: v = v_reset; theta += theta_plus; lockout = lockout_ticks; hold = hold_ticks
    else: lockout = max(lockout - 1, 0); if hold > 0: v = v_reset; hold -= 1

All counters are small integers, exact in float32 however long the run.

Registration
------------
Before ``setup()``::

    model.register_soma_type(
        name="user_customized_diehl_cook_soma",
        step_func=user_customized_diehl_cook_soma,
        step_func_path=Path("customized_step_functions/user_customized_diehl_cook_soma.py").resolve(),
    )

``name`` is the breed key under ``soma:`` in ``diehl_cook_config.yaml``. A soma kernel
that ends its parameter list with ``synaptic_input, synaptic_input_idx`` gets an inbox.

Hyperparameters -- the YAML key order *is* the index order read here:

    [0]  v_rest        resting potential (mV)
    [1]  v_reset       reset potential (mV)
    [2]  v_thresh      threshold (mV)
    [3]  tau_v         membrane time constant (s)
    [4]  e_inh         inhibitory reversal potential (mV)
    [5]  theta_plus    threshold increase per spike (mV); 0 freezes the threshold
    [6]  offset        threshold offset (mV); theta starts at it
    [7]  lockout_ticks spikes forbidden for this many ticks after a spike
    [8]  hold_ticks    v held at v_reset for this many ticks after a spike
    [9]  tau_syn_0     excitatory conductance time constant (s), tau_ge
    [10] tau_syn_1     inhibitory conductance time constant (s), tau_gi

Internal states:

    [0] v          membrane potential (mV)
    [1] theta      adaptive threshold (mV)
    [2] lockout    ticks of spike lockout left
    [3] hold       ticks of v clamp left
    [4] I_syn_0    ge, the summed excitatory conductance
    [5] I_syn_1    gi, the summed inhibitory conductance
"""

from cupyx import jit

from superneuroabm.step_functions.synaptic_input import read_synaptic_input


@jit.rawkernel(device="cuda")
def user_customized_diehl_cook_soma(
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

    v_rest = neuron_params[agent_index][0]
    v_reset = neuron_params[agent_index][1]
    v_thresh = neuron_params[agent_index][2]
    tau_v = neuron_params[agent_index][3]
    e_inh = neuron_params[agent_index][4]
    theta_plus = neuron_params[agent_index][5]
    offset = neuron_params[agent_index][6]
    lockout_ticks = neuron_params[agent_index][7]
    hold_ticks = neuron_params[agent_index][8]
    tau_ge = neuron_params[agent_index][9]
    tau_gi = neuron_params[agent_index][10]

    # Conductances: decay each channel, add what the afferents deposited last tick.
    ge = internal_states[agent_index][4] * (1.0 - dt / tau_ge)
    gi = internal_states[agent_index][5] * (1.0 - dt / tau_gi)
    ge = ge + read_synaptic_input(agent_index, synaptic_input, synaptic_input_idx, 0)
    gi = gi + read_synaptic_input(agent_index, synaptic_input, synaptic_input_idx, 1)
    internal_states[agent_index][4] = ge
    internal_states[agent_index][5] = gi

    v = internal_states[agent_index][0]
    theta = internal_states[agent_index][1]
    lockout = internal_states[agent_index][2]
    hold = internal_states[agent_index][3]

    # Euler membrane update, every tick (the inhibitory clamp is applied after the
    # threshold test, below).
    v = v + (dt / tau_v) * ((v_rest - v) + ge * (0.0 - v) + gi * (e_inh - v))

    s = 0.0
    if lockout <= 0.0 and v > (v_thresh - offset + theta):
        s = 1.0

    if s > 0.0:
        v = v_reset
        theta = theta + theta_plus
        lockout = lockout_ticks
        hold = hold_ticks
    else:
        lockout = lockout - 1.0
        if lockout < 0.0:
            lockout = 0.0
        if hold > 0.0:
            v = v_reset
            hold = hold - 1.0

    internal_states[agent_index][0] = v
    internal_states[agent_index][1] = theta
    internal_states[agent_index][2] = lockout
    internal_states[agent_index][3] = hold

    # Width-2 spike ring: the synapses read slot t % 2 later in this same tick.
    output_spikes_tensor[agent_index][t_current % 2] = s

    buffer_idx = t_current % len(internal_states_buffer[agent_index])
    internal_states_buffer[agent_index][buffer_idx][0] = v
    internal_states_buffer[agent_index][buffer_idx][1] = theta
    internal_states_buffer[agent_index][buffer_idx][2] = lockout
    internal_states_buffer[agent_index][buffer_idx][3] = hold
    internal_states_buffer[agent_index][buffer_idx][4] = ge
    internal_states_buffer[agent_index][buffer_idx][5] = gi
