"""
user_customized_brunel_soma -- a LIF soma with one exponential synaptic current per inbox
channel, tutorial-owned.

What it shows (tutorial 07, section 2): everything a soma step function has to do. Each tick it
reads its parameters and state by position, reads what the synapses delivered into its inbox
last tick, updates its state, and writes its spike and its history::

    I_c <- I_c * exp(-dt / tau_syn_c) + (what the synapses delivered into channel c last tick)
    V   <- E_L + (V - E_L) * P22 + I_syn_0 * P21_0 + I_syn_1 * P21_1 + I_bias * P20
           (V is held while refractory, i.e. for t_ref after a spike)
    spike when V >= V_th and not refractory, then V <- V_reset

The membrane step is exact, not an approximation that improves with a smaller dt: between two
ticks the currents decay exponentially and nothing else happens, and for that the membrane
equation dV/dt = -(V - E_L) / tau_m + I / C_m has a closed-form solution. P22, P21_c and P20 are
its coefficients for one tick (how much of V survives, and how much one unit of each current and
of the bias adds). NEST's ``iaf_psc_exp`` integrates the same way, so the two give the same
membrane for the same input. It needs ``tau_syn_c != tau_m``.

Every value is in SI units: volts, amperes, farads, seconds. The current is a single exponential,
not an alpha function. The soma filters its input itself, so it reads the inbox with
``read_synaptic_input``; a soma that wants the standard filter (forward Euler, ``1 - dt/tau``) can
call ``synaptic_current`` instead, as the built-in ``lif_soma`` does.

Registration
------------
Before ``setup()``::

    model.register_soma_type(
        name="user_customized_brunel_soma",
        step_func=user_customized_brunel_soma,
        step_func_path=Path("customized_step_functions/user_customized_brunel_soma.py").resolve(),
    )

A soma kernel that ends its parameter list with ``synaptic_input, synaptic_input_idx`` gets an
inbox. This one reads slots 0 and 1, so it needs ``NeuromorphicModel(synaptic_channels=2)`` (the
default). Both channels have the same time constant in the tutorial's config, where one channel
would give the same result (tutorial 07, section 4).

Hyperparameters -- the YAML key order *is* the index order read here:

    [0] C_m        membrane capacitance (F)
    [1] tau_m      membrane time constant (s)
    [2] tau_syn_0  decay of channel 0's current (s); the kernel calls it tau_syn_ex
    [3] tau_syn_1  decay of channel 1's current (s); the kernel calls it tau_syn_in
    [4] V_th       spike threshold (V)
    [5] V_reset    reset potential (V)
    [6] E_L        leak reversal potential (V)
    [7] t_ref      refractory period (s)

The YAML names ``tau_syn_0`` and ``tau_syn_1`` are required by a check in ``setup()``, not by this
kernel: any key starting ``tau_syn_`` makes it require ``tau_syn_0..K-1`` and ``I_syn_0..K-1``.

Internal states, also by position:

    [0] V_m      membrane potential (V)
    [1] tcount   ticks since the start
    [2] tlast    time of the last spike (s); starts below -t_ref so tick 0 is not refractory
    [3] I_syn_0  channel 0's current (A)
    [4] I_syn_1  channel 1's current (A)

The history buffer records the same five values.
"""

from cupyx import jit
import cupy as cp

from superneuroabm.step_functions.synaptic_input import read_synaptic_input


@jit.rawkernel(device="cuda")
def user_customized_brunel_soma(
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

    # === Read neuron parameters, by YAML position ===
    C_m = neuron_params[agent_index][0]           # F
    tau_m = neuron_params[agent_index][1]         # s
    tau_syn_ex = neuron_params[agent_index][2]    # s (channel 0)
    tau_syn_in = neuron_params[agent_index][3]    # s (channel 1)
    V_th = neuron_params[agent_index][4]          # V
    V_reset = neuron_params[agent_index][5]       # V
    E_L = neuron_params[agent_index][6]           # V
    t_ref = neuron_params[agent_index][7]         # s

    # === Read internal state, by YAML position ===
    V = internal_states[agent_index][0]           # V
    tcount = internal_states[agent_index][1]      # ticks since the start
    tlast = internal_states[agent_index][2]       # s
    I_syn_exc = internal_states[agent_index][3]   # A (channel 0)
    I_syn_inh = internal_states[agent_index][4]   # A (channel 1)

    # === Update synaptic currents ===
    # What the synapses delivered into each channel last tick
    I_syn_exc_in = read_synaptic_input(agent_index, synaptic_input, synaptic_input_idx, 0)
    I_syn_inh_in = read_synaptic_input(agent_index, synaptic_input, synaptic_input_idx, 1)

    # Exponential filter: decay, then add what arrived
    decay_exc = cp.exp(-dt / tau_syn_ex)
    decay_inh = cp.exp(-dt / tau_syn_in)

    I_syn_exc = I_syn_exc * decay_exc + I_syn_exc_in
    I_syn_inh = I_syn_inh * decay_inh + I_syn_inh_in

    # === Membrane voltage integration ===
    # dV/dt = -(V - E_L) / tau_m + I / C_m, solved exactly over one tick for currents that
    # decay exponentially from their values now:
    #   P22   the part of V - E_L that is left after one tick
    #   P21   what one ampere of a channel's current adds to V over the tick
    #   P20   what one ampere of constant bias adds to V over the tick
    P22 = cp.exp(-dt / tau_m)
    P21_exc = tau_syn_ex * tau_m / (C_m * (tau_m - tau_syn_ex)) * (P22 - decay_exc)
    P21_inh = tau_syn_in * tau_m / (C_m * (tau_m - tau_syn_in)) * (P22 - decay_inh)
    P20 = tau_m / C_m * (1.0 - P22)

    # Integrate only if enough time has passed since the last spike
    time_since_spike = (dt * tcount) - tlast  # s
    in_refractory = time_since_spike < t_ref

    if not in_refractory:
        V = E_L + (V - E_L) * P22 + I_syn_exc * P21_exc + I_syn_inh * P21_inh + I_bias * P20
    # else: V remains unchanged during the refractory period

    # === Spike detection and reset ===
    spike = 1.0 if (V >= V_th and not in_refractory) else 0.0

    if spike > 0.5:
        V = V_reset
        tlast = dt * tcount

    # === Update internal state ===
    internal_states[agent_index][0] = V
    internal_states[agent_index][1] += 1.0  # increment tcount
    internal_states[agent_index][2] = tlast
    internal_states[agent_index][3] = I_syn_exc
    internal_states[agent_index][4] = I_syn_inh

    # === Record output spike ===
    output_spikes_tensor[agent_index][t_current % 2] = spike

    # === Update state history buffer ===
    if len(internal_states_buffer[agent_index]) > 0:
        buffer_idx = t_current % len(internal_states_buffer[agent_index])
        internal_states_buffer[agent_index][buffer_idx][0] = V
        internal_states_buffer[agent_index][buffer_idx][1] = internal_states[agent_index][1]
        internal_states_buffer[agent_index][buffer_idx][2] = tlast
        internal_states_buffer[agent_index][buffer_idx][3] = I_syn_exc
        internal_states_buffer[agent_index][buffer_idx][4] = I_syn_inh
