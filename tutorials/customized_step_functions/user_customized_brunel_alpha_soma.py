"""
user_customized_brunel_alpha_soma -- NEST's ``iaf_psc_alpha`` as a soma kernel, with a
transmission delay, tutorial-owned.

It is ``user_customized_brunel_soma`` with two changes, the two that NEST's classic Brunel
example (``brunel_alpha_nest.py``) and its HPC benchmark need:

1. **Alpha-shaped currents.** One event of weight ``w`` makes the channel current
   ``w * (t / tau_syn) * exp(1 - t / tau_syn)``, peaking at ``w`` after ``tau_syn``. That
   takes two state variables per channel, the current ``I`` and its derivative ``dI``.
   Each tick, exactly as NEST GPU's ``iaf_psc_alpha_Update`` and then its input delivery::

       V_rel <- V_rel + P30 I_bias + sum_c (P31_c dI_c + P32_c I_c) + expm1(-dt/tau_m) V_rel
                (skipped while refractory, which counts down round(t_ref / dt) ticks)
       I_c   <- P21_c dI_c + P22_c I_c
       dI_c  <- P11_c dI_c
       spike when V_rel >= V_th - E_L, then V_rel <- V_reset - E_L
       dI_c  <- dI_c + (what arrived in channel c) * e / tau_syn_c

   ``V_rel = V - E_L``. ``P11..P33`` are the exact one-tick propagators of the linear system
   (NEST's ``propagator_31`` / ``propagator_32``, non-singular branch), so the step is exact
   for any ``dt``. That branch needs ``tau_syn_c`` away from ``tau_m``: NEST switches to a
   different form within 0.1 ms of equality, this kernel does not.

2. **A transmission delay.** The framework delivers a spike in a fixed number of ticks. A
   longer delay is added here, at the sending side: the soma keeps its last
   ``delay_ticks`` spikes in a bit queue and *emits* (writes to ``output_spikes_tensor``)
   the spike it fired ``delay_ticks`` ticks ago. Every outgoing synapse then delivers it
   the usual way. This is exact for one uniform delay, as in both Brunel networks, and
   costs one state slot, where a per-channel ring of arrivals on the receiving side would
   cost ``2 * delay_ticks`` slots, and since every agent's state row is padded to the
   widest breed, every synapse would carry those as well. Two consequences for a script:

   * recorded spike times (``get_all_spike_times``) are emission times, ``delay_ticks``
     after the spike: subtract it, and run ``delay_ticks`` extra ticks to flush the queue;
   * injected input (``add_spikes`` on an external synapse) does not pass through any
     soma's queue, so it must be injected ``delay_ticks`` later to see the same delay.

   The queue is one float32 holding an integer, exact up to 24 bits: ``delay_ticks <= 23``.

Every value is in SI units: volts, amperes, farads, seconds; ``dI`` is in A/s.

Registration
------------
Before ``setup()``::

    model.register_soma_type(
        name="user_customized_brunel_alpha_soma",
        step_func=user_customized_brunel_alpha_soma,
        step_func_path=Path("customized_step_functions/user_customized_brunel_alpha_soma.py").resolve(),
    )

It reads inbox slots 0 and 1, so it needs ``NeuromorphicModel(synaptic_channels=2)`` (the
default).

Hyperparameters -- the YAML key order *is* the index order read here:
    [0] C_m          membrane capacitance (F)
    [1] tau_m        membrane time constant (s)
    [2] tau_syn_0    rise/decay time of channel 0's alpha current (s); NEST's tau_syn_ex
    [3] tau_syn_1    the same for channel 1 (s); NEST's tau_syn_in
    [4] V_th         spike threshold (V)
    [5] V_reset      reset potential (V)
    [6] E_L          leak reversal potential (V)
    [7] t_ref        refractory period (s), counted as round(t_ref / dt) ticks
    [8] delay_ticks  ticks between a spike and its emission (0 to 23)

Internal states, also by position:
    [0] V_m          membrane potential (V)
    [1] refractory   refractory ticks left
    [2] I_syn_0      channel 0's current (A)
    [3] I_syn_1      channel 1's current (A)
    [4] dI_syn_0     channel 0's current derivative (A/s)
    [5] dI_syn_1     channel 1's current derivative (A/s)
    [6] spike_queue  the last delay_ticks spikes, newest in bit 0

The history buffer records the same seven values.
"""

from cupyx import jit
import cupy as cp

from superneuroabm.step_functions.synaptic_input import read_synaptic_input


@jit.rawkernel(device="cuda")
def user_customized_brunel_alpha_soma(
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
    tau_ex = neuron_params[agent_index][2]        # s (channel 0)
    tau_in = neuron_params[agent_index][3]        # s (channel 1)
    V_th = neuron_params[agent_index][4]          # V
    V_reset = neuron_params[agent_index][5]       # V
    E_L = neuron_params[agent_index][6]           # V
    t_ref = neuron_params[agent_index][7]         # s
    delay_ticks = int(neuron_params[agent_index][8])

    # === Read internal state, by YAML position ===
    V_rel = internal_states[agent_index][0] - E_L  # V, relative to E_L as in NEST
    refractory = internal_states[agent_index][1]   # ticks left
    I_ex = internal_states[agent_index][2]         # A
    I_in = internal_states[agent_index][3]         # A
    dI_ex = internal_states[agent_index][4]        # A/s
    dI_in = internal_states[agent_index][5]        # A/s
    queue = int(internal_states[agent_index][6])

    # === One-tick propagators (NEST GPU computes the same in iaf_psc_alpha::Calibrate) ===
    P11_ex = cp.exp(-dt / tau_ex)                  # dI decay; also I's own decay (P22)
    P11_in = cp.exp(-dt / tau_in)
    P21_ex = dt * P11_ex                           # what dI adds to I
    P21_in = dt * P11_in
    expm1_tau_m = cp.exp(-dt / tau_m) - 1.0        # V_rel's change per volt
    P30 = -tau_m / C_m * expm1_tau_m               # what one ampere of bias adds to V
    # NEST's propagator_31 (what one A/s of dI adds to V) and propagator_32 (what one A of I
    # adds), non-singular branch. Written out here: setup() imports each kernel module with
    # `import *`, which would not bring private helper functions along.
    P31_ex = (1.0 / C_m * (P11_ex * (cp.exp(-dt / tau_m + dt / tau_ex) - 1.0)
                           / (tau_m / tau_ex - 1.0) * tau_m - dt * P11_ex)
              / (-1.0 + tau_m / tau_ex) * tau_m)
    P31_in = (1.0 / C_m * (P11_in * (cp.exp(-dt / tau_m + dt / tau_in) - 1.0)
                           / (tau_m / tau_in - 1.0) * tau_m - dt * P11_in)
              / (-1.0 + tau_m / tau_in) * tau_m)
    P32_ex = (-tau_m / (C_m * (1.0 - tau_m / tau_ex)) * P11_ex
              * (cp.exp(dt * (1.0 / tau_ex - 1.0 / tau_m)) - 1.0))
    P32_in = (-tau_m / (C_m * (1.0 - tau_m / tau_in)) * P11_in
              * (cp.exp(dt * (1.0 / tau_in - 1.0 / tau_m)) - 1.0))

    # === Membrane, unless refractory (from the currents as they were before this tick) ===
    if refractory > 0.5:
        refractory = refractory - 1.0
    else:
        V_rel = (P30 * I_bias + P31_ex * dI_ex + P32_ex * I_ex + P31_in * dI_in + P32_in * I_in
                 + expm1_tau_m * V_rel + V_rel)

    # === Alpha currents ===
    I_ex = P21_ex * dI_ex + P11_ex * I_ex
    dI_ex = dI_ex * P11_ex
    I_in = P21_in * dI_in + P11_in * I_in
    dI_in = dI_in * P11_in

    # === Spike detection and reset ===
    spike = 0
    if V_rel >= V_th - E_L:
        spike = 1
        V_rel = V_reset - E_L
        refractory = float(int(t_ref / dt + 0.5))  # round(t_ref / dt), as NEST GPU

    # === Input: what the synapses delivered last tick, into dI (after the update, as NEST GPU) ===
    dI_ex = dI_ex + read_synaptic_input(agent_index, synaptic_input, synaptic_input_idx, 0) * (
        2.718281828459045 / tau_ex)
    dI_in = dI_in + read_synaptic_input(agent_index, synaptic_input, synaptic_input_idx, 1) * (
        2.718281828459045 / tau_in)

    # === Transmission delay: emit the spike fired delay_ticks ticks ago ===
    emitted = spike
    if delay_ticks > 0:
        emitted = (queue >> (delay_ticks - 1)) & 1
        queue = ((queue << 1) | spike) & ((1 << delay_ticks) - 1)

    # === Update internal state ===
    internal_states[agent_index][0] = V_rel + E_L
    internal_states[agent_index][1] = refractory
    internal_states[agent_index][2] = I_ex
    internal_states[agent_index][3] = I_in
    internal_states[agent_index][4] = dI_ex
    internal_states[agent_index][5] = dI_in
    internal_states[agent_index][6] = float(queue)

    # === Record output spike (the emitted one) ===
    output_spikes_tensor[agent_index][t_current % 2] = float(emitted)

    # === Update state history buffer ===
    if len(internal_states_buffer[agent_index]) > 0:
        buffer_idx = t_current % len(internal_states_buffer[agent_index])
        for k in range(7):
            internal_states_buffer[agent_index][buffer_idx][k] = internal_states[agent_index][k]
