"""
brunel_nest_soma.py -- LIF soma matching NEST iaf_psc_alpha for Brunel network validation.

This soma implements the Leaky Integrate-and-Fire (LIF) neuron model used in the Brunel
network, designed to match NEST's iaf_psc_alpha exactly for single-spike and network
validation.

Neuron Model
-----------
Membrane voltage evolution:
    dV/dt = -(V - E_L) / (R * C_m) + I_syn / C_m

where:
- V: membrane voltage (mV)
- E_L: leak reversal potential (0 mV)
- R: membrane resistance (implicit; R = tau_m / C_m)
- C_m: membrane capacitance (250 pF)
- tau_m: membrane time constant (20 ms)
- I_syn: synaptic input current (pA)

Spike condition: V >= V_th (20 mV)
After spike: V is reset to V_reset (0 mV)
Refractory period: t_ref (2 ms, no integration during this window)

Synaptic Input Channels
-----------------------
Two channels integrate synaptic inputs independently with alpha-shaped filters:
- Channel 0 (Excitatory): tau_syn_ex = 0.5 ms
- Channel 1 (Inhibitory): tau_syn_in = 0.5 ms

Each channel's synaptic current is filtered as:
    dI_c/dt = -I_c / tau_syn_c + I_c_in(t)

where I_c_in(t) is the delivered synaptic input for that channel (typically from delivery
synapses). The total synaptic current is the sum: I_syn = I_exc + I_inh.

Hyperparameters (indexed in agent_params array)
-----------------------------------------
[0]  C_m              -- membrane capacitance (pF), default 250.0
[1]  tau_m            -- membrane time constant (ms), default 20.0
[2]  tau_syn_ex       -- excitatory synaptic time constant (ms), default 0.5
[3]  tau_syn_in       -- inhibitory synaptic time constant (ms), default 0.5
[4]  V_th             -- spike threshold (mV), default 20.0
[5]  V_reset          -- reset potential (mV), default 0.0
[6]  E_L              -- leak reversal potential (mV), default 0.0
[7]  t_ref            -- refractory period (ms), default 2.0
[8]  I_bias           -- external input current (pA), default 0.0

Internal State (indexed in internal_states array)
-----------------------------------------
[0]  V                -- membrane voltage (mV)
[1]  tcount           -- spike counter (incremented each timestep)
[2]  tlast            -- last spike time (in dt units)
[3]  I_syn_exc        -- excitatory synaptic current (pA)
[4]  I_syn_inh        -- inhibitory synaptic current (pA)

Synaptic Input Delivery
-----------------------
Synapses deliver current to the soma's inbox (synaptic_input) indexed by channel:
- Channel 0: excitatory input (E spikes, positive weight)
- Channel 1: inhibitory input (I spikes, negative weight)

The inbox holds the delivered values; this soma reads and integrates them each timestep.

Registration (in SuperNeuroABM script)
--------------------------------------
Before model.setup():

    from pathlib import Path
    from brunel_nest_soma import brunel_nest_lif_soma

    model.register_soma_type(
        name="brunel_nest_lif",
        step_func=brunel_nest_lif_soma,
        step_func_path=Path("brunel_nest_soma.py").resolve(),
        num_channels=2,  # excitatory and inhibitory
        num_internal_state_variables=5,  # V, tcount, tlast, I_syn_exc, I_syn_inh
    )

Configuration (YAML, e.g., brunel_nest_config.yaml)
----------------------------------------------------
neuron_models:
  brunel_nest_lif:
    hyperparameters:
      C_m: 250.0
      tau_m: 20.0
      tau_syn_ex: 0.5
      tau_syn_in: 0.5
      V_th: 20.0
      V_reset: 0.0
      E_L: 0.0
      t_ref: 2.0
      I_bias: 0.0
    initial_state:
      V: 0.0
      tcount: 0.0
      tlast: 0.0
      I_syn_exc: 0.0
      I_syn_inh: 0.0

References
----------
Brunel, N. (2000). Dynamics of sparsely connected networks of excitatory and inhibitory
spiking neurons. Journal of Computational Neuroscience 8, 183-208.
"""

from cupyx import jit
import cupy as cp

from superneuroabm.step_functions.synaptic_input import synaptic_current


@jit.rawkernel(device="cuda")
def brunel_nest_lif_soma(
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
    """
    Brunel LIF soma step function.

    Parameters
    ----------
    tick : int
        Current simulation timestep.
    agent_index : int
        Index of this neuron in the agent array.
    dt : float
        Integration timestep (ms).
    I_bias : float
        External bias current (pA).
    agent_ids : cupy array
        Global IDs of all agents.
    breeds : cupy array
        Breed IDs of all agents.
    locations : cupy array
        Spatial locations (unused here).
    neuron_params : cupy array
        Neuron hyperparameters for all neurons (8 params each).
    learning_params : cupy array
        Learning parameters (unused).
    internal_states : cupy array
        Neuron internal state for all neurons (5 state vars each).
    learning_internal_states : cupy array
        Learning state (unused).
    synapse_history : cupy array
        Synapse history (unused).
    input_spikes_tensor : cupy array
        External spike input (unused).
    output_spikes_tensor : cupy array
        Output spike record (2-entry circular buffer per neuron).
    internal_states_buffer : cupy array
        State history buffer (unused here).
    learning_internal_states_buffer : cupy array
        Learning history buffer (unused).
    synaptic_input : cupy array
        Synaptic input from all synapses (summed per channel per neuron).
    synaptic_input_idx : cupy array
        Indices into synaptic_input (unused; indexed by agent_index and channel).
    """
    t_current = int(tick)

    # === Read neuron parameters ===
    C_m = neuron_params[agent_index][0]           # pF
    tau_m = neuron_params[agent_index][1]         # ms
    tau_syn_ex = neuron_params[agent_index][2]    # ms (excitatory)
    tau_syn_in = neuron_params[agent_index][3]    # ms (inhibitory)
    V_th = neuron_params[agent_index][4]          # mV
    V_reset = neuron_params[agent_index][5]       # mV
    E_L = neuron_params[agent_index][6]           # mV
    t_ref = neuron_params[agent_index][7]         # ms

    # === Read internal state ===
    V = internal_states[agent_index][0]           # mV
    tcount = internal_states[agent_index][1]      # tick counter
    tlast = internal_states[agent_index][2]       # last spike time (in dt units)
    I_syn_exc = internal_states[agent_index][3]   # pA (excitatory current)
    I_syn_inh = internal_states[agent_index][4]   # pA (inhibitory current)

    # === Update synaptic currents (alpha filters) ===
    # Read delivered synaptic input from inbox (summed by the model per channel)
    # Channel 0: excitatory, Channel 1: inhibitory
    I_syn_exc_in = synaptic_input[agent_index * 2 + 0] if agent_index * 2 + 0 < len(synaptic_input) else 0.0
    I_syn_inh_in = synaptic_input[agent_index * 2 + 1] if agent_index * 2 + 1 < len(synaptic_input) else 0.0

    # Alpha filter: dI/dt = -I/tau + I_in
    # Discrete: I(t+dt) = I(t) * exp(-dt/tau) + I_in
    decay_exc = cp.exp(-dt / tau_syn_ex)
    decay_inh = cp.exp(-dt / tau_syn_in)

    I_syn_exc = I_syn_exc * decay_exc + I_syn_exc_in
    I_syn_inh = I_syn_inh * decay_inh + I_syn_inh_in

    # === Total synaptic current ===
    I_syn = I_syn_exc + I_syn_inh  # pA

    # === Membrane voltage integration ===
    # dV/dt = -(V - E_L) / (R * C_m) + I_syn / C_m
    # where R = tau_m / C_m (implicitly defined by tau_m = R * C_m)
    # So: dV/dt = -(V - E_L) / tau_m + I_syn / C_m
    
    # Check refractory period: integrate only if enough time has passed since last spike
    time_since_spike = (dt * tcount) - tlast  # ms
    in_refractory = time_since_spike < t_ref

    if not in_refractory:
        # Integrate membrane voltage
        dV = (-(V - E_L) / tau_m + (I_syn + I_bias) / C_m) * dt
        V += dV
    # else: V remains unchanged during refractory period

    # === Spike detection and reset ===
    # Spike occurs if V >= V_th and not in refractory period
    spike = 1.0 if (V >= V_th and not in_refractory) else 0.0

    # Reset after spike
    if spike > 0.5:
        V = V_reset
        tlast = dt * tcount
    # If not spiking and past refractory period, V can integrate normally (handled above)

    # === Update internal state ===
    internal_states[agent_index][0] = V
    internal_states[agent_index][1] += 1.0  # increment tcount
    internal_states[agent_index][2] = tlast
    internal_states[agent_index][3] = I_syn_exc
    internal_states[agent_index][4] = I_syn_inh

    # === Record output spike ===
    output_spikes_tensor[agent_index][t_current % 2] = spike

    # === Update state history buffer (if used by the model) ===
    if len(internal_states_buffer[agent_index]) > 0:
        buffer_idx = t_current % len(internal_states_buffer[agent_index])
        internal_states_buffer[agent_index][buffer_idx][0] = V
        internal_states_buffer[agent_index][buffer_idx][1] = internal_states[agent_index][1]
        internal_states_buffer[agent_index][buffer_idx][2] = tlast
        internal_states_buffer[agent_index][buffer_idx][3] = I_syn_exc
        internal_states_buffer[agent_index][buffer_idx][4] = I_syn_inh
