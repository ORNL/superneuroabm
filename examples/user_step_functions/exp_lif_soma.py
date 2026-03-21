"""
Custom Exponential LIF (Leaky Integrate-and-Fire) soma step function.

Uses exponential decay (exact solution) instead of Euler integration:

    v(t+dt) = vrest + (v(t) - vrest) * exp(-dt/tau_m) + I * dt / tau_m

Hyperparameters (in order):
    [0] tau_m            - Membrane time constant (s)
    [1] vthr             - Spike threshold (mV)
    [2] vrest            - Resting potential (mV)
    [3] vreset           - Reset potential after spike (mV)
    [4] tref             - Absolute refractory period (s)
    [5] I_in             - Constant external input current
    [6] scaling_factor   - Synaptic current scaling factor

Internal state (in order):
    [0] v                - Membrane potential (mV)
    [1] tcount           - Tick counter
    [2] tlast            - Last spike time (s)
"""

import cupy as cp
from cupyx import jit


@jit.rawkernel(device="cuda")
def exp_lif_soma_step_func(
    tick,
    agent_index,
    dt,
    I_bias,
    agent_ids,
    breeds,
    locations,
    neuron_params,
    learning_params,
    internal_state,
    internal_learning_state,
    synapse_history,
    input_spikes_tensor,
    output_spikes_tensor,
    internal_states_buffer,
    internal_learning_states_buffer,
):
    # Sum synaptic currents from connected synapses
    synapse_indices = locations[agent_index]
    I_synapse = 0.0
    for i in range(len(synapse_indices)):
        synapse_index = int(synapse_indices[i])
        if synapse_index >= 0 and not cp.isnan(synapse_indices[i]):
            I_synapse += internal_state[synapse_index][0]

    t_current = int(tick)

    # Read hyperparameters
    tau_m = neuron_params[agent_index][0]
    vthr = neuron_params[agent_index][1]
    vrest = neuron_params[agent_index][2]
    vreset = neuron_params[agent_index][3]
    tref = neuron_params[agent_index][4]
    I_in = neuron_params[agent_index][5]
    scaling_factor = neuron_params[agent_index][6]

    # Read internal state
    v = internal_state[agent_index][0]
    tcount = internal_state[agent_index][1]
    tlast = internal_state[agent_index][2]

    # Exponential decay toward rest + input
    decay = cp.exp(-dt / tau_m)
    I_total = I_synapse * scaling_factor + I_in

    # Only integrate if not in refractory period
    in_refractory = (tlast > 0) and (dt * tcount <= tlast + tref)
    if not in_refractory:
        v = vrest + (v - vrest) * decay + I_total * dt / tau_m

    # Spike detection
    s = 1.0 * ((v >= vthr) and not in_refractory)

    # Reset on spike
    tlast = tlast * (1 - s) + dt * tcount * s
    v = v * (1 - s) + vreset * s

    # Write back state
    internal_state[agent_index][0] = v
    internal_state[agent_index][1] = tcount + 1
    internal_state[agent_index][2] = tlast

    output_spikes_tensor[agent_index][t_current % 2] = s

    buffer_idx = t_current % len(internal_states_buffer[agent_index])
    internal_states_buffer[agent_index][buffer_idx][0] = v
    internal_states_buffer[agent_index][buffer_idx][1] = tcount + 1
    internal_states_buffer[agent_index][buffer_idx][2] = tlast
