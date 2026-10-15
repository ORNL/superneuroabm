"""
LIF soma with an adaptive threshold: the threshold jumps by ``delta_thr`` at every spike
and relaxes back to ``vthr_initial`` with ``tau_decay_thr``. Synaptic input comes from
the soma inbox through one exponential filter per channel
(step_functions/synaptic_input.synaptic_current), as in lif_soma.

Hyperparameters (neuron_params): [0] C, [1] R, [2] vthr_initial, [3] tref, [4] vrest,
[5] vreset, [6] tref_allows_integration, [7] I_in, [8] scaling_factor, [9] delta_thr,
[10] tau_decay_thr, [11 + c] tau_syn_c.
Internal state: [0] v, [1] tcount, [2] tlast, [3] vthr, [4 + c] I_syn_c.
"""

from cupyx import jit
import cupy as cp

from superneuroabm.step_functions.synaptic_input import synaptic_current


@jit.rawkernel(device="cuda")
def lif_soma_adaptive_thr_step_func(
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

    I_synapse = synaptic_current(agent_index, dt, neuron_params, 11, internal_states, 4,
                                 synaptic_input, synaptic_input_idx)

    C = neuron_params[agent_index][0]
    R = neuron_params[agent_index][1]
    vthr_initial = neuron_params[agent_index][2]
    tref = neuron_params[agent_index][3]
    vrest = neuron_params[agent_index][4]
    vreset = neuron_params[agent_index][5]
    tref_allows_integration = neuron_params[agent_index][6]
    I_in = neuron_params[agent_index][7]
    scaling_factor = neuron_params[agent_index][8]
    delta_thr = neuron_params[agent_index][9]
    tau_decay_thr = neuron_params[agent_index][10]

    v = internal_states[agent_index][0]
    tcount = internal_states[agent_index][1]
    tlast = internal_states[agent_index][2]
    vthr = internal_states[agent_index][3]

    dv = (vrest - v) / (R * C) + (I_synapse * scaling_factor + I_bias + I_in) / C
    v += (
        (dv * dt)
        if ((dt * tcount) > (tlast + tref)) or tref_allows_integration
        else 0.0
    )
    s = 1.0 * ((v >= vthr) and ((dt * tcount > tlast + tref) if tlast > 0 else True))

    # adaptive threshold: jump at a spike, otherwise relax towards the initial value
    if s == 1.0:
        vthr += delta_thr
    else:
        vthr = vthr_initial + (vthr - vthr_initial) * cp.exp(-dt / tau_decay_thr)

    tlast = tlast * (1 - s) + dt * tcount * s
    v = v * (1 - s) + vreset * s

    internal_states[agent_index][0] = v
    internal_states[agent_index][1] += 1
    internal_states[agent_index][2] = tlast
    internal_states[agent_index][3] = vthr

    output_spikes_tensor[agent_index][t_current % 2] = s

    buffer_idx = t_current % len(internal_states_buffer[agent_index])
    internal_states_buffer[agent_index][buffer_idx][0] = v
    internal_states_buffer[agent_index][buffer_idx][1] = internal_states[agent_index][1] + 1
    internal_states_buffer[agent_index][buffer_idx][2] = tlast
    internal_states_buffer[agent_index][buffer_idx][3] = vthr
    internal_states_buffer[agent_index][buffer_idx][4] = I_synapse
