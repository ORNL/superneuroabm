"""
Izhikevich soma. Synaptic input comes from the soma inbox through one exponential
filter per channel (step_functions/synaptic_input.synaptic_current); the default config
sets every ``tau_syn_c`` equal to ``dt`` (decay factor zero), i.e. instantaneous
delivery, which is what this soma was paired with before.

Hyperparameters (neuron_params): [0] k, [1] vthr, [2] C, [3] a, [4] b, [5] vpeak,
[6] vrest, [7] d, [8] vreset, [9] I_in, [10 + c] tau_syn_c.
Internal state: [0] v, [1] u, [2 + c] I_syn_c.
"""

import cupy as cp
from cupyx import jit

from superneuroabm.step_functions.synaptic_input import synaptic_current


@jit.rawkernel(device="cuda")
def izh_soma_step_func(
    tick,
    agent_index,
    dt,
    I_bias,
    agent_ids,
    breeds,
    locations,
    neuron_params,  # k, vthr, C, a, b, vpeak, vrest, d, vreset, I_in, tau_syn_0, tau_syn_1, ...
    learning_params,
    internal_states,  # v, u, I_syn_0, I_syn_1, ...
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

    I_synapse = synaptic_current(agent_index, dt, neuron_params, 10, internal_states, 2,
                                 synaptic_input, synaptic_input_idx)

    k = neuron_params[agent_index][0]
    vthr = neuron_params[agent_index][1]
    C = neuron_params[agent_index][2]
    a = neuron_params[agent_index][3]
    b = neuron_params[agent_index][4]
    vpeak = neuron_params[agent_index][5]
    vrest = neuron_params[agent_index][6]
    d = neuron_params[agent_index][7]
    vreset = neuron_params[agent_index][8]
    I_in = neuron_params[agent_index][9]

    # From https://www.izhikevich.org/publications/spikes.htm
    #   C v' = k (v - vrest)(v - vthr) - u + I
    #   u'   = a (b (v - vrest) - u)
    #   if v >= vpeak: v = vreset, u = u + d, spike
    v = internal_states[agent_index][0]
    u = internal_states[agent_index][1]

    dv = (k * (v - vrest) * (v - vthr) - u + I_synapse + I_bias + I_in) / C
    v = v + dt * dv * 1e3
    u += dt * 1e3 * (a * (b * (v - vrest) - u))

    s = 1 * (v >= vpeak)
    u = u + d * s
    v = v * (1 - s) + vreset * s

    internal_states[agent_index][0] = v
    internal_states[agent_index][1] = u

    output_spikes_tensor[agent_index][t_current % 2] = s

    buffer_idx = t_current % len(internal_states_buffer[agent_index])
    internal_states_buffer[agent_index][buffer_idx][0] = v
    internal_states_buffer[agent_index][buffer_idx][1] = u
    internal_states_buffer[agent_index][buffer_idx][2] = I_synapse
