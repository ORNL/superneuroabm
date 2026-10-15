"""
Leaky integrate-and-fire soma with the synaptic current filtered on the soma, one
exponential filter per inbox channel (NEST ``iaf_psc_exp`` / ``iaf_psc_exp_multisynapse``).

The afferent delivery synapses deposit ``spike * scale * weight`` into the soma's inbox
(see step_functions/synaptic_input.py); this kernel decays channel c with
``tau_syn_c``, adds the deposits and feeds the sum into the membrane. With one time
constant this is the same recursion as summing per-synapse exponential currents:

    sum_i I_i(t) = (sum_i I_i(t-1)) * (1 - dt/tau) + sum_i s_i(t) * scale_i * w_i

A model whose afferents use K distinct time constants needs K channels
(``NeuromorphicModel(synaptic_channels=K)``) and a config with ``tau_syn_0..K-1`` /
``I_syn_0..K-1``.

Hyperparameters (neuron_params): [0] C, [1] R, [2] vthr, [3] tref, [4] vrest,
[5] vreset, [6] tref_allows_integration, [7] I_in, [8] scaling_factor,
[9 + c] tau_syn_c (s). Internal state: [0] v, [1] tcount, [2] tlast, [3 + c] I_syn_c.
The history buffer records the summed current at [3].
"""

from cupyx import jit
import cupy as cp

from superneuroabm.step_functions.synaptic_input import synaptic_current


@jit.rawkernel(device="cuda")
def lif_soma_step_func(
    tick,
    agent_index,
    dt,
    I_bias,
    agent_ids,
    breeds,
    locations,
    neuron_params,  # C, R, vthr, tref, vrest, vreset, tref_allows_integration, I_in, scaling_factor, tau_syn_0, tau_syn_1, ...
    learning_params,
    internal_states,  # v, tcount, tlast, I_syn_0, I_syn_1, ...
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

    I_synapse = synaptic_current(agent_index, dt, neuron_params, 9, internal_states, 3,
                                 synaptic_input, synaptic_input_idx)

    v = internal_states[agent_index][0]
    tcount = internal_states[agent_index][1]
    tlast = internal_states[agent_index][2]

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

    buffer_idx = t_current % len(internal_states_buffer[agent_index])
    internal_states_buffer[agent_index][buffer_idx][0] = v
    internal_states_buffer[agent_index][buffer_idx][1] = internal_states[agent_index][1] + 1
    internal_states_buffer[agent_index][buffer_idx][2] = tlast
    internal_states_buffer[agent_index][buffer_idx][3] = I_synapse
