"""
LIF soma with the synaptic current filtered on the soma (NEST ``iaf_psc_exp`` layout).

Same membrane dynamics as soma/lif.py. The difference is where the exponential synaptic
current lives: here it is one state variable per soma, ``I_syn``, decayed every tick
with ``tau_syn`` and fed by what the afferent synapses deposited into the inbox on the
previous tick (push mode; pair it with synapse/delivery.py). With equal time constants
this is the same recursion as summing per-synapse single_exp currents:

    sum_i I_i(t) = (sum_i I_i(t-1)) * (1 - dt/tau) + sum_i s_i(t) * scale_i * w_i

so spike trains match the lif_soma + single_exp_synapse model (verified identical on a
2,000-neuron / 30-tick dump; trajectories part later from float rounding).

Hyperparameters (neuron_params): the nine of lif_soma, then [9] tau_syn (s).
Internal state: [0] v, [1] tcount, [2] tlast, [3] I_syn.
"""

from cupyx import jit
import cupy as cp

from superneuroabm.step_functions.synaptic_input import read_synaptic_input


@jit.rawkernel(device="cuda")
def lif_psc_soma_step_func(
    tick,
    agent_index,
    dt,
    I_bias,
    agent_ids,
    breeds,
    locations,
    neuron_params,  # C, R, vthr, tref, vrest, vreset, tref_allows_integration, I_in, scaling_factor, tau_syn
    learning_params,
    internal_states,  # v, tcount, tlast, I_syn
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
    tau_syn = neuron_params[agent_index][9]

    # Synaptic current on the soma: decay, then add last tick's deposits
    inbox = read_synaptic_input(agent_index, synaptic_input, synaptic_input_idx, 0)
    I_synapse = internal_states[agent_index][3] * (1 - dt / tau_syn) + inbox
    internal_states[agent_index][3] = I_synapse

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
