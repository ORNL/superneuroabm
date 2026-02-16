"""
Hathway-Goodman (2018) LIF soma step function.

Implements the exact 3-variable LIF used in the author's Brian2 reimplementation
of Masquelier et al. (2008):

  du/dt = (A*a)/taus + (X*x - u)/taum
  dx/dt = -x/tausyn          (handled by single_exp synapse)
  da/dt = -a/taus

where x is the summed synaptic current from all single_exp synapses,
A = -K2*T controls afterhyperpolarization strength, and X normalizes the
peak PSP to 1.0 per unit weight.

Integration uses the exact analytical solution for linear ODEs (matching
Brian2's method='linear').

On spike: u = 2*T, a = deltaa.  x = 0 is handled by the STDP kernel
(learning_rule_selector zeros synapse internal_state on post-spike).

Hyperparameters (neuron_params):
  [0] T         - spike threshold (e.g. 500)
  [1] tref      - absolute refractory period (s)
  [2] X         - PSP normalization = (taus/taum)^(taum/(taus-taum))
  [3] A         - afterhyperpolarization amplitude = -K2 * T
  [4] taus      - afterhyperpolarization / synaptic time constant (s)
  [5] taum      - membrane time constant (s)
  [6] deltaa    - afterhyperpolarization reset value on spike
  [7] I_in      - constant external input current

Internal state:
  [0] u      - membrane potential
  [1] tcount - time counter (ticks since start)
  [2] tlast  - time of last spike (in seconds)
  [3] a      - afterhyperpolarization variable
"""

from cupyx import jit
import cupy as cp


@jit.rawkernel(device="cuda")
def hg_lif_soma_step_func(
    tick,
    agent_index,
    globals,
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
    synapse_indices = locations[agent_index]

    # Sum synaptic currents (x_total) from all connected single_exp synapses
    x_total = 0.0
    for i in range(len(synapse_indices)):
        synapse_index = int(synapse_indices[i])
        if synapse_index >= 0 and not cp.isnan(synapse_indices[i]):
            x_total += internal_state[synapse_index][0]  # I_synapse

    t_current = int(tick)
    dt = globals[0]

    # Hyperparameters
    T = neuron_params[agent_index][0]
    tref = neuron_params[agent_index][1]
    X = neuron_params[agent_index][2]
    A = neuron_params[agent_index][3]
    taus = neuron_params[agent_index][4]
    taum = neuron_params[agent_index][5]
    deltaa = neuron_params[agent_index][6]
    I_in = neuron_params[agent_index][7]

    # Internal state
    u = internal_state[agent_index][0]
    tcount = internal_state[agent_index][1]
    tlast = internal_state[agent_index][2]
    a = internal_state[agent_index][3]

    # Exact integration (matching Brian2 method='linear')
    decay_u = cp.exp(-dt / taum)
    decay_a = cp.exp(-dt / taus)

    # Coupling coefficients for the exact solution:
    #   u(t+dt) = u(t)*decay_u + c_a*a(t) + c_x*x(t)
    # where c_a and c_x account for the exponentially decaying driving terms
    # being filtered by the membrane time constant.
    #   c_a = A / (taus * alpha) * (decay_a - decay_u)
    #   c_x = X / (taum * alpha) * (decay_a - decay_u)
    # with alpha = 1/taum - 1/taus  (note: tausyn = taus)
    alpha = 1.0 / taum - 1.0 / taus
    exp_diff = decay_a - decay_u  # exp(-dt/taus) - exp(-dt/taum)

    c_a = A / (taus * alpha) * exp_diff
    c_x = X / (taum * alpha) * exp_diff

    # Update membrane potential (exact linear integration)
    u = u * decay_u + c_a * a + c_x * x_total + I_in

    # Decay afterhyperpolarization variable (exact)
    a = a * decay_a

    # Check for spike (not in refractory period)
    # Use integer tick comparison to avoid float32 precision issues
    tref_ticks = tref / dt  # refractory period in ticks (e.g. 1.0 for 1ms)
    not_refractory = ((tcount - tlast) > tref_ticks) if tlast > 0 else True
    s = 1.0 * ((u >= T) and not_refractory)

    # On spike: reset u to 2*T, activate afterhyperpolarization
    u = u * (1.0 - s) + (2.0 * T) * s
    a = a * (1.0 - s) + deltaa * s
    tlast = tlast * (1.0 - s) + tcount * s

    # Update internal state
    internal_state[agent_index][0] = u
    internal_state[agent_index][1] = tcount + 1
    internal_state[agent_index][2] = tlast
    internal_state[agent_index][3] = a

    output_spikes_tensor[agent_index][t_current] = s

    # Safe buffer indexing
    buffer_idx = t_current % len(internal_states_buffer[agent_index])
    internal_states_buffer[agent_index][buffer_idx][0] = u
    internal_states_buffer[agent_index][buffer_idx][1] = tcount + 1
    internal_states_buffer[agent_index][buffer_idx][2] = tlast
    internal_states_buffer[agent_index][buffer_idx][3] = a
