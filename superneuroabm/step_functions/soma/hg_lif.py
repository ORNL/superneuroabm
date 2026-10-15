"""
Hathway-Goodman (2018) LIF soma step function.

Implements the exact 3-variable LIF used in the author's Brian2 reimplementation
of Masquelier et al. (2008):

  du/dt = (A*a)/taus + (X*x - u)/taum
  dx/dt = -x/tausyn
  da/dt = -a/taus

where x is the summed synaptic current. x lives on the soma: the afferent delivery
synapses deposit into the inbox and ``synaptic_current`` decays one filter per channel
with ``tau_syn_c`` (set it to tausyn; the default config uses taus, as in the paper).
A = -K2*T controls afterhyperpolarization strength, and X normalizes the peak PSP to
1.0 per unit weight. Integration uses the exact analytical solution for the linear
membrane ODE (matching Brian2's method='linear').

On spike: u = 2*T, a = deltaa. (The paper's x = 0 reset at a post spike is a modelling
choice of that experiment, applied by its own soma kernel; this built-in does not.)

Hyperparameters (neuron_params):
  [0] T, [1] tref, [2] X, [3] A, [4] taus, [5] taum, [6] deltaa, [7] I_in,
  [8 + c] tau_syn_c
Internal state:
  [0] u, [1] tcount, [2] tlast, [3] a, [4 + c] I_syn_c
"""

from cupyx import jit
import cupy as cp

from superneuroabm.step_functions.synaptic_input import synaptic_current


@jit.rawkernel(device="cuda")
def hg_lif_soma_step_func(
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

    x_total = synaptic_current(agent_index, dt, neuron_params, 8, internal_states, 4,
                               synaptic_input, synaptic_input_idx)

    T = neuron_params[agent_index][0]
    tref = neuron_params[agent_index][1]
    X = neuron_params[agent_index][2]
    A = neuron_params[agent_index][3]
    taus = neuron_params[agent_index][4]
    taum = neuron_params[agent_index][5]
    deltaa = neuron_params[agent_index][6]
    I_in = neuron_params[agent_index][7]

    u = internal_states[agent_index][0]
    tcount = internal_states[agent_index][1]
    tlast = internal_states[agent_index][2]
    a = internal_states[agent_index][3]

    # Exact integration (matching Brian2 method='linear'):
    #   u(t+dt) = u(t)*decay_u + c_a*a(t) + c_x*x(t)
    decay_u = cp.exp(-dt / taum)
    decay_a = cp.exp(-dt / taus)
    alpha = 1.0 / taum - 1.0 / taus
    exp_diff = decay_a - decay_u
    c_a = A / (taus * alpha) * exp_diff
    c_x = X / (taum * alpha) * exp_diff

    u = u * decay_u + c_a * a + c_x * x_total + I_in
    a = a * decay_a

    tref_ticks = tref / dt
    not_refractory = ((tcount - tlast) > tref_ticks) if tlast > 0 else True
    s = 1.0 * ((u >= T) and not_refractory)

    u = u * (1.0 - s) + (2.0 * T) * s
    a = a * (1.0 - s) + deltaa * s
    tlast = tlast * (1.0 - s) + tcount * s

    internal_states[agent_index][0] = u
    internal_states[agent_index][1] = tcount + 1
    internal_states[agent_index][2] = tlast
    internal_states[agent_index][3] = a

    output_spikes_tensor[agent_index][t_current % 2] = s

    buffer_idx = t_current % len(internal_states_buffer[agent_index])
    internal_states_buffer[agent_index][buffer_idx][0] = u
    internal_states_buffer[agent_index][buffer_idx][1] = tcount + 1
    internal_states_buffer[agent_index][buffer_idx][2] = tlast
    internal_states_buffer[agent_index][buffer_idx][3] = a
    internal_states_buffer[agent_index][buffer_idx][4] = x_total
