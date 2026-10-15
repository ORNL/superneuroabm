"""
learning_devices -- on-chip learning where the learning rule IS the device.

A memristive synapse does not "have" a learning rule; it has programming pulses. Each
presynaptic spike reads the device (current = w_scale * g) and, while learning is on,
applies one programming pulse whose polarity the neuron decides: SET (potentiate) if the
post-neuron's activity trace is above theta, RESET (depress) otherwise. What a pulse does
to the conductance is device physics, and that is what the kernel below models:

  nl         conductance-dependent step: SET steps shrink near g_max and RESET steps
             shrink near g_min ((1-g)^nl and (g-g_min)^nl). nl = 1 is the textbook
             multiplicative (soft-bounded) rule and is the "ideal" device here: it
             converges to a graded template. nl = 0 is a constant step with hard bounds
             (abrupt, state-independent switching), which binarizes the template
  sigma_c2c  cycle-to-cycle write noise: every pulse lands with a random error
  sigma_d2d  device-to-device variation: every device has its own SET and RESET step size
  n_levels   finite conductance states; a pulse smaller than one level switches the
             device to the next level with probability step/level (stochastic switching)
  p_stuck    stuck cells (half at g_min, half at g_max) that ignore programming

Why this cannot be pre-baked into weights: the weight is the device's state, and it is
produced during the run by the activity. And why NEST GPU's plasticity hook cannot host
it even with a new C++ rule: that hook is SynapseUpdate(syn_group, &weight, Dt) -- one
float of state, a spike-time difference, and parameters shared by the whole group. Write
noise and stochastic switching need a random number per pulse; d2d variation needs
per-synapse parameters. Neither is passed in.

The neuron publishes its trace in internal_states[1]; the device reads it from its
post-neuron, which is always on the same GPU (post-owns).

Layouts (YAML key order = index order):
  learn_relay     hp [0] thr                             is [0] v [1] x (unused)
  learn_lif       hp [0] vthr [1] tau_m [2] v_floor [3] tau_x
                  is [0] v [1] x (activity trace read by the neuron's devices)
  learn_wire      hp [0] weight [1] synaptic_delay       is [0] I
  learn_device    hp [0] weight (unused; the weight is the state g) [1] synaptic_delay
                     [2] w_scale [3] g_min [4] alpha_p [5] alpha_d [6] nl
                     [7] sigma_c2c [8] sigma_d2d [9] n_levels [10] p_stuck [11] theta
                     [12] t_learn_end (tick; programming stops here)
                     [13] g_ref (reference column: the neuron sees g - g_ref)
                  is [0] I [1] g
"""

import cupy as cp
from cupyx import jit

from sagesim.math_utils import rand_normal, rand_uniform_philox
from superneuroabm.step_functions.synapse.util import get_soma_spike


@jit.rawkernel(device="cuda")
def learn_relay(
    tick, agent_index, dt, I_bias, agent_ids, breeds, locations,
    neuron_params, learning_params, internal_states, learning_internal_states,
    synapse_history, input_spikes_tensor, output_spikes_tensor,
    internal_states_buffer, learning_internal_states_buffer,
):
    """Spike whenever the summed input reaches thr (input encoders and teachers)."""
    synapse_indices = locations[agent_index]
    I = 0.0
    for i in range(len(synapse_indices)):
        synapse_index = int(synapse_indices[i])
        if synapse_index >= 0 and not cp.isnan(synapse_indices[i]):
            I += internal_states[synapse_index][0]
    s = 0.0
    if I >= neuron_params[agent_index][0]:
        s = 1.0
    output_spikes_tensor[agent_index][int(tick) % 2] = s


@jit.rawkernel(device="cuda")
def learn_lif(
    tick, agent_index, dt, I_bias, agent_ids, breeds, locations,
    neuron_params, learning_params, internal_states, learning_internal_states,
    synapse_history, input_spikes_tensor, output_spikes_tensor,
    internal_states_buffer, learning_internal_states_buffer,
):
    """Float LIF (reset by subtraction) that also keeps an activity trace for its devices."""
    synapse_indices = locations[agent_index]
    I = 0.0
    for i in range(len(synapse_indices)):
        synapse_index = int(synapse_indices[i])
        if synapse_index >= 0 and not cp.isnan(synapse_indices[i]):
            I += internal_states[synapse_index][0]

    vthr = neuron_params[agent_index][0]
    tau_m = neuron_params[agent_index][1]
    v_floor = neuron_params[agent_index][2]
    tau_x = neuron_params[agent_index][3]

    v = internal_states[agent_index][0]
    x = internal_states[agent_index][1]
    v = v - v * dt / tau_m + I
    if v < v_floor:
        v = v_floor
    s = 0.0
    if v >= vthr:
        s = 1.0
        v = v - vthr
    x = x * cp.exp(-dt / tau_x) + s

    internal_states[agent_index][0] = v
    internal_states[agent_index][1] = x
    output_spikes_tensor[agent_index][int(tick) % 2] = s


@jit.rawkernel(device="cuda")
def learn_wire(
    tick, agent_index, dt, I_bias, agent_ids, breeds, locations,
    synapse_params, learning_params, internal_states, learning_internal_states,
    synapse_history, input_spikes_tensor, output_spikes_tensor,
    internal_states_buffer, learning_internal_states_buffer,
):
    """Fixed digital connection (external drive and teacher paths)."""
    spike = get_soma_spike(
        tick, agent_index, dt, I_bias, agent_ids, locations[agent_index][0], int(tick),
        input_spikes_tensor, output_spikes_tensor,
    )
    if spike == 0.0:
        if internal_states[agent_index][0] != 0.0:
            internal_states[agent_index][0] = 0.0
        return
    internal_states[agent_index][0] = spike * synapse_params[agent_index][0]


@jit.rawkernel(device="cuda")
def learn_device(
    tick, agent_index, dt, I_bias, agent_ids, breeds, locations,
    synapse_params, learning_params, internal_states, learning_internal_states,
    synapse_history, input_spikes_tensor, output_spikes_tensor,
    internal_states_buffer, learning_internal_states_buffer,
):
    """A memristive synapse that is read and programmed by the spikes that cross it."""
    t = int(tick)
    pre_soma_index = locations[agent_index][0]
    post_soma_index = locations[agent_index][1]
    spike = get_soma_spike(
        tick, agent_index, dt, I_bias, agent_ids, pre_soma_index, t,
        input_spikes_tensor, output_spikes_tensor,
    )
    if spike == 0.0:
        if internal_states[agent_index][0] != 0.0:
            internal_states[agent_index][0] = 0.0
        return

    w_scale = synapse_params[agent_index][2]
    g_min = synapse_params[agent_index][3]
    p_stuck = synapse_params[agent_index][10]

    g = internal_states[agent_index][1]
    stuck = 0
    u = rand_uniform_philox(0, agent_index, 1)   # frozen: this cell is stuck or it is not
    if u < 0.5 * p_stuck:
        g = g_min
        stuck = 1
    elif u < p_stuck:
        g = 1.0
        stuck = 1

    # Read, against a reference column at g_ref (so only above-reference conductance
    # drives the neuron; the common part of every template cancels).
    internal_states[agent_index][0] = spike * w_scale * (g - synapse_params[agent_index][13])

    # Program: one pulse per presynaptic spike, polarity from the post-neuron's trace.
    if t < synapse_params[agent_index][12] and stuck == 0:
        alpha_p = synapse_params[agent_index][4]
        alpha_d = synapse_params[agent_index][5]
        nl = synapse_params[agent_index][6]
        sigma_c2c = synapse_params[agent_index][7]
        sigma_d2d = synapse_params[agent_index][8]
        n_levels = synapse_params[agent_index][9]
        theta = synapse_params[agent_index][11]
        span = 1.0 - g_min

        if internal_states[post_soma_index][1] > theta:
            a = alpha_p * (1.0 + sigma_d2d * rand_normal(0, agent_index, 2))
            if a < 0.0:
                a = 0.0
            step = a * ((1.0 - g) / span) ** nl
        else:
            a = alpha_d * (1.0 + sigma_d2d * rand_normal(0, agent_index, 3))
            if a < 0.0:
                a = 0.0
            step = -a * ((g - g_min) / span) ** nl
        if sigma_c2c > 0.0:
            step = step + sigma_c2c * a * rand_normal(tick, agent_index, 4)

        g = g + step
        if g < g_min:
            g = g_min
        if g > 1.0:
            g = 1.0
        if n_levels > 1.0:
            pos = (g - g_min) / span * (n_levels - 1.0)
            k = cp.floor(pos)
            if rand_uniform_philox(tick, agent_index, 5) < pos - k:
                k = k + 1.0
            g = g_min + k * span / (n_levels - 1.0)
        internal_states[agent_index][1] = g
