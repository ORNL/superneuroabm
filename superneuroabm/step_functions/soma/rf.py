"""
Resonate-and-fire soma for SuperNeuroABM.

Dynamics:

    x = v - vrest

    dx/dt = -damping*x - omega*u + I_total
    du/dt =  omega*x - damping*u

where omega is the angular resonant frequency in rad/s.

The update uses the exact solution over one timestep, assuming the input
current is constant during that timestep.

Hyperparameter order must match the YAML configuration:

    [0] damping
    [1] omega
    [2] vthr
    [3] vrest
    [4] vreset
    [5] ureset
    [6] tref
    [7] I_in
    [8] scaling_factor

Internal-state order:

    [0] v
    [1] u
    [2] tcount
    [3] refractory_remaining
"""

import cupy as cp
from cupyx import jit


@jit.rawkernel(device="cuda")
def resonate_and_fire_soma_step_func(
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
):
    # ------------------------------------------------------------
    # Accumulate current from all incoming synapses.
    #
    # SuperNeuroABM requires a synapse to publish its output current
    # in internal_states[synapse_index][0].
    # ------------------------------------------------------------
    synapse_indices = locations[agent_index]

    I_synapse = 0.0

    for i in range(len(synapse_indices)):
        synapse_index = int(synapse_indices[i])

        if synapse_index >= 0 and not cp.isnan(synapse_indices[i]):
            I_synapse += internal_states[synapse_index][0]

    t_current = int(tick)

    # ------------------------------------------------------------
    # Hyperparameters
    # ------------------------------------------------------------
    damping = neuron_params[agent_index][0]
    omega = neuron_params[agent_index][1]
    vthr = neuron_params[agent_index][2]
    vrest = neuron_params[agent_index][3]
    vreset = neuron_params[agent_index][4]
    ureset = neuron_params[agent_index][5]
    tref = neuron_params[agent_index][6]
    I_in = neuron_params[agent_index][7]
    scaling_factor = neuron_params[agent_index][8]

    # ------------------------------------------------------------
    # Internal state
    # ------------------------------------------------------------
    v = internal_states[agent_index][0]
    u = internal_states[agent_index][1]
    tcount = internal_states[agent_index][2]
    refractory_remaining = internal_states[agent_index][3]

    # Include global bias, constant neuron input, and synaptic input.
    I_total = (
        scaling_factor * I_synapse
        + I_bias
        + I_in
    )

    # Use membrane voltage relative to rest.
    x = v - vrest
    v_previous = v

    # ------------------------------------------------------------
    # Exact integration
    #
    # A = [[-damping, -omega],
    #      [ omega,   -damping]]
    #
    # For a constant input during the timestep, the equilibrium is:
    #
    # x_eq = damping * I_total / (damping^2 + omega^2)
    # u_eq = omega   * I_total / (damping^2 + omega^2)
    # ------------------------------------------------------------
    denominator = damping * damping + omega * omega

    x_equilibrium = damping * I_total / denominator
    u_equilibrium = omega * I_total / denominator

    decay = cp.exp(-damping * dt)
    angle = omega * dt

    cosine = cp.cos(angle)
    sine = cp.sin(angle)

    x_relative = x - x_equilibrium
    u_relative = u - u_equilibrium

    x_next = (
        x_equilibrium
        + decay * (x_relative * cosine - u_relative * sine)
    )

    u_next = (
        u_equilibrium
        + decay * (x_relative * sine + u_relative * cosine)
    )

    v_next = vrest + x_next

    # ------------------------------------------------------------
    # Refractory-period update
    #
    # The resonator continues evolving during the refractory period,
    # but spike emission is suppressed.
    # ------------------------------------------------------------
    if refractory_remaining > 0.0:
        refractory_remaining -= dt

        if refractory_remaining < 0.0:
            refractory_remaining = 0.0

    not_refractory = refractory_remaining <= 0.0

    # Detect an upward threshold crossing. Using a crossing rather
    # than simply v >= vthr prevents repeated spikes while the
    # oscillation remains above threshold.
    crossed_threshold = (
        (v_previous < vthr)
        and (v_next >= vthr)
    )

    s = 1.0 * (crossed_threshold and not_refractory)

    # Reset after a spike.
    v_next = v_next * (1.0 - s) + vreset * s
    u_next = u_next * (1.0 - s) + ureset * s

    if s > 0.0:
        refractory_remaining = tref

    # ------------------------------------------------------------
    # Store updated neuron state
    # ------------------------------------------------------------
    internal_states[agent_index][0] = v_next
    internal_states[agent_index][1] = u_next
    internal_states[agent_index][2] = tcount + 1.0
    internal_states[agent_index][3] = refractory_remaining

    # Required SuperNeuroABM soma output contract.
    output_spikes_tensor[agent_index][t_current % 2] = s

    # Safe for both tracking-enabled and tracking-disabled modes.
    buffer_idx = t_current % len(
        internal_states_buffer[agent_index]
    )

    internal_states_buffer[agent_index][buffer_idx][0] = v_next
    internal_states_buffer[agent_index][buffer_idx][1] = u_next
    internal_states_buffer[agent_index][buffer_idx][2] = tcount + 1.0
    internal_states_buffer[agent_index][buffer_idx][3] = (
        refractory_remaining
    )
