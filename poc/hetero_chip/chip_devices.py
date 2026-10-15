"""
chip_devices -- device-level kernels for a heterogeneous neuromorphic chip.

Every neuron circuit and every synaptic device below is an ordinary SuperNeuroABM
step function: a ~30-line Python function compiled to CUDA, registered on the model at
runtime, and chosen per agent. Nothing here touches the installed package.

Three properties of the framework do the work:

1. **Per-agent kernels.** Each synapse runs the kernel of its own device technology, so
   SRAM cells, RRAM cells and PCM pairs can feed the same neuron, chosen synapse by
   synapse (see the `hetero_*` designs in run_chip_poc.py).
2. **Per-tick device state coupled to the neuron.** A passive-crossbar RRAM cell
   conducts on every tick, not only when its row is pulsed, because the column it sits
   on is the neuron's membrane node. The kernel reads the post-neuron's membrane voltage
   each tick. That read is valid on any number of GPUs because a synapse is always
   placed on its post-neuron's rank (post-owns).
3. **Randomness that belongs to the device.** `rand_normal(0, agent_index, salt)` is
   rewritten by codegen to a Philox draw keyed on (seed, tick=0, *global* agent id,
   salt). A tick-0 draw is therefore a frozen per-device constant (programming error,
   stuck-at fault, threshold mismatch) that is recomputed, never stored, and is the same
   number whichever GPU the device lands on. `rand_normal(tick, agent_index, salt)` is
   fresh per-tick noise with the same property. A seed is a chip instance.

Units: membrane voltage is in units of the firing threshold (vthr = 1); a synapse's
published current is the membrane increment it causes this tick; conductances are
normalized to G_max = 1.

Hyperparameter layouts (YAML key order is the index order read here):

  poisson_encoder        [0] rate (Hz)
  digital_lif / analog_lif
                         [0] vthr  [1] tau_m (s)  [2] bias (per tick)  [3] v_floor
    digital_lif          [4] frac_bits   (fixed-point membrane; 0 = float)
    analog_lif           [4] sigma_vth  [5] sigma_tau  (frozen mismatch, relative)
                         [6] sigma_noise (thermal noise per tick)
  sram_synapse           [0] weight [1] synaptic_delay [2] w_max [3] n_bits (0 = float)
  rram_synapse           [0] weight [1] synaptic_delay [2] w_max [3] g_min
                         [4] sigma_prog [5] sigma_read [6] p_sa0 [7] p_sa1
                         [8] passive (0 = 1T1R, 1 = selector-less/selector crossbar)
                         [9] v_swing (membrane threshold / read voltage)
                         [10] v_nl (selector nonlinearity; 0 = linear I-V)
                         [11] sigma_nl
                         [12] dynamic (1 = full device model, 0 = static-only)
  pcm_synapse            [0] weight [1] synaptic_delay [2] w_max [3] g_min
                         [4] sigma_prog [5] sigma_read [6] nu_mean [7] nu_sigma
                         [8] t0 (s) [9] t_deploy (s) [10] gdc (global drift comp.)
                         [11] dynamic (1 = full device model, 0 = static-only)

Static-only mode (dynamic = 0) is the ablation for "what could a fixed-weight simulator
such as NEST GPU represent?": it keeps every effect that is a constant per device --
programming error, stuck-at faults, PCM drift at the deployment age, drift compensation,
and (passive arrays) the cell's leak linearized at 0 V, which sums to a constant extra
leak conductance per neuron -- and drops every effect that happens during the run: read
noise, the read current's dependence on the membrane voltage, and the selector's
nonlinear I-V. Same device draws, so it is the same chip instance.

Internal states: somas [0] v (membrane); synapses [0] I (membrane increment).
"""

import cupy as cp
from cupyx import jit

from sagesim.math_utils import rand_normal, rand_uniform_philox
from superneuroabm.step_functions.synapse.util import get_soma_spike


# ----------------------------------------------------------------------------- somas


@jit.rawkernel(device="cuda")
def chip_poisson_encoder(
    tick, agent_index, dt, I_bias, agent_ids, breeds, locations,
    neuron_params, learning_params, internal_states, learning_internal_states,
    synapse_history, input_spikes_tensor, output_spikes_tensor,
    internal_states_buffer, learning_internal_states_buffer,
):
    """On-chip rate encoder: spikes with probability rate*dt each tick."""
    t = int(tick)
    rate = neuron_params[agent_index][0]
    s = 0.0
    if rate > 0.0:
        if rand_uniform_philox(tick, agent_index, 1) < rate * dt:
            s = 1.0
    output_spikes_tensor[agent_index][t % 2] = s


@jit.rawkernel(device="cuda")
def chip_digital_lif(
    tick, agent_index, dt, I_bias, agent_ids, breeds, locations,
    neuron_params, learning_params, internal_states, learning_internal_states,
    synapse_history, input_spikes_tensor, output_spikes_tensor,
    internal_states_buffer, learning_internal_states_buffer,
):
    """Digital LIF core: deterministic, membrane held in a fixed-point register."""
    synapse_indices = locations[agent_index]
    I = 0.0
    for i in range(len(synapse_indices)):
        synapse_index = int(synapse_indices[i])
        if synapse_index >= 0 and not cp.isnan(synapse_indices[i]):
            I += internal_states[synapse_index][0]

    t = int(tick)
    vthr = neuron_params[agent_index][0]
    tau_m = neuron_params[agent_index][1]
    bias = neuron_params[agent_index][2]
    v_floor = neuron_params[agent_index][3]
    frac_bits = neuron_params[agent_index][4]

    v = internal_states[agent_index][0]
    v = v - v * dt / tau_m + I + bias
    if frac_bits > 0.0:
        lsb = 2.0 ** (-frac_bits)
        v = cp.floor(v / lsb) * lsb
    if v < v_floor:
        v = v_floor

    s = 0.0
    if v >= vthr:
        s = 1.0
        v = v - vthr  # reset by subtraction

    internal_states[agent_index][0] = v
    output_spikes_tensor[agent_index][t % 2] = s


@jit.rawkernel(device="cuda")
def chip_analog_lif(
    tick, agent_index, dt, I_bias, agent_ids, breeds, locations,
    neuron_params, learning_params, internal_states, learning_internal_states,
    synapse_history, input_spikes_tensor, output_spikes_tensor,
    internal_states_buffer, learning_internal_states_buffer,
):
    """Analog LIF circuit: frozen threshold/leak mismatch plus thermal noise."""
    synapse_indices = locations[agent_index]
    I = 0.0
    for i in range(len(synapse_indices)):
        synapse_index = int(synapse_indices[i])
        if synapse_index >= 0 and not cp.isnan(synapse_indices[i]):
            I += internal_states[synapse_index][0]

    t = int(tick)
    sigma_vth = neuron_params[agent_index][4]
    sigma_tau = neuron_params[agent_index][5]
    sigma_noise = neuron_params[agent_index][6]

    # Fabrication mismatch: tick-0 draws keyed on this circuit's global id.
    vthr = neuron_params[agent_index][0] * (1.0 + sigma_vth * rand_normal(0, agent_index, 2))
    tau_m = neuron_params[agent_index][1] * (1.0 + sigma_tau * rand_normal(0, agent_index, 3))
    if vthr < 0.05:
        vthr = 0.05
    if tau_m < dt:
        tau_m = dt
    bias = neuron_params[agent_index][2]
    v_floor = neuron_params[agent_index][3]

    v = internal_states[agent_index][0]
    v = v - v * dt / tau_m + I + bias + sigma_noise * rand_normal(tick, agent_index, 4)
    if v < v_floor:
        v = v_floor

    s = 0.0
    if v >= vthr:
        s = 1.0
        v = v - vthr

    internal_states[agent_index][0] = v
    output_spikes_tensor[agent_index][t % 2] = s


# -------------------------------------------------------------------------- synapses


@jit.rawkernel(device="cuda")
def chip_sram_synapse(
    tick, agent_index, dt, I_bias, agent_ids, breeds, locations,
    synapse_params, learning_params, internal_states, learning_internal_states,
    synapse_history, input_spikes_tensor, output_spikes_tensor,
    internal_states_buffer, learning_internal_states_buffer,
):
    """Digital SRAM cell: signed n-bit weight, exact."""
    t = int(tick)
    pre_soma_index = locations[agent_index][0]
    spike = get_soma_spike(
        tick, agent_index, dt, I_bias, agent_ids, pre_soma_index, t,
        input_spikes_tensor, output_spikes_tensor,
    )
    if spike == 0.0:
        if internal_states[agent_index][0] != 0.0:
            internal_states[agent_index][0] = 0.0
        return

    w = synapse_params[agent_index][0]
    w_max = synapse_params[agent_index][2]
    n_bits = synapse_params[agent_index][3]
    if n_bits > 0.0:
        levels = 2.0 ** (n_bits - 1.0) - 1.0
        q = cp.rint(w / w_max * levels)
        if q > levels:
            q = levels
        if q < -levels:
            q = -levels
        w = q / levels * w_max

    internal_states[agent_index][0] = spike * w


@jit.rawkernel(device="cuda")
def chip_rram_synapse(
    tick, agent_index, dt, I_bias, agent_ids, breeds, locations,
    synapse_params, learning_params, internal_states, learning_internal_states,
    synapse_history, input_spikes_tensor, output_spikes_tensor,
    internal_states_buffer, learning_internal_states_buffer,
):
    """Single-ended RRAM cell on a column that IS the neuron's membrane node.

    The weight is encoded around a mid-range reference conductance g_ref, which an
    ideal reference column (at virtual ground) subtracts. The cell's own column is not
    at virtual ground: it sits at the post-neuron's membrane voltage. So

      - a read sees V_read - v_col (read current saturates as the membrane charges);
      - in a passive array (no access transistor) the cell also conducts when its row
        is idle, leaking the membrane through its own conductance on EVERY tick;
      - a selector's sinh I-V (per-device nonlinearity v_nl) suppresses that leak.
    """
    t = int(tick)
    pre_soma_index = locations[agent_index][0]
    post_soma_index = locations[agent_index][1]
    spike = get_soma_spike(
        tick, agent_index, dt, I_bias, agent_ids, pre_soma_index, t,
        input_spikes_tensor, output_spikes_tensor,
    )
    passive = synapse_params[agent_index][8]
    if spike == 0.0 and passive == 0.0:
        # 1T1R: the access transistor is off, the cell is disconnected.
        if internal_states[agent_index][0] != 0.0:
            internal_states[agent_index][0] = 0.0
        return

    w = synapse_params[agent_index][0]
    w_max = synapse_params[agent_index][2]
    g_min = synapse_params[agent_index][3]
    sigma_prog = synapse_params[agent_index][4]
    sigma_read = synapse_params[agent_index][5]
    p_sa0 = synapse_params[agent_index][6]
    p_sa1 = synapse_params[agent_index][7]
    v_swing = synapse_params[agent_index][9]
    v_nl = synapse_params[agent_index][10]
    sigma_nl = synapse_params[agent_index][11]
    dynamic = synapse_params[agent_index][12]

    half = 0.5 * (1.0 - g_min)
    g_ref = g_min + half

    # Programmed conductance, with this device's frozen programming error.
    x = w / w_max
    if x > 1.0:
        x = 1.0
    if x < -1.0:
        x = -1.0
    g = g_ref + half * x + sigma_prog * (1.0 - g_min) * rand_normal(0, agent_index, 1)

    # Stuck-at faults, also frozen per device.
    u = rand_uniform_philox(0, agent_index, 2)
    if u < p_sa0:
        g = g_min
    elif u > 1.0 - p_sa1:
        g = 1.0
    if g < g_min:
        g = g_min
    if g > 1.0:
        g = 1.0

    # Voltage across the cell: row at V_read = 1 when pulsed (else 0), column at the
    # post-neuron's membrane. The post soma is local (post-owns), and it has already
    # been updated this tick (somas run at priority 0, synapses at 100).
    v_col = internal_states[post_soma_index][0] * v_swing
    v_nl_dev = v_nl
    if v_nl > 0.0:
        v_nl_dev = v_nl * (1.0 + sigma_nl * rand_normal(0, agent_index, 4))

    if dynamic > 0.0:
        # Cycle-to-cycle read noise.
        if spike != 0.0:
            g = g * (1.0 + sigma_read * rand_normal(tick, agent_index, 3))
        v_dev = spike - v_col
        if v_nl > 0.0:
            f = cp.sinh(v_dev / v_nl_dev) / cp.sinh(1.0 / v_nl_dev)
        else:
            f = v_dev
        I = w_max * (g * f - g_ref * spike) / half
    else:
        # A fixed weight, plus (passive) a leak linear in v with a fixed slope per cell.
        slope = 1.0
        if v_nl > 0.0:
            slope = 1.0 / (v_nl_dev * cp.sinh(1.0 / v_nl_dev))
        I = w_max * ((g - g_ref) * spike - passive * g * slope * v_col) / half
    internal_states[agent_index][0] = I


@jit.rawkernel(device="cuda")
def chip_pcm_synapse(
    tick, agent_index, dt, I_bias, agent_ids, breeds, locations,
    synapse_params, learning_params, internal_states, learning_internal_states,
    synapse_history, input_spikes_tensor, output_spikes_tensor,
    internal_states_buffer, learning_internal_states_buffer,
):
    """Differential PCM pair (G+ - G-) with power-law conductance drift.

    G(t) = G_prog * (t / t0) ** -nu, with nu drawn per device. Optional global drift
    compensation (gdc) rescales by the array's mean drift, which removes the common
    decay but not the device-to-device spread in nu.
    """
    t = int(tick)
    pre_soma_index = locations[agent_index][0]
    spike = get_soma_spike(
        tick, agent_index, dt, I_bias, agent_ids, pre_soma_index, t,
        input_spikes_tensor, output_spikes_tensor,
    )
    if spike == 0.0:
        if internal_states[agent_index][0] != 0.0:
            internal_states[agent_index][0] = 0.0
        return

    w = synapse_params[agent_index][0]
    w_max = synapse_params[agent_index][2]
    g_min = synapse_params[agent_index][3]
    sigma_prog = synapse_params[agent_index][4]
    sigma_read = synapse_params[agent_index][5]
    nu_mean = synapse_params[agent_index][6]
    nu_sigma = synapse_params[agent_index][7]
    t0 = synapse_params[agent_index][8]
    t_deploy = synapse_params[agent_index][9]
    gdc = synapse_params[agent_index][10]
    dynamic = synapse_params[agent_index][11]

    x = w / w_max
    if x > 1.0:
        x = 1.0
    if x < -1.0:
        x = -1.0
    gp = g_min
    gm = g_min
    if x > 0.0:
        gp = g_min + (1.0 - g_min) * x
    else:
        gm = g_min - (1.0 - g_min) * x
    gp = gp + sigma_prog * (1.0 - g_min) * rand_normal(0, agent_index, 1)
    gm = gm + sigma_prog * (1.0 - g_min) * rand_normal(0, agent_index, 2)
    if gp < g_min:
        gp = g_min
    if gp > 1.0:
        gp = 1.0
    if gm < g_min:
        gm = g_min
    if gm > 1.0:
        gm = 1.0

    nu_p = nu_mean + nu_sigma * rand_normal(0, agent_index, 3)
    nu_m = nu_mean + nu_sigma * rand_normal(0, agent_index, 4)
    if nu_p < 0.0:
        nu_p = 0.0
    if nu_m < 0.0:
        nu_m = 0.0
    t_age = t_deploy / t0
    if dynamic > 0.0:
        t_age = (t_deploy + t * dt) / t0
    if t_age > 1.0:
        gp = gp * t_age ** (-nu_p)
        gm = gm * t_age ** (-nu_m)

    if dynamic > 0.0:
        gp = gp * (1.0 + sigma_read * rand_normal(tick, agent_index, 5))
        gm = gm * (1.0 + sigma_read * rand_normal(tick, agent_index, 6))

    I = spike * w_max * (gp - gm) / (1.0 - g_min)
    if gdc > 0.0 and t_age > 1.0:
        I = I * t_age ** nu_mean
    internal_states[agent_index][0] = I
