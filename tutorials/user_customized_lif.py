"""
user_customized_lif -- a tutorial-owned LIF soma for the digits notebook.

This is what "bring your own soma" looks like: a CUDA kernel that lives next to
the notebook, is registered on the model at runtime, and needs no change to the
installed ``superneuroabm`` package.

Why this network does not use the built-in ``lif_soma``
-------------------------------------------------------
``digits_config.yaml`` starts every neuron at ``tlast: -.inf``, so the very first
spike is never blocked by the refractory test whatever ``tref`` is set to. The
built-in kernel resets branchlessly::

    s     = 1.0 * ((v >= vthr) and ...)
    tlast = tlast * (1 - s) + dt * tcount * s

With ``tlast = -inf`` the first spike evaluates ``-inf * 0.0``, which is NaN.
From then on ``tlast`` is NaN, every comparison against it is False, and the
refractory period is silently gone for the rest of the run.

This kernel writes the reset as an explicit ``if``, so ``-inf`` survives untouched
until the first spike and is then replaced by a finite time. Two related
corrections come with it:

* **Integration and spiking are separate decisions.** ``tref_allows_integration``
  lets ``v`` keep evolving during the refractory period; it must not let the
  neuron emit a second spike inside it.
* **The refractory test is inclusive and sentinel-free.**
  ``(current_time - tlast) >= tref`` -- a neuron that spiked at ``ts`` may spike
  again at exactly ``ts + tref``, and the test is naturally True while ``tlast``
  is ``-inf``, so no ``tlast > 0`` sentinel is needed.

Registration
------------
Before ``setup()`` -- ``register_soma_type`` raises ``RuntimeError`` afterwards::

    model.register_soma_type(
        name="user_customized_lif",
        step_func=user_customized_lif,
        step_func_path=Path("user_customized_lif.py").resolve(),
    )

``name`` must match the breed key under ``soma:`` in ``digits_config.yaml``, and
cannot be ``"lif_soma"``: that name is already taken by the built-in, and
re-registering it raises ``ValueError``.

The *file* basename and the *function* name both have to be globally unique.
``setup()`` code-generates ``from <module stem> import *`` for every registered
breed, so a file called ``lif.py`` or a function called ``lif_soma_step_func``
would shadow the built-in. See ``docs/CUSTOM_COMPONENTS.md``.

Hyperparameters -- the YAML key order *is* the index order read here:
    [0] C                        membrane capacitance (F)
    [1] R                        leak resistance (Ohm)
    [2] vthr                     spike threshold (mV)
    [3] tref                     absolute refractory period (s)
    [4] vrest                    resting potential (mV)
    [5] vreset                   post-spike reset potential (mV)
    [6] tref_allows_integration  non-zero: keep integrating while refractory
    [7] I_in                     constant input current (A)
    [8] scaling_factor           multiplier on the summed synaptic current

Internal states:
    [0] v       membrane potential (mV)
    [1] tcount  ticks elapsed since the start of the simulation
    [2] tlast   time of the last spike (s); may start at -inf
"""

import cupy as cp
from cupyx import jit


@jit.rawkernel(device="cuda")
def user_customized_lif(
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
    # Sum the current published by every incoming synapse at its slot 0.
    # SAGESim rewrites these lines into CSR form by pattern-matching, so the
    # idiom `synapse_indices = locations[agent_index]` followed by
    # `for i in range(len(synapse_indices))` has to be kept literally.
    synapse_indices = locations[agent_index]

    I_synapse = 0.0
    for i in range(len(synapse_indices)):
        synapse_index = int(synapse_indices[i])
        if synapse_index >= 0 and not cp.isnan(synapse_indices[i]):
            I_synapse += internal_states[synapse_index][0]

    t_current = int(tick)

    # ------------------------------------------------------------------
    # Hyperparameters
    # ------------------------------------------------------------------
    C = neuron_params[agent_index][0]  # membrane capacitance
    R = neuron_params[agent_index][1]  # leak resistance
    vthr = neuron_params[agent_index][2]  # spike threshold
    tref = neuron_params[agent_index][3]  # refractory period
    vrest = neuron_params[agent_index][4]  # resting potential
    vreset = neuron_params[agent_index][5]  # reset potential
    tref_allows_integration = neuron_params[agent_index][6]
    I_in = neuron_params[agent_index][7]  # constant input current
    scaling_factor = neuron_params[agent_index][8]  # synaptic current scaling

    # ------------------------------------------------------------------
    # Internal state
    # ------------------------------------------------------------------
    v = internal_states[agent_index][0]  # membrane potential
    tcount = internal_states[agent_index][1]  # ticks since simulation start
    tlast = internal_states[agent_index][2]  # time of last spike

    # Physical time of the update being computed on this tick.
    current_time = dt * tcount

    # True before the first spike, because tlast starts at -inf.
    # A neuron that spiked at time ts may spike again at ts + tref.
    refractory_complete = (current_time - tlast) >= tref

    # ------------------------------------------------------------------
    # Membrane update
    # ------------------------------------------------------------------
    dv = (vrest - v) / (R * C) + (I_synapse * scaling_factor + I_bias + I_in) / C

    # tref_allows_integration lets v evolve during the refractory period.
    # It deliberately does NOT let the neuron spike again inside it.
    integration_allowed = refractory_complete or tref_allows_integration != 0.0

    if integration_allowed:
        v += dv * dt

    # ------------------------------------------------------------------
    # Spike generation and reset
    # ------------------------------------------------------------------
    s = 0.0

    if refractory_complete and v >= vthr:
        s = 1.0
        tlast = current_time
        v = vreset

    # ------------------------------------------------------------------
    # Store updated state
    # ------------------------------------------------------------------
    next_tcount = tcount + 1.0

    internal_states[agent_index][0] = v
    internal_states[agent_index][1] = next_tcount
    internal_states[agent_index][2] = tlast

    output_spikes_tensor[agent_index][t_current % 2] = s

    # Safe buffer indexing: modulo keeps this in bounds when tracking is
    # disabled and the buffer is length 1, so t_current % 1 = 0 always.
    buffer_idx = t_current % len(internal_states_buffer[agent_index])
    internal_states_buffer[agent_index][buffer_idx][0] = v
    internal_states_buffer[agent_index][buffer_idx][1] = next_tcount
    internal_states_buffer[agent_index][buffer_idx][2] = tlast
