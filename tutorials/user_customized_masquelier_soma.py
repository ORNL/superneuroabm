"""
user_customized_masquelier_soma -- the output neuron of Masquelier, Guyonneau & Thorpe
(2008), as a tutorial-owned CUDA kernel.

Like the digits tutorial's soma, this file lives next to the notebook and is registered
on the model at runtime; the installed ``superneuroabm`` package is not touched. Change
the dynamics here and re-run the notebook.

The model
---------
The three-variable leaky integrate-and-fire neuron of the Hathway & Goodman (2018)
re-implementation of the paper (their Brian2 equations, dimensionless units):

    du/dt = A * a / taus + (X * x - u) / taum     membrane potential
    dx/dt = -x / tausyn                           summed excitatory post-synaptic potential
    da/dt = -a / taus                             after-hyperpolarisation

An afferent spike through a synapse of weight w adds ``w`` to ``x`` (the synapse kernel
deposits it into this soma's inbox, see user_customized_masquelier_synapse.py).
When ``u`` crosses the threshold ``T`` outside the refractory period the neuron fires and
resets ``u = 2T`` (a 1-tick spike pulse), ``a = deltaa`` (the after-hyperpolarisation,
``A < 0`` pulls the membrane down while ``a`` decays) and ``x = 0`` (the pending input
is discarded, which is what stops the neuron firing twice on one pattern).

The integration is exact for one tick of length ``dt`` (Brian2's ``method='linear'``):
with ``decay_u = exp(-dt/taum)`` and ``decay_a = exp(-dt/taus)``

    u <- u * decay_u + c_a * a + c_x * x
    a <- a * decay_a
    x <- x * (1 - dt/tausyn) + (what the afferents deposited last tick)

where ``c_a = A / (taus * alpha) * (decay_a - decay_u)``,
``c_x = X / (taum * alpha) * (decay_a - decay_u)`` and ``alpha = 1/taum - 1/taus``.
``X = (taus/taum)^(taum/(taus-taum))`` normalises one unit of ``x`` to a peak EPSP of 1.

Where the synaptic variable lives
---------------------------------
In the paper every synapse has its own ``x_i``; the neuron sums them. Here the 2,000
synapses are event-woken delivery agents with no per-tick state, so the *sum* is kept
on the soma as one state variable ``x``. With all synapses sharing ``tausyn`` and the
same post-spike reset, summing the per-synapse recursion gives exactly the recursion
above (``docs/EVENT_DRIVEN_NOTES.md``, "Masquelier on the event path"). The reset of
``x`` uses this soma's spike from two ticks ago (slot ``t % 2`` of the width-2 spike ring,
read before this tick overwrites it): that is the tick on which the per-synapse kernels
of the clocked reference applied it, so the two formulations agree to float rounding.

Registration
------------
Before ``setup()``::

    model.register_soma_type(
        name="user_customized_masquelier_soma",
        step_func=user_customized_masquelier_soma,
        step_func_path=Path("user_customized_masquelier_soma.py").resolve(),
    )

``name`` is the breed key under ``soma:`` in ``masquelier_config.yaml``. The file
basename and the function name must both be globally unique: ``setup()`` generates
``from <module stem> import *`` for every registered breed, so a file called ``hg_lif.py``
or a function called ``hg_lif_soma_step_func`` would shadow the built-in of the same
name (``docs/CUSTOM_COMPONENTS.md``).

Hyperparameters -- the YAML key order *is* the index order read here:

    [0] T        spike threshold (0.5 * number of pattern afferents)
    [1] tref     absolute refractory period (s)
    [2] X        EPSP normalisation (taus/taum)^(taum/(taus-taum))
    [3] A        after-hyperpolarisation amplitude, -K2 * T
    [4] taus     after-hyperpolarisation time constant (s)
    [5] taum     membrane time constant (s)
    [6] deltaa   value of a after a spike
    [7] I_in     constant input (0 in the paper)
    [8] tausyn   EPSP time constant (s)

Internal states:

    [0] u        membrane potential
    [1] tcount   ticks elapsed since the start of the simulation
    [2] tlast    tick of the last spike (0 before the first)
    [3] a        after-hyperpolarisation variable
    [4] x        summed EPSP
"""

import cupy as cp
from cupyx import jit

from superneuroabm.step_functions.synaptic_input import read_synaptic_input


@jit.rawkernel(device="cuda")
def user_customized_masquelier_soma(
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

    # ------------------------------------------------------------------
    # Hyperparameters
    # ------------------------------------------------------------------
    T = neuron_params[agent_index][0]
    tref = neuron_params[agent_index][1]
    X = neuron_params[agent_index][2]
    A = neuron_params[agent_index][3]
    taus = neuron_params[agent_index][4]
    taum = neuron_params[agent_index][5]
    deltaa = neuron_params[agent_index][6]
    I_in = neuron_params[agent_index][7]
    tausyn = neuron_params[agent_index][8]

    # ------------------------------------------------------------------
    # Summed EPSP x: decay, add last tick's deposits from every afferent synapse
    # (inbox channel 0), and reset to 0 if this neuron fired two ticks ago.
    # ------------------------------------------------------------------
    deposits = read_synaptic_input(agent_index, synaptic_input, synaptic_input_idx, 0)
    fired_two_ticks_ago = output_spikes_tensor[agent_index][t_current % 2]
    x = (internal_states[agent_index][4] * (1.0 - dt / tausyn) + deposits) * (1.0 - fired_two_ticks_ago)
    internal_states[agent_index][4] = x

    # ------------------------------------------------------------------
    # Internal state
    # ------------------------------------------------------------------
    u = internal_states[agent_index][0]
    tcount = internal_states[agent_index][1]
    tlast = internal_states[agent_index][2]
    a = internal_states[agent_index][3]

    # ------------------------------------------------------------------
    # Exact one-tick integration of the linear system
    # ------------------------------------------------------------------
    decay_u = cp.exp(-dt / taum)
    decay_a = cp.exp(-dt / taus)
    alpha = 1.0 / taum - 1.0 / taus
    exp_diff = decay_a - decay_u
    c_a = A / (taus * alpha) * exp_diff
    c_x = X / (taum * alpha) * exp_diff

    u = u * decay_u + c_a * a + c_x * x + I_in
    a = a * decay_a

    # ------------------------------------------------------------------
    # Threshold, refractory period, reset
    # ------------------------------------------------------------------
    tref_ticks = tref / dt
    not_refractory = ((tcount - tlast) > tref_ticks) if tlast > 0 else True
    s = 1.0 * ((u >= T) and not_refractory)

    u = u * (1.0 - s) + (2.0 * T) * s      # spike pulse: u jumps to 2T for one tick
    a = a * (1.0 - s) + deltaa * s         # after-hyperpolarisation restarts
    tlast = tlast * (1.0 - s) + tcount * s

    # ------------------------------------------------------------------
    # Store updated state
    # ------------------------------------------------------------------
    internal_states[agent_index][0] = u
    internal_states[agent_index][1] = tcount + 1
    internal_states[agent_index][2] = tlast
    internal_states[agent_index][3] = a

    output_spikes_tensor[agent_index][t_current % 2] = s

    # Safe buffer indexing: modulo keeps this in bounds when tracking is
    # disabled and the buffer is length 1, so t_current % 1 = 0 always.
    buffer_idx = t_current % len(internal_states_buffer[agent_index])
    internal_states_buffer[agent_index][buffer_idx][0] = u
    internal_states_buffer[agent_index][buffer_idx][1] = tcount + 1
    internal_states_buffer[agent_index][buffer_idx][2] = tlast
    internal_states_buffer[agent_index][buffer_idx][3] = a
    internal_states_buffer[agent_index][buffer_idx][4] = x
