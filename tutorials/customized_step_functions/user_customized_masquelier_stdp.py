"""
user_customized_masquelier_stdp -- additive nearest-neighbour STDP (Masquelier et al.
2008), tutorial-owned, in the event-woken form.

Registered at runtime with ``model.register_learning_rule(step_func=...,
step_func_path=..., event_capable=True)`` before ``setup()``. The returned id is what a
synapse puts in ``stdp_type`` (``learning_hyperparameters[0]``) to select this rule.

The rule
--------
Each synapse keeps a presynaptic trace ``pre_trace`` (time constant ``tau_plus``) and a
postsynaptic trace ``post_trace`` (``tau_minus``):

* afferent spike:  ``pre_trace = a_plus`` (not accumulated: nearest-neighbour), the
  weight is depressed by ``post_trace`` and ``post_trace`` is cleared;
* output spike:    ``post_trace = a_minus``, the weight is potentiated by ``pre_trace``
  and ``pre_trace`` is cleared;
* the weight is clipped to ``[wmin, wmax]``.

So only the *nearest* pre/post pair counts, each pairing is used once, and the update is
additive (independent of the current weight). ``a_minus = 0.85 * a_plus``: depression
slightly weaker per pairing, but far more frequent before the neuron has become
selective, which is what drives the non-pattern weights to 0.

When the rule runs
------------------
With ``synapse_stepping="event"`` a synapse's rule is stepped only on the ticks on which
its afferent or the output neuron fired, so between two runs the traces have to be
decayed in closed form over the ``n`` ticks the rule slept: ``exp(-n dt / tau)``, which
is exact (and the same decay Brian2 applies). The tick the rule last ran is kept in
``learning_internal_states[3]`` (``last_tick`` in the config); ``event_capable=True``
at registration tells the model the rule does this.

Learning hyperparameters -- the YAML key order *is* the index order read here:

    [0] stdp_type   rule id (pinned; set per synapse from register_learning_rule())
    [1] tau_plus    presynaptic trace time constant (s)
    [2] tau_minus   postsynaptic trace time constant (s)
    [3] a_plus      potentiation per pairing
    [4] a_minus     depression per pairing
    [5] wmin        lower weight bound
    [6] wmax        upper weight bound

Learning internal states:

    [0] pre_trace
    [1] post_trace
    [2] dW          last weight change (for inspection)
    [3] last_tick   tick the rule last ran (pinned at index 3 for event stepping)
"""

import cupy as cp
from cupyx import jit

from superneuroabm.step_functions.synapse.util import get_soma_spike


@jit.rawkernel(device="cuda")
def user_customized_masquelier_stdp(
    tick,
    agent_index,
    dt,
    I_bias,
    agent_ids,
    breeds,
    locations,
    synapse_params,
    learning_params,
    internal_states,
    learning_internal_states,
    synapse_history,
    input_spikes_tensor,
    output_spikes_tensor,
    internal_states_buffer,
    learning_internal_states_buffer,
):
    t_current = int(tick)

    weight = synapse_params[agent_index][0]

    tau_plus = learning_params[agent_index][1]
    tau_minus = learning_params[agent_index][2]
    a_plus = learning_params[agent_index][3]
    a_minus = learning_params[agent_index][4]
    wmin = learning_params[agent_index][5]
    wmax = learning_params[agent_index][6]

    pre_trace = learning_internal_states[agent_index][0]
    post_trace = learning_internal_states[agent_index][1]

    _nbrs = locations[agent_index]
    pre_soma_index = _nbrs[0]
    post_soma_index = _nbrs[1]

    pre_spike = get_soma_spike(
        tick, agent_index, dt, I_bias, agent_ids,
        pre_soma_index, t_current,
        input_spikes_tensor, output_spikes_tensor,
    )
    post_spike = get_soma_spike(
        tick, agent_index, dt, I_bias, agent_ids,
        post_soma_index, t_current,
        input_spikes_tensor, output_spikes_tensor,
    )

    # --- Decay the traces over the ticks since the rule last ran (exact) ---
    n_ticks = t_current - int(learning_internal_states[agent_index][3])
    if n_ticks < 1:
        n_ticks = 1
    pre_trace = pre_trace * cp.exp(-n_ticks * dt / tau_plus)
    post_trace = post_trace * cp.exp(-n_ticks * dt / tau_minus)

    # --- Afferent spike: depress by the post trace, restart the pre trace ---
    pre_trace = pre_trace * (1.0 - pre_spike) + pre_spike * a_plus
    ltd = post_trace * pre_spike
    post_trace = post_trace * (1.0 - pre_spike)

    # --- Output spike: potentiate by the pre trace, restart the post trace ---
    post_trace = post_trace * (1.0 - post_spike) + post_spike * a_minus
    ltp = pre_trace * post_spike
    pre_trace = pre_trace * (1.0 - post_spike)

    # --- Additive update with hard bounds ---
    dW = ltp - ltd
    weight += dW
    weight = weight if weight <= wmax else wmax
    weight = weight if weight >= wmin else wmin
    synapse_params[agent_index][0] = weight

    learning_internal_states[agent_index][0] = pre_trace
    learning_internal_states[agent_index][1] = post_trace
    learning_internal_states[agent_index][2] = dW
    learning_internal_states[agent_index][3] = 1.0 * t_current

    buffer_idx = t_current % len(learning_internal_states_buffer[agent_index])
    learning_internal_states_buffer[agent_index][buffer_idx][0] = pre_trace
    learning_internal_states_buffer[agent_index][buffer_idx][1] = post_trace
    learning_internal_states_buffer[agent_index][buffer_idx][2] = dW
