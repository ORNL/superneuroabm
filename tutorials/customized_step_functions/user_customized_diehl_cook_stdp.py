"""
user_customized_diehl_cook_stdp -- triplet STDP with hard weight bounds, the rule the
released code of Diehl & Cook (2015) runs, tutorial-owned.

Registered at runtime by tutorial 04 with ``model.register_learning_rule(step_func=...,
step_func_path=...)`` before ``setup()``. The returned id is what a synapse puts in
``stdp_type`` (``learning_hyperparameters[0]``) to select this rule.

The rule
--------
The authors' Brian 1 equations (peter-u-diehl/stdp-mnist)::

    dpre/dt   = -pre   / tc_pre
    dpost1/dt = -post1 / tc_post_1
    dpost2/dt = -post2 / tc_post_2

    on a pre spike:   w -= nu_pre * post1;          pre = 1
    on a post spike:  w += nu_post * pre * post2;   post1 = 1; post2 = 1
    w clipped to [wmin, wmax]

Depression is small and driven by the fast post trace; potentiation needs a recent pre
spike *and* a recent post spike (the slow ``post2``, read before it is reset), which is
what makes it a triplet rule. The paper's text also describes a power-law weight-dependent
variant; the code the authors released runs this one.

Timing
------
* The traces decay exactly, ``exp(-dt / tau)`` per tick, as Brian 1's linear updater does.
* The pre spike is the one that *arrived* at this synapse this tick, after the synapse's
  delay: the synapse kernel stores it in ``internal_states[0]`` and this rule runs right
  after it on the same thread.
* The post spike is the post soma's spike of this tick (slot ``t % 2`` of its spike
  ring), so pre and post are seen without relative skew; on a tick with both, the pre
  rule is applied first, as in Brian.

The rule is stepped every tick (``synapse_stepping="clocked"``), so it is not
``event_capable``.

Learning hyperparameters -- the YAML key order *is* the index order read here:

    [0] stdp_type   rule id (pinned; set per synapse from register_learning_rule())
    [1] tc_pre      presynaptic trace time constant (s)
    [2] tc_post_1   fast postsynaptic trace time constant (s), drives depression
    [3] tc_post_2   slow postsynaptic trace time constant (s), drives potentiation
    [4] nu_pre      depression learning rate
    [5] nu_post     potentiation learning rate
    [6] wmax        upper weight bound
    [7] wmin        lower weight bound

Learning internal states:

    [0] pre     presynaptic trace
    [1] post1   fast postsynaptic trace
    [2] post2   slow postsynaptic trace
"""

import cupy as cp
from cupyx import jit


@jit.rawkernel(device="cuda")
def user_customized_diehl_cook_stdp(
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

    tc_pre = learning_params[agent_index][1]
    tc_post_1 = learning_params[agent_index][2]
    tc_post_2 = learning_params[agent_index][3]
    nu_pre = learning_params[agent_index][4]
    nu_post = learning_params[agent_index][5]
    wmax = learning_params[agent_index][6]
    wmin = learning_params[agent_index][7]

    pre = learning_internal_states[agent_index][0]
    post1 = learning_internal_states[agent_index][1]
    post2 = learning_internal_states[agent_index][2]

    post_soma_index = locations[agent_index][1]

    # The pre spike as it arrived at this synapse this tick (set by the synapse
    # kernel just before), and the post soma's spike of this tick.
    pre_spike = internal_states[agent_index][0]
    post_spike = output_spikes_tensor[post_soma_index][t_current % 2]

    # --- Trace decay (exact) ---
    pre = pre * cp.exp(-dt / tc_pre)
    post1 = post1 * cp.exp(-dt / tc_post_1)
    post2 = post2 * cp.exp(-dt / tc_post_2)

    # --- Pre spike: depress by the fast post trace, then set the pre trace ---
    weight = weight - nu_pre * post1 * pre_spike
    pre = pre * (1.0 - pre_spike) + pre_spike

    # --- Post spike: potentiate by pre * post2 (before its reset), set both post traces ---
    weight = weight + nu_post * pre * post2 * post_spike
    post1 = post1 * (1.0 - post_spike) + post_spike
    post2 = post2 * (1.0 - post_spike) + post_spike

    weight = weight if weight <= wmax else wmax
    weight = weight if weight >= wmin else wmin
    synapse_params[agent_index][0] = weight

    learning_internal_states[agent_index][0] = pre
    learning_internal_states[agent_index][1] = post1
    learning_internal_states[agent_index][2] = post2

    buffer_idx = t_current % len(learning_internal_states_buffer[agent_index])
    learning_internal_states_buffer[agent_index][buffer_idx][0] = pre
    learning_internal_states_buffer[agent_index][buffer_idx][1] = post1
    learning_internal_states_buffer[agent_index][buffer_idx][2] = post2
