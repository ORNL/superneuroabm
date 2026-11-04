"""
user_customized_naive_stdp -- pair-wise STDP that is right only when stepped every tick.
Tutorial-owned, a counter-example: do not use it in a real model.

This is the built-in ``exp_pair_wise_stdp`` with one line changed. Each time it runs it
decays its two traces by ONE tick's factor, ``1 - dt/tau``:

    pre_trace  = pre_trace  * (1 - dt/tau_pre)  + pre_spike  * a_exp_pre
    post_trace = post_trace * (1 - dt/tau_post) + post_spike * a_exp_post
    dW = pre_trace * post_spike - post_trace * pre_spike

That is correct when the rule runs on every tick (``synapse_stepping="clocked"``). With
event stepping a rule runs only on the ticks after its pre or post neuron fired, so a
trace that should have decayed over the 20 ticks between two spikes has decayed over one,
and the weight change comes out too large. The built-in rule decays by
``trace_decay(dt, tau, n)`` = ``(1 - dt/tau)^n`` over the ``n`` ticks since it last ran
(the tick kept in ``learning_internal_states[3]``), which gives the same traces in both
steppings.

Tutorial 05 registers it with ``event_capable=True`` on purpose, which is a false promise,
to show what the promise means:

    model.register_learning_rule(
        step_func=user_customized_naive_stdp,
        step_func_path=Path("customized_step_functions/user_customized_naive_stdp.py").resolve(),
        event_capable=True,   # FALSE: this rule does not catch up over the ticks it slept
    )

Learning hyperparameters -- the YAML key order *is* the index order read here (the
built-in rule's layout):

    [0] stdp_type, [1] tau_pre_stdp, [2] tau_post_stdp, [3] a_exp_pre, [4] a_exp_post,
    [5] stdp_history_length (unused)

Learning internal states: [0] pre_trace, [1] post_trace, [2] dW, [3] last_tick.
"""

from cupyx import jit

from superneuroabm.step_functions.synapse.util import get_soma_spike


@jit.rawkernel(device="cuda")
def user_customized_naive_stdp(
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
    pre_trace = learning_internal_states[agent_index][0]
    post_trace = learning_internal_states[agent_index][1]

    pre_soma_index = locations[agent_index][0]
    post_soma_index = locations[agent_index][1]
    pre_soma_spike = get_soma_spike(
        tick, agent_index, dt, I_bias, agent_ids, pre_soma_index,
        t_current, input_spikes_tensor, output_spikes_tensor,
    )
    post_soma_spike = get_soma_spike(
        tick, agent_index, dt, I_bias, agent_ids, post_soma_index,
        t_current, input_spikes_tensor, output_spikes_tensor,
    )

    tau_pre_stdp = learning_params[agent_index][1]
    tau_post_stdp = learning_params[agent_index][2]
    a_exp_pre = learning_params[agent_index][3]
    a_exp_post = learning_params[agent_index][4]

    # The one difference from the built-in rule: one tick's decay per run, however many
    # ticks have passed since the last run.
    pre_trace = pre_trace * (1.0 - dt / tau_pre_stdp) + pre_soma_spike * a_exp_pre
    post_trace = post_trace * (1.0 - dt / tau_post_stdp) + post_soma_spike * a_exp_post
    dW = pre_trace * post_soma_spike - post_trace * pre_soma_spike

    synapse_params[agent_index][0] = synapse_params[agent_index][0] + dW

    learning_internal_states[agent_index][0] = pre_trace
    learning_internal_states[agent_index][1] = post_trace
    learning_internal_states[agent_index][2] = dW
    learning_internal_states[agent_index][3] = 1.0 * t_current

    buffer_idx = t_current % len(learning_internal_states_buffer[agent_index])
    learning_internal_states_buffer[agent_index][buffer_idx][0] = pre_trace
    learning_internal_states_buffer[agent_index][buffer_idx][1] = post_trace
    learning_internal_states_buffer[agent_index][buffer_idx][2] = dW
