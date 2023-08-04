"""
Neuron and synapse step functions for spiking neural networks

"""
import math


def neuron_step_func(
    global_data_vector,
    breeds,
    thresholds,
    reset_states,
    leaks,
    refractory_periods,
    output_synapsess,
    t_elapses,
    internal_state,
    neuron_delay_regs,
    input_spikess,
    output_spikess,
    my_idx,
):
    t_current = int(global_data_vector[0])
    # update t_elapse from refractory period time
    t_elapses[my_idx] = 0 if t_elapses[my_idx] == 0 else t_elapses[my_idx] - 1
    # If still in refractory period, return
    if t_elapses[my_idx] > 0:
        return
    # Otherwise update Vm, etc
    # Update Vm with the input spikes if any:
    for spike in input_spikess[my_idx]:
        if spike[0] == t_current:
            internal_state[my_idx] += spike[1]
    internal_state[my_idx] -= leaks[my_idx]

    # Apply axonal delay to the output of the spike:
    # Shift the register each time step
    head = t_current % len(neuron_delay_regs[my_idx])
    neuron_delay_regs[my_idx, head] = (
        1 if internal_state[my_idx] > thresholds[my_idx] else 0
    )

    # Update outgoing synapses
    oldest_idx = (
        0
        if len(neuron_delay_regs[my_idx]) == 1
        or head + 1 >= len(neuron_delay_regs[my_idx])
        else head + 1
    )  # index of oldest entry on delay_reg
    for synapse_idx, synapse_info in enumerate(output_synapsess[my_idx]):
        out_neuron_id = synapse_info[0]
        weight = synapse_info[1]
        synapse_register = synapse_info[2:]
        syn_delay_reg_len = 0
        for val in synapse_register:
            if math.isnan(val):
                break
            syn_delay_reg_len += 1
        if not syn_delay_reg_len:
            continue
        syn_delay_reg_head = t_current % (syn_delay_reg_len)
        # Check if delayed Vm was over threshold, if so spike
        output_synapsess[my_idx][synapse_idx][2 + syn_delay_reg_head] = (
            1 if neuron_delay_regs[my_idx][oldest_idx] else 0
        )

    # If spike reset and refactor
    if neuron_delay_regs[my_idx][oldest_idx]:
        # reset Vm
        internal_state[my_idx] = reset_states[my_idx]
        # start refractory period
        t_elapses[my_idx] = refractory_periods[my_idx]
        output_spikess[my_idx][t_current] = 1


def synapse_step_func(
    global_data_vector,
    breeds,
    thresholds,
    reset_states,
    leaks,
    refractory_periods,
    output_synapsess,
    t_elapses,
    internal_state,
    neuron_delay_regs,
    input_spikess,
    output_spikess,
    my_idx,
):
    t_current = int(global_data_vector[0])
    # Update outgoing synapses if any
    for synapse_idx in range(len(output_synapsess[my_idx])):
        out_synapse_info = output_synapsess[my_idx][synapse_idx]
        out_neuron_id = out_synapse_info[0]
        if math.isnan(out_neuron_id):
            break
        weight = out_synapse_info[1]
        synapse_register = out_synapse_info[2:]
        # Check if delayed Vm was over threshold, if so spike
        syn_delay_reg_len = 0
        for val in synapse_register:
            if math.isnan(val):
                break
            syn_delay_reg_len += 1
        if not syn_delay_reg_len:
            continue
        syn_delay_reg_head = t_current % (syn_delay_reg_len)
        syn_delay_reg_tail = (
            0
            if syn_delay_reg_len == 1
            or syn_delay_reg_head + 1 >= syn_delay_reg_len
            else syn_delay_reg_head + 1
        )
        Vm = synapse_register[syn_delay_reg_tail]
        if not math.isnan(Vm) and Vm != 0:
            internal_state[int(out_neuron_id)] += weight


def synapse_with_stdp_step_func(
    global_data_vector,
    breeds,
    thresholds,
    reset_states,
    leaks,
    refractory_periods,
    output_synapsess,
    t_elapses,
    internal_state,
    neuron_delay_regs,
    input_spikess,
    output_spikess,
    my_idx,
):
    t_current = int(global_data_vector[0])
    # Update outgoing synapses if any
    out_synapses_info = output_synapsess[my_idx]
    for synapse_idx in range(len(out_synapses_info)):
        out_synapse_info = out_synapses_info[synapse_idx]
        out_neuron_id = out_synapse_info[0]
        if math.isnan(out_neuron_id):
            break
        weight = out_synapse_info[1]
        synapse_register = out_synapse_info[2:]
        # Check if delayed Vm was over threshold, if so spike
        # We'll perform a rotation of the synapse_register to
        # emulate FIFO behavior of the spikes.
        # For this find the delay register head and tail
        syn_delay_reg_len = 0
        for val in synapse_register:
            if math.isnan(val):
                break
            syn_delay_reg_len += 1
        if not syn_delay_reg_len:
            continue
        # The delay register head pointer moves forward along
        # the delay register with ticks and wraps around once
        # the end is reached.
        syn_delay_reg_head = t_current % (syn_delay_reg_len)
        syn_delay_reg_tail = (
            0
            if syn_delay_reg_len == 1
            or syn_delay_reg_head + 1 >= syn_delay_reg_len
            else syn_delay_reg_head + 1
        )
        # The earliest spike still in the register is at
        # syn_delay_reg_tail and affects the out_neuron during
        # this tick.
        Vm = synapse_register[syn_delay_reg_tail]
        if not math.isnan(Vm) and Vm != 0:
            internal_state[int(out_neuron_id)] += weight

        # Perform STDP weight change
        if t_current < 2:
            return
        stdp_timesteps = 3
        A_pos = 0.6
        A_neg = 0.3
        tau_pos = 8
        tau_neg = 5

        presynaptic_spikes = output_spikess[my_idx]
        postsynaptic_spikes = output_spikess[int(out_neuron_id)]
        delta_w = 0
        for delta_t in range(stdp_timesteps):
            pre_to_post_correlation = (
                presynaptic_spikes[t_current - 1 - delta_t]
                * postsynaptic_spikes[t_current]
            )
            delta_w += (
                A_pos * math.exp(delta_t / tau_pos) * pre_to_post_correlation
            )
            post_to_pre_correlation = (
                postsynaptic_spikes[t_current]
                * presynaptic_spikes[t_current - 1 - delta_t]
            )
            delta_w -= (
                A_neg * math.exp(delta_t / tau_neg) * post_to_pre_correlation
            )
        w_old = weight
        sigma = 0.8  # weight change rate
        w_max = 1000
        if delta_w > 0:
            w_new = w_old + sigma * delta_w * (w_max - w_old)
        else:
            w_new = w_old + sigma * delta_w * (w_old - w_max)
        # set new weight of synapse. Weight is at index 1
        output_synapsess[my_idx][synapse_idx][1] = w_new
