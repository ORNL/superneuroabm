"""
Izhikevich Neuron and weighted synapse step functions for spiking neural networks

"""

import math
import cupy as cp

# def step_func(
#     agent_ids, agent_index, globals, breeds, locations, states, preventative_measures
# ):

#     neighbor_ids = locations[agent_index]  # network location is defined by neighbors
#     rand = random()  # 0.1#step_func_helper_get_random_float(rng_states, id)

#     p_infection = globals[1]

#     agent_preventative_measures = preventative_measures[agent_index]

#     for i in range(len(neighbor_ids)):

#         neighbor_index = -1
#         i = 0
#         while i < len(agent_ids) and agent_ids[i] != neighbor_ids[0]:
#             i += 1
#         if i < len(agent_ids):
#             neighbor_index = i
#             neighbor_state = int(states[neighbor_index])
#             neighbor_preventative_measures = preventative_measures[neighbor_index]
#             abs_safety_of_interaction = 0.0
#             for n in range(len(agent_preventative_measures)):
#                 for m in range(len(neighbor_preventative_measures)):
#                     abs_safety_of_interaction += (
#                         agent_preventative_measures[n]
#                         * neighbor_preventative_measures[m]
#                     )
#             normalized_safety_of_interaction = abs_safety_of_interaction / (
#                 len(agent_preventative_measures) ** 2
#             )
#             if neighbor_state == 2 and rand < p_infection * (
#                 1 - normalized_safety_of_interaction
#             ):
#                 states[agent_index] = 2

# output_synapsess, #shape: num_agents x max(synpses of an agent) x max(delay of syn)+2


def izh_soma_step_func(  # NOTE: update the name to soma_step_func from neuron_step_func
    agent_ids,
    agent_index,
    globals,
    breeds,
    locations,
    neuron_params,  # k, vth, C, a, b,
    internal_state,  # v, u
    synapse_history,  # Synapse delay
    output_spikes_tensor,
):
    synapse_ids = locations[agent_index]  # network location is defined by neighbors

    I_synapse = 0
    for i in range(len(synapse_ids)):

        synapse_index = -1  # synapse index local to the current mpi rank
        i = 0
        while i < len(agent_ids) and agent_ids[i] != synapse_ids[0]:
            i += 1
        if i < len(agent_ids):
            synapse_index = i
            I_synapse += internal_state[synapse_index][0]

    # Get the current time step value:
    t_current = int(globals[0])

    dt = globals[1]  # time step size

    # update t_elapse from refractory period time
    # t_elapses[my_idx] = 0 if t_elapses[my_idx] == 0 else t_elapses[my_idx] - 1
    # If still in refractory period, return
    # if t_elapses[my_idx] > 0:
    #    return
    # Otherwise update Vm, etc

    """
    # Update Vm with the input spikes if any:
    # for spike in input_spikess[my_idx]:
    for i in range(len(input_spikes_tensor[agent_index])):
        if input_spikes_tensor[agent_index][i][0] == t_current:
            I_external += input_spikes_tensor[agent_index][i][1]
    """

    # internal_state[my_idx] = (
    #    internal_state[my_idx] - leaks[my_idx]
    #    if internal_state[my_idx] - leaks[my_idx] > reset_states[my_idx]
    #    else reset_states[my_idx]
    # )
    ###################TODO:
    # NOTE: neuron_params would need to as long as the max number of params in any spiking neuron model
    k = neuron_params[agent_index][0]
    vthr = neuron_params[agent_index][1]
    C = neuron_params[agent_index][2]
    a = neuron_params[agent_index][3]
    b = neuron_params[agent_index][4]
    vpeak = neuron_params[agent_index][5]
    vrest = neuron_params[agent_index][6]
    d = neuron_params[agent_index][7]
    vreset = neuron_params[agent_index][8]

    # From https://www.izhikevich.org/publications/spikes.htm
    # v' = 0.04v^2 + 5v + 140 -u + I
    # u' = a(bv - u)
    # if v=30mV: v = c, u = u + d, spike

    # dv = (k*(internal_state[my_idx]-vrest)*(internal_state[my_idx]-vthr)-u[my_idx]+I) / C
    # internal_state: [0] - v, [1] - u
    # NOTE: size of internal_state would need to be set as the maximum possible state varaibles of any spiking neuron

    v = internal_state[agent_index][0]
    u = internal_state[agent_index][1]

    I_bias = globals[2]  # bias current

    dv = (k * (v - vrest) * (v - vthr) - u + I_synapse + I_bias) / C
    v = v + dt * dv

    u += dt * (a * (b * (v - vrest) - u))
    s = 1 * (v >= vpeak)  # output spike
    u = u + d * s  # If spiked, update recovery variable
    v = v * (1 - s) + vreset * s  # If spiked, reset membrane potential
    # self.v_ = self.v

    internal_state[agent_index][0] = v
    internal_state[agent_index][1] = u

    output_spikes_tensor[agent_index][t_current] = s


def stdp_aux():
    pass


def synapse_single_exp_step_func(
    agent_ids,
    agent_index,
    globals,
    breeds,
    locations,
    synapse_params,  # scale, time constant (tau_rise and tau_fall)
    internal_state,  #
    synapse_history,  # delay
    output_spikes_tensor,
):
    t_current = int(globals[0])

    dt = globals[1]  # time step size

    weight = synapse_params[agent_index][0]
    synaptic_delay = synapse_params[agent_index][1]
    scale = synapse_params[agent_index][2]
    tau_fall = synapse_params[agent_index][3]
    tau_rise = synapse_params[agent_index][4]

    pre_soma_id = locations[agent_index][0]

    if not cp.isnan(pre_soma_id):
        i = 0
        while i < len(agent_ids) and agent_ids[i] != pre_soma_id:
            i += 1
        spike = output_spikes_tensor[i][t_current]
    else:
        spike = 0

    # r = self.r*(1-self.dt/self.td) + spike/self.td
    # self.r = r

    I_synapse = internal_state[agent_index][0]

    I_synapse = I_synapse * (1 - dt / tau_fall) + spike * scale * weight

    internal_state[agent_index][0] = I_synapse

    # # Update outgoing synapses if any
    # for synapse_idx in range(len(output_synapsess[my_idx])):
    #     out_synapse_info = output_synapsess[my_idx][synapse_idx]
    #     out_neuron_id = out_synapse_info[0]
    #     if math.isnan(out_neuron_id):
    #         break
    #     out_neuron_id = int(out_neuron_id)
    #     # If out_neuron still in refractory period, return
    #     if t_elapses[out_neuron_id] > 0:
    #         continue
    #     weight = out_synapse_info[1]
    #     synapse_register = out_synapse_info[2:]
    #     # Check if delayed Vm was over threshold, if so spike
    #     syn_delay_reg_len = 0
    #     for val in synapse_register:
    #         if math.isnan(val):
    #             break
    #         syn_delay_reg_len += 1
    #     if not syn_delay_reg_len:
    #         continue
    #     syn_delay_reg_head = t_current % (syn_delay_reg_len)
    #     syn_delay_reg_tail = (
    #         0
    #         if syn_delay_reg_len == 1
    #         or syn_delay_reg_head + 1 >= syn_delay_reg_len
    #         else syn_delay_reg_head + 1
    #     )
    #     Vm = synapse_register[syn_delay_reg_tail]
    #     if not math.isnan(Vm) and Vm != 0:
    #         internal_state[out_neuron_id] += weight


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
    output_synapses_learning_paramss,
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
        out_neuron_id = int(out_neuron_id)
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
            if syn_delay_reg_len == 1 or syn_delay_reg_head + 1 >= syn_delay_reg_len
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
        output_synapse_learning_params = output_synapses_learning_paramss[my_idx][
            synapse_idx
        ]
        stdp_timesteps = output_synapse_learning_params[0]
        A_pos = output_synapse_learning_params[1]  # 0.6
        A_neg = output_synapse_learning_params[2]  # 0.3
        tau_pos = output_synapse_learning_params[3]  # 8
        tau_neg = output_synapse_learning_params[4]  # 5

        presynaptic_spikes = output_spikess[my_idx]
        postsynaptic_spikes = output_spikess[int(out_neuron_id)]
        delta_w = 0
        for delta_t in range(int(stdp_timesteps)):
            pre_to_post_correlation = (
                presynaptic_spikes[t_current - 1 - delta_t]
                * postsynaptic_spikes[t_current]
            )
            delta_w += A_pos * math.exp(delta_t / tau_pos) * pre_to_post_correlation
            post_to_pre_correlation = (
                postsynaptic_spikes[t_current]
                * presynaptic_spikes[t_current - 1 - delta_t]
            )
            delta_w -= A_neg * math.exp(delta_t / tau_neg) * post_to_pre_correlation
        w_old = weight
        sigma = 0.8  # weight change rate
        w_max = 1000
        if delta_w > 0:
            w_new = w_old + sigma * delta_w * (w_max - w_old)
        else:
            w_new = w_old + sigma * delta_w * (w_old - w_max)
        # set new weight of synapse. Weight is at index 1
        output_synapsess[my_idx][synapse_idx][1] = w_new
