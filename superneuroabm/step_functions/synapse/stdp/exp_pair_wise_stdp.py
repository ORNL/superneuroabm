"""
Exponential STDP (Spike-Timing Dependent Plasticity) step function for spiking neural networks

"""

from cupyx import jit

from superneuroabm.step_functions.synapse.util import get_pre_soma_spike


@jit.rawkernel(device="cuda")
def exp_stdp_all_to_all(
    tick,
    agent_index,
    globals,
    agent_ids,
    breeds,
    locations,
    synapse_params,  # scale, time constant (tau_rise and tau_fall)
    internal_state,  #
    synapse_history,  # delay
    input_spikes_tensor,  # input spikes
    output_spikes_tensor,
    internal_states_buffer,
):
    t_current = int(tick)

    dt = globals[1]  # time step size

    weight = synapse_params[agent_index][0]
    synaptic_delay = synapse_params[agent_index][1]
    scale = synapse_params[agent_index][2]
    tau_fall = synapse_params[agent_index][3]
    tau_rise = synapse_params[agent_index][4]
    tau_pre_stdp = synapse_params[agent_index][5]
    tau_post_stdp = synapse_params[agent_index][6]
    a_exp_pre = synapse_params[agent_index][7]
    a_exp_post = synapse_params[agent_index][8]
    stdp_history_length = synapse_params[agent_index][9]
    #Wmax, Wmin
    
    pre_soma_id = locations[agent_index][0]
    pre_soma_spike = get_pre_soma_spike(
        agent_index,
        globals,
        agent_ids,
        pre_soma_id,
        t_current,
        input_spikes_tensor,
        output_spikes_tensor,
    )

    post_soma_id = locations[agent_index][1]
    post_soma_spike = get_pre_soma_spike(
        agent_index,
        globals,
        agent_ids,
        post_soma_id,
        t_current,
        input_spikes_tensor,
        output_spikes_tensor,
    )

    pre_trace = pre_trace * (1 - dt / tau_pre_stdp) + pre_soma_spike *a_exp_pre
    post_trace = post_trace * (1 - dt / tau_post_stdp) + post_soma_spike * a_exp_post

    spike_pre_[t_current] = pre_soma_spike #spike_pre_ is an array of size (stdp_history_length, number of input neurons), pre_soma_spike is (number of input neurons,)
    spike_post_[:, t_current] = post_soma_spike#spike_post_ is an array of size (number of output neurons,stdp_history_length), post_soma_spike is (number of output neurons,)
    trace_pre_[t_current] = pre_trace #Corresponding traces an array of size (stdp_history_length,number of input neurons), pre_trace is (number of input neurons,)
    trace_post_[:, t_current] = post_trace #Corresponding traces is an array of size (number of output neurons,stdp_history_length)
    
    if t_current == stdp_history_length:
        dW = cp.dot(spike_post_, trace_pre_)#(1,stdp_history_length) dot (stdp_history_length,1) we might need additional learning rate and multiplicative STDP*(wmax - W)*
        dW -=cp.dot(trace_post_, spike_pre_)#(1,stdp_history_length) dot (stdp_history_length,1),  add learning rat*W for multiplicative STDP
        clipped_dW = cp.clip(dW / stdp_history_length, dw_max, dw_min)  # Clip the weight change if needed
        weight = cp.clip(weight+clipped_dW,wmin, wmax)  # Update the weight
        #reset the traces and spikes buffers
        spike_pre_ = cp.zeros((stdp_history_length, number_of_input_neurons), dtype=cp.float32)
        spike_post_ = cp.zeros((number_of_output_neurons, stdp_history_length), dtype=cp.float32)
        trace_pre_ = cp.zeros((stdp_history_length, number_of_input_neurons), dtype=cp.float32)
        trace_post_ = cp.zeros((number_of_output_neurons, stdp_history_length), dtype=cp.float32)
        


    internal_state[agent_index][2] = pre_trace
    internal_state[agent_index][3] = post_trace
    internal_states_buffer[agent_index][t_current][2] = pre_trace
    internal_states_buffer[agent_index][t_current][3] = post_trace
    #if t_current==100:
        #print("Calculation at t_current=100:")

