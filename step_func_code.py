# Auto-generated GPU kernel with cross-breed synchronization
# Contains all necessary imports and modified step functions

import os
import sys
module_path = os.path.abspath('/lustre/orion/lrn088/proj-shared/objective3/wfishell/superneuroabm/superneuroabm/step_functions/soma')
if module_path not in sys.path:
	sys.path.append(module_path)
from izh import *
module_path = os.path.abspath('/lustre/orion/lrn088/proj-shared/objective3/wfishell/superneuroabm/superneuroabm/step_functions/soma')
if module_path not in sys.path:
	sys.path.append(module_path)
from lif import *
module_path = os.path.abspath('/lustre/orion/lrn088/proj-shared/objective3/wfishell/superneuroabm/superneuroabm/step_functions/synapse')
if module_path not in sys.path:
	sys.path.append(module_path)
from single_exp import *
module_path = os.path.abspath('/lustre/orion/lrn088/proj-shared/objective3/wfishell/superneuroabm/superneuroabm/step_functions/synapse/stdp')
if module_path not in sys.path:
	sys.path.append(module_path)
from learning_rule_selector import *

# Modified step functions with double buffering
@jit.rawkernel(device='cuda')
def izh_soma_step_func_double_buffer(tick, agent_index, globals, agent_ids, breeds, locations, neuron_params, learning_params, internal_state, internal_learning_state, synapse_history, input_spikes_tensor, output_spikes_tensor, internal_states_buffer, internal_learning_states_buffer, write_internal_state, write_output_spikes_tensor, write_internal_states_buffer):
    synapse_indices = locations[agent_index]
    I_synapse = 0.0
    for i in range(len(synapse_indices)):
        synapse_index = int(synapse_indices[i])
        if synapse_index >= 0 and (not cp.isnan(synapse_indices[i])):
            I_synapse += internal_state[synapse_index][0]
    t_current = int(tick)
    dt = globals[0]
    I_bias = globals[1]
    k = neuron_params[agent_index][0]
    vthr = neuron_params[agent_index][1]
    C = neuron_params[agent_index][2]
    a = neuron_params[agent_index][3]
    b = neuron_params[agent_index][4]
    vpeak = neuron_params[agent_index][5]
    vrest = neuron_params[agent_index][6]
    d = neuron_params[agent_index][7]
    vreset = neuron_params[agent_index][8]
    I_in = neuron_params[agent_index][9]
    v = internal_state[agent_index][0]
    u = internal_state[agent_index][1]
    dv = (k * (v - vrest) * (v - vthr) - u + I_synapse + I_bias + I_in) / C
    v = v + dt * dv * 1000.0
    u += dt * 1000.0 * (a * (b * (v - vrest) - u))
    s = 1 * (v >= vpeak)
    u = u + d * s
    v = v * (1 - s) + vreset * s
    write_internal_state[agent_index][0] = v
    write_internal_state[agent_index][1] = u
    write_output_spikes_tensor[agent_index][t_current] = s
    buffer_idx = t_current % len(internal_states_buffer[agent_index])
    write_internal_states_buffer[agent_index][buffer_idx][0] = v
    write_internal_states_buffer[agent_index][buffer_idx][1] = u

@jit.rawkernel(device='cuda')
def lif_soma_step_func_double_buffer(tick, agent_index, globals, agent_ids, breeds, locations, neuron_params, learning_params, internal_state, internal_learning_state, synapse_history, input_spikes_tensor, output_spikes_tensor, internal_states_buffer, internal_learning_states_buffer, write_internal_state, write_output_spikes_tensor, write_internal_states_buffer):
    synapse_indices = locations[agent_index]
    I_synapse = 0.0
    for i in range(len(synapse_indices)):
        synapse_index = int(synapse_indices[i])
        if synapse_index >= 0 and (not cp.isnan(synapse_indices[i])):
            I_synapse += internal_state[synapse_index][0]
    t_current = int(tick)
    dt = globals[0]
    I_bias = globals[1]
    C = neuron_params[agent_index][0]
    R = neuron_params[agent_index][1]
    vthr = neuron_params[agent_index][2]
    tref = neuron_params[agent_index][3]
    vrest = neuron_params[agent_index][4]
    vreset = neuron_params[agent_index][5]
    tref_allows_integration = neuron_params[agent_index][6]
    I_in = neuron_params[agent_index][7]
    scaling_factor = neuron_params[agent_index][8]
    v = internal_state[agent_index][0]
    tcount = internal_state[agent_index][1]
    tlast = internal_state[agent_index][2]
    dv = (vrest - v) / (R * C) + (I_synapse * scaling_factor + I_bias + I_in) / C
    v += dv * dt if dt * tcount > tlast + tref or tref_allows_integration else 0.0
    s = 1.0 * (v >= vthr and (dt * tcount > tlast + tref if tlast > 0 else True))
    tlast = tlast * (1 - s) + dt * tcount * s
    v = v * (1 - s) + vreset * s
    write_internal_state[agent_index][0] = v
    write_internal_state[agent_index][1] = internal_state[agent_index][1] + 1
    write_internal_state[agent_index][2] = tlast
    write_output_spikes_tensor[agent_index][t_current] = s
    buffer_idx = t_current % len(internal_states_buffer[agent_index])
    write_internal_states_buffer[agent_index][buffer_idx][0] = v
    write_internal_states_buffer[agent_index][buffer_idx][1] = internal_state[agent_index][1] + 1
    write_internal_states_buffer[agent_index][buffer_idx][2] = tlast

@jit.rawkernel(device='cuda')
def synapse_single_exp_step_func_double_buffer(tick, agent_index, globals, agent_ids, breeds, locations, synapse_params, learning_params, internal_state, internal_learning_state, synapse_history, input_spikes_tensor, output_spikes_tensor, internal_states_buffer, internal_learning_states_buffer, write_internal_state, write_output_spikes_tensor, write_internal_states_buffer):
    t_current = int(tick)
    dt = globals[0]
    weight = synapse_params[agent_index][0]
    synaptic_delay = synapse_params[agent_index][1]
    scale = synapse_params[agent_index][2]
    tau_fall = synapse_params[agent_index][3]
    tau_rise = synapse_params[agent_index][4]
    pre_soma_index, post_soma_index = (locations[agent_index][0], locations[agent_index][1])
    spike = get_soma_spike(tick, agent_index, globals, agent_ids, pre_soma_index, t_current, input_spikes_tensor, output_spikes_tensor)
    I_synapse = internal_state[agent_index][0]
    I_synapse = I_synapse * (1 - dt / tau_fall) + spike * scale * weight
    write_internal_state[agent_index][0] = I_synapse
    buffer_idx = t_current % len(internal_states_buffer[agent_index])
    write_internal_states_buffer[agent_index][buffer_idx][0] = I_synapse
    write_internal_states_buffer[agent_index][buffer_idx][1] = spike
    write_internal_states_buffer[agent_index][buffer_idx][2] = t_current

@jit.rawkernel(device='cuda')
def learning_rule_selector_double_buffer(tick, agent_index, globals, agent_ids, breeds, locations, synapse_params, learning_params, internal_state, internal_learning_state, synapse_history, input_spikes_tensor, output_spikes_tensor, internal_states_buffer, internal_learning_states_buffer, write_internal_state, write_output_spikes_tensor, write_internal_states_buffer):
    stdpType = learning_params[agent_index][0]
    if stdpType == -1:
        pass
    elif stdpType == 0:
        exp_pair_wise_stdp(tick, agent_index, globals, agent_ids, breeds, locations, synapse_params, learning_params, internal_state, internal_learning_state, synapse_history, input_spikes_tensor, output_spikes_tensor, internal_states_buffer, internal_learning_states_buffer)

@jit.rawkernel(device='cuda')
def stepfunc(
global_tick,
device_global_data_vector,
a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,write_a4,write_a8,write_a9,
sync_workers_every_n_ticks,
num_rank_local_agents,
agent_ids,
current_priority_index,
):
	thread_id = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
	agent_index = thread_id
	if agent_index < num_rank_local_agents:
		breed_id = a0[agent_index]
		for tick in range(sync_workers_every_n_ticks):
			thread_local_tick = int(global_tick) + tick

			if current_priority_index == 0:
				if breed_id == 0:
					izh_soma_step_func_double_buffer(
						thread_local_tick,
						agent_index,
						device_global_data_vector,
						agent_ids,
						a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,write_a4,write_a8,write_a9,
					)
				if breed_id == 1:
					lif_soma_step_func_double_buffer(
						thread_local_tick,
						agent_index,
						device_global_data_vector,
						agent_ids,
						a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,write_a4,write_a8,write_a9,
					)
			if current_priority_index == 1:
				if breed_id == 2:
					synapse_single_exp_step_func_double_buffer(
						thread_local_tick,
						agent_index,
						device_global_data_vector,
						agent_ids,
						a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,write_a4,write_a8,write_a9,
					)
			if current_priority_index == 2:
				if breed_id == 2:
					learning_rule_selector_double_buffer(
						thread_local_tick,
						agent_index,
						device_global_data_vector,
						agent_ids,
						a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,write_a4,write_a8,write_a9,
					)