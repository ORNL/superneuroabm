from cupyx import jit
import os
import sys
module_path = os.path.abspath('/home/xxz/superneuroabm/superneuroabm/step_functions/soma')
if module_path not in sys.path:
	sys.path.append(module_path)
from izh import *
module_path = os.path.abspath('/home/xxz/superneuroabm/superneuroabm/step_functions/soma')
if module_path not in sys.path:
	sys.path.append(module_path)
from lif import *
module_path = os.path.abspath('/home/xxz/superneuroabm/superneuroabm/step_functions/synapse')
if module_path not in sys.path:
	sys.path.append(module_path)
from single_exp import *


@jit.rawkernel(device='cuda')
def stepfunc(
device_global_data_vector,
a0,a1,a2,a3,a4,a5,a6,a7,
sync_workers_every_n_ticks,
num_rank_local_agents,
agent_ids,
):
	thread_id = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
	agent_index = thread_id
	if agent_index < num_rank_local_agents:
		breed_id = a0[agent_index]
		for tick in range(sync_workers_every_n_ticks):

			thread_local_tick = int(device_global_data_vector[0]) + tick

			if breed_id == 0:
				izh_soma_step_func(
					thread_local_tick,
					agent_index,
					device_global_data_vector,
					agent_ids,
					a0,a1,a2,a3,a4,a5,a6,a7,
				)
			if breed_id == 1:
				lif_soma_step_func(
					thread_local_tick,
					agent_index,
					device_global_data_vector,
					agent_ids,
					a0,a1,a2,a3,a4,a5,a6,a7,
				)
			if breed_id == 2:
				synapse_single_exp_step_func(
					thread_local_tick,
					agent_index,
					device_global_data_vector,
					agent_ids,
					a0,a1,a2,a3,a4,a5,a6,a7,
				)