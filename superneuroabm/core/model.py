"""
SuperNeuroABM basic Model class

"""
from typing import Dict, List, Callable, Set, Any
import math
import heapq
from multiprocessing import Pool, Manager
import inspect

from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states
from tqdm import tqdm

from superneuroabm.core.agent import AgentFactory, Breed


class Model:
    THREADSPERBLOCK = 32

    def __init__(self, name: str = "Untitled", use_cuda: bool = True) -> None:
        self._name = name
        self._agent_factory = AgentFactory()
        self._use_cuda = use_cuda
        if self._use_cuda:
            if not cuda.is_available():
                raise EnvironmentError(
                    "CUDA requested but no cuda installation detected."
                )
            else:
                pass
                # device = cuda.get_current_device()
                # device.reset()

    def register_breed(self, breed: Breed) -> None:
        if self._agent_factory.num_agents > 0:
            raise Exception(
                f"Breeds must be registered before agents are created!"
            )
        self._agent_factory.register_breed(breed)

    def create_agent_of_breed(self, breed: Breed, **kwargs) -> int:
        agent_id = self._agent_factory.create_agent(breed, **kwargs)
        return agent_id

    def get_synapse_weight(
        self, presynaptic_neuron_id: int, post_synaptic_neuron_id: int
    ) -> float:
        """
        Return synaptic weight. This function is useful if STDP is
        turned on as weights will be updated during simulation.

        :param presynaptic_neuron_id: int
        :param presynaptic_neuron_id: int
        :returns: float, final weight of synapse
        """
        out_synapses = self.get_agent_property_value(
            presynaptic_neuron_id, "output_synapses"
        )
        weight = math.nan
        for out_synapse in out_synapses:
            if out_synapse[0] == post_synaptic_neuron_id:
                weight = out_synapse[1]
                break
        return weight

    def get_agent_property_value(self, id: int, property_name: str) -> Any:
        return self._agent_factory.get_agent_property_value(
            property_name=property_name, agent_id=id
        )

    def set_agent_property_value(
        self,
        id: int,
        property_name: str,
        value: Any,
        dims: List[int] = None,
    ) -> None:
        self._agent_factory.set_agent_property_value(
            property_name=property_name, agent_id=id, value=value, dims=dims
        )

    def get_agents_with(self, query: Callable) -> Set[List[Any]]:
        return self._agent_factory.get_agents_with(query=query)

    def setup(self) -> None:
        # Create record of agent step functions by breed and priority
        self._breed_idx_2_step_func_by_priority: List[Dict[int, Callable]] = []
        heap_priority_breedidx_func = []
        for breed in self._agent_factory.breeds:
            for priority, func in breed.step_funcs.items():
                heap_priority_breedidx_func.append(
                    (priority, (breed._breedidx, func))
                )
        heapq.heapify(heap_priority_breedidx_func)
        last_priority = None
        while heap_priority_breedidx_func:
            priority, breed_idx_func = heapq.heappop(
                heap_priority_breedidx_func
            )
            if last_priority == priority:
                # same slot in self._breed_idx_2_step_func_by_priority
                self._breed_idx_2_step_func_by_priority[-1].update(
                    {breed_idx_func[0]: breed_idx_func[1]}
                )
            else:
                # new slot
                self._breed_idx_2_step_func_by_priority.append(
                    {breed_idx_func[0]: breed_idx_func[1]}
                )
                last_priority = priority

        # Generate agent data tensor
        self._agent_data_tensors = (
            self._agent_factory.generate_agent_data_tensors(self._use_cuda)
        )

    def simulate(
        self, ticks: int, update_data_ticks: int = 1, num_cpu_proc: int = 4
    ) -> None:
        # Generate global data tensor
        self._global_data_vector = [0]  # index 0 reserved for step
        if not self._use_cuda:
            with Pool(num_cpu_proc) as pool:
                with Manager() as manager:
                    shared_global_data_vector = manager.list(
                        self._global_data_vector
                    )
                    breed_ids = self._agent_data_tensors[0]
                    jobs = []
                    for (
                        breed_idx_2_step_func
                    ) in self._breed_idx_2_step_func_by_priority:
                        jobs_in_priority = [
                            (
                                breed_idx_2_step_func[breed_ids[agent_id]],
                                (
                                    shared_global_data_vector,
                                    *self._agent_data_tensors,
                                    agent_id,
                                ),
                            )
                            for agent_id in range(
                                self._agent_factory._num_agents
                            )
                            if breed_ids[agent_id] in breed_idx_2_step_func
                        ]
                        jobs.append(jobs_in_priority)

                    for tick in tqdm(
                        range(ticks),
                        desc="Simulation Progress",
                        unit="Tick",
                        unit_scale=True,
                        unit_divisor=1000,
                        dynamic_ncols=True,
                    ):
                        shared_global_data_vector[0] = tick
                        for jobs_in_priority in jobs:
                            _ = list(map(smap, jobs_in_priority))
        else:
            blockspergrid = int(
                math.ceil(
                    self._agent_factory.num_agents / Model.THREADSPERBLOCK
                )
            )
            step_funcs_code_obj = generate_gpu_func(
                len(self._agent_data_tensors),
                self._breed_idx_2_step_func_by_priority,
                ticks,
            )
            ################
            shared_global_data_vector = [0]
            device_global_data_vector = cuda.to_device(
                shared_global_data_vector
            )
            exec(step_funcs_code_obj)

        self._agent_factory.update_agents_properties(
            self._agent_data_tensors, self._use_cuda
        )

    @property
    def name(self) -> str:
        return self._name


def smap(func_args):
    return func_args[0](*func_args[1])


def generate_gpu_func(
    n_properties,
    breed_idx_2_step_func_by_priority,
    ticks,
):
    args = [f"a{i}" for i in range(n_properties)]
    sim_loop = ""
    for breed_idx_2_step_func in breed_idx_2_step_func_by_priority:
        for breedidx, breed_step_func in breed_idx_2_step_func.items():
            step_source = inspect.getsource(breed_step_func)
            step_func_name = getattr(
                breed_step_func, "__name__", repr(callable)
            )
            step_source = "\t\t\t\t".join(step_source.splitlines(True))
            sim_loop += f"""
            \n\t\t\tif breed_id == {breedidx}:
            \n\t\t\t\t{step_source} 
            \t{step_func_name}(
            \t\tdevice_global_data_vector,
            \t\t{','.join(args)},
            \t\tagent_id,
            \t)
            g.sync()
            #cuda.syncthreads()
                
            """
    func = f"""
    \ndef stepfunc(
    device_global_data_vector,
    {','.join(args)},
    ):
        thread_id = int(cuda.grid(1))
        if thread_id >= len(a0):
            return
        agent_id = thread_id
        g = cuda.cg.this_grid()
        breed_id = a0[agent_id]
        for tick in range({ticks}):
            device_global_data_vector[0] = tick
            {sim_loop}
    \nstepfunc = cuda.jit(stepfunc)
    \nstepfunc[blockspergrid, Model.THREADSPERBLOCK](
        device_global_data_vector,
        *self._agent_data_tensors,
    )
    \ncuda.synchronize()
    """
    func = func.replace("\t", "    ")
    return compile(func, "<string>", "exec")
