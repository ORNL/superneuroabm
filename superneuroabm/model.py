"""
Model class for building an SNN
"""

from typing import Dict, Callable, List
from pathlib import Path

import numpy as np
from sagesim.space import NetworkSpace
from sagesim.model import Model
from sagesim.breed import Breed
from pathlib import Path

from superneuroabm.step_functions.soma.izh import izh_soma_step_func
from superneuroabm.step_functions.soma.lif import lif_soma_step_func
from superneuroabm.step_functions.synapse.single_exp import synapse_single_exp_step_func
from superneuroabm.step_functions.synapse.stdp.exp_pair_wise_stdp import (
    exp_stdp_all_to_all,
)

CURRENT_DIR_ABSPATH = Path(__file__).resolve().parent


class NeuromorphicModel(Model):
    def __init__(
        self,
        soma_breed_info: Dict[str, List[Callable]] = {
            "IZH_Soma": [
                (
                    izh_soma_step_func,
                    CURRENT_DIR_ABSPATH / "step_functions" / "soma" / "izh.py",
                )
            ],
            "LIF_Soma": [
                (
                    lif_soma_step_func,
                    CURRENT_DIR_ABSPATH / "step_functions" / "soma" / "lif.py",
                )
            ],
        },
        synapse_breed_info: Dict[str, List[Callable]] = {
            "Single_Exp_Synapse": [
                (
                    synapse_single_exp_step_func,
                    CURRENT_DIR_ABSPATH
                    / "step_functions"
                    / "synapse"
                    / "single_exp.py",
                )
            ],
        },
    ) -> None:
        """
        Creates an SNN Model and provides methods to create, simulate,
        and monitor soma and synapses.

        :param use_gpu: True if the system supports CUDA GPU
            acceleration.
        :param soma_breed_info: Dict of breed name to List of
            Callable step functions. If specifed, will override
            the default soma breed and soma and synapse step
            functions, allowing for multi-breed simulations.
            Step functions will be executed on the respective
            breed every simulation step in the order specifed in the
            list.
        """
        super().__init__(space=NetworkSpace())

        soma_properties = {
            "parameters": [0.0, 0.0, 0.0, 0.0, 0.0],  # k, vth, C, a, b,
            "internal_state": [0.0, 0.0, 0.0, 0.0],  # v, u
            "synapse_delay_reg": [],  # Synapse delay
            "input_spikes_tensor": [],  # input spikes tensor
            "output_spikes_tensor": [],
            "internal_states_buffer": [],
        }
        synapse_properties = {
            "parameters": [
                0.0 for _ in range(10)
            ],  # weight, delay, scale, Tau_fall, Tau_rise, tau_pre_stdp, tau_post_stdp, a_exp_pre, a_exp_post, stdp_history_length
            "internal_state": [
                0.0 for _ in range(4)
            ],  # Isyn, Isyn_supp, pre_trace, post_trace
            "synapse_delay_reg": [],  # Synapse delay
            "input_spikes_tensor": [],  # input spikes tensor
            "output_spikes_tensor": [],
            "internal_states_buffer": [],
        }
        self._synapse_ids = []
        self._soma_ids = []
        self._soma_reset_states = {}

        self._soma_breeds: Dict[str, Breed] = {}
        for breed_name, step_funcs in soma_breed_info.items():
            soma_breed = Breed(breed_name)  # Strt here Ashish
            for prop_name, default_val in soma_properties.items():
                soma_breed.register_property(prop_name, default_val)
            for step_func_order, (step_func, module_fpath) in enumerate(step_funcs):
                module_fpath = (
                    CURRENT_DIR_ABSPATH / "izh_soma.py"
                    if module_fpath is None
                    else module_fpath
                )
                soma_breed.register_step_func(
                    step_func=step_func,
                    module_fpath=module_fpath,
                    priority=step_func_order,
                )
            self.register_breed(soma_breed)
            self._soma_breeds[breed_name] = soma_breed

        self._synapse_breeds: Dict[str, Breed] = {}
        for breed_name, step_funcs in synapse_breed_info.items():
            synapse_breed = Breed(breed_name)  # Strt here Ashish
            for prop_name, default_val in synapse_properties.items():
                synapse_breed.register_property(prop_name, default_val)
            for step_func_order, (step_func, module_fpath) in enumerate(step_funcs):
                module_fpath = (
                    CURRENT_DIR_ABSPATH / "izh_soma.py"
                    if module_fpath is None
                    else module_fpath
                )
                synapse_breed.register_step_func(
                    step_func=step_func,
                    module_fpath=module_fpath,
                    priority=100 + step_func_order,
                )
            self.register_breed(synapse_breed)
            self._synapse_breeds[breed_name] = synapse_breed

        self._synapse_index_map = {}

    def set_global_property_value(name: str, value: float) -> None:
        if name in super().globals:
            super().set_global_property_value(name, value)
        else:
            super().register_global_property(name, value)

    def get_global_property_value(name: str) -> float:
        return super().get_global_property_value(name)

    def setup(
        self,
        use_gpu: bool = False,
        retain_parameters=False,
    ) -> None:
        """
        Resets the simulation and initializes agents.

        :param retain_parameters: False by default. If True, parameters are
            reset to their default values upon setup.
        """
        synapse_ids = self._synapse_ids
        soma_ids = self._soma_ids
        for synapse_id in synapse_ids:
            # Clear input spikes
            new_input_spikes = []
            super().set_agent_property_value(
                id=synapse_id,
                property_name="input_spikes_tensor",
                value=new_input_spikes,
            )
            # Clear synapse delay registers
            synapse_delay = len(
                super().get_agent_property_value(
                    id=synapse_id, property_name="synapse_delay_reg"
                )
            )
            synapse_delay_reg = [0 for _ in range(synapse_delay)]
            self.set_agent_property_value(
                synapse_id,
                "synapse_delay_reg",
                synapse_delay_reg,
            )
            # Reset parameters to defaults if retain_parameters is True
            if retain_parameters:
                # Reset all synapse parameters to their default values
                default_synapse_parameters = [
                    0.0 for _ in range(10)
                ]  # weight, delay, scale, Tau_fall, Tau_rise, tau_pre_stdp, tau_post_stdp, a_exp_pre, a_exp_post, stdp_history_length
                super().set_agent_property_value(
                    id=synapse_id,
                    property_name="parameters",
                    value=default_synapse_parameters,
                )
            # Clear internal states
            synapse_internal_state = super().get_agent_property_value(
                id=synapse_id,
                property_name="internal_state",
            )
            synapse_internal_state = [0.0 for _ in synapse_internal_state]
            super().set_agent_property_value(
                id=synapse_id,
                property_name="internal_state",
                value=synapse_internal_state,
            )
        for soma_id in soma_ids:
            # Clear internal states
            super().set_agent_property_value(
                id=soma_id,
                property_name="internal_state",
                value=self._soma_reset_states[soma_id],
            )
        super().setup(use_gpu=use_gpu)

    def simulate(
        self, ticks: int, update_data_ticks: int = 1  # , num_cpu_proc: int = 4
    ) -> None:
        """
        Override of superneuroabm.core.model mainly to register an
        AgentDataCollector to monitor marked output somas.

        """
        for soma_id in self._soma_ids:
            # Clear output buffer
            output_buffer = [0 for _ in range(ticks)]
            super().set_agent_property_value(
                id=soma_id,
                property_name="output_spikes_tensor",
                value=output_buffer,
            )
            initial_internal_state = super().get_agent_property_value(
                id=soma_id, property_name="internal_state"
            )
            internal_states_buffer = [initial_internal_state[::] for _ in range(ticks)]
            super().set_agent_property_value(
                id=soma_id,
                property_name="internal_states_buffer",
                value=internal_states_buffer,
            )
        for synapse_id in self._synapse_ids:
            initial_internal_state = super().get_agent_property_value(
                id=synapse_id, property_name="internal_state"
            )
            internal_states_buffer = [initial_internal_state[::] for _ in range(ticks)]
            super().set_agent_property_value(
                id=synapse_id,
                property_name="internal_states_buffer",
                value=internal_states_buffer,
            )
        super().simulate(ticks, update_data_ticks)  # , num_cpu_proc)

    def create_soma(
        self,
        breed: str,
        parameters: List[float],
        default_internal_state: List[float],
    ) -> int:
        """
        Creates and soma agent.

        :return: SAGESim agent id of soma

        """
        soma_id = super().create_agent_of_breed(
            breed=self._soma_breeds[breed],  # TODO fix
            parameters=parameters,
            internal_state=default_internal_state,
        )
        self._soma_ids.append(soma_id)
        self._soma_reset_states[soma_id] = default_internal_state
        return soma_id

    def create_synapse(
        self,
        breed: str,
        pre_soma_id: int,  # TODO edit
        post_soma_id: int,
        parameters: List[float],
        default_internal_state: List[float],
    ) -> int:
        """
        Creates and adds Synapse agent.

        :param pre_soma_id: int presynaptic soma id
        :param post_soma_id: int postsynaptic soma id
        :param weight: weight of synapse
        :param synaptic delay: number of timesteps to delay synapse by
        :param synapse_learning_params: Optional. Any parameters used in a learning
            enabled step function. Must be specified in order of use
            in step function.
        """
        synaptic_delay = int(parameters[1])
        delay_reg = [0 for _ in range(synaptic_delay)]
        synapse_id = self.create_agent_of_breed(
            breed=self._synapse_breeds[breed],
            parameters=parameters,
            internal_state=default_internal_state,
            synapse_delay_reg=delay_reg,
        )
        self._synapse_ids.append(synapse_id)

        network_space: NetworkSpace = self.get_space()
        if not np.isnan(pre_soma_id):
            network_space.connect_agents(synapse_id, pre_soma_id, directed=True)
        if not np.isnan(post_soma_id):
            network_space.connect_agents(post_soma_id, synapse_id, directed=True)
        return synapse_id

    def update_synapse(
        self,
        pre_soma_id: int,
        post_soma_id: int,
        parameters: List[float] = None,
    ):
        raise NotImplementedError("update_synapse is not implemented.")
        if pre_soma_id in self._synapse_index_map:
            output_synapse_index_map = self._synapse_index_map[pre_soma_id]
            if post_soma_id in output_synapse_index_map:
                synapse_idx = output_synapse_index_map[post_soma_id]
            else:
                raise ValueError(
                    f"Synapse {pre_soma_id} -> {post_soma_id} does not exist"
                )
        else:
            raise ValueError(f"Synapse {pre_soma_id} -> {post_soma_id} does not exist")

        # Update new synapse params
        if weight != None:
            output_synapses = self.get_agent_property_value(
                pre_soma_id, "output_synapses"
            )
            output_synapses[synapse_idx][1] = weight
            self.set_agent_property_value(
                pre_soma_id,
                "output_synapses",
                output_synapses,
            )

        # Update or enter learning params
        if synapse_learning_params:
            synapses_learning_params = self.get_agent_property_value(
                pre_soma_id, "output_synapses_learning_params"
            )
            synapses_learning_params[synapse_idx] = synapse_learning_params
            self.set_agent_property_value(
                id=pre_soma_id,
                property_name="output_synapses_learning_params",
                value=synapses_learning_params,
            )

    def add_spike(self, synapse_id: int, tick: int, value: float) -> None:
        """
        Schedules an external input spike to this soma.

        :param tick: tick at which spike should be triggered
        :param value: spike value
        """
        spikes = self.get_agent_property_value(
            id=synapse_id,
            property_name="input_spikes_tensor",
        )
        spikes.append([tick, value])
        self.set_agent_property_value(
            synapse_id, "input_spikes_tensor", spikes  # , [len(spikes), 2]
        )

    def get_spike_times(self, soma_id: int) -> np.array:
        spike_train = super().get_agent_property_value(
            id=soma_id,
            property_name="output_spikes_tensor",
        )
        spike_times = [i for i in range(len(spike_train)) if spike_train[i] > 0]
        return spike_times

    def get_internal_states_history(self, agent_id: int) -> np.array:
        return super().get_agent_property_value(
            id=agent_id, property_name="internal_states_buffer"
        )

    def summary(self) -> str:
        """
        Verbose summary of the network structure.

        :return: str information of netowkr struture
        """
        raise NotImplementedError("summary is not implemented.")
        summary = []
        summary.append("soma information:")
        for soma_id in range(self._agent_factory.num_agents):
            spikes = self.get_agent_property_value(soma_id, "output_spikes")
            summary.append(f"soma: {soma_id} Spike Train: {str(spikes)}")

        summary.append("\n\n\nSynapse information:")
        for presynaptic_soma_id in range(self._agent_factory.num_agents):
            synapses = self.get_agent_property_value(
                presynaptic_soma_id, "output_synapses"
            )
            for synapse in synapses:
                postsynaptic_soma_id = synapse[0]
                weight = synapse[1]
                summary.append(
                    (
                        f"soma {presynaptic_soma_id} -> {postsynaptic_soma_id}"
                        f": weight: {weight}"
                    )
                )

        return "\n".join(summary)

    def save(self, fpath: str):
        super().save(self, fpath)

    def load(self, fpath: str):
        self = super().load(fpath)
