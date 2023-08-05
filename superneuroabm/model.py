"""
Model class for building an SNN
"""
from typing import Dict, Callable, List

import numpy as np

from superneuroabm.core.model import Model
from superneuroabm.core.agent import Breed
from superneuroabm.neuron import (
    synapse_step_func,
    neuron_step_func,
)


class NeuromorphicModel(Model):
    def __init__(
        self,
        neuron_breed_info: Dict[str, List[Callable]] = {
            "Neuron": [neuron_step_func, synapse_step_func]
        },
    ) -> None:
        """
        Creates an SNN Model and provides methods to create, simulate,
        and monitor neurons and synapses.

        :param use_cuda: True if the system supports CUDA GPU
            acceleration.
        :param neuron_breed_info: Dict of breed name to List of
            Callable step functions. If specifed, will override
            the default neuron breed and neuron and synapse step
            functions, allowing for multi-breed simulations.
            Step functions will be executed on the respective
            breed every simulation step in the order specifed in the
            list.
        """
        super().__init__()

        axonal_delay = 1
        neuron_properties = {
            "threshold": 1,
            "reset_state": 0,
            "leak": 0,
            "refractory_period": 0,
            "output_synapses": [],
            "t_elapse": 0,
            "internal_state": 0,
            "neuron_delay_reg": [0 for _ in range(axonal_delay)],
            "input_spikes": [],
            "output_synapses_learning_params": [],
            "output_spikes": [],
        }
        max_dims = {
            "threshold": [],
            "reset_state": [],
            "leak": [],
            "refractory_period": [],
            "output_synapses": [],
            "t_elapse": [],
            "internal_state": [],
            "neuron_delay_reg": None,
            "input_spikes": None,
            "output_synapses_learning_params": None,
            "output_spikes": None,
        }

        self._neuron_breeds: Dict[str, Breed] = {}
        for breed_name, step_funcs in neuron_breed_info.items():
            neuron_breed = Breed(breed_name)
            for prop_name, default_val in neuron_properties.items():
                neuron_breed.register_property(
                    prop_name, default_val, max_dims[prop_name]
                )
            for step_func_order, step_func in enumerate(step_funcs):
                neuron_breed.register_step_func(
                    step_func=step_func, priority=step_func_order
                )
            self.register_breed(neuron_breed)
            self._neuron_breeds[breed_name] = neuron_breed

        self._output_synapsess_max_dim = [0, 2]
        self._original_output_synapse_weights = {}
        self._synapse_index_map = {}

    def setup(
        self,
        use_cuda: bool = False,
        output_buffer_len: int = 1000,
        retain_weights=False,
    ) -> None:
        """
        Resets the simulation and initializes agents.

        :param retain_weights: False by default. If True, updated weights are
            not reset upon setup.
        """

        neuron_ids = self._agent_factory.num_agents
        for neuron_id in range(neuron_ids):
            # Clear input spikes
            new_input_spikes = []
            super().set_agent_property_value(
                id=neuron_id,
                property_name="input_spikes",
                value=new_input_spikes,
                dims=[0, 2],
            )
            # Clear output buffer
            output_buffer = [0 for _ in range(output_buffer_len)]
            super().set_agent_property_value(
                id=neuron_id,
                property_name="output_spikes",
                value=output_buffer,
                dims=[output_buffer_len],
            )
            # Clear internal states
            reset_state = super().get_agent_property_value(
                id=neuron_id, property_name="reset_state"
            )
            super().set_agent_property_value(
                id=neuron_id,
                property_name="internal_state",
                value=reset_state,
                dims=[],
            )
            # Clear neuron delay registers
            axonal_delay = len(
                super().get_agent_property_value(
                    id=neuron_id, property_name="neuron_delay_reg"
                )
            )
            neuron_delay_reg = [0 for _ in range(axonal_delay)]
            self.set_agent_property_value(
                neuron_id, "neuron_delay_reg", neuron_delay_reg, [axonal_delay]
            )
            # Clear synaptic delay registers
            output_synapses = self.get_agent_property_value(
                neuron_id, "output_synapses"
            )
            max_synapse_info_len = 2
            # Iterate through synapse info lists for this neuron
            for i in range(len(output_synapses)):
                # Clear weight (2nd element) if necessary
                if not retain_weights:
                    output_synapses[i][
                        1
                    ] = self._original_output_synapse_weights[neuron_id][i]
                # Clear the rest of the list, which is the delay register
                j = 2
                while j < len(output_synapses[i]):
                    output_synapses[i][j] = 0
                    j += 1
                max_synapse_info_len = max(
                    len(output_synapses[i]), max_synapse_info_len
                )
            self.set_agent_property_value(
                neuron_id,
                "output_synapses",
                output_synapses,
                [len(output_synapses), max_synapse_info_len],
            )

        super().setup(use_cuda=use_cuda)

    def simulate(
        self, ticks: int, update_data_ticks: int = 1, num_cpu_proc: int = 4
    ) -> None:
        """
        Override of superneuroabm.core.model mainly to register an
        AgentDataCollector to monitor marked output Neurons.

        """
        super().simulate(ticks, update_data_ticks, num_cpu_proc)

    def create_neuron(
        self,
        breed: str = "Neuron",
        threshold: float = 1,
        reset_state: float = 0,
        leak: float = 0,
        refractory_period: int = 0,
        axonal_delay: int = 1,
    ) -> int:
        """
        Creates and Neuron agent.

        :return: SAGESim agent id of neuron

        """
        delay_reg = [0 for _ in range(axonal_delay)]
        neuron_id = super().create_agent_of_breed(
            breed=self._neuron_breeds[breed],
            threshold=threshold,
            reset_state=reset_state,
            leak=leak,
            refractory_period=refractory_period,
        )
        self.set_agent_property_value(
            neuron_id, "neuron_delay_reg", delay_reg, [axonal_delay]
        )
        self.set_agent_property_value(
            neuron_id, "internal_state", reset_state, []
        )
        # synapse_infos = []
        # self._output_synapsess.append(synapse_infos)
        return neuron_id

    def create_synapse(
        self,
        pre_neuron_id: int,
        post_neuron_id: int,
        weight: int = 1,
        synaptic_delay: int = 1,
        synapse_learning_params: List[float] = None,
    ) -> None:
        """
        Creates and adds Synapse agent.

        :param pre_neuron_id: int presynaptic neuron id
        :param post_neuron_id: int postsynaptic neuron id
        :param weight: weight of synapse
        :param synaptic delay: number of timesteps to delay synapse by
        :param synapse_learning_params: Optional. Any parameters used in a learning
            enabled step function. Must be specified in order of use
            in step function.
        """
        output_synapses = self.get_agent_property_value(
            pre_neuron_id, "output_synapses"
        )
        if pre_neuron_id in self._synapse_index_map:
            output_synapse_index_map = self._synapse_index_map[pre_neuron_id]
            if post_neuron_id in output_synapse_index_map:
                raise ValueError(
                    f"Synapse {pre_neuron_id} -> {post_neuron_id}already exists"
                )
            else:
                output_synapse_index_map[post_neuron_id] = len(output_synapses)
        else:
            output_synapse_index_map = {post_neuron_id: 0}
        self._synapse_index_map[pre_neuron_id] = output_synapse_index_map

        # Update or enter new synapse params
        output_synapses = self.get_agent_property_value(
            pre_neuron_id, "output_synapses"
        )
        delay_reg = [0 for _ in range(synaptic_delay)]
        synapse_info = [post_neuron_id, weight]
        synapse_info.extend(delay_reg)
        output_synapses.append(synapse_info)
        self.set_agent_property_value(
            pre_neuron_id,
            "output_synapses",
            output_synapses,
            [len(output_synapses), len(synapse_info)],
        )

        # store original weights
        self._original_output_synapse_weights[
            pre_neuron_id
        ] = self._original_output_synapse_weights.get(pre_neuron_id, [])
        self._original_output_synapse_weights[pre_neuron_id].append(weight)

        # Update or enter learning params
        if synapse_learning_params:
            synapses_learning_params = self.get_agent_property_value(
                pre_neuron_id, "output_synapses_learning_params"
            )
            synapses_learning_params.append(synapse_learning_params)
            self.set_agent_property_value(
                id=pre_neuron_id,
                property_name="output_synapses_learning_params",
                value=synapses_learning_params,
                dims=[
                    len(synapses_learning_params),
                    len(synapse_learning_params),
                ],
            )

    def update_synapse(
        self,
        pre_neuron_id: int,
        post_neuron_id: int,
        weight: int = None,
        synapse_learning_params: List[float] = None,
    ):
        if pre_neuron_id in self._synapse_index_map:
            output_synapse_index_map = self._synapse_index_map[pre_neuron_id]
            if post_neuron_id in output_synapse_index_map:
                synapse_idx = output_synapse_index_map[post_neuron_id]
            else:
                raise ValueError(
                    f"Synapse {pre_neuron_id} -> {post_neuron_id} does not exist"
                )
        else:
            raise ValueError(
                f"Synapse {pre_neuron_id} -> {post_neuron_id} does not exist"
            )

        # Update new synapse params
        if weight != None:
            output_synapses = self.get_agent_property_value(
                pre_neuron_id, "output_synapses"
            )
            output_synapses[synapse_idx][1] = weight
            self.set_agent_property_value(
                pre_neuron_id,
                "output_synapses",
                output_synapses,
                [len(output_synapses), len(output_synapses[synapse_idx])],
            )

        # Update or enter learning params
        if synapse_learning_params:
            synapses_learning_params = self.get_agent_property_value(
                pre_neuron_id, "output_synapses_learning_params"
            )
            synapses_learning_params[synapse_idx] = synapse_learning_params
            self.set_agent_property_value(
                id=pre_neuron_id,
                property_name="output_synapses_learning_params",
                value=synapses_learning_params,
                dims=[
                    len(synapses_learning_params),
                    len(synapse_learning_params),
                ],
            )

    def spike(self, neuron_id: int, tick: int, value: float) -> None:
        """
        Schedules an external input spike to this Neuron.

        :param tick: tick at which spike should be triggered
        :param value: spike value
        """
        spikes = self.get_agent_property_value(
            id=neuron_id,
            property_name="input_spikes",
        )
        spikes.append([tick, value])
        self.set_agent_property_value(
            neuron_id, "input_spikes", spikes, [len(spikes), 2]
        )

    def get_spikes(self, neuron_id: int) -> np.array:
        spike_train = super().get_agent_property_value(
            id=neuron_id,
            property_name="output_spikes",
        )
        spike_times = [
            i for i in range(len(spike_train)) if spike_train[i] > 0
        ]
        return spike_times

    def summary(self) -> str:
        """
        Verbose summary of the network structure.

        :return: str information of netowkr struture
        """
        summary = []
        summary.append("Neuron information:")
        for neuron_id in range(self._agent_factory.num_agents):
            spikes = self.get_agent_property_value(neuron_id, "output_spikes")
            summary.append(f"Neuron: {neuron_id} Spike Train: {str(spikes)}")

        summary.append("\n\n\nSynapse information:")
        for presynaptic_neuron_id in range(self._agent_factory.num_agents):
            synapses = self.get_agent_property_value(
                presynaptic_neuron_id, "output_synapses"
            )
            for synapse in synapses:
                postsynaptic_neuron_id = synapse[0]
                weight = synapse[1]
                summary.append(
                    (
                        f"Neuron {presynaptic_neuron_id} -> {postsynaptic_neuron_id}"
                        f": weight: {weight}"
                    )
                )

        return "\n".join(summary)

    def save(self, fpath: str):
        super().save(self, fpath)

    def load(self, fpath: str):
        self = super().load(fpath)
