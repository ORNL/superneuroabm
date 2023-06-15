"""
Model class for building an SNN
"""

import numpy as np

from superneuroabm.core.model import Model
from superneuroabm.core.agent import Breed
from superneuroabm.neuron import (
    synapse_step_func,
    neuron_step_func,
)


class NeuromorphicModel(Model):
    def __init__(self, use_cuda: bool = False) -> None:
        """
        Creates an SNN Model and provides methods to create, simulate,
        and monitor neurons and synapses.

        :param use_cuda: True if the system supports CUDA GPU acceleration
        """
        super().__init__(name="NeuromorphicModel", use_cuda=use_cuda)
        # dictionary for datacollectors for each neuron to be tracked
        """self._neuron_spike_collectors = {}"""

        axonal_delay = 1
        neuron_properties = {
            "threshold": 1,
            "reset_state": 0,
            "leak": 0,
            "refactory_period": 0,
            "output_synapses": [],
            "t_elapse": 0,
            "internal_state": 0,
            "neuron_delay_reg": [0 for _ in range(axonal_delay)],
            "input_spikes": [],
            "output_spikes": [],
        }
        max_dims = {
            "threshold": [],
            "reset_state": [],
            "leak": [],
            "refactory_period": [],
            "output_synapses": None,
            "t_elapse": [],
            "internal_state": [],
            "neuron_delay_reg": None,
            "input_spikes": None,
            "output_spikes": None,
        }

        self._neuron_breed = Breed("Neuron")
        for prop_name, default_val in neuron_properties.items():
            self._neuron_breed.register_property(
                prop_name, default_val, max_dims[prop_name]
            )
        self._neuron_breed.register_step_func(
            step_func=neuron_step_func, priority=0
        )
        self._neuron_breed.register_step_func(
            step_func=synapse_step_func, priority=1
        )

        self.register_breed(self._neuron_breed)
        self.register_breed(self._neuron_breed)
        # self._output_synapsess = []
        self._output_synapsess_max_dim = [0, 2]

    def setup(self, output_buffer_len: int = 1000) -> None:
        def get_neurons(breed: int, **kwargs):
            return breed == 1

        neuron_ids = self.get_agents_with(query=get_neurons)
        for neuron_id in neuron_ids:
            output_buffer = [0 for _ in range(output_buffer_len)]
            super().set_agent_property_value(
                id=neuron_id,
                property_name="output_spikes",
                value=output_buffer,
                dims=[output_buffer_len],
            )
            """super().set_agent_property_value(
                id=neuron_id,
                property_name="output_synapses",
                value=self._output_synapsess[neuron_id],
                dims=
            )"""
        super().setup()

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
        threshold: float = 1,
        reset_state: float = 0,
        leak: float = 0,
        refactory_period: int = 0,
        axonal_delay: int = 1,
    ) -> int:
        """
        Creates and Neuron agent.

        :return: SAGESim agent id of neuron

        """
        delay_reg = [0 for _ in range(axonal_delay)]
        neuron_id = super().create_agent_of_breed(
            breed=self._neuron_breed,
            threshold=threshold,
            reset_state=reset_state,
            leak=leak,
            refactory_period=refactory_period,
        )
        self.set_agent_property_value(
            neuron_id, "neuron_delay_reg", delay_reg, [axonal_delay]
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
    ) -> None:
        """
        Creates and adds Synapse agent.

        """
        delay_reg = [0 for _ in range(synaptic_delay)]
        synapse_info = [post_neuron_id, weight]
        synapse_info.extend(delay_reg)
        # self._output_synapsess[pre_neuron_id].append(synapse_info)
        output_synapses = self.get_agent_property_value(
            pre_neuron_id, "output_synapses"
        )
        output_synapses.append(synapse_info)
        self.set_agent_property_value(
            pre_neuron_id,
            "output_synapses",
            output_synapses,
            [len(output_synapses), len(synapse_info)],
        )

        max_synapses = max(
            self._output_synapsess_max_dim[0],
            len(output_synapses),
        )
        max_delay_reg_len = max(
            self._output_synapsess_max_dim[0], len(synapse_info)
        )
        self._output_synapsess_max_dim = [max_synapses, max_delay_reg_len]

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
        # Function to print
        # TODO: Have a better way to represent the network structure(?)
        summary = []
        for k in range(len(self._agents)):
            summary.append(f"Agent: {k} Type:{type(self.get_agent(k))}")
        return "\n".join(summary)
