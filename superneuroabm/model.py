"""
Model class for building an SNN
"""

from collections import defaultdict
from typing import Dict, Callable, List, Set
from pathlib import Path

import numpy as np
import cupy as cp
from sagesim.space import NetworkSpace
from sagesim.model import Model
from sagesim.breed import Breed
from pathlib import Path

from superneuroabm.step_functions.soma.izh import izh_soma_step_func
from superneuroabm.step_functions.soma.lif import lif_soma_step_func
from superneuroabm.step_functions.synapse.single_exp import synapse_single_exp_step_func
from superneuroabm.step_functions.synapse.stdp.learning_rule_selector import (
    learning_rule_selector,
)
from superneuroabm.util import load_component_configurations
import copy

CURRENT_DIR_ABSPATH = Path(__file__).resolve().parent


class NeuromorphicModel(Model):
    def __init__(
        self,
        soma_breed_info: Dict[str, List[Callable]] = {
            "izh_soma": [
                (
                    izh_soma_step_func,
                    CURRENT_DIR_ABSPATH / "step_functions" / "soma" / "izh.py",
                )
            ],
            "lif_soma": [
                (
                    lif_soma_step_func,
                    CURRENT_DIR_ABSPATH / "step_functions" / "soma" / "lif.py",
                )
            ],
        },
        synapse_breed_info: Dict[str, List[Callable]] = {
            "single_exp_synapse": [
                (
                    synapse_single_exp_step_func,
                    CURRENT_DIR_ABSPATH
                    / "step_functions"
                    / "synapse"
                    / "single_exp.py",
                ),
                (
                    learning_rule_selector,
                    CURRENT_DIR_ABSPATH
                    / "step_functions"
                    / "synapse"
                    / "stdp"
                    / "learning_rule_selector.py",
                ),
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
            "connectivity": [],  # placeholder for connectivity info
            "hyperparameters": [0.0, 0.0, 0.0, 0.0, 0.0],  # k, vth, C, a, b,
            "learning_hyperparameters": [
                0.0 for _ in range(5)
            ],  # STDP_function name, tau_pre_stdp, tau_post_stdp, a_exp_pre, a_exp_post, Wmax, Wmin
            "internal_state": [0.0, 0.0, 0.0, 0.0],  # v, u
            "internal_learning_state": [
                0.0 for _ in range(3)
            ],  # pre_trace, post_trace, dW
            "synapse_delay_reg": [],  # Synapse delay
            "input_spikes_tensor": [],  # input spikes tensor
            "output_spikes_tensor": [],
            "internal_states_buffer": [],
            "internal_learning_states_buffer": [],  # learning states buffer
        }
        synapse_properties = {
            "connectivity": [],  # hold pre and post soma ids, needed since locations is unordered
            "hyperparameters": [
                0.0 for _ in range(10)
            ],  # weight, delay, scale, Tau_fall, Tau_rise, tau_pre_stdp, tau_post_stdp, a_exp_pre, a_exp_post, stdp_history_length
            "learning_hyperparameters": [
                0.0 for _ in range(5)
            ],  # STDP_function name, tau_pre_stdp, tau_post_stdp, a_exp_pre, a_exp_post, Wmax, Wmin
            "internal_state": [
                0.0 for _ in range(4)
            ],  # Isyn, Isyn_supp, pre_trace, post_trace
            "internal_learning_state": [
                0.0 for _ in range(3)
            ],  # pre_trace, post_trace, dW
            "synapse_delay_reg": [],  # Synapse delay
            "input_spikes_tensor": [],  # input spikes tensor
            "output_spikes_tensor": [],
            "internal_states_buffer": [],
            "internal_learning_states_buffer": [],  # learning states buffer
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

        self.tag2component = defaultdict(set)  # tag -> agent_id

        # Load and hold configurations
        self.agentid2config = {}
        self._component_configurations = load_component_configurations()

        self.synapse2soma_map = defaultdict(
            dict
        )  # synapse_id -> "pre" or "post" -> soma_id
        self.soma2synapse_map = defaultdict(
            lambda: defaultdict(set)
        )  # soma_id -> "pre" or "post" -> List[synapse_id]
        self._synapse2defaultparameters: Dict[int, List[float]] = {}
        self._synapse2defaultlearningparameters: Dict[int, List[float]] = {}
        self._synapse2defaultinternalstate: Dict[int, List[float]] = {}
        self._synapse2defaultinternallearningstate: Dict[int, List[float]] = {}

    def set_global_property_value(name: str, value: float) -> None:
        if name in super().globals:
            super().set_global_property_value(name, value)
        else:
            super().register_global_property(name, value)

    def get_global_property_value(name: str) -> float:
        return super().get_global_property_value(name)

    def get_agent_config_name(self, agent_id: int) -> Dict[str, any]:
        """
        Returns the configuration of the agent with the given ID.
        """
        return self.agentid2config.get(agent_id, None)

    def get_agent_breed(self, agent_id: int) -> str:
        """
        Returns the breed of the agent with the given ID.
        """
        breed_idx = int(
            self.get_agent_property_value(id=agent_id, property_name="breed")
        )
        return list(self._agent_factory.breeds)[breed_idx]

    def get_synapse_connectivity(self, synapse_id: int) -> List[int]:
        """
        Returns the connectivity of the synapse with the given ID.
        The connectivity is a list of length 2 containing pre and post soma IDs.
        """
        return self.get_agent_property_value(
            id=synapse_id, property_name="connectivity"
        )

    def get_agent_config_diff(self, agent_id: int) -> Dict[str, any]:
        """
        Returns the configuration overrides for the agent with the given ID.
        """
        component_class = (
            "soma" if agent_id in self.get_agents_with_tag("soma") else "synapse"
        )
        breed_name = self.get_agent_breed(agent_id).name
        config_name = self.get_agent_config_name(agent_id)
        config = self._component_configurations[component_class][breed_name][
            config_name
        ]
        overrides = {}
        # Must use Python 3.7+ dict comprehension syntax for ordered dicts
        property_names = config.keys()
        for property_name in property_names:
            config_property_key_values = config.get(property_name, {})
            current_property_key_values = self.get_agent_property_value(
                id=agent_id, property_name=property_name
            )
            diffs = {
                k: (
                    v,
                    current_property_key_values[i],
                    v - current_property_key_values[i],
                )
                for i, (k, v) in enumerate(config_property_key_values.items())
                if v != current_property_key_values[i]
            }

            overrides[property_name] = {
                k: current_property_key_values[i]
                for i, (k, v) in enumerate(config_property_key_values.items())
                if v != current_property_key_values[i]
            }
        return overrides

    def get_agents_with_tag(self, tag: str) -> Set[int]:
        """
        Returns a list of agent IDs associated with the given tag.

        :param tag: The tag to filter agents by.
        :return: List of agent IDs that have the specified tag.
        """
        return self.tag2component.get(tag, set())

    def setup(
        self,
        use_gpu: bool = True,
        retain_parameters=True,
    ) -> None:
        """
        Resets the simulation and initializes agents.

        :param retain_parameters: False by default. If True, parameters are
            reset to their default values upon setup.
        """

        self.register_global_property("dt", 1e-3)  # Time step (100 μs)
        self.register_global_property("I_bias", 0)  # No bias current

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
            if not retain_parameters:
                # Reset all synapse parameters to their default values
                default_synapse_parameters = self._synapse2defaultparameters[
                    synapse_id
                ]  # weight, delay, scale, Tau_fall, Tau_rise, tau_pre_stdp, tau_post_stdp, a_exp_pre, a_exp_post, stdp_history_length
                super().set_agent_property_value(
                    id=synapse_id,
                    property_name="hyperparameters",
                    value=default_synapse_parameters,
                )
                default_synapse_learning_hyperparameters = (
                    self._synapse2defaultlearningparameters[synapse_id]
                )
                super().set_agent_property_value(
                    id=synapse_id,
                    property_name="learning_hyperparameters",
                    value=default_synapse_learning_hyperparameters,
                )
            # Clear internal states

            synapse_internal_state = self._synapse2defaultinternalstate[synapse_id]
            super().set_agent_property_value(
                id=synapse_id,
                property_name="internal_state",
                value=synapse_internal_state,
            )
            synapse_internal_learning_state = (
                self._synapse2defaultinternallearningstate[synapse_id]
            )
            super().set_agent_property_value(
                id=synapse_id,
                property_name="internal_learning_state",
                value=synapse_internal_learning_state,
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

            initial_internal_learning_state = super().get_agent_property_value(
                id=synapse_id, property_name="internal_learning_state"
            )
            internal_learning_states_buffer = [
                initial_internal_learning_state[::] for _ in range(ticks)
            ]
            super().set_agent_property_value(
                id=synapse_id,
                property_name="internal_learning_states_buffer",
                value=internal_learning_states_buffer,
            )
        super().simulate(ticks, update_data_ticks)  # , num_cpu_proc)

    def create_soma(
        self,
        breed: str,
        config_name: str,
        hyperparameters_overrides: Dict[str, float] = None,
        default_internal_state_overrides: Dict[str, float] = None,
        tags: Set[str] = None,
    ) -> int:
        """
        Creates and soma agent.

        :return: SAGESim agent id of soma

        """
        tags = tags if tags else set()

        # Get relevant configuration
        config = copy.deepcopy(
            self._component_configurations["soma"][breed][config_name]
        )
        # Apply overrides to hyperparameters and default internal state
        if hyperparameters_overrides:
            for parameter_name, parameter_value in hyperparameters_overrides.items():
                config["hyperparameters"][parameter_name] = parameter_value
        if default_internal_state_overrides:
            for state_name, state_value in default_internal_state_overrides.items():
                config["internal_state"][state_name] = state_value

        hyperparameters = [float(val) for val in config["hyperparameters"].values()]
        default_internal_state = [
            float(val) for val in config["internal_state"].values()
        ]

        soma_id = super().create_agent_of_breed(
            breed=self._soma_breeds[breed],  # TODO fix
            hyperparameters=hyperparameters,
            internal_state=default_internal_state,
        )

        self._soma_ids.append(soma_id)
        self._soma_reset_states[soma_id] = default_internal_state

        self.agentid2config[soma_id] = config_name

        tags.update({"soma", breed})
        for tag in tags:
            self.tag2component[tag].add(soma_id)
        return soma_id

    def create_synapse(
        self,
        breed: str,
        pre_soma_id: int,  # TODO edit
        post_soma_id: int,
        config_name: str,
        hyperparameters_overrides: List[float] = None,
        default_internal_state_overrides: List[float] = None,
        learning_hyperparameters_overrides: List[float] = None,
        default_internal_learning_state_overrides: List[float] = None,
        tags: Set[str] = None,
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
        tags = tags if tags else set()

        # Get relevant configuration
        config = copy.deepcopy(
            self._component_configurations["synapse"][breed][config_name]
        )
        # Apply overrides to hyperparameters and default internal state
        if hyperparameters_overrides:
            for parameter_name, parameter_value in hyperparameters_overrides.items():
                config["hyperparameters"][parameter_name] = parameter_value
        if default_internal_state_overrides:
            for state_name, state_value in default_internal_state_overrides.items():
                config["internal_state"][state_name] = state_value
        if learning_hyperparameters_overrides:
            for (
                parameter_name,
                parameter_value,
            ) in learning_hyperparameters_overrides.items():
                config["learning_hyperparameters"][parameter_name] = parameter_value
        if default_internal_learning_state_overrides:
            for (
                state_name,
                state_value,
            ) in default_internal_learning_state_overrides.items():
                config["internal_learning_state"][state_name] = state_value
        hyperparameters = [float(val) for val in config["hyperparameters"].values()]
        default_internal_state = [
            float(val) for val in config["internal_state"].values()
        ]
        learning_hyperparameters = [
            float(val)
            for val in config.get(
                "learning_hyperparameters", {"stdp_type": -1}
            ).values()
        ]
        default_internal_learning_state = [
            float(val) for val in config.get("internal_learning_state", {}).values()
        ]

        synaptic_delay = int(hyperparameters[1])
        delay_reg = [0 for _ in range(synaptic_delay)]
        synapse_id = self.create_agent_of_breed(
            breed=self._synapse_breeds[breed],
            hyperparameters=hyperparameters,
            learning_hyperparameters=learning_hyperparameters,
            internal_state=default_internal_state,
            internal_learning_state=default_internal_learning_state,
            synapse_delay_reg=delay_reg,
        )
        self._synapse2defaultparameters[synapse_id] = hyperparameters
        self._synapse2defaultlearningparameters[synapse_id] = learning_hyperparameters
        self._synapse2defaultinternalstate[synapse_id] = default_internal_state
        self._synapse2defaultinternallearningstate[synapse_id] = (
            default_internal_learning_state
        )
        self._synapse_ids.append(synapse_id)

        network_space: NetworkSpace = self.get_space()
        if not np.isnan(pre_soma_id):
            network_space.connect_agents(synapse_id, pre_soma_id, directed=True)
            self.soma2synapse_map[pre_soma_id]["post"].add(synapse_id)
            self.synapse2soma_map[synapse_id]["pre"] = pre_soma_id
        else:
            self.synapse2soma_map[synapse_id]["pre"] = float("nan")  # External input
            tags.add("input_synapse")
        if not np.isnan(post_soma_id):
            network_space.connect_agents(
                synapse_id, post_soma_id, directed=True
            )  # Necessary due to STDP
            network_space.connect_agents(
                post_soma_id, synapse_id, directed=True
            )  # Necessary due to STDP
            self.synapse2soma_map[synapse_id]["post"] = post_soma_id
            self.soma2synapse_map[post_soma_id]["pre"].add(synapse_id)
        else:
            self.synapse2soma_map[synapse_id]["post"] = float("nan")

        # Locations is unordered so we need to store pre and post soma ids
        # in an ordered manner using a property
        self.set_agent_property_value(
            id=synapse_id,
            property_name="connectivity",
            value=[pre_soma_id, post_soma_id],
        )

        self.agentid2config[synapse_id] = config_name
        tags.update({"synapse", breed})
        for tag in tags:
            self.tag2component[tag].add(synapse_id)
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

    def get_internal_learning_states_history(self, agent_id: int) -> np.array:
        return super().get_agent_property_value(
            id=agent_id, property_name="internal_learning_states_buffer"
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
