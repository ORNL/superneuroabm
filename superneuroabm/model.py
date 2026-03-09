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

from superneuroabm.step_functions.soma.izh import izh_soma_step_func
from superneuroabm.step_functions.soma.lif import lif_soma_step_func
from superneuroabm.step_functions.soma.lif_soma_adaptive_thr import lif_soma_adaptive_thr_step_func
from superneuroabm.step_functions.soma.srm import srm_soma_step_func
from superneuroabm.step_functions.soma.hg_lif import hg_lif_soma_step_func
from superneuroabm.step_functions.synapse.single_exp import synapse_single_exp_step_func
from superneuroabm.step_functions.synapse.double_exp import synapse_double_exp_step_func
from superneuroabm.step_functions.synapse.stdp.learning_rule_selector import (
    learning_rule_selector,
)
from superneuroabm.util import load_component_configurations
import copy
from mpi4py import MPI

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
            "lif_soma_adaptive_thr": [
                (
                    lif_soma_adaptive_thr_step_func,
                    CURRENT_DIR_ABSPATH / "step_functions" / "soma" / "lif_soma_adaptive_thr.py",
                )
            ],
            "srm_soma": [
                (
                    srm_soma_step_func,
                    CURRENT_DIR_ABSPATH / "step_functions" / "soma" / "srm.py",
                )
            ],
            "hg_lif_soma": [
                (
                    hg_lif_soma_step_func,
                    CURRENT_DIR_ABSPATH / "step_functions" / "soma" / "hg_lif.py",
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
            "double_exp_synapse": [
                (
                    synapse_double_exp_step_func,
                    CURRENT_DIR_ABSPATH
                    / "step_functions"
                    / "synapse"
                    / "double_exp.py",
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
        enable_internal_state_tracking: bool = True,
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
        :param enable_internal_state_tracking: If True, tracks and stores
            internal states history for all agents during simulation.
            If False, disables tracking to reduce memory usage and improve
            performance. Default is True for backward compatibility.
        """
        super().__init__(space=NetworkSpace(ordered=True))

        self.enable_internal_state_tracking = enable_internal_state_tracking

        self.register_global_property("dt", 1e-3)  # Time step (100 μs)
        self.register_global_property("I_bias", 0)  # No bias current

        # Soma properties: (default_value, neighbor_visible)
        # neighbor_visible=True means the property is sent to neighbors during MPI sync
        # Only output_spikes_tensor is read by neighbors (synapses read soma spikes)
        soma_properties = {
            "hyperparameters": ([0.0, 0.0, 0.0, 0.0, 0.0], False),  # k, vth, C, a, b,
            "learning_hyperparameters": (
                [0.0 for _ in range(5)], False
            ),  # STDP_function name, tau_pre_stdp, tau_post_stdp, a_exp_pre, a_exp_post, Wmax, Wmin
            "internal_state": ([0.0, 0.0, 0.0, 0.0], False),  # v, u
            "internal_learning_state": (
                [0.0 for _ in range(3)], False
            ),  # pre_trace, post_trace, dW
            "synapse_delay_reg": ([], False),  # Synapse delay
            "input_spikes_tensor": ([], False),  # input spikes tensor
            "output_spikes_tensor": ([], True),  # NEIGHBOR-VISIBLE: synapses read soma spikes
            "internal_states_buffer": ([], False),
            "internal_learning_states_buffer": ([], False),  # learning states buffer
        }
        # Synapse properties: (default_value, neighbor_visible)
        # Only internal_state is read by neighbors (somas read I_synapse from synapses)
        synapse_properties = {
            "hyperparameters": (
                [0.0 for _ in range(10)], False
            ),  # weight, delay, scale, Tau_fall, Tau_rise, tau_pre_stdp, tau_post_stdp, a_exp_pre, a_exp_post, stdp_history_length
            "learning_hyperparameters": (
                [0.0 for _ in range(5)], False
            ),  # STDP_function name, tau_pre_stdp, tau_post_stdp, a_exp_pre, a_exp_post, Wmax, Wmin
            "internal_state": (
                [0.0 for _ in range(4)], True
            ),  # NEIGHBOR-VISIBLE: somas read Isyn; Isyn, Isyn_supp, pre_trace, post_trace
            "internal_learning_state": (
                [0.0 for _ in range(3)], False
            ),  # pre_trace, post_trace, dW
            "synapse_delay_reg": ([], False),  # Synapse delay
            "input_spikes_tensor": ([], False),  # input spikes tensor
            "output_spikes_tensor": ([], False),
            "internal_states_buffer": ([], False),
            "internal_learning_states_buffer": ([], False),  # learning states buffer
        }
        self._synapse_ids = []
        self._soma_ids = []
        self._soma_reset_states = {}

        # Disable double buffering for properties written by soma:
        # - internal_state: self-access only (v, tcount, tlast)
        # - output_spikes_tensor: synapse reads [t-1], soma writes [t] (different indices)
        # - internal_states_buffer: self-access only, no cross-agent reads
        soma_no_double_buffer = [
            "internal_state",
            "output_spikes_tensor",
            "internal_states_buffer",
        ]

        self._soma_breeds: Dict[str, Breed] = {}
        for breed_name, step_funcs in soma_breed_info.items():
            soma_breed = Breed(breed_name)
            for prop_name, (default_val, neighbor_visible) in soma_properties.items():
                soma_breed.register_property(prop_name, default_val, neighbor_visible=neighbor_visible)
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
                    no_double_buffer=soma_no_double_buffer,
                )
            self.register_breed(soma_breed)
            self._soma_breeds[breed_name] = soma_breed

        # Disable double buffering for properties written by synapse:
        # - internal_state: soma reads at P0 before synapse writes at P100
        # - internal_states_buffer: self-access only, no cross-agent reads
        synapse_no_double_buffer = [
            "internal_state",
            "internal_states_buffer",
        ]

        self._synapse_breeds: Dict[str, Breed] = {}
        for breed_name, step_funcs in synapse_breed_info.items():
            synapse_breed = Breed(breed_name)
            for prop_name, (default_val, neighbor_visible) in synapse_properties.items():
                synapse_breed.register_property(prop_name, default_val, neighbor_visible=neighbor_visible)
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
                    no_double_buffer=synapse_no_double_buffer,
                )
            self.register_breed(synapse_breed)
            self._synapse_breeds[breed_name] = synapse_breed

        # Spike recording state (GPU buffers allocated lazily)
        self._recorded_spikes = []
        self._spike_record_gpu = None
        self._spike_record_count_gpu = None
        self._recorded_soma_ids = None   # None = record all, list = subset
        self._spike_mask_gpu = None      # CuPy float32 bitmask, built lazily
        self._spikes_need_gather = False

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

        Note: This returns the ordered locations [pre_soma_id, post_soma_id].
        These are agent IDs, not local indices.
        """

        return self.get_agent_property_value(
            id=synapse_id, property_name="locations"
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

    def set_recorded_somas(self, soma_ids: list):
        """
        Set a subset of soma IDs whose spikes should be recorded on GPU.
        Non-target somas are filtered out at the kernel level (O(1) bitmask).
        If never called, all somas are recorded (default behavior).

        :param soma_ids: List of soma agent IDs to record.
        """
        self._recorded_soma_ids = soma_ids
        self._spike_mask_gpu = None  # force rebuild on next prepare

    def _reset_agents(self, retain_parameters: bool = True) -> None:
        """
        Internal method to reset all soma and synapse agents to their initial states.

        :param retain_parameters: If True, keeps current learned parameters.
            If False, resets parameters to their default values.
        """
        # Reset all synapses
        for synapse_id in self._synapse_ids:
            # OPTIMIZED: Use depth-2 flattened format [tick, value, tick, value, ...] instead of depth-3 [[tick, value], ...]
            super().set_agent_property_value(
                id=synapse_id,
                property_name="input_spikes_tensor",
                value=[-1, 0.0],  # Flattened: [tick, value]
            )
            # Reset synapse delay registers
            synapse_delay = len(
                super().get_agent_property_value(
                    id=synapse_id, property_name="synapse_delay_reg"
                )
            )
            synapse_delay_reg = [0 for _ in range(synapse_delay)]
            super().set_agent_property_value(
                id=synapse_id,
                property_name="synapse_delay_reg",
                value=synapse_delay_reg,
            )
            # Reset internal states
            super().set_agent_property_value(
                id=synapse_id,
                property_name="internal_state",
                value=self._synapse2defaultinternalstate[synapse_id].copy(),
            )
            super().set_agent_property_value(
                id=synapse_id,
                property_name="internal_learning_state",
                value=self._synapse2defaultinternallearningstate[synapse_id].copy(),
            )
            # Reset parameters to defaults if retain_parameters is False
            if not retain_parameters:
                super().set_agent_property_value(
                    id=synapse_id,
                    property_name="hyperparameters",
                    value=self._synapse2defaultparameters[synapse_id].copy(),
                )
                super().set_agent_property_value(
                    id=synapse_id,
                    property_name="learning_hyperparameters",
                    value=self._synapse2defaultlearningparameters[synapse_id].copy(),
                )

        # Reset all somas
        for soma_id in self._soma_ids:
            # Reset internal states
            super().set_agent_property_value(
                id=soma_id,
                property_name="internal_state",
                value=self._soma_reset_states[soma_id].copy(),
            )
            # Circular buffer: only 2 slots needed (soma writes [t%2], synapse reads [(t-1)%2])
            super().set_agent_property_value(
                id=soma_id,
                property_name="output_spikes_tensor",
                value=[0.0, 0.0],
            )

    def reset(self, retain_parameters: bool = True) -> None:
        """
        Resets all soma and synapse agents to their initial states.

        :param retain_parameters: If True, keeps current learned parameters
            (e.g. STDP weights). If False, resets parameters to defaults.
        """
        # Step 1: SAGESim syncs GPU->AgentFactory, regenerates tensors, frees GPU
        # After this, AgentFactory has all GPU-learned values (including weights)
        super().reset()

        # Step 2: Reset agent states on AgentFactory (keeps hyperparameters if retain=True)
        self._reset_agents(retain_parameters=retain_parameters)

        # Step 3: Regenerate data tensors to reflect the reset states
        super()._regenerate_data_tensors()

        # Step 4: Clear recording state + caches
        self._recorded_spikes = []
        self._spike_record_gpu = None
        self._spike_record_count_gpu = None
        self._spike_mask_gpu = None  # rebuild mask on next prepare
        self._spikes_need_gather = False
        self._agent_factory._prev_agent_data.clear()
        
    def setup(
        self,
        use_gpu: bool = True,
    ) -> None:
        """
        One-time heavy initialization: code gen, JIT, priority analysis.
        Always resets to default state. Call once before simulation loop.
        """
        self._reset_agents(retain_parameters=False)
        self._recorded_spikes = []
        self._spike_record_gpu = None
        self._spike_record_count_gpu = None
        self._spike_mask_gpu = None  # rebuild mask on next prepare
        self._spikes_need_gather = False
        super().setup(use_gpu=use_gpu)

    def simulate(
        self, ticks: int, update_data_ticks: int = 1  # , num_cpu_proc: int = 4
    ) -> None:
        """
        Override of superneuroabm.core.model mainly to register an
        AgentDataCollector to monitor marked output somas.

        """
        _rank = MPI.COMM_WORLD.Get_rank()
        print(f"[Rank {_rank}] simulate preamble start", flush=True)
        for soma_id in self._soma_ids:
            initial_internal_state = super().get_agent_property_value(
                id=soma_id, property_name="internal_state"
            )
            # Allocate full buffer when tracking enabled, minimal dummy buffer when disabled
            if self.enable_internal_state_tracking:
                internal_states_buffer = [initial_internal_state[::] for _ in range(ticks)]
            else:
                # Minimal dummy buffer - single element that gets overwritten each tick
                internal_states_buffer = [initial_internal_state[::]]
            super().set_agent_property_value(
                id=soma_id,
                property_name="internal_states_buffer",
                value=internal_states_buffer,
            )
            # Allocate internal_learning_states_buffer for somas too (for MPI consistency)
            # Somas don't use this, but having consistent structure across agent types
            # prevents issues when MPI workers have different agent type distributions
            initial_internal_learning_state = super().get_agent_property_value(
                id=soma_id, property_name="internal_learning_state"
            )
            if self.enable_internal_state_tracking:
                internal_learning_states_buffer = [
                    initial_internal_learning_state[::] for _ in range(ticks)
                ]
            else:
                internal_learning_states_buffer = [initial_internal_learning_state[::]]
            super().set_agent_property_value(
                id=soma_id,
                property_name="internal_learning_states_buffer",
                value=internal_learning_states_buffer,
            )
        for synapse_id in self._synapse_ids:
            # Sort input spikes by tick for early-exit optimization in
            # get_soma_spike — scan stops once past t_current.
            # Layout: [-1, 0.0, tick, val, tick, val, ...]
            spikes = super().get_agent_property_value(
                id=synapse_id, property_name="input_spikes_tensor"
            )
            if len(spikes) > 2:
                pairs = [(spikes[i], spikes[i + 1]) for i in range(2, len(spikes), 2)]
                pairs.sort(key=lambda p: p[0])
                sorted_spikes = [spikes[0], spikes[1]]
                for t, v in pairs:
                    sorted_spikes.append(t)
                    sorted_spikes.append(v)
                super().set_agent_property_value(
                    id=synapse_id,
                    property_name="input_spikes_tensor",
                    value=sorted_spikes,
                )

            initial_internal_state = super().get_agent_property_value(
                id=synapse_id, property_name="internal_state"
            )
            # Allocate full buffer when tracking enabled, minimal dummy buffer when disabled
            if self.enable_internal_state_tracking:
                internal_states_buffer = [initial_internal_state[::] for _ in range(ticks)]
            else:
                # Minimal dummy buffer - single element that gets overwritten each tick
                internal_states_buffer = [initial_internal_state[::]]
            super().set_agent_property_value(
                id=synapse_id,
                property_name="internal_states_buffer",
                value=internal_states_buffer,
            )

            initial_internal_learning_state = super().get_agent_property_value(
                id=synapse_id, property_name="internal_learning_state"
            )
            # Allocate full buffer when tracking enabled, minimal dummy buffer when disabled
            if self.enable_internal_state_tracking:
                internal_learning_states_buffer = [
                    initial_internal_learning_state[::] for _ in range(ticks)
                ]
            else:
                # Minimal dummy buffer - single element that gets overwritten each tick
                internal_learning_states_buffer = [initial_internal_learning_state[::]]
            super().set_agent_property_value(
                id=synapse_id,
                property_name="internal_learning_states_buffer",
                value=internal_learning_states_buffer,
            )
        self._recorded_spikes = []
        self._spikes_need_gather = False
        print(f"[Rank {_rank}] calling super().simulate()", flush=True)
        super().simulate(ticks, update_data_ticks)  # , num_cpu_proc)
        print(f"[Rank {_rank}] super().simulate() done", flush=True)
        if MPI.COMM_WORLD.Get_size() > 1:
            self._spikes_need_gather = True

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
            breed=self._soma_breeds[breed],
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
        pre_soma_id: int,
        post_soma_id: int,
        config_name: str,
        hyperparameters_overrides: Dict[str, float] = None,
        default_internal_state_overrides: Dict[str, float] = None,
        learning_hyperparameters_overrides: Dict[str, float] = None,
        default_internal_learning_state_overrides: Dict[str, float] = None,
        tags: Set[str] = None,
    ) -> int:
        """
        Creates and adds a Synapse agent.

        Parameters:
            breed (str): Synapse breed name (e.g., 'single_exp_synapse').
            pre_soma_id (int): Presynaptic soma agent ID (or -1 for external input).
            post_soma_id (int): Postsynaptic soma agent ID (or -1 for external output).
            config_name (str): Name of the configuration to use for this synapse.
            hyperparameters_overrides (dict, optional): Dict of hyperparameter overrides.
            default_internal_state_overrides (dict, optional): Dict of internal state overrides.
            learning_hyperparameters_overrides (dict, optional): Dict of learning hyperparameter overrides.
            default_internal_learning_state_overrides (dict, optional): Dict of internal learning state overrides.
            tags (set of str, optional): Tags to associate with this synapse.

        Returns:
            int: SAGESim agent ID of the created synapse.
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

        # Connect synapse to somas using SAGESim's API
        # With ordered=True, connections are maintained in insertion order
        # So synapse's locations will be [pre_soma_id, post_soma_id] after we connect them

        # IMPORTANT: Connect in order [pre, post] to maintain ordered locations
        # First connection: pre_soma (if exists)
        # -1 indicates external input
        if pre_soma_id != -1:
            network_space.connect_agents(synapse_id, pre_soma_id, directed=True)
            self.soma2synapse_map[pre_soma_id]["post"].add(synapse_id)
            self.synapse2soma_map[synapse_id]["pre"] = pre_soma_id
        else:
            # For external input, manually add -1 to locations to maintain [pre, post] order
            network_space.get_location(synapse_id).append(-1)
            self.synapse2soma_map[synapse_id]["pre"] = -1  # External input
            tags.add("input_synapse")


        # Second connection: post_soma (if exists)
        # -1 indicates external output
        if post_soma_id != -1:
            network_space.connect_agents(synapse_id, post_soma_id, directed=True)
            network_space.connect_agents(post_soma_id, synapse_id, directed=True)  # Bidirectional for STDP
            self.synapse2soma_map[synapse_id]["post"] = post_soma_id
            self.soma2synapse_map[post_soma_id]["pre"].add(synapse_id)
        else:
            self.synapse2soma_map[synapse_id]["post"] = -1
            # For external output (rare), manually add -1
            network_space.get_location(synapse_id).append(-1)

        self.agentid2config[synapse_id] = config_name
        tags.update({"synapse", breed})
        for tag in tags:
            self.tag2component[tag].add(synapse_id)
        return synapse_id

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
        # OPTIMIZED: Store as flattened [tick, value, tick, value, ...] (depth 2) instead of [[tick, value], ...] (depth 3) 
        spikes.append(tick)
        spikes.append(value)
        self.set_agent_property_value(
            synapse_id, "input_spikes_tensor", spikes
        )
    
    def add_spike_list(self, synapse_id: int, spike_list):
        """
        Schedules a list of external input spikes to this synapse.

        :param spike_list: List of [tick, value] pairs
        """
        spikes = self.get_agent_property_value(
            id=synapse_id,
            property_name="input_spikes_tensor",
        )
        # OPTIMIZED: Flatten [[tick, value], ...] to [tick, value, tick, value, ...]
        for spike_pair in spike_list:
            spikes.append(spike_pair[0])  # tick
            spikes.append(spike_pair[1])  # value
        self.set_agent_property_value(
            synapse_id, "input_spikes_tensor", spikes
        )

    # ------------------------------------------------------------------
    # GPU kernel extension hooks for spike recording
    # ------------------------------------------------------------------

    def _get_extra_kernel_config(self) -> dict:
        prop_idx = self._agent_factory._property_name_2_index["output_spikes_tensor"]
        return {
            'extra_kernel_params': ['spike_record', 'spike_record_count', 'spike_mask'],
            'post_breed_step_code': [
                (
                    [
                        f'_sv = a{prop_idx}[_real_idx][thread_local_tick % 2]',
                        'if _sv > 0.0 and spike_mask[_real_idx] > 0.0:',
                        '\t_slot = jit.atomic_add(spike_record_count, 0, 1)',
                        '\tspike_record[_slot * 2] = agent_ids[_real_idx]',
                        '\tspike_record[_slot * 2 + 1] = float(thread_local_tick)',
                    ],
                    True,  # once_per_breed
                    0,     # only_priority — only emit for soma priority
                )
            ],
        }

    def _prepare_kernel_extras(self, num_local_agents, sync_ticks):
        import cupy as cp
        if self._spike_record_gpu is None:
            max_slots = max(10000, num_local_agents * sync_ticks // 100)
            self._spike_record_gpu = cp.full(max_slots * 2, cp.nan, dtype=cp.float32)
            self._spike_record_count_gpu = cp.zeros(1, dtype=cp.int32)
        self._spike_record_count_gpu[0] = 0
        # Build spike mask: 1.0 for target somas, 0.0 for others
        if self._spike_mask_gpu is None:
            buf = self._gpu_buffers
            mask = cp.zeros(buf.agent_capacity, dtype=cp.float32)
            if self._recorded_soma_ids is None:
                mask[:num_local_agents] = 1.0  # record all
            else:
                for sid in self._recorded_soma_ids:
                    idx = buf.agent_id_to_index.get(sid, -1)
                    if 0 <= idx < num_local_agents:
                        mask[idx] = 1.0
            self._spike_mask_gpu = mask
        return (self._spike_record_gpu, self._spike_record_count_gpu, self._spike_mask_gpu)

    def _process_kernel_extras(self):
        count = int(self._spike_record_count_gpu[0].get())
        if count > 0:
            self._recorded_spikes.extend(
                self._spike_record_gpu[:count * 2].get().tolist()
            )

    def _ensure_spikes_gathered(self):
        """MPI allgather of recorded spikes (collective, idempotent)."""
        if not self._spikes_need_gather:
            return
        comm = MPI.COMM_WORLD
        all_spikes = comm.allgather(self._recorded_spikes)
        self._recorded_spikes = []
        for rank_spikes in all_spikes:
            self._recorded_spikes.extend(rank_spikes)
        self._spikes_need_gather = False

    def get_spike_times(self, soma_id: int) -> list:
        self._ensure_spikes_gathered()
        spikes = []
        data = self._recorded_spikes
        for i in range(0, len(data), 2):
            if int(data[i]) == soma_id:
                spikes.append(int(data[i + 1]))
        return spikes

    def get_all_spike_times(self) -> dict:
        """Return {soma_id: [tick, ...]} for all recorded spikes."""
        self._ensure_spikes_gathered()
        result = defaultdict(list)
        data = self._recorded_spikes
        for i in range(0, len(data), 2):
            result[int(data[i])].append(int(data[i + 1]))
        return dict(result)

    def get_internal_states_history(self, agent_id: int) -> np.array:
        if not self.enable_internal_state_tracking:
            return []
        return super().get_agent_property_value(
            id=agent_id, property_name="internal_states_buffer"
        )

    def get_internal_learning_states_history(self, agent_id: int) -> np.array:
        if not self.enable_internal_state_tracking:
            return []
        return super().get_agent_property_value(
            id=agent_id, property_name="internal_learning_states_buffer"
        )

