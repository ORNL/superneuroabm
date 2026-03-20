"""
Model class for building an SNN
"""

from collections import defaultdict
from typing import Dict, List, Set
from pathlib import Path

import numpy as np
import cupy as cp
from sagesim.space import NetworkSpace
from sagesim.model import Model
from sagesim.breed import Breed

from superneuroabm.step_functions.soma.izh import izh_soma_step_func
from superneuroabm.step_functions.soma.lif import lif_soma_step_func
from superneuroabm.step_functions.soma.lif_soma_adaptive_thr import lif_soma_adaptive_thr_step_func
from superneuroabm.step_functions.soma.hg_lif import hg_lif_soma_step_func
from superneuroabm.step_functions.synapse.single_exp import synapse_single_exp_step_func
from superneuroabm.step_functions.synapse.weighted_synapse import weighted_synapse_step_func
from superneuroabm.util import load_component_configurations
import importlib.util
import sys
from mpi4py import MPI

CURRENT_DIR_ABSPATH = Path(__file__).resolve().parent


def _default_soma_breeds():
    return {
        "izh_soma": (izh_soma_step_func, CURRENT_DIR_ABSPATH / "step_functions" / "soma" / "izh.py"),
        "lif_soma": (lif_soma_step_func, CURRENT_DIR_ABSPATH / "step_functions" / "soma" / "lif.py"),
        "lif_soma_adaptive_thr": (lif_soma_adaptive_thr_step_func, CURRENT_DIR_ABSPATH / "step_functions" / "soma" / "lif_soma_adaptive_thr.py"),
        "hg_lif_soma": (hg_lif_soma_step_func, CURRENT_DIR_ABSPATH / "step_functions" / "soma" / "hg_lif.py"),
    }


def _default_synapse_breeds():
    return {
        "single_exp_synapse": (synapse_single_exp_step_func, CURRENT_DIR_ABSPATH / "step_functions" / "synapse" / "single_exp.py"),
        "weighted_synapse": (weighted_synapse_step_func, CURRENT_DIR_ABSPATH / "step_functions" / "synapse" / "weighted_synapse.py"),
    }


def _default_learning_rules():
    return {
        0: {
            "func_name": "exp_pair_wise_stdp",
            "import_line": "from superneuroabm.step_functions.synapse.stdp.exp_pair_wise_stdp import exp_pair_wise_stdp",
        },
        1: {
            "func_name": "exp_pair_wise_stdp_quantized",
            "import_line": "from superneuroabm.step_functions.synapse.stdp.exp_pair_wise_stdp_quantized import exp_pair_wise_stdp_quantized",
        },
        2: {
            "func_name": "exp_pair_wise_stdp_bounded",
            "import_line": "from superneuroabm.step_functions.synapse.stdp.exp_pair_wise_stdp_bounded import exp_pair_wise_stdp_bounded",
        },
        3: {
            "func_name": "exp_pair_wise_stdp_memristive",
            "import_line": "from superneuroabm.step_functions.synapse.stdp.exp_pair_wise_stdp_memristive import *",
        },
    }


def _compute_max_property_sizes(configurations: dict) -> dict:
    """Return {property_name: max_length} across all component classes, breeds, and configs."""
    property_maxes = {}
    for component_class in configurations:
        for breed in configurations[component_class]:
            for config_name in configurations[component_class][breed]:
                config = configurations[component_class][breed][config_name]
                for prop_type, prop_dict in config.items():
                    if isinstance(prop_dict, dict):
                        property_maxes[prop_type] = max(
                            property_maxes.get(prop_type, 0), len(prop_dict)
                        )
    return property_maxes


class NeuromorphicModel(Model):
    def __init__(
        self,
        soma_breed_info=None,
        synapse_breed_info=None,
        learning_rule_info=None,
        user_config=None,
        enable_internal_state_tracking: bool = True,
    ) -> None:
        """
        Creates an SNN Model and provides methods to create, simulate,
        and monitor soma and synapses.

        :param use_gpu: True if the system supports CUDA GPU
            acceleration.
        :param soma_breed_info: Dict of breed name to
            (step_func, step_func_path) tuple. If specified, will override
            the default soma breeds.
        :param learning_rule_info: Dict of rule id to dict with
            "func_name" and "import_line" keys. If specified, will
            override the default learning rules.
        :param enable_internal_state_tracking: If True, tracks and stores
            internal states history for all agents during simulation.
            If False, disables tracking to reduce memory usage and improve
            performance. Default is True for backward compatibility.
        """
        super().__init__(space=NetworkSpace(ordered=True))

        if soma_breed_info is None:
            soma_breed_info = _default_soma_breeds()
        if synapse_breed_info is None:
            synapse_breed_info = _default_synapse_breeds()

        self.enable_internal_state_tracking = enable_internal_state_tracking
        self._config_list_cache = {}

        self.register_global_property("dt", 1e-3)  # Time step (100 μs)
        self.register_global_property("I_bias", 0)  # No bias current
        self.register_global_property("seed", int(np.random.randint(0, 2**31)))

        # Load and hold configurations (needed before property dicts are built)
        self.agentid2config = {}
        if user_config is not None:
            self._component_configurations = load_component_configurations(user_config)
        else:
            self._component_configurations = load_component_configurations()

        max_sizes = _compute_max_property_sizes(self._component_configurations)

        # Separate learning rule configs before building property dicts
        self._learning_rule_configurations = self._component_configurations.pop("learning_rule", {})

        # Track which learning rule each synapse uses: agent_id -> (rule_breed, rule_config) or None
        self.agentid2learning_rule = {}

        # Soma properties: (default_value, neighbor_visible)
        # neighbor_visible=True means the property is sent to neighbors during MPI sync
        # Only output_spikes_tensor is read by neighbors (synapses read soma spikes)
        soma_properties = {
            "hyperparameters": ([0.0] * max_sizes.get("hyperparameters", 0), False),
            "learning_hyperparameters": (
                [0.0] * max_sizes.get("learning_hyperparameters", 0), False
            ),
            "internal_state": ([0.0] * max_sizes.get("internal_state", 0), False),
            "internal_learning_state": (
                [0.0] * max_sizes.get("internal_learning_state", 0), False
            ),
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
                [0.0] * max_sizes.get("hyperparameters", 0), False
            ),
            "learning_hyperparameters": (
                [0.0] * max_sizes.get("learning_hyperparameters", 0), False
            ),
            "internal_state": (
                [0.0] * max_sizes.get("internal_state", 0), True
            ),  # NEIGHBOR-VISIBLE: somas read Isyn
            "internal_learning_state": (
                [0.0] * max_sizes.get("internal_learning_state", 0), False
            ),
            "synapse_delay_reg": ([], False),  # Synapse delay
            "input_spikes_tensor": ([], False),  # input spikes tensor
            "output_spikes_tensor": ([], False),
            "internal_states_buffer": ([], False),
            "internal_learning_states_buffer": ([], False),  # learning states buffer
        }
        self._synapse_ids = []
        self._soma_ids = []
        self._soma_reset_states = {}

        # Store property definitions for use by registration API
        self._soma_properties = soma_properties
        self._soma_no_double_buffer = list(soma_properties.keys())

        self._soma_breeds: Dict[str, Breed] = {}
        for breed_name, (step_func, step_func_path) in soma_breed_info.items():
            soma_breed = self._make_soma_breed(breed_name, step_func, step_func_path)
            self.register_breed(soma_breed)
            self._soma_breeds[breed_name] = soma_breed

        # Store property definitions for use by registration API
        self._synapse_properties = synapse_properties
        self._synapse_no_double_buffer = list(synapse_properties.keys())

        self._synapse_breeds: Dict[str, Breed] = {}
        for breed_name, (step_func, step_func_path) in synapse_breed_info.items():
            synapse_breed = self._make_synapse_breed(breed_name, step_func, step_func_path)
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

        # Learning rule registry
        if learning_rule_info is None:
            learning_rule_info = _default_learning_rules()
        self._learning_rules = learning_rule_info
        self._learning_rule_names = {r["func_name"]: rid for rid, r in self._learning_rules.items()}
        self._next_learning_rule_id = len(self._learning_rules)
        self._setup_called = False

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

        # Diff synapse/soma config properties (hyperparameters, internal_state)
        for property_name in config:
            config_property_key_values = config.get(property_name, {})
            current_property_key_values = self.get_agent_property_value(
                id=agent_id, property_name=property_name
            )
            overrides[property_name] = {
                k: current_property_key_values[i]
                for i, (k, v) in enumerate(config_property_key_values.items())
                if v != current_property_key_values[i]
            }

        # For synapses, also diff learning rule properties
        lr_info = self.agentid2learning_rule.get(agent_id)
        if lr_info is not None:
            lr_breed, lr_config_name = lr_info
            lr_config = self._learning_rule_configurations[lr_breed][lr_config_name]
            for property_name in lr_config:
                config_property_key_values = lr_config.get(property_name, {})
                current_property_key_values = self.get_agent_property_value(
                    id=agent_id, property_name=property_name
                )
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

    def _make_soma_breed(self, name: str, step_func, step_func_path: Path) -> Breed:
        breed = Breed(name)
        for prop_name, (default_val, neighbor_visible) in self._soma_properties.items():
            breed.register_property(prop_name, default_val, neighbor_visible=neighbor_visible)
        breed.register_step_func(
            step_func=step_func,
            module_fpath=step_func_path,
            priority=0,
            no_double_buffer=self._soma_no_double_buffer,
        )
        return breed

    def _make_synapse_breed(self, name: str, step_func, step_func_path: Path) -> Breed:
        breed = Breed(name)
        for prop_name, (default_val, neighbor_visible) in self._synapse_properties.items():
            breed.register_property(prop_name, default_val, neighbor_visible=neighbor_visible)
        breed.register_step_func(
            step_func=step_func,
            module_fpath=step_func_path,
            priority=100,
            no_double_buffer=self._synapse_no_double_buffer,
        )
        return breed

    def register_soma_type(self, name: str, step_func, step_func_path: Path) -> None:
        """Register a custom soma type with its step function.

        Must be called before setup().

        :param name: Unique name for the soma type.
        :param step_func: The step function callable.
        :param step_func_path: Path to the module containing the step function.
        """
        if self._setup_called:
            raise RuntimeError(
                "Cannot register soma type after setup() has been called."
            )
        if name in self._soma_breeds:
            raise ValueError(f"Soma type '{name}' is already registered.")

        soma_breed = self._make_soma_breed(name, step_func, step_func_path)
        self.register_breed(soma_breed)
        self._soma_breeds[name] = soma_breed

    def register_synapse_type(self, name: str, step_func, step_func_path: Path) -> None:
        """Register a custom synapse type with its step function.

        Must be called before setup(). The learning rule selector is
        auto-attached to all synapse breeds during setup().

        :param name: Unique name for the synapse type.
        :param step_func: The step function callable.
        :param step_func_path: Path to the module containing the step function.
        """
        if self._setup_called:
            raise RuntimeError(
                "Cannot register synapse type after setup() has been called."
            )
        if name in self._synapse_breeds:
            raise ValueError(f"Synapse type '{name}' is already registered.")

        synapse_breed = self._make_synapse_breed(name, step_func, step_func_path)
        self.register_breed(synapse_breed)
        self._synapse_breeds[name] = synapse_breed

    def register_learning_rule(
        self, step_func, step_func_path: Path
    ) -> int:
        """Register a custom learning rule; returns auto-assigned integer ID.

        Must be called before setup().

        :param step_func: The learning rule step function.
        :param step_func_path: Path to the module containing the step function.
        :return: The auto-assigned integer ID for the learning rule.
        """
        if self._setup_called:
            raise RuntimeError(
                "Cannot register learning rule after setup() has been called."
            )
        if step_func.__name__ in self._learning_rule_names:
            raise ValueError(f"Learning rule '{step_func.__name__}' is already registered.")

        rule_id = self._next_learning_rule_id
        self._next_learning_rule_id += 1

        step_func_path = Path(step_func_path).resolve()
        func_name = step_func.__name__
        module_stem = step_func_path.stem
        sys_path_entry = str(step_func_path.parent)

        self._learning_rules[rule_id] = {
            "func_name": func_name,
            "import_line": f"from {module_stem} import {func_name}",
            "sys_path_entry": sys_path_entry,
        }
        self._learning_rule_names[func_name] = rule_id

        return rule_id

    def _generate_learning_rule_selector(self):
        """Generate a new learning_rule_selector.py with all registered rules.

        Writes to superneuroabm/_generated/learning_rule_selector.py,
        imports the module, and returns (func, path).
        """
        CALL_ARGS = (
            "            tick, agent_index, globals, agent_ids, breeds, locations,\n"
            "            synapse_params, learning_params, internal_state,\n"
            "            internal_learning_state, synapse_history, input_spikes_tensor,\n"
            "            output_spikes_tensor, internal_states_buffer,\n"
            "            internal_learning_states_buffer,\n"
        )

        # Collect sys.path entries and import lines
        sys_path_lines = []
        import_lines = []
        for rule_id in sorted(self._learning_rules.keys()):
            rule = self._learning_rules[rule_id]
            entry = rule.get("sys_path_entry")
            if entry:
                line = f"sys.path.insert(0, {entry!r})"
                if line not in sys_path_lines:
                    sys_path_lines.append(line)
            import_lines.append(rule["import_line"])

        # Build if/elif branches
        branches = []
        branches.append("    stdpType = learning_params[agent_index][0]")
        branches.append("    if stdpType == -1:")
        branches.append("        pass")
        for rule_id in sorted(self._learning_rules.keys()):
            rule = self._learning_rules[rule_id]
            func_name = rule["func_name"]
            branches.append(f"    elif stdpType == {rule_id}:")
            branches.append(f"        {func_name}(")
            branches.append(CALL_ARGS + "        )")

        # Assemble source
        lines = ["import sys", "from cupyx import jit", ""]
        lines.append(
            "from superneuroabm.step_functions.synapse.util import get_soma_spike"
        )
        lines.append("")
        for line in sys_path_lines:
            lines.append(line)
        if sys_path_lines:
            lines.append("")
        for line in import_lines:
            lines.append(line)
        lines.append("")
        lines.append("")
        lines.append('@jit.rawkernel(device="cuda")')
        lines.append("def learning_rule_selector(")
        lines.append(
            "    tick, agent_index, globals, agent_ids, breeds, locations,"
        )
        lines.append("    synapse_params, learning_params, internal_state,")
        lines.append(
            "    internal_learning_state, synapse_history, input_spikes_tensor,"
        )
        lines.append("    output_spikes_tensor, internal_states_buffer,")
        lines.append("    internal_learning_states_buffer,")
        lines.append("):")
        lines.extend(branches)
        lines.append("")

        source = "\n".join(lines)

        # Only rank 0 writes to avoid race conditions on shared filesystems
        gen_dir = CURRENT_DIR_ABSPATH / "_generated"
        gen_file = gen_dir / "learning_rule_selector.py"
        comm = MPI.COMM_WORLD
        if comm.Get_rank() == 0:
            gen_dir.mkdir(exist_ok=True)
            (gen_dir / "__init__.py").touch()
            gen_file.write_text(source)
        comm.Barrier()

        # Evict stale module and invalidate caches before re-importing
        module_name = "superneuroabm._generated.learning_rule_selector"
        sys.modules.pop(module_name, None)
        importlib.invalidate_caches()

        # Import via importlib
        spec = importlib.util.spec_from_file_location(
            module_name,
            str(gen_file),
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module  # Register so inspect.getmodule() works
        spec.loader.exec_module(module)

        return (module.learning_rule_selector, gen_file)

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
        Iterates only local agents and writes directly to data tensors,
        avoiding MPI broadcasts from get/set_agent_property_value.

        :param retain_parameters: If True, keeps current learned parameters.
            If False, resets parameters to their default values.
        """
        af = self._agent_factory
        try:
            rank = MPI.COMM_WORLD.Get_rank()
        except Exception:
            rank = 0
        local_agent_map = af._rank2agentid2agentidx.get(rank, {})
        data = af._property_name_2_agent_data_tensor
        synapse_set = set(self._synapse_ids)

        for agent_id, idx in local_agent_map.items():
            if agent_id in synapse_set:
                # Reset synapse
                data["input_spikes_tensor"][idx] = [-1, 0.0]
                synapse_delay = int(self._synapse2defaultparameters[agent_id][1])
                data["synapse_delay_reg"][idx] = [0] * synapse_delay
                data["internal_state"][idx] = self._synapse2defaultinternalstate[agent_id].copy()
                data["internal_learning_state"][idx] = self._synapse2defaultinternallearningstate[agent_id].copy()
                if not retain_parameters:
                    data["hyperparameters"][idx] = self._synapse2defaultparameters[agent_id].copy()
                    data["learning_hyperparameters"][idx] = self._synapse2defaultlearningparameters[agent_id].copy()
            else:
                # Reset soma
                if agent_id in self._soma_reset_states:
                    data["internal_state"][idx] = self._soma_reset_states[agent_id].copy()
                    data["output_spikes_tensor"][idx] = [0.0, 0.0]

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
        # self._agent_factory._prev_agent_data.clear()
        
    def setup(
        self,
        use_gpu: bool = True,
    ) -> None:
        """
        One-time heavy initialization: code gen, JIT, priority analysis.
        Always resets to default state. Call once before simulation loop.
        """
        self._setup_called = True

        # Always generate selector from registry, attach to all synapse breeds
        new_func, new_path = self._generate_learning_rule_selector()
        for breed in self._synapse_breeds.values():
            # Priority = one after the last synapse step func
            max_priority = max(breed.step_funcs.keys())
            breed.register_step_func(
                step_func=new_func,
                module_fpath=new_path,
                priority=max_priority + 1,
                no_double_buffer=self._synapse_no_double_buffer,
            )

        # Skip redundant reset on first setup — agents already have defaults from creation.
        # Only needed on subsequent setup() calls (after simulate() has run).
        if getattr(self, '_has_simulated', False):
            self._reset_agents(retain_parameters=False)

        self._recorded_spikes = []
        self._spike_record_gpu = None
        self._spike_record_count_gpu = None
        self._spike_mask_gpu = None  # rebuild mask on next prepare
        self._spikes_need_gather = False
        super().setup(use_gpu=use_gpu, skip_priority_barriers={100})

        if not self.enable_internal_state_tracking:
            af = self._agent_factory
            rank = MPI.COMM_WORLD.Get_rank()
            local_agent_map = af._rank2agentid2agentidx.get(rank, {})
            data = af._property_name_2_agent_data_tensor
            for agent_id, idx in local_agent_map.items():
                state = data["internal_state"][idx]
                data["internal_states_buffer"][idx] = [state[::]]
                ls = data["internal_learning_state"][idx]
                data["internal_learning_states_buffer"][idx] = [ls[::]]

    def simulate(
        self, ticks: int, update_data_ticks: int = 1  # , num_cpu_proc: int = 4
    ) -> None:
        """
        Override of superneuroabm.core.model mainly to register an
        AgentDataCollector to monitor marked output somas.

        """
        import time
        t_construction_start = time.time()

        # Direct data tensor access — bypasses MPI broadcasts entirely.
        # Each rank only touches its own local agents.
        af = self._agent_factory
        rank = MPI.COMM_WORLD.Get_rank()
        local_agent_map = af._rank2agentid2agentidx.get(rank, {})
        data = af._property_name_2_agent_data_tensor
        soma_set = set(self._soma_ids)

        if self.enable_internal_state_tracking:
            for agent_id, idx in local_agent_map.items():
                state = data["internal_state"][idx]
                data["internal_states_buffer"][idx] = [state[::] for _ in range(ticks)]

                ls = data["internal_learning_state"][idx]
                data["internal_learning_states_buffer"][idx] = [ls[::] for _ in range(ticks)]

                if agent_id not in soma_set:
                    spikes = data["input_spikes_tensor"][idx]
                    if len(spikes) > 2:
                        pairs = [(spikes[i], spikes[i + 1]) for i in range(2, len(spikes), 2)]
                        pairs.sort(key=lambda p: p[0])
                        sorted_spikes = [spikes[0], spikes[1]]
                        for t, v in pairs:
                            sorted_spikes.append(t)
                            sorted_spikes.append(v)
                        data["input_spikes_tensor"][idx] = sorted_spikes
        else:
            for agent_id, idx in local_agent_map.items():
                if agent_id not in soma_set:
                    spikes = data["input_spikes_tensor"][idx]
                    if len(spikes) > 2:
                        pairs = [(spikes[i], spikes[i + 1]) for i in range(2, len(spikes), 2)]
                        pairs.sort(key=lambda p: p[0])
                        sorted_spikes = [spikes[0], spikes[1]]
                        for t, v in pairs:
                            sorted_spikes.append(t)
                            sorted_spikes.append(v)
                        data["input_spikes_tensor"][idx] = sorted_spikes
        t_construction_end = time.time()
        self._construction_time = t_construction_end - t_construction_start

        self._recorded_spikes = []
        self._spikes_need_gather = False

        t_sim_start = time.time()
        super().simulate(ticks, update_data_ticks)  # , num_cpu_proc)
        self._simulation_time = time.time() - t_sim_start

        if self._verbose_timing and MPI.COMM_WORLD.Get_rank() == 0:
            print(f"[TIMING] Construction (pre-sim buffer alloc): {self._construction_time:.4f}s")
            print(f"[TIMING] Simulation (state propagation): {self._simulation_time:.4f}s")

        self._has_simulated = True

        if MPI.COMM_WORLD.Get_size() > 1:
            self._spikes_need_gather = True

    def create_soma(
        self,
        breed: str,
        config_name: str,
        overrides: Dict[str, Dict[str, float]] = None,
        tags: Set[str] = None,
    ) -> int:
        """
        Creates a soma agent.

        :param overrides: Dict keyed by property type, e.g.
            {"hyperparameters": {"R": 1.1e6}, "internal_state": {"v": -55.0}}
        :return: SAGESim agent id of soma
        """
        tags = tags if tags else set()
        overrides = overrides or {}

        # Cached config list construction — avoids copy.deepcopy per agent
        cache_key = ("soma", breed, config_name)
        if cache_key not in self._config_list_cache:
            config = self._component_configurations["soma"][breed][config_name]
            hp_keys = list(config["hyperparameters"].keys())
            hp_vals = [float(v) for v in config["hyperparameters"].values()]
            is_keys = list(config["internal_state"].keys())
            is_vals = [float(v) for v in config["internal_state"].values()]
            self._config_list_cache[cache_key] = (hp_keys, hp_vals, is_keys, is_vals)
        hp_keys, hp_defaults, is_keys, is_defaults = self._config_list_cache[cache_key]

        hyperparameters = hp_defaults[:]
        for k, v in overrides.get("hyperparameters", {}).items():
            hyperparameters[hp_keys.index(k)] = float(v)

        default_internal_state = is_defaults[:]
        for k, v in overrides.get("internal_state", {}).items():
            default_internal_state[is_keys.index(k)] = float(v)

        soma_id = super().create_agent_of_breed(
            breed=self._soma_breeds[breed],
            hyperparameters=hyperparameters,
            internal_state=default_internal_state,
            output_spikes_tensor=[0.0, 0.0],
        )

        self._soma_ids.append(soma_id)
        self._soma_reset_states[soma_id] = default_internal_state

        self.agentid2config[soma_id] = config_name

        tags.update({"soma", breed})
        for tag in tags:
            self.tag2component[tag].add(soma_id)
        return soma_id

    def create_soma_at_index(
        self,
        agent_id: int,
        local_idx: int,
        breed: str,
        config_name: str,
        overrides: Dict[str, Dict[str, float]] = None,
        tags: Set[str] = None,
    ) -> int:
        """
        Creates a soma at a pre-allocated index from bulk_register_agents.

        :param agent_id: Global agent ID (pre-assigned)
        :param local_idx: Local index in property tensors
        :param overrides: Dict keyed by property type, e.g.
            {"hyperparameters": {"R": 1.1e6}, "internal_state": {"v": -55.0}}
        :return: agent_id
        """
        tags = tags if tags else set()
        overrides = overrides or {}

        # Cached config list construction (same as create_soma)
        cache_key = ("soma", breed, config_name)
        if cache_key not in self._config_list_cache:
            config = self._component_configurations["soma"][breed][config_name]
            hp_keys = list(config["hyperparameters"].keys())
            hp_vals = [float(v) for v in config["hyperparameters"].values()]
            is_keys = list(config["internal_state"].keys())
            is_vals = [float(v) for v in config["internal_state"].values()]
            self._config_list_cache[cache_key] = (hp_keys, hp_vals, is_keys, is_vals)
        hp_keys, hp_defaults, is_keys, is_defaults = self._config_list_cache[cache_key]

        hyperparameters = hp_defaults[:]
        for k, v in overrides.get("hyperparameters", {}).items():
            hyperparameters[hp_keys.index(k)] = float(v)

        default_internal_state = is_defaults[:]
        for k, v in overrides.get("internal_state", {}).items():
            default_internal_state[is_keys.index(k)] = float(v)

        # Use shared location reference from space
        location_ref = self.get_space()._locations[agent_id]

        self._agent_factory.create_agent_at_index(
            agent_id, local_idx, self._soma_breeds[breed],
            hyperparameters=hyperparameters,
            internal_state=default_internal_state,
            output_spikes_tensor=[0.0, 0.0],
            locations=location_ref,
        )

        self._soma_ids.append(agent_id)
        self._soma_reset_states[agent_id] = default_internal_state
        self.agentid2config[agent_id] = config_name

        tags.update({"soma", breed})
        for tag in tags:
            self.tag2component[tag].add(agent_id)
        return agent_id

    def create_synapse_at_index(
        self,
        agent_id: int,
        local_idx: int,
        breed: str,
        pre_soma_id: int,
        post_soma_id: int,
        config_name: str,
        learning_rule: str = None,
        learning_rule_config: str = "default",
        overrides: Dict[str, Dict[str, float]] = None,
        tags: Set[str] = None,
    ) -> int:
        """
        Creates a synapse at a pre-allocated index from bulk_register_agents.

        :param agent_id: Global agent ID (pre-assigned)
        :param local_idx: Local index in property tensors
        :param learning_rule: Learning rule breed name (e.g. "exp_pair_wise_stdp"), or None for no learning.
        :param learning_rule_config: Config name within the learning rule breed (default: "default").
        :param overrides: Dict keyed by property type, e.g.
            {"hyperparameters": {"weight": 0.5}, "learning_hyperparameters": {"a_exp_pre": 0.01}}
        :return: agent_id
        """
        tags = tags if tags else set()
        overrides = overrides or {}

        # Synapse config cache (hp + is only — no learning params in synapse config)
        cache_key = ("synapse", breed, config_name)
        if cache_key not in self._config_list_cache:
            config = self._component_configurations["synapse"][breed][config_name]
            hp_keys = list(config["hyperparameters"].keys())
            hp_vals = [float(v) for v in config["hyperparameters"].values()]
            is_keys = list(config["internal_state"].keys())
            is_vals = [float(v) for v in config["internal_state"].values()]
            self._config_list_cache[cache_key] = (hp_keys, hp_vals, is_keys, is_vals)
        hp_keys, hp_defaults, is_keys, is_defaults = self._config_list_cache[cache_key]

        # Learning rule config (separate cache)
        if learning_rule is not None:
            lr_cache_key = ("learning_rule", learning_rule, learning_rule_config)
            if lr_cache_key not in self._config_list_cache:
                lr_config = self._learning_rule_configurations[learning_rule][learning_rule_config]
                lhp_keys = list(lr_config["learning_hyperparameters"].keys())
                lhp_vals = [float(v) for v in lr_config["learning_hyperparameters"].values()]
                ils_keys = list(lr_config.get("internal_learning_state", {}).keys())
                ils_vals = [float(v) for v in lr_config.get("internal_learning_state", {}).values()]
                self._config_list_cache[lr_cache_key] = (lhp_keys, lhp_vals, ils_keys, ils_vals)
            lhp_keys, lhp_defaults, ils_keys, ils_defaults = self._config_list_cache[lr_cache_key]
        else:
            lhp_keys, lhp_defaults = ["stdp_type"], [-1.0]
            ils_keys, ils_defaults = [], []

        hyperparameters = hp_defaults[:]
        for k, v in overrides.get("hyperparameters", {}).items():
            hyperparameters[hp_keys.index(k)] = float(v)

        default_internal_state = is_defaults[:]
        for k, v in overrides.get("internal_state", {}).items():
            default_internal_state[is_keys.index(k)] = float(v)

        learning_hyperparameters = lhp_defaults[:]
        for k, v in overrides.get("learning_hyperparameters", {}).items():
            learning_hyperparameters[lhp_keys.index(k)] = float(v)

        default_internal_learning_state = ils_defaults[:]
        for k, v in overrides.get("internal_learning_state", {}).items():
            default_internal_learning_state[ils_keys.index(k)] = float(v)

        synaptic_delay = int(hyperparameters[1])
        delay_reg = [0 for _ in range(synaptic_delay)]

        # Use shared location reference from space
        location_ref = self.get_space()._locations[agent_id]

        self._agent_factory.create_agent_at_index(
            agent_id, local_idx, self._synapse_breeds[breed],
            hyperparameters=hyperparameters,
            learning_hyperparameters=learning_hyperparameters,
            internal_state=default_internal_state,
            internal_learning_state=default_internal_learning_state,
            synapse_delay_reg=delay_reg,
            input_spikes_tensor=[-1, 0.0],
            locations=location_ref,
        )

        self._synapse2defaultparameters[agent_id] = hyperparameters
        self._synapse2defaultlearningparameters[agent_id] = learning_hyperparameters
        self._synapse2defaultinternalstate[agent_id] = default_internal_state
        self._synapse2defaultinternallearningstate[agent_id] = default_internal_learning_state
        self._synapse_ids.append(agent_id)
        self.agentid2learning_rule[agent_id] = (learning_rule, learning_rule_config) if learning_rule else None

        network_space = self.get_space()

        if pre_soma_id != -1:
            network_space.connect_agents(agent_id, pre_soma_id, directed=True)
            self.soma2synapse_map[pre_soma_id]["post"].add(agent_id)
            self.synapse2soma_map[agent_id]["pre"] = pre_soma_id
        else:
            network_space.get_location(agent_id).append(-1)
            self.synapse2soma_map[agent_id]["pre"] = -1
            tags.add("input_synapse")

        if post_soma_id != -1:
            network_space.connect_agents(agent_id, post_soma_id, directed=True)
            network_space.connect_agents(post_soma_id, agent_id, directed=True)
            self.synapse2soma_map[agent_id]["post"] = post_soma_id
            self.soma2synapse_map[post_soma_id]["pre"].add(agent_id)
        else:
            self.synapse2soma_map[agent_id]["post"] = -1
            network_space.get_location(agent_id).append(-1)

        self.agentid2config[agent_id] = config_name
        tags.update({"synapse", breed})
        for tag in tags:
            self.tag2component[tag].add(agent_id)
        return agent_id

    def create_synapse(
        self,
        breed: str,
        pre_soma_id: int,
        post_soma_id: int,
        config_name: str,
        learning_rule: str = None,
        learning_rule_config: str = "default",
        overrides: Dict[str, Dict[str, float]] = None,
        tags: Set[str] = None,
    ) -> int:
        """
        Creates and adds a Synapse agent.

        Parameters:
            breed (str): Synapse breed name (e.g., 'single_exp_synapse').
            pre_soma_id (int): Presynaptic soma agent ID (or -1 for external input).
            post_soma_id (int): Postsynaptic soma agent ID (or -1 for external output).
            config_name (str): Name of the configuration to use for this synapse.
            learning_rule (str, optional): Learning rule breed name (e.g. "exp_pair_wise_stdp"), or None for no learning.
            learning_rule_config (str): Config name within the learning rule breed (default: "default").
            overrides (dict, optional): Dict keyed by property type, e.g.
                {"hyperparameters": {"weight": 0.5}, "learning_hyperparameters": {"a_exp_pre": 0.01}}
            tags (set of str, optional): Tags to associate with this synapse.

        Returns:
            int: SAGESim agent ID of the created synapse.
        """
        tags = tags if tags else set()
        overrides = overrides or {}

        # Synapse config cache (hp + is only — no learning params in synapse config)
        cache_key = ("synapse", breed, config_name)
        if cache_key not in self._config_list_cache:
            config = self._component_configurations["synapse"][breed][config_name]
            hp_keys = list(config["hyperparameters"].keys())
            hp_vals = [float(v) for v in config["hyperparameters"].values()]
            is_keys = list(config["internal_state"].keys())
            is_vals = [float(v) for v in config["internal_state"].values()]
            self._config_list_cache[cache_key] = (hp_keys, hp_vals, is_keys, is_vals)
        hp_keys, hp_defaults, is_keys, is_defaults = self._config_list_cache[cache_key]

        # Learning rule config (separate cache)
        if learning_rule is not None:
            lr_cache_key = ("learning_rule", learning_rule, learning_rule_config)
            if lr_cache_key not in self._config_list_cache:
                lr_config = self._learning_rule_configurations[learning_rule][learning_rule_config]
                lhp_keys = list(lr_config["learning_hyperparameters"].keys())
                lhp_vals = [float(v) for v in lr_config["learning_hyperparameters"].values()]
                ils_keys = list(lr_config.get("internal_learning_state", {}).keys())
                ils_vals = [float(v) for v in lr_config.get("internal_learning_state", {}).values()]
                self._config_list_cache[lr_cache_key] = (lhp_keys, lhp_vals, ils_keys, ils_vals)
            lhp_keys, lhp_defaults, ils_keys, ils_defaults = self._config_list_cache[lr_cache_key]
        else:
            lhp_keys, lhp_defaults = ["stdp_type"], [-1.0]
            ils_keys, ils_defaults = [], []

        hyperparameters = hp_defaults[:]
        for k, v in overrides.get("hyperparameters", {}).items():
            hyperparameters[hp_keys.index(k)] = float(v)

        default_internal_state = is_defaults[:]
        for k, v in overrides.get("internal_state", {}).items():
            default_internal_state[is_keys.index(k)] = float(v)

        learning_hyperparameters = lhp_defaults[:]
        for k, v in overrides.get("learning_hyperparameters", {}).items():
            learning_hyperparameters[lhp_keys.index(k)] = float(v)

        default_internal_learning_state = ils_defaults[:]
        for k, v in overrides.get("internal_learning_state", {}).items():
            default_internal_learning_state[ils_keys.index(k)] = float(v)

        synaptic_delay = int(hyperparameters[1])
        delay_reg = [0 for _ in range(synaptic_delay)]
        synapse_id = self.create_agent_of_breed(
            breed=self._synapse_breeds[breed],
            hyperparameters=hyperparameters,
            learning_hyperparameters=learning_hyperparameters,
            internal_state=default_internal_state,
            internal_learning_state=default_internal_learning_state,
            synapse_delay_reg=delay_reg,
            input_spikes_tensor=[-1, 0.0],
        )
        self._synapse2defaultparameters[synapse_id] = hyperparameters
        self._synapse2defaultlearningparameters[synapse_id] = learning_hyperparameters
        self._synapse2defaultinternalstate[synapse_id] = default_internal_state
        self._synapse2defaultinternallearningstate[synapse_id] = (
            default_internal_learning_state
        )
        self._synapse_ids.append(synapse_id)
        self.agentid2learning_rule[synapse_id] = (learning_rule, learning_rule_config) if learning_rule else None

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

