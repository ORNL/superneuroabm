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
        enable_internal_states_tracking: bool = True,
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
        :param enable_internal_states_tracking: If True, tracks and stores
            internal states history for all agents during simulation.
            If False, disables tracking to reduce memory usage and improve
            performance. Default is True for backward compatibility.
        """
        super().__init__(space=NetworkSpace(ordered=True))

        if soma_breed_info is None:
            soma_breed_info = _default_soma_breeds()
        if synapse_breed_info is None:
            synapse_breed_info = _default_synapse_breeds()

        self.enable_internal_states_tracking = enable_internal_states_tracking
        self._config_list_cache = {}

        self.register_global_property("dt", 1e-3)      # Time step (100 μs)
        self.register_global_property("I_bias", 0)     # No bias current

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
            "internal_states": ([0.0] * max_sizes.get("internal_states", 0), False),
            "learning_internal_states": (
                [0.0] * max_sizes.get("learning_internal_states", 0), False
            ),
            "synapse_delay_reg": ([], False),  # Synapse delay
            "input_spikes_tensor": ([], False),  # input spikes tensor
            "output_spikes_tensor": ([], True),  # NEIGHBOR-VISIBLE: synapses read soma spikes
            "internal_states_buffer": ([], False),
            "learning_internal_states_buffer": ([], False),  # learning states buffer
        }
        # Synapse properties: (default_value, neighbor_visible)
        # Only internal_states is read by neighbors (somas read I_synapse from synapses)
        synapse_properties = {
            "hyperparameters": (
                [0.0] * max_sizes.get("hyperparameters", 0), False
            ),
            "learning_hyperparameters": (
                [0.0] * max_sizes.get("learning_hyperparameters", 0), False
            ),
            "internal_states": (
                [0.0] * max_sizes.get("internal_states", 0), True
            ),  # NEIGHBOR-VISIBLE: somas read Isyn
            "learning_internal_states": (
                [0.0] * max_sizes.get("learning_internal_states", 0), False
            ),
            "synapse_delay_reg": ([], False),  # Synapse delay
            "input_spikes_tensor": ([], False),  # input spikes tensor
            "output_spikes_tensor": ([], False),
            "internal_states_buffer": ([], False),
            "learning_internal_states_buffer": ([], False),  # learning states buffer
        }
        self._synapse_ids = set()
        self._soma_ids = set()

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

        self._soma_outgoing_synapses = defaultdict(set)  # soma_id -> set(synapse_ids)
        self.agentid2overrides = {}  # agent_id -> overrides dict

        self._breed_names = list(self._agent_factory._breeds.keys())

        # Learning rule registry
        if learning_rule_info is None:
            learning_rule_info = _default_learning_rules()
        self._learning_rules = learning_rule_info
        self._learning_rule_names = {r["func_name"]: rid for rid, r in self._learning_rules.items()}
        self._next_learning_rule_id = len(self._learning_rules)
        self._setup_called = False
        # Construction-mode lock: a model is built EITHER incrementally via
        # create_soma/create_synapse OR in one shot via load_from_file — never
        # both. load_from_file overwrites the agent factory wholesale, so the two
        # paths cannot be combined. Set True by load_from_file().
        self._built_from_file = False

    def get_agent_config_name(self, agent_id: int) -> Dict[str, any]:
        """
        Returns the configuration of the agent with the given ID.
        """
        return self.agentid2config.get(agent_id, None)

    def get_agent_breed(self, agent_id: int) -> str:
        """
        Returns the breed of the agent with the given ID.
        """
        return self._breed_names[self._agent_factory._agent2breed[agent_id]]

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

    def get_soma_outgoing_synapses(self, soma_id: int) -> Set[int]:
        """
        Returns the set of synapse IDs where this soma is the pre-synaptic source.
        """
        return self._soma_outgoing_synapses.get(soma_id, set())


    def get_agent_config_diff(self, agent_id: int) -> Dict[str, any]:
        """
        Returns the configuration overrides for the agent with the given ID.
        """
        component_class = "soma" if agent_id in self._soma_ids else "synapse"
        breed_name = self.get_agent_breed(agent_id)
        config_name = self.get_agent_config_name(agent_id)
        config = self._component_configurations[component_class][breed_name][
            config_name
        ]
        overrides = {}

        # Diff synapse/soma config properties (hyperparameters, internal_states)
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
        self._breed_names = list(self._agent_factory._breeds.keys())

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
        self._breed_names = list(self._agent_factory._breeds.keys())

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
            "            tick, agent_index, _seed, dt, I_bias,\n"
            "            agent_ids, logical_ids, breeds, locations,\n"
            "            synapse_params, learning_params, internal_states,\n"
            "            learning_internal_states, synapse_history, input_spikes_tensor,\n"
            "            output_spikes_tensor, internal_states_buffer,\n"
            "            learning_internal_states_buffer,\n"
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
            "    tick, agent_index, dt, I_bias, agent_ids, breeds, locations,"
        )
        lines.append("    synapse_params, learning_params, internal_states,")
        lines.append(
            "    learning_internal_states, synapse_history, input_spikes_tensor,"
        )
        lines.append("    output_spikes_tensor, internal_states_buffer,")
        lines.append("    learning_internal_states_buffer,")
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
        Recomputes defaults from (breed, config, overrides) via config cache.

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

        for agent_id, idx in local_agent_map.items():
            if agent_id in self._synapse_ids:
                # Reset synapse — recompute defaults from config
                breed_name = self._breed_names[af._agent2breed[agent_id]]
                config_name = self.agentid2config[agent_id]
                overrides = self.agentid2overrides.get(agent_id, {})
                lr = self.agentid2learning_rule.get(agent_id)
                lr_breed, lr_config = lr if lr else (None, "default")
                _, hp, lhp, is_state, ils = self._get_synapse_properties(
                    breed_name, config_name, overrides, lr_breed, lr_config)

                data["input_spikes_tensor"][idx] = [-1, 0.0]
                data["synapse_delay_reg"][idx] = [0] * int(hp[1])
                data["internal_states"][idx] = is_state
                data["learning_internal_states"][idx] = ils
                if not retain_parameters:
                    data["hyperparameters"][idx] = hp
                    data["learning_hyperparameters"][idx] = lhp
            elif agent_id in self._soma_ids:
                # Reset soma — recompute defaults from config
                breed_name = self._breed_names[af._agent2breed[agent_id]]
                config_name = self.agentid2config[agent_id]
                overrides = self.agentid2overrides.get(agent_id, {})
                hp, is_state = self._get_soma_properties(breed_name, config_name, overrides)
                data["internal_states"][idx] = is_state
                data["output_spikes_tensor"][idx] = [0.0, 0.0]
                if not retain_parameters:
                    data["hyperparameters"][idx] = hp

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
        self._breed_names = list(self._agent_factory._breeds.keys())

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
        self.set_property_neighbor_visible("breed", False)  # no step func reads neighbor breeds
        super().setup(use_gpu=use_gpu, skip_priority_barriers={100})

        if not self.enable_internal_states_tracking:
            af = self._agent_factory
            rank = MPI.COMM_WORLD.Get_rank()
            local_agent_map = af._rank2agentid2agentidx.get(rank, {})
            data = af._property_name_2_agent_data_tensor
            for agent_id, idx in local_agent_map.items():
                state = data["internal_states"][idx]
                data["internal_states_buffer"][idx] = [state[::]]
                ls = data["learning_internal_states"][idx]
                data["learning_internal_states_buffer"][idx] = [ls[::]]

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
        if self.enable_internal_states_tracking:
            for agent_id, idx in local_agent_map.items():
                state = data["internal_states"][idx]
                data["internal_states_buffer"][idx] = [state[::] for _ in range(ticks)]

                ls = data["learning_internal_states"][idx]
                data["learning_internal_states_buffer"][idx] = [ls[::] for _ in range(ticks)]

                if agent_id in self._synapse_ids:
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
                if agent_id in self._synapse_ids:
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
        agent_id: int = None,
    ) -> int:
        """
        Creates a soma agent.

        :param overrides: Dict keyed by property type, e.g.
            {"hyperparameters": {"R": 1.1e6}, "internal_states": {"v": -55.0}}
        :param agent_id: Explicit global agent ID. If provided, uses this ID
            instead of auto-incrementing; must be unique (a collision with an
            existing agent raises). Used for partition-based loading.
        :return: SAGESim agent id of soma
        """
        if self._built_from_file:
            raise RuntimeError(
                "Cannot create_soma() on a model built via load_from_file(). "
                "load_from_file() builds the entire model in one shot and is "
                "mutually exclusive with incremental create_soma/create_synapse — "
                "use one construction path per model."
            )
        overrides = overrides or {}

        # Cached config list construction — avoids copy.deepcopy per agent
        cache_key = ("soma", breed, config_name)
        if cache_key not in self._config_list_cache:
            config = self._component_configurations["soma"][breed][config_name]
            hp_keys = list(config["hyperparameters"].keys())
            hp_vals = [float(v) for v in config["hyperparameters"].values()]
            is_keys = list(config["internal_states"].keys())
            is_vals = [float(v) for v in config["internal_states"].values()]
            self._config_list_cache[cache_key] = (hp_keys, hp_vals, is_keys, is_vals)
        hp_keys, hp_defaults, is_keys, is_defaults = self._config_list_cache[cache_key]

        hyperparameters = hp_defaults[:]
        for k, v in overrides.get("hyperparameters", {}).items():
            hyperparameters[hp_keys.index(k)] = float(v)

        default_internal_states = is_defaults[:]
        for k, v in overrides.get("internal_states", {}).items():
            default_internal_states[is_keys.index(k)] = float(v)

        soma_id = super().create_agent_of_breed(
            breed=self._soma_breeds[breed],
            agent_id=agent_id,
            hyperparameters=hyperparameters,
            internal_states=default_internal_states,
            output_spikes_tensor=[0.0, 0.0],
        )

        self._soma_ids.add(soma_id)
        self.agentid2config[soma_id] = config_name
        self.agentid2overrides[soma_id] = overrides
        return soma_id

    def create_synapse(
        self,
        breed: str,
        pre_soma_id: int,
        post_soma_id: int,
        config_name: str,
        learning_rule: str = None,
        learning_rule_config: str = "default",
        overrides: Dict[str, Dict[str, float]] = None,
        agent_id: int = None,
        skip_post_connection: bool = False,
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
            agent_id (int, optional): Explicit global agent ID; must be unique
                (a collision with an existing agent raises). Used for partition-based loading.
            skip_post_connection (bool): If True, skip the reverse connection from
                post_soma to synapse. Used when post_soma is on a remote rank —
                the remote rank handles its own ghost connection.

        Returns:
            int: SAGESim agent ID of the created synapse.
        """
        if self._built_from_file:
            raise RuntimeError(
                "Cannot create_synapse() on a model built via load_from_file(). "
                "load_from_file() builds the entire model in one shot and is "
                "mutually exclusive with incremental create_soma/create_synapse — "
                "use one construction path per model."
            )
        overrides = overrides or {}

        # Synapse config cache (hp + is only — no learning params in synapse config)
        cache_key = ("synapse", breed, config_name)
        if cache_key not in self._config_list_cache:
            config = self._component_configurations["synapse"][breed][config_name]
            hp_keys = list(config["hyperparameters"].keys())
            hp_vals = [float(v) for v in config["hyperparameters"].values()]
            is_keys = list(config["internal_states"].keys())
            is_vals = [float(v) for v in config["internal_states"].values()]
            self._config_list_cache[cache_key] = (hp_keys, hp_vals, is_keys, is_vals)
        hp_keys, hp_defaults, is_keys, is_defaults = self._config_list_cache[cache_key]

        # Learning rule config (separate cache)
        if learning_rule is not None:
            lr_cache_key = ("learning_rule", learning_rule, learning_rule_config)
            if lr_cache_key not in self._config_list_cache:
                lr_config = self._learning_rule_configurations[learning_rule][learning_rule_config]
                lhp_keys = list(lr_config["learning_hyperparameters"].keys())
                lhp_vals = [float(v) for v in lr_config["learning_hyperparameters"].values()]
                ils_keys = list(lr_config.get("learning_internal_states", {}).keys())
                ils_vals = [float(v) for v in lr_config.get("learning_internal_states", {}).values()]
                self._config_list_cache[lr_cache_key] = (lhp_keys, lhp_vals, ils_keys, ils_vals)
            lhp_keys, lhp_defaults, ils_keys, ils_defaults = self._config_list_cache[lr_cache_key]
        else:
            lhp_keys, lhp_defaults = ["stdp_type"], [-1.0]
            ils_keys, ils_defaults = [], []

        hyperparameters = hp_defaults[:]
        for k, v in overrides.get("hyperparameters", {}).items():
            hyperparameters[hp_keys.index(k)] = float(v)

        default_internal_states = is_defaults[:]
        for k, v in overrides.get("internal_states", {}).items():
            default_internal_states[is_keys.index(k)] = float(v)

        learning_hyperparameters = lhp_defaults[:]
        for k, v in overrides.get("learning_hyperparameters", {}).items():
            learning_hyperparameters[lhp_keys.index(k)] = float(v)

        default_learning_internal_states = ils_defaults[:]
        for k, v in overrides.get("learning_internal_states", {}).items():
            default_learning_internal_states[ils_keys.index(k)] = float(v)

        synaptic_delay = int(hyperparameters[1])
        delay_reg = [0 for _ in range(synaptic_delay)]
        synapse_id = self.create_agent_of_breed(
            breed=self._synapse_breeds[breed],
            agent_id=agent_id,
            hyperparameters=hyperparameters,
            learning_hyperparameters=learning_hyperparameters,
            internal_states=default_internal_states,
            learning_internal_states=default_learning_internal_states,
            synapse_delay_reg=delay_reg,
            input_spikes_tensor=[-1, 0.0],
        )

        self._synapse_ids.add(synapse_id)
        self.agentid2config[synapse_id] = config_name
        self.agentid2overrides[synapse_id] = overrides
        self.agentid2learning_rule[synapse_id] = (learning_rule, learning_rule_config) if learning_rule else None

        network_space: NetworkSpace = self.get_space()

        # Connect synapse to somas using SAGESim's API
        # With ordered=True, connections are maintained in insertion order
        # So synapse's locations will be [pre_soma_id, post_soma_id] after we connect them

        # First connection: pre_soma (if exists)
        if pre_soma_id != -1:
            network_space.connect_agents(synapse_id, pre_soma_id, directed=True)
            self._soma_outgoing_synapses[pre_soma_id].add(synapse_id)
        else:
            # For external input, manually add -1 to locations to maintain [pre, post] order
            network_space.get_location(synapse_id).append(-1)

        # Second connection: post_soma (if exists)
        if post_soma_id != -1:
            network_space.connect_agents(synapse_id, post_soma_id, directed=True)
            if not skip_post_connection:
                network_space.connect_agents(post_soma_id, synapse_id, directed=True)  # Bidirectional for STDP
        else:
            # For external output (rare), manually add -1
            network_space.get_location(synapse_id).append(-1)

        return synapse_id

    def _get_soma_properties(self, breed: str, config_name: str, overrides: dict = None):
        """Compute soma property values from config without creating an agent.

        :return: (hyperparameters, internal_states) as lists
        """
        overrides = overrides or {}
        cache_key = ("soma", breed, config_name)
        if cache_key not in self._config_list_cache:
            config = self._component_configurations["soma"][breed][config_name]
            hp_keys = list(config["hyperparameters"].keys())
            hp_vals = [float(v) for v in config["hyperparameters"].values()]
            is_keys = list(config["internal_states"].keys())
            is_vals = [float(v) for v in config["internal_states"].values()]
            self._config_list_cache[cache_key] = (hp_keys, hp_vals, is_keys, is_vals)
        hp_keys, hp_defaults, is_keys, is_defaults = self._config_list_cache[cache_key]

        hp = hp_defaults[:]
        for k, v in overrides.get("hyperparameters", {}).items():
            hp[hp_keys.index(k)] = float(v)
        is_state = is_defaults[:]
        for k, v in overrides.get("internal_states", {}).items():
            is_state[is_keys.index(k)] = float(v)
        return hp, is_state

    def _get_synapse_properties(self, breed: str, config_name: str,
                                overrides: dict = None, learning_rule: str = None,
                                learning_rule_config: str = "default"):
        """Compute synapse property values from config without creating an agent.

        :param overrides: Dict keyed by property type, e.g.
            {"hyperparameters": {"weight": 14.0}, "learning_hyperparameters": {"a_exp_pre": 0.01}}
        :return: (props_dict, hp, lhp, is_state, ils)
        """
        overrides = overrides or {}
        cache_key = ("synapse", breed, config_name)
        if cache_key not in self._config_list_cache:
            config = self._component_configurations["synapse"][breed][config_name]
            hp_keys = list(config["hyperparameters"].keys())
            hp_vals = [float(v) for v in config["hyperparameters"].values()]
            is_keys = list(config["internal_states"].keys())
            is_vals = [float(v) for v in config["internal_states"].values()]
            self._config_list_cache[cache_key] = (hp_keys, hp_vals, is_keys, is_vals)
        hp_keys, hp_defaults, is_keys, is_defaults = self._config_list_cache[cache_key]

        if learning_rule is not None:
            lr_cache_key = ("learning_rule", learning_rule, learning_rule_config)
            if lr_cache_key not in self._config_list_cache:
                lr_config = self._learning_rule_configurations[learning_rule][learning_rule_config]
                lhp_keys = list(lr_config["learning_hyperparameters"].keys())
                lhp_vals = [float(v) for v in lr_config["learning_hyperparameters"].values()]
                ils_keys = list(lr_config.get("learning_internal_states", {}).keys())
                ils_vals = [float(v) for v in lr_config.get("learning_internal_states", {}).values()]
                self._config_list_cache[lr_cache_key] = (lhp_keys, lhp_vals, ils_keys, ils_vals)
            lhp_keys, lhp_defaults, ils_keys, ils_defaults = self._config_list_cache[lr_cache_key]
        else:
            lhp_keys, lhp_defaults = ["stdp_type"], [-1.0]
            ils_keys, ils_defaults = [], []

        hp = hp_defaults[:]
        for k, v in overrides.get("hyperparameters", {}).items():
            hp[hp_keys.index(k)] = float(v)
        is_state = is_defaults[:]
        for k, v in overrides.get("internal_states", {}).items():
            is_state[is_keys.index(k)] = float(v)
        lhp = lhp_defaults[:]
        for k, v in overrides.get("learning_hyperparameters", {}).items():
            lhp[lhp_keys.index(k)] = float(v)
        ils = ils_defaults[:]
        for k, v in overrides.get("learning_internal_states", {}).items():
            ils[ils_keys.index(k)] = float(v)

        synaptic_delay = int(hp[1])
        delay_reg = [0 for _ in range(synaptic_delay)]

        return {
            'hyperparameters': hp,
            'learning_hyperparameters': lhp,
            'internal_states': is_state,
            'learning_internal_states': ils,
            'synapse_delay_reg': delay_reg,
            'input_spikes_tensor': [-1, 0.0],
        }, hp, lhp, is_state, ils

    @staticmethod
    def _normalize_snn_partition(data: dict) -> dict:
        """Map the SNN-native file schema to the internal representation.

        Input (canonical): {'somas': [...], 'synapses': [...], 'remote_ranks': {...}}
          - soma:    {'id', 'breed', 'config', 'overrides'}
          - synapse: {'id', 'pre', 'post', 'breed', 'config', 'learning_rule',
                      'learning_rule_config', 'overrides'}  (pre = -1 → input)
        Any legacy 'metadata' key on a soma/synapse is ignored — labels are an
        application concern, not framework state.
        Output (internal): {'nodes': [...], 'edges': [...], 'remote_nodes': {...}}
          where each edge has 'source'/'target'/'synapse_id'.

        Raises on the legacy graph-centric schema (nodes/edges/source/target) so old
        files fail loudly instead of silently dropping unmatched override keys.
        """
        if 'somas' in data or 'synapses' in data:
            edges = []
            for syn in data.get('synapses', []):
                edge = {k: v for k, v in syn.items() if k not in ('id', 'pre', 'post')}
                edge['synapse_id'] = syn['id']
                edge['source'] = syn['pre']
                edge['target'] = syn['post']
                edges.append(edge)
            return {
                'nodes': list(data.get('somas', [])),
                'edges': edges,
                'remote_nodes': data.get('remote_ranks', data.get('remote_nodes', {})),
            }
        if 'nodes' in data or 'edges' in data:
            raise ValueError(
                "Legacy graph-centric network format detected (found "
                f"{sorted(k for k in ('nodes', 'edges') if k in data)}). The schema is now "
                "SNN-native: use 'somas'/'synapses' with per-synapse 'id'/'pre'/'post' and "
                "'remote_ranks'. Regenerate the file with the updated producer "
                "(e.g. build_network_from_data.py)."
            )
        raise ValueError(
            "Unrecognized network file: expected top-level 'somas' and 'synapses' keys."
        )

    @staticmethod
    def _read_partition_file(partition_file: str) -> dict:
        """Read a per-rank partition file, auto-detecting format from extension.

        Normalizes all formats to the same internal dict structure:
            {'nodes': [...], 'edges': [...], 'remote_nodes': {...}}

        Supported formats: .pkl/.pickle, .graphml, .json
        """
        import json as json_mod

        fpath = Path(partition_file)
        ext = fpath.suffix.lower()

        if ext in ('.pkl', '.pickle'):
            import pickle
            with open(partition_file, 'rb') as f:
                data = pickle.load(f)
            return NeuromorphicModel._normalize_snn_partition(data)

        elif ext == '.graphml':
            import networkx as nx
            graph = nx.read_graphml(partition_file)

            def _try_parse(val):
                """Deserialize JSON-string attributes from GraphML."""
                if isinstance(val, str):
                    stripped = val.strip()
                    if (stripped.startswith('{') and stripped.endswith('}')) or \
                       (stripped.startswith('[') and stripped.endswith(']')):
                        try:
                            return json_mod.loads(stripped)
                        except (json_mod.JSONDecodeError, ValueError):
                            pass
                return val

            nodes = []
            for node_id, attrs in graph.nodes(data=True):
                node = {'id': int(node_id)}
                for k, v in attrs.items():
                    node[k] = _try_parse(v)
                nodes.append(node)

            edges = []
            remote_nodes = {}
            for u, v, attrs in graph.edges(data=True):
                edge = {'source': int(u), 'target': int(v)}
                for k, val in attrs.items():
                    edge[k] = _try_parse(val)
                edges.append(edge)

            # Extract remote_nodes from graph-level attribute if present
            if 'remote_nodes' in graph.graph:
                remote_nodes = _try_parse(graph.graph['remote_nodes'])
                if isinstance(remote_nodes, dict):
                    remote_nodes = {int(k): int(v) for k, v in remote_nodes.items()}

            return {'nodes': nodes, 'edges': edges, 'remote_nodes': remote_nodes}

        elif ext == '.json':
            with open(partition_file, 'r') as f:
                data = json_mod.load(f)
            # JSON keys are always strings — convert node IDs and remote_nodes keys to int
            if 'remote_nodes' in data:
                data['remote_nodes'] = {int(k): int(v) for k, v in data['remote_nodes'].items()}
            return data

        else:
            raise ValueError(
                f"Unsupported partition file format: {ext}. "
                f"Supported: .pkl, .pickle, .graphml, .json"
            )

    def load_from_file(self, partition_file: str,
                       soma_breed: str = "lif_soma",
                       soma_config: str = "config_0",
                       synapse_breed: str = "single_exp_synapse",
                       synapse_config: str = "config_0") -> None:
        """Load a network definition file and build the neuromorphic model.

        Reads an SNN-native network file (pkl), computes property values from
        breed configs, builds connections, and calls SAGESim's
        build_from_local_data() for bulk creation.

        This is a one-shot, whole-model builder: it overwrites the agent factory
        and is MUTUALLY EXCLUSIVE with incremental create_soma/create_synapse.
        Call it exactly once, on a fresh model, and do not mix the two paths
        (doing so raises RuntimeError).

        Canonical file schema (a dict with 2 required keys + 1 optional):
            {
              "somas":    [{"id", "breed", "config", "overrides"}, ...],
              "synapses": [{"id", "pre", "post", "breed", "config",
                            "learning_rule", "learning_rule_config",
                            "overrides"}, ...],   # pre = -1 → external input
              "remote_ranks": {agent_id: rank}   # optional; only for multi-GPU
            }
        A legacy "metadata" key on any soma/synapse is ignored (labels are an
        application concern, not framework state).
        `overrides` is grouped: "hyperparameters", "internal_states",
        "learning_hyperparameters", "learning_internal_states". Legacy
        graph-centric files (nodes/edges/source/target) are rejected with a
        clear error — regenerate them with the updated producer.

        :param partition_file: Path to network file (.pkl)
        :param soma_breed: Default soma breed name
        :param soma_config: Default soma config name
        :param synapse_breed: Default synapse breed name
        :param synapse_config: Default synapse config name
        """
        if self._soma_ids or self._synapse_ids:
            raise RuntimeError(
                "load_from_file() builds the entire model in one shot and overwrites "
                "the agent factory wholesale; it cannot run on a model that already "
                f"has agents ({len(self._soma_ids)} somas, {len(self._synapse_ids)} "
                "synapses). load_from_file() and create_soma/create_synapse are "
                "mutually exclusive — use one construction path per model. (This also "
                "prevents calling load_from_file() more than once.)"
            )
        data = self._read_partition_file(partition_file)

        nodes = data['nodes']
        graph_edges = data['edges']
        # remote_agents maps ALL remote agent IDs (somas + synapses) to ranks
        remote_agents = data.get('remote_agents', data.get('remote_nodes', {}))
        local_node_ids = set(n['id'] for n in nodes)

        # Separate local edges (source local → create synapse agent) from
        # ghost edges (source remote → just a connection for STDP reads)
        local_edges = [e for e in graph_edges
                       if e['source'] in local_node_ids or e['source'] == -1]
        ghost_edges = [e for e in graph_edges
                       if e['source'] not in local_node_ids and e['source'] != -1]

        # Validate that all edges have synapse_id
        for edge in local_edges:
            if 'synapse_id' not in edge:
                raise ValueError(
                    f"Edge from {edge['source']} to {edge['target']} missing "
                    f"required 'synapse_id'. Assign synapse IDs before saving."
                )

        agents = []
        connections = []
        remote_agent_ranks = dict(remote_agents)

        # --- Somas from nodes ---
        for node in nodes:
            breed = node.get('breed', soma_breed)
            config = node.get('config', soma_config)
            overrides = node.get('overrides', {})
            hp, is_state = self._get_soma_properties(breed, config, overrides)

            agents.append({
                'id': node['id'],
                'breed': self._soma_breeds[breed],
                'properties': {
                    'hyperparameters': hp,
                    'internal_states': is_state,
                    'output_spikes_tensor': [0.0, 0.0],
                },
            })

            nid = node['id']
            self._soma_ids.add(nid)
            self.agentid2config[nid] = config
            self.agentid2overrides[nid] = overrides

        # --- Synapses from local edges ---
        for edge in local_edges:
            syn_id = edge['synapse_id']
            pre_id = edge['source']
            post_id = edge['target']
            breed = edge.get('breed', synapse_breed)
            config = edge.get('config', synapse_config)
            overrides = edge.get('overrides', {})
            learning_rule = edge.get('learning_rule', None)
            learning_rule_config = edge.get('learning_rule_config', 'default')

            props, hp, lhp, is_state, ils = self._get_synapse_properties(
                breed, config, overrides, learning_rule, learning_rule_config)

            agents.append({
                'id': syn_id,
                'breed': self._synapse_breeds[breed],
                'properties': props,
            })

            # Connections: synapse reads pre_soma and post_soma;
            # post_soma reads synapse (for STDP) only if post_soma is local.
            # For input synapses (pre_id == -1), we still add (syn_id, -1)
            # so that -1 occupies index 0 in the locations list.
            connections.append((syn_id, pre_id))
            if post_id != -1:
                connections.append((syn_id, post_id))
            if post_id != -1 and post_id in local_node_ids:
                connections.append((post_id, syn_id))

            # Bookkeeping
            self._synapse_ids.add(syn_id)
            self.agentid2config[syn_id] = config
            self.agentid2overrides[syn_id] = overrides
            self.agentid2learning_rule[syn_id] = (learning_rule, learning_rule_config) if learning_rule else None
            if pre_id != -1:
                self._soma_outgoing_synapses[pre_id].add(syn_id)

        # --- Ghost edges: local post-soma reads remote synapse ---
        # The synapse agent lives on the source's rank. The local post-soma
        # connects to it as a ghost neighbor. synapse_id and its rank must
        # be in the file (synapse_id on the edge, rank in remote_agents).
        for edge in ghost_edges:
            syn_id = edge['synapse_id']
            post_id = edge['target']
            connections.append((post_id, syn_id))

        # --- Bulk build via SAGESim ---
        self.build_from_local_data(agents, connections, remote_agent_ranks, directed=True)
        self._built_from_file = True

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
                ),
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
                # Record all somas (not synapses — they don't produce output spikes)
                for sid in self._soma_ids:
                    idx = buf.agent_id_to_index.get(sid, -1)
                    if 0 <= idx < num_local_agents:
                        mask[idx] = 1.0
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
        if not self.enable_internal_states_tracking:
            return []
        return super().get_agent_property_value(
            id=agent_id, property_name="internal_states_buffer"
        )

    def get_learning_internal_states_history(self, agent_id: int) -> np.array:
        if not self.enable_internal_states_tracking:
            return []
        return super().get_agent_property_value(
            id=agent_id, property_name="learning_internal_states_buffer"
        )

