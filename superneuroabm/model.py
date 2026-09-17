"""
Model class for building an SNN
"""

from collections import defaultdict, namedtuple
from collections.abc import Mapping
from typing import Dict, List, Set
from pathlib import Path

import numpy as np
import cupy as cp
from sagesim.space import NetworkSpace
from sagesim.model import Model
from sagesim.columns import IndexedColumn
from sagesim.breed import Breed

from superneuroabm.step_functions.soma.izh import izh_soma_step_func
from superneuroabm.step_functions.soma.lif import lif_soma_step_func
from superneuroabm.step_functions.soma.lif_soma_adaptive_thr import lif_soma_adaptive_thr_step_func
from superneuroabm.step_functions.soma.hg_lif import hg_lif_soma_step_func
from superneuroabm.step_functions.synapse.single_exp import synapse_single_exp_step_func
from superneuroabm.step_functions.synapse.weighted_synapse import weighted_synapse_step_func
from superneuroabm.util import load_component_configurations


#: What an agent IS, as far as its initial property values are concerned. Two agents with
#: the same Combo get byte-identical property rows, so the model stores one row per combo
#: and one int32 code per agent instead of one row per agent. A model where every agent
#: differs degrades to one combo per agent, which is what the per-agent dicts cost today.
Combo = namedtuple(
    "Combo",
    "component_class breed config learning_rule learning_rule_config overrides_key",
)


#: What every reader produces and the one builder consumes. Deliberately holds no
#: per-agent Python objects: ids and codes are arrays, and the combo table lives on the
#: model (the reader interns into it). ``nbr_values`` are GLOBAL agent ids, -1 allowed as
#: the external-input sentinel.
BuildSpec = namedtuple(
    "BuildSpec", "agent_ids combo_codes nbr_offsets nbr_values remote_ranks")


class _ComboField(Mapping):
    """Read-only ``agent_id -> field`` view over the model's combo store.

    Replaces the ``agentid2config`` / ``agentid2overrides`` / ``agentid2learning_rule``
    dicts. Those were three entries per agent; this is one int32 code per agent plus a
    table with one row per DISTINCT agent description. Kept as a Mapping so existing
    readers (``superneuroabm/util.py:93``, user code) are unchanged.
    """

    __slots__ = ("_model", "_get")

    def __init__(self, model, get):
        self._model = model
        self._get = get

    def __getitem__(self, agent_id):
        combo = self._model._combo_of(agent_id)
        if combo is None:
            raise KeyError(agent_id)
        return self._get(combo)

    def __iter__(self):
        return iter(self._model._combo_agent_ids())

    def __len__(self):
        return sum(1 for _ in self._model._combo_agent_ids())

    def __repr__(self):
        return f"{type(self).__name__}({dict(self)!r})"

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
        super().__init__(space=NetworkSpace(ordered=True),
                         agent_slack_factor=1.0, csr_slack_factor=1.0)

        if soma_breed_info is None:
            soma_breed_info = _default_soma_breeds()
        if synapse_breed_info is None:
            synapse_breed_info = _default_synapse_breeds()

        self.enable_internal_states_tracking = enable_internal_states_tracking
        self._config_list_cache = {}

        self.register_global_property("dt", 1e-3)      # Time step (100 μs)
        self.register_global_property("I_bias", 0)     # No bias current

        # Load and hold configurations (needed before property dicts are built)
        self.agentid2config = _ComboField(self, lambda c: c.config)
        if user_config is not None:
            self._component_configurations = load_component_configurations(user_config)
        else:
            self._component_configurations = load_component_configurations()

        max_sizes = _compute_max_property_sizes(self._component_configurations)

        # Separate learning rule configs before building property dicts
        self._learning_rule_configurations = self._component_configurations.pop("learning_rule", {})

        # What each agent IS: one int32 code per agent into a table of DISTINCT agent
        # descriptions. `agentid2config` / `agentid2overrides` / `agentid2learning_rule`
        # are read-only Mapping views over this (see _ComboField), so existing readers
        # are unchanged while the storage stops being three dict entries per agent.
        self._combo_table = []          # list[Combo]
        self._combo_index = {}          # Combo -> code, build-time interning
        self._agentid2combo = {}        # agent_id -> code, for create_soma/create_synapse
        self._combo_codes = None        # int32[n_local] by LOCAL ROW, for bulk builds
        self._overrides_cache = {}      # overrides_key -> rebuilt dict
        self._combo_reset_cache = {}    # code -> property rows to reset to
        self.agentid2learning_rule = _ComboField(
            self, lambda c: (c.learning_rule, c.learning_rule_config)
            if c.learning_rule is not None else None)

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
            "input_spikes_tensor": ([-1.0, 0.0], False),  # [last_delivered_tick, value]
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
            "input_spikes_tensor": ([-1.0, 0.0], False),  # [last_delivered_tick, value]
            "output_spikes_tensor": ([], False),
            "internal_states_buffer": ([], False),
            "learning_internal_states_buffer": ([], False),  # learning states buffer
        }
        # Backing stores for the _soma_ids / _synapse_ids properties. Incremental
        # creation adds to them directly; a bulk build leaves them None so the sets are
        # materialised from the combo codes only if something actually asks, which a
        # 12.5M-synapse run never does (it reads _num_synapses / _input_synapse_ids).
        self._synapse_ids_set = set()
        self._soma_ids_set = set()

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

        # External input spikes: host-side (ids, ticks, values) chunks, compiled
        # lazily into a tick-major device event list (see add_spike).
        self._input_events = []
        self._input_events_dirty = True
        self._input_events_gpu = None
        self._input_next_tick = 0      # tick the next launch is expected to start at

        # Learning mode (see eval()/train()). True = plasticity active.
        self._learning_enabled = True
        self._saved_stdp_type = {}       # synapse_id -> stdp_type saved while disabled

        self._soma_outgoing_synapses = defaultdict(set)  # soma_id -> set(synapse_ids)
        self.agentid2overrides = _ComboField(
            self, lambda c: NeuromorphicModel._overrides_from_key(c.overrides_key))

        self._breed_names = list(self._agent_factory._breeds.keys())

        # Learning rule registry
        if learning_rule_info is None:
            learning_rule_info = _default_learning_rules()
        self._learning_rules = learning_rule_info
        self._learning_rule_names = {r["func_name"]: rid for rid, r in self._learning_rules.items()}
        self._next_learning_rule_id = len(self._learning_rules)
        self._setup_called = False
        # Construction-mode lock: a model is built EITHER incrementally via
        # create_soma/create_synapse OR in one shot via a loader
        # (load_post_owned / load_from_adjacency) — never both. A loader
        # overwrites the agent factory wholesale, so the two paths cannot be
        # combined. Set True by either loader.
        self._built_from_file = False

    # ------------------------------------------------------------------
    # Combo store: what each agent IS, without a dict entry per agent
    # ------------------------------------------------------------------

    @staticmethod
    def _overrides_key(overrides) -> tuple:
        """Hashable, value-normalised form of an overrides dict.

        Values are coerced with ``float`` because the config path already does
        (``_get_soma_properties``), so ``"1e-3"`` and ``"1E-3"`` and ``0.001`` describe
        the same agent. Interning on the raw value would split combos the model treats
        as identical -- Cora's overrides carry exactly those string spellings.
        """
        if not overrides:
            return ()
        return tuple(sorted(
            (prop, tuple(sorted((k, float(v)) for k, v in d.items())))
            for prop, d in overrides.items() if d
        ))

    @staticmethod
    def _overrides_from_key(key) -> dict:
        """Inverse of _overrides_key: rebuild the grouped overrides dict."""
        return {prop: dict(kvs) for prop, kvs in key}

    def _intern_combo(self, component_class, breed, config,
                      learning_rule=None, learning_rule_config=None, overrides=None) -> int:
        """Return the code for this agent description, adding it to the table if new."""
        combo = Combo(component_class, breed, config, learning_rule,
                      learning_rule_config, self._overrides_key(overrides))
        code = self._combo_index.get(combo)
        if code is None:
            code = len(self._combo_table)
            self._combo_index[combo] = code
            self._combo_table.append(combo)
        return code

    @property
    def _local_rank(self) -> int:
        """This rank, cached. The combo accessors are called per agent in hot loops
        (_reset_agents alone makes one call per agent), and MPI.COMM_WORLD.Get_rank()
        inside a try/except is far too expensive to pay 35k times per reset."""
        rank = getattr(self, "_local_rank_cached", None)
        if rank is None:
            try:
                rank = MPI.COMM_WORLD.Get_rank()
            except Exception:
                rank = 0
            self._local_rank_cached = rank
        return rank

    def _combo_code_of(self, agent_id: int):
        """Combo code for an agent, or None if it is not locally known.

        Two backings: a dict for incrementally created agents (create_soma /
        create_synapse assign local rows in call order and accept explicit ids, so a
        row-indexed array would need care there for no benefit), and a row-indexed
        int32 array for bulk builds, where a dict would be the per-agent structure this
        store exists to avoid.
        """
        code = self._agentid2combo.get(agent_id)
        if code is not None:
            return code
        codes = self._combo_codes
        if codes is None:
            return None
        idx = self._agent_factory._rank2agentid2agentidx.get(
            self._local_rank, {}).get(agent_id)
        if idx is None or idx >= len(codes):
            return None
        return int(codes[idx])

    def _combo_of(self, agent_id: int):
        code = self._combo_code_of(agent_id)
        return self._combo_table[code] if code is not None else None

    def _combo_agent_ids(self):
        """Every locally known agent id that has a combo. Used only by the views."""
        yield from self._agentid2combo
        if self._combo_codes is not None:
            local = self._agent_factory._rank2agentid2agentidx.get(self._local_rank, {})
            n = len(self._combo_codes)
            for aid, idx in local.items():
                if idx < n and aid not in self._agentid2combo:
                    yield aid

    def _ids_of_class(self, component_class: str) -> set:
        """Local agent ids of one component class, derived from the combo codes."""
        codes = self._combo_codes
        ids = getattr(self, "_agent_ids", None)
        if codes is None or ids is None:
            return set()
        wanted = np.array([c.component_class == component_class
                           for c in self._combo_table], dtype=bool)
        return {int(a) for a in np.asarray(ids)[wanted[codes]]}

    @property
    def _soma_ids(self) -> set:
        if self._soma_ids_set is None:
            self._soma_ids_set = self._ids_of_class("soma")
        return self._soma_ids_set

    @property
    def _synapse_ids(self) -> set:
        """Every local synapse id.

        Materialised on demand rather than at build time: at 12.5M synapses this set is
        the kind of per-agent structure the columnar path exists to avoid, and the
        callers that matter at that scale read _num_synapses / _input_synapse_ids
        instead. Building it eagerly is what the columnar loader used to skip entirely,
        which silently emptied reset() and eval() -- lazy keeps it correct AND cheap.
        """
        if self._synapse_ids_set is None:
            self._synapse_ids_set = self._ids_of_class("synapse")
        return self._synapse_ids_set

    def _component_class_of(self, agent_id: int):
        combo = self._combo_of(agent_id)
        if combo is not None:
            return combo.component_class
        # Incremental models built before the combo store existed, and any agent the
        # store does not know: fall back to the id sets.
        return "soma" if agent_id in self._soma_ids else "synapse"

    def _config_of(self, agent_id: int):
        combo = self._combo_of(agent_id)
        return combo.config if combo is not None else None

    def _overrides_of(self, agent_id: int) -> dict:
        combo = self._combo_of(agent_id)
        return self._overrides_of_combo(combo) if combo is not None else {}

    def _overrides_of_combo(self, combo) -> dict:
        """Rebuilt once per COMBO, not once per agent that shares it."""
        cache = self._overrides_cache
        got = cache.get(combo.overrides_key)
        if got is None:
            got = self._overrides_from_key(combo.overrides_key)
            cache[combo.overrides_key] = got
        return got

    def _learning_rule_of(self, agent_id: int):
        combo = self._combo_of(agent_id)
        if combo is None or combo.learning_rule is None:
            return None
        return (combo.learning_rule, combo.learning_rule_config)

    def get_agent_config_name(self, agent_id: int) -> Dict[str, any]:
        """
        Returns the configuration of the agent with the given ID.
        """
        return self._config_of(agent_id)

    def get_agent_breed(self, agent_id: int) -> str:
        """
        Returns the breed of the agent with the given ID.
        """
        return self._breed_names[self._agent_factory._agent2breed[agent_id]]

    def get_neighbors(self, agent_id: int) -> List[int]:
        """The agent's neighbour list, in its stored slot order (global agent ids).

        For a synapse that is positional -- ``[pre]`` or ``[pre, post]``, with ``-1`` in
        slot 0 for an external input. For a soma it is its incoming synapse ids.

        Reads the CSR the builder produced rather than the ``locations`` property,
        because a bulk build hands SAGESim a prebuilt CSR and ``set_prebuilt_csr``
        clears the per-agent lists. It also survives ``reset()``, which frees the GPU
        buffers that the property route depends on.
        """
        offsets = getattr(self, "_nbr_offsets", None)
        if offsets is None:
            # Incrementally built (create_soma / create_synapse): the space still holds
            # real per-agent lists.
            return list(self.get_space().get_location(agent_id))
        row = self._agent_factory._rank2agentid2agentidx.get(
            self._local_rank, {}).get(agent_id)
        if row is None:
            raise KeyError(f"agent {agent_id} is not local to this rank")
        return [int(v) for v in self._nbr_values[offsets[row]:offsets[row + 1]]]

    def get_synapse_connectivity(self, synapse_id: int) -> List[int]:
        """
        Returns the connectivity of the synapse with the given ID.
        The connectivity is a list of length 2 containing pre and post soma IDs.

        Note: This returns the ordered locations [pre_soma_id, post_soma_id].
        These are agent IDs, not local indices.
        """
        return self.get_neighbors(synapse_id)

    def get_soma_outgoing_synapses(self, soma_id: int) -> Set[int]:
        """
        Returns the set of synapse IDs where this soma is the pre-synaptic source.
        """
        return self._soma_outgoing_synapses.get(soma_id, set())


    def get_agent_config_diff(self, agent_id: int) -> Dict[str, any]:
        """
        Returns the configuration overrides for the agent with the given ID.
        """
        component_class = self._component_class_of(agent_id)
        breed_name = self.get_agent_breed(agent_id)
        config_name = self._config_of(agent_id)
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
        lr_info = self._learning_rule_of(agent_id)
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

    # ------------------------------------------------------------------
    # Named parameter access
    # ------------------------------------------------------------------
    # Properties are stored as flat float vectors whose element order is the key
    # order of the agent's YAML block. These methods let callers work in parameter
    # names instead of positions, so a config reordering cannot silently change
    # which parameter a write lands on.

    # A few positions are contract rather than convention, because device code reads
    # them directly and cannot consult the config: every synapse step func reads
    # synapse_params[agent_index][0] and [1] for weight and synaptic_delay (see
    # step_functions/synapse/*.py), and the generated learning-rule selector reads
    # learning_params[agent_index][0] for stdp_type. Resolving these by name would
    # imply a flexibility that does not exist and would let a reordered config
    # mis-dispatch silently on the GPU, so they are named here and *validated* at
    # config-load time instead.
    _WEIGHT_INDEX = 0
    _SYNAPTIC_DELAY_INDEX = 1
    _STDP_TYPE_INDEX = 0

    @staticmethod
    def _assert_kernel_pinned_order(keys: list, name: str, index: int, where: str) -> None:
        """Fail loudly if a config puts a kernel-pinned parameter at the wrong position."""
        if len(keys) <= index or keys[index] != name:
            found = keys[index] if len(keys) > index else "<missing>"
            raise ValueError(
                f"{where}: {name!r} must be key #{index} because device code reads it "
                f"positionally, but found {found!r}. Key order is {keys}. Reorder the "
                f"config so {name!r} comes {'first' if index == 0 else f'at position {index}'}."
            )

    #: Hyperparameters that are consumed at creation time and cannot be changed on a
    #: live agent. Maps name -> the explanation raised to the caller.
    _CREATION_ONLY_HYPERPARAMETERS = {
        "synaptic_delay": (
            "synaptic_delay is fixed at create_synapse() time: the delay register is "
            "sized from it and rebuilt from config on every reset(), so writing it on "
            "a live agent has no effect. Change it in the component config and "
            "rebuild the model instead."
        ),
    }

    @staticmethod
    def _as_id_list(ids) -> tuple:
        """Normalize an id or iterable of ids to ``(list, was_scalar)``."""
        if isinstance(ids, (int, np.integer)):
            return [int(ids)], True
        return [int(i) for i in ids], False

    def _hp_key_index(self, agent_id: int, property_name: str) -> Dict[str, int]:
        """Map parameter name -> index into ``agent_id``'s ``property_name`` vector.

        The index is the key's position in the agent's YAML block, which is exactly
        how _get_soma_properties / _get_synapse_properties build the vector, so it is
        derived here rather than hard-coded. Reuses _config_list_cache.
        """
        if property_name == "hyperparameters":
            component_class = self._component_class_of(agent_id)
            breed_name = self.get_agent_breed(agent_id)
            config_name = self._config_of(agent_id)
            cache_key = (component_class, breed_name, config_name)
            if cache_key not in self._config_list_cache:
                config = self._component_configurations[component_class][breed_name][config_name]
                self._config_list_cache[cache_key] = (
                    list(config["hyperparameters"].keys()),
                    [float(v) for v in config["hyperparameters"].values()],
                    list(config["internal_states"].keys()),
                    [float(v) for v in config["internal_states"].values()],
                )
            keys = self._config_list_cache[cache_key][0]
        elif property_name == "learning_hyperparameters":
            lr_info = self._learning_rule_of(agent_id)
            if lr_info is None:
                # Synapses created without a rule carry the synthetic single-element
                # vector create_synapse() gives them.
                keys = ["stdp_type"]
            else:
                lr_breed, lr_config_name = lr_info
                cache_key = ("learning_rule", lr_breed, lr_config_name)
                if cache_key not in self._config_list_cache:
                    lr_config = self._learning_rule_configurations[lr_breed][lr_config_name]
                    self._config_list_cache[cache_key] = (
                        list(lr_config["learning_hyperparameters"].keys()),
                        [float(v) for v in lr_config["learning_hyperparameters"].values()],
                        list(lr_config.get("learning_internal_states", {}).keys()),
                        [float(v) for v in lr_config.get("learning_internal_states", {}).values()],
                    )
                keys = self._config_list_cache[cache_key][0]
        else:
            raise ValueError(
                f"No name mapping for {property_name!r}; expected 'hyperparameters' "
                f"or 'learning_hyperparameters'."
            )
        return {name: i for i, name in enumerate(keys)}

    def _get_named_properties(self, ids, property_name: str):
        id_list, scalar = self._as_id_list(ids)
        out = []
        for agent_id in id_list:
            index = self._hp_key_index(agent_id, property_name)
            values = self.get_agent_property_value(id=agent_id, property_name=property_name)
            out.append({name: values[i] for name, i in index.items()})
        return out[0] if scalar else out

    def _set_named_properties(self, ids, updates: dict, property_name: str) -> None:
        if not updates:
            return
        id_list, _ = self._as_id_list(ids)
        if not id_list:
            return

        if property_name == "hyperparameters":
            for name in updates:
                if name in self._CREATION_ONLY_HYPERPARAMETERS:
                    raise ValueError(self._CREATION_ONLY_HYPERPARAMETERS[name])

        # Writes land on the CPU-side AgentFactory, but after a simulate() the GPU
        # holds values the CPU has never seen (STDP weights, membrane state). The
        # next simulate() rebuilds the GPU from the CPU copy, so writing here would
        # silently discard everything the kernel learned. reset() is currently the
        # only GPU->CPU sync, so require it rather than corrupting the model.
        # See docs/CPU_GPU_DATA_FLOW.md; this guard goes away once the sync layer
        # tracks staleness directly.
        if self._buffers_live():
            raise RuntimeError(
                f"set_{property_name}() cannot run while GPU buffers hold unsynced "
                f"state: the write would land on the CPU copy and the next simulate() "
                f"would re-upload it, discarding GPU-learned values such as STDP "
                f"weights. Call reset(retain_parameters=True) first -- it syncs the "
                f"GPU back to the CPU and keeps learned parameters. "
                f"See docs/CPU_GPU_DATA_FLOW.md."
            )

        for agent_id in id_list:
            index = self._hp_key_index(agent_id, property_name)
            unknown = [name for name in updates if name not in index]
            if unknown:
                raise KeyError(
                    f"Unknown {property_name} {unknown} for agent {agent_id}; "
                    f"valid names are {list(index)}."
                )
            # Copy before mutating: for a list column get_agent_property_value returns
            # the STORED row object (sagesim/agent.py:221-224), so writing into it in
            # place edits the column directly -- and once rows are shared between agents
            # that carry identical parameters, it would edit every agent sharing the row.
            # _write_stdp_type copies for the same reason.
            values = list(self.get_agent_property_value(
                id=agent_id, property_name=property_name))
            for name, value in updates.items():
                values[index[name]] = float(value)
            self.set_agent_property_value(agent_id, property_name, values)

    def get_hyperparameters(self, ids):
        """Read hyperparameters by name.

        :param ids: One agent id, or an iterable of them.
        :return: ``{name: value}`` for a single id, or a list of such dicts, in the
            order the ids were given.
        """
        return self._get_named_properties(ids, "hyperparameters")

    def set_hyperparameters(self, ids, updates: dict) -> None:
        """Update hyperparameters by name, leaving unnamed ones untouched.

        Must be called before the first simulate(), or after a
        ``reset(retain_parameters=True)`` -- see docs/CPU_GPU_DATA_FLOW.md.

        :param ids: One agent id, or an iterable of them.
        :param updates: ``{name: value}``; an empty dict is a no-op.
        :raises KeyError: a name is not a hyperparameter of that agent's breed/config.
        :raises ValueError: the name can only be set at creation time.
        :raises RuntimeError: GPU buffers hold unsynced state; reset() first.
        """
        self._set_named_properties(ids, updates, "hyperparameters")

    def get_learning_hyperparameters(self, ids):
        """Read learning-rule hyperparameters by name. See get_hyperparameters."""
        return self._get_named_properties(ids, "learning_hyperparameters")

    def set_learning_hyperparameters(self, ids, updates: dict) -> None:
        """Update learning-rule hyperparameters by name. See set_hyperparameters.

        Note that plasticity is better switched with eval() / train(), which
        snapshot and restore stdp_type for you.
        """
        self._set_named_properties(ids, updates, "learning_hyperparameters")

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

    # ------------------------------------------------------------------
    # Learning mode (plasticity on/off)
    # ------------------------------------------------------------------

    @property
    def learning_enabled(self) -> bool:
        """Whether plasticity is currently active. See eval() / train()."""
        return self._learning_enabled

    def _plastic_synapse_ids(self) -> list:
        """Synapses that actually carry a learning rule.

        Synapses created without one already default to stdp_type = -1, so they
        must be left alone: writing the sentinel would be a no-op but restoring
        it later could resurrect a rule that was never there.

        Derived from the combo store rather than from _synapse_ids, which a bulk build
        never fills -- that is why eval()/train() were silent no-ops on a columnar-loaded
        model. Only combos that carry a rule are scanned, so a network with no plasticity
        costs one pass over the (tiny) combo table and nothing per synapse.
        """
        plastic = {code for code, combo in enumerate(self._combo_table)
                   if combo.component_class == "synapse" and combo.learning_rule is not None}
        if not plastic:
            return []
        out = [sid for sid, code in self._agentid2combo.items() if code in plastic]
        codes = self._combo_codes
        if codes is not None:
            local = self._agent_factory._rank2agentid2agentidx.get(self._local_rank, {})
            n = len(codes)
            out.extend(sid for sid, idx in local.items()
                       if idx < n and int(codes[idx]) in plastic
                       and sid not in self._agentid2combo)
        return out

    def _buffers_live(self) -> bool:
        return (
            getattr(self, "_setup_called", False)
            and hasattr(self, "_gpu_buffers")
            and self._gpu_buffers.is_initialized
        )

    def _read_stdp_type(self, synapse_ids: list) -> dict:
        """Read stdp_type (learning_hyperparameters[0]) for locally-owned synapses."""
        af = self._agent_factory
        prop_idx = af._property_name_2_index["learning_hyperparameters"]
        out = {}
        if self._buffers_live():
            buf = self._gpu_buffers
            tensor = buf.property_tensors[prop_idx]
            idxs, ids = [], []
            for sid in synapse_ids:
                i = buf.agent_id_to_index.get(sid, -1)
                if 0 <= i < buf.num_local_agents:
                    idxs.append(i)
                    ids.append(sid)
            if idxs:
                rows = self._gpu_buffers.rows(prop_idx, cp.asarray(idxs))
                vals = rows[:, self._STDP_TYPE_INDEX].get().tolist()
                out = dict(zip(ids, vals))
        else:
            rank = MPI.COMM_WORLD.Get_rank()
            local_agent_map = af._rank2agentid2agentidx.get(rank, {})
            data = af._property_name_2_agent_data_tensor["learning_hyperparameters"]
            for sid in synapse_ids:
                idx = local_agent_map.get(sid)
                if idx is not None:
                    out[sid] = float(data[idx][self._STDP_TYPE_INDEX])
        return out

    def _write_stdp_type(self, id2value: dict) -> None:
        """Batch-write stdp_type for locally-owned synapses.

        Writes the CPU-side AgentFactory, which is the durable copy, then
        invalidates the GPU buffers once so the next simulate() picks the change
        up when it rebuilds them. Same semantics as set_agent_property_value, but
        without one MPI-collective read and one invalidation per synapse.

        Writing the GPU tensor instead would be faster but wrong: the two copies
        would diverge, and any later property write triggers a rebuild that
        repopulates the GPU from the AgentFactory, silently reverting this.
        """
        if not id2value:
            return
        af = self._agent_factory
        local_agent_map = af._rank2agentid2agentidx.get(self._local_rank, {})
        data = af._property_name_2_agent_data_tensor["learning_hyperparameters"]

        rows, vals = [], []
        for sid, value in id2value.items():
            idx = local_agent_map.get(sid)
            if idx is not None:
                rows.append(idx)
                vals.append(float(value))
        touched = bool(rows)
        rows = np.asarray(rows, dtype=np.int64)
        vals = np.asarray(vals, dtype=np.float64)

        # Read-modify-write the row: a column stored as a padded array hands out a copy,
        # so an element write through the alias would be lost (and on a list column
        # shared between synapses it would have changed every synapse holding that row).
        if isinstance(data, IndexedColumn) and touched:
            # Group by (current table code, new value). Every synapse in a group starts
            # from the same row and wants the same stdp_type, so the group has ONE
            # result -- interned once and scattered, instead of hashing a row per
            # synapse. eval() over a large plastic population is a single group.
            codes = np.asarray(data.codes)[rows]
            groups = []
            for code, value in {(int(c), float(v)) for c, v in zip(codes, vals)}:
                sel = rows[(codes == code) & (vals == value)]
                row = list(data[int(sel[0])])
                row[self._STDP_TYPE_INDEX] = value
                groups.append((row, sel))          # snapshot before mutating any codes
            for row, sel in groups:
                self._scatter_row(data, row, sel)
        else:
            for i, value in zip(rows.tolist(), vals.tolist()):
                row = list(data[i])
                row[self._STDP_TYPE_INDEX] = value
                data[i] = row

        # _generate_agent_data_tensors() hands out references to these same
        # lists, so the write above is already visible model-side; no
        # _regenerate_data_tensors() needed. Only the GPU copy is now stale.
        if touched and self._buffers_live():
            self._gpu_buffers.is_initialized = False
            self._cached_all_args = None

    def set_learning_enabled(self, enabled: bool, synapse_ids: list = None) -> None:
        """
        Enable or disable plasticity by flipping stdp_type, which is element 0 of
        each synapse's ``learning_hyperparameters``.

        Disabling snapshots the current stdp_type and writes -1.0, the sentinel the
        generated learning-rule selector treats as "no rule". Enabling restores the
        snapshot. Weights are never touched, so disabling freezes them in place.

        :param enabled: True to restore plasticity, False to freeze it.
        :param synapse_ids: Synapses to affect. Defaults to every synapse that was
            created with a learning rule.
        """
        targets = self._plastic_synapse_ids() if synapse_ids is None else list(synapse_ids)
        if not enabled:
            # Snapshot only what we have not already saved, so repeated eval()
            # calls cannot overwrite the real values with the sentinel.
            unsaved = [sid for sid in targets if sid not in self._saved_stdp_type]
            self._saved_stdp_type.update(self._read_stdp_type(unsaved))
            self._write_stdp_type({sid: -1.0 for sid in targets})
        else:
            restore = {
                sid: self._saved_stdp_type.pop(sid)
                for sid in targets
                if sid in self._saved_stdp_type
            }
            self._write_stdp_type(restore)
        if synapse_ids is None:
            self._learning_enabled = enabled

    def eval(self):
        """Switch to inference mode: freeze plasticity on all learning synapses.

        A pure mode flip, like PyTorch's -- it does not clear membrane voltage or
        synaptic current. Call ``reset()`` separately for that. The mode survives
        ``reset()`` in either order.

        :return: self, so calls can be chained.
        """
        self.set_learning_enabled(False)
        return self

    def train(self):
        """Switch back to training mode: restore plasticity.

        :return: self, so calls can be chained.
        """
        self.set_learning_enabled(True)
        return self

    @staticmethod
    def _scatter_row(col, row, rows) -> None:
        """Write one row into many positions of a column, without a per-row write.

        Every agent sharing a combo resets to the SAME values, so this is a scatter, not
        a loop. Each column representation gets its cheapest form, using public API only:

        * IndexedColumn -- intern the row once (one hash lookup), then assign its code to
          every position. ``codes`` is a live view, so this is one numpy scatter. Writing
          row by row instead would re-hash the row for every agent, which is what made
          reset 4x slower than it needed to be once the builder started emitting
          IndexedColumns.
        * ArrayColumn -- one fancy-index assignment into ``values`` plus ``lengths``.
        * plain list -- one SHARED row object per position, which keeps the column
          eligible for SAGESim's identity-based interning.
        """
        if not len(rows):
            return
        rows = np.asarray(rows)
        if isinstance(col, IndexedColumn):
            col[int(rows[0])] = row                  # interns; one hash lookup
            col.codes[rows] = col.codes[int(rows[0])]
            return
        values = getattr(col, "values", None)
        if values is not None and not col.degraded and values.ndim == 2:
            n = len(row)
            if n > values.shape[1]:
                col[int(rows[0])] = row              # let the column grow, then retry
                values = col.values
            values[np.ix_(rows, np.arange(len(row)))] = row
            if values.shape[1] > len(row):
                values[rows, len(row):] = col.fill
            col.lengths[rows] = len(row)
            return
        shared = list(row)
        for i in rows.tolist():
            col[int(i)] = shared

    def _combo_reset_rows(self, code: int) -> dict:
        """Property rows a combo resets to. Computed once per combo, then cached."""
        cached = self._combo_reset_cache.get(code)
        if cached is None:
            cached = self._combo_property_rows(self._combo_table[code])
            self._combo_reset_cache[code] = cached
        return cached

    def _reset_agents(self, retain_parameters: bool = True) -> None:
        """
        Internal method to reset all soma and synapse agents to their initial states.
        Recomputes defaults from (breed, config, overrides) via config cache.

        Grouped by combo, not by agent: every agent with the same description resets to
        byte-identical rows, so the rows are computed once per combo and scattered. The
        old per-agent form recomputed the same handful of results N times and wrote them
        N times -- 205k column writes on a 35k-agent network.

        :param retain_parameters: If True, keeps current learned parameters.
            If False, resets parameters to their default values.
        """
        af = self._agent_factory
        local_agent_map = af._rank2agentid2agentidx.get(self._local_rank, {})
        if not local_agent_map:
            return
        data = af._property_name_2_agent_data_tensor
        n_local = len(local_agent_map)

        # Combo code per LOCAL ROW. Bulk builds already hold exactly this array;
        # incrementally created models are assembled from the id map.
        codes = self._combo_codes
        if codes is not None and len(codes) >= n_local:
            row_codes = np.asarray(codes[:n_local], dtype=np.int64)
        else:
            row_codes = np.full(n_local, -1, dtype=np.int64)
            by_id = self._agentid2combo
            for aid, idx in local_agent_map.items():
                c = by_id.get(aid)
                if c is not None and idx < n_local:
                    row_codes[idx] = c

        # Parameters are kept on retain_parameters=True; state always resets.
        keep = {"hyperparameters", "learning_hyperparameters"} if retain_parameters else set()

        for code in np.unique(row_codes):
            code = int(code)
            if code < 0:
                continue          # agent with no combo: nothing known to reset it to
            rows = np.flatnonzero(row_codes == code)
            for prop_name, row in self._combo_reset_rows(code).items():
                if prop_name in keep:
                    continue
                self._scatter_row(data[prop_name], row, rows)


    def reset(self, retain_parameters: bool = True) -> None:
        """
        Resets all soma and synapse agents to their initial states.

        :param retain_parameters: If True, keeps current learned parameters
            (e.g. STDP weights). If False, resets parameters to defaults.
        """
        # Step 1: SAGESim syncs GPU->AgentFactory, regenerates tensors, frees GPU.
        # Only the columns we are KEEPING need to come back off the device. Everything
        # else is about to be restored from the combo table by _reset_agents, so reading
        # it back first is a device->host copy of data we immediately overwrite. The
        # learned values live in hyperparameters (weight) and learning_hyperparameters;
        # the STDP traces in learning_internal_states are state, not parameters, and are
        # meant to be cleared.
        super().reset(sync_properties=("hyperparameters", "learning_hyperparameters")
                      if retain_parameters else ())

        # Step 2: Reset agent states on AgentFactory (keeps hyperparameters if retain=True)
        self._reset_agents(retain_parameters=retain_parameters)

        # Step 2b: Learning mode is sticky. With retain_parameters=False the step
        # above restores learning_hyperparameters from config, which would silently
        # re-enable plasticity after eval(); re-apply the sentinel here.
        if not self._learning_enabled:
            targets = self._plastic_synapse_ids()
            unsaved = [sid for sid in targets if sid not in self._saved_stdp_type]
            self._saved_stdp_type.update(self._read_stdp_type(unsaved))
            self._write_stdp_type({sid: -1.0 for sid in targets})

        # Step 3: Regenerate data tensors to reflect the reset states
        super()._regenerate_data_tensors()

        # Step 3b: super().reset() synced the GPU back with .get().tolist(), which
        # replaced every column with fresh per-agent lists. Re-share the write-only
        # history buffers so a chunked run does not rebuild them row by row on the
        # next tick. Learned values in the other columns are correctly per-agent now.
        if not self.enable_internal_states_tracking:
            self._share_history_buffers()
            super()._regenerate_data_tensors()

        # Step 4: Clear recording state + caches. Injected input spikes are
        # discarded too (reset has always done that); re-inject after reset().
        self._recorded_spikes = []
        self._spike_record_gpu = None
        self._spike_record_count_gpu = None
        self._spike_mask_gpu = None  # rebuild mask on next prepare
        self._spikes_need_gather = False
        self.clear_input_spikes()
        self._input_next_tick = 0
        # self._agent_factory._prev_agent_data.clear()
        
    def setup(self) -> None:
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
            self.clear_input_spikes()
        # Buffers are rebuilt by setup(); the compiled event list refers to the
        # old buffer rows, so recompile at the next tick.
        self._input_events_dirty = True
        self._input_events_gpu = None
        self._input_next_tick = 0

        import time
        _t_pre = time.time()
        self._recorded_spikes = []
        self._spike_record_gpu = None
        self._spike_record_count_gpu = None
        self._spike_mask_gpu = None  # rebuild mask on next prepare
        self._spikes_need_gather = False
        self.set_property_neighbor_visible("breed", False)  # no step func reads neighbor breeds
        _t_super = time.time()
        super().setup(skip_priority_barriers={100})
        self._setup_timings["snn_pre_super"] = _t_super - _t_pre
        self._setup_timings["sagesim_setup"] = time.time() - _t_super

        if not self.enable_internal_states_tracking:
            # Tracking off does not remove the history buffers: every synapse and
            # learning kernel writes them unconditionally via
            # buffer[agent][t % len(buffer[agent])][k], so each row is shrunk to a
            # single write-only slot instead. That per-agent loop is timed here
            # because it is the whole of setup()'s cost outside SAGESim.
            _t_buf = time.time()
            self._share_history_buffers()
            self._setup_timings["snn_shrink_buffers"] = time.time() - _t_buf

    def _share_history_buffers(self) -> None:
        """Give every agent the SAME one-slot history-buffer row, when tracking is off.

        Tracking off does not remove these buffers: every synapse and learning kernel
        writes them unconditionally as ``buffer[agent][t % len(buffer[agent])][k]``, so
        the column cannot be empty. Each agent therefore gets a single write-only slot.

        What this does NOT change is the value, the outer length (1) or the inner width,
        so the GPU tensor keeps its ``(capacity, 1, W)`` shape. What it changes is the
        number of Python objects behind those values: one per agent becomes one in total,
        referenced by every agent. That removes an N-iteration allocation loop, and it is
        what lets SAGESim's converter collapse the column instead of walking every row.

        Sharing is safe because nothing reads these rows: the kernels only write them,
        ``get_internal_states_history`` returns [] while tracking is off, and no host
        code mutates a buffer row in place (only whole-row replacement). Device rows are
        independent regardless, since the converter copies into a dense tensor.
        """
        af = self._agent_factory
        data = af._property_name_2_agent_data_tensor
        n_local = len(data["internal_states"])
        if not n_local:
            return
        # Width must be the max over agents, not any one agent's: breeds differ (a
        # single_exp_synapse carries 1 internal state, a lif_soma 3), the tensor width is
        # the column max, and kernels write up to their own state width. A narrower
        # shared row would shrink the tensor into unchecked out-of-bounds device writes.
        def _max_len(col):
            if hasattr(col, "max_length"):          # ArrayColumn: no per-row walk
                return col.max_length()
            return max(map(len, col)) if col else 0

        def _fill(col, row):
            # Write in place, do not rebind. Model.setup() caches
            # __rank_local_agent_data_tensors as references to these very column
            # objects (_generate_agent_data_tensors returns dict.values()), so
            # replacing the dict entry would leave the GPU build reading the old column.
            if hasattr(col, "fill_rows"):           # ArrayColumn: one broadcast write
                col.fill_rows(row)
            else:
                col[:] = [row] * n_local            # one shared list object per row

        w_is = _max_len(data["internal_states"])
        w_lis = _max_len(data["learning_internal_states"])
        _fill(data["internal_states_buffer"], [[0.0] * w_is])
        _fill(data["learning_internal_states_buffer"], [[0.0] * w_lis])

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
        # Input spikes need no host-side preparation here: they are compiled into
        # the tick-major event list in _prepare_kernel_extras when the kernel launches.
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
                "Cannot create_soma() on a model built via load_post_owned()/"
                "load_from_adjacency(). A loader builds the entire model in one "
                "shot and is mutually exclusive with incremental "
                "create_soma/create_synapse — use one construction path per model."
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
        self._agentid2combo[soma_id] = self._intern_combo(
            "soma", breed, config_name, overrides=overrides)
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

        Returns:
            int: SAGESim agent ID of the created synapse.
        """
        if self._built_from_file:
            raise RuntimeError(
                "Cannot create_synapse() on a model built via load_post_owned()/"
                "load_from_adjacency(). A loader builds the entire model in one "
                "shot and is mutually exclusive with incremental "
                "create_soma/create_synapse — use one construction path per model."
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
            self._assert_kernel_pinned_order(
                hp_keys, "weight", self._WEIGHT_INDEX, f"synapse config {breed}/{config_name}")
            self._assert_kernel_pinned_order(
                hp_keys, "synaptic_delay", self._SYNAPTIC_DELAY_INDEX,
                f"synapse config {breed}/{config_name}")
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
                self._assert_kernel_pinned_order(
                    lhp_keys, "stdp_type", self._STDP_TYPE_INDEX,
                    f"learning rule config {learning_rule}/{learning_rule_config}")
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

        # `synapse_delay_reg` is registered (positional kernel argument 12 stays in place)
        # but carries no payload: no built-in or duplicate kernel indexes it, and a
        # 12.5 M-synapse network paid 50 MB on the GPU for the zeros.
        delay_reg = []
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
        self._agentid2combo[synapse_id] = self._intern_combo(
            "synapse", breed, config_name,
            learning_rule=learning_rule if learning_rule else None,
            learning_rule_config=learning_rule_config if learning_rule else None,
            overrides=overrides)

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

        # Second connection: post_soma (if exists). Under post-owns, an
        # incrementally created synapse's post-soma is always local, so the
        # reverse connection (post reads synapse, for STDP) is always made.
        if post_soma_id != -1:
            network_space.connect_agents(synapse_id, post_soma_id, directed=True)
            network_space.connect_agents(post_soma_id, synapse_id, directed=True)
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
            self._assert_kernel_pinned_order(
                hp_keys, "weight", self._WEIGHT_INDEX, f"synapse config {breed}/{config_name}")
            self._assert_kernel_pinned_order(
                hp_keys, "synaptic_delay", self._SYNAPTIC_DELAY_INDEX,
                f"synapse config {breed}/{config_name}")
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
                self._assert_kernel_pinned_order(
                    lhp_keys, "stdp_type", self._STDP_TYPE_INDEX,
                    f"learning rule config {learning_rule}/{learning_rule_config}")
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

        delay_reg = []                    # see create_synapse: registered, no payload

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
        """Validate and return the SNN-native file schema verbatim.

        Canonical schema: {'somas': [...], 'synapses': [...], 'remote_ranks': {...}}
          - soma:    {'id', 'breed', 'config', 'overrides'}
          - synapse: {'id', 'pre', 'post', 'breed', 'config', 'learning_rule',
                      'learning_rule_config', 'overrides'}  (pre = -1 → input)
        Any legacy 'metadata' key on a soma/synapse is ignored — labels are an
        application concern, not framework state.

        The loader consumes 'somas'/'synapses' directly, so no translation is
        done here. Raises on the legacy graph-centric schema (nodes/edges) so old
        files fail loudly instead of silently mis-loading.
        """
        if 'somas' in data or 'synapses' in data:
            return {
                'somas': list(data.get('somas', [])),
                'synapses': list(data.get('synapses', [])),
                'remote_ranks': dict(data.get('remote_ranks', {})),
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
        """Read a per-rank partition file (pickle) into the canonical schema.

        Returns {'somas': [...], 'synapses': [...], 'remote_ranks': {...}}.

        Only pickle (.pkl/.pickle) is supported; it is the format every in-repo
        producer emits (build_snn_from_data.py, brunel.py).
        """
        ext = Path(partition_file).suffix.lower()

        if ext in ('.pkl', '.pickle'):
            import pickle
            with open(partition_file, 'rb') as f:
                data = pickle.load(f)
            return NeuromorphicModel._normalize_snn_partition(data)

        raise ValueError(
            f"Unsupported partition file format: {ext}. Only .pkl/.pickle is "
            "supported (the format emitted by the in-repo partition producers)."
        )

    @staticmethod
    def _read_columnar_partition_file(partition_file: str) -> dict:
        """Read a columnar (.npz) post-owned partition into an array dict.

        Memory-maps the file (``mmap_mode='r'``) so nothing is eagerly copied into
        host RAM; the columnar build reads each array with vectorized numpy and
        only the derived structures (property columns, CSR) are materialized. The
        on-disk schema is the one ``brunel_partition(output_format='columns')``
        writes (see its docstring).
        """
        import numpy as np
        data = np.load(partition_file, mmap_mode='r', allow_pickle=True)
        arrays = {k: data[k] for k in data.files}
        schema = str(arrays.get('schema', ''))
        if schema != 'columnar_post_owned_v1':
            raise ValueError(
                f"Unrecognized columnar partition schema {schema!r}; expected "
                "'columnar_post_owned_v1' (regenerate with "
                "brunel_partition(output_format='columns')).")
        return arrays

    @staticmethod
    def _normalize_neighbor_partition(data: dict) -> dict:
        """Validate and return the explicit-neighbors (load_from_adjacency) schema.

        Schema: {'somas': [...], 'synapses': [...], 'remote_ranks': {...}}
          - soma:    {'id', 'breed', 'config', 'overrides',
                      'neighbors': [incoming_syn_id, ...]}
          - synapse: {'id', 'breed', 'config', 'learning_rule',
                      'learning_rule_config', 'overrides',
                      'neighbors': [pre[, post]]}
        Synapse 'neighbors' is POSITIONAL ([pre] or [pre, post]); soma 'neighbors'
        is order-free. Returned verbatim — neighbor lists are NEVER reordered here.

        Rejects the legacy graph-centric schema (nodes/edges) AND a Method-1
        pre/post file (no 'neighbors') so the wrong file fails loudly instead of
        loading with empty neighbor lists.
        """
        if 'nodes' in data or 'edges' in data:
            raise ValueError(
                "Legacy graph-centric network format detected (found "
                f"{sorted(k for k in ('nodes', 'edges') if k in data)}). "
                "load_from_adjacency() expects the explicit-neighbors schema: "
                "'somas'/'synapses', each entry carrying a 'neighbors' list."
            )
        if 'somas' not in data and 'synapses' not in data:
            raise ValueError(
                "Unrecognized network file: expected top-level 'somas' and "
                "'synapses' keys for load_from_adjacency()."
            )

        somas = list(data.get('somas', []))
        synapses = list(data.get('synapses', []))

        # Every entry must carry 'neighbors'. A Method-1 (pre/post) file lacks it
        # and fails here rather than silently loading empty lists.
        for kind, entries in (('soma', somas), ('synapse', synapses)):
            for e in entries:
                if 'neighbors' not in e:
                    raise ValueError(
                        f"load_from_adjacency(): {kind} id={e.get('id')} has no "
                        "'neighbors' list. This looks like a load_post_owned() "
                        "(pre/post) partition — use load_post_owned() for those, "
                        "or regenerate with an explicit-neighbors producer."
                    )

        return {
            'somas': somas,
            'synapses': synapses,
            'remote_ranks': dict(data.get('remote_ranks', {})),
        }

    @staticmethod
    def _read_neighbor_partition_file(partition_file: str) -> dict:
        """Read an explicit-neighbors partition file (pickle).

        Returns {'somas': [...], 'synapses': [...], 'remote_ranks': {...}} where
        each soma/synapse carries its own 'neighbors' list. Only pickle is
        supported, matching _read_partition_file().
        """
        ext = Path(partition_file).suffix.lower()
        if ext in ('.pkl', '.pickle'):
            import pickle
            with open(partition_file, 'rb') as f:
                data = pickle.load(f)
            return NeuromorphicModel._normalize_neighbor_partition(data)
        raise ValueError(
            f"Unsupported partition file format: {ext}. Only .pkl/.pickle is "
            "supported (the format emitted by the in-repo partition producers)."
        )

    def _assert_unbuilt(self, who: str) -> None:
        """Guard the one-shot, whole-model construction contract for the loaders.

        Raises if the model already has agents — either from a prior loader call
        or from incremental create_soma/create_synapse. ``who`` names the calling
        loader in the error message.
        """
        if self._soma_ids or self._synapse_ids:
            raise RuntimeError(
                f"{who}() builds the entire model in one shot and overwrites the "
                "agent factory wholesale; it cannot run on a model that already "
                f"has agents ({len(self._soma_ids)} somas, "
                f"{len(self._synapse_ids)} synapses). load_post_owned(), "
                "load_from_adjacency(), and create_soma/create_synapse are "
                "mutually exclusive — use one construction path per model. (This "
                f"also prevents calling {who}() more than once.)"
            )

    def load_post_owned(self, partition_file: str,
                        soma_breed: str = "lif_soma",
                        soma_config: str = "config_0",
                        synapse_breed: str = "single_exp_synapse",
                       synapse_config: str = "config_0") -> None:
        """Load a POST-OWNED network file (Method 1) and build the model.

        The producer lists each synapse by its ``pre``/``post`` endpoints; this
        loader DERIVES all connectivity, including each post-soma's incoming-
        synapse list. The method name carries its CONSTRAINT: every synapse's
        post-soma must be local on the synapse's rank (post-owns / NEST). This is
        required because a post-soma builds its incoming list by scanning the
        synapses in its OWN file — so it can only discover incoming synapses that
        are listed locally. There is no way to name a *remote* incoming synapse in
        this schema; if you need that, use ``load_from_adjacency()`` instead.

        The constraint is ENFORCED here: a synapse whose ``post`` is not a local
        soma raises (it would otherwise be silently miswired).

        One-shot, whole-model builder: overwrites the agent factory and is
        MUTUALLY EXCLUSIVE with incremental create_soma/create_synapse and with
        ``load_from_adjacency()``. Call exactly once on a fresh model.

        File schema (a dict with 2 required keys + 1 optional):
            {
              "somas":    [{"id", "breed", "config", "overrides"}, ...],
              "synapses": [{"id", "pre", "post", "breed", "config",
                            "learning_rule", "learning_rule_config",
                            "overrides"}, ...],   # pre = -1 → external input
              "remote_ranks": {agent_id: rank}   # optional; remote pre-somas
            }
        `overrides` is grouped: "hyperparameters", "internal_states",
        "learning_hyperparameters", "learning_internal_states". Legacy
        graph-centric files (nodes/edges/source/target) are rejected.

        :param partition_file: Path to network file (.pkl)
        :param soma_breed: Default soma breed name
        :param soma_config: Default soma config name
        :param synapse_breed: Default synapse breed name
        :param synapse_config: Default synapse config name
        """
        self._assert_unbuilt("load_post_owned")
        # Encoding is a READER detail, not a builder: row-oriented records and columnar
        # arrays describe the same post-owned network and end at the same _build(spec).
        # `partition_file` may also be the already-loaded data, so a producer that builds
        # in memory has a public entry point instead of reaching for a private builder.
        source = partition_file
        if isinstance(source, (str, Path)):
            source = (self._read_columnar_partition_file(source)
                      if Path(source).suffix.lower() == '.npz'
                      else self._read_partition_file(source))
        if 'soma_ids' in source:
            spec = self._spec_from_post_owned_columns(source)
        elif 'somas' in source:
            spec = self._spec_from_post_owned_records(
                source['somas'], source['synapses'], source.get('remote_ranks', {}),
                soma_breed, soma_config, synapse_breed, synapse_config)
        else:
            raise ValueError(
                "load_post_owned(): expected a path, a columnar array dict (with "
                "'soma_ids'), or a record dict (with 'somas'/'synapses'); got keys "
                f"{sorted(source)[:8]}.")
        self._build(spec)

    def create_from_lists(self, somas: list, synapses: list,
                          soma_breed: str = "lif_soma",
                          soma_config: str = "config_0",
                          synapse_breed: str = "single_exp_synapse",
                          synapse_config: str = "config_0") -> None:
        """Bulk-create the whole network from in-memory soma/synapse lists.

        Single-GPU bulk alternative to calling ``create_soma()`` /
        ``create_synapse()`` one at a time: hand over every soma and synapse as
        a list and the entire model is built in one shot. No file, no rank/remote
        concept — every soma is local on the single device.

        ::

            # one-by-one (incremental):
            a = model.create_soma(breed="lif_soma", config_name="config_0")
            b = model.create_soma(breed="lif_soma", config_name="config_0")
            model.create_synapse(pre_soma_id=-1, post_soma_id=a, ...)
            model.create_synapse(pre_soma_id=a,  post_soma_id=b, ...)

            # in bulk (equivalent network, one call):
            model.create_from_lists(
                somas=[{"id": 0}, {"id": 1}],
                synapses=[{"id": 10, "pre": -1, "post": 0},
                          {"id": 11, "pre": 0,  "post": 1}],
            )

        One-shot, whole-model builder: MUTUALLY EXCLUSIVE with the incremental
        ``create_soma``/``create_synapse`` and with the file loaders. Call
        exactly once on a fresh model.

        Entry-dict schema (the caller assigns every ``id``):
            somas:    [{"id", "breed"?, "config"?, "overrides"?}, ...]
            synapses: [{"id", "pre", "post", "breed"?, "config"?, "overrides"?,
                        "learning_rule"?, "learning_rule_config"?}, ...]
        ``pre = -1`` marks an external-input synapse (no pre-synaptic soma).
        Omitted ``breed``/``config`` fall back to the method defaults below.
        ``overrides`` is grouped by property type: "hyperparameters",
        "internal_states", "learning_hyperparameters", "learning_internal_states".

        :param somas: List of soma entry dicts.
        :param synapses: List of synapse entry dicts.
        :param soma_breed: Default soma breed (per-entry "breed" overrides it).
        :param soma_config: Default soma config name.
        :param synapse_breed: Default synapse breed name.
        :param synapse_config: Default synapse config name.
        """
        self._assert_unbuilt("create_from_lists")
        self._build(self._spec_from_post_owned_records(
            list(somas), list(synapses), {},
            soma_breed, soma_config, synapse_breed, synapse_config))

    def _combo_breed_index(self, combo) -> int:
        breeds = self._soma_breeds if combo.component_class == "soma" else self._synapse_breeds
        if combo.breed not in breeds:
            raise ValueError(
                f"{combo.component_class} breed {combo.breed!r} is not registered; "
                f"known: {sorted(breeds)}")
        return breeds[combo.breed]._breedidx

    def _combo_property_rows(self, combo) -> dict:
        """Property rows for one combo -- computed k times, not n times."""
        overrides = self._overrides_from_key(combo.overrides_key)
        if combo.component_class == "soma":
            hp, is_state = self._get_soma_properties(combo.breed, combo.config, overrides)
            return {
                "hyperparameters": hp,
                "internal_states": is_state,
                "output_spikes_tensor": [0.0, 0.0],
            }
        props, _hp, _lhp, _is, _ils = self._get_synapse_properties(
            combo.breed, combo.config, overrides,
            combo.learning_rule, combo.learning_rule_config or "default")
        return props

    def _build(self, spec: "BuildSpec") -> None:
        """Build the whole local model from a BuildSpec. The only bulk builder.

        Encoding-agnostic by construction: every reader (.pkl records, .npz columns,
        adjacency records) produces a BuildSpec, and the only thing they differ in is
        whether the neighbour CSR is *derived* from pre/post or *transcribed* verbatim.
        """
        import time
        timings = self._load_timings if isinstance(getattr(self, "_load_timings", None), dict) else {}
        _t = time.time()

        agent_ids = np.ascontiguousarray(spec.agent_ids, dtype=np.int64)
        codes = np.ascontiguousarray(spec.combo_codes, dtype=np.int32)
        offsets = np.ascontiguousarray(spec.nbr_offsets, dtype=np.int64)
        values = np.ascontiguousarray(spec.nbr_values, dtype=np.int64)
        n = agent_ids.size

        if codes.size != n:
            raise ValueError(f"combo_codes has {codes.size} entries for {n} agents")
        if offsets.size != n + 1:
            raise ValueError(f"nbr_offsets has {offsets.size} entries, expected {n + 1}")
        if n:
            uniq, counts = np.unique(agent_ids, return_counts=True)
            if uniq.size != n:
                dup = uniq[counts > 1]
                raise ValueError(
                    f"duplicate agent id(s) in this partition, e.g. {int(dup[0])} "
                    f"({dup.size} distinct ids repeat). Soma and synapse ids share one "
                    "namespace and must be unique.")

        # --- Breed order. build_from_local_columns requires non-decreasing breed
        # indices, and a prebuilt CSR cannot lean on setup's sort_by_breed (reordering
        # would desync the separately held CSR -- see AgentFactory._agents_prebreed_sorted).
        # So sort here, permuting ids, codes and the CSR together. Identity check first:
        # the common somas-then-synapses layout costs one np.diff. ---
        combo_breed = np.array(
            [self._combo_breed_index(c) for c in self._combo_table], dtype=np.int64)
        breed_indices = combo_breed[codes] if n else np.empty(0, dtype=np.int64)
        if n and np.any(np.diff(breed_indices) < 0):
            perm = np.argsort(breed_indices, kind="stable")
            row_counts = np.diff(offsets)[perm]
            new_off = np.empty(n + 1, dtype=np.int64)
            new_off[0] = 0
            np.cumsum(row_counts, out=new_off[1:])
            total = int(new_off[-1])
            gather = (np.repeat(offsets[perm], row_counts)
                      + (np.arange(total, dtype=np.int64)
                         - np.repeat(new_off[:-1], row_counts)))
            values = values[gather]
            offsets = new_off
            agent_ids = agent_ids[perm]
            codes = codes[perm]
            breed_indices = breed_indices[perm]

        timings.setdefault("assemble", 0.0)
        timings["assemble"] += time.time() - _t
        _t = time.time()

        # --- Property columns: one row per COMBO, one int32 code per agent. A property
        # a combo does not set falls back to its registered default, which is how somas
        # get the synapse-only columns and vice versa. ---
        default = self._agent_factory._property_name_2_defaults
        combo_rows = [self._combo_property_rows(c) for c in self._combo_table]
        names = set()
        for rows in combo_rows:
            names.update(rows)
        names -= {"breed", "locations"}

        property_columns = {}
        for name in sorted(names):
            rows = [[float(x) for x in r.get(name, default.get(name, []))]
                    for r in combo_rows]
            width = max((len(r) for r in rows), default=0)
            table = np.full((max(len(rows), 1), width), np.nan, dtype=np.float32)
            lengths = np.zeros(max(len(rows), 1), dtype=np.int32)
            for k, r in enumerate(rows):
                if r:
                    table[k, :len(r)] = r
                lengths[k] = len(r)
            property_columns[name] = IndexedColumn(table, codes, lengths)

        # History buffers stay dense (N, 1, w) zero arrays: they are 3-D and write-only,
        # and setup()'s _share_history_buffers re-fills them. lengths 0 reads back as [].
        for buf_name, src in (("internal_states_buffer", "internal_states"),
                              ("learning_internal_states_buffer", "learning_internal_states")):
            col = property_columns.get(src)
            w = int(col.lengths.max()) if col is not None and len(col.lengths) else 0
            property_columns[buf_name] = (np.zeros((n, 1, w), dtype=np.float32),
                                          np.zeros(n, dtype=np.int32))

        timings["property_columns"] = time.time() - _t
        _t = time.time()

        self.build_from_local_columns(
            agent_ids, breed_indices, property_columns,
            offsets.astype(np.int32), values, spec.remote_ranks)

        # Keep the CSR model-side too. SAGESim's set_prebuilt_csr deliberately CLEARS
        # space._locations (so a wrong-but-plausible empty neighbour list cannot leak),
        # which leaves get_location and any `locations` read broken for every bulk-built
        # model. get_neighbors() reads these instead, and unlike the old route through
        # get_agent_property_value it keeps working after reset() frees the GPU buffers.
        self._nbr_offsets = offsets
        self._nbr_values = values

        # --- Bookkeeping: codes by local row, and the vectorized id views callers use
        # instead of scanning a 50M-element set. ---
        self._combo_codes = codes
        self._agent_ids = agent_ids
        is_syn = np.array([c.component_class == "synapse" for c in self._combo_table],
                          dtype=bool)[codes] if n else np.zeros(0, dtype=bool)
        self._soma_ids_set = None          # materialise lazily, see the properties
        self._synapse_ids_set = None
        self._num_synapses = int(is_syn.sum())
        self._built_from_file = True

        timings["bookkeeping"] = time.time() - _t
        self._load_timings = timings

    @staticmethod
    def _post_owned_csr(soma_ids, synapse_ids, pre, post):
        """Derive the post-owned neighbour CSR. Shared by both post-owned readers.

        Row layout is the agent order [somas..., synapses...]. Soma i's neighbours are
        its incoming synapse ids (post == i) in synapse-array order; synapse j's are
        [pre] then [post] when post != -1 (positional slot 0 = pre, 1 = post).

        Enforces the post-owns constraint the loader is named for: every synapse's
        post-soma must be local, because a post-soma discovers its incoming synapses by
        scanning the synapses listed in its OWN partition.

        Neighbours are emitted VERBATIM -- no dedup. The record path used to route this
        through space.bulk_connect, which dedups in ordered mode, so an autapse
        (pre == post) silently lost positional slot 1 and the two builders produced
        different networks for the same input.
        """
        soma_ids = np.ascontiguousarray(soma_ids, dtype=np.int64)
        synapse_ids = np.ascontiguousarray(synapse_ids, dtype=np.int64)
        pre = np.ascontiguousarray(pre, dtype=np.int64)
        post = np.ascontiguousarray(post, dtype=np.int64)
        N, M = soma_ids.size, synapse_ids.size

        order = np.argsort(soma_ids, kind='stable')
        sorted_soma = soma_ids[order]
        has_post = post >= 0
        sp = np.searchsorted(sorted_soma, post)
        sp_clipped = np.clip(sp, 0, max(N - 1, 0))
        found = (sp < N) & (sorted_soma[sp_clipped] == post)
        if not np.all(found | ~has_post):
            bad = synapse_ids[has_post & ~found]
            raise ValueError(
                f"{bad.size} synapse(s) have a post-soma that is not local (e.g. "
                f"synapse {int(bad[0])} post {int(post[has_post & ~found][0])}); "
                "every synapse's post-soma must be co-located (post-owns). Use "
                "load_from_adjacency() to lift this constraint.")
        post_local = order[sp_clipped]

        counts_soma = np.bincount(post_local[has_post], minlength=N)
        soma_offsets = np.empty(N + 1, dtype=np.int64)
        soma_offsets[0] = 0
        np.cumsum(counts_soma, out=soma_offsets[1:])
        grp = np.argsort(post_local[has_post], kind='stable')
        soma_values = synapse_ids[has_post][grp]

        syn_counts = 1 + has_post.astype(np.int64)
        syn_offsets = np.empty(M + 1, dtype=np.int64)
        syn_offsets[0] = 0
        np.cumsum(syn_counts, out=syn_offsets[1:])
        syn_values = np.empty(int(syn_offsets[-1]), dtype=np.int64)
        syn_values[syn_offsets[:-1]] = pre
        syn_values[syn_offsets[:-1][has_post] + 1] = post[has_post]

        offsets = np.concatenate([soma_offsets, int(soma_offsets[-1]) + syn_offsets[1:]])
        values = np.concatenate([soma_values, syn_values]).astype(np.int64)
        return offsets, values

    def _spec_from_post_owned_records(self, somas, synapses, remote_ranks,
                                      soma_breed, soma_config,
                                      synapse_breed, synapse_config) -> "BuildSpec":
        """Row-oriented post-owned records (.pkl, or in-memory lists) -> BuildSpec.

        One Python pass over each list to pull out id / pre / post and intern the combo;
        everything after it is the same vectorized code the .npz reader runs. That pass
        is the only per-record work in the design and is irreducible for this encoding --
        the file IS a list of dicts -- but it is paid once at load, not on every GPU
        buffer rebuild.
        """
        import time
        timings = {}
        _t = time.time()

        n_s, n_y = len(somas), len(synapses)
        soma_ids = np.empty(n_s, dtype=np.int64)
        synapse_ids = np.empty(n_y, dtype=np.int64)
        pre = np.empty(n_y, dtype=np.int64)
        post = np.empty(n_y, dtype=np.int64)
        codes = np.empty(n_s + n_y, dtype=np.int32)

        intern = self._intern_combo
        for i, soma in enumerate(somas):
            soma_ids[i] = soma['id']
            codes[i] = intern("soma",
                              soma.get('breed') or soma_breed,
                              soma.get('config') or soma_config,
                              overrides=soma.get('overrides'))
        for j, syn in enumerate(synapses):
            synapse_ids[j] = syn['id']
            pre[j] = syn['pre']
            post[j] = syn['post']
            rule = syn.get('learning_rule') or None
            codes[n_s + j] = intern(
                "synapse",
                syn.get('breed') or synapse_breed,
                syn.get('config') or synapse_config,
                learning_rule=rule,
                learning_rule_config=(syn.get('learning_rule_config') or 'default')
                if rule else None,
                overrides=syn.get('overrides'))

        timings['property_columns'] = 0.0
        timings['records_to_columns'] = time.time() - _t
        _t = time.time()

        neighbor_offsets, neighbor_values_ids = self._post_owned_csr(
            soma_ids, synapse_ids, pre, post)
        timings['neighbor_csr'] = time.time() - _t

        self._input_synapse_ids = synapse_ids[pre < 0].astype(np.int64)
        self._load_timings = timings
        return BuildSpec(
            agent_ids=np.concatenate([soma_ids, synapse_ids]),
            combo_codes=codes,
            nbr_offsets=neighbor_offsets,
            nbr_values=neighbor_values_ids,
            remote_ranks=dict(remote_ranks or {}),
        )

    def _spec_from_adjacency_records(self, somas, synapses, remote_ranks,
                                     soma_breed, soma_config,
                                     synapse_breed, synapse_config) -> "BuildSpec":
        """Explicit-neighbour records -> BuildSpec.

        The one thing that genuinely differs from post-owned: the CSR is TRANSCRIBED
        from each entry's ``neighbors`` list, not derived from pre/post. That is what
        releases the post-owns constraint -- a post-soma names its incoming synapses
        explicitly, so one of them may live on another rank.

        Slot order is preserved verbatim and nothing is sorted or deduped: a synapse's
        list is positional (slot 0 = pre, slot 1 = post), so an autapse legitimately
        reads [x, x].
        """
        import time
        timings = {}
        _t = time.time()

        n_s, n_y = len(somas), len(synapses)
        n = n_s + n_y
        agent_ids = np.empty(n, dtype=np.int64)
        codes = np.empty(n, dtype=np.int32)
        counts = np.empty(n, dtype=np.int64)
        flat = []

        intern = self._intern_combo
        for i, soma in enumerate(somas):
            agent_ids[i] = soma['id']
            codes[i] = intern("soma",
                              soma.get('breed') or soma_breed,
                              soma.get('config') or soma_config,
                              overrides=soma.get('overrides'))
            nbrs = soma['neighbors']
            counts[i] = len(nbrs)
            flat.extend(nbrs)
        input_ids = []
        for j, syn in enumerate(synapses):
            k = n_s + j
            agent_ids[k] = syn['id']
            rule = syn.get('learning_rule') or None
            codes[k] = intern(
                "synapse",
                syn.get('breed') or synapse_breed,
                syn.get('config') or synapse_config,
                learning_rule=rule,
                learning_rule_config=(syn.get('learning_rule_config') or 'default')
                if rule else None,
                overrides=syn.get('overrides'))
            nbrs = syn['neighbors']
            counts[k] = len(nbrs)
            flat.extend(nbrs)
            if nbrs and nbrs[0] == -1:
                input_ids.append(syn['id'])

        offsets = np.empty(n + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(counts, out=offsets[1:])
        values = np.asarray(flat, dtype=np.int64) if flat else np.empty(0, dtype=np.int64)

        timings['records_to_columns'] = time.time() - _t
        self._input_synapse_ids = np.asarray(input_ids, dtype=np.int64)
        self._load_timings = timings
        return BuildSpec(
            agent_ids=agent_ids,
            combo_codes=codes,
            nbr_offsets=offsets,
            nbr_values=values,
            remote_ranks=dict(remote_ranks or {}),
        )

    def _spec_from_post_owned_columns(self, arrays: dict) -> "BuildSpec":
        """Columnar (.npz) post-owned arrays -> BuildSpec.

        The producer stores breed / config / learning rule as SCALARS because the
        network it describes is uniform. Those are read here as one-entry categorical
        axes -- a scalar is the k = 1 case of a combo table -- so a future producer that
        varies them per agent needs a new reader, not a new builder. The synapse
        hyperparameter override matrix is the axis that already varies, and np.unique
        turns it into combos and codes directly.
        """
        import time
        timings = {}
        _t = time.time()

        def _scalar(name):
            return np.asarray(arrays[name]).item()

        soma_ids = np.ascontiguousarray(arrays['soma_ids'], dtype=np.int64)
        synapse_ids = np.ascontiguousarray(arrays['synapse_ids'], dtype=np.int64)
        pre = np.ascontiguousarray(arrays['pre'], dtype=np.int64)
        post = np.ascontiguousarray(arrays['post'], dtype=np.int64)
        soma_breed = _scalar('soma_breed')
        soma_config = _scalar('soma_config')
        synapse_breed = _scalar('synapse_breed')
        synapse_config = _scalar('synapse_config')
        lr_name = _scalar('learning_rule')
        learning_rule = None if lr_name == '' else lr_name
        learning_rule_config = _scalar('learning_rule_config') if learning_rule else None
        hp_keys = [str(k) for k in np.asarray(arrays['syn_hp_keys'])]
        hp_vals = np.ascontiguousarray(arrays['syn_hp_vals'], dtype=np.float64)
        remote_ids = np.asarray(arrays.get('remote_ids', np.empty(0, np.int64)))
        remote_rank_of = np.asarray(arrays.get('remote_rank_of', np.empty(0, np.int64)))

        N, M = len(soma_ids), len(synapse_ids)
        timings['decode_arrays'] = time.time() - _t
        _t = time.time()

        timings['constraint_check'] = 0.0

        # --- Combos: one for the (uniform) somas, one per distinct synapse override row.
        soma_code = self._intern_combo("soma", soma_breed, soma_config, overrides={})
        combo_codes = np.empty(N + M, dtype=np.int32)
        combo_codes[:N] = soma_code
        if M:
            distinct, inverse = np.unique(hp_vals, axis=0, return_inverse=True)
            syn_codes = np.array([
                self._intern_combo(
                    "synapse", synapse_breed, synapse_config,
                    learning_rule=learning_rule,
                    learning_rule_config=learning_rule_config,
                    overrides={"hyperparameters": dict(zip(hp_keys, row))})
                for row in distinct
            ], dtype=np.int32)
            combo_codes[N:] = syn_codes[np.asarray(inverse, dtype=np.intp).ravel()]

        neighbor_offsets, neighbor_values_ids = self._post_owned_csr(
            soma_ids, synapse_ids, pre, post)

        timings['neighbor_csr'] = time.time() - _t

        self._input_synapse_ids = synapse_ids[pre < 0].astype(np.int64)
        self._load_timings = timings
        return BuildSpec(
            agent_ids=np.concatenate([soma_ids, synapse_ids]),
            combo_codes=combo_codes,
            nbr_offsets=neighbor_offsets,
            nbr_values=neighbor_values_ids,
            remote_ranks={int(i): int(r) for i, r in zip(remote_ids, remote_rank_of)},
        )

    def load_from_adjacency(self, partition_file: str,
                            soma_breed: str = "lif_soma",
                            soma_config: str = "config_0",
                            synapse_breed: str = "single_exp_synapse",
                            synapse_config: str = "config_0") -> None:
        """Load an EXPLICIT-NEIGHBORS network file (Method 2) and build the model.

        The producer supplies each agent's neighbor list DIRECTLY — soma AND
        synapse — and this loader reads them verbatim. This RELEASES the post-owns
        constraint of ``load_post_owned()``: because a post-soma's incoming
        synapses are named explicitly in its ``neighbors`` (not derived by
        scanning local synapses), an incoming synapse may live on another rank.
        Declare any such cross-rank neighbor in ``remote_ranks`` and SAGESim's
        ghost exchange delivers its ``internal_states`` each tick — the same
        machinery that already serves a synapse's remote pre-soma.

        Neighbor-list slot order is preserved VERBATIM (never sorted/deduped):
          - synapse ``neighbors`` is POSITIONAL: ``[pre]`` or ``[pre, post]``
            (slot 0 = pre, read for the incoming spike; slot 1 = post, for STDP);
            ``pre = -1`` occupies slot 0 for an external-input synapse.
          - soma ``neighbors`` is its incoming synapse ids, order-free.

        One-shot, whole-model builder; MUTUALLY EXCLUSIVE with create_soma/
        create_synapse and with ``load_post_owned()``.

        File schema (a dict with 2 required keys + 1 optional):
            {
              "somas":    [{"id", "breed", "config", "overrides",
                            "neighbors": [incoming_syn_id, ...]}, ...],
              "synapses": [{"id", "breed", "config", "learning_rule",
                            "learning_rule_config", "overrides",
                            "neighbors": [pre[, post]]}, ...],
              "remote_ranks": {agent_id: rank}   # optional; any cross-rank id
            }

        :param partition_file: Path to network file (.pkl)
        :param soma_breed: Default soma breed name
        :param soma_config: Default soma config name
        :param synapse_breed: Default synapse breed name
        :param synapse_config: Default synapse config name
        """
        self._assert_unbuilt("load_from_adjacency")
        data = self._read_neighbor_partition_file(partition_file)

        # Validate the partition BEFORE building anything (cheap, local). These are
        # the "Bug A" guards docs/PARTITION_LOADING.md §3.3 endorses; completeness
        # (Bug B — a soma missing some incoming synapse) is impossible to see
        # locally and remains the producer's job.
        remote_ids = data['remote_ranks']
        local_ids = ({s['id'] for s in data['somas']}
                     | {s['id'] for s in data['synapses']})
        for syn in data['synapses']:
            nbrs = syn['neighbors']
            if not 1 <= len(nbrs) <= 2:
                raise ValueError(
                    f"load_from_adjacency(): synapse {syn['id']} has "
                    f"{len(nbrs)} neighbors {list(nbrs)}; a synapse must have "
                    "exactly [pre] (external input) or [pre, post]."
                )
        # Every neighbor id must be the external-input sentinel -1, a local agent,
        # or declared remote. A ref that is none of these would be silently skipped
        # by the kernels (e.g. lif.py guards synapse_index >= 0).
        for entry in (*data['somas'], *data['synapses']):
            for nb in entry['neighbors']:
                if nb == -1 or nb in local_ids or nb in remote_ids:
                    continue
                raise ValueError(
                    f"load_from_adjacency(): agent {entry['id']} references "
                    f"neighbor {nb}, which is neither a local agent nor named in "
                    "remote_ranks. The producer must list every cross-rank "
                    "neighbor in remote_ranks."
                )

        # A soma's incoming list is a set of distinct synapses; a repeat would double
        # count that synapse's current every tick. The old path routed through
        # space.bulk_connect, which deduped in ordered mode and so silently REPAIRED a
        # producer bug -- while contradicting this loader's documented "never
        # sorted/deduped" promise. Now the list is transcribed verbatim, so say so.
        # Synapses are exempt: their [pre, post] is positional and an autapse is [x, x].
        for soma in data['somas']:
            nbrs = soma['neighbors']
            if len(set(nbrs)) != len(nbrs):
                dupes = sorted({n for n in nbrs if list(nbrs).count(n) > 1})
                raise ValueError(
                    f"load_from_adjacency(): soma {soma['id']} lists incoming "
                    f"synapse(s) {dupes} more than once. A repeated incoming synapse "
                    "would have its current counted twice every tick.")

        # Neighbour lists are read VERBATIM: positional slot order is load-bearing for
        # synapses (slot 0 = pre, slot 1 = post) and soma lists are copied the same way.
        self._build(self._spec_from_adjacency_records(
            data['somas'], data['synapses'], dict(remote_ids),
            soma_breed, soma_config, synapse_breed, synapse_config))

    # ------------------------------------------------------------------
    # External input spikes: a tick-major event list delivered by the kernel
    # ------------------------------------------------------------------
    #
    # Injected spikes are not stored per synapse. They accumulate host-side as
    # (synapse_id, tick, value) chunks and are compiled, when the kernel next
    # launches, into three device arrays sorted by tick:
    #     ev_offsets[t] .. ev_offsets[t + 1]   the events of tick t
    #     ev_syn[k]                           local row of the target synapse
    #     ev_val[k]                           summed value
    # At the top of every tick the generated kernel scatters that tick's events
    # into each target's `input_spikes_tensor` row as [tick, value] (see
    # _get_extra_kernel_config) and get_soma_spike reads the row back with one
    # comparison. Per-tick cost is the number of events at that tick, so run
    # length does not matter, and injecting between simulate() calls does not
    # touch the property buffers, so nothing the kernels learned is lost.

    _MAX_INPUT_TICK = 1 << 24   # ticks are stored in float32 on the device

    def add_spike(self, synapse_id: int, tick: int, value: float) -> None:
        """
        Schedules an external input spike on a synapse whose ``pre_soma_id`` is -1.

        :param tick: tick at which the spike arrives (absolute simulation tick)
        :param value: spike value; several spikes on one tick are summed
        """
        self._append_input_events([synapse_id], [tick], [value])

    def add_local_spike(self, synapse_id: int, tick: int, value: float) -> None:
        """Schedule an input spike on a LOCALLY-OWNED synapse.

        Same effect as add_spike (injection is rank-local either way: each rank keeps
        only the spikes of synapses it owns and drops the rest when the event list is
        built), but this variant checks ownership eagerly and raises KeyError for a
        synapse this rank does not own, as the distributed API has always done.
        """
        self._append_input_events([synapse_id], [tick], [value], local=True)

    def add_spike_list(self, synapse_id: int, spike_list) -> None:
        """
        Schedules a list of external input spikes on one synapse.

        :param spike_list: ``[[tick, value], ...]`` or an ``(N, 2)`` array
        """
        pairs = np.asarray(spike_list, dtype=np.float64).reshape(-1, 2)
        self._append_input_events(
            np.full(len(pairs), synapse_id, dtype=np.int64), pairs[:, 0], pairs[:, 1])

    def add_local_spike_list(self, synapse_id: int, spike_list) -> None:
        """Local counterpart of add_spike_list (see add_local_spike)."""
        pairs = np.asarray(spike_list, dtype=np.float64).reshape(-1, 2)
        self._append_input_events(
            np.full(len(pairs), synapse_id, dtype=np.int64), pairs[:, 0], pairs[:, 1],
            local=True)

    def add_spikes(self, synapse_ids, ticks, values=1.0) -> None:
        """
        Bulk injection in the shape of Brian2's ``SpikeGeneratorGroup``: flat
        arrays of synapse ids and ticks (any order), optionally per-spike values.
        This is the fast path for long experiments: one call for millions of spikes.
        """
        ids = np.asarray(synapse_ids, dtype=np.int64).ravel()
        ticks = np.asarray(ticks, dtype=np.float64).ravel()
        vals = np.broadcast_to(np.asarray(values, dtype=np.float64), ids.shape).ravel()
        if len(ticks) != len(ids):
            raise ValueError("synapse_ids and ticks must have the same length")
        self._append_input_events(ids, ticks, vals)

    def get_input_spikes(self, synapse_id: int):
        """Return ``(ticks, values)`` injected on ``synapse_id`` from this rank's
        host store, sorted by tick. Events set with set_input_events are device-only
        and not reported here."""
        ticks, vals = [], []
        for ids_c, ticks_c, vals_c in self._input_events:
            m = ids_c == synapse_id
            if m.any():
                ticks.append(ticks_c[m]); vals.append(vals_c[m])
        if not ticks:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
        t = np.concatenate(ticks); v = np.concatenate(vals)
        order = np.argsort(t, kind="stable")
        return t[order], v[order]

    def clear_input_spikes(self) -> None:
        """Discard every pending input spike (host store and device event list)."""
        self._input_events = []
        self._input_events_gpu = None
        self._input_events_dirty = True

    def set_input_events(self, syn_rows, ticks, values=None) -> None:
        """Low-level, device-side replacement of ALL pending input spikes.

        ``syn_rows`` are local GPU buffer row indices (``_gpu_buffers.agent_id_to_index``),
        ``ticks`` absolute ticks, ``values`` per-event values (default 1.0); CuPy or numpy
        arrays, compiled on the device without a host round trip. Meant for drivers that
        generate their input on the GPU per presentation (Diehl & Cook). Replaces anything
        added with add_spike*.
        """
        import cupy as cp
        rows = cp.asarray(syn_rows).astype(cp.int64).ravel()
        tks = cp.asarray(ticks).astype(cp.int64).ravel()
        if values is None:
            vals = cp.ones(rows.size, dtype=cp.float64)
        else:
            vals = cp.broadcast_to(cp.asarray(values, dtype=cp.float64), rows.shape).ravel()
        if rows.size != tks.size:
            raise ValueError("syn_rows and ticks must have the same length")
        self._input_events = []
        self._input_events_gpu = self._build_event_list(cp, rows, tks, vals)
        self._input_events_dirty = False

    # -- internals -----------------------------------------------------------

    def _append_input_events(self, ids, ticks, values, local: bool = False) -> None:
        ids = np.asarray(ids, dtype=np.int64).ravel()
        ticks = np.asarray(ticks, dtype=np.float64).ravel()
        vals = np.asarray(values, dtype=np.float64).ravel()
        if not (len(ids) == len(ticks) == len(vals)):
            raise ValueError("ids, ticks and values must have the same length")
        if len(ids) == 0:
            return
        # On one rank every agent is local, so an unknown id is an error right away.
        # Under MPI the collective idiom (every rank calls with the same ids) must
        # tolerate ids owned elsewhere; only the add_local_* variants insist.
        if local or MPI.COMM_WORLD.Get_size() == 1:
            owned = self._agent_factory._rank2agentid2agentidx.get(MPI.COMM_WORLD.Get_rank(), {})
            unknown = [i for i in np.unique(ids).tolist() if i not in owned]
            if unknown:
                raise KeyError(f"synapse id(s) not owned by this rank: {unknown[:5]}")
        if not np.all(ticks == np.floor(ticks)):
            raise ValueError("input spike ticks must be integers")
        if ticks.min() < 0 or ticks.max() >= self._MAX_INPUT_TICK:
            raise ValueError(f"input spike ticks must lie in [0, {self._MAX_INPUT_TICK})")
        if not np.all(np.isfinite(vals)):
            raise ValueError("input spike values must be finite")
        self._input_events.append(
            (ids, ticks.astype(np.int64), vals.astype(np.float32)))
        self._input_events_dirty = True

    @staticmethod
    def _build_event_list(xp, rows, ticks, vals):
        """Aggregate duplicate (tick, row) pairs, sort by tick, build tick offsets.
        ``xp`` is numpy or cupy; arrays are int64 rows, int64 ticks, float64 values."""
        if rows.size == 0:
            return (xp.zeros(1, dtype=xp.int32), xp.zeros(1, dtype=xp.int32),
                    xp.zeros(1, dtype=xp.float32))
        n_rows = int(rows.max()) + 1
        key = ticks * n_rows + rows                     # sorts by tick, then row
        ukey, inv = xp.unique(key, return_inverse=True)
        summed = xp.bincount(inv.ravel(), weights=vals, minlength=len(ukey))
        ev_tick = ukey // n_rows
        max_tick = int(ev_tick[-1])
        offsets = xp.zeros(max_tick + 2, dtype=xp.int64)
        offsets[1:] = xp.cumsum(xp.bincount(ev_tick, minlength=max_tick + 1))
        return (offsets.astype(xp.int32), (ukey % n_rows).astype(xp.int32),
                summed.astype(xp.float32))

    def _compile_input_events(self, buf) -> None:
        """Turn the host store into device arrays (once per change)."""
        import cupy as cp
        if not self._input_events_dirty and self._input_events_gpu is not None:
            return
        self._input_events_dirty = False
        if not self._input_events:
            if self._input_events_gpu is None:
                self._input_events_gpu = self._build_event_list(
                    np, np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0))
                self._input_events_gpu = tuple(cp.asarray(a) for a in self._input_events_gpu)
            return
        ids = np.concatenate([c[0] for c in self._input_events])
        ticks = np.concatenate([c[1] for c in self._input_events])
        vals = np.concatenate([c[2] for c in self._input_events]).astype(np.float64)
        # Resolve ids -> buffer rows once per distinct id, not once per event.
        uniq, inverse = np.unique(ids, return_inverse=True)
        owned = self._agent_factory._rank2agentid2agentidx.get(MPI.COMM_WORLD.Get_rank(), {})
        rows_u = np.fromiter(
            (buf.agent_id_to_index.get(i, -1) if i in owned else -1 for i in uniq.tolist()),
            dtype=np.int64, count=len(uniq))
        if MPI.COMM_WORLD.Get_size() == 1 and (rows_u < 0).any():
            raise KeyError(f"add_spike*: unknown synapse id(s) {uniq[rows_u < 0][:5].tolist()}")
        rows = rows_u[inverse.ravel()]
        keep = rows >= 0                                # other ranks' synapses
        off, syn, val = self._build_event_list(np, rows[keep], ticks[keep], vals[keep])
        self._input_events_gpu = (cp.asarray(off), cp.asarray(syn), cp.asarray(val))

    # ------------------------------------------------------------------
    # GPU kernel extension hooks for spike recording
    # ------------------------------------------------------------------

    def _get_extra_kernel_config(self) -> dict:
        prop_idx = self._agent_factory._property_name_2_index["output_spikes_tensor"]
        in_idx = self._agent_factory._property_name_2_index["input_spikes_tensor"]
        return {
            'extra_kernel_params': ['spike_record', 'spike_record_count', 'spike_mask',
                                    'n_spike_slots',
                                    'ev_offsets', 'ev_syn', 'ev_val', 'n_ev_offsets'],
            # The delivery code below writes this property; the framework's write
            # analysis cannot see generated code, and a property any kernel writes must
            # never be interned into a shared table.
            'writes_properties': ['input_spikes_tensor'],
            # Deliver this tick's external input events: all threads stride over
            # ev_syn[ev_offsets[t]:ev_offsets[t+1]] and stamp [tick, value] into the
            # target synapse rows. Every thread reads the same offsets, so the branch
            # is uniform and the barrier is only paid on ticks that have events.
            'pre_tick_code': [
                'if thread_local_tick + 1 < int(n_ev_offsets):',
                '\t_e0 = int(ev_offsets[thread_local_tick])',
                '\t_e1 = int(ev_offsets[thread_local_tick + 1])',
                '\tif _e1 > _e0:',
                '\t\t_k = _e0 + int(thread_id)',
                '\t\twhile _k < _e1:',
                f'\t\t\ta{in_idx}[int(ev_syn[_k])][0] = 1.0 * thread_local_tick',
                f'\t\t\ta{in_idx}[int(ev_syn[_k])][1] = ev_val[_k]',
                '\t\t\t_k = _k + int(total_threads)',
                '\t\t__GRID_BARRIER__',
            ],
            'post_breed_step_code': [
                (
                    [
                        f'_sv = a{prop_idx}[_real_idx][thread_local_tick % 2]',
                        'if _sv > 0.0 and spike_mask[_real_idx] > 0.0:',
                        '\t_slot = jit.atomic_add(spike_record_count, 0, 1)',
                        # never write past the record; the host reads the count and
                        # reports an overflow instead of corrupting device memory
                        '\tif _slot < int(n_spike_slots):',
                        '\t\tspike_record[_slot * 2] = agent_ids[_real_idx]',
                        '\t\tspike_record[_slot * 2 + 1] = float(thread_local_tick)',
                    ],
                    True,  # once_per_breed
                    0,     # only_priority — only emit for soma priority
                ),
            ],
        }

    def _prepare_kernel_extras(self, num_local_agents, sync_ticks):
        import cupy as cp
        # Size the record for THIS launch (1 % of agents firing per tick on average, at
        # least 10k spikes); grow when a longer launch needs more. Overflow is caught by
        # the guard in the recording code and reported by _process_kernel_extras.
        max_slots = max(10000, num_local_agents * sync_ticks // 100)
        if self._spike_record_gpu is None or self._spike_record_gpu.size < max_slots * 2:
            self._spike_record_gpu = cp.full(max_slots * 2, cp.nan, dtype=cp.float32)
        if self._spike_record_count_gpu is None:
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
        self._compile_input_events(self._gpu_buffers)
        # A driver that rewinds the clock (Diehl & Cook sets model.tick = 0 before
        # every presentation) would otherwise see last presentation's stamps
        # [tick, value] match again on the same tick numbers; clear them.
        if self.tick < self._input_next_tick:
            in_idx = self._agent_factory._property_name_2_index["input_spikes_tensor"]
            self._gpu_buffers.property_tensors[in_idx][:, 0] = -1.0
        self._input_next_tick = self.tick + sync_ticks
        ev_offsets, ev_syn, ev_val = self._input_events_gpu
        return (self._spike_record_gpu, self._spike_record_count_gpu, self._spike_mask_gpu,
                cp.int32(self._spike_record_gpu.size // 2),
                ev_offsets, ev_syn, ev_val, cp.int32(ev_offsets.size))

    def _process_kernel_extras(self):
        count = int(self._spike_record_count_gpu[0].get())
        capacity = self._spike_record_gpu.size // 2
        if count > capacity:
            raise RuntimeError(
                f"spike record overflow: {count:,} spikes in one simulate() call but room for "
                f"{capacity:,} (sized as max(10000, agents * ticks / 100)). Call simulate() "
                f"in shorter pieces or record fewer somas (set_recorded_somas).")
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

