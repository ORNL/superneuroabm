"""
Model class for building an SNN
"""

from collections import defaultdict, namedtuple
from collections.abc import Mapping
from typing import Dict, List, Set
from pathlib import Path
import array
import ast
import inspect
import textwrap

import numpy as np
import cupy as cp
from sagesim.space import NetworkSpace
from sagesim.model import Model
from sagesim.columns import ArrayColumn, IndexedColumn
from sagesim.agent_index import rows_of
from sagesim.breed import Breed

from superneuroabm.step_functions.soma.izh import izh_soma_step_func
from superneuroabm.step_functions.soma.lif import lif_soma_step_func
from superneuroabm.step_functions.soma.lif_soma_adaptive_thr import lif_soma_adaptive_thr_step_func
from superneuroabm.step_functions.soma.hg_lif import hg_lif_soma_step_func
from superneuroabm.step_functions.soma.adex import adex_soma_step_func
from superneuroabm.step_functions.synapse.delivery import delivery_synapse_step_func
from superneuroabm.util import load_component_configurations


#: A KIND of agent: what many agents share. The model stores each kind's property rows
#: once and one int32 code per agent. Numbers given for a single agent (its weight, say)
#: are values, not kinds: they fill slots of that agent's row (see _value_column), so a
#: model where every synapse has its own weight still has a handful of kinds.
Combo = namedtuple(
    "Combo",
    "component_class breed config learning_rule learning_rule_config",
)


class _ComboField(Mapping):
    """Read-only ``agent_id -> field`` view over the model's combo store.

    Replaces the ``agentid2config`` / ``agentid2overrides`` / ``agentid2learning_rule``
    dicts. Those were three entries per agent; this is one int32 code per agent plus a
    table with one row per kind. Kept as a Mapping so existing readers
    (``superneuroabm/util.py:93``, user code) are unchanged. ``get`` receives the
    agent's kind and its id.
    """

    __slots__ = ("_model", "_get")

    def __init__(self, model, get):
        self._model = model
        self._get = get

    def __getitem__(self, agent_id):
        combo = self._model._combo_of(agent_id)
        if combo is None:
            raise KeyError(agent_id)
        return self._get(combo, agent_id)

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
        "adex_soma": (adex_soma_step_func, CURRENT_DIR_ABSPATH / "step_functions" / "soma" / "adex.py"),
    }


def _default_synapse_breeds():
    # One built-in synapse: it delivers spike * scale * weight into a channel of the
    # post soma's inbox; the synaptic kinetics (time constants) live on the soma.
    return {
        "delivery_synapse": (delivery_synapse_step_func, CURRENT_DIR_ABSPATH / "step_functions" / "synapse" / "delivery.py"),
    }


# Synapse kinds with no per-tick state, so they may be event-woken (synapse_stepping="event"):
# stepped only on the tick after their pre fired or an input event arrived.
EVENT_CAPABLE_SYNAPSES = {"delivery_synapse"}
# Built-in learning rules that catch their traces up on wake (trace_decay + last_tick slot),
# so the learning-rule selector can be event-woken by pre or post spikes.
EVENT_CAPABLE_RULES = {"exp_pair_wise_stdp", "exp_pair_wise_stdp_bounded",
                       "exp_pair_wise_stdp_quantized", "exp_pair_wise_stdp_memristive",
                       "triplet_stdp", "suppression_stdp"}


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
        4: {
            "func_name": "triplet_stdp",
            "import_line": "from superneuroabm.step_functions.synapse.stdp.triplet_stdp import triplet_stdp",
        },
        5: {
            "func_name": "suppression_stdp",
            "import_line": "from superneuroabm.step_functions.synapse.stdp.suppression_stdp import suppression_stdp",
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


def _unique_rows(mat):
    """``np.unique(mat, axis=0, return_inverse=True)`` for a 2-D numeric matrix, fast.

    Same distinct rows in the same (lexicographic) order, from a 1-D unique per column
    combined into one int64 key. ``axis=0`` sorts the rows as structured records, which
    took 7.4 s for Brunel's 12.5 M x 2 override matrix; this takes 0.5 s. When the next
    column would overflow the key, the key is first re-compressed to dense codes (order
    kept, at most ``n`` of them), so any number of columns stays on this path; and once
    every row is distinct the remaining columns can neither merge nor reorder rows, so
    they are not read (all-distinct parameters cost one column's unique).
    """
    mat = np.asarray(mat)
    n, k = mat.shape
    if k == 0:
        return np.empty((1 if n else 0, 0), dtype=mat.dtype), np.zeros(n, dtype=np.intp)
    uniques, key, span = [], np.zeros(n, dtype=np.int64), 1
    compressed = dense = False            # dense: key is exactly the codes 0..span-1
    for j in range(k):                    # column 0 most significant = lexicographic order
        u, inv = np.unique(mat[:, j], return_inverse=True)
        if span * len(u) > np.iinfo(np.int64).max:
            distinct_keys, key = np.unique(key, return_inverse=True)
            key, span, compressed = key.ravel().astype(np.int64), len(distinct_keys), True
            if span == n:
                dense = True
                break
        span *= len(u)
        uniques.append(u)
        key = key * len(u) + inv.ravel()
        if j == 0 and span == n:
            dense = True
            break
    if dense:
        inverse, n_distinct = key, span
    else:
        distinct_keys, inverse = np.unique(key, return_inverse=True)
        inverse, n_distinct = inverse.ravel(), len(distinct_keys)
    if dense or compressed:
        # the key no longer spells every column: take each distinct row from one of the
        # rows it stands for (they compare equal)
        rep = np.empty(n_distinct, dtype=np.intp)
        rep[inverse] = np.arange(n)
        return mat[rep], inverse
    distinct = np.empty((n_distinct, k), dtype=mat.dtype)
    rest = distinct_keys.copy()
    for j in range(k - 1, -1, -1):
        rest, digit = np.divmod(rest, len(uniques[j]))
        distinct[:, j] = uniques[j][digit]
    return distinct, inverse


SYNAPTIC_INPUT_PARAMS = ("synaptic_input", "synaptic_input_idx")


def _synaptic_input_arrays(step_func) -> List[str]:
    """Breed-local arrays a soma/synapse kernel declares: ["synaptic_input"] when its
    parameter list ends with the inbox pair, else [] (a custom soma that reads its
    afferents' state directly). See superneuroabm/step_functions/synaptic_input.py."""
    params = list(inspect.signature(step_func).parameters)
    return ["synaptic_input"] if tuple(params[-2:]) == SYNAPTIC_INPUT_PARAMS else []


def _reads_neighbours(step_func, n_globals: int) -> bool:
    """True when a kernel's body uses its neighbour-list parameter (``locations``: the one
    after tick, agent_index, the globals, agent_ids and breeds). Any use counts, also
    handing it to a helper, and a kernel whose source cannot be read counts as a reader:
    the answer errs towards building the list."""
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(step_func)))
    except (OSError, TypeError, SyntaxError):
        return True
    func = next((node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)), None)
    index = 2 + n_globals + 2
    if func is None or len(func.args.args) <= index:
        return True
    name = func.args.args[index].arg
    return any(isinstance(node, ast.Name) and node.id == name for node in ast.walk(func))


class _Rows:
    """A growing table of numeric columns: numpy chunks from the array calls plus one
    ``array.array`` per column for the one-at-a-time calls. The per-item buffers are
    flushed into a chunk whenever an array call or a read arrives, so the chunks stay in
    call order."""

    __slots__ = ("dtypes", "chunks", "one")

    _DTYPES = {"q": np.int64, "i": np.int32, "d": np.float64}

    def __init__(self, *typecodes):
        self.dtypes = [np.dtype(self._DTYPES[t]) for t in typecodes]
        self.chunks = []
        self.one = [array.array(t) for t in typecodes]

    def flush(self) -> None:
        if len(self.one[0]):
            self.chunks.append(tuple(np.frombuffer(buf, dtype=dt).copy()
                                     for buf, dt in zip(self.one, self.dtypes)))
            for buf in self.one:
                del buf[:]

    def add_block(self, *columns) -> None:
        """Columns of one array call. Copied: the caller's arrays stay the caller's."""
        self.flush()
        self.chunks.append(tuple(np.array(c, dtype=dt, copy=True)
                                 for c, dt in zip(columns, self.dtypes)))

    def columns(self) -> tuple:
        self.flush()
        if not self.chunks:
            return tuple(np.empty(0, dtype=dt) for dt in self.dtypes)
        if len(self.chunks) > 1:
            self.chunks = [tuple(np.concatenate(col) for col in zip(*self.chunks))]
        return self.chunks[0]

    def __len__(self) -> int:
        return len(self.one[0]) + sum(len(chunk[0]) for chunk in self.chunks)


class _Values:
    """The values given for one parameter (a property and a key in it), by creation
    ordinal. An array call covers a run of consecutive ordinals and stores only where the
    run starts; one-at-a-time calls store (ordinal, value) pairs."""

    __slots__ = ("entries", "rows")

    def __init__(self):
        self.entries = []                 # (first ordinal, values) or (ordinals, values)
        self.rows = _Rows("q", "d")

    def add_one(self, seq: int, value: float) -> None:
        self.rows.one[0].append(seq)
        self.rows.one[1].append(value)

    def add_run(self, first_seq: int, values) -> None:
        self._flush()
        self.entries.append((int(first_seq), np.array(values, dtype=np.float64, copy=True)))

    def _flush(self) -> None:
        if len(self.rows):
            self.entries.append(self.rows.columns())
            self.rows.chunks = []          # the per-item arrays themselves stay (cached appends)

    def __iter__(self):
        self._flush()
        return iter(self.entries)


class _NetworkDraft:
    """What the construction calls have described so far: flat arrays, no agents.

    ``create_soma`` / ``create_synapse`` append one row, ``create_somas`` /
    ``create_synapses`` a block, ``add_neighbors`` pairs, ``load_network`` a rank's part.
    NeuromorphicModel._build_from_draft turns it into SAGESim columns in one pass.
    ``seq`` is an agent's creation ordinal (somas and synapses share the count): it fixes
    the order of agents, addresses the values given for them, and deals somas to ranks.
    """

    def __init__(self):
        self.n_created = 0            # creation ordinal of the next agent
        self.next_id = 0              # next automatic agent id
        self.explicit_ids = False     # some id was chosen by the caller: check uniqueness
        self.somas = _Rows("q", "i", "q")                  # id, kind code, seq
        self.synapses = _Rows("q", "q", "q", "i", "q")     # id, pre, post, kind code, seq
        self.extras = _Rows("q", "q")                      # agent id, neighbour id
        self.values = {}                                   # (property, key) -> _Values


def _positions_in_group(group: np.ndarray) -> np.ndarray:
    """For each element, how many earlier elements have the same key (0, 1, 2, ...)."""
    n = len(group)
    order = np.argsort(group, kind="stable")
    sorted_group = group[order]
    first = np.ones(n, dtype=bool)
    first[1:] = sorted_group[1:] != sorted_group[:-1]
    run_start = np.maximum.accumulate(np.where(first, np.arange(n), 0))
    positions = np.empty(n, dtype=np.int64)
    positions[order] = np.arange(n) - run_start
    return positions


class NeuromorphicModel(Model):
    def __init__(
        self,
        soma_breed_info=None,
        synapse_breed_info=None,
        learning_rule_info=None,
        user_config=None,
        enable_internal_states_tracking: bool = True,
        synapse_stepping: str = "event",
        synaptic_channels: int = 2,
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
        :param synapse_stepping: "event" (default): a synapse is stepped only on the
            tick after its pre-soma fired or an injected input event arrived, so the
            synapse phase costs O(spikes x fan-out) instead of O(synapses); learning
            rules are woken by their pre or post spike and must be event-capable
            (decay-on-wake traces, see the built-in rules or
            register_learning_rule(..., event_capable=True)); a custom synapse type
            needs register_synapse_type(..., event_capable=True). "clocked": every
            synapse is stepped every tick (for verification, or for a custom synapse
            type with continuous per-tick state).
        :param synaptic_channels: width of the soma inbox: the number of distinct
            synaptic time constants one soma can receive. A delivery_synapse picks a
            channel with its ``channel`` hyperparameter and the soma filters channel c
            with ``tau_syn_c`` (NEST's receptor ports). Default 2.
        """
        super().__init__(space=NetworkSpace(ordered=True),
                         agent_slack_factor=1.0, csr_slack_factor=1.0)

        if synapse_stepping not in ("clocked", "event"):
            raise ValueError(f"synapse_stepping must be 'clocked' or 'event', got {synapse_stepping!r}")
        self._synapse_stepping = synapse_stepping
        if int(synaptic_channels) < 1 or int(synaptic_channels) != synaptic_channels:
            raise ValueError(f"synaptic_channels must be a positive int, got {synaptic_channels!r}")
        self._synaptic_channels = int(synaptic_channels)

        if soma_breed_info is None:
            soma_breed_info = _default_soma_breeds()
        if synapse_breed_info is None:
            synapse_breed_info = _default_synapse_breeds()

        self.enable_internal_states_tracking = enable_internal_states_tracking
        self._config_list_cache = {}

        self.register_global_property("dt", 1e-3)      # Time step (100 μs)
        self.register_global_property("I_bias", 0)     # No bias current

        # Load and hold configurations (needed before property dicts are built)
        self.agentid2config = _ComboField(self, lambda c, _id: c.config)
        if user_config is not None:
            self._component_configurations = load_component_configurations(user_config)
        else:
            self._component_configurations = load_component_configurations()

        max_sizes = _compute_max_property_sizes(self._component_configurations)

        # Separate learning rule configs before building property dicts
        self._learning_rule_configurations = self._component_configurations.pop("learning_rule", {})

        # What each agent IS: one int32 code per agent into a table of kinds.
        # `agentid2config` / `agentid2overrides` / `agentid2learning_rule` are read-only
        # Mapping views over this (see _ComboField).
        self._combo_table = []          # list[Combo]: the kinds
        self._combo_index = {}          # kind key -> code
        self._kind_keys = []            # per kind: {property: [key names]}
        self._combo_reset_cache = {}    # kind code -> its property rows, from the config
        self._combo_codes = None        # int32[n_local] by LOCAL ROW, after a build
        self._initial_values = {}       # property -> packed rows it was built with
        self.agentid2learning_rule = _ComboField(
            self, lambda c, _id: (c.learning_rule, c.learning_rule_config)
            if c.learning_rule is not None else None)

        # The network as described so far, and whether SAGESim holds it (_ensure_built).
        self._draft = _NetworkDraft()
        self._built = False             # SAGESim holds the draft as it stands
        self._building = False
        self._partitioned = False       # load_network(): this rank was given its own part
        self._remote_ranks = {}         # agent id -> rank, from a loaded part
        self._host_writes = []          # property writes before setup(), replayed on a rebuild
        self._value_appenders = {}      # (kind code, property, key) -> cached appends
        self._all_ids = None            # several ranks, create calls: every agent id (sorted),
        self._all_codes = None          #   its kind code
        self._all_wiring = None         #   and every synapse's pre and post
        self._nbr_offsets = None
        self._nbr_values = None
        self._agent_ids = None
        self._num_synapses = 0
        self._input_synapse_ids = np.empty(0, dtype=np.int64)

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
        # Nothing of a synapse is read by a neighbour: a soma gets its input through the
        # inbox (step_functions/synaptic_input.py), not by reading its synapses' rows.
        # Note that SAGESim keeps ONE visibility flag per property NAME across breeds, so
        # flagging a synapse property visible would ship the SOMA column of the same name
        # to every ghost soma every tick as well (internal_states: 5 floats per ghost that
        # no kernel reads; the pull-era flag cost 28 B per ghost per tick instead of 8). A
        # custom kernel that reads a neighbour's property on another rank opts in with
        # model.set_property_neighbor_visible(name, True) before setup().
        synapse_properties = {
            "hyperparameters": (
                [0.0] * max_sizes.get("hyperparameters", 0), False
            ),
            "learning_hyperparameters": (
                [0.0] * max_sizes.get("learning_hyperparameters", 0), False
            ),
            "internal_states": (
                [0.0] * max_sizes.get("internal_states", 0), False
            ),
            "learning_internal_states": (
                [0.0] * max_sizes.get("learning_internal_states", 0), False
            ),
            "synapse_delay_reg": ([], False),  # Synapse delay
            "input_spikes_tensor": ([-1.0, 0.0], False),  # [last_delivered_tick, value]
            "output_spikes_tensor": ([], False),
            "internal_states_buffer": ([], False),
            "learning_internal_states_buffer": ([], False),  # learning states buffer
        }
        # Backing stores for the _soma_ids / _synapse_ids properties: materialised from
        # the kind codes only if something asks, which a 12.5M-synapse run never does
        # (it reads _num_synapses / _input_synapse_ids).
        self._synapse_ids_set = None
        self._soma_ids_set = None

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

        self._soma_reads_neighbours = {}  # soma breed name -> bool, see _soma_keeps_neighbour_list
        self.agentid2overrides = _ComboField(self, lambda c, agent_id: self._overrides_of(agent_id))

        self._breed_names = list(self._agent_factory._breeds.keys())

        # Learning rule registry
        if learning_rule_info is None:
            learning_rule_info = _default_learning_rules()
        self._learning_rules = learning_rule_info
        self._learning_rule_names = {r["func_name"]: rid for rid, r in self._learning_rules.items()}
        self._next_learning_rule_id = len(self._learning_rules)
        self._setup_called = False

    # ------------------------------------------------------------------
    # Combo store: what each agent IS, without a dict entry per agent
    # ------------------------------------------------------------------

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

    def _local_row(self, agent_id: int):
        """Row of a local agent in the built columns, or None."""
        self._ensure_built()
        return self._agent_factory._rank2agentid2agentidx.get(
            self._local_rank, {}).get(agent_id)

    def _combo_code_of(self, agent_id: int):
        """Kind code of an agent, or None if this rank does not know it.

        A local agent's code is read by its row. With the create calls on several ranks
        every rank has described the whole network, so an agent held by another rank is
        still known here (through _all_ids); under load_network() only the rank's own
        part is.
        """
        row = self._local_row(agent_id)
        if row is not None and row < len(self._combo_codes):
            return int(self._combo_codes[row])
        if self._all_ids is not None and len(self._all_ids):
            at = int(np.searchsorted(self._all_ids, agent_id))
            if at < len(self._all_ids) and int(self._all_ids[at]) == agent_id:
                return int(self._all_codes[at])
        return None

    def _combo_of(self, agent_id: int):
        code = self._combo_code_of(agent_id)
        return self._combo_table[code] if code is not None else None

    def _synapses_with_rule(self, rows, agent_ids):
        """Wake-eligibility mask for the learning-rule phase: True where the synapse
        was created with a learning rule (buffer rows of local agents are the local
        rows of the kind codes)."""
        has_rule = np.array([c.learning_rule is not None for c in self._combo_table], dtype=bool)
        codes = self._combo_codes
        if codes is not None and len(rows) and int(rows.max()) < len(codes):
            return has_rule[np.asarray(codes)[np.asarray(rows)]]
        return np.array([
            (lambda c: c is not None and c.learning_rule is not None)(self._combo_of(int(a)))
            for a in agent_ids], dtype=bool)

    def _combo_agent_ids(self):
        """Every agent id this rank knows the kind of. Used only by the views."""
        self._ensure_built()
        if self._all_ids is not None:
            return iter(self._all_ids.tolist())
        return iter(np.asarray(self._agent_ids).tolist())

    def _ids_of_class(self, component_class: str) -> set:
        """Agent ids of one component class, from the kind codes: every such agent this
        rank knows of (see _combo_code_of), which on one rank or under load_network()
        is its own."""
        self._ensure_built()
        if self._all_ids is not None:
            ids, codes = self._all_ids, self._all_codes
        else:
            ids, codes = self._agent_ids, self._combo_codes
        if codes is None or ids is None or not len(ids):
            return set()
        wanted = np.array([c.component_class == component_class
                           for c in self._combo_table], dtype=bool)
        return set(np.asarray(ids)[wanted[codes]].tolist())

    @property
    def _soma_ids(self) -> set:
        self._ensure_built()
        if self._soma_ids_set is None:
            self._soma_ids_set = self._ids_of_class("soma")
        return self._soma_ids_set

    @property
    def _synapse_ids(self) -> set:
        """Every synapse id (see _ids_of_class).

        Materialised on demand rather than at build time: at 12.5M synapses this set is
        the kind of per-agent structure the build exists to avoid, and the callers that
        matter at that scale read _num_synapses / _input_synapse_ids instead.
        """
        self._ensure_built()
        if self._synapse_ids_set is None:
            self._synapse_ids_set = self._ids_of_class("synapse")
        return self._synapse_ids_set

    def _component_class_of(self, agent_id: int):
        combo = self._combo_of(agent_id)
        if combo is None:
            raise KeyError(agent_id)
        return combo.component_class

    def _config_of(self, agent_id: int):
        combo = self._combo_of(agent_id)
        return combo.config if combo is not None else None

    def _overrides_of(self, agent_id: int) -> dict:
        """The values a local agent was built with that differ from its config, grouped
        by property: what its ``overrides`` amounted to."""
        row = self._local_row(agent_id)
        if row is None:
            return {}
        code = int(self._combo_codes[row])
        out = {}
        for prop, kept in self._initial_values.items():
            names = self._kind_keys[code].get(prop, [])
            if not names:
                continue
            if len(kept) == 3:
                table, codes, _lengths = kept
                built = table[codes[row]]
            else:
                built = kept[0][row]
            config = self._combo_reset_cache[code].get(prop, [])
            diff = {name: float(built[i]) for i, name in enumerate(names)
                    if built.dtype.type(config[i]) != built[i]}
            if diff:
                out[prop] = diff
        return out

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
        combo = self._combo_of(agent_id)
        if combo is None:
            raise KeyError(agent_id)
        return combo.breed

    def get_synapse_connectivity(self, synapse_id: int) -> List[int]:
        """The synapse's neighbour list as global agent ids: ``[pre_soma_id, post_soma_id]``,
        with ``-1`` for a missing one, followed by anything add_neighbors() gave it. This
        is what the synapse's kernel sees in ``locations[agent_index]`` (there as local
        rows). Works for any agent this rank holds, a soma included; with the create
        calls on several ranks, where every rank described the whole network, also for
        an agent another rank holds.

        It is the one neighbour accessor. The model does not hand a soma's synapses
        back: whoever built the network knows them, and a soma has no neighbour list
        unless its kernel uses one (_soma_keeps_neighbour_list) or add_neighbors() gave
        it one.
        """
        row = self._local_row(synapse_id)
        if row is not None:
            offsets = self._nbr_offsets
            return [int(v) for v in self._nbr_values[offsets[row]:offsets[row + 1]]]
        combo = self._combo_of(synapse_id) if self._all_wiring is not None else None
        if combo is None:
            raise KeyError(f"agent {synapse_id} is not local to this rank")
        ids, pre, post, extra_agent, extra_neighbor = self._all_wiring    # sorted by id
        if combo.component_class == "synapse":
            at = int(np.searchsorted(ids, synapse_id))
            own = [int(pre[at]), int(post[at])]
        elif self._soma_keeps_neighbour_list(combo.breed):
            own = ids[post == synapse_id].tolist()          # in id order
        else:
            own = []
        return own + extra_neighbor[extra_agent == synapse_id].tolist()

    def _soma_keeps_neighbour_list(self, breed_name: str) -> bool:
        """Whether somas of this breed get a neighbour list (their incoming synapse ids).

        Synapses deliver into the soma's inbox, so a soma kernel normally never looks at
        its synapses, and then no list is built, stored or exchanged for it. A soma
        breed whose kernel uses its ``locations`` parameter (to read its afferents'
        weights, say) gets the list on every build route.
        """
        keeps = self._soma_reads_neighbours.get(breed_name)
        if keeps is None:
            n_globals = len(self._global_tensors)
            keeps = any(_reads_neighbours(func, n_globals)
                        for func, _path in self._soma_breeds[breed_name].step_funcs.values())
            self._soma_reads_neighbours[breed_name] = keeps
        return keeps

    def _somas_keeping_neighbour_lists(self, soma_codes) -> np.ndarray:
        """Per soma (given its combo code): does its breed keep a neighbour list?"""
        by_combo = np.array(
            [c.component_class == "soma" and self._soma_keeps_neighbour_list(c.breed)
             for c in self._combo_table], dtype=bool)
        return by_combo[np.asarray(soma_codes, dtype=np.intp)]

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

        # Writes land on the CPU-side AgentFactory. If the device holds values the host
        # has not seen (STDP weights, membrane state after a run), SAGESim's setter
        # downloads them first, then the write lands, then the device is refreshed in
        # place before the next launch: nothing learned is lost, no reset() needed.
        # See docs/CPU_GPU_DATA_FLOW.md section 5.
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

        May be called at any time; after a simulate() the device state is synced back
        first so the write lands on current values -- see docs/CPU_GPU_DATA_FLOW.md.

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
            breed_local_arrays=_synaptic_input_arrays(step_func),
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
            breed_local_arrays=_synaptic_input_arrays(step_func),
            # event stepping: woken by neighbour 0 (the pre-soma) raising a spike, and
            # by injected input events (see _get_extra_kernel_config)
            wake_on_roles=[0] if self._synapse_stepping == "event" else None,
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

    def register_synapse_type(self, name: str, step_func, step_func_path: Path,
                              event_capable: bool = False) -> None:
        """Register a custom synapse type with its step function.

        Must be called before setup(). The learning rule selector is
        auto-attached to all synapse breeds during setup().

        :param name: Unique name for the synapse type.
        :param step_func: The step function callable.
        :param step_func_path: Path to the module containing the step function.
        :param event_capable: The kernel keeps no per-tick state, so it may be stepped
            only when woken (synapse_stepping="event"). Required in event mode.
        """
        if self._setup_called:
            raise RuntimeError(
                "Cannot register synapse type after setup() has been called."
            )
        if name in self._synapse_breeds:
            raise ValueError(f"Synapse type '{name}' is already registered.")
        if self._synapse_stepping == "event" and not event_capable:
            raise ValueError(
                f"synapse_stepping='event': synapse type '{name}' must be event_capable "
                "(no per-tick state; stepped only on the tick after its pre fired)")

        synapse_breed = self._make_synapse_breed(name, step_func, step_func_path)
        self.register_breed(synapse_breed)
        self._synapse_breeds[name] = synapse_breed
        self._breed_names = list(self._agent_factory._breeds.keys())

    def register_learning_rule(
        self, step_func, step_func_path: Path, event_capable: bool = False
    ) -> int:
        """Register a custom learning rule; returns auto-assigned integer ID.

        :param event_capable: The rule catches its traces up with
            ``trace_decay(dt, tau, t_current - last_tick)`` and records the tick it ran
            in ``learning_internal_states[agent_index][3]`` (see the built-in rules), so
            it may be stepped only when woken by a pre or post spike. Required in
            synapse_stepping="event".

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
            "event_capable": bool(event_capable),
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
        # regenerated in place: drop any cached source text (loader-sourced linecache
        # entries are never invalidated) so the JIT reads the new file
        import linecache
        linecache.cache.pop(str(gen_file), None)
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

        The ones this rank holds, from the kind codes: only kinds that carry a rule are
        looked at, so a network with no plasticity costs one pass over the (tiny) kind
        table and nothing per synapse.
        """
        self._ensure_built()
        plastic = np.array([combo.component_class == "synapse" and combo.learning_rule is not None
                            for combo in self._combo_table], dtype=bool)
        codes = self._combo_codes
        if not plastic.any() or codes is None or not len(codes):
            return []
        return np.asarray(self._agent_ids)[plastic[codes]].tolist()

    def _buffers_live(self) -> bool:
        return (
            getattr(self, "_setup_called", False)
            and hasattr(self, "_gpu_buffers")
            and self._gpu_buffers.is_initialized
        )

    def _read_stdp_type(self, synapse_ids: list) -> dict:
        """Read stdp_type (learning_hyperparameters[0]) for locally-owned synapses."""
        self._ensure_built()
        af = self._agent_factory
        prop_idx = af._property_name_2_index["learning_hyperparameters"]
        out = {}
        if self._buffers_live():
            buf = self._gpu_buffers
            tensor = buf.property_tensors[prop_idx]
            sids = list(synapse_ids)
            r = rows_of(buf.agent_id_to_index, sids)
            keep = (r >= 0) & (r < buf.num_local_agents)
            idxs = r[keep]
            ids = [sid for sid, k in zip(sids, keep.tolist()) if k]
            if idxs.size:
                rows = self._gpu_buffers.rows(prop_idx, cp.asarray(idxs))
                vals = rows[:, self._STDP_TYPE_INDEX].get().tolist()
                out = dict(zip(ids, vals))
        else:
            rank = MPI.COMM_WORLD.Get_rank()
            local_agent_map = af._rank2agentid2agentidx.get(rank, {})
            data = af._property_name_2_agent_data_tensor["learning_hyperparameters"]
            sids = list(synapse_ids)
            for sid, idx in zip(sids, rows_of(local_agent_map, sids).tolist()):
                if idx >= 0:
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
        self._ensure_built()
        # If the device holds state the host has not seen (a run since the last
        # build/reset/flush), download it first so this host write lands on current
        # data and the in-place refresh below does not discard anything learned.
        self._ensure_host_current()
        af = self._agent_factory
        local_agent_map = af._rank2agentid2agentidx.get(self._local_rank, {})
        data = af._property_name_2_agent_data_tensor["learning_hyperparameters"]

        r = rows_of(local_agent_map, list(id2value))
        keep = r >= 0
        rows = r[keep]
        vals = np.fromiter((float(v) for v in id2value.values()), dtype=np.float64,
                           count=len(id2value))[keep]
        touched = bool(rows.size)

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
        # _regenerate_data_tensors() needed. The device copy is refreshed in place
        # before the next launch or device read.
        if touched:
            self._host_columns_changed()

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

    def _reset_agents(self, retain_parameters: bool = True) -> None:
        """
        Internal method to reset all soma and synapse agents to their initial states:
        the rows of each agent's kind, with the values that were given for the agent
        itself (overrides) put back.

        Nothing is done per agent. A property with per-agent values is replaced by a
        fresh column from the rows it was built with (_initial_column); every other
        property is one row per kind, scattered to that kind's agents, and a property a
        kind does not have (a soma's learning_hyperparameters) is left as it is.

        :param retain_parameters: If True, keeps current learned parameters.
            If False, resets parameters to the values the model was built with.
        """
        self._ensure_built()
        af = self._agent_factory
        codes = self._combo_codes
        if codes is None or not len(codes):
            return
        data = af._property_name_2_agent_data_tensor

        # Parameters are kept on retain_parameters=True; state always resets.
        keep = {"hyperparameters", "learning_hyperparameters"} if retain_parameters else set()

        with_values = [name for name in self._initial_values if name not in keep]
        for prop_name in with_values:
            data[prop_name] = self._initial_column(prop_name)
        row_codes = np.asarray(codes, dtype=np.int64)
        for code in np.unique(row_codes).tolist():
            rows = np.flatnonzero(row_codes == code)
            for prop_name, row in self._combo_reset_cache[code].items():
                if prop_name in keep or prop_name in with_values:
                    continue
                self._scatter_row(data[prop_name], row, rows)


    def reset(self, retain_parameters: bool = True, release_buffers: bool = False) -> None:
        """
        Resets all soma and synapse agents to their initial states.

        :param retain_parameters: If True, keeps current learned parameters
            (e.g. STDP weights). If False, resets parameters to defaults.
        :param release_buffers: False (the default) keeps the GPU buffers and refreshes
            them in place from the reset host columns, so a reset costs a few column
            uploads rather than a rebuild of the device structures (index map, CSR,
            interning); the right choice for a reset per sample. True frees them and
            the next simulate() rebuilds everything.
        """
        # Step 1: SAGESim syncs GPU->AgentFactory, regenerates tensors, frees GPU.
        # Only the columns we are KEEPING need to come back off the device. Everything
        # else is about to be restored from the combo table by _reset_agents, so reading
        # it back first is a device->host copy of data we immediately overwrite. The
        # learned values live in hyperparameters (weight) and learning_hyperparameters;
        # the STDP traces in learning_internal_states are state, not parameters, and are
        # meant to be cleared.
        super().reset(sync_properties=("hyperparameters", "learning_hyperparameters")
                      if retain_parameters else (),
                      release_buffers=release_buffers)

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
        # The host columns are final now: refresh the live device buffers from them
        # (no-op after release_buffers=True, the next simulate() rebuilds).
        self._flush_pending_reset()
        
    def _check_synaptic_channels(self) -> None:
        """A soma config that filters per channel (keys tau_syn_0..) must declare every
        channel, and a synapse config that picks a channel must pick an existing one.
        Checked here rather than in the kernels: a missing key would read NaN padding, an
        out-of-range slot would index past the inbox row."""
        n_ch = self._synaptic_channels
        for breed_name, configs in self._component_configurations.get("soma", {}).items():
            if breed_name not in self._soma_breeds:
                continue
            for cfg_name, cfg in configs.items():
                hp = cfg.get("hyperparameters", {}) or {}
                if not any(str(k).startswith("tau_syn_") for k in hp):
                    continue
                states = cfg.get("internal_states", {}) or {}
                missing = ([f"tau_syn_{c}" for c in range(n_ch) if f"tau_syn_{c}" not in hp]
                           + [f"I_syn_{c}" for c in range(n_ch) if f"I_syn_{c}" not in states])
                if missing:
                    raise ValueError(
                        f"soma config {breed_name}/{cfg_name} filters per inbox channel but "
                        f"lacks {missing} for synaptic_channels={n_ch}")
        for breed_name, breed in self._synapse_breeds.items():
            delivers = any(f.__name__ == "delivery_synapse_step_func"
                           for f, _p in breed.step_funcs.values())
            for cfg_name, cfg in self._component_configurations.get("synapse", {}).get(
                    breed_name, {}).items():
                hp = cfg.get("hyperparameters", {}) or {}
                if "channel" in hp:
                    self._check_channel(hp["channel"], f"synapse config {breed_name}/{cfg_name}")
                elif delivers:
                    raise ValueError(
                        f"synapse config {breed_name}/{cfg_name} uses the delivery kernel and "
                        "needs a 'channel' hyperparameter (the inbox slot it deposits into)")

    def _check_channel(self, value, where: str) -> None:
        ch = float(value)
        if ch != int(ch) or not 0 <= int(ch) < self._synaptic_channels:
            raise ValueError(
                f"{where}: channel {value!r} is not in 0..{self._synaptic_channels - 1} "
                f"(NeuromorphicModel(synaptic_channels={self._synaptic_channels}))")

    def setup(self) -> None:
        """
        One-time heavy initialization: code gen, JIT, priority analysis.
        Always resets to default state. Call once before simulation loop.

        Builds the model from what the construction calls described, if that has not
        happened yet. The network is fixed from here on.
        """
        self._ensure_built()
        self._setup_called = True
        self._draft = _NetworkDraft()     # the description is no longer needed
        self._host_writes = []
        self._value_appenders = {}
        self._breed_names = list(self._agent_factory._breeds.keys())

        # Generate the selector from the registry and attach it to all synapse breeds.
        # With no rules registered (learning_rule_info={}) there is nothing to select:
        # attaching it would still step every synapse once more per tick (12.5 M idle
        # agents on Brunel) at priority 101.
        new_func, new_path = self._generate_learning_rule_selector()
        if self._synapse_stepping == "event":
            not_capable = [r["func_name"] for r in self._learning_rules.values()
                           if not (r.get("event_capable") or r["func_name"] in EVENT_CAPABLE_RULES)]
            if not_capable:
                raise ValueError(
                    f"synapse_stepping='event': learning rules {not_capable} are not "
                    "event-capable (register_learning_rule(..., event_capable=True) after "
                    "converting them to trace_decay + last_tick, see the built-in rules)")
        for breed in (self._synapse_breeds.values() if self._learning_rules else ()):
            # Priority = one after the last synapse step func
            max_priority = max(breed.step_funcs.keys())
            breed.register_step_func(
                step_func=new_func,
                module_fpath=new_path,
                priority=max_priority + 1,
                no_double_buffer=self._synapse_no_double_buffer,
                breed_local_arrays=[],  # the selector does not touch the soma inbox
                # event stepping: a rule runs on the tick after its pre (role 0) or its
                # post (role 1) fired, or after an input event (pre-tick code)
                wake_on_roles=[0, 1] if self._synapse_stepping == "event" else None,
            )
            if self._synapse_stepping == "event":
                # only synapses that carry a rule have anything to do at this priority
                self.set_wake_eligibility(max_priority + 1, self._synapses_with_rule)

        # Synaptic input: every soma breed has a row in the inbox (a reduce breed-local
        # array, step_functions/synaptic_input.py). neighbor_visible: a synapse whose post
        # soma lives on another rank deposits into the soma's ghost row and SAGESim sums
        # it back into the owner's row, so the partition never changes the result.
        self._check_synaptic_channels()
        if not any(b['name'] == "synaptic_input" for b in self._breed_local_arrays):
            members = list(self._soma_breeds.values())
            self.register_breed_local_array(
                "synaptic_input", members, shape_per_agent=(self._synaptic_channels,),
                neighbor_visible=bool(members), reduce="sum")
        for name, breed in self._synapse_breeds.items():
            for priority, (func, _path) in breed.step_funcs.items():
                if priority == 100 and breed.step_func_blas.get(priority) != ["synaptic_input"]:
                    raise ValueError(
                        f"every synapse kernel must deliver into the soma inbox, but synapse "
                        f"type '{name}' ({func.__name__}) does not take the "
                        "'synaptic_input, synaptic_input_idx' parameters")
        if self._synapse_stepping == "event":
            # a soma raises the event by writing its spike into output_spikes_tensor[t % 2]
            self.configure_wake(event_property="output_spikes_tensor")

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
        # delivery_synapse carries 1 internal state, a lif_soma 5), the tensor width is
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

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    #
    # Every construction call only records rows in the draft (_NetworkDraft); nothing is
    # handed to SAGESim until _ensure_built(), which setup() and every read or write of
    # the built model call. So one soma, a million synapses from arrays and a rank's part
    # read from a file all end in the same arrays and the same single build.
    #
    # A KIND is what many agents share: component class, breed, config and learning rule
    # (a Combo, one small code per agent). A VALUE is a number given for one agent, such
    # as its weight: it fills a slot of that agent's row and never makes a new kind.

    _VALUE_PROPERTIES = ("hyperparameters", "internal_states",
                         "learning_hyperparameters", "learning_internal_states")

    def _kind_code(self, component_class: str, breed: str, config_name: str,
                   learning_rule: str = None, learning_rule_config: str = None) -> int:
        """Code of a kind of agent, registering (and validating) it on first use."""
        key = (component_class, breed, config_name, learning_rule, learning_rule_config)
        code = self._combo_index.get(key)
        if code is None:
            combo = Combo(*key)
            breeds = self._soma_breeds if component_class == "soma" else self._synapse_breeds
            # config first, then breed: both a KeyError, as they always were
            self._component_configurations[component_class][breed][config_name]
            if breed not in breeds:
                raise KeyError(
                    f"{component_class} breed {breed!r} is not registered; "
                    f"known: {sorted(breeds)}")
            rows = self._combo_property_rows(combo)       # validates the pinned key order
            if component_class == "soma":
                hp_keys, _, is_keys, _ = self._config_list_cache[("soma", breed, config_name)]
                keys = {"hyperparameters": hp_keys, "internal_states": is_keys}
            else:
                hp_keys, _, is_keys, _ = self._config_list_cache[("synapse", breed, config_name)]
                if learning_rule is not None:
                    lhp_keys, _, ils_keys, _ = self._config_list_cache[
                        ("learning_rule", learning_rule, learning_rule_config)]
                else:
                    lhp_keys, ils_keys = ["stdp_type"], []
                keys = {"hyperparameters": hp_keys, "internal_states": is_keys,
                        "learning_hyperparameters": lhp_keys,
                        "learning_internal_states": ils_keys}
            code = len(self._combo_table)
            self._combo_table.append(combo)
            self._combo_index[key] = code
            self._kind_keys.append(keys)
            self._combo_reset_cache[code] = rows
        return code

    def _open_draft(self, who: str) -> "_NetworkDraft":
        if self._setup_called:
            raise RuntimeError(
                f"{who}() after setup(): create every soma and synapse before setup().")
        if self._partitioned:
            raise RuntimeError(
                f"Cannot {who}() on a model built with load_network(): a rank that loads "
                "its own part gets the whole of it from that call. (add_neighbors() is "
                "the one call allowed afterwards.)")
        self._built = False
        return self._draft

    def _value_appender(self, code: int, prop: str, key: str) -> tuple:
        """(append ordinal, append value, is a synapse channel) for one parameter of one
        kind: validated here once, then one dict lookup per create call."""
        combo = self._combo_table[code]
        names = self._kind_keys[code].get(prop)
        if names is None:
            raise ValueError(
                f"{combo.component_class} {combo.breed}/{combo.config} has no "
                f"{prop!r} to override; it has {list(self._kind_keys[code])}")
        if key not in names:
            raise ValueError(
                f"{key!r} is not a {prop} name of {combo.component_class} "
                f"{combo.breed}/{combo.config}; valid names are {names}")
        column = self._draft.values.get((prop, key))
        if column is None:
            column = self._draft.values[(prop, key)] = _Values()
        append = (column.rows.one[0].append, column.rows.one[1].append,
                  key == "channel" and prop == "hyperparameters"
                  and combo.component_class == "synapse")
        self._value_appenders[(code, prop, key)] = append
        return append

    def _record_values(self, code: int, overrides: dict, first_seq: int, n: int = None) -> None:
        """Record the values given for one agent (``n`` is None) or for the ``n`` agents
        whose creation ordinals start at ``first_seq``."""
        keys_of = self._kind_keys[code]
        combo = self._combo_table[code]
        values = self._draft.values
        for prop, given in overrides.items():
            if not given:
                continue
            names = keys_of.get(prop)
            if names is None:
                raise ValueError(
                    f"{combo.component_class} {combo.breed}/{combo.config} has no "
                    f"{prop!r} to override; it has {list(keys_of)}")
            for key, value in given.items():
                if key not in names:
                    raise ValueError(
                        f"{key!r} is not a {prop} name of {combo.component_class} "
                        f"{combo.breed}/{combo.config}; valid names are {names}")
                column = values.get((prop, key))
                if column is None:
                    column = values[(prop, key)] = _Values()
                if n is None:
                    value = float(value)
                    if key == "channel" and prop == "hyperparameters" \
                            and combo.component_class == "synapse":
                        self._check_channel(
                            value, f"synapse {combo.breed}/{combo.config} override")
                    column.add_one(first_seq, value)
                else:
                    array_value = np.asarray(value, dtype=np.float64)
                    if array_value.ndim == 0:
                        array_value = np.full(n, float(array_value))
                    elif array_value.shape != (n,):
                        raise ValueError(
                            f"{prop} {key!r}: expected one value or {n} values, got "
                            f"shape {array_value.shape}")
                    if key == "channel" and prop == "hyperparameters" \
                            and combo.component_class == "synapse":
                        for channel in np.unique(array_value).tolist():
                            self._check_channel(
                                channel, f"synapse {combo.breed}/{combo.config} override")
                    column.add_run(first_seq, array_value)

    def create_soma(
        self,
        breed: str,
        config_name: str,
        overrides: Dict[str, Dict[str, float]] = None,
        agent_id: int = None,
    ) -> int:
        """
        Creates one soma.

        :param overrides: Dict keyed by property type, e.g.
            {"hyperparameters": {"R": 1.1e6}, "internal_states": {"v": -55.0}}
        :param agent_id: Explicit global agent ID instead of the next automatic one; must
            be unique among somas and synapses.
        :return: agent id of the soma
        """
        if self._setup_called or self._partitioned:
            self._open_draft("create_soma")              # raises
        self._built = False
        draft = self._draft
        code = self._combo_index.get(("soma", breed, config_name, None, None))
        if code is None:
            code = self._kind_code("soma", breed, config_name)
        seq = draft.n_created
        if agent_id is None:
            agent_id = draft.next_id
            draft.next_id = agent_id + 1
        else:
            agent_id = int(agent_id)
            draft.explicit_ids = True
            if agent_id >= draft.next_id:
                draft.next_id = agent_id + 1
        draft.n_created = seq + 1
        ids, codes, seqs = draft.somas.one
        ids.append(agent_id)
        codes.append(code)
        seqs.append(seq)
        if overrides:
            appenders = self._value_appenders
            for prop, given in overrides.items():
                for key, value in given.items():
                    append = appenders.get((code, prop, key))
                    if append is None:
                        append = self._value_appender(code, prop, key)
                    value = float(value)
                    if append[2]:
                        self._check_channel(value, "synapse channel override")
                    append[0](seq)
                    append[1](value)
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
        agent_id: int = None,
    ) -> int:
        """
        Creates one synapse.

        Parameters:
            breed (str): Synapse breed name (e.g., 'delivery_synapse').
            pre_soma_id (int): Presynaptic soma agent ID (or -1 for external input).
            post_soma_id (int): Postsynaptic soma agent ID (or -1 for none).
            config_name (str): Name of the configuration to use for this synapse.
            learning_rule (str, optional): Learning rule name (e.g. "exp_pair_wise_stdp"), or None for no learning.
            learning_rule_config (str): Config name within the learning rule (default: "default").
            overrides (dict, optional): Dict keyed by property type, e.g.
                {"hyperparameters": {"weight": 0.5}, "learning_hyperparameters": {"a_exp_pre": 0.01}}
            agent_id (int, optional): Explicit global agent ID; must be unique.

        Returns:
            int: agent ID of the created synapse.

        The synapse's neighbour list is ``[pre_soma_id, post_soma_id]`` (see
        add_neighbors for more). It is stepped on the rank of its post soma.
        """
        if self._setup_called or self._partitioned:
            self._open_draft("create_synapse")           # raises
        self._built = False
        draft = self._draft
        if learning_rule:
            code = self._kind_code("synapse", breed, config_name,
                                   learning_rule, learning_rule_config)
        else:
            code = self._combo_index.get(("synapse", breed, config_name, None, None))
            if code is None:
                code = self._kind_code("synapse", breed, config_name)
        seq = draft.n_created
        if agent_id is None:
            agent_id = draft.next_id
            draft.next_id = agent_id + 1
        else:
            agent_id = int(agent_id)
            draft.explicit_ids = True
            if agent_id >= draft.next_id:
                draft.next_id = agent_id + 1
        draft.n_created = seq + 1
        ids, pre, post, codes, seqs = draft.synapses.one
        ids.append(agent_id)
        pre.append(pre_soma_id)
        post.append(post_soma_id)
        codes.append(code)
        seqs.append(seq)
        if overrides:
            appenders = self._value_appenders
            for prop, given in overrides.items():
                for key, value in given.items():
                    append = appenders.get((code, prop, key))
                    if append is None:
                        append = self._value_appender(code, prop, key)
                    value = float(value)
                    if append[2]:
                        self._check_channel(value, "synapse channel override")
                    append[0](seq)
                    append[1](value)
        return agent_id

    def _take_ids(self, draft, n, agent_ids, who: str):
        """Ids and creation ordinals for a block of ``n`` agents."""
        if agent_ids is None:
            if n is None:
                raise ValueError(f"{who}(): give n or agent_ids")
            n = int(n)
            ids = np.arange(draft.next_id, draft.next_id + n, dtype=np.int64)
            draft.next_id += n
        else:
            ids = np.array(agent_ids, dtype=np.int64).ravel()
            if n is not None and int(n) != ids.size:
                raise ValueError(f"{who}(): n is {n} but {ids.size} agent_ids were given")
            n = ids.size
            draft.explicit_ids = True
            if n:
                draft.next_id = max(draft.next_id, int(ids.max()) + 1)
        first_seq = draft.n_created
        draft.n_created += n
        return ids, n, first_seq

    def create_somas(self, n: int, breed: str, config_name: str,
                     overrides: dict = None, agent_ids=None) -> np.ndarray:
        """
        Creates ``n`` somas of one breed and config.

        :param overrides: as for create_soma; each value is one number for all ``n`` somas
            or an array with one number per soma, e.g.
            {"internal_states": {"v": initial_voltages}}
        :param agent_ids: explicit global ids (then ``n`` may be None)
        :return: the agent ids, an int64 array
        """
        draft = self._open_draft("create_somas")
        code = self._kind_code("soma", breed, config_name)
        ids, n, first_seq = self._take_ids(draft, n, agent_ids, "create_somas")
        draft.somas.add_block(ids, np.full(n, code, dtype=np.int32),
                              np.arange(first_seq, first_seq + n, dtype=np.int64))
        if overrides:
            self._record_values(code, overrides, first_seq, n)
        return ids

    def create_synapses(self, pre_soma_ids, post_soma_ids, breed: str, config_name: str,
                        learning_rule: str = None, learning_rule_config: str = "default",
                        overrides: dict = None, agent_ids=None) -> np.ndarray:
        """
        Creates one synapse per (pre, post) pair, all of one breed, config and learning
        rule. ``pre_soma_ids`` and ``post_soma_ids`` are arrays of the same length, or one
        of them a single id (``-1`` for external input synapses onto many somas).

        :param overrides: as for create_synapse; each value is one number for all the
            synapses or an array with one number per synapse, e.g.
            {"hyperparameters": {"weight": weights}}. A number per synapse costs no more
            than one for all: it fills that synapse's slot directly.
        :param agent_ids: explicit global ids
        :return: the agent ids, an int64 array, in the order of the pairs
        """
        draft = self._open_draft("create_synapses")
        if learning_rule:
            code = self._kind_code("synapse", breed, config_name,
                                   learning_rule, learning_rule_config)
        else:
            code = self._kind_code("synapse", breed, config_name)
        pre, post = np.broadcast_arrays(np.asarray(pre_soma_ids, dtype=np.int64),
                                        np.asarray(post_soma_ids, dtype=np.int64))
        pre, post = pre.ravel(), post.ravel()
        ids, n, first_seq = self._take_ids(
            draft, pre.size if agent_ids is None else None, agent_ids, "create_synapses")
        if n != pre.size:
            raise ValueError(
                f"create_synapses(): {pre.size} (pre, post) pairs but {n} agent_ids")
        draft.synapses.add_block(ids, pre, post, np.full(n, code, dtype=np.int32),
                                 np.arange(first_seq, first_seq + n, dtype=np.int64))
        if overrides:
            self._record_values(code, overrides, first_seq, n)
        return ids

    def add_neighbors(self, agent_ids, neighbor_ids) -> None:
        """
        Gives agents extra neighbours: ``neighbor_ids[k]`` is appended to the neighbour
        list of ``agent_ids[k]`` (single ids, or arrays; one side may be a single id).

        Any agent may neighbour any agent: a synapse another synapse, a soma a soma. The
        extras come after what the agent has by itself, in the order given: a synapse's
        list is ``[pre, post, extras...]``, a soma's is its incoming synapses (only when
        its step function reads its neighbours) followed by its extras. The built-in step
        functions read slots 0 and 1 only; the extras are for your own step functions,
        which see them in ``locations``.

        Under load_network(), a neighbour on another rank must be named in the part's
        ``remote_ranks``.
        """
        if self._setup_called:
            raise RuntimeError("add_neighbors() after setup(): add neighbours before setup().")
        self._built = False
        extras = self._draft.extras
        if isinstance(agent_ids, (int, np.integer)) and isinstance(neighbor_ids, (int, np.integer)):
            extras.one[0].append(agent_ids)
            extras.one[1].append(neighbor_ids)
            return
        agents, neighbors = np.broadcast_arrays(np.asarray(agent_ids, dtype=np.int64),
                                                np.asarray(neighbor_ids, dtype=np.int64))
        extras.add_block(agents.ravel(), neighbors.ravel())

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
        """Read an array-layout (.npz) part into an array dict.

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

    def load_network(self, part,
                     soma_breed: str = "lif_soma",
                     soma_config: str = "config_0",
                     synapse_breed: str = "delivery_synapse",
                     synapse_config: str = "config_0") -> None:
        """Build this rank's model from ITS OWN PART of a network.

        The create calls describe a whole network and SuperNeuroABM splits it across
        ranks. ``load_network`` is for a network that was split beforehand (a large one,
        or one partitioned with a graph partitioner): every rank calls it with its own
        part and keeps everything in it. A part holds

        - the somas this rank owns,
        - the synapses this rank steps, each by its ``pre`` and ``post`` soma id
          (``pre = -1``: an external input). Either soma may live on another rank;
        - ``remote_ranks``: the rank of every agent named here that this rank does not
          hold.

        Every soma and every synapse must be in exactly one rank's part. A soma whose own
        step function reads its incoming synapses finds only the synapses of its own
        part; name any other with add_neighbors(). On one rank the part is the whole
        network.

        ``part`` is a file path or the same data in memory, in one of two layouts:

        records (``.pkl``, or a dict)::

            {"somas":    [{"id", "breed", "config", "overrides"}, ...],
             "synapses": [{"id", "pre", "post", "breed", "config", "learning_rule",
                           "learning_rule_config", "overrides"}, ...],
             "remote_ranks": {agent_id: rank}}

        arrays (``.npz``, or a dict; ``schema = "columnar_post_owned_v1"``)::

            soma_ids, synapse_ids, pre, post             int arrays
            soma_breed, soma_config, synapse_breed,      one name each, or an array with
            synapse_config, learning_rule,               one name per agent ("" = no
            learning_rule_config                         learning rule)
            syn_hp_keys, syn_hp_vals                     hyperparameter names and one row
                                                         of values per synapse (NaN: keep
                                                         the config's value)
            remote_ids, remote_rank_of                   optional

        In the records layout ``overrides`` is grouped by property ("hyperparameters",
        "internal_states", "learning_hyperparameters", "learning_internal_states") and a
        missing ``breed`` / ``config`` falls back to the arguments below.

        :param part: path (.pkl records, .npz arrays) or the dict itself
        :param soma_breed: default soma breed for records that name none
        :param soma_config: default soma config
        :param synapse_breed: default synapse breed
        :param synapse_config: default synapse config
        """
        if self._setup_called or self._partitioned or self._draft.n_created:
            raise RuntimeError(
                "load_network() builds this rank's whole model: call it once, on a "
                "model with no somas or synapses yet, before setup(). It cannot be "
                "combined with create_soma/create_synapse/create_somas/create_synapses.")
        import time
        _t = time.time()
        source = part
        if isinstance(source, (str, Path)):
            source = (self._read_columnar_partition_file(source)
                      if Path(source).suffix.lower() == '.npz'
                      else self._read_partition_file(source))
        if 'soma_ids' in source:
            self._draft_from_columns(source)
        elif 'somas' in source or 'synapses' in source:
            self._draft_from_records(
                self._normalize_snn_partition(source),
                soma_breed, soma_config, synapse_breed, synapse_config)
        else:
            raise ValueError(
                "load_network(): expected a path, an array dict (with 'soma_ids'), or a "
                f"record dict (with 'somas'/'synapses'); got keys {sorted(source)[:8]}.")
        self._partitioned = True
        self._built = False
        read_seconds = time.time() - _t
        self._ensure_built()
        self._load_timings["read"] = read_seconds

    def _draft_from_records(self, data: dict, soma_breed, soma_config,
                            synapse_breed, synapse_config) -> None:
        """Record layout -> draft. One Python pass over the records, which is what this
        layout costs; everything after it is the same array code as every other route."""
        draft = self._draft
        somas, synapses = data['somas'], data['synapses']
        n_s, n_y = len(somas), len(synapses)
        first = draft.n_created
        soma_ids = np.empty(n_s, dtype=np.int64)
        soma_codes = np.empty(n_s, dtype=np.int32)
        synapse_ids = np.empty(n_y, dtype=np.int64)
        pre = np.empty(n_y, dtype=np.int64)
        post = np.empty(n_y, dtype=np.int64)
        synapse_codes = np.empty(n_y, dtype=np.int32)
        kind = self._kind_code
        for i, soma in enumerate(somas):
            soma_ids[i] = soma['id']
            code = kind("soma", soma.get('breed') or soma_breed,
                        soma.get('config') or soma_config)
            soma_codes[i] = code
            overrides = soma.get('overrides')
            if overrides:
                self._record_values(code, overrides, first + i)
        for j, syn in enumerate(synapses):
            if 'pre' not in syn:
                raise ValueError(
                    f"load_network(): synapse {syn.get('id')} has no 'pre'/'post'. "
                    "Explicit 'neighbors' lists are no longer read: give every synapse "
                    "its pre and post, and add any further neighbour with add_neighbors().")
            synapse_ids[j] = syn['id']
            pre[j] = syn['pre']
            post[j] = syn.get('post', -1)
            rule = syn.get('learning_rule') or None
            if rule:
                code = kind("synapse", syn.get('breed') or synapse_breed,
                            syn.get('config') or synapse_config,
                            rule, syn.get('learning_rule_config') or 'default')
            else:
                code = kind("synapse", syn.get('breed') or synapse_breed,
                            syn.get('config') or synapse_config)
            synapse_codes[j] = code
            overrides = syn.get('overrides')
            if overrides:
                self._record_values(code, overrides, first + n_s + j)
        draft.somas.add_block(soma_ids, soma_codes,
                              np.arange(first, first + n_s, dtype=np.int64))
        draft.synapses.add_block(synapse_ids, pre, post, synapse_codes,
                                 np.arange(first + n_s, first + n_s + n_y, dtype=np.int64))
        draft.n_created = first + n_s + n_y
        draft.explicit_ids = True
        top = max(int(soma_ids.max()) if n_s else -1, int(synapse_ids.max()) if n_y else -1)
        draft.next_id = max(draft.next_id, top + 1)
        self._remote_ranks = {int(k): int(v) for k, v in data['remote_ranks'].items()}

    def _draft_from_columns(self, arrays: dict) -> None:
        """Array layout -> draft. A name given once is the kind of every agent; a name
        array gives each agent its own, and the kinds are the distinct combinations."""
        draft = self._draft
        soma_ids = np.asarray(arrays['soma_ids'], dtype=np.int64)
        synapse_ids = np.asarray(arrays['synapse_ids'], dtype=np.int64)
        pre = np.asarray(arrays['pre'], dtype=np.int64)
        post = np.asarray(arrays['post'], dtype=np.int64)
        n_s, n_y = soma_ids.size, synapse_ids.size
        first = draft.n_created

        def kinds(n, component_class, *names):
            """Kind code per agent from name columns (each one name, or n names)."""
            columns = [np.asarray(arrays[name]) for name in names]
            if all(col.ndim == 0 for col in columns):
                labels = [col.item() for col in columns]
                return np.full(n, self._kind_from_labels(component_class, labels),
                               dtype=np.int32)
            inverses, uniques = [], []
            for col in columns:
                col = np.broadcast_to(col, (n,)) if col.ndim == 0 else col
                u, inv = np.unique(col, return_inverse=True)
                uniques.append(u)
                inverses.append(inv.ravel())
            key = np.zeros(n, dtype=np.int64)
            for u, inv in zip(uniques, inverses):
                key = key * len(u) + inv
            distinct, which = np.unique(key, return_inverse=True)
            codes = np.empty(len(distinct), dtype=np.int32)
            for k, value in enumerate(distinct.tolist()):
                labels = []
                for u in reversed(uniques):
                    value, digit = divmod(value, len(u))
                    labels.append(u[digit].item())
                codes[k] = self._kind_from_labels(component_class, labels[::-1])
            return codes[which.ravel()]

        soma_codes = kinds(n_s, "soma", 'soma_breed', 'soma_config')
        synapse_codes = kinds(n_y, "synapse", 'synapse_breed', 'synapse_config',
                              'learning_rule', 'learning_rule_config')
        draft.somas.add_block(soma_ids, soma_codes,
                              np.arange(first, first + n_s, dtype=np.int64))
        draft.synapses.add_block(synapse_ids, pre, post, synapse_codes,
                                 np.arange(first + n_s, first + n_s + n_y, dtype=np.int64))
        draft.n_created = first + n_s + n_y
        draft.explicit_ids = True
        top = max(int(soma_ids.max()) if n_s else -1, int(synapse_ids.max()) if n_y else -1)
        draft.next_id = max(draft.next_id, top + 1)

        if 'syn_hp_keys' in arrays and n_y:
            hp_keys = [str(k) for k in np.asarray(arrays['syn_hp_keys'])]
            hp_vals = np.asarray(arrays['syn_hp_vals'])
            for column, key in enumerate(hp_keys):
                self._record_block_values(
                    np.unique(synapse_codes), synapse_codes, "hyperparameters", key,
                    hp_vals[:, column], first + n_s)
        remote_ids = np.asarray(arrays.get('remote_ids', np.empty(0, np.int64)))
        remote_rank_of = np.asarray(arrays.get('remote_rank_of', np.empty(0, np.int64)))
        self._remote_ranks = {int(i): int(r) for i, r in zip(remote_ids, remote_rank_of)}

    def _kind_from_labels(self, component_class: str, labels: list) -> int:
        if component_class == "soma":
            return self._kind_code("soma", labels[0], labels[1])
        rule = labels[2] or None
        if rule:
            return self._kind_code("synapse", labels[0], labels[1], rule, labels[3])
        return self._kind_code("synapse", labels[0], labels[1])

    def _record_block_values(self, kind_codes, codes, prop: str, key: str, values,
                             first_seq: int) -> None:
        """One value per agent of a block that may mix kinds (the array layout). NaN keeps
        the config's value, so a key only has to exist for the kinds that are given one."""
        values = np.asarray(values, dtype=np.float64)
        given = ~np.isnan(values)
        for code in np.asarray(kind_codes).tolist():
            names = self._kind_keys[code].get(prop, [])
            if key not in names:
                rows = codes == code
                if np.any(given[rows]):
                    combo = self._combo_table[code]
                    raise ValueError(
                        f"{key!r} is not a {prop} name of {combo.component_class} "
                        f"{combo.breed}/{combo.config}; valid names are {names}")
            elif key == "channel" and prop == "hyperparameters":
                for channel in np.unique(values[(codes == code) & given]).tolist():
                    self._check_channel(channel, "synapse channel column")
        column = self._draft.values.get((prop, key))
        if column is None:
            column = self._draft.values[(prop, key)] = _Values()
        column.add_run(first_seq, values)

    def _combo_breed_index(self, combo) -> int:
        breeds = self._soma_breeds if combo.component_class == "soma" else self._synapse_breeds
        return breeds[combo.breed]._breedidx

    def _combo_property_rows(self, combo) -> dict:
        """Property rows of one kind, from its config: computed once per kind."""
        if combo.component_class == "soma":
            hp, is_state = self._get_soma_properties(combo.breed, combo.config, {})
            return {
                "hyperparameters": hp,
                "internal_states": is_state,
                "output_spikes_tensor": [0.0, 0.0],
            }
        props, _hp, _lhp, _is, _ils = self._get_synapse_properties(
            combo.breed, combo.config, {},
            combo.learning_rule, combo.learning_rule_config or "default")
        return props

    # ------------------------------------------------------------------
    # The one build: draft -> SAGESim columns and neighbour table
    # ------------------------------------------------------------------

    def _ensure_built(self) -> None:
        """Hand the current draft to SAGESim if that has not happened since it last
        changed. Called by setup() and by everything that reads or writes agents."""
        if not self._built and not self._building:
            self._build_from_draft()

    # SAGESim's agent accessors, behind the lazy build. A property write made before
    # setup() is also logged: if more of the network is described afterwards the model is
    # built again, and the write is made again on the new columns.

    def get_space(self):
        self._ensure_built()
        return super().get_space()

    def get_agent_property_value(self, id: int, property_name: str):
        self._ensure_built()
        return super().get_agent_property_value(id, property_name)

    def get_local_agent_property_value(self, id: int, property_name: str):
        self._ensure_built()
        return super().get_local_agent_property_value(id, property_name)

    def set_agent_property_value(self, id: int, property_name: str, value) -> None:
        self._ensure_built()
        if not self._setup_called:
            kept = list(value) if isinstance(value, (list, tuple, np.ndarray)) else value
            self._host_writes.append(("set_agent_property_value", id, property_name, kept))
        super().set_agent_property_value(id, property_name, value)

    def set_local_agent_property_value(self, id: int, property_name: str, value) -> None:
        self._ensure_built()
        if not self._setup_called:
            kept = list(value) if isinstance(value, (list, tuple, np.ndarray)) else value
            self._host_writes.append(("set_local_agent_property_value", id, property_name, kept))
        super().set_local_agent_property_value(id, property_name, value)

    def get_breed_data(self, *args, **kwargs):
        self._ensure_built()
        return super().get_breed_data(*args, **kwargs)

    def get_breed_agent_ids(self, *args, **kwargs):
        self._ensure_built()
        return super().get_breed_agent_ids(*args, **kwargs)

    def get_agents_with(self, query):
        self._ensure_built()
        return super().get_agents_with(query)

    def _build_from_draft(self) -> None:
        import time
        timings = {}
        _t = time.time()
        self._building = True
        try:
            self._do_build(timings, _t)
        finally:
            self._building = False

    def _do_build(self, timings: dict, _t: float) -> None:
        import time
        draft = self._draft
        size = MPI.COMM_WORLD.Get_size()
        me = self._local_rank
        s_ids, s_codes, s_seq = draft.somas.columns()
        y_ids, y_pre, y_post, y_codes, y_seq = draft.synapses.columns()
        x_agent, x_neighbor = draft.extras.columns()
        total = draft.n_created

        if draft.explicit_ids:
            every = np.concatenate([s_ids, y_ids])
            uniq, counts = np.unique(every, return_counts=True)
            if uniq.size != every.size:
                dup = uniq[counts > 1]
                raise ValueError(
                    f"duplicate agent id(s), e.g. {int(dup[0])} ({dup.size} distinct ids "
                    "repeat). Soma and synapse ids share one namespace and must be unique.")

        # ---- Which rank holds what. A rank given its own part keeps all of it. With
        # the create calls every rank has described the whole network: somas (and
        # post-less synapses) are dealt to ranks in creation order, and a synapse goes
        # with its post soma.
        remote_ranks = dict(self._remote_ranks)
        self._all_ids = self._all_codes = self._all_wiring = None
        if size > 1 and not self._partitioned:
            dealt = np.zeros(total, dtype=bool)
            dealt[s_seq] = True
            postless = y_post < 0
            dealt[y_seq[postless]] = True
            ordinal = np.cumsum(dealt) - 1
            s_rank = ordinal[s_seq] % size
            y_rank = np.full(y_ids.size, -1, dtype=np.int64)
            y_rank[postless] = ordinal[y_seq[postless]] % size
            every_id = np.concatenate([s_ids, y_ids])
            every_rank = np.concatenate([s_rank, y_rank])
            by_id = np.argsort(every_id, kind="stable")
            sorted_id = every_id[by_id]
            waiting = ~postless
            for _ in range(64):               # a synapse whose post is itself a synapse
                if not waiting.any():
                    break
                at = np.searchsorted(sorted_id, y_post[waiting])
                at_clipped = np.clip(at, 0, max(sorted_id.size - 1, 0))
                exists = (at < sorted_id.size) & (sorted_id[at_clipped] == y_post[waiting])
                if not exists.all():
                    bad = np.flatnonzero(waiting)[~exists]
                    raise ValueError(
                        f"{bad.size} synapse(s) name a post agent that was never created, "
                        f"e.g. synapse {int(y_ids[bad[0]])} post {int(y_post[bad[0]])}")
                every_rank[s_ids.size:] = y_rank
                found = every_rank[by_id[at_clipped]]
                index = np.flatnonzero(waiting)
                y_rank[index[found >= 0]] = found[found >= 0]
                if not (found >= 0).any():
                    raise ValueError("synapses name each other as post in a cycle")
                waiting[index[found >= 0]] = False
            every_rank[s_ids.size:] = y_rank
            self._all_ids = sorted_id
            self._all_codes = np.concatenate([s_codes, y_codes])[by_id]
            all_rank = every_rank[by_id]
            # the whole wiring, for get_synapse_connectivity() of an agent held elsewhere
            by_synapse = np.argsort(y_ids, kind="stable")
            self._all_wiring = (y_ids[by_synapse], y_pre[by_synapse], y_post[by_synapse],
                                x_agent, x_neighbor)
            keep_s, keep_y = s_rank == me, y_rank == me
            s_ids, s_codes, s_seq = s_ids[keep_s], s_codes[keep_s], s_seq[keep_s]
            y_ids, y_pre, y_post, y_codes, y_seq = (
                y_ids[keep_y], y_pre[keep_y], y_post[keep_y], y_codes[keep_y], y_seq[keep_y])

        # ---- Row order: somas in creation order, then synapses. Event stepping lays
        # the synapse rows out by pre soma, so the synapses one spike wakes are one
        # contiguous run (external inputs, pre == -1, first; creation order within a run).
        if self._synapse_stepping == "event" and y_pre.size:
            by_pre = np.argsort(y_pre, kind="stable")
            y_ids, y_pre, y_post, y_codes, y_seq = (
                y_ids[by_pre], y_pre[by_pre], y_post[by_pre], y_codes[by_pre], y_seq[by_pre])
        n_s, n_y = s_ids.size, y_ids.size
        n = n_s + n_y
        ids = np.concatenate([s_ids, y_ids])
        codes = np.concatenate([s_codes, y_codes]).astype(np.int32)
        seq = np.concatenate([s_seq, y_seq])

        # SAGESim wants rows grouped by non-decreasing breed index. `source` is, for each
        # final row, its position in the [somas..., synapses...] order above.
        combo_breed = np.array(
            [self._combo_breed_index(c) for c in self._combo_table], dtype=np.int64)
        breed_indices = combo_breed[codes] if n else np.empty(0, dtype=np.int64)
        row_of_source = np.arange(n, dtype=np.int64)
        if n and np.any(np.diff(breed_indices) < 0):
            source = np.argsort(breed_indices, kind="stable")
            ids, codes, seq, breed_indices = (
                ids[source], codes[source], seq[source], breed_indices[source])
            row_of_source = np.empty(n, dtype=np.int64)
            row_of_source[source] = np.arange(n, dtype=np.int64)
        soma_row = row_of_source[:n_s]
        synapse_row = row_of_source[n_s:]
        timings["assemble"] = time.time() - _t
        _t = time.time()

        # ---- Neighbour table. A synapse: [pre, post, extras...], -1 for none. A soma:
        # its incoming synapses in creation order when its step function reads its
        # neighbours (otherwise none: it takes its input from the inbox), then extras.
        by_soma = np.argsort(s_ids, kind="stable")
        sorted_soma = s_ids[by_soma]
        at = np.searchsorted(sorted_soma, y_post)
        at_clipped = np.clip(at, 0, max(n_s - 1, 0))
        post_local = (y_post >= 0) & (at < n_s) & (
            sorted_soma[at_clipped] == y_post if n_s else np.zeros(n_y, dtype=bool))
        away = np.flatnonzero((y_post >= 0) & ~post_local)   # post is not a soma held here
        if away.size and self._partitioned:
            elsewhere = np.fromiter(remote_ranks.keys(), dtype=np.int64,
                                    count=len(remote_ranks))
            unknown = away[~np.isin(y_post[away], elsewhere) & ~np.isin(y_post[away], y_ids)]
            if unknown.size:
                raise ValueError(
                    f"{unknown.size} synapse(s) have a post soma that is neither in "
                    "this rank's part nor named in remote_ranks, e.g. synapse "
                    f"{int(y_ids[unknown[0]])} post {int(y_post[unknown[0]])}")
        elif away.size and size == 1:
            missing = away[~np.isin(y_post[away], y_ids)]
            if missing.size:
                raise ValueError(
                    f"{missing.size} synapse(s) name a post agent that was never "
                    f"created, e.g. synapse {int(y_ids[missing[0]])} "
                    f"post {int(y_post[missing[0]])}")

        counts = np.zeros(n, dtype=np.int64)
        counts[synapse_row] = 2
        keeps = self._somas_keeping_neighbour_lists(s_codes) if n_s else np.zeros(0, dtype=bool)
        incoming_row = incoming_id = np.empty(0, dtype=np.int64)
        if keeps.any():
            post_soma = by_soma[at_clipped]                  # index into the soma arrays
            listed = post_local & keeps[post_soma]
            in_creation_order = np.argsort(y_seq[listed], kind="stable")
            incoming_row = soma_row[post_soma[listed][in_creation_order]]
            incoming_id = y_ids[listed][in_creation_order]
            counts += np.bincount(incoming_row, minlength=n)
        base = counts.copy()
        extra_row = extra_id = np.empty(0, dtype=np.int64)
        if x_agent.size:
            by_local = np.argsort(ids, kind="stable")
            sorted_local = ids[by_local]
            where = np.searchsorted(sorted_local, x_agent)
            where_clipped = np.clip(where, 0, max(n - 1, 0))
            here = (where < n) & (sorted_local[where_clipped] == x_agent if n
                                  else np.zeros(x_agent.size, dtype=bool))
            if self._partitioned and not here.all():
                raise ValueError(
                    f"add_neighbors(): agent {int(x_agent[~here][0])} is not in this "
                    "rank's part")
            extra_row = by_local[where_clipped[here]]
            extra_id = x_neighbor[here]
            counts += np.bincount(extra_row, minlength=n)
        offsets = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        values = np.empty(int(offsets[-1]), dtype=np.int64)
        values[offsets[synapse_row]] = y_pre
        values[offsets[synapse_row] + 1] = y_post
        if incoming_row.size:
            values[offsets[incoming_row] + _positions_in_group(incoming_row)] = incoming_id
        if extra_row.size:
            values[offsets[extra_row] + base[extra_row]
                   + _positions_in_group(extra_row)] = extra_id

        # ---- Agents on other ranks that this rank's lists name.
        if size > 1 and not self._partitioned and values.size:
            named = np.unique(values[values >= 0])
            named = named[~np.isin(named, ids)]
            if named.size:
                at_all = np.searchsorted(self._all_ids, named)
                at_all_clipped = np.clip(at_all, 0, self._all_ids.size - 1)
                known = self._all_ids[at_all_clipped] == named
                remote_ranks.update(zip(named[known].tolist(),
                                        all_rank[at_all_clipped[known]].tolist()))
        elif self._partitioned and x_agent.size:
            named = np.unique(extra_id[extra_id >= 0])
            named = named[~np.isin(named, ids)]
            unknown = [int(a) for a in named.tolist() if a not in remote_ranks]
            if unknown:
                raise ValueError(
                    f"add_neighbors(): neighbour {unknown[0]} is neither in this rank's "
                    "part nor named in remote_ranks")
        timings["neighbor_csr"] = time.time() - _t
        _t = time.time()

        # ---- Property columns: one row per kind and one code per agent; the values
        # given per agent overwrite slots (see _value_column).
        row_of_seq = np.full(total, -1, dtype=np.int64)
        row_of_seq[seq] = np.arange(n, dtype=np.int64)
        self._combo_codes = codes
        self._initial_values = {}
        default = self._agent_factory._property_name_2_defaults
        names = set()
        for rows in self._combo_reset_cache.values():
            names.update(rows)
        names -= {"breed", "locations"}
        property_columns = {}
        for name in sorted(names):
            table, lengths = self._kind_table(name, default)
            entries = []
            if name in self._VALUE_PROPERTIES:
                for (prop, key), column in draft.values.items():
                    if prop != name:
                        continue
                    for where, given in column:
                        rows = (row_of_seq[where:where + len(given)]
                                if isinstance(where, int) else row_of_seq[where])
                        local = rows >= 0
                        if not local.all():
                            rows, given = rows[local], given[local]
                        if rows.size:
                            entries.append((key, rows, given))
            if entries:
                property_columns[name] = self._value_column(name, table, lengths, codes, entries)
            else:
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
            ids, breed_indices, property_columns,
            offsets.astype(np.int32), values, remote_ranks)

        # The neighbour table is kept model-side too: SAGESim clears its per-agent lists
        # for a prebuilt table, and get_synapse_connectivity() reads this copy.
        self._nbr_offsets = offsets
        self._nbr_values = values
        self._agent_ids = ids
        self._soma_ids_set = None          # materialised lazily, see the properties
        self._synapse_ids_set = None
        self._num_synapses = n_y
        self._input_synapse_ids = y_ids[y_pre < 0]
        self._built = True

        # Property writes made before setup() went to the columns of an earlier build of
        # this draft; make them again, in order. The same for a frozen learning mode.
        for setter, agent_id, property_name, value in self._host_writes:
            getattr(Model, setter)(self, agent_id, property_name, value)
        if self._saved_stdp_type:
            frozen = list(self._saved_stdp_type)
            self._saved_stdp_type = dict(self._read_stdp_type(frozen))
            self._write_stdp_type({sid: -1.0 for sid in frozen})

        timings["bookkeeping"] = time.time() - _t
        self._load_timings = timings

    def _kind_table(self, name: str, default: dict):
        """Rows of one property, one per kind (NaN padded), and their lengths. In
        float64: the host returns a parameter as it was given until the device, which
        holds float32, has run."""
        rows = [[float(x) for x in self._combo_reset_cache[code].get(name, default.get(name, []))]
                for code in range(len(self._combo_table))]
        width = max((len(r) for r in rows), default=0)
        table = np.full((max(len(rows), 1), width), np.nan, dtype=np.float64)
        lengths = np.zeros(max(len(rows), 1), dtype=np.int32)
        for k, r in enumerate(rows):
            if r:
                table[k, :len(r)] = r
            lengths[k] = len(r)
        return table, lengths

    #: Above this many distinct rows a value column is stored one row per agent.
    _MAX_SHARED_ROWS = 65536

    def _value_column(self, name: str, table, lengths, codes, entries):
        """A property column with per-agent values: every agent starts from its kind's
        row and the given values overwrite their slots (later ones win).

        Packed for SAGESim by what came out: few distinct rows (a handful of weights,
        say) stay a small table plus one code per agent, which the device can share;
        many distinct rows (random or learned weights) become one row per agent. Either
        way no Python object is made per row, and the host keeps the values as they were
        given (float64) until the device, which holds float32, has run. The packed arrays
        are kept as the state reset() returns to.
        """
        dense = table[codes]
        row_lengths = lengths[codes]
        touched = set()
        for key, rows, given in entries:
            slot_of_kind = np.array(
                [keys.get(name, []).index(key) if key in keys.get(name, []) else -1
                 for keys in self._kind_keys], dtype=np.int64)
            slots = slot_of_kind[codes[rows]]
            valid = ~np.isnan(given)
            if not valid.all():
                rows, slots, given = rows[valid], slots[valid], given[valid]
            if np.any(slots < 0):
                bad = self._combo_table[int(codes[rows[slots < 0][0]])]
                raise ValueError(
                    f"{key!r} is not a {name} name of {bad.component_class} "
                    f"{bad.breed}/{bad.config}")
            dense[rows, slots] = given
            touched.update(np.unique(slot_of_kind[slot_of_kind >= 0]).tolist())

        # Rows can only differ in their kind and in the slots that were given values, so
        # distinct rows are looked for on those columns alone (each costs a sort), first
        # on a sample: random or learned weights are recognised there and go dense.
        n = len(dense)
        limit = self._MAX_SHARED_ROWS
        shared = None
        if n:
            keyed = np.column_stack([codes.astype(np.float64)]
                                    + [dense[:, slot] for slot in sorted(touched)])
            probe = keyed if n <= 4 * limit else keyed[::n // (4 * limit)]
            if len(_unique_rows(probe)[0]) <= limit:
                distinct, inverse = _unique_rows(keyed)
                if len(distinct) <= limit:
                    inverse = np.asarray(inverse, dtype=np.int32).ravel()
                    first = np.empty(len(distinct), dtype=np.intp)
                    first[inverse] = np.arange(n)            # one row of each group
                    shared = (np.ascontiguousarray(dense[first]), inverse,
                              row_lengths[first].astype(np.int32))
        if shared is not None:
            self._initial_values[name] = shared
            return IndexedColumn(shared[0].copy(), shared[1], shared[2])
        self._initial_values[name] = (dense, row_lengths)
        return ArrayColumn(dense.copy(), row_lengths.copy())

    def _initial_column(self, name: str):
        """A fresh column holding what the property was built with (for reset())."""
        kept = self._initial_values.get(name)
        if kept is None:
            table, lengths = self._kind_table(name, self._agent_factory._property_name_2_defaults)
            return IndexedColumn(table, self._combo_codes, lengths)
        if len(kept) == 3:
            return IndexedColumn(kept[0].copy(), kept[1], kept[2])
        dense, row_lengths = kept
        return ArrayColumn(dense.copy(), row_lengths.copy())

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
        self._ensure_built()
        # On one rank every agent is local, so an unknown id is an error right away.
        # Under MPI the collective idiom (every rank calls with the same ids) must
        # tolerate ids owned elsewhere; only the add_local_* variants insist.
        if local or MPI.COMM_WORLD.Get_size() == 1:
            owned = self._agent_factory._rank2agentid2agentidx.get(MPI.COMM_WORLD.Get_rank(), {})
            uniq = np.unique(ids)
            unknown = uniq[rows_of(owned, uniq) < 0].tolist()
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
        rows_u = np.where(rows_of(owned, uniq) >= 0, rows_of(buf.agent_id_to_index, uniq), -1)
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
                # event stepping: an input event wakes its synapse at every woken
                # priority (the synapse kernel, and the learning rule if rules exist)
                *[line for p in sorted(self._wake_config)
                  for line in (f'\t\t\t_wb{p} = jit.atomic_add(wake_count_{p}, 0, 1)',
                               f'\t\t\twake_queue_{p}[_wb{p}] = int(ev_syn[_k])')],
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
        # Size the record for THIS launch: room for a quarter of the local somas firing
        # on every tick (250 Hz at dt = 1 ms), at least 10k spikes; grow when a longer
        # launch needs more. Only somas spike, so the count must not include synapses:
        # sized by agents, a fused 400-tick Brunel launch allocated and NaN-filled 400 MB
        # for 162k spikes. A soma fires at most once per tick, so a one-tick launch (every
        # multi-rank run, and verbose timing) gets room for all of them: the 10k floor
        # overflowed when a bursting 12,500-soma rank fired 10,085 in one tick. Overflow is
        # caught by the guard in the recording code and reported by _process_kernel_extras.
        n_somas = len(self._soma_ids) if self._recorded_soma_ids is None else len(self._recorded_soma_ids)
        max_slots = max(10000, n_somas, n_somas * sync_ticks // 4)
        if self._spike_record_gpu is None or self._spike_record_gpu.size < max_slots * 2:
            self._spike_record_gpu = cp.full(max_slots * 2, cp.nan, dtype=cp.float32)
        if self._spike_record_count_gpu is None:
            self._spike_record_count_gpu = cp.zeros(1, dtype=cp.int32)
        self._spike_record_count_gpu[0] = 0
        # Build spike mask: 1.0 for target somas, 0.0 for others. One scatter: a
        # per-soma device write was 33 ms per launch on Cora (2,715 somas), and the mask
        # is rebuilt after every reset().
        if self._spike_mask_gpu is None:
            buf = self._gpu_buffers
            mask = cp.zeros(buf.agent_capacity, dtype=cp.float32)
            # all somas (synapses produce no output spikes), or the recorded subset
            ids = self._soma_ids if self._recorded_soma_ids is None else self._recorded_soma_ids
            idx = rows_of(buf.agent_id_to_index, ids)
            idx = idx[(idx >= 0) & (idx < num_local_agents)]
            if idx.size:
                mask[cp.asarray(idx)] = 1.0
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
                f"spike record overflow: {count:,} spikes in one kernel launch but room for "
                f"{capacity:,} (sized as max(10000, somas, somas * ticks / 4)). Call simulate() "
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

