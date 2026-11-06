#!/usr/bin/env python
"""load_network(): a rank builds its own part of a network; and add_neighbors().

The headline case is a synapse that is NOT on its post soma's rank (the "split synapse"):

    external_input(pre=-1) -> A      [rank 0]
    A --S--> B                       S is in rank 0's part; its post soma B is on rank 1

A synapse delivers by adding into its post soma's inbox. On rank 0 that is B's ghost row,
and SAGESim sums ghost deposits back into the owner's row, so where a synapse sits never
changes the result (measured on a Brunel network at 1-4 ranks, docs/API_DESIGN_NOTES.md,
2026-11-06). Pass criterion: B's spike train equals that of the same topology built with
create_soma / create_synapse.

Run with 1 and 2 ranks (tests/run_mpi_tests.sh); other rank counts skip the split cases.
"""

import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR.parent))

from superneuroabm.model import NeuromorphicModel
from tests.step_funcs.soma_reads_afferents import soma_reads_afferents

# Fixed global agent ids for the split-synapse fixture.
A = 0           # presynaptic soma (rank 0)
B = 1           # postsynaptic soma (rank 1 in the 2-rank case)
S_EXT = 10      # external-input synapse into A (rank 0)
S = 11          # the A->B synapse (in rank 0's part, with its pre)

INPUT_TICK = 2
SIM_TICKS = 60
READER = "soma_reads_afferents"
READER_PATH = CURRENT_DIR / "step_funcs" / "soma_reads_afferents.py"


def _get_mpi():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    return comm, comm.Get_rank(), comm.Get_size()


def _split_parts(size, soma_b=None):
    """Each rank's part of the split-synapse network (one part holding everything for
    size 1). ``soma_b`` is B's record when it needs more than its id."""
    soma_b = soma_b or {"id": B}
    syn_ext = {"id": S_EXT, "pre": -1, "post": A}
    syn_s = {"id": S, "pre": A, "post": B}
    if size == 1:
        return [{"somas": [{"id": A}, soma_b], "synapses": [syn_ext, syn_s]}]
    if size == 2:
        return [{"somas": [{"id": A}], "synapses": [syn_ext, syn_s], "remote_ranks": {B: 1}},
                {"somas": [soma_b], "synapses": [], "remote_ranks": {}}]
    raise ValueError(f"fixture supports size 1 or 2, got {size}")


def _run(model):
    model.setup()
    # add_spike is collective; ranks that do not hold S_EXT drop it.
    model.add_spike(synapse_id=S_EXT, tick=INPUT_TICK, value=1)
    model.simulate(ticks=SIM_TICKS, update_data_ticks=1)
    return model.get_spike_times(B)          # collective getter


def _reference_spike_times(stepping="event"):
    """Spike train of B for the A->B topology built with the create calls. Every rank
    must build it (setup / simulate are collective); the synapse sits with B there."""
    model = NeuromorphicModel(enable_internal_states_tracking=False, synapse_stepping=stepping)
    a = model.create_soma(breed="lif_soma", config_name="config_0")
    b = model.create_soma(breed="lif_soma", config_name="config_0")
    ext = model.create_synapse(breed="delivery_synapse", pre_soma_id=-1,
                               post_soma_id=a, config_name="config_0")
    model.create_synapse(breed="delivery_synapse", pre_soma_id=a,
                         post_soma_id=b, config_name="config_0")
    model.setup()
    model.add_spike(synapse_id=ext, tick=INPUT_TICK, value=1)
    model.simulate(ticks=SIM_TICKS, update_data_ticks=1)
    return model.get_spike_times(b)


class TestSplitSynapse(unittest.TestCase):
    """A synapse in the part of its PRE soma's rank; its post soma on another rank."""

    def _check(self, stepping):
        comm, rank, size = _get_mpi()
        if size not in (1, 2):
            self.skipTest(f"fixture supports 1 or 2 ranks, got {size}")
        model = NeuromorphicModel(enable_internal_states_tracking=False,
                                  synapse_stepping=stepping)
        model.load_network(_split_parts(size)[rank])
        got = _run(model)
        expected = _reference_spike_times(stepping)
        if rank == 0:
            self.assertGreater(len(expected), 0)
            self.assertEqual(sorted(got), sorted(expected),
                             f"size={size}, {stepping}: B did not get the delivery of a "
                             "synapse held by another rank")

    def test_event_stepping(self):
        self._check("event")

    def test_clocked_stepping(self):
        self._check("clocked")

    def test_part_from_a_file(self):
        """The same parts read from .pkl files."""
        comm, rank, size = _get_mpi()
        if size not in (1, 2):
            self.skipTest(f"fixture supports 1 or 2 ranks, got {size}")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / f"part_{rank}.pkl"
            with open(path, "wb") as fh:
                pickle.dump(_split_parts(size)[rank], fh)
            model = NeuromorphicModel(enable_internal_states_tracking=False)
            model.load_network(str(path))
        got = _run(model)
        expected = _reference_spike_times()
        if rank == 0:
            self.assertEqual(sorted(got), sorted(expected))

    def test_a_soma_reading_a_synapse_held_by_another_rank(self):
        """A soma whose own kernel reads its incoming synapses finds only those of its
        part; the synapse another rank holds is named with add_neighbors() and reaches
        the soma as a ghost."""
        comm, rank, size = _get_mpi()
        if size not in (1, 2):
            self.skipTest(f"fixture supports 1 or 2 ranks, got {size}")
        weight = 3.0
        model = NeuromorphicModel(enable_internal_states_tracking=False)
        model.register_soma_type(name=READER, step_func=soma_reads_afferents,
                                 step_func_path=READER_PATH)
        model._component_configurations["soma"][READER] = {"config_0": {
            "hyperparameters": {"unused": 0.0},
            "internal_states": {"n_afferents": -1.0, "weight_sum": -1.0}}}
        part = _split_parts(size, soma_b={"id": B, "breed": READER})[rank]
        for synapse in part["synapses"]:
            if synapse["id"] == S:
                synapse["overrides"] = {"hyperparameters": {"weight": weight}}
        if size == 2 and rank == 1:
            part["remote_ranks"] = {S: 0}            # B's rank is told where S lives
        model.load_network(part)
        if size == 2 and rank == 1:
            model.add_neighbors(B, S)                # ... and that B reads it
        # the reader kernel reads its synapses' hyperparameters, also on a ghost row
        model.set_property_neighbor_visible("hyperparameters", True)
        model.setup()
        model.simulate(ticks=2, update_data_ticks=1)
        count, total = model.get_agent_property_value(B, "internal_states")[:2]
        if rank == 0:
            self.assertEqual((count, total), (1.0, weight))


class TestLoadNetwork(unittest.TestCase):
    def setUp(self):
        if _get_mpi()[2] > 1:
            self.skipTest("single-rank test")

    def test_records_match_the_create_calls(self):
        model = NeuromorphicModel(enable_internal_states_tracking=False)
        model.load_network(_split_parts(1)[0])
        self.assertEqual(sorted(_run(model)), sorted(_reference_spike_times()))

    def test_arrays_match_the_create_calls(self):
        model = NeuromorphicModel(enable_internal_states_tracking=False)
        model.load_network({
            "schema": "columnar_post_owned_v1",
            "soma_ids": np.array([A, B]), "soma_breed": "lif_soma", "soma_config": "config_0",
            "synapse_ids": np.array([S_EXT, S]), "pre": np.array([-1, A]),
            "post": np.array([A, B]),
            "synapse_breed": "delivery_synapse", "synapse_config": "config_0",
            "learning_rule": "", "learning_rule_config": "",
        })
        self.assertEqual(sorted(_run(model)), sorted(_reference_spike_times()))

    def test_arrays_can_mix_kinds(self):
        """A name per agent instead of one name: each agent gets its own breed, config
        and learning rule (the array layout used to allow one of each)."""
        model = NeuromorphicModel(enable_internal_states_tracking=False)
        model.load_network({
            "schema": "columnar_post_owned_v1",
            "soma_ids": np.array([A, B]),
            "soma_breed": np.array(["lif_soma", "izh_soma"]), "soma_config": "config_0",
            "synapse_ids": np.array([S_EXT, S]), "pre": np.array([-1, A]),
            "post": np.array([A, B]),
            "synapse_breed": "delivery_synapse",
            "synapse_config": np.array(["config_0", "config_1"]),
            "learning_rule": np.array(["", "exp_pair_wise_stdp"]),
            "learning_rule_config": "default",
            "syn_hp_keys": np.array(["weight"]),
            "syn_hp_vals": np.array([[np.nan], [7.5]]),        # NaN: the config's weight
        })
        self.assertEqual(model.get_agent_breed(A), "lif_soma")
        self.assertEqual(model.get_agent_breed(B), "izh_soma")
        self.assertEqual(model.get_agent_config_name(S_EXT), "config_0")
        self.assertEqual(model.get_agent_config_name(S), "config_1")
        self.assertIsNone(model.agentid2learning_rule[S_EXT])
        self.assertEqual(model.agentid2learning_rule[S], ("exp_pair_wise_stdp", "default"))
        self.assertEqual(model.get_hyperparameters(S)["weight"], 7.5)
        config_weight = model._component_configurations["synapse"]["delivery_synapse"][
            "config_0"]["hyperparameters"]["weight"]
        self.assertEqual(model.get_hyperparameters(S_EXT)["weight"], float(config_weight))

    def test_post_soma_on_another_rank_must_be_named(self):
        """A post soma that is neither in the part nor in remote_ranks is an error; one
        named in remote_ranks is accepted (there is no placement rule)."""
        part = {"somas": [{"id": A}], "synapses": [{"id": S, "pre": A, "post": B}]}
        with self.assertRaises(ValueError):
            NeuromorphicModel(enable_internal_states_tracking=False).load_network(part)
        model = NeuromorphicModel(enable_internal_states_tracking=False)
        model.load_network({**part, "remote_ranks": {B: 1}})
        self.assertEqual(model.get_synapse_connectivity(S), [A, B])

    def test_explicit_neighbour_lists_are_refused(self):
        """The old explicit-neighbours layout fails loudly instead of loading empty."""
        part = {"somas": [{"id": A, "neighbors": [S]}],
                "synapses": [{"id": S, "neighbors": [-1, A]}]}
        with self.assertRaises(ValueError):
            NeuromorphicModel(enable_internal_states_tracking=False).load_network(part)

    def test_one_route_per_model(self):
        model = NeuromorphicModel(enable_internal_states_tracking=False)
        model.load_network(_split_parts(1)[0])
        with self.assertRaises(RuntimeError):
            model.create_soma(breed="lif_soma", config_name="config_0")
        with self.assertRaises(RuntimeError):
            model.load_network(_split_parts(1)[0])
        other = NeuromorphicModel(enable_internal_states_tracking=False)
        other.create_soma(breed="lif_soma", config_name="config_0")
        with self.assertRaises(RuntimeError):
            other.load_network(_split_parts(1)[0])

    def test_nothing_is_created_after_setup(self):
        model = NeuromorphicModel(enable_internal_states_tracking=False)
        a = model.create_soma(breed="lif_soma", config_name="config_0")
        model.setup()
        with self.assertRaises(RuntimeError):
            model.create_soma(breed="lif_soma", config_name="config_0")
        with self.assertRaises(RuntimeError):
            model.add_neighbors(a, a)


class TestCreateCallsOnAnyRankCount(unittest.TestCase):
    """The create calls describe the whole network on every rank and SuperNeuroABM splits
    it; these run at every rank count (every rank makes the same calls)."""

    N, K = 40, 6

    def _wiring(self):
        rng = np.random.default_rng(3)
        post = np.repeat(np.arange(self.N), self.K)
        pre = rng.integers(0, self.N, size=post.size)
        weights = rng.uniform(20.0, 60.0, size=post.size)
        return pre, post, weights

    def _run(self, model, somas, inputs):
        model.setup()
        model.add_spikes(np.asarray(inputs)[::2], np.full(len(inputs[::2]), INPUT_TICK), 1.0)
        model.simulate(ticks=SIM_TICKS, update_data_ticks=1)
        return {int(s): sorted(model.get_spike_times(int(s))) for s in somas}

    def test_array_calls_match_single_calls(self):
        pre, post, weights = self._wiring()
        single = NeuromorphicModel(enable_internal_states_tracking=False)
        somas = [single.create_soma(breed="lif_soma", config_name="config_0")
                 for _ in range(self.N)]
        inputs = [single.create_synapse(breed="delivery_synapse", pre_soma_id=-1,
                                        post_soma_id=s, config_name="config_1") for s in somas]
        for p, q, w in zip(pre.tolist(), post.tolist(), weights.tolist()):
            single.create_synapse(breed="delivery_synapse", pre_soma_id=somas[p],
                                  post_soma_id=somas[q], config_name="config_0",
                                  overrides={"hyperparameters": {"weight": w}})
        expected = self._run(single, somas, inputs)

        arrays = NeuromorphicModel(enable_internal_states_tracking=False)
        ids = arrays.create_somas(self.N, breed="lif_soma", config_name="config_0")
        ins = arrays.create_synapses(-1, ids, breed="delivery_synapse", config_name="config_1")
        arrays.create_synapses(ids[pre], ids[post], breed="delivery_synapse",
                               config_name="config_0",
                               overrides={"hyperparameters": {"weight": weights}})
        got = self._run(arrays, ids, ins)

        if _get_mpi()[1] == 0:
            self.assertEqual(ids.tolist(), somas)
            self.assertGreater(sum(map(len, expected.values())), self.N)
            self.assertEqual(got, expected)

    def test_extra_neighbour_held_by_another_rank(self):
        """add_neighbors() with the create calls: the neighbour may be dealt to another
        rank; the reading soma gets it as a ghost."""
        model = NeuromorphicModel(enable_internal_states_tracking=False)
        model.register_soma_type(name=READER, step_func=soma_reads_afferents,
                                 step_func_path=READER_PATH)
        model._component_configurations["soma"][READER] = {"config_0": {
            "hyperparameters": {"unused": 0.0},
            "internal_states": {"n_afferents": -1.0, "weight_sum": -1.0}}}
        a = model.create_soma(breed=READER, config_name="config_0")
        b = model.create_soma(breed=READER, config_name="config_0")   # the next rank, if any
        ext = model.create_synapse(breed="delivery_synapse", pre_soma_id=-1, post_soma_id=a,
                                   config_name="config_0",
                                   overrides={"hyperparameters": {"weight": 2.0}})
        ab = model.create_synapse(breed="delivery_synapse", pre_soma_id=a, post_soma_id=b,
                                  config_name="config_0",
                                  overrides={"hyperparameters": {"weight": 5.0}})
        model.add_neighbors(a, ab)                 # ab sits with b
        model.set_property_neighbor_visible("hyperparameters", True)
        model.setup()
        model.simulate(ticks=2, update_data_ticks=1)
        state_a = model.get_agent_property_value(a, "internal_states")[:2]
        state_b = model.get_agent_property_value(b, "internal_states")[:2]
        if _get_mpi()[1] == 0:
            self.assertEqual(tuple(state_a), (2.0, 7.0))       # ext and ab
            self.assertEqual(tuple(state_b), (1.0, 5.0))       # ab
            self.assertEqual(model.get_synapse_connectivity(a), [ext, ab])


class TestAddNeighbors(unittest.TestCase):
    def setUp(self):
        if _get_mpi()[2] > 1:
            self.skipTest("single-rank test")

    def _network(self):
        model = NeuromorphicModel(enable_internal_states_tracking=False)
        a = model.create_soma(breed="lif_soma", config_name="config_0")
        b = model.create_soma(breed="lif_soma", config_name="config_0")
        ext = model.create_synapse(breed="delivery_synapse", pre_soma_id=-1,
                                   post_soma_id=a, config_name="config_0")
        ab = model.create_synapse(breed="delivery_synapse", pre_soma_id=a,
                                  post_soma_id=b, config_name="config_0")
        return model, a, b, ext, ab

    def test_any_agent_any_neighbour(self):
        """A synapse gets a synapse, a soma gets a soma and a synapse; extras come after
        pre and post, in the order given, and the kernels see them in ``locations``."""
        model, a, b, ext, ab = self._network()
        model.add_neighbors(ab, ext)                           # one pair
        model.add_neighbors([b, b, ab], [a, ab, b])            # arrays
        self.assertEqual(model.get_synapse_connectivity(ab), [a, b, ext, b])
        self.assertEqual(model.get_synapse_connectivity(ext), [-1, a])
        self.assertEqual(model.get_synapse_connectivity(b), [a, ab])
        self.assertEqual(model.get_synapse_connectivity(a), [])
        model.setup()
        model.simulate(ticks=2)
        self.assertEqual(list(model.get_agent_property_value(ab, "locations")), [a, b, ext, b])
        self.assertEqual(list(model.get_agent_property_value(b, "locations")), [a, ab])

    def test_extras_do_not_change_what_the_built_in_kernels_do(self):
        plain, a, b, ext, ab = self._network()
        plain.setup()
        plain.add_spike(ext, INPUT_TICK, 1)
        plain.simulate(ticks=SIM_TICKS)
        extra, a, b, ext, ab = self._network()
        extra.add_neighbors([ab, b], [ext, a])
        extra.setup()
        extra.add_spike(ext, INPUT_TICK, 1)
        extra.simulate(ticks=SIM_TICKS)
        self.assertGreater(len(plain.get_spike_times(b)), 0)
        self.assertEqual(plain.get_spike_times(b), extra.get_spike_times(b))

    def test_an_agent_with_neither_pre_nor_post(self):
        model, a, b, ext, ab = self._network()
        free = model.create_synapse(breed="delivery_synapse", pre_soma_id=-1,
                                    post_soma_id=-1, config_name="config_0")
        model.add_neighbors(free, [ab, ext])
        self.assertEqual(model.get_synapse_connectivity(free), [-1, -1, ab, ext])
        model.setup()
        model.simulate(ticks=2)
        self.assertEqual(list(model.get_agent_property_value(free, "locations")),
                         [-1, -1, ab, ext])

    def test_a_synapse_in_the_pre_slot(self):
        model, a, b, ext, ab = self._network()
        chained = model.create_synapse(breed="delivery_synapse", pre_soma_id=ab,
                                       post_soma_id=b, config_name="config_0")
        self.assertEqual(model.get_synapse_connectivity(chained), [ab, b])

    def test_under_load_network_a_neighbour_elsewhere_must_be_named(self):
        model = NeuromorphicModel(enable_internal_states_tracking=False)
        model.load_network(_split_parts(1)[0])
        model.add_neighbors(B, S)                              # in the part: fine
        self.assertEqual(model.get_synapse_connectivity(B), [S])
        model.add_neighbors(B, 999)                            # nowhere
        with self.assertRaises(ValueError):
            model.get_synapse_connectivity(B)


if __name__ == "__main__":
    unittest.main()
