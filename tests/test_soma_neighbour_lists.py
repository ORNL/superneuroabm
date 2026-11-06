"""Who is whose neighbour.

A synapse's neighbours are its pre and post soma; every synapse kernel reads them. A soma
has a neighbour list (its incoming synapses) only when its kernel uses ``locations``:
synapses deliver into the soma's inbox, so the built-in somas never look at their synapses
and nothing is built, stored or exchanged for them. Both halves must hold on every build
route. docs/API_DESIGN_NOTES.md (2026-11-06) has the measurements behind this.
"""

import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from superneuroabm.model import NeuromorphicModel, _reads_neighbours
from superneuroabm.step_functions.soma.lif import lif_soma_step_func
from superneuroabm.step_functions.synapse.delivery import delivery_synapse_step_func
from tests.step_funcs.soma_reads_afferents import soma_reads_afferents

READER = "soma_reads_afferents"
READER_PATH = Path(__file__).resolve().parent / "step_funcs" / "soma_reads_afferents.py"
WEIGHTS = (1.0, 2.0, 4.0)          # three synapses A -> B
EXTERNAL_WEIGHT = 8.0              # one external input -> A
A, B, EXT, FIRST = 0, 1, 2, 3      # ids on the bulk routes


def _model(soma_breed):
    model = NeuromorphicModel(enable_internal_states_tracking=False)
    if soma_breed == READER:
        model.register_soma_type(name=READER, step_func=soma_reads_afferents,
                                 step_func_path=READER_PATH)
        model._component_configurations["soma"][READER] = {"config_0": {
            "hyperparameters": {"unused": 0.0},
            "internal_states": {"n_afferents": -1.0, "weight_sum": -1.0}}}
    return model


def _weight(value):
    return {"hyperparameters": {"weight": value}}


def _per_agent(soma_breed):
    model = _model(soma_breed)
    a = model.create_soma(breed=soma_breed, config_name="config_0")
    b = model.create_soma(breed=soma_breed, config_name="config_0")
    ext = model.create_synapse(breed="delivery_synapse", pre_soma_id=-1, post_soma_id=a,
                               config_name="config_0", overrides=_weight(EXTERNAL_WEIGHT))
    syns = [model.create_synapse(breed="delivery_synapse", pre_soma_id=a, post_soma_id=b,
                                 config_name="config_0", overrides=_weight(w)) for w in WEIGHTS]
    return model, a, b, ext, syns


def _records():
    synapses = [{"id": EXT, "pre": -1, "post": A, "overrides": _weight(EXTERNAL_WEIGHT)}]
    synapses += [{"id": FIRST + k, "pre": A, "post": B, "overrides": _weight(w)}
                 for k, w in enumerate(WEIGHTS)]
    return [{"id": A}, {"id": B}], synapses


def _plural(soma_breed):
    model = _model(soma_breed)
    a, b = (int(i) for i in model.create_somas(2, soma_breed, "config_0"))
    ext = int(model.create_synapses(-1, [a], "delivery_synapse", "config_0",
                                    overrides=_weight(EXTERNAL_WEIGHT))[0])
    syns = model.create_synapses(a, np.full(len(WEIGHTS), b), "delivery_synapse", "config_0",
                                 overrides=_weight(np.array(WEIGHTS)))
    return model, a, b, ext, [int(s) for s in syns]


def _from_records(soma_breed):
    model = _model(soma_breed)
    somas, synapses = _records()
    model.load_network({"somas": somas, "synapses": synapses}, soma_breed=soma_breed)
    return model, A, B, EXT, [FIRST + k for k in range(len(WEIGHTS))]


def _from_arrays(soma_breed):
    model = _model(soma_breed)
    n = len(WEIGHTS)
    model.load_network({
        "schema": "columnar_post_owned_v1",
        "soma_ids": np.array([A, B]), "soma_breed": soma_breed, "soma_config": "config_0",
        "synapse_ids": np.arange(EXT, EXT + 1 + n),
        "pre": np.array([-1] + [A] * n), "post": np.array([A] + [B] * n),
        "synapse_breed": "delivery_synapse", "synapse_config": "config_0",
        "learning_rule": "", "learning_rule_config": "",
        "syn_hp_keys": np.array(["weight"]),
        "syn_hp_vals": np.array([[EXTERNAL_WEIGHT]] + [[w] for w in WEIGHTS]),
    })
    return model, A, B, EXT, [FIRST + k for k in range(n)]


def _from_file(soma_breed):
    model = _model(soma_breed)
    somas, synapses = _records()
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "network.pkl"
        with open(path, "wb") as fh:
            pickle.dump({"somas": somas, "synapses": synapses}, fh)
        model.load_network(str(path), soma_breed=soma_breed)
    return model, A, B, EXT, [FIRST + k for k in range(len(WEIGHTS))]


ROUTES = (_per_agent, _plural, _from_records, _from_arrays, _from_file)


def _single_rank_only(test):
    from mpi4py import MPI
    if MPI.COMM_WORLD.Get_size() > 1:
        test.skipTest("single-rank test")


class TestSomaNeighbourLists(unittest.TestCase):
    def test_which_kernels_read_their_neighbours(self):
        self.assertFalse(_reads_neighbours(lif_soma_step_func, 2))
        self.assertTrue(_reads_neighbours(delivery_synapse_step_func, 2))
        self.assertTrue(_reads_neighbours(soma_reads_afferents, 2))
        model = NeuromorphicModel()
        for breed in model._soma_breeds:                       # every built-in soma
            self.assertFalse(model._soma_keeps_neighbour_list(breed), breed)

    def test_synapse_lists_pre_and_post_and_a_builtin_soma_lists_nothing(self):
        _single_rank_only(self)
        for route in ROUTES:
            with self.subTest(route=route.__name__):
                model, a, b, ext, syns = route("lif_soma")
                self.assertEqual(model.get_synapse_connectivity(ext), [-1, a])
                for s in syns:
                    self.assertEqual(model.get_synapse_connectivity(s), [a, b])
                model.setup()
                model.simulate(ticks=2)
                # what the kernels were handed on the device
                for soma in (a, b):
                    self.assertEqual(list(model.get_agent_property_value(soma, "locations")), [])
                self.assertEqual(list(model.get_agent_property_value(syns[0], "locations")), [a, b])

    def test_a_soma_whose_kernel_reads_its_synapses_gets_them(self):
        _single_rank_only(self)
        for route in ROUTES:
            with self.subTest(route=route.__name__):
                model, a, b, ext, syns = route(READER)
                self.assertTrue(model._soma_keeps_neighbour_list(READER))
                model.setup()
                model.simulate(ticks=2)
                self.assertEqual(sorted(model.get_agent_property_value(b, "locations")), sorted(syns))
                self.assertEqual(list(model.get_agent_property_value(a, "locations")), [ext])
                # creation order, whatever order the synapse rows are laid out in
                self.assertEqual(list(model.get_agent_property_value(b, "locations")), syns)
                count_a, sum_a = model.get_agent_property_value(a, "internal_states")[:2]
                count_b, sum_b = model.get_agent_property_value(b, "internal_states")[:2]
                self.assertEqual((count_a, sum_a), (1.0, EXTERNAL_WEIGHT))
                self.assertEqual((count_b, sum_b), (float(len(WEIGHTS)), sum(WEIGHTS)))

    def test_the_removed_getters_are_gone(self):
        model = NeuromorphicModel()
        self.assertFalse(hasattr(model, "get_neighbors"))
        self.assertFalse(hasattr(model, "get_soma_outgoing_synapses"))


if __name__ == "__main__":
    unittest.main()
