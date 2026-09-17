#!/usr/bin/env python
"""The combo store: what each agent IS, without a dict entry per agent.

`agentid2config` / `agentid2overrides` / `agentid2learning_rule` used to be three dict
entries per agent, written by the record builders and **skipped entirely** by the
columnar one -- which is why, on a columnar-loaded model, the named-parameter API raised
`KeyError: None`, `eval()`/`train()` were silent no-ops, and `reset()` silently left every
synapse untouched. They are now read-only views over one int32 code per agent plus a table
of distinct agent descriptions.

These tests pin the behaviour that was broken, so a future builder cannot drop the
bookkeeping again without a red test.

Usage:
    python -m pytest tests/test_combo_store.py
"""

import sys
import unittest
from pathlib import Path

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR.parent))

from superneuroabm.model import NeuromorphicModel
from superneuroabm.brunel import brunel_partition


BASE = dict(somas_per_rank=12, excitatory_in_degree=4, inhibitory_in_degree=2,
            external_synapses_per_soma=1, inhibitory_weight_ratio=5.0, seed=5,
            num_partitions=1, partition_rank=0, excitatory_weight=14.0,
            external_weight=14.0, synaptic_delay_ms=1.5)


def _columnar_model(learning_rule=None):
    """A columnar-built model, optionally with a learning rule on every synapse.

    brunel_partition never emits a learning rule, so the plastic case patches the
    array the loader reads -- the same field a producer would write.
    """
    arrays = dict(brunel_partition(output_format="columns", **BASE))
    if learning_rule is not None:
        arrays["learning_rule"] = np.asarray(learning_rule)
        arrays["learning_rule_config"] = np.asarray("default")
    m = NeuromorphicModel(enable_internal_states_tracking=False)
    m.load_post_owned(arrays)
    return m


def _record_model():
    rec = brunel_partition(output_format="records", **BASE)
    m = NeuromorphicModel(enable_internal_states_tracking=False)
    m.load_post_owned(rec)
    return m


def _a_synapse(model):
    """Any locally-owned synapse id of a columnar-built model."""
    return next(aid for aid in model._agent_factory._rank2agentid2agentidx[0]
                if model._component_class_of(aid) == "synapse")


class TestComboStoreOnColumnarBuild(unittest.TestCase):
    """Everything here raised or silently did nothing before the combo store."""

    def test_named_parameter_read_works(self):
        m = _columnar_model()
        syn = _a_synapse(m)
        hp = m.get_hyperparameters(syn)          # was KeyError: None
        self.assertIn("weight", hp)
        self.assertIn("tau_fall", hp)

    def test_named_parameter_write_works(self):
        m = _columnar_model()
        syn = _a_synapse(m)
        m.set_hyperparameters(syn, {"weight": 3.25})
        self.assertAlmostEqual(m.get_hyperparameters(syn)["weight"], 3.25, places=5)

    def test_config_diff_works(self):
        m = _columnar_model()
        syn = _a_synapse(m)
        diff = m.get_agent_config_diff(syn)       # was KeyError: None
        self.assertIn("hyperparameters", diff)

    def test_component_class_is_known_for_synapses(self):
        m = _columnar_model()
        syn = _a_synapse(m)
        self.assertEqual(m._component_class_of(syn), "synapse")

    def test_plastic_synapses_are_found(self):
        m = _columnar_model(learning_rule="exp_pair_wise_stdp")
        self.assertTrue(m._plastic_synapse_ids(),
                        "a columnar model with a learning rule reported no plastic "
                        "synapses, so eval()/train() would be silent no-ops")

    def test_eval_and_train_flip_stdp_type(self):
        m = _columnar_model(learning_rule="exp_pair_wise_stdp")
        syn = m._plastic_synapse_ids()[0]
        lhp_index = m._hp_key_index(syn, "learning_hyperparameters")
        before = m.get_agent_property_value(
            id=syn, property_name="learning_hyperparameters")[lhp_index["stdp_type"]]
        self.assertNotEqual(before, -1.0)

        m.eval()
        after = m.get_agent_property_value(
            id=syn, property_name="learning_hyperparameters")[lhp_index["stdp_type"]]
        self.assertEqual(after, -1.0, "eval() did not disable plasticity")

        m.train()
        restored = m.get_agent_property_value(
            id=syn, property_name="learning_hyperparameters")[lhp_index["stdp_type"]]
        self.assertEqual(restored, before, "train() did not restore the rule")

    def test_reset_actually_resets_synapses(self):
        """The branch that used to drop every synapse on the floor.

        _reset_agents tested `agent_id in self._synapse_ids`, which a columnar build
        never populates, so synapses matched neither branch and were never reset.
        """
        m = _columnar_model()
        syn = _a_synapse(m)
        idx = m._agent_factory._rank2agentid2agentidx[0][syn]
        data = m._agent_factory._property_name_2_agent_data_tensor

        default_is = list(data["internal_states"][idx])
        data["internal_states"][idx] = [123.0] * len(default_is)

        m._reset_agents(retain_parameters=False)

        self.assertEqual(list(data["internal_states"][idx]), default_is,
                         "synapse internal_states survived reset -- it was skipped")


class TestComboInterning(unittest.TestCase):

    def test_table_holds_distinct_descriptions_not_one_per_agent(self):
        m = _columnar_model()
        n_agents = len(m._agent_factory._rank2agentid2agentidx[0])
        self.assertLess(len(m._combo_table), n_agents,
                        "combo table has an entry per agent; interning is not working")
        self.assertGreaterEqual(len(m._combo_table), 2)   # at least soma + synapse

    def test_codes_cover_every_local_agent(self):
        m = _columnar_model()
        n_agents = len(m._agent_factory._rank2agentid2agentidx[0])
        self.assertEqual(len(m._combo_codes), n_agents)
        self.assertTrue((m._combo_codes >= 0).all())
        self.assertTrue((m._combo_codes < len(m._combo_table)).all())

    def test_overrides_key_normalizes_value_spelling(self):
        """'1e-3', '1E-3' and 0.001 describe the same agent.

        The config path coerces with float(), so interning on the raw value would split
        combos the model treats as identical -- Cora's overrides carry exactly those
        string spellings.
        """
        k1 = NeuromorphicModel._overrides_key({"hyperparameters": {"tau_fall": "1e-3"}})
        k2 = NeuromorphicModel._overrides_key({"hyperparameters": {"tau_fall": "1E-3"}})
        k3 = NeuromorphicModel._overrides_key({"hyperparameters": {"tau_fall": 0.001}})
        self.assertEqual(k1, k2)
        self.assertEqual(k2, k3)

    def test_empty_and_missing_overrides_agree(self):
        self.assertEqual(NeuromorphicModel._overrides_key(None),
                         NeuromorphicModel._overrides_key({}))
        self.assertEqual(NeuromorphicModel._overrides_key({"hyperparameters": {}}),
                         NeuromorphicModel._overrides_key({}))

    def test_overrides_round_trip(self):
        original = {"hyperparameters": {"weight": 2.0, "tau_fall": 0.01}}
        key = NeuromorphicModel._overrides_key(original)
        self.assertEqual(NeuromorphicModel._overrides_from_key(key), original)


class TestBreedOrder(unittest.TestCase):
    """The builder sorts by breed instead of refusing to build.

    build_from_local_columns requires non-decreasing breed indices, and a prebuilt CSR
    cannot lean on setup's sort_by_breed (reordering would desync it). The columnar
    loader used to *raise* when somas did not sort first, which rejected any model whose
    soma breed was registered after a synapse breed -- exactly what register_soma_type
    produces, since it appends.
    """

    def test_soma_breed_registered_after_synapse_breed(self):
        import copy
        from superneuroabm.step_functions.soma.lif import lif_soma_step_func

        m = NeuromorphicModel(enable_internal_states_tracking=False)
        lif_path = (CURRENT_DIR.parent / "superneuroabm" / "step_functions"
                    / "soma" / "lif.py")
        m.register_soma_type("late_lif", lif_soma_step_func, lif_path)
        m._component_configurations["soma"]["late_lif"] = copy.deepcopy(
            m._component_configurations["soma"]["lif_soma"])

        late_idx = m._soma_breeds["late_lif"]._breedidx
        syn_idx = m._synapse_breeds["single_exp_synapse"]._breedidx
        self.assertGreater(late_idx, syn_idx,
                           "test premise: the late soma breed must sort after synapses")

        arrays = dict(brunel_partition(output_format="columns", **BASE))
        arrays["soma_breed"] = np.asarray("late_lif")
        m.load_post_owned(arrays)      # used to raise RuntimeError

        # Agents come out in non-decreasing breed order, with the synapses first.
        af = m._agent_factory
        ids = list(af._rank2agentid2agentidx[0])
        breeds = [af._agent2breed[i] for i in ids]
        self.assertEqual(breeds, sorted(breeds), "agents are not in breed order")
        self.assertEqual(breeds[0], syn_idx)
        self.assertEqual(breeds[-1], late_idx)

    def test_csr_follows_the_breed_sort(self):
        """Permuting rows must permute the CSR with them, or the network is miswired."""
        import copy
        from superneuroabm.step_functions.soma.lif import lif_soma_step_func

        m = NeuromorphicModel(enable_internal_states_tracking=False)
        lif_path = (CURRENT_DIR.parent / "superneuroabm" / "step_functions"
                    / "soma" / "lif.py")
        m.register_soma_type("late_lif", lif_soma_step_func, lif_path)
        m._component_configurations["soma"]["late_lif"] = copy.deepcopy(
            m._component_configurations["soma"]["lif_soma"])
        arrays = dict(brunel_partition(output_format="columns", **BASE))
        arrays["soma_breed"] = np.asarray("late_lif")
        m.load_post_owned(arrays)

        # Reference: the unsorted build, whose rows are [somas..., synapses...].
        ref = NeuromorphicModel(enable_internal_states_tracking=False)
        ref.load_post_owned(
            dict(brunel_partition(output_format="columns", **BASE)))

        def neighbours(model):
            off = np.asarray(model.get_space()._prebuilt_csr_offsets)
            val = np.asarray(model.get_space()._prebuilt_csr_values)
            ids = list(model._agent_factory._rank2agentid2agentidx[0])
            return {ids[i]: list(val[off[i]:off[i + 1]]) for i in range(len(ids))}

        self.assertEqual(neighbours(m), neighbours(ref),
                         "each agent's neighbour list changed under the breed sort")


class TestEvalTrainScatter(unittest.TestCase):
    """eval()/train() write stdp_type grouped by (current row, new value).

    Grouping is only safe if synapses that differ keep differing. A bug here would
    collapse every plastic synapse onto one learning rule on the way back from eval().
    """

    def _two_rule_model(self):
        m = NeuromorphicModel(enable_internal_states_tracking=False)
        a = m.create_soma(breed="lif_soma", config_name="config_0")
        b = m.create_soma(breed="lif_soma", config_name="config_0")
        s1 = m.create_synapse(breed="single_exp_synapse", pre_soma_id=a, post_soma_id=b,
                              config_name="config_0",
                              learning_rule="exp_pair_wise_stdp")
        s2 = m.create_synapse(breed="single_exp_synapse", pre_soma_id=b, post_soma_id=a,
                              config_name="config_0",
                              learning_rule="exp_pair_wise_stdp_bounded")
        s3 = m.create_synapse(breed="single_exp_synapse", pre_soma_id=-1, post_soma_id=a,
                              config_name="config_0")          # no rule
        return m, s1, s2, s3

    def _stdp_type(self, m, sid):
        i = m._hp_key_index(sid, "learning_hyperparameters")["stdp_type"]
        return m.get_agent_property_value(
            id=sid, property_name="learning_hyperparameters")[i]

    def test_distinct_rules_survive_an_eval_train_round_trip(self):
        m, s1, s2, s3 = self._two_rule_model()
        before = {s: self._stdp_type(m, s) for s in (s1, s2, s3)}
        self.assertNotEqual(before[s1], before[s2],
                            "test premise: the two rules must have different stdp_type")
        self.assertEqual(before[s3], -1.0, "a rule-less synapse should already be -1")

        m.eval()
        self.assertEqual(self._stdp_type(m, s1), -1.0)
        self.assertEqual(self._stdp_type(m, s2), -1.0)

        m.train()
        self.assertEqual(self._stdp_type(m, s1), before[s1],
                         "train() gave synapse 1 the wrong rule back")
        self.assertEqual(self._stdp_type(m, s2), before[s2],
                         "train() gave synapse 2 the wrong rule back")
        self.assertEqual(self._stdp_type(m, s3), -1.0,
                         "train() resurrected a rule on a synapse that never had one")

    def test_round_trip_on_a_bulk_built_model(self):
        m = _columnar_model(learning_rule="exp_pair_wise_stdp")
        plastic = m._plastic_synapse_ids()
        before = {s: self._stdp_type(m, s) for s in plastic[:5]}
        m.eval()
        for s in before:
            self.assertEqual(self._stdp_type(m, s), -1.0)
        m.train()
        for s, v in before.items():
            self.assertEqual(self._stdp_type(m, s), v)


class TestLazyIdSets(unittest.TestCase):
    """_soma_ids / _synapse_ids are correct on every path, and built only on demand.

    A bulk build used to leave _synapse_ids empty -- the bug behind reset() skipping
    synapses. Filling it eagerly would recreate the per-agent structure the columnar
    path exists to avoid, so it is derived from the combo codes when first asked.
    """

    def test_columnar_build_has_both_sets(self):
        m = _columnar_model()
        ids = set(m._agent_factory._rank2agentid2agentidx[0])
        self.assertTrue(m._synapse_ids, "a bulk build reported no synapses")
        self.assertTrue(m._soma_ids)
        self.assertEqual(m._soma_ids | m._synapse_ids, ids)
        self.assertFalse(m._soma_ids & m._synapse_ids, "an agent is in both sets")

    def test_record_build_has_both_sets(self):
        m = _record_model()
        ids = set(m._agent_factory._rank2agentid2agentidx[0])
        self.assertEqual(m._soma_ids | m._synapse_ids, ids)

    def test_not_materialised_until_asked(self):
        m = _columnar_model()
        self.assertIsNone(m._synapse_ids_set, "the set was built eagerly")
        _ = m._synapse_ids
        self.assertIsNotNone(m._synapse_ids_set, "the set was not cached")

    def test_incremental_build_still_tracks_ids(self):
        m = NeuromorphicModel(enable_internal_states_tracking=False)
        a = m.create_soma(breed="lif_soma", config_name="config_0")
        syn = m.create_synapse(breed="single_exp_synapse", pre_soma_id=-1,
                               post_soma_id=a, config_name="config_0")
        self.assertEqual(m._soma_ids, {a})
        self.assertEqual(m._synapse_ids, {syn})


class TestNeighbourAccessors(unittest.TestCase):
    """get_neighbors works on every construction path.

    Every bulk build now hands SAGESim a prebuilt CSR, and set_prebuilt_csr clears
    space._locations, so reading the `locations` property returns nothing. These pin the
    replacement -- including that it keeps working for an incrementally built model,
    which still has real per-agent lists.
    """

    def _check(self, m):
        ids = list(m._agent_factory._rank2agentid2agentidx[0])
        syn = next(i for i in ids if m._component_class_of(i) == "synapse")
        nbrs = m.get_neighbors(syn)
        self.assertIn(len(nbrs), (1, 2),
                      f"synapse neighbours must be [pre] or [pre, post], got {nbrs}")
        self.assertEqual(m.get_synapse_connectivity(syn), nbrs)

        soma = next(i for i in ids if m._component_class_of(i) == "soma")
        for s in m.get_neighbors(soma):
            self.assertEqual(m._component_class_of(s), "synapse",
                             "a soma's neighbours are its incoming synapses")

    def test_record_build(self):
        self._check(_record_model())

    def test_columnar_build(self):
        self._check(_columnar_model())

    def test_incremental_build(self):
        m = NeuromorphicModel(enable_internal_states_tracking=False)
        a = m.create_soma(breed="lif_soma", config_name="config_0")
        b = m.create_soma(breed="lif_soma", config_name="config_0")
        m.create_synapse(breed="single_exp_synapse", pre_soma_id=a, post_soma_id=b,
                         config_name="config_0")
        self._check(m)

    def test_unknown_agent_raises(self):
        m = _columnar_model()
        with self.assertRaises(KeyError):
            m.get_neighbors(-99999)


class TestMappingViews(unittest.TestCase):
    """The three public dicts still read like dicts, on both construction paths."""

    def _check(self, m):
        aid = next(iter(m._agent_factory._rank2agentid2agentidx[0]))
        self.assertIsNotNone(m.agentid2config[aid])
        self.assertIsNotNone(m.agentid2config.get(aid))
        self.assertIsInstance(m.agentid2overrides.get(aid), dict)
        # .get on a rule-less synapse is None, not a KeyError
        self.assertIsNone(m.agentid2learning_rule.get(-99999))
        self.assertIn(aid, m.agentid2config)

    def test_views_on_record_build(self):
        self._check(_record_model())

    def test_views_on_columnar_build(self):
        self._check(_columnar_model())

    def test_view_raises_keyerror_for_unknown_agent(self):
        m = _columnar_model()
        with self.assertRaises(KeyError):
            m.agentid2config[-99999]


if __name__ == "__main__":
    unittest.main()
