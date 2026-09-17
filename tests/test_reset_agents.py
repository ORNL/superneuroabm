#!/usr/bin/env python
"""_reset_agents must put every agent back to exactly its freshly-built state.

It is grouped by combo rather than by agent: agents sharing a description reset to
byte-identical rows, so the rows are computed once and scattered. That is a big change
to a method every reset() depends on, and the three column representations take three
different scatter paths, so each is pinned here against the definition of the operation
-- "the model as the builder produced it" -- rather than against the old implementation.

Usage:
    python -m pytest tests/test_reset_agents.py
"""

import sys
import unittest
from pathlib import Path

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR.parent))

from superneuroabm.model import NeuromorphicModel
from superneuroabm.brunel import brunel_partition
from sagesim.columns import ArrayColumn, IndexedColumn


BASE = dict(somas_per_rank=10, excitatory_in_degree=4, inhibitory_in_degree=2,
            external_synapses_per_soma=1, inhibitory_weight_ratio=5.0, seed=11,
            num_partitions=1, partition_rank=0, excitatory_weight=14.0,
            external_weight=14.0, synaptic_delay_ms=1.5)

RESET_PROPS = ("hyperparameters", "learning_hyperparameters",
               "internal_states", "learning_internal_states",
               "input_spikes_tensor", "output_spikes_tensor")


def _record_model():
    rec = brunel_partition(output_format="records", **BASE)
    m = NeuromorphicModel(enable_internal_states_tracking=False)
    m.load_post_owned(rec)
    return m


def _created_model():
    """Built one agent at a time, so its columns are plain Python lists."""
    m = NeuromorphicModel(enable_internal_states_tracking=False)
    a = m.create_soma(breed="lif_soma", config_name="config_0")
    b = m.create_soma(breed="lif_soma", config_name="config_0")
    m.create_synapse(breed="single_exp_synapse", pre_soma_id=-1, post_soma_id=a,
                     config_name="config_0")
    m.create_synapse(breed="single_exp_synapse", pre_soma_id=a, post_soma_id=b,
                     config_name="config_0", learning_rule="exp_pair_wise_stdp")
    return m


def _snapshot(model):
    data = model._agent_factory._property_name_2_agent_data_tensor
    n = len(model._agent_factory._rank2agentid2agentidx[0])
    return {p: [list(data[p][i]) for i in range(n)] for p in RESET_PROPS if p in data}


def _owned_props(model, agent_id):
    """Properties reset OWNS for this agent -- the ones its component class writes.

    A soma has no learning_hyperparameters to restore and a synapse has no
    output_spikes_tensor; neither the old per-agent reset nor the combo-grouped one
    touches them, because no kernel of that class writes them.
    """
    combo = model._combo_of(agent_id)
    return set(model._combo_property_rows(combo)) if combo is not None else set()


def _perturb(model):
    """Move every agent away from its built value, on the properties reset owns."""
    af = model._agent_factory
    data = af._property_name_2_agent_data_tensor
    for agent_id, i in af._rank2agentid2agentidx[0].items():
        for prop in _owned_props(model, agent_id):
            if prop not in data:
                continue
            row = list(data[prop][i])
            if row:
                data[prop][i] = [float(x) + 7.5 for x in row]


class TestResetRestoresBuiltState(unittest.TestCase):

    def _check_full_reset(self, model):
        built = _snapshot(model)
        _perturb(model)
        self.assertNotEqual(_snapshot(model), built, "test premise: perturb did nothing")
        model._reset_agents(retain_parameters=False)
        self.assertEqual(_snapshot(model), built,
                         "reset did not restore the freshly-built state")

    def test_record_built_indexed_columns(self):
        self._check_full_reset(_record_model())

    def test_incrementally_created_list_columns(self):
        self._check_full_reset(_created_model())

    def test_array_columns(self):
        """The representation a model has after a GPU->host sync."""
        m = _record_model()
        data = m._agent_factory._property_name_2_agent_data_tensor
        n = len(m._agent_factory._rank2agentid2agentidx[0])
        for prop in RESET_PROPS:
            col = data[prop]
            if not isinstance(col, IndexedColumn):
                continue
            width = max((len(col[i]) for i in range(n)), default=0)
            vals = np.full((n, width), np.nan, dtype=np.float32)
            lens = np.zeros(n, dtype=np.int32)
            for i in range(n):
                row = col[i]
                vals[i, :len(row)] = row
                lens[i] = len(row)
            data[prop] = ArrayColumn(vals, lens)
        self.assertTrue(any(isinstance(data[p], ArrayColumn) for p in RESET_PROPS),
                        "test premise: no column became an ArrayColumn")
        self._check_full_reset(m)


class TestRetainParameters(unittest.TestCase):

    def test_retain_keeps_parameters_and_clears_state(self):
        m = _record_model()
        data = m._agent_factory._property_name_2_agent_data_tensor
        built_state = [list(r) for r in _snapshot(m)["internal_states"]]

        _perturb(m)
        learned_hp = [list(r) for r in _snapshot(m)["hyperparameters"]]

        m._reset_agents(retain_parameters=True)
        after = _snapshot(m)
        self.assertEqual(after["hyperparameters"], learned_hp,
                         "retain_parameters=True discarded the learned parameters")
        self.assertEqual(after["internal_states"], built_state,
                         "retain_parameters=True failed to clear state")

    def test_no_retain_clears_parameters_too(self):
        m = _record_model()
        built_hp = [list(r) for r in _snapshot(m)["hyperparameters"]]
        _perturb(m)
        m._reset_agents(retain_parameters=False)
        self.assertEqual(_snapshot(m)["hyperparameters"], built_hp)


class TestScatterDoesNotLeak(unittest.TestCase):
    """Agents sharing a combo share a table row; a reset must not bleed between combos."""

    def test_distinct_combos_reset_to_their_own_values(self):
        m = _created_model()
        ids = list(m._agent_factory._rank2agentid2agentidx[0])
        somas = [i for i in ids if m._component_class_of(i) == "soma"]
        syns = [i for i in ids if m._component_class_of(i) == "synapse"]
        self.assertTrue(somas and syns)

        built = {i: list(m.get_agent_property_value(id=i, property_name="internal_states"))
                 for i in ids}
        _perturb(m)
        m._reset_agents(retain_parameters=False)

        for i in ids:
            self.assertEqual(
                list(m.get_agent_property_value(id=i, property_name="internal_states")),
                built[i], f"agent {i} did not reset to its own built values")

        # A soma and a synapse must not have collapsed onto one shared row.
        self.assertNotEqual(built[somas[0]], built[syns[0]],
                            "test premise: soma and synapse states are indistinguishable")

    def test_unowned_properties_are_left_alone(self):
        """Reset restores what a class writes, and deliberately nothing else."""
        m = _record_model()
        af = m._agent_factory
        data = af._property_name_2_agent_data_tensor
        soma = next(i for i in af._rank2agentid2agentidx[0]
                    if m._component_class_of(i) == "soma")
        idx = af._rank2agentid2agentidx[0][soma]
        self.assertNotIn("learning_hyperparameters", _owned_props(m, soma))

        marker = [9.0] * len(data["learning_hyperparameters"][idx])
        if marker:
            data["learning_hyperparameters"][idx] = marker
            m._reset_agents(retain_parameters=False)
            self.assertEqual(list(data["learning_hyperparameters"][idx]), marker,
                             "reset touched a property the soma never writes")


if __name__ == "__main__":
    unittest.main()
