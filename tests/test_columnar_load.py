#!/usr/bin/env python
"""Tests for the columnar (.npz) post-owned partition path.

The columnar producer/loader encode the SAME post-owned Brunel network as the
record (.pkl) path, but as typed arrays built straight into SAGESim's property
tensors + neighbor CSR — no per-synapse Python object. These tests pin the
contract that "columnar encoding" must never mean "different network":

  * test_columnar_matches_record_build (CPU, no GPU): the built agent-factory
    property tensors and the neighbor CSR are IDENTICAL to the record path's,
    across weight/delay/partition variants — the plan's headline identity check.
  * test_columnar_load_from_npz (CPU, no GPU): the full disk round-trip,
    save_brunel_partition(output_format="columns") -> load_post_owned(".npz"),
    reproduces the same build (dispatch + mmap read + columnar build).
  * test_columnar_vs_record_firing (GPU): load -> setup -> simulate reproduces
    the record path's spikes; run on a compute node with a GPU.

Usage:
    python -m unittest superneuroabm.tests.test_columnar_load       # CPU checks
    srun -N1 -n1 --gpu-bind=closest python -m unittest \
        superneuroabm.tests.test_columnar_load.TestColumnarLoad.test_columnar_vs_record_firing
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR.parent))

from superneuroabm.model import NeuromorphicModel
from superneuroabm.brunel import brunel_partition, save_brunel_partition
from sagesim.internal_utils import build_csr_from_ragged


BASE = dict(somas_per_rank=16, excitatory_in_degree=6, inhibitory_in_degree=2,
            external_synapses_per_soma=2, inhibitory_weight_ratio=5.0, seed=3)

VARIANTS = {
    "single_full": dict(num_partitions=1, partition_rank=0, excitatory_weight=14.0,
                        external_weight=14.0, synaptic_delay_ms=1.5),
    "no_delay": dict(num_partitions=1, partition_rank=0, excitatory_weight=14.0,
                     external_weight=14.0, synaptic_delay_ms=None),
    "inherit_weight": dict(num_partitions=1, partition_rank=0, excitatory_weight=None,
                           external_weight=None, synaptic_delay_ms=1.0),
    "multi_global_r0": dict(num_partitions=3, partition_rank=0, topology="global",
                            excitatory_weight=14.0, external_weight=14.0, synaptic_delay_ms=1.5),
    "multi_global_r2": dict(num_partitions=3, partition_rank=2, topology="global",
                            excitatory_weight=14.0, external_weight=14.0, synaptic_delay_ms=1.5),
}


def _record_model(params):
    rec = brunel_partition(output_format="records", **params)
    m = NeuromorphicModel(enable_internal_states_tracking=False)
    m.load_post_owned(rec)
    return m


def _csr_of(model):
    """(offsets, values) for a model, from whichever representation it holds.

    Both builders now hand SAGESim a prebuilt CSR, so `locations` is empty and the
    neighbour lists live in the space. The ragged fallback covers a model built by
    create_soma/create_synapse, which still keeps per-agent lists.
    """
    space = model.get_space()
    off = getattr(space, "_prebuilt_csr_offsets", None)
    if off is not None:
        return np.asarray(off), np.asarray(space._prebuilt_csr_values)
    locs = model._agent_factory._property_name_2_agent_data_tensor["locations"]
    return build_csr_from_ragged(locs)


def _assert_same_build(test, mr, mc):
    afr = mr._agent_factory._property_name_2_agent_data_tensor
    afc = mc._agent_factory._property_name_2_agent_data_tensor
    ids_r = list(mr._agent_factory._rank2agentid2agentidx[0].keys())
    ids_c = list(mc._agent_factory._rank2agentid2agentidx[0].keys())
    test.assertEqual(ids_r, ids_c, "agent id order differs")

    test.assertEqual(set(afr), set(afc))
    for prop in afr:
        if prop == "locations":
            continue
        a, b = afr[prop], afc[prop]
        test.assertEqual(len(a), len(b), f"{prop}: length differs")
        for i, (x, y) in enumerate(zip(a, b)):
            xl = list(x) if isinstance(x, (list, tuple, np.ndarray)) else x
            yl = list(y) if isinstance(y, (list, tuple, np.ndarray)) else y
            # Compared EXACTLY. Both builders store float32 now, so the downcast this
            # assertion used to need on the record side is gone -- a parity test that
            # needs a fudge factor is not quite a parity test.
            test.assertEqual(xl, yl, f"{prop}[{i}] (agent {ids_r[i]}) differs")

    rec_off, rec_val = _csr_of(mr)
    col_off, col_val = _csr_of(mc)
    test.assertTrue(np.array_equal(rec_off, col_off), "CSR offsets differ")
    test.assertTrue(np.array_equal(rec_val, col_val), "CSR values differ")

    # Remote (ghost) ranks match.
    def _remote(m):
        local = m._agent_factory._rank2agentid2agentidx[0]
        return {k: v for k, v in m._agent_factory._agent2rank.items() if k not in local}
    test.assertEqual(_remote(mr), _remote(mc), "remote ranks differ")


class TestColumnarLoad(unittest.TestCase):

    def test_columnar_matches_record_build(self):
        """Columnar build == record build (af tensors + CSR) across variants."""
        for name, params in VARIANTS.items():
            with self.subTest(variant=name):
                p = {**BASE, **params}
                mr = _record_model(p)
                mc = NeuromorphicModel(enable_internal_states_tracking=False)
                mc.load_post_owned(brunel_partition(output_format="columns", **p))
                _assert_same_build(self, mr, mc)

                # Input synapses (external, pre == -1) exposed vectorized.
                ext_expected = sorted(
                    s["id"] for s in brunel_partition(output_format="records", **p)["synapses"]
                    if s["pre"] == -1)
                self.assertEqual(sorted(int(x) for x in mc._input_synapse_ids), ext_expected)

    def test_columnar_load_from_npz(self):
        """Full disk round-trip: save columns -> load_post_owned('.npz') == record."""
        p = {**BASE, **VARIANTS["multi_global_r0"]}
        with tempfile.TemporaryDirectory() as d:
            path = save_brunel_partition(output_dir=d, output_format="columns", **p)
            self.assertTrue(path.endswith(".npz"))
            mc = NeuromorphicModel(enable_internal_states_tracking=False)
            mc.load_post_owned(path)  # dispatch on .npz + mmap read + columnar build
        mr = _record_model(p)
        _assert_same_build(self, mr, mc)

    def test_columnar_vs_record_firing(self):
        """load('.npz') -> setup -> simulate reproduces the record path's spikes (GPU)."""
        p = {**BASE, **VARIANTS["single_full"]}
        INPUT_TICK, TICKS = 2, 40

        # External input synapses fire on both models identically.
        rec = brunel_partition(output_format="records", **p)
        ext_ids = [s["id"] for s in rec["synapses"] if s["pre"] == -1]
        soma_ids = [s["id"] for s in rec["somas"]]

        m_rec = NeuromorphicModel(enable_internal_states_tracking=False)
        m_rec.create_from_lists(rec["somas"], rec["synapses"])
        m_rec.setup()
        for sid in ext_ids:
            m_rec.add_spike(synapse_id=sid, tick=INPUT_TICK, value=1)
        m_rec.simulate(ticks=TICKS, update_data_ticks=1)
        rec_spikes = {s: sorted(m_rec.get_spike_times(s)) for s in soma_ids}

        with tempfile.TemporaryDirectory() as d:
            path = save_brunel_partition(output_dir=d, output_format="columns", **p)
            m_col = NeuromorphicModel(enable_internal_states_tracking=False)
            m_col.load_post_owned(path)
            m_col.setup()
            for sid in m_col._input_synapse_ids.tolist():
                m_col.add_spike(synapse_id=int(sid), tick=INPUT_TICK, value=1)
            m_col.simulate(ticks=TICKS, update_data_ticks=1)
            col_spikes = {s: sorted(m_col.get_spike_times(s)) for s in soma_ids}

        self.assertEqual(rec_spikes, col_spikes,
                         "columnar-loaded network fired differently from the record path")


class TestAutapseKeepsBothSlots(unittest.TestCase):
    """A synapse onto its own pre-soma must keep BOTH positional slots.

    A synapse's neighbour row is positional: slot 0 = pre (read every tick for the
    incoming spike), slot 1 = post (read by STDP). The record path used to route its
    adjacency through space.bulk_connect, which dedups in ordered mode, so an autapse
    (pre == post) came out as [pre] -- slot 1 silently deleted -- while the columnar
    path emitted [pre, post]. The two builders produced different networks for the same
    input, and nothing caught it because brunel_partition resamples autapses away.
    """

    def test_create_from_lists_keeps_pre_and_post(self):
        A, S = 0, 1
        m = NeuromorphicModel(enable_internal_states_tracking=False)
        m.create_from_lists(somas=[{"id": A}], synapses=[{"id": S, "pre": A, "post": A}])

        off, val = _csr_of(m)
        row = m._agent_factory._rank2agentid2agentidx[0][S]
        self.assertEqual(list(val[off[row]:off[row + 1]]), [A, A],
                         "autapse lost its post slot; slot 1 is read by STDP")

    def test_record_and_columnar_agree_on_an_autapse(self):
        A, S = 0, 1
        somas = [{"id": A}]
        synapses = [{"id": S, "pre": A, "post": A}]

        m_rec = NeuromorphicModel(enable_internal_states_tracking=False)
        m_rec.create_from_lists(somas=somas, synapses=synapses)

        arrays = {
            "schema": np.asarray("columnar_post_owned_v1"),
            "soma_ids": np.array([A], dtype=np.int64),
            "soma_breed": np.asarray("lif_soma"),
            "soma_config": np.asarray("config_0"),
            "synapse_ids": np.array([S], dtype=np.int64),
            "pre": np.array([A], dtype=np.int64),
            "post": np.array([A], dtype=np.int64),
            "synapse_breed": np.asarray("single_exp_synapse"),
            "synapse_config": np.asarray("config_0"),
            "learning_rule": np.asarray(""),
            "learning_rule_config": np.asarray("default"),
            "syn_hp_keys": np.asarray([], dtype=object),
            "syn_hp_vals": np.zeros((1, 0), dtype=np.float64),
            "remote_ids": np.empty(0, np.int64),
            "remote_rank_of": np.empty(0, np.int64),
        }
        m_col = NeuromorphicModel(enable_internal_states_tracking=False)
        m_col.load_post_owned(arrays)

        rec_off, rec_val = _csr_of(m_rec)
        col_off, col_val = _csr_of(m_col)
        self.assertTrue(np.array_equal(rec_off, col_off), "CSR offsets differ")
        self.assertTrue(np.array_equal(rec_val, col_val), "CSR values differ")


if __name__ == "__main__":
    unittest.main()
