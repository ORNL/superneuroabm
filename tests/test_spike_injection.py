#!/usr/bin/env python
"""Unit tests for the spike-injection API family on ``NeuromorphicModel``.

Covers the ways external input spikes are scheduled on a synapse whose
``pre_soma_id`` is -1 (all defined in ``superneuroabm/model.py``):

- ``add_spike(synapse_id, tick, value)``
- ``add_spike_list(synapse_id, [[tick, value], ...])``   (also accepts an (N, 2) array)
- ``add_spikes(synapse_ids, ticks, values)``              (bulk, flat arrays, any order)
- ``add_local_spike`` / ``add_local_spike_list``          (raise KeyError for a non-owned id)

Storage contract: spikes accumulate in a host-side event store (additive; never
clears until ``reset()``/``clear_input_spikes()``), readable with
``get_input_spikes(synapse_id) -> (ticks, values)`` sorted by tick. They are
compiled into a tick-major event list when the kernel launches and delivered
into the synapse's ``input_spikes_tensor`` row as ``[tick, value]`` on the tick
they fall on. Delivery timing itself is covered by ``test_input_spike_events.py``.

Usage:
    python -m pytest tests/test_spike_injection.py -v
    mpirun -n 2 python -m unittest \
        tests.test_spike_injection.TestAddLocalSpike.test_local_owner_injection_multirank
    (the single-rank tests skip themselves under mpirun: they build agents on every
    rank and would desynchronise the collective simulate)
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


SIM_TICKS = 200
BIT_EXACT_PROPS = ["internal_states", "output_spikes_tensor", "hyperparameters"]


def _get_mpi():
    """Return (comm, rank, size); (None, 0, 1) when mpi4py is unavailable."""
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        return comm, comm.Get_rank(), comm.Get_size()
    except ImportError:
        return None, 0, 1


def _single_rank_only(test):
    _comm, _rank, size = _get_mpi()
    if size != 1:
        test.skipTest("single-rank fixture; run this one without mpirun")


def _build_chain():
    """Build ``external -> soma_0 -> soma_1`` and return (model, soma_0, soma_1, syn_ext)."""
    model = NeuromorphicModel()
    model.set_seed(42)
    soma_0 = model.create_soma(breed="lif_soma", config_name="config_0")
    soma_1 = model.create_soma(breed="lif_soma", config_name="config_0")
    syn_ext = model.create_synapse(
        breed="single_exp_synapse", pre_soma_id=-1,
        post_soma_id=soma_0, config_name="config_0")
    model.create_synapse(
        breed="single_exp_synapse", pre_soma_id=soma_0,
        post_soma_id=soma_1, config_name="config_0")
    model.setup()
    return model, soma_0, soma_1, syn_ext


def _injected(model, syn):
    """(ticks, values) as plain lists, for readable assertions."""
    t, v = model.get_input_spikes(syn)
    return t.tolist(), v.tolist()


def _assert_bit_exact(test, model_a, somas_a, model_b, somas_b):
    """Assert two models' somas are bit-identical in state and spike times."""
    for sa, sb in zip(somas_a, somas_b):
        for prop in BIT_EXACT_PROPS:
            va = np.array(model_a.get_agent_property_value(sa, prop), dtype=np.float32)
            vb = np.array(model_b.get_agent_property_value(sb, prop), dtype=np.float32)
            np.testing.assert_array_equal(
                va.view(np.uint32), vb.view(np.uint32),
                err_msg=f"{prop}: bit-differs between the two injection paths")
        test.assertEqual(
            model_a.get_spike_times(soma_id=sa),
            model_b.get_spike_times(soma_id=sb),
            "spike times differ between the two injection paths")


class TestAddSpike(unittest.TestCase):
    """Direct coverage of add_spike's storage contract."""

    def test_round_trip(self):
        """Two add_spike calls are both recorded, sorted by tick, values kept."""
        _single_rank_only(self)
        model, _soma_0, _soma_1, syn_ext = _build_chain()
        self.assertEqual(_injected(model, syn_ext), ([], []))

        model.add_spike(synapse_id=syn_ext, tick=50, value=1.0)
        model.add_spike(synapse_id=syn_ext, tick=2, value=0.5)

        self.assertEqual(_injected(model, syn_ext), ([2, 50], [0.5, 1.0]))

    def test_unknown_id_raises(self):
        """On one rank every agent is local, so an unknown id is an error at call time."""
        _single_rank_only(self)
        model, _soma_0, _soma_1, _syn_ext = _build_chain()
        with self.assertRaises(KeyError):
            model.add_spike(synapse_id=10_000_000, tick=2, value=1.0)

    def test_tick_validation(self):
        _single_rank_only(self)
        model, _soma_0, _soma_1, syn_ext = _build_chain()
        with self.assertRaises(ValueError):
            model.add_spike(synapse_id=syn_ext, tick=2.5, value=1.0)
        with self.assertRaises(ValueError):
            model.add_spike(synapse_id=syn_ext, tick=-1, value=1.0)
        with self.assertRaises(ValueError):
            model.add_spike(synapse_id=syn_ext, tick=1 << 24, value=1.0)


class TestAddSpikeList(unittest.TestCase):
    """Comprehensive coverage of the bulk add_spike_list / add_spikes API."""

    def test_pairs_recorded(self):
        _single_rank_only(self)
        model, _soma_0, _soma_1, syn_ext = _build_chain()
        model.add_spike_list(syn_ext, [[2, 1.0], [50, 1.0], [80, 1.0]])
        self.assertEqual(_injected(model, syn_ext), ([2, 50, 80], [1.0, 1.0, 1.0]))

    def test_array_input(self):
        """An (N, 2) numpy array is accepted as well as a list of pairs."""
        _single_rank_only(self)
        model, _soma_0, _soma_1, syn_ext = _build_chain()
        model.add_spike_list(syn_ext, np.array([[2, 1.0], [50, 1.0]]))
        self.assertEqual(_injected(model, syn_ext), ([2, 50], [1.0, 1.0]))

    def test_additive_accumulation(self):
        """Successive calls concatenate rather than overwrite."""
        _single_rank_only(self)
        model, _s0, _s1, syn_ext = _build_chain()
        model.add_spike_list(syn_ext, [[2, 1.0], [50, 1.0]])
        model.add_spike_list(syn_ext, [[80, 1.0]])
        self.assertEqual(_injected(model, syn_ext), ([2, 50, 80], [1.0, 1.0, 1.0]))

        model2, _a, _b, syn2 = _build_chain()
        model2.add_spike(synapse_id=syn2, tick=2, value=1.0)
        model2.add_spike_list(syn2, [[50, 1.0], [80, 1.0]])
        self.assertEqual(_injected(model2, syn2), ([2, 50, 80], [1.0, 1.0, 1.0]))

    def test_equivalent_to_add_spike(self):
        """add_spike_list must be a faithful bulk form of repeated add_spike calls."""
        _single_rank_only(self)
        model_a, a0, a1, syn_a = _build_chain()
        model_a.add_spike_list(syn_a, [[2, 1.0], [50, 1.0]])

        model_b, b0, b1, syn_b = _build_chain()
        model_b.add_spike(synapse_id=syn_b, tick=2, value=1.0)
        model_b.add_spike(synapse_id=syn_b, tick=50, value=1.0)

        model_a.simulate(ticks=SIM_TICKS, update_data_ticks=1)
        model_b.simulate(ticks=SIM_TICKS, update_data_ticks=1)

        _assert_bit_exact(self, model_a, [a0, a1], model_b, [b0, b1])

    def test_add_spikes_equivalent(self):
        """Bulk flat-array injection (unsorted) matches per-synapse pairs bit-for-bit."""
        _single_rank_only(self)
        model_a, a0, a1, syn_a = _build_chain()
        model_a.add_spikes([syn_a, syn_a, syn_a], [50, 2, 80])   # any order, value 1.0

        model_b, b0, b1, syn_b = _build_chain()
        model_b.add_spike_list(syn_b, [[2, 1.0], [50, 1.0], [80, 1.0]])

        self.assertEqual(_injected(model_a, syn_a), _injected(model_b, syn_b))
        model_a.simulate(ticks=SIM_TICKS, update_data_ticks=1)
        model_b.simulate(ticks=SIM_TICKS, update_data_ticks=1)
        _assert_bit_exact(self, model_a, [a0, a1], model_b, [b0, b1])

    def test_add_spikes_length_mismatch(self):
        _single_rank_only(self)
        model, _s0, _s1, syn_ext = _build_chain()
        with self.assertRaises(ValueError):
            model.add_spikes([syn_ext, syn_ext], [1])

    def test_bulk_train_drives_firing(self):
        """A bulk-injected spike train drives the downstream soma to fire."""
        _single_rank_only(self)
        model, soma_0, soma_1, syn_ext = _build_chain()
        model.add_spike_list(syn_ext, [[t, 1.0] for t in (2, 5, 8, 11)])
        model.simulate(ticks=SIM_TICKS, update_data_ticks=1)
        self.assertGreaterEqual(len(model.get_spike_times(soma_id=soma_0)), 1,
                                "directly-stimulated soma_0 should fire")


class TestAddLocalSpike(unittest.TestCase):
    """add_local_spike: rank-local injection with an eager ownership check.

    Injection is rank-local for every variant; the add_local_* names differ only
    in raising KeyError for a synapse this rank does not own. The multi-rank case
    auto-skips unless launched under mpirun -n 2.
    """

    def test_local_round_trip(self):
        _single_rank_only(self)
        model, _soma_0, _soma_1, syn_ext = _build_chain()
        model.add_local_spike(synapse_id=syn_ext, tick=2, value=1.0)
        model.add_local_spike_list(syn_ext, [[5, 1.0]])
        self.assertEqual(_injected(model, syn_ext), ([2, 5], [1.0, 1.0]))

    def test_local_equivalent_to_add_spike(self):
        """On one rank, add_local_spike matches add_spike bit-for-bit."""
        _single_rank_only(self)
        model_a, a0, a1, syn_a = _build_chain()
        model_a.add_local_spike(synapse_id=syn_a, tick=2, value=1.0)
        model_a.add_local_spike(synapse_id=syn_a, tick=50, value=1.0)

        model_b, b0, b1, syn_b = _build_chain()
        model_b.add_spike(synapse_id=syn_b, tick=2, value=1.0)
        model_b.add_spike(synapse_id=syn_b, tick=50, value=1.0)

        model_a.simulate(ticks=SIM_TICKS, update_data_ticks=1)
        model_b.simulate(ticks=SIM_TICKS, update_data_ticks=1)

        _assert_bit_exact(self, model_a, [a0, a1], model_b, [b0, b1])

    def test_local_keyerror_on_unknown_id(self):
        _single_rank_only(self)
        model, _soma_0, _soma_1, _syn_ext = _build_chain()
        with self.assertRaises(KeyError):
            model.add_local_spike(synapse_id=10_000_000, tick=2, value=1.0)

    def test_local_owner_injection_multirank(self):
        """Owner injects locally; non-owner raises KeyError; each soma still fires."""
        comm, rank, size = _get_mpi()
        if size == 1:
            self.skipTest("multi-rank add_local_spike contract needs mpirun -n 2")
        if size != 2:
            self.skipTest(f"fixture supports exactly 2 ranks, got {size}")

        with tempfile.TemporaryDirectory() as tmpdir:
            for r in (0, 1):
                soma_id, syn_id = r, 100 + r
                f = Path(tmpdir) / f"local_spike_rank{r}.pkl"
                with open(f, "wb") as fh:
                    pickle.dump({
                        "somas": [{"id": soma_id, "neighbors": [syn_id]}],
                        "synapses": [{"id": syn_id, "neighbors": [-1]}],
                        "remote_ranks": {},
                    }, fh)

            model = NeuromorphicModel(enable_internal_states_tracking=False)
            model.load_from_adjacency(str(Path(tmpdir) / f"local_spike_rank{rank}.pkl"))
            model.setup()

            my_syn = 100 + rank
            other_syn = 100 + (1 - rank)

            with self.assertRaises(KeyError):
                model.add_local_spike(synapse_id=other_syn, tick=2, value=1.0)

            for t in (2, 5, 8, 11):
                model.add_local_spike(synapse_id=my_syn, tick=t, value=1.0)

            model.simulate(ticks=SIM_TICKS, update_data_ticks=1)

            fired = {sid: len(model.get_spike_times(soma_id=sid)) for sid in (0, 1)}

        self.assertGreaterEqual(fired[0], 1, "soma on rank 0 should fire from its local injection")
        self.assertGreaterEqual(fired[1], 1, "soma on rank 1 should fire from its local injection")


if __name__ == "__main__":
    unittest.main()
