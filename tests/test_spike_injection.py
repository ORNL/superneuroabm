#!/usr/bin/env python
"""Unit tests for the spike-injection API family on ``NeuromorphicModel``.

Covers the three ways external input spikes are scheduled onto a synapse's
``input_spikes_tensor`` (all defined in ``superneuroabm/model.py``):

- ``add_spike(synapse_id, tick, value)``          (model.py:1411, collective)
- ``add_spike_list(synapse_id, [[tick, value]])`` (model.py:1447, collective bulk)
- ``add_local_spike(synapse_id, tick, value)``    (model.py:1429, MPI-local)

These were previously untested as subjects-in-themselves: ``add_spike`` appeared
only as plumbing in other tests, and the other two had no coverage at all.

The storage contract for all three is identical: append ``[tick, value]`` pairs,
flattened, into the synapse's ``input_spikes_tensor`` (additive; never clears).

Usage:
    # single-process (GPU), runs everything except the multi-rank local case:
    python -m pytest tests/test_spike_injection.py -v
    # the distributed add_local_spike contract (needs GPU + 2 ranks):
    mpirun -n 2 python -m unittest tests.test_spike_injection.TestAddLocalSpike
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


def _build_chain():
    """Build ``external -> soma_0 -> soma_1`` and return (model, soma_0, soma_1, syn_ext).

    Standard build order used across the suite: create -> setup. Spikes are injected
    by the caller AFTER this returns, since ``setup`` creates the property tensors.
    """
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


def _as_floats(tensor):
    """Read-back helper: coerce a stored spike tensor to a plain list of floats.

    The tensor may come back as ints, a numpy array, or floats depending on the
    storage path; compare on value, not representation."""
    return [float(x) for x in tensor]


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
    """Direct coverage of add_spike's storage contract (elsewhere only plumbing)."""

    def test_flattened_tensor(self):
        """Two add_spike calls append, flattened and in order, additively.

        The tensor ships with a [-1, 0] never-fires sentinel; assert against the
        captured baseline rather than the sentinel value so the test survives an
        init change."""
        model, _soma_0, _soma_1, syn_ext = _build_chain()
        base = _as_floats(model.get_agent_property_value(syn_ext, "input_spikes_tensor"))

        model.add_spike(synapse_id=syn_ext, tick=2, value=1.0)
        model.add_spike(synapse_id=syn_ext, tick=50, value=1.0)

        tensor = model.get_agent_property_value(syn_ext, "input_spikes_tensor")
        self.assertEqual(_as_floats(tensor), base + [2.0, 1.0, 50.0, 1.0])


class TestAddSpikeList(unittest.TestCase):
    """Comprehensive coverage of the bulk add_spike_list API."""

    def test_flattened_tensor(self):
        """A [[tick, value], ...] list is flattened to [tick, value, ...] in order."""
        model, _soma_0, _soma_1, syn_ext = _build_chain()
        base = _as_floats(model.get_agent_property_value(syn_ext, "input_spikes_tensor"))

        model.add_spike_list(syn_ext, [[2, 1.0], [50, 1.0], [80, 1.0]])

        tensor = model.get_agent_property_value(syn_ext, "input_spikes_tensor")
        self.assertEqual(_as_floats(tensor), base + [2.0, 1.0, 50.0, 1.0, 80.0, 1.0])

    def test_additive_accumulation(self):
        """Successive calls concatenate rather than overwrite."""
        # Two add_spike_list calls accumulate.
        model, _s0, _s1, syn_ext = _build_chain()
        base = _as_floats(model.get_agent_property_value(syn_ext, "input_spikes_tensor"))
        model.add_spike_list(syn_ext, [[2, 1.0], [50, 1.0]])
        model.add_spike_list(syn_ext, [[80, 1.0]])
        self.assertEqual(
            _as_floats(model.get_agent_property_value(syn_ext, "input_spikes_tensor")),
            base + [2.0, 1.0, 50.0, 1.0, 80.0, 1.0])

        # add_spike followed by add_spike_list also accumulates onto the same tensor.
        model2, _a, _b, syn2 = _build_chain()
        base2 = _as_floats(model2.get_agent_property_value(syn2, "input_spikes_tensor"))
        model2.add_spike(synapse_id=syn2, tick=2, value=1.0)
        model2.add_spike_list(syn2, [[50, 1.0], [80, 1.0]])
        self.assertEqual(
            _as_floats(model2.get_agent_property_value(syn2, "input_spikes_tensor")),
            base2 + [2.0, 1.0, 50.0, 1.0, 80.0, 1.0])

    def test_equivalent_to_add_spike(self):
        """add_spike_list must be a faithful bulk form of repeated add_spike calls."""
        model_a, a0, a1, syn_a = _build_chain()
        model_a.add_spike_list(syn_a, [[2, 1.0], [50, 1.0]])

        model_b, b0, b1, syn_b = _build_chain()
        model_b.add_spike(synapse_id=syn_b, tick=2, value=1.0)
        model_b.add_spike(synapse_id=syn_b, tick=50, value=1.0)

        model_a.simulate(ticks=SIM_TICKS, update_data_ticks=1)
        model_b.simulate(ticks=SIM_TICKS, update_data_ticks=1)

        _assert_bit_exact(self, model_a, [a0, a1], model_b, [b0, b1])

    def test_bulk_train_drives_firing(self):
        """A bulk-injected spike train drives the downstream soma to fire."""
        model, soma_0, soma_1, syn_ext = _build_chain()
        model.add_spike_list(syn_ext, [[t, 1.0] for t in (2, 5, 8, 11)])
        model.simulate(ticks=SIM_TICKS, update_data_ticks=1)
        self.assertGreaterEqual(len(model.get_spike_times(soma_id=soma_0)), 1,
                                "directly-stimulated soma_0 should fire")


@unittest.skipUnless(
    hasattr(NeuromorphicModel, "get_local_agent_property_value"),
    "local accessors require the editable SAGESim install")
class TestAddLocalSpike(unittest.TestCase):
    """add_local_spike: non-collective, rank-local spike injection.

    On a single-process run rank 0 owns every agent, so the happy path and the
    KeyError-on-unknown-id path are exercised here. The distinctive cross-rank
    contract (owner injects locally; a non-owner raises KeyError; the spike still
    reaches the soma through the collective simulate) is exercised by the
    multi-rank case below, which auto-skips unless launched under mpirun -n 2.
    """

    def test_local_flattened_tensor(self):
        """add_local_spike appends [tick, value] to the owner's tensor."""
        model, _soma_0, _soma_1, syn_ext = _build_chain()
        base = _as_floats(model.get_local_agent_property_value(syn_ext, "input_spikes_tensor"))
        model.add_local_spike(synapse_id=syn_ext, tick=2, value=1.0)
        tensor = model.get_local_agent_property_value(syn_ext, "input_spikes_tensor")
        self.assertEqual(_as_floats(tensor), base + [2.0, 1.0])

    def test_local_equivalent_to_add_spike(self):
        """On one rank, add_local_spike matches collective add_spike bit-for-bit."""
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
        """A non-local (here: nonexistent) id raises KeyError, not a silent no-op.

        On a single rank "owned elsewhere" and "does not exist" are indistinguishable;
        distinguishing them requires a real 2-rank run (see the multi-rank case)."""
        model, _soma_0, _soma_1, _syn_ext = _build_chain()
        with self.assertRaises(KeyError):
            model.add_local_spike(synapse_id=10_000_000, tick=2, value=1.0)

    def test_local_owner_injection_multirank(self):
        """Owner injects locally; non-owner raises KeyError; each soma still fires.

        Two disjoint single-soma networks, one per rank (soma id == rank, external
        synapse id == 100 + rank). Each rank injects only into the synapse it owns;
        injecting into the other rank's synapse must raise KeyError. After the
        collective simulate, both somas have fired.
        """
        comm, rank, size = _get_mpi()
        if size == 1:
            self.skipTest("multi-rank add_local_spike contract needs mpirun -n 2")
        if size != 2:
            self.skipTest(f"fixture supports exactly 2 ranks, got {size}")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Every rank writes both files deterministically (distinct names, no race)
            # then loads only its own.
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

            # Non-owner path: this rank does not own the other rank's synapse.
            with self.assertRaises(KeyError):
                model.add_local_spike(synapse_id=other_syn, tick=2, value=1.0)

            # Owner path: inject a short train locally so the local soma fires.
            for t in (2, 5, 8, 11):
                model.add_local_spike(synapse_id=my_syn, tick=t, value=1.0)

            model.simulate(ticks=SIM_TICKS, update_data_ticks=1)

            # get_spike_times is collective/owner-agnostic, so every rank can read both.
            fired = {sid: len(model.get_spike_times(soma_id=sid)) for sid in (0, 1)}

        self.assertGreaterEqual(fired[0], 1, "soma on rank 0 should fire from its local injection")
        self.assertGreaterEqual(fired[1], 1, "soma on rank 1 should fire from its local injection")


if __name__ == "__main__":
    unittest.main()
