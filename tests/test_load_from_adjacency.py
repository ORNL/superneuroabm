#!/usr/bin/env python
"""Tests for ``NeuromorphicModel.load_from_adjacency`` (Method 2, explicit neighbors).

The headline test is the case ``load_post_owned`` (Method 1) **cannot express**: a
synapse owned by ``rank(pre)`` whose post-soma lives on ANOTHER rank. Under Method 1 the
post-soma can only discover incoming synapses listed in its own file, so a remote
incoming synapse is invisible. Method 2 lets the post-soma's ``neighbors`` name that
synapse explicitly (declared in ``remote_ranks``); SAGESim's ghost exchange then carries
the synapse's current to the post-soma's rank — the same machinery that already serves a
synapse's remote pre-soma.

Topology (the "split synapse"):

    external_input(pre=-1) -> A      [rank 0]
    A --S--> B                       S owned by rank(pre)=rank(A)=0; post-soma B on rank 1

Pass criterion: B's spike train under the distributed Method-2 load equals the spike
train of the SAME topology built single-process with create_soma/create_synapse.

Usage:
    # single-rank smoke (no MPI):
    python -m unittest tests.test_load_from_adjacency.TestLoadFromAdjacency
    # split-synapse across 2 ranks (run on a compute node — needs GPU + MPI):
    srun -N1 -n2 --gpu-bind=closest python -m unittest \
        tests.test_load_from_adjacency.TestLoadFromAdjacency
"""

import pickle
import sys
import tempfile
import unittest
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR.parent))

from superneuroabm.model import NeuromorphicModel


# Fixed global agent ids for the split-synapse fixture.
A = 0           # presynaptic soma (rank 0)
B = 1           # postsynaptic soma (rank 1 in the 2-rank case)
S_EXT = 10      # external-input synapse into A (rank 0)
S = 11          # the A->B synapse (owned by rank(pre)=0)

INPUT_TICK = 2
SIM_TICKS = 60


def _get_mpi():
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        return comm, comm.Get_rank(), comm.Get_size()
    except ImportError:
        return None, 0, 1


def _split_synapse_files(tmpdir, size):
    """Write this run's Method-2 partition file(s) for the split-synapse topology.

    Returns the list of per-rank file paths (length == size). For size==1 the whole
    network is one file with no remote_ranks; for size==2 it is split so that S is
    owned by rank 0 (rank(pre)=rank(A)) while post-soma B lives on rank 1.
    """
    soma_A = {"id": A, "neighbors": [S_EXT]}          # A's incoming = the external synapse
    soma_B = {"id": B, "neighbors": [S]}              # B's incoming = the A->B synapse S
    syn_ext = {"id": S_EXT, "neighbors": [-1]}        # external input: slot0 = -1
    syn_S = {"id": S, "neighbors": [A, B]}            # slot0=pre=A, slot1=post=B

    paths = []
    if size == 1:
        f = Path(tmpdir) / "adj_np1_rank0.pkl"
        with open(f, "wb") as fh:
            pickle.dump({"somas": [soma_A, soma_B],
                         "synapses": [syn_ext, syn_S],
                         "remote_ranks": {}}, fh)
        paths.append(str(f))
    elif size == 2:
        # rank 0: A, the external synapse, and S (owned here since pre=A is local).
        # S references B in slot1 -> B is a remote neighbor here.
        f0 = Path(tmpdir) / "adj_np2_rank0.pkl"
        with open(f0, "wb") as fh:
            pickle.dump({"somas": [soma_A],
                         "synapses": [syn_ext, syn_S],
                         "remote_ranks": {B: 1}}, fh)
        # rank 1: B only. B claims the REMOTE synapse S in its neighbors -> S remote.
        f1 = Path(tmpdir) / "adj_np2_rank1.pkl"
        with open(f1, "wb") as fh:
            pickle.dump({"somas": [soma_B],
                         "synapses": [],
                         "remote_ranks": {S: 0}}, fh)
        paths = [str(f0), str(f1)]
    else:
        raise ValueError(f"fixture supports size 1 or 2, got {size}")
    return paths


def _post_owned_file(tmpdir):
    """Single-rank Method-1 (pre/post) file for the A->B topology (post-owns)."""
    f = Path(tmpdir) / "post_owned_np1.pkl"
    with open(f, "wb") as fh:
        pickle.dump({
            "somas": [{"id": A}, {"id": B}],
            "synapses": [
                {"id": S_EXT, "pre": -1, "post": A},   # external input -> A
                {"id": S, "pre": A, "post": B},        # A -> B
            ],
            "remote_ranks": {},
        }, fh)
    return str(f)


def _reference_spike_times():
    """Spike train of B for the A->B topology, built single-process with create_*."""
    model = NeuromorphicModel(enable_internal_states_tracking=False)
    a = model.create_soma(breed="lif_soma", config_name="config_0")
    b = model.create_soma(breed="lif_soma", config_name="config_0")
    ext = model.create_synapse(breed="single_exp_synapse", pre_soma_id=-1,
                               post_soma_id=a, config_name="config_0")
    model.create_synapse(breed="single_exp_synapse", pre_soma_id=a,
                         post_soma_id=b, config_name="config_0")
    model.setup(use_gpu=True)
    model.add_spike(synapse_id=ext, tick=INPUT_TICK, value=1)
    model.simulate(ticks=SIM_TICKS, update_data_ticks=1)
    return model.get_spike_times(b)


class TestLoadFromAdjacency(unittest.TestCase):

    def test_split_synapse_matches_reference(self):
        """Method-2 load reproduces the create_* reference; works split across ranks.

        At size==1 this exercises both loader loops + validation with no MPI. At
        size==2 it proves the released constraint: B (rank 1) integrates the current of
        synapse S (owned by rank 0) via ghost exchange.
        """
        comm, rank, size = _get_mpi()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Every rank builds the file set deterministically (no shared FS race on
            # distinct names); each rank then loads only its own file.
            paths = _split_synapse_files(tmpdir, size)
            my_file = paths[rank]

            model = NeuromorphicModel(enable_internal_states_tracking=False)
            model.load_from_adjacency(my_file)
            model.setup(use_gpu=True)

            # The external input synapse S_EXT is owned by rank 0 (it feeds A, on rank 0).
            # add_spike is collective; non-owners no-op, so every rank may call it.
            model.add_spike(synapse_id=S_EXT, tick=INPUT_TICK, value=1)
            model.simulate(ticks=SIM_TICKS, update_data_ticks=1)

            got = model.get_spike_times(B)          # collective getter, owner-agnostic

        if rank == 0:
            expected = _reference_spike_times()
            self.assertEqual(
                sorted(got), sorted(expected),
                f"size={size}: B spike train {sorted(got)} != reference "
                f"{sorted(expected)}. Method-2 split load did not reproduce the "
                "create_* topology (post-soma failed to integrate the remote synapse).")

    def test_post_owned_matches_reference(self):
        """load_post_owned (Method 1) reproduces the create_* reference (refactor guard).

        Single-rank; no MPI. Confirms the shared _build_from_partition core, exercised
        through the post-owns derivation closures, builds the same model as create_*.
        """
        comm, rank, size = _get_mpi()
        if size > 1:
            self.skipTest("post-owns single-file regression is a 1-rank test")
        with tempfile.TemporaryDirectory() as tmpdir:
            model = NeuromorphicModel(enable_internal_states_tracking=False)
            model.load_post_owned(_post_owned_file(tmpdir))
            model.setup(use_gpu=True)
            model.add_spike(synapse_id=S_EXT, tick=INPUT_TICK, value=1)
            model.simulate(ticks=SIM_TICKS, update_data_ticks=1)
            got = model.get_spike_times(B)
        self.assertEqual(sorted(got), sorted(_reference_spike_times()))

    def test_post_owned_rejects_nonlocal_post(self):
        """load_post_owned enforces its name: a non-local post-soma raises."""
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "bad_postowns.pkl"
            with open(f, "wb") as fh:
                # Synapse S has post=B, but B is NOT a local soma in this file.
                pickle.dump({"somas": [{"id": A}],
                             "synapses": [{"id": S, "pre": A, "post": B}],
                             "remote_ranks": {}}, fh)
            model = NeuromorphicModel(enable_internal_states_tracking=False)
            with self.assertRaises(ValueError):
                model.load_post_owned(str(f))

    def test_rejects_method1_file(self):
        """A Method-1 (pre/post, no neighbors) file fails loudly in the normalizer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "m1.pkl"
            with open(f, "wb") as fh:
                pickle.dump({"somas": [{"id": A}],
                             "synapses": [{"id": S, "pre": -1, "post": A}],
                             "remote_ranks": {}}, fh)
            model = NeuromorphicModel(enable_internal_states_tracking=False)
            with self.assertRaises(ValueError):
                model.load_from_adjacency(str(f))

    def test_rejects_bad_synapse_neighbor_count(self):
        """A synapse with 3 neighbors (not [pre] or [pre,post]) is rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "bad.pkl"
            with open(f, "wb") as fh:
                pickle.dump({"somas": [{"id": A, "neighbors": []},
                                       {"id": B, "neighbors": [S]}],
                             "synapses": [{"id": S, "neighbors": [A, B, A]}],
                             "remote_ranks": {}}, fh)
            model = NeuromorphicModel(enable_internal_states_tracking=False)
            with self.assertRaises(ValueError):
                model.load_from_adjacency(str(f))

    def test_rejects_undeclared_remote_neighbor(self):
        """A neighbor that is neither local, -1, nor in remote_ranks is rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "dangling.pkl"
            with open(f, "wb") as fh:
                # B references S, but S is neither local (no such synapse here) nor remote.
                pickle.dump({"somas": [{"id": B, "neighbors": [S]}],
                             "synapses": [],
                             "remote_ranks": {}}, fh)
            model = NeuromorphicModel(enable_internal_states_tracking=False)
            with self.assertRaises(ValueError):
                model.load_from_adjacency(str(f))


if __name__ == "__main__":
    unittest.main()
