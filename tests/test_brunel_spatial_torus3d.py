#!/usr/bin/env python
"""Tests for the Brunel generator's ``topology="torus3d"`` 3D spatial-radius stencil.

These are pure generator tests -- they inspect the partition dicts ``brunel_partition`` returns
and need no GPU or MPI. ``torus3d`` is the **weak-scaling / point-to-point** wiring (see
``docs/BRUNEL_SCALING.md`` §5.5a): the ``num_partitions`` ranks tile a periodic ``a x b x c``
torus, each rank's contiguous id block is a ``gx x gy x gz`` spatial sub-block, and every neuron
draws its ``K`` recurrent sources uniformly within a Euclidean ``connection_radius`` of its own
position. This gives **volume-local compute + surface-halo communication**: interior neurons are
fully local, only near-boundary neurons reach neighbor tiles, and the realized remote peer set is
a subset of the 26 Moore neighbors that **ramps** with the machine (0, 1, ..., 26) and
**plateaus** at 26 once every grid dimension is >= 3.

Unlike the strict 2D ``torus2d`` (which requires ``a, b >= 3``), the 3D grid factorization is
**relaxed** so the weak-scaling sweep can start at a single GPU (``1x1x1``) and grow.

Usage:
    python -m unittest tests.test_brunel_spatial_torus3d
"""

import sys
import unittest
from pathlib import Path

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR.parent))

from superneuroabm.brunel import (brunel_partition, _factor_near_cube,
                                   _grid_factorization_3d, _select_neighbor_tiles_3d,
                                   _soma_positions, _positions_to_soma,
                                   _ball_offsets, _radius_for_indegree)


def _recurrent_sources_by_post(partition):
    """{post_soma_id: [pre ids]} for recurrent (pre != -1) synapses only."""
    by_post = {}
    for s in partition["synapses"]:
        if s["pre"] != -1:
            by_post.setdefault(s["post"], []).append(s["pre"])
    return by_post


class TestGridFactorization3d(unittest.TestCase):
    def test_product_and_ordering(self):
        """a*b*c == M, a <= b <= c, dims >= 1, for any M >= 1 (relaxed -- no minimum)."""
        for M in (1, 2, 4, 8, 16, 27, 32, 64, 128, 256, 512, 1024, 1000, 216):
            a, b, c = _grid_factorization_3d(M)
            self.assertEqual(a * b * c, M, f"M={M}")
            self.assertLessEqual(a, b, f"M={M}")
            self.assertLessEqual(b, c, f"M={M}")
            self.assertGreaterEqual(a, 1, f"M={M}")

    def test_known_near_cube_factorizations(self):
        self.assertEqual(_grid_factorization_3d(1), (1, 1, 1))
        self.assertEqual(_grid_factorization_3d(8), (2, 2, 2))
        self.assertEqual(_grid_factorization_3d(27), (3, 3, 3))
        self.assertEqual(_grid_factorization_3d(64), (4, 4, 4))
        self.assertEqual(_grid_factorization_3d(512), (8, 8, 8))

    def test_is_most_cube_like(self):
        """The chosen (a,b,c) minimizes c-a over all valid factorizations."""
        def brute(M):
            best = None
            for a in range(1, M + 1):
                if M % a:
                    continue
                for b in range(a, M // a + 1):
                    if (M // a) % b:
                        continue
                    c = M // a // b
                    if c < b:
                        continue
                    if best is None or (c - a) < best[0]:
                        best = (c - a, (a, b, c))
            return best[1]
        for M in (12, 30, 60, 100, 128, 360):
            self.assertEqual(_grid_factorization_3d(M), brute(M), f"M={M}")


class TestNeighborRampAndPlateau(unittest.TestCase):
    def test_peer_count_ramps_then_plateaus(self):
        """Moore-neighbor count ramps with the grid and plateaus at 26 once every dim >= 3."""
        expected = {1: 0, 2: 1, 4: 3, 8: 7, 27: 26, 64: 26, 512: 26}
        for M, want in expected.items():
            a, b, c = _grid_factorization_3d(M)
            self.assertEqual(len(_select_neighbor_tiles_3d(0, a, b, c)), want, f"M={M}")

    def test_plateau_is_26_for_all_ranks(self):
        """With all dims >= 3, every rank has exactly 26 distinct Moore neighbors (not self)."""
        for M in (27, 64, 128, 216, 512):
            a, b, c = _grid_factorization_3d(M)
            for r in range(M):
                nbrs = _select_neighbor_tiles_3d(r, a, b, c)
                self.assertEqual(len(nbrs), 26, f"M={M} rank {r}")
                self.assertNotIn(r, nbrs, f"M={M} rank {r}: self is a neighbor")


class TestCoordinateBijection(unittest.TestCase):
    M = 27
    NPP = 1000

    def test_round_trip_all_ids(self):
        ids = np.arange(self.M * self.NPP)
        X, Y, Z = _soma_positions(ids, self.M, self.NPP)
        back = _positions_to_soma(X, Y, Z, self.M, self.NPP)
        self.assertTrue(np.array_equal(back, ids))

    def test_positions_unique_and_complete(self):
        """The bijection covers the whole A x B x C grid exactly once."""
        ids = np.arange(self.M * self.NPP)
        X, Y, Z = _soma_positions(ids, self.M, self.NPP)
        pts = set(zip(X.tolist(), Y.tolist(), Z.tolist()))
        self.assertEqual(len(pts), self.M * self.NPP)

    def test_rank_block_is_one_spatial_tile(self):
        """Each rank's contiguous id block maps to exactly one (tile_x, tile_y, tile_z)."""
        a, b, c = _grid_factorization_3d(self.M)
        gx, gy, gz = _factor_near_cube(self.NPP)
        for r in (0, 5, 13, 26):
            ids = np.arange(r * self.NPP, (r + 1) * self.NPP)
            X, Y, Z = _soma_positions(ids, self.M, self.NPP)
            tiles = set(zip((X // gx).tolist(), (Y // gy).tolist(), (Z // gz).tolist()))
            self.assertEqual(len(tiles), 1, f"rank {r}: block spans {tiles}")


class TestBallOffsets(unittest.TestCase):
    def test_excludes_origin_and_respects_radius(self):
        off = _ball_offsets(3.0)
        d2 = (off ** 2).sum(axis=1)
        self.assertTrue((d2 > 0).all())          # no self
        self.assertTrue((d2 <= 9).all())         # within radius
        self.assertNotIn((0, 0, 0), {tuple(o) for o in off})

    def test_radius_for_indegree_has_headroom(self):
        for K in (50, 100, 500, 1250):
            r = _radius_for_indegree(K)
            self.assertGreaterEqual(_ball_offsets(r).shape[0], 2 * K, f"K={K}")


class TestSpatialTorus3d(unittest.TestCase):
    # 3x3x3 grid, 1000 somas/rank (10x10x10 tile), K = 80 + 20.
    M = 27
    NPP = 1000
    C_E = 80
    C_I = 20
    EXC_FRACTION = 0.8
    SEED = 42

    @property
    def exc_per_rank(self):
        return round(self.EXC_FRACTION * self.NPP)

    def _is_exc(self, soma_id):
        return (soma_id % self.NPP) < self.exc_per_rank

    def _partition(self, rank, **overrides):
        kwargs = dict(
            somas_per_rank=self.NPP, num_partitions=self.M, partition_rank=rank,
            excitatory_in_degree=self.C_E, inhibitory_in_degree=self.C_I,
            excitatory_fraction=self.EXC_FRACTION, excitatory_weight=0.1,
            inhibitory_weight_ratio=5.0, topology="torus3d", seed=self.SEED,
        )
        kwargs.update(overrides)
        return brunel_partition(**kwargs)

    def test_fixed_in_degree(self):
        """Every soma receives exactly K = C_E + C_I recurrent inputs (fixed compute)."""
        K = self.C_E + self.C_I
        for rank in (0, 13, 26):
            by_post = _recurrent_sources_by_post(self._partition(rank))
            self.assertEqual(len(by_post), self.NPP)
            for post, pres in by_post.items():
                self.assertEqual(len(pres), K, f"rank {rank} soma {post}")

    def test_no_autapses(self):
        for rank in (0, 13):
            for s in self._partition(rank)["synapses"]:
                self.assertNotEqual(s["pre"], s["post"], f"autapse at {s['post']}")

    def test_contiguous_id_block_per_rank(self):
        for rank in (0, 7, 26):
            ids = sorted(s["id"] for s in self._partition(rank)["somas"])
            self.assertEqual(ids, list(range(rank * self.NPP, (rank + 1) * self.NPP)))

    def test_peers_subset_of_moore_neighbors(self):
        """No recurrent source lives outside the own tile + 26 Moore-neighbor tiles."""
        a, b, c = _grid_factorization_3d(self.M)
        for rank in (0, 13, 26):
            part = self._partition(rank)
            allowed = set(_select_neighbor_tiles_3d(rank, a, b, c)) | {rank}
            peers = {s["pre"] // self.NPP for s in part["synapses"] if s["pre"] >= 0}
            self.assertTrue(peers <= allowed, f"rank {rank}: {peers - allowed} outside halo")

    def test_own_rank_is_dominant(self):
        """Most incoming edges are local (own rank) -- the surface/volume stencil property."""
        for rank in (0, 13, 26):
            part = self._partition(rank)
            pre = np.array([s["pre"] for s in part["synapses"] if s["pre"] != -1])
            local_fraction = np.mean((pre // self.NPP) == rank)
            self.assertGreater(local_fraction, 0.5, f"rank {rank}: only {local_fraction:.2f} local")

    def test_local_fraction_rises_with_tile_size(self):
        """Bigger tiles -> more interior -> higher local fraction (lighter halo). Surface/volume."""
        def local_frac(npp):
            part = self._partition(0, somas_per_rank=npp)
            pre = np.array([s["pre"] for s in part["synapses"] if s["pre"] != -1])
            return np.mean((pre // npp) == 0)
        self.assertLess(local_frac(1000), local_frac(8000))

    def test_ghost_is_bounded_not_whole_tiles(self):
        """Remote sources form a surface halo far smaller than whole neighbor tiles."""
        part = self._partition(13)
        remote = [s["pre"] for s in part["synapses"] if s["pre"] >= 0
                  and s["pre"] // self.NPP != 13]
        n_peers = len({p // self.NPP for p in remote})
        distinct_remote = len(set(remote))
        # A whole-tile pull would be n_peers * NPP; the surface halo is a fraction of that.
        self.assertLess(distinct_remote, n_peers * self.NPP,
                        "ghost should be surface strips, not whole tiles")

    def test_remote_ranks_recorded(self):
        """remote_ranks maps every off-tile presynaptic soma to its owning rank."""
        part = self._partition(13)
        rr = part["remote_ranks"]
        for s in part["synapses"]:
            if s["pre"] >= 0 and s["pre"] // self.NPP != 13:
                self.assertIn(s["pre"], rr)
                self.assertEqual(rr[s["pre"]], s["pre"] // self.NPP)

    def test_inhibitory_weight_is_negative(self):
        part = self._partition(0)
        for s in part["synapses"]:
            if s["pre"] >= 0 and not self._is_exc(s["pre"]):
                w = s["overrides"]["hyperparameters"]["weight"]
                self.assertLess(w, 0.0, f"inhibitory synapse weight {w} not negative")

    def test_single_partition_baseline_all_local(self):
        """M=1 (the 1-GPU weak-scaling baseline): every source is local, no remote_ranks."""
        part = brunel_partition(somas_per_rank=self.NPP, num_partitions=1, partition_rank=0,
                                excitatory_in_degree=self.C_E, inhibitory_in_degree=self.C_I,
                                topology="torus3d", excitatory_weight=0.1, seed=self.SEED)
        self.assertNotIn("remote_ranks", part)
        pres = [s["pre"] for s in part["synapses"] if s["pre"] != -1]
        self.assertTrue(all(0 <= p < self.NPP for p in pres))

    def test_reaches_all_26_peers_when_grid_saturated(self):
        """With every dim >= 3 and a large enough tile, all 26 Moore neighbors are realized."""
        a, b, c = _grid_factorization_3d(self.M)
        part = self._partition(13)
        peers = {s["pre"] // self.NPP for s in part["synapses"] if s["pre"] >= 0} - {13}
        self.assertEqual(peers, set(_select_neighbor_tiles_3d(13, a, b, c)))

    def test_explicit_connection_radius(self):
        """A user-set connection_radius is honored (larger radius -> more peers/ghost)."""
        small = self._partition(13, somas_per_rank=8000, connection_radius=3.0)
        pre = np.array([s["pre"] for s in small["synapses"] if s["pre"] != -1])
        self.assertGreater(np.mean((pre // 8000) == 13), 0.8)  # small radius -> very local


if __name__ == "__main__":
    unittest.main()
