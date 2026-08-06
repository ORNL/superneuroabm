#!/usr/bin/env python
"""Tests for the Brunel generator's ``topology="torus2d"`` spatial wiring.

These are pure generator tests -- they inspect the partition dicts ``brunel_partition``
returns and need no GPU or MPI. The invariant under test is the one that makes weak scaling
hold for SuperNeuroABM under a *geometric* partition (see ``docs/BRUNEL_SCALING.md``): the
``num_partitions`` ranks form an ``a x b`` torus grid of tiles (one tile = one rank's
contiguous soma block), and every soma draws its recurrent sources only from its own tile
plus its **8 Moore-neighbor tiles**. So each rank's distinct remote peer count is a
**hard constant == 8**, independent of ``num_partitions`` -- the spatial-locality analogue
of the bounded-fanout invariant, but with the peer set fixed by geometry rather than random.

We check: the grid factorization is near-square with both dims >= 3 (and rejects
primes / M < 9); every rank reaches exactly its 8 Moore neighbors (torus-wrapped, so corner
ranks are not special); no recurrent source lives outside the 9-tile neighborhood; each
rank owns a contiguous soma block; the fixed two-pool in-degree and E/I layout are
preserved; inhibitory synapses carry a negative weight; and there are no autapses.

(The within-tile ``(X, Y)`` coordinate layout is not exercised here because the tile-block
kernel draws uniformly over whole tiles -- it never uses intra-tile positions. Those
coordinates + their id<->(X,Y) bijection arrive with the continuous distance kernel.)

Usage:
    python -m unittest tests.test_brunel_spatial_torus
"""

import math
import sys
import unittest
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR.parent))

from superneuroabm.brunel import (brunel_partition, _grid_factorization,
                                   _select_neighbor_tiles)


def _recurrent_sources_by_post(partition):
    """{post_soma_id: [pre ids]} for recurrent (pre != -1) synapses only."""
    by_post = {}
    for s in partition["synapses"]:
        if s["pre"] != -1:
            by_post.setdefault(s["post"], []).append(s["pre"])
    return by_post


def _largest_divisor_leq_sqrt(M):
    best = 1
    for d in range(1, math.isqrt(M) + 1):
        if M % d == 0:
            best = d
    return best


class TestGridFactorization(unittest.TestCase):
    def test_valid_grids_are_near_square(self):
        """a*b == M, a <= b, both >= 3, and a is the largest divisor <= sqrt(M)."""
        for M in (9, 12, 16, 32, 64, 128, 256, 1024):
            a, b = _grid_factorization(M)
            self.assertEqual(a * b, M, f"M={M}")
            self.assertLessEqual(a, b, f"M={M}")
            self.assertGreaterEqual(a, 3, f"M={M}: dim < 3")
            self.assertEqual(a, _largest_divisor_leq_sqrt(M),
                             f"M={M}: a={a} is not the most-square factor")

    def test_known_factorizations(self):
        self.assertEqual(_grid_factorization(16), (4, 4))
        self.assertEqual(_grid_factorization(32), (4, 8))
        self.assertEqual(_grid_factorization(64), (8, 8))
        self.assertEqual(_grid_factorization(128), (8, 16))

    def test_rejects_primes_and_too_small(self):
        """Primes and M < 9 (and M like 2*prime) cannot make an a,b>=3 torus."""
        for bad in (1, 2, 4, 5, 7, 8, 11, 13, 17):
            with self.assertRaises(ValueError, msg=f"M={bad} should raise"):
                _grid_factorization(bad)


class TestNeighborTiles(unittest.TestCase):
    def test_rank0_on_4x4_wraps_to_8_neighbors(self):
        """Rank 0 (a corner) on a 4x4 torus wraps to row 3 / col 3 -> 8 distinct tiles."""
        # (dr,dc) offsets from (0,0) on a 4x4 torus -> tiles {15,12,13,3,1,7,4,5}.
        self.assertEqual(_select_neighbor_tiles(0, 4, 4),
                         sorted([15, 12, 13, 3, 1, 7, 4, 5]))

    def test_always_8_distinct_excluding_self(self):
        for M in (16, 32, 64, 128):
            a, b = _grid_factorization(M)
            for r in range(M):
                nbrs = _select_neighbor_tiles(r, a, b)
                self.assertEqual(len(nbrs), 8, f"M={M} rank {r}: {nbrs}")
                self.assertNotIn(r, nbrs, f"M={M} rank {r}: self is a neighbor")


class TestSpatialTorus(unittest.TestCase):
    # 4x4 grid, 200 somas/rank, K = 16 + 4. npp*K/9 ~ 444 draws/tile -> all 8 realized.
    P = 16
    NPP = 200
    C_E = 16
    C_I = 4
    EXC_FRACTION = 0.8
    SEED = 42

    @property
    def exc_per_rank(self):
        return round(self.EXC_FRACTION * self.NPP)

    def _is_exc(self, soma_id):
        """E/I identity is interleaved per rank: E iff (id % npp) < exc_per_rank."""
        return (soma_id % self.NPP) < self.exc_per_rank

    def _partition(self, rank, **overrides):
        kwargs = dict(
            somas_per_rank=self.NPP, num_partitions=self.P, partition_rank=rank,
            excitatory_in_degree=self.C_E, inhibitory_in_degree=self.C_I,
            excitatory_fraction=self.EXC_FRACTION, excitatory_weight=14.0,
            inhibitory_weight_ratio=5.0, topology="torus2d", seed=self.SEED,
        )
        kwargs.update(overrides)
        return brunel_partition(**kwargs)

    def test_exactly_8_peer_ranks(self):
        """Every rank communicates with exactly 8 other ranks -- its Moore neighbors."""
        a, b = _grid_factorization(self.P)
        for rank in range(self.P):
            part = self._partition(rank)
            peers = set(part["remote_ranks"].values())
            self.assertEqual(len(peers), 8,
                             f"rank {rank}: {len(peers)} peers ({sorted(peers)})")
            self.assertNotIn(rank, peers)

    def test_peers_are_the_moore_neighbors(self):
        """The realized peer set is *exactly* the 8 grid neighbors (not just <= 8)."""
        a, b = _grid_factorization(self.P)
        for rank in range(self.P):
            part = self._partition(rank)
            peers = set(part["remote_ranks"].values())
            self.assertEqual(peers, set(_select_neighbor_tiles(rank, a, b)),
                             f"rank {rank}: peers != Moore neighbors")

    def test_peer_count_independent_of_num_partitions(self):
        """The whole point: peer count stays == 8 as the machine grows."""
        for P in (16, 32, 64, 128):
            part = brunel_partition(
                somas_per_rank=self.NPP, num_partitions=P, partition_rank=0,
                excitatory_in_degree=self.C_E, inhibitory_in_degree=self.C_I,
                excitatory_weight=14.0, topology="torus2d", seed=self.SEED,
            )
            self.assertEqual(len(set(part["remote_ranks"].values())), 8, f"P={P}")

    def test_sources_bounded_to_9_tiles(self):
        """No recurrent source lives outside {own tile} u (the 8 neighbor tiles)."""
        a, b = _grid_factorization(self.P)
        for rank in range(self.P):
            part = self._partition(rank)
            allowed = set(_select_neighbor_tiles(rank, a, b)) | {rank}
            for s in part["synapses"]:
                if s["pre"] >= 0:
                    self.assertIn(s["pre"] // self.NPP, allowed,
                                  f"rank {rank}: source {s['pre']} outside {allowed}")

    def test_contiguous_id_block_per_rank(self):
        """Each rank owns exactly the contiguous soma block [r*npp, (r+1)*npp)."""
        for rank in range(self.P):
            part = self._partition(rank)
            ids = sorted(s["id"] for s in part["somas"])
            self.assertEqual(ids, list(range(rank * self.NPP, (rank + 1) * self.NPP)),
                             f"rank {rank}: soma block not contiguous")

    def test_fixed_in_degree_and_ei_split(self):
        """Each soma gets exactly C_E excitatory + C_I inhibitory recurrent inputs."""
        for rank in range(self.P):
            part = self._partition(rank)
            by_post = _recurrent_sources_by_post(part)
            self.assertEqual(len(by_post), self.NPP)
            for post, pres in by_post.items():
                self.assertEqual(len(pres), self.C_E + self.C_I,
                                 f"rank {rank} soma {post}: in-degree {len(pres)}")
                n_exc = sum(self._is_exc(p) for p in pres)
                self.assertEqual(n_exc, self.C_E,
                                 f"rank {rank} soma {post}: {n_exc} exc (want {self.C_E})")

    def test_per_rank_ei_composition(self):
        """Each rank owns exc_per_rank excitatory + the rest inhibitory somas."""
        for rank in range(self.P):
            part = self._partition(rank)
            ids = [s["id"] for s in part["somas"]]
            self.assertEqual(sum(self._is_exc(i) for i in ids), self.exc_per_rank)

    def test_inhibitory_weight_is_negative(self):
        part = self._partition(0)
        for s in part["synapses"]:
            if s["pre"] >= 0 and not self._is_exc(s["pre"]):
                w = s["overrides"]["hyperparameters"]["weight"]
                self.assertLess(w, 0.0, f"inhibitory synapse weight {w} not negative")

    def test_no_autapses(self):
        for rank in range(self.P):
            part = self._partition(rank)
            for s in part["synapses"]:
                self.assertNotEqual(s["pre"], s["post"],
                                    f"rank {rank}: autapse at soma {s['post']}")

    def test_invalid_grid_raises(self):
        """A worker count that can't form an a,b>=3 grid is rejected."""
        with self.assertRaises(ValueError):
            brunel_partition(somas_per_rank=self.NPP, num_partitions=8, partition_rank=0,
                             excitatory_in_degree=self.C_E, inhibitory_in_degree=self.C_I,
                             topology="torus2d", seed=self.SEED)


if __name__ == "__main__":
    unittest.main()
