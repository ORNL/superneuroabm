#!/usr/bin/env python
"""Tests for the Brunel generator's ``remote_rank_fanout`` (exactly-R remote peers).

These are pure generator tests -- they inspect the partition dicts
``brunel_partition`` returns and need no GPU or MPI. With ``remote_rank_fanout=R`` set, every
partition draws its recurrent sources from its own rank plus **exactly R** distinct remote
ranks -- a hard-constant *read-from* (incoming) peer count, independent of ``num_partitions``
(see ``docs/BRUNEL_SCALING.md`` D4).

**Direction matters (D4).** ``remote_rank_fanout=R`` pins only the INCOMING peers -- the
ranks a partition reads ghost soma-state *from* (its ``remote_ranks``). The OUTGOING peers
-- the ranks a partition must *send* to, i.e. the ranks that drew from it -- are the emergent
in-degree of the reads-from digraph: their count is exactly ``R`` **on average** (edge
conservation: every read-from edge is one rank's send-to edge, so ``sum = P*R``), but an
individual rank's send-to count can exceed ``R`` (balls-in-bins; max grows ~``log P``, not
linearly). SAGESim's per-tick ``mpi_num_peers`` is the send-to count, so the flat quantity
there is the *mean*, not every rank's value. Both directions are checked below.

(For a hard-constant peer count in *both* directions with no ring/torus, see the geometric
``topology="torus2d"`` wiring and ``test_brunel_spatial_torus.py`` -- there every rank both
reads from and sends to exactly its 8 grid neighbors.)

We also check the E/I layout is interleaved per rank (so any peer can supply both E and I
sources), the fixed in-degree is preserved, inhibitory synapses carry a negative weight,
and the legacy global-uniform path (``None``) is intact.

Usage:
    python -m unittest tests.test_brunel_bounded_fanout
"""

import sys
import unittest
from collections import Counter
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR.parent))

from superneuroabm.brunel import brunel_partition


def _recurrent_sources_by_post(partition):
    """{post_soma_id: [pre ids]} for recurrent (pre != -1) synapses only."""
    by_post = {}
    for s in partition["synapses"]:
        if s["pre"] != -1:
            by_post.setdefault(s["post"], []).append(s["pre"])
    return by_post


class TestBoundedFanout(unittest.TestCase):
    # Small but non-degenerate: 8 ranks, 500 somas/rank, K = 40 + 10.
    P = 8
    NPP = 500
    C_E = 40
    C_I = 10
    R = 2
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
            inhibitory_weight_ratio=5.0, remote_rank_fanout=self.R, seed=self.SEED,
        )
        kwargs.update(overrides)
        return brunel_partition(**kwargs)

    def test_exactly_R_incoming_peer_ranks(self):
        """Every rank READS FROM exactly R other ranks -- not more, not less.

        This is the INCOMING (receive) direction: ``remote_ranks`` records the ranks a
        partition draws ghost soma-state *from*, and ``remote_rank_fanout=R`` pins that
        count to exactly ``R`` for every rank, independent of ``num_partitions``. (The
        OUTGOING send-to count is a different, emergent quantity -- see
        ``test_outgoing_peers_mean_is_R``.)
        """
        for rank in range(self.P):
            part = self._partition(rank)
            # remote_ranks maps {remote_soma_id: owner_rank}; the peer count is the number
            # of DISTINCT owner ranks, and it must be exactly R for every rank.
            peers = set(part["remote_ranks"].values())
            self.assertEqual(
                len(peers), self.R,
                f"rank {rank}: expected exactly {self.R} read-from peer ranks, got "
                f"{len(peers)}: {sorted(peers)}",
            )
            self.assertNotIn(rank, peers, f"rank {rank}: self appears in remote_ranks")

    def test_outgoing_peers_mean_is_R(self):
        """Send-to peer count is R on average (edge conservation), not R per rank.

        A rank ``q`` must send its soma-state to every rank that drew a source from ``q``
        -- the in-degree of ``q`` in the reads-from digraph. Because every read-from edge
        (there are exactly ``P * R`` of them, ``R`` per rank) is exactly one rank's send-to
        edge, the total send-to edge count equals ``P * R`` and the mean send-to count is
        exactly ``R``. Individual ranks may exceed ``R`` (balls-in-bins), so this is the
        quantity that is flat *in the mean* -- which is what SAGESim's per-tick
        ``mpi_num_peers`` measures -- rather than constant per rank.
        """
        # Build the reads-from digraph across all partitions: edge r -> q iff rank r draws
        # a source from rank q (q is then obliged to send to r).
        send_to = Counter()
        total_read_edges = 0
        for rank in range(self.P):
            peers = set(self._partition(rank)["remote_ranks"].values())
            total_read_edges += len(peers)
            for q in peers:
                send_to[q] += 1
        # Edge conservation: every rank reads from exactly R, so the digraph has P*R edges,
        # counted identically whether summed over read-from (out) or send-to (in) degree.
        self.assertEqual(total_read_edges, self.P * self.R)
        self.assertEqual(sum(send_to.values()), self.P * self.R)
        self.assertAlmostEqual(sum(send_to.values()) / self.P, self.R)

    def test_fixed_in_degree_and_ei_split(self):
        """Each soma gets exactly C_E excitatory + C_I inhibitory recurrent inputs."""
        for rank in range(self.P):
            part = self._partition(rank)
            by_post = _recurrent_sources_by_post(part)
            # every local soma is a target
            self.assertEqual(len(by_post), self.NPP)
            for post, pres in by_post.items():
                self.assertEqual(len(pres), self.C_E + self.C_I,
                                 f"rank {rank} soma {post}: in-degree {len(pres)}")
                n_exc = sum(self._is_exc(p) for p in pres)
                self.assertEqual(n_exc, self.C_E,
                                 f"rank {rank} soma {post}: {n_exc} exc sources "
                                 f"(want {self.C_E})")

    def test_sources_bounded_to_allowed_ranks(self):
        """No recurrent source lives outside {own rank} u (the R chosen peers)."""
        for rank in range(self.P):
            part = self._partition(rank)
            allowed = set(part["remote_ranks"].values()) | {rank}
            for s in part["synapses"]:
                if s["pre"] >= 0:
                    self.assertIn(s["pre"] // self.NPP, allowed,
                                  f"rank {rank}: source {s['pre']} outside {allowed}")

    def test_per_rank_ei_composition(self):
        """Each rank owns exc_per_rank excitatory + the rest inhibitory somas."""
        for rank in range(self.P):
            part = self._partition(rank)
            ids = [s["id"] for s in part["somas"]]
            self.assertEqual(len(ids), self.NPP)
            n_exc = sum(self._is_exc(i) for i in ids)
            self.assertEqual(n_exc, self.exc_per_rank)

    def test_inhibitory_weight_is_negative(self):
        """A synapse from an inhibitory source carries a negative weight (the sign is
        topological -- the kernel has no sign logic)."""
        part = self._partition(0)
        for s in part["synapses"]:
            if s["pre"] >= 0 and not self._is_exc(s["pre"]):
                w = s["overrides"]["hyperparameters"]["weight"]
                self.assertLess(w, 0.0, f"inhibitory synapse weight {w} is not negative")

    def test_no_autapses(self):
        for rank in range(self.P):
            part = self._partition(rank)
            for s in part["synapses"]:
                self.assertNotEqual(s["pre"], s["post"],
                                    f"rank {rank}: autapse at soma {s['post']}")

    def test_peer_count_independent_of_num_partitions(self):
        """The whole point: peer count stays == R as the machine grows."""
        for P in (4, 8, 16, 32):
            part = brunel_partition(
                somas_per_rank=self.NPP, num_partitions=P, partition_rank=0,
                excitatory_in_degree=self.C_E, inhibitory_in_degree=self.C_I,
                excitatory_weight=14.0, remote_rank_fanout=self.R, seed=self.SEED,
            )
            peers = set(part["remote_ranks"].values())
            self.assertEqual(len(peers), self.R,
                             f"P={P}: peer count {len(peers)} != R={self.R}")

    def test_validation_errors(self):
        with self.assertRaises(ValueError):   # R must be < num_partitions
            brunel_partition(somas_per_rank=self.NPP, num_partitions=4, partition_rank=0,
                             excitatory_in_degree=self.C_E, inhibitory_in_degree=self.C_I,
                             remote_rank_fanout=4, seed=self.SEED)
        with self.assertRaises(ValueError):   # R must be non-negative
            brunel_partition(somas_per_rank=self.NPP, num_partitions=8, partition_rank=0,
                             excitatory_in_degree=self.C_E, inhibitory_in_degree=self.C_I,
                             remote_rank_fanout=-1, seed=self.SEED)

    def test_global_uniform_still_works(self):
        """remote_rank_fanout=None keeps the legacy whole-population draw."""
        part = brunel_partition(
            somas_per_rank=self.NPP, num_partitions=self.P, partition_rank=0,
            excitatory_in_degree=self.C_E, inhibitory_in_degree=self.C_I,
            excitatory_weight=14.0, remote_rank_fanout=None, seed=self.SEED,
        )
        by_post = _recurrent_sources_by_post(part)
        for pres in by_post.values():
            self.assertEqual(len(pres), self.C_E + self.C_I)
            self.assertEqual(sum(self._is_exc(p) for p in pres), self.C_E)
        # global-uniform over 8 ranks: rank 0 should touch many peers (not bounded to R)
        peers = set(part["remote_ranks"].values())
        self.assertGreater(len(peers), self.R)


if __name__ == "__main__":
    unittest.main()
