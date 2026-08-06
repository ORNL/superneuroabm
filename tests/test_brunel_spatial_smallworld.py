#!/usr/bin/env python
"""Tests for ``spatial_smallworld_partition`` -- the spatial economical small-world (Deliverable 2).

Pure generator tests (no GPU/MPI). The realistic-scenario network (Bassett & Bullmore 2017): 3D
distance-dependent connectivity (variable in-degree) with a tighter inhibitory kernel, plus a
Watts-Strogatz long-range shortcut fraction ``beta``. This is a *software stress test*, not a
weak-scaling benchmark -- its shortcuts make the peer count grow with the machine, by design.

We check: variable in-degree with the right mean; distance-decay (inhibition more local than
excitation); the small-world signature (beta=0 -> local lattice = 26-neighbor halo, no shortcuts;
beta>0 -> long-range edges appear and the peer count climbs past 26); fine E/I identity; no
autapses; negative inhibitory weights; unique synapse ids; and that the reserved ``longrange_form``
values raise.

Usage:
    python -m unittest tests.test_brunel_spatial_smallworld
"""

import sys
import unittest
from pathlib import Path

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR.parent))

from superneuroabm.brunel import (spatial_smallworld_partition, _is_excitatory_spatial,
                                   _soma_positions, _grid_factorization_3d,
                                   _factor_near_cube, _select_neighbor_tiles_3d)


def _sources_by_post(partition):
    by_post = {}
    for s in partition["synapses"]:
        if s["pre"] != -1:
            by_post.setdefault(s["post"], []).append(s["pre"])
    return by_post


def _torus_dist(id_a, id_b, M, npp):
    """Euclidean distance between two somas on the periodic global grid."""
    a, b, c = _grid_factorization_3d(M)
    gx, gy, gz = _factor_near_cube(npp)
    A, B, C = a * gx, b * gy, c * gz
    xa, ya, za = _soma_positions(id_a, M, npp)
    xb, yb, zb = _soma_positions(id_b, M, npp)
    dx = np.minimum((xa - xb) % A, (xb - xa) % A)
    dy = np.minimum((ya - yb) % B, (yb - ya) % B)
    dz = np.minimum((za - zb) % C, (zb - za) % C)
    return np.sqrt(dx * dx + dy * dy + dz * dz)


class TestFineEIIdentity(unittest.TestCase):
    def test_global_fraction_matches(self):
        ids = np.arange(50000)
        for frac in (0.8, 0.5, 0.9):
            got = _is_excitatory_spatial(ids, frac).mean()
            self.assertAlmostEqual(got, frac, delta=0.01, msg=f"frac={frac}")

    def test_pure_function_of_id(self):
        """E/I is a deterministic function of id (every rank agrees on a remote source)."""
        ids = np.array([0, 1, 2, 12345, 99999])
        a = _is_excitatory_spatial(ids, 0.8)
        b = _is_excitatory_spatial(ids, 0.8)
        self.assertTrue(np.array_equal(a, b))

    def test_both_populations_in_a_small_ball(self):
        """Any small spatial neighbourhood contains both E and I (fine interleave)."""
        M, NPP = 27, 1000
        # 27 spatially-contiguous ids (a 3x3x3 intra-tile block near an origin) -> both E and I.
        ids = np.arange(27)
        exc = _is_excitatory_spatial(ids, 0.8)
        self.assertTrue(exc.any() and (~exc).any())


class TestSpatialSmallWorld(unittest.TestCase):
    M = 64          # 4x4x4 -> the 26-neighbour halo (27 tiles) is a strict subset of 64
    NPP = 125       # 5x5x5 tile
    MEANK = 60
    SEED = 11

    def _part(self, rank, **ov):
        kw = dict(somas_per_rank=self.NPP, num_partitions=self.M, partition_rank=rank,
                  mean_in_degree=self.MEANK, kernel_form="gaussian",
                  kernel_width_exc=1.5, kernel_width_inh=0.8, longrange_fraction=0.0,
                  excitatory_weight=0.1, seed=self.SEED)
        kw.update(ov)
        return spatial_smallworld_partition(**kw)

    def test_variable_in_degree(self):
        """In-degree varies neuron-to-neuron (degree heterogeneity / hubs), mean near target."""
        by_post = _sources_by_post(self._part(0, longrange_fraction=0.1))
        deg = np.array([len(v) for v in by_post.values()])
        self.assertEqual(len(by_post), self.NPP)
        self.assertGreater(deg.std(), 0.0, "in-degree should vary")
        # mean is a target subject to kernel feasibility; allow a broad band.
        self.assertGreater(deg.mean(), 0.4 * self.MEANK)
        self.assertLess(deg.mean(), 1.5 * self.MEANK)

    def test_inhibition_is_more_local_than_excitation(self):
        """Local-tier inhibitory sources sit closer than excitatory ones (sigma_I < sigma_E)."""
        part = self._part(20, longrange_fraction=0.0)  # local tier only
        e_d, i_d = [], []
        for s in part["synapses"]:
            if s["pre"] == -1:
                continue
            d = float(_torus_dist(np.array(s["pre"]), np.array(s["post"]), self.M, self.NPP))
            (e_d if _is_excitatory_spatial(np.array([s["pre"]]), 0.8)[0] else i_d).append(d)
        self.assertLess(np.mean(i_d), np.mean(e_d), "inhibition should be more local")

    def test_beta0_is_local_lattice(self):
        """beta=0: sources stay within the 26-neighbour halo (no long-range shortcuts)."""
        a, b, c = _grid_factorization_3d(self.M)
        part = self._part(20, longrange_fraction=0.0)
        halo = set(_select_neighbor_tiles_3d(20, a, b, c)) | {20}
        tiles = {s["pre"] // self.NPP for s in part["synapses"] if s["pre"] >= 0}
        self.assertTrue(tiles <= halo, f"beta=0 leaked outside halo: {tiles - halo}")

    def test_beta_grows_peers_and_adds_shortcuts(self):
        """The small-world signature: beta>0 adds out-of-halo shortcuts and grows the peer count."""
        a, b, c = _grid_factorization_3d(self.M)

        def out_of_halo_and_peers(beta):
            oh = peers = 0
            for r in range(self.M):
                part = self._part(r, longrange_fraction=beta)
                halo = set(_select_neighbor_tiles_3d(r, a, b, c)) | {r}
                tiles = np.array([s["pre"] // self.NPP for s in part["synapses"] if s["pre"] >= 0])
                oh += int(np.sum(~np.isin(tiles, list(halo))))
                peers += len(set(tiles.tolist()) - {r})
            return oh, peers / self.M

        oh0, peers0 = out_of_halo_and_peers(0.0)
        oh1, peers1 = out_of_halo_and_peers(0.1)
        self.assertEqual(oh0, 0)
        self.assertGreater(oh1, 0, "beta>0 should create out-of-halo shortcuts")
        self.assertLessEqual(peers0, 26)
        self.assertGreater(peers1, peers0, "peer count should grow with beta (not weak-scaling)")

    def test_no_autapses(self):
        for s in self._part(3, longrange_fraction=0.2)["synapses"]:
            self.assertNotEqual(s["pre"], s["post"])

    def test_inhibitory_weight_negative(self):
        part = self._part(0, longrange_fraction=0.1)
        for s in part["synapses"]:
            if s["pre"] >= 0 and not _is_excitatory_spatial(np.array([s["pre"]]), 0.8)[0]:
                self.assertLess(s["overrides"]["hyperparameters"]["weight"], 0.0)

    def test_synapse_ids_unique(self):
        part = self._part(7, longrange_fraction=0.15)
        sids = [s["id"] for s in part["synapses"]]
        self.assertEqual(len(sids), len(set(sids)))

    def test_contract_shape(self):
        """Emits the post-owns {somas, synapses, remote_ranks} contract; 1 external synapse/soma."""
        part = self._part(5, longrange_fraction=0.1)
        self.assertEqual(len(part["somas"]), self.NPP)
        self.assertIn("remote_ranks", part)
        n_ext = sum(1 for s in part["synapses"] if s["pre"] == -1)
        self.assertEqual(n_ext, self.NPP)

    def test_single_partition_all_local(self):
        """M=1: no remote_ranks (whole net local); still small-world-shaped internally."""
        part = spatial_smallworld_partition(somas_per_rank=self.NPP, num_partitions=1,
                                            partition_rank=0, mean_in_degree=self.MEANK,
                                            kernel_width_exc=1.5, kernel_width_inh=0.8,
                                            excitatory_weight=0.1, seed=self.SEED)
        self.assertNotIn("remote_ranks", part)

    def test_kernel_forms(self):
        """gaussian / exponential / power-law all build; an unknown form raises."""
        for form in ("gaussian", "exponential", "powerlaw"):
            p = self._part(0, kernel_form=form)
            self.assertEqual(len(p["somas"]), self.NPP)
        with self.assertRaises(ValueError):
            self._part(0, kernel_form="lorentzian")

    def test_longrange_form_uniform_only(self):
        with self.assertRaises(ValueError):
            self._part(0, longrange_fraction=0.1, longrange_form="powerlaw")


if __name__ == "__main__":
    unittest.main()
