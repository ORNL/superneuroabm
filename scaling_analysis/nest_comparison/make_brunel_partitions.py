"""Write a Brunel network as per-rank partition files for run_partitioned_sna.py and
run_partitioned_nestgpu.py.

Two ways, both through superneuroabm.brunel.brunel_partition (fixed in-degree; sources from
the whole network, or with --topology torus3d from a ball around each soma on a 3D torus):

  * one network at every rank count (the default): generated once as a single-rank partition,
    then split by post soma into N post-owned parts (per_tick_probes/split_partition.py). The
    defaults are NEST's brunel_alpha_nest.py at order 2500: 12,500 neurons (80 % excitatory),
    in-degree 1,000 excitatory + 250 inhibitory.

        python make_brunel_partitions.py partitions/brunel_alpha_12500 --split 2 4

  * a network that grows with the rank count (--ranks): --neurons on EVERY rank, each rank's
    part drawn on its own, nothing global ever held. This is how NEST GPU's hpc_benchmark
    scales (11,250 neurons per process, in-degree 9,000 + 2,250 drawn from the whole network).
    Under MPI the tasks share the parts of each rank count between them.

        srun -n 8 python make_brunel_partitions.py partitions/hpc --ranks 1 2 4 8 \\
            --neurons 11250 --excitatory-in-degree 9000 --inhibitory-in-degree 2250 \\
            --config hpc_alpha_config.yaml

    The same size with only the structure changed, the 3D torus of the weak-scaling runs
    (docs/BRUNEL_SCALING.md 5.5a): every rank is one 18 x 25 x 25 tile of neurons and a soma
    draws its 11,250 sources, without repeats, from the neurons within --connection-radius of
    it, so a rank listens to its neighbour tiles only (26 of them from 64 ranks on). The
    radius must stay below the smallest grid dimension (18 on one rank). With --exact-ei the
    soma draws its 9,000 excitatory and 2,250 inhibitory sources apart, from the neurons of
    each kind in the ball, as the whole-network draw does: brunel_partition's own torus3d
    draws one pool, a rank's inhibitory neurons are one slab of its tile, and an excitatory
    soma then gets up to 44 % more inhibitory sources than an inhibitory one, which silences
    the excitatory neurons at these weights (measured 2026-11-15: 0.4 Hz, against 8 to 10 Hz).

        srun -n 8 python make_brunel_partitions.py partitions/torus --ranks 1 2 4 8 \\
            --neurons 11250 --excitatory-in-degree 9000 --inhibitory-in-degree 2250 \\
            --config hpc_alpha_config.yaml --topology torus3d --connection-radius 17 --exact-ei

    Strong scaling on the torus (--drawn-as): the network is drawn once, as --drawn-as ranks
    of --neurons each, and a rank of --ranks holds a brick of those tiles. Every soma, synapse
    and weight is then the same at every rank count (the drive and the starting potentials
    follow the soma id), so the spikes should be too. 92,160 neurons, 1.04 G synapses, on 8
    to 256 ranks:

        srun -n 8 python make_brunel_partitions.py partitions/strong --ranks 8 \\
            --neurons 180 --drawn-as 512 --excitatory-in-degree 9000 --inhibitory-in-degree 2250 \\
            --config hpc_alpha_config.yaml --topology torus3d --connection-radius 17 \\
            --exact-ei --interleave-ei

The weights are the alpha neuron's, from --config (inhibitory = -g x excitatory), with one
external input synapse per soma. Both write <root>/np<N>/partition_<r>.npz.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import partitioned_common as pc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "per_tick_probes"))
from split_partition import split                                   # noqa: E402
from superneuroabm import brunel                                    # noqa: E402
from superneuroabm.brunel import brunel_partition                   # noqa: E402


def exact_ei_ball_draw(c_e, c_i, interleave, excitatory_fraction=0.8, chunk=1024):
    """A stand-in for brunel._draw_sources_radius (same arguments, same (somas, in-degree)
    result, same grid of rank tiles and the same ball) that gives every soma exactly c_e
    excitatory sources followed by c_i inhibitory ones, each drawn without repeats among the
    ball's neurons of that kind.

    interleave: where a rank's inhibitory neurons sit in its tile. brunel_partition's ids put
    them last in the rank's block, which its positions turn into one slab of the tile; with
    interleave they are spread evenly instead (every fifth position at 20 %), so that every
    ball holds both kinds in the network's proportion. The ids, and with them which neuron is
    inhibitory, stay brunel_partition's."""

    def draw(rng, rank, ranks, npp, in_degree, radius, allow_autapses):
        assert in_degree == c_e + c_i
        a, b, c = brunel._grid_factorization_3d(ranks)
        gx, gy, gz = brunel._factor_near_cube(npp)
        A, B, C = a * gx, b * gy, c * gz
        if radius >= min(A, B, C):                  # also what keeps a soma out of its own ball
            raise ValueError(f"radius {radius} must be below the smallest grid dimension {min(A, B, C)}")
        exc_per_rank = round(excitatory_fraction * npp)             # brunel_partition's rule
        q = np.arange(npp, dtype=np.int64)          # positions in the tile, x-major
        if interleave:
            n_inh = npp - exc_per_rank
            inh_at = ((q + 1) * n_inh) // npp > (q * n_inh) // npp
            position_of = np.concatenate([q[~inh_at], q[inh_at]])   # of a rank's k-th soma
        else:
            position_of = q
        soma_at = np.empty(npp, dtype=np.int64)
        soma_at[position_of] = q
        ix, rest = np.divmod(position_of, gy * gz)
        iy, iz = np.divmod(rest, gz)
        tx, rest = divmod(rank, b * c)
        ty, tz = divmod(rest, c)
        X, Y, Z = tx * gx + ix, ty * gy + iy, tz * gz + iz           # this rank's somas
        off = brunel._ball_offsets(radius)
        pre = np.empty((npp, in_degree), dtype=np.int64)
        for lo in range(0, npp, chunk):
            hi = min(lo + chunk, npp)
            cx, jx = np.divmod((X[lo:hi, None] + off[None, :, 0]) % A, gx)
            cy, jy = np.divmod((Y[lo:hi, None] + off[None, :, 1]) % B, gy)
            cz, jz = np.divmod((Z[lo:hi, None] + off[None, :, 2]) % C, gz)
            k = soma_at[jx * (gy * gz) + jy * gz + jz]
            cand = (cx * (b * c) + cy * c + cz) * npp + k
            is_exc = k < exc_per_rank
            del cx, cy, cz, jx, jy, jz, k
            keys = rng.random(cand.shape)
            for first, n, kind in ((0, c_e, is_exc), (c_e, c_i, ~is_exc)):
                if kind.sum(axis=1).min() < n:
                    raise ValueError(f"a ball of radius {radius} holds fewer than {n} neurons of "
                                     "one kind for some soma: use a larger --connection-radius")
                chosen = np.argpartition(np.where(kind, keys, 2.0), n - 1, axis=1)[:, :n]
                pre[lo:hi, first:first + n] = np.take_along_axis(cand, chosen, axis=1)
        return pre

    return draw


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root")
    ap.add_argument("--neurons", type=int, default=12500,
                    help="neurons in the network, or on each rank with --ranks")
    ap.add_argument("--excitatory-in-degree", type=int, default=1000)
    ap.add_argument("--inhibitory-in-degree", type=int, default=250)
    ap.add_argument("--g", type=float, default=5.0, help="inhibitory / excitatory weight")
    ap.add_argument("--seed", type=int, default=42, help="connectivity seed")
    ap.add_argument("--config", default=str(pc.ALPHA_YAML), help="the alpha neuron's YAML")
    ap.add_argument("--topology", choices=("global", "torus3d"), default="global",
                    help="where a soma's sources come from: the whole network, or a ball "
                         "around it on a 3D torus of rank tiles")
    ap.add_argument("--connection-radius", type=float, default=None,
                    help="torus3d: the ball's radius in neurons (default: the smallest ball "
                         "holding twice the in-degree)")
    ap.add_argument("--exact-ei", action="store_true",
                    help="torus3d: exactly --excitatory-in-degree and --inhibitory-in-degree "
                         "sources per soma, each from its own kind in the ball")
    ap.add_argument("--interleave-ei", action="store_true",
                    help="with --exact-ei: a rank's inhibitory neurons spread evenly over its "
                         "tile, not in one slab of it")
    ap.add_argument("--drawn-as", type=int, default=None, metavar="RANKS",
                    help="torus3d with --ranks: the network is that of this many ranks of "
                         "--neurons each, whatever --ranks; a rank holds a brick of their tiles")
    ap.add_argument("--split", type=int, nargs="*", default=[], help="rank counts to split for")
    ap.add_argument("--ranks", type=int, nargs="*", default=[],
                    help="rank counts of a network with --neurons on every rank")
    args = ap.parse_args()

    par = pc.alpha_parameters(args.config)
    w = par["J_E"] * 1e-12                                       # pA -> A, the YAML's units
    if args.exact_ei:
        if args.topology != "torus3d":
            ap.error("--exact-ei goes with --topology torus3d (the global draw is already exact)")
        brunel._draw_sources_radius = exact_ei_ball_draw(
            args.excitatory_in_degree, args.inhibitory_in_degree, args.interleave_ei)
    elif args.interleave_ei:
        ap.error("--interleave-ei goes with --exact-ei")
    root = Path(args.root)

    def part(ranks, rank):
        return brunel_partition(
            somas_per_rank=args.neurons, num_partitions=ranks, partition_rank=rank,
            excitatory_in_degree=args.excitatory_in_degree,
            inhibitory_in_degree=args.inhibitory_in_degree, topology=args.topology,
            connection_radius=args.connection_radius,
            excitatory_weight=w, inhibitory_weight_ratio=args.g, external_weight=w,
            soma_breed=pc.SOMA_BREED, soma_config=pc.SOMA_CONFIG,
            synapse_breed=pc.SYNAPSE_BREED, synapse_config=pc.SYNAPSE_CONFIG,
            seed=args.seed, output_format="columns")

    def brick_part(ranks, rank):
        """Rank `rank` of `ranks` of the network drawn as --drawn-as ranks: the tiles of the
        brick it holds, merged into one part. The grid of `ranks` must divide the tiles' grid."""
        a, b, c = brunel._grid_factorization_3d(args.drawn_as)
        A, B, C = brunel._grid_factorization_3d(ranks)
        if a % A or b % B or c % C:
            raise ValueError(f"{ranks} ranks ({A}x{B}x{C}) do not divide the {a}x{b}x{c} tiles "
                             f"of --drawn-as {args.drawn_as}")
        tx, rest = np.divmod(np.arange(args.drawn_as), b * c)
        ty, tz = np.divmod(rest, c)
        owner = (tx // (a // A)) * (B * C) + (ty // (b // B)) * C + tz // (c // C)
        tiles = [part(args.drawn_as, int(t)) for t in np.flatnonzero(owner == rank)]
        net = dict(tiles[0])
        for key in ("soma_ids", "synapse_ids", "pre", "post", "syn_hp_vals"):
            net[key] = np.concatenate([np.asarray(t[key]) for t in tiles])
        pre = net["pre"]
        remote = np.unique(pre[(pre >= 0) & (owner[np.maximum(pre, 0) // args.neurons] != rank)])
        net["remote_ids"] = remote.astype(np.int64)
        net["remote_rank_of"] = owner[remote // args.neurons].astype(np.int64)
        return net

    if args.drawn_as and not (args.ranks and args.topology == "torus3d"):
        ap.error("--drawn-as goes with --ranks and --topology torus3d")

    if args.ranks:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        todo = [(n, r) for n in args.ranks for r in range(n)][comm.Get_rank()::comm.Get_size()]
        for n, r in todo:
            out = root / f"np{n}" / f"partition_{r}.npz"
            if out.exists():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            net = brick_part(n, r) if args.drawn_as else part(n, r)
            tmp = out.with_suffix(".tmp.npz")
            np.savez(tmp, **net)
            tmp.rename(out)                       # a part is either whole or absent
            ids = np.sort(np.asarray(net["soma_ids"]))
            pre, n = np.asarray(net["pre"]), ids.size
            own = ids[np.minimum(np.searchsorted(ids, pre), n - 1)] == pre
            inh = np.asarray(net["syn_hp_vals"])[:, 0] < 0
            inh = np.bincount(np.searchsorted(ids, np.asarray(net["post"])[inh]), minlength=n)
            print(f"{out}: {n:,} somas, {pre.size:,} synapses ({100 * own.sum() / (pre >= 0).sum():.1f} % "
                  f"from its own rank; inhibitory per soma {inh.min():,} to {inh.max():,}), "
                  f"{np.asarray(net['remote_ids']).size:,} remote somas on "
                  f"{np.unique(np.asarray(net['remote_rank_of'])).size} ranks, "
                  f"{time.time() - t0:.1f} s", flush=True)
            del pre, own, inh, ids
            del net
        return

    (root / "np1").mkdir(parents=True, exist_ok=True)
    net = part(1, 0)
    src = root / "np1" / "partition_0.npz"
    np.savez(src, **net)
    pre = np.asarray(net["pre"])
    print(f"{src}: {np.asarray(net['soma_ids']).size:,} somas, {(pre >= 0).sum():,} recurrent "
          f"+ {(pre < 0).sum():,} external synapses")
    if args.split:
        split(str(src), str(root), args.split)


if __name__ == "__main__":
    main()
