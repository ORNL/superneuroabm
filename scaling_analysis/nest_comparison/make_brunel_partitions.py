"""Write a Brunel network as per-rank partition files for run_partitioned_sna.py and
run_partitioned_nestgpu.py.

Two ways, both through superneuroabm.brunel.brunel_partition (global fixed in-degree):

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
from superneuroabm.brunel import brunel_partition                   # noqa: E402


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
    ap.add_argument("--split", type=int, nargs="*", default=[], help="rank counts to split for")
    ap.add_argument("--ranks", type=int, nargs="*", default=[],
                    help="rank counts of a network with --neurons on every rank")
    args = ap.parse_args()

    par = pc.alpha_parameters(args.config)
    w = par["J_E"] * 1e-12                                       # pA -> A, the YAML's units
    root = Path(args.root)

    def part(ranks, rank):
        return brunel_partition(
            somas_per_rank=args.neurons, num_partitions=ranks, partition_rank=rank,
            excitatory_in_degree=args.excitatory_in_degree,
            inhibitory_in_degree=args.inhibitory_in_degree, topology="global",
            excitatory_weight=w, inhibitory_weight_ratio=args.g, external_weight=w,
            soma_breed=pc.SOMA_BREED, soma_config=pc.SOMA_CONFIG,
            synapse_breed=pc.SYNAPSE_BREED, synapse_config=pc.SYNAPSE_CONFIG,
            seed=args.seed, output_format="columns")

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
            net = part(n, r)
            tmp = out.with_suffix(".tmp.npz")
            np.savez(tmp, **net)
            tmp.rename(out)                       # a part is either whole or absent
            print(f"{out}: {np.asarray(net['soma_ids']).size:,} somas, "
                  f"{np.asarray(net['pre']).size:,} synapses, {time.time() - t0:.1f} s", flush=True)
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
