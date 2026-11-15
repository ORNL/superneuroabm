"""Write a Brunel network as per-rank partition files for run_partitioned_sna.py and
run_partitioned_nestgpu.py.

The network is generated once as a single-rank partition (superneuroabm.brunel.brunel_partition,
global fixed in-degree), then split by post soma into N post-owned parts
(per_tick_probes/split_partition.py), so it is the same network at every rank count. The
defaults are NEST's brunel_alpha_nest.py at order 2500: 12,500 neurons (80 % excitatory),
in-degree 1,000 excitatory + 250 inhibitory, with the alpha neuron's weights from
tutorials/customized_configs/brunel_alpha_config.yaml (inhibitory = -5 x excitatory) and one
external input synapse per soma.

    python make_brunel_partitions.py partitions/brunel_alpha_12500 --split 2 4

writes <root>/np1/partition_0.npz and <root>/np<N>/partition_<r>.npz.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

import partitioned_common as pc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "per_tick_probes"))
from split_partition import split                                   # noqa: E402
from superneuroabm.brunel import brunel_partition                   # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root")
    ap.add_argument("--neurons", type=int, default=12500)
    ap.add_argument("--excitatory-in-degree", type=int, default=1000)
    ap.add_argument("--inhibitory-in-degree", type=int, default=250)
    ap.add_argument("--g", type=float, default=5.0, help="inhibitory / excitatory weight")
    ap.add_argument("--seed", type=int, default=42, help="connectivity seed")
    ap.add_argument("--split", type=int, nargs="*", default=[], help="rank counts to split for")
    args = ap.parse_args()

    par = pc.alpha_parameters()
    w = par["J_E"] * 1e-12                                       # pA -> A, the YAML's units
    root = Path(args.root)
    (root / "np1").mkdir(parents=True, exist_ok=True)
    net = brunel_partition(
        somas_per_rank=args.neurons, num_partitions=1, partition_rank=0,
        excitatory_in_degree=args.excitatory_in_degree,
        inhibitory_in_degree=args.inhibitory_in_degree, topology="global",
        excitatory_weight=w, inhibitory_weight_ratio=args.g, external_weight=w,
        soma_breed=pc.SOMA_BREED, soma_config=pc.SOMA_CONFIG,
        synapse_breed=pc.SYNAPSE_BREED, synapse_config=pc.SYNAPSE_CONFIG,
        seed=args.seed, output_format="columns")
    src = root / "np1" / "partition_0.npz"
    np.savez(src, **net)
    pre = np.asarray(net["pre"])
    print(f"{src}: {np.asarray(net['soma_ids']).size:,} somas, {(pre >= 0).sum():,} recurrent "
          f"+ {(pre < 0).sum():,} external synapses")
    if args.split:
        split(str(src), str(root), args.split)


if __name__ == "__main__":
    main()
