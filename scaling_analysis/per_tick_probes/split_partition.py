"""Split a single-rank columnar Brunel partition into N post-owned partitions.

Rank r owns a contiguous block of soma ids (the convention of superneuroabm.brunel); a
synapse goes with its post; remote_ids / remote_rank_of list the pre-somas owned
elsewhere. Synapse ids and every value stay as in the single-rank file, so the network
is IDENTICAL at every N: the same spikes are expected at every rank count (up to the
float accumulation order, see docs/EVENT_DRIVEN_NOTES.md section 16).

usage: split_partition.py <partition_0.npz> <out_root> <N> [<N> ...]
writes <out_root>/np<N>/partition_<r>.npz
"""
import sys
from pathlib import Path

import numpy as np


def split(src, out_root, ns):
    with np.load(src, allow_pickle=True) as z:
        d = {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files}
    soma_ids = np.asarray(d["soma_ids"])
    n_somas = soma_ids.size
    assert np.array_equal(soma_ids, np.arange(n_somas)), "expects soma ids 0..n-1"
    pre, post = np.asarray(d["pre"]), np.asarray(d["post"])
    scalars = {k: v for k, v in d.items() if not isinstance(v, np.ndarray) or v.ndim == 0}
    out_root = Path(out_root)
    for N in ns:
        block = -(-n_somas // N)
        owner = np.minimum(np.arange(n_somas) // block, N - 1)
        out = out_root / f"np{N}"
        out.mkdir(parents=True, exist_ok=True)
        for r in range(N):
            mine = owner[post] == r
            local_somas = soma_ids[owner == r]
            p = pre[mine]
            remote = np.unique(p[(p >= 0) & (owner[np.maximum(p, 0)] != r)])
            np.savez(out / f"partition_{r}.npz",
                     **scalars,
                     soma_ids=local_somas,
                     synapse_ids=np.asarray(d["synapse_ids"])[mine],
                     pre=p, post=post[mine],
                     syn_hp_keys=np.asarray(d["syn_hp_keys"]),
                     syn_hp_vals=np.asarray(d["syn_hp_vals"])[mine],
                     remote_ids=remote.astype(np.int64),
                     remote_rank_of=owner[remote].astype(np.int64))
            print(f"N={N} rank {r}: somas {local_somas.size:,} synapses {int(mine.sum()):,} "
                  f"remote pre-somas {remote.size:,}", flush=True)


if __name__ == "__main__":
    split(sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3:]])
