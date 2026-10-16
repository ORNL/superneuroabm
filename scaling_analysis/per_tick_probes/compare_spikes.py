"""Per-spike comparison of the spikes_np<N>.npy files multirank_tick.py writes: which rank
counts agree, and at which tick the first difference appears, so a late divergence
(float accumulation order, amplified by a chaotic network) can be told from a systematic
timing shift (a one-tick delay shows at the first spikes).

usage: compare_spikes.py <split_root> <N> [<N> ...]
"""
import sys
from itertools import combinations

import numpy as np


def main(root, ns):
    sets = {}
    for n in ns:
        a = np.load(f"{root}/spikes_np{n}.npy")
        sets[n] = set(map(tuple, a.astype(np.int64).tolist()))
        print(f"N={n}: {len(sets[n]):,} spikes, first tick {int(a[:, 1].min()) if a.size else None}, "
              f"last {int(a[:, 1].max()) if a.size else None}")
    for a, b in combinations(ns, 2):
        only_a, only_b = sets[a] - sets[b], sets[b] - sets[a]
        first = min([t for _, t in only_a] + [t for _, t in only_b], default=None)
        common_before = sum(1 for (_, t) in sets[a] if first is None or t < first)
        print(f"N={a} vs N={b}: identical={not only_a and not only_b}  only_in_{a}={len(only_a):,} "
              f"only_in_{b}={len(only_b):,}  first_differing_tick={first}  spikes_before_that={common_before:,}")


if __name__ == "__main__":
    main(sys.argv[1], [int(x) for x in sys.argv[2:]])
