"""Spike-by-spike comparison of run_partitioned_*.py outputs, every pair of the files given.

For each pair: the spike counts, the firing rate after the burn-in, the first tick at which the
two differ, and the share of identical (soma, tick) spikes per 150 ms window. A one-tick shift
at the very first spikes is a timing error; a late, growing divergence is float rounding
amplified by the network's chaos.

usage: compare_partitioned_spikes.py a.npz b.npz [c.npz ...] [--burn-in-ms 500]
"""

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np

DT_MS = 0.1


def load(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--burn-in-ms", type=float, default=500.0)
    ap.add_argument("--window-ms", type=float, default=150.0)
    args = ap.parse_args()
    runs = {Path(f).stem: load(f) for f in args.files}
    burn = int(round(args.burn_in_ms / DT_MS))
    width = int(round(args.window_ms / DT_MS))
    for name, r in runs.items():
        n, steps = int(r["n_somas"]), int(r["steps"])
        late = r["tick"] >= burn
        print(f"{name:28s} {r['tick'].size:9,} spikes   rate {late.sum() / n / ((steps - burn) * DT_MS / 1000):8.3f} Hz")
    def keys(r, n):
        """A run's spikes as sorted integers, tick * n + soma. save_spikes orders by tick then
        soma, so this is already sorted and 10^8 spikes compare in seconds."""
        k = r["tick"].astype(np.int64) * n + r["soma"]
        return k if np.all(k[1:] > k[:-1]) else np.unique(k)

    def missing_from(x, y):
        """Which entries of sorted x are not in sorted y."""
        at = np.minimum(np.searchsorted(y, x), max(y.size - 1, 0))
        return y[at] != x if y.size else np.ones(x.size, dtype=bool)

    for (na, a), (nb, b) in combinations(runs.items(), 2):
        n = max(int(a["n_somas"]), int(b["n_somas"]), int(a["soma"].max(initial=0)) + 1,
                int(b["soma"].max(initial=0)) + 1)
        A, B = keys(a, n), keys(b, n)
        a_only, b_only = missing_from(A, B), missing_from(B, A)
        print(f"\n{na} vs {nb}:")
        if not a_only.any() and not b_only.any():
            print("  identical over the whole run")
            continue
        early = [[(int(k // n), int(k % n)) for k in x[only][:3]] for x, only in ((A, a_only), (B, b_only))]
        first = min(e[0][0] for e in early if e)
        print(f"  first difference at tick {first} ({first * DT_MS:.1f} ms); "
              f"only in first: {early[0]}  only in second: {early[1]}  (tick, soma)")
        edges = np.arange(0, int(a["steps"]) + width, width)
        both = np.histogram(A[~a_only] // n, edges)[0]
        either = np.histogram(A // n, edges)[0] + np.histogram(B[b_only] // n, edges)[0]
        shares = " ".join(f"{100 * x / max(y, 1):.1f}" for x, y in zip(both, either))
        print(f"  % identical per {args.window_ms:.0f} ms window: {shares}")
        counts = [f"{c:,}" for pair in zip(np.histogram(a["tick"], edges)[0],
                                           np.histogram(b["tick"], edges)[0]) for c in pair]
        print(f"  spikes per window (first, second): {' '.join(counts)}")


if __name__ == "__main__":
    main()
