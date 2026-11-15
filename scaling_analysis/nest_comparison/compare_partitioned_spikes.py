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
    args = ap.parse_args()
    runs = {Path(f).stem: load(f) for f in args.files}
    burn = int(round(args.burn_in_ms / DT_MS))
    for name, r in runs.items():
        n, steps = int(r["n_somas"]), int(r["steps"])
        late = r["tick"] >= burn
        print(f"{name:28s} {r['tick'].size:9,} spikes   rate {late.sum() / n / ((steps - burn) * DT_MS / 1000):8.3f} Hz")
    for (na, a), (nb, b) in combinations(runs.items(), 2):
        A = set(zip(a["soma"].tolist(), a["tick"].tolist()))
        B = set(zip(b["soma"].tolist(), b["tick"].tolist()))
        diff = A ^ B
        print(f"\n{na} vs {nb}:")
        if not diff:
            print("  identical over the whole run")
            continue
        first = min(t for _, t in diff)
        print(f"  first difference at tick {first} ({first * DT_MS:.1f} ms); "
              f"only in first: {sorted((t, s) for s, t in A - B)[:3]}  "
              f"only in second: {sorted((t, s) for s, t in B - A)[:3]}  (tick, soma)")
        steps = int(a["steps"])
        windows = []
        for lo in range(0, steps, 1500):
            a_w = {p for p in A if lo <= p[1] < lo + 1500}
            b_w = {p for p in B if lo <= p[1] < lo + 1500}
            windows.append(f"{100 * len(a_w & b_w) / max(len(a_w | b_w), 1):.1f}")
        print(f"  % identical per 150 ms window: {' '.join(windows)}")


if __name__ == "__main__":
    main()
