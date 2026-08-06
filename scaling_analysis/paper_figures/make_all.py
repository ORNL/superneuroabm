"""Build both paper figures, and refuse to do it quietly if the data is incomplete.

    python make_all.py            # 2 figures -> figs/, as PNG + PDF pairs
    python make_all.py --print    # plus every plotted value as text

Exits non-zero if any expected grid point is missing from the final CSVs, so a figure drawn from
a truncated sweep cannot be mistaken for a finished one. `--print` is the table-view twin of the
figure set: the same numbers, readable without opening a PDF, so a caption can be checked against
the source without trusting a rendering.

This reads `outputs/weak_3d_final.csv` and `outputs/strong_3d_final.csv` only. Regenerate those
with `python analyze_weak.py` / `python analyze_strong.py` after any change to the per-tick data
or the warm-up window. Setup, conventions and reproduction commands are in `README.md`.
"""

import argparse
import sys

import strong_speedup
import weak_efficiency
from _common import read_strong, read_weak

# (tag, module, blurb). Both consume a whole curve set.
WEAK_FIGURES = [
    ("F1", weak_efficiency, "weak: efficiency across the constant-peer plateau"),
]
STRONG_FIGURES = [
    ("F2", strong_speedup, "strong: speedup in wall time"),
]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--print", dest="show", action="store_true",
                   help="dump every plotted value as text")
    args = p.parse_args(argv)

    weak, weak_missing = read_weak()
    strong, strong_missing = read_strong()

    written = []
    for data, figures in ((weak, WEAK_FIGURES), (strong, STRONG_FIGURES)):
        for tag, mod, blurb in figures:
            print(f"\n{tag} {mod.STEM} -- {blurb}")
            if args.show:
                mod.report(data)
            written += mod.plot(data, args.dpi)

    print(f"\nWrote {len(written)} files:")
    for path in written:
        print(f"  {path}")

    if weak_missing or strong_missing:
        if weak_missing:
            print(f"\nINCOMPLETE -- {len(weak_missing)} missing weak (K, w) point(s): "
                  f"{weak_missing}", file=sys.stderr)
        if strong_missing:
            print(f"INCOMPLETE -- missing strong w: {strong_missing}", file=sys.stderr)
        return 1
    print("\nCoverage: weak 36/36, strong 8/8 -- complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
