"""F1 -- weak-scaling parallel efficiency across the constant-peer regime, 64 -> 2048 GPUs.

Carries the sustained per-tick rate as legend text, so the absolute number a reader would quote
sits on the same panel as the scaling claim -- the role GGap's `weak_scaling_efficiency.py` gives
its corner pill. The figure set is two panels; everything the deleted companions carried --
the pre-saturation ramp point by point, the construction/step split, the setup-amortisation
horizon -- is in `README.md`.

WHY THE FIGURE STARTS AT 64, AND WHY THAT IS ALSO THE BASELINE
    A rank's Moore stencil always has 26 = 3^3-1 directions, but on a periodic torus several of
    them can land on the SAME rank: if the grid is only two ranks wide along an axis, +x and -x
    wrap around to the same neighbour. So the number of DISTINCT peers a rank talks to climbs
    with the grid -- 0, 1, 3, 7, 11, 17, 26 -- and only reaches 26 once every axis holds at least
    four ranks, which first happens at 64 workers. (A 1-GPU run has no peers at all.)

    Below 64, then, a run measures a CHEAPER COMMUNICATION PATTERN, not a faster code, which is
    why it is neither the baseline nor on the panel: normalising there charges the plateau for
    the one-time cost of switching communication on, and drawing the ramp beside the plateau puts
    the only steep thing on the figure in the regime that is not the claim -- a reader's eye lands
    there and extrapolates a decline that does not exist. Against T(64) the plateau is a 32x span
    at a FIXED communication topology, which is the only span over which weak scaling means
    anything here. This is the published convention for point-to-point codes: WOMBAT (Mendygral
    et al. 2017, ApJS 228:23) reports off-node communication saturating between 3 and 27 nodes for
    the same geometric reason, with update times "nearly flat for larger node counts" past it.

    What the convention does not permit is dropping the ramp silently. It is not dropped: it is
    tabulated point by point in `README.md` (peers 0, 1, 3, 7, 11, 17 -> 26 with the step time at
    each), `--print` here dumps every point against both baselines, and both numbers go in the
    caption -- the T(1) figure alone invites the reader to extrapolate the ramp, the T(64) figure
    alone hides what communication costs.

WHY THE CURVE IS THE STEP AND NOT simulate()
    GGap's `_scaling_common.py:derive_metrics` builds its efficiency from
    `simulation_time = first_tick + steady_state`, defending it as more honest than a bare
    steady-state tick. That is right at GGap's ratio -- their first tick is 55-58% of simulate(),
    a genuine blend. Ours is 99.7-99.9%, so the same curve would be construction with a rounding
    error of simulation: non-monotonic, no knee at w=64, no plateau, and its worst excursions are
    construction run-to-run variance rather than a scaling signal. Publishing that shape as weak
    scaling is precisely what invalidated the previous campaign. The construction/step split it
    comes from, and the run length at which it would stop dominating, are in `README.md`.
"""

import argparse
import statistics as st
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend choice)

from _common import (IN_DEGREES, MOORE_PEERS, PLATEAU_START, read_weak,  # noqa: E402
                     save_figure, setup_log2_xaxis, step_efficiency)
from _style import (COLOR_IDEAL, FIG_H, F_ANNOT, INK_MUTED, INK_SECONDARY,  # noqa: E402
                    LW_IDEAL, MARK, SERIES, apply_rcparams)

STEM = "weak_efficiency"

# The plateau alone, so the axis can zoom on it. Truncated at 90 rather than run to zero: from
# zero the three curves collapse into one stripe and the K=4000 w=2048 dip is invisible.
YLIM = (90, 105)


def plateau(curve):
    """The constant-peer points -- w=64 is both the first of them and the efficiency baseline."""
    return [r for r in curve if r["workers"] >= PLATEAU_START]


def sustained_ms(curve):
    """The per-tick rate a reader would quote: the median step across the constant-peer plateau.

    The plateau, not the whole sweep. GGap averages over every point of its sweep, but every one
    of those points has the same communication pattern; ours do not, and folding ramp points that
    talk to 0-17 peers into the average would report a rate the machine never sustains at scale.
    """
    return st.median([r["step_s"] * 1e3 for r in plateau(curve)])


def plot(curves, dpi):
    apply_rcparams()
    fig, ax = plt.subplots(figsize=(3.4, FIG_H + 0.15))
    w = [r["workers"] for r in plateau(curves[IN_DEGREES[0]])]

    ax.axhline(100, color=COLOR_IDEAL, ls="--", lw=LW_IDEAL, label="ideal", zorder=2)

    for color, K in zip(SERIES, IN_DEGREES):
        curve = curves.get(K)
        if not curve:
            continue
        pl = plateau(curve)
        eff = step_efficiency(pl, PLATEAU_START)
        # Sustained rate and end efficiency ride in the legend label rather than in their own
        # text block: they belong beside the swatch that carries them, the panel already holds a
        # note, and putting the percentage here frees the right margin of end labels.
        ax.plot([r["workers"] for r in pl], eff, "o-", color=color, zorder=4,
                label=f"K={K}: ~{sustained_ms(curve):.1f} ms/tick, {eff[-1]:.0f}%", **MARK)

    setup_log2_xaxis(ax, w)
    ax.set_ylabel("Parallel efficiency (%)", color=INK_SECONDARY)
    ax.set_ylim(*YLIM)
    ax.set_yticks([90, 95, 100, 105])
    # Without the ramp on the panel, "100% at w=64" would be an unexplained choice of origin, and
    # nothing else states what is held constant. Scale figures come from the CSV, not from
    # arithmetic here -- this is a presentation layer and a second derivation would drift.
    ref = plateau(curves[IN_DEGREES[0]])
    ax.annotate(
        f"K = in-degree (incoming synapses per neuron)\n"
        f"weak scaling: {int(ref[0]['neurons_per_worker']):,} neurons/GPU "
        f"({ref[-1]['total_neurons'] / 1e6:.1f} M at w={int(ref[-1]['workers'])})\n"
        f"baseline w={PLATEAU_START} — the smallest grid with 4+ ranks along\n"
        f"every axis, so a rank's {MOORE_PEERS} = 3³-1 stencil directions each\n"
        f"land on a different rank: {MOORE_PEERS} distinct MPI peers",
        xy=(PLATEAU_START, 0.40), xycoords=("data", "axes fraction"),
        xytext=(0, 0), textcoords="offset points", ha="left", va="top",
        fontsize=F_ANNOT, color=INK_MUTED, linespacing=1.4, zorder=5)

    # Above the curves: with the axis starting at 90 the top of the panel is the empty corner.
    ax.legend(loc="upper left", frameon=False, labelcolor=INK_SECONDARY, ncol=2,
              fontsize=F_ANNOT, handletextpad=0.4, borderaxespad=0.2, handlelength=1.3,
              columnspacing=0.8)

    fig.tight_layout(pad=0.2)
    written = save_figure(fig, STEM, dpi)
    plt.close(fig)
    return written


def report(curves):
    """Every value as text -- the table-view twin of the figure."""
    print(f"\n{STEM}: parallel efficiency vs T({PLATEAU_START}); plotted range {PLATEAU_START}..2048")
    for K in IN_DEGREES:
        curve = curves.get(K)
        if not curve:
            continue
        e1 = step_efficiency(curve)
        e64 = step_efficiency(curve, PLATEAU_START)
        print(f"  K={K}: ~{sustained_ms(curve):.3f} ms/tick sustained (plateau median)")
        print(f"    {'w':>6} {'peers':>6} {'step(ms)':>9} {'E vs T(64)':>11} {'E vs T(1)':>10}")
        for r, a, b in zip(curve, e1, e64):
            mark = " " if r["workers"] >= PLATEAU_START else "."  # '.' = pre-saturation ramp
            print(f"  {mark} {int(r['workers']):>6} {int(r['peers_mean']):>6} "
                  f"{r['step_s'] * 1e3:>9.3f} {b:>10.1f}% {a:>9.1f}%")
        pl = [r["step_s"] for r in plateau(curve)]
        pe = [b for r, b in zip(curve, e64) if r["workers"] >= PLATEAU_START]
        print(f"    plateau: step spread {(max(pl) - min(pl)) / min(pl) * 100:.2f}%, "
              f"E vs T(64) {min(pe):.1f}-{max(pe):.1f}%, "
              f"E vs T(1) {min(a for r, a in zip(curve, e1) if r['workers'] >= PLATEAU_START):.1f}"
              f"-{max(a for r, a in zip(curve, e1) if r['workers'] >= PLATEAU_START):.1f}%")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--print", dest="show", action="store_true", help="dump plotted values")
    args = p.parse_args(argv)

    curves, missing = read_weak()
    if missing:
        print(f"INCOMPLETE -- missing weak grid points: {missing}", file=sys.stderr)
    if args.show:
        report(curves)
    for path in plot(curves, args.dpi):
        print(f"  {path}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
