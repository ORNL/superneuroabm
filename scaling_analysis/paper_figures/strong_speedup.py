"""F2 -- strong scaling: fixed 204,800-neuron problem, 16 -> 2048 GPUs.

Structural mirror of `GGap/SC2026/figures/scripts/strong_scaling_speedup.py`: one measured curve
against an ideal reference.

THE METRIC IS WALL TIME -- `total_time`, the whole run
    Generation, model creation, GPU setup, construction and every tick. It is what a user waits
    for, and it is the textbook strong-scaling metric: Amdahl's law is defined on total runtime,
    and the question a fixed-size sweep asks is "more processors, how much sooner is the answer?"
    It reaches 13.9x on 64x the GPUs.

    Note what this does NOT say. Construction is ~99.8% of a 100-tick run (see `README.md`), and construction
    is what parallelises here; the steady-state step peaks at 1.44x at w=256 and then reverses to
    0.76x at w=2048, because per-tick GPU compute is pinned near 0.370 ms across a 128x span of
    per-rank work -- the kernel is launch-bound, so shrinking the tile only grows the halo. That
    turn lands exactly where the peer count leaves 26 (w=512), which the right-hand axis shows.
    So 13.9x is a wall-time result,
    not a claim that the simulation step scales; the step numbers stay in `--print` and in the
    caption rather than on the panel.

    READ THE PEER AXIS AS CONTEXT, NOT CAUSE. Construction stops halving at the same w=512 --
    1.79x, then 1.19x, 1.06x, 1.05x -- so the two move together, but that is equally well explained
    by construction asymptoting to a fixed ~15 s floor (plain Amdahl) and this sweep cannot
    separate the two. The axis is here because the communication topology changing is worth
    knowing, not because it is established as the reason the curve flattens.

    The peer count is deterministic geometry rather than a measurement: rank grid x tile shape x
    connection radius reproduces it exactly through w=1024. At w=2048 the measured mean is 95.55
    against 104 reachable tiles -- the 4x5x5 tile is small enough that a rank's K=1000 draws miss
    ~8 of the tiles it could reach, which is also why it is the only non-integer value.

BASELINE
    w=16, not 1 GPU. The fixed problem needs ~205 M synapses, which does not fit on one GCD, so
    a 1-GPU baseline never ran and quoting speedup against it would be fiction. w=16 itself sits
    in the ramp at 11 peers -- it is not a clean serial reference, and the axis label says so.
"""

import argparse
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend choice)

from _common import (BREAKDOWN_W, label_ends, peers_axis, read_strong, save_figure,  # noqa: E402
                     setup_log2_xaxis, strong_series)
from _style import (COLOR_IDEAL, FIG_H_TALL, F_ANNOT, INK_MUTED, INK_SECONDARY,  # noqa: E402
                    LW_IDEAL, MARK, SERIES, apply_rcparams)

STEM = "strong_speedup"


def plot(curve, dpi):
    apply_rcparams()
    s = strong_series(curve)
    w, p0 = s["workers"], s["baseline"]
    fig, ax = plt.subplots(figsize=(3.6, FIG_H_TALL))

    # Orange is spent on the peer axis here, so the measured curve takes the green slot.
    ax.plot(w, s["ideal"], "--", color=COLOR_IDEAL, lw=LW_IDEAL, label="ideal", zorder=2)
    ax.plot(w, s["end2end"], "^-", color=SERIES[2], label="wall time", zorder=4, **MARK)

    # Every point labelled: eight of them, so the four-digit pair 1024/2048 still clears.
    setup_log2_xaxis(ax, w)
    ax.set_yscale("log", base=2)
    ax.set_ylabel(f"Speedup vs {p0} GPUs", color=INK_SECONDARY)
    # Floor at 1: the w=16 point is pinned there by definition and nothing is drawn below it.
    # The ceiling still has to hold `ideal`, which reaches 128x at w=2048.
    ax.set_ylim(0.9, 200)
    ax.set_yticks([1, 2, 4, 8, 16, 32, 64, 128])
    ax.set_yticklabels(["1", "2", "4", "8", "16", "32", "64", "128"])
    peers_axis(ax, w, [r["peers_mean"] for r in curve])

    label_ends(ax, [(w[-1], s["end2end"][-1], f"{s['end2end'][-1]:.1f}x", SERIES[2])])

    # Speedup is a ratio, so nothing else on this panel says whether a run is seconds or hours.
    # Two numbers rather than eight: wall time is exactly total_time(16)/speedup, so per-point
    # labels would add precision, not information, and the three at the right (19.1, 18.5,
    # 19.2 s) would crowd the flattest, least interesting part of the curve. The endpoints are
    # what a reader quotes.
    ax.text(0.98, 0.04,
            f"wall clock {curve[0]['total_time']:.0f} s → {curve[-1]['total_time']:.0f} s "
            f"(w={p0} → {int(w[-1])})",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=F_ANNOT, color=INK_MUTED, zorder=6)

    ax.legend(loc="upper left", frameon=False, labelcolor=INK_SECONDARY,
              handletextpad=0.4, borderaxespad=0.2, handlelength=1.5)
    fig.tight_layout(pad=0.2)
    written = save_figure(fig, STEM, dpi)
    plt.close(fig)
    return written


def report(curve):
    s = strong_series(curve)
    r0 = curve[0]
    print(f"\n{STEM}: N={int(r0['total_neurons']):,} fixed, K={int(r0['in_degree'])}, "
          f"baseline w={s['baseline']} (itself in the ramp at "
          f"{int(r0['peers_mean'])} peers)")
    print(f"    {'w':>6} {'n/GPU':>8} {'step(ms)':>9} {'step x':>8} {'wall x':>10} "
          f"{'nocomm x':>9} {'eff':>7} {'peers':>6} {'comm%':>7}")
    for r, st_, e2, nc, ef in zip(curve, s["step"], s["end2end"], s["nocomm"], s["eff"]):
        print(f"    {int(r['workers']):>6} {int(r['neurons_per_worker']):>8,} "
              f"{r['step_s'] * 1e3:>9.3f} {st_:>7.2f}x {e2:>9.2f}x {nc:>8.2f}x {ef:>6.1f}% "
              f"{int(r['peers_mean']):>6} {r['comm_s'] / r['step_s'] * 100:>6.1f}%")
    peak = max(range(len(curve)), key=lambda i: s["step"][i])
    print(f"    step peaks {s['step'][peak]:.2f}x at w={int(curve[peak]['workers'])}, "
          f"ends {s['step'][-1]:.2f}x at w={int(curve[-1]['workers'])}")
    best = max(range(len(curve)), key=lambda i: s["end2end"][i])
    print(f"    wall time peaks {s['end2end'][best]:.2f}x at "
          f"w={int(curve[best]['workers'])}")
    print(f"    breakdown regime (peers > 26) begins at w={BREAKDOWN_W}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--print", dest="show", action="store_true", help="dump plotted values")
    args = p.parse_args(argv)

    curve, missing = read_strong()
    if missing:
        print(f"INCOMPLETE -- missing strong grid points: {missing}", file=sys.stderr)
    if args.show:
        report(curve)
    for path in plot(curve, args.dpi):
        print(f"  {path}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
