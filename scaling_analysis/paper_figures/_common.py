"""Data access and shared axis furniture for the paper figures.

This is a PRESENTATION layer. It reads the two final CSVs and nothing else -- consolidation,
the warm-up window and every derived timing belong to `analyze_weak.py` / `analyze_strong.py`,
which own the per-tick files. Re-deriving anything here would give the paper and the diagnostic
figures two sources of truth that drift apart silently, which is the failure mode this whole
campaign was rebuilt to avoid.

The regime constants below are the subject of the figure set, so they live in one place:

    PLATEAU_START   the first worker count whose rank grid is >= 4 in every dimension, which is
                    where a rank finally has all 26 Moore neighbours as DISTINCT peers. Below it
                    the periodic torus wraps several stencil directions onto the same rank, so
                    small-w points measure a cheaper communication pattern, not a better code.

    BREAKDOWN_W     strong scaling only: where the shrinking tile falls under the fixed stencil
                    radius, the bounded-peer premise stops holding and the peer count leaves 26.
"""

import csv
from pathlib import Path

from _style import AXIS, F_ANNOT, F_TICK, GRID, INK_MUTED, INK_SECONDARY, SURFACE

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parent / "outputs"
FIGS = HERE / "figs"

MOORE_PEERS = 26      # 3^3 - 1: a closed 3D Moore stencil
PLATEAU_START = 64    # first w with a rank grid >= 4 in every dimension
BREAKDOWN_W = 512     # strong: first w where the stencil outgrows the tile

IN_DEGREES = [4000, 2000, 1000]
WEAK_WORKERS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
STRONG_WORKERS = [16, 32, 64, 128, 256, 512, 1024, 2048]


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def _rows(path):
    if not path.exists():
        raise SystemExit(f"{path} missing -- run analyze_weak.py / analyze_strong.py first")
    with open(path, newline="") as f:
        out = []
        for r in csv.DictReader(f):
            rec = {}
            for k, v in r.items():
                try:
                    rec[k] = float(v)
                except (TypeError, ValueError):
                    rec[k] = v
            out.append(rec)
    return out


def read_weak(path=None):
    """Weak curves grouped by in-degree, each sorted by worker count."""
    curves = {}
    for r in _rows(path or OUTPUTS / "weak_3d_final.csv"):
        curves.setdefault(int(r["in_degree"]), []).append(r)
    for K in curves:
        curves[K].sort(key=lambda r: r["workers"])
    missing = [(K, w) for K in IN_DEGREES
               for w in WEAK_WORKERS
               if w not in {r["workers"] for r in curves.get(K, [])}]
    return curves, missing


def read_strong(path=None):
    """The strong curve, sorted by worker count."""
    curve = sorted(_rows(path or OUTPUTS / "strong_3d_final.csv"), key=lambda r: r["workers"])
    present = {r["workers"] for r in curve}
    return curve, [p for p in STRONG_WORKERS if p not in present]


# ---------------------------------------------------------------------------
# Derived series
# ---------------------------------------------------------------------------

def step_efficiency(curve, baseline=1):
    """Weak-scaling efficiency T(baseline)/T(w) as a percentage, on the steady-state step.

    `baseline=PLATEAU_START` is the one PLOTTED: it reads efficiency across the constant-peer
    regime, where the communication pattern is fixed and a 32x span of workers is therefore
    comparable like for like. `baseline=1` is the textbook definition and stays available for the
    text dump, because a 1-GPU run has no halo at all and the gap between the two numbers IS the
    one-time cost of turning communication on. Both are quoted in the captions: the T(1) figure
    alone invites the reader to extrapolate the ramp, the T(64) figure alone hides what
    communication costs.
    """
    base = next(r["step_s"] for r in curve if r["workers"] == baseline)
    return [base / r["step_s"] * 100 for r in curve]


def strong_series(curve):
    """Speedup and efficiency against the smallest worker count present.

    Mirrors `analyze_strong.py:scaling_series` rather than importing it -- that module pulls in
    matplotlib state and the whole consolidation path for four lines of arithmetic.

    Three curves, because one of them alone misleads:
      * `step`      -- the simulation step. Reverses past w=256; reported, not plotted by F2.
      * `end2end`   -- wall time (`total_time`): what a user waits for, and F2's only measured
                       curve. It keeps improving well past the point the step stops, because
                       construction still parallelises.
      * `nocomm`    -- the step with the measured ghost exchange removed, against its own
                       baseline so both read as scaling SHAPES rather than one looking superlinear.
    """
    p0 = curve[0]["workers"]
    t0 = curve[0]["step_s"]
    e0 = curve[0]["total_time"]
    nocomm = [max(r["step_s"] - r["comm_s"], 1e-9) for r in curve]
    speedup = [t0 / r["step_s"] for r in curve]
    ideal = [r["workers"] / p0 for r in curve]
    return {
        "baseline": int(p0),
        "workers": [r["workers"] for r in curve],
        "ideal": ideal,
        "step": speedup,
        "end2end": [e0 / r["total_time"] for r in curve],
        "nocomm": [nocomm[0] / t for t in nocomm],
        "eff": [s / i * 100 for s, i in zip(speedup, ideal)],
    }


# ---------------------------------------------------------------------------
# Axis furniture
# ---------------------------------------------------------------------------

def setup_log2_xaxis(ax, workers, label="Workers (GPUs)", rotation=0):
    """Powers-of-two x-axis with a labelled tick at every measured point.

    `rotation` buys the room to label ALL of them when the sweep is long: over twelve points the
    four-digit pair 1024/2048 is the only one that collides horizontally, and slanting clears it
    without dropping labels the reader would then have to infer.
    """
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(workers))
    ax.set_xticklabels([str(int(w)) for w in workers], fontsize=F_TICK, rotation=rotation,
                       ha="right" if rotation else "center",
                       rotation_mode="anchor" if rotation else None)
    ax.minorticks_off()
    ax.set_xlabel(label, color=INK_SECONDARY)
    ax.set_xlim(workers[0] / 1.5, workers[-1] * 1.5)
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK_MUTED, labelsize=F_TICK, length=2, width=0.6)


def peers_axis(ax, workers, peers, label="MPI peers / rank", color=None):
    """Twin y-axis carrying the peer count -- the communication topology behind the main curve.

    `color` is overridable because the default orange collides with a series colour on figures
    that already spend three hues on the left axis.

    A LINE WITH MARKERS, NOT A STEP. `peers_mean` is a mean over ranks: on the strong sweep it
    reads 95.55 at w=2048, which is no rank's actual peer count. A step draws it as a discrete
    level held across a range, which misstates it twice -- it is an average, and it asserts
    constancy between worker counts that were never run. Straight segments between measured points
    are also the convention every other series here uses.

    The underlying quantity is deterministic geometry, not a measurement: rank grid x tile shape x
    connection radius reproduces the measured value exactly through w=1024. It is NOT monotonic in
    w, though -- w=25 factors 1x5x5 and gives 8 peers against w=16's 11 -- so it must never be
    drawn as a curve through worker counts the sweep did not run.
    """
    from _style import COLOR_XRANK
    color = color or COLOR_XRANK
    ax2 = ax.twinx()
    # No reference rule at MOORE_PEERS: the curve already sits visibly on 26 across the plateau
    # and departs above it, so a full-width dotted line labelled a value the data was showing
    # anyway, at the cost of a third orange element on the panel.
    ax2.plot(workers, peers, "o-", color=color, lw=1.0, ms=2.4, alpha=0.9, zorder=2)
    ax2.set_ylabel(label, color=color)
    ax2.set_ylim(0, max(peers) * 1.25)
    ax2.tick_params(axis="y", colors=color, labelsize=F_TICK, length=2, width=0.6)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(color)
    ax2.spines["right"].set_linewidth(0.6)
    ax2.set_zorder(0)
    ax.set_zorder(1)
    ax.patch.set_visible(False)
    return ax2


def label_end(ax, x, y, text, color, dy=0):
    ax.annotate(text, xy=(x, y), xytext=(4, dy), textcoords="offset points",
                ha="left", va="center", fontsize=F_ANNOT, color=color, fontweight="bold",
                zorder=6)


def label_ends(ax, entries, min_gap_pt=6.0):
    """Right-edge labels for several curves, nudged apart so close values stay readable.

    `entries` is [(x, y, text, color), ...]. Curves whose final values sit within a couple of
    percent of each other -- which is exactly what a converged plateau looks like -- would
    otherwise print one label on top of another.
    """
    if not entries:
        return
    order = sorted(range(len(entries)), key=lambda i: entries[i][1])
    ys_pt = {i: ax.transData.transform((0, entries[i][1]))[1] for i in order}
    placed = {}
    for i in order:
        want = ys_pt[i]
        if placed:
            floor = max(placed.values()) + min_gap_pt
            want = max(want, floor)
        placed[i] = want
    for i, (x, y, text, color) in enumerate(entries):
        # transform() works in display pixels; convert the nudge back to points for the offset.
        dy_px = placed[i] - ys_pt[i]
        label_end(ax, x, y, text, color, dy=dy_px * 72.0 / ax.figure.dpi)


def save_figure(fig, stem, dpi=600):
    """Write the PNG/PDF pair GGap's `save_figure` writes: raster for slides, vector for LaTeX."""
    FIGS.mkdir(parents=True, exist_ok=True)
    png, pdf = FIGS / f"{stem}.png", FIGS / f"{stem}.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight", facecolor=SURFACE)
    fig.savefig(pdf, format="pdf", bbox_inches="tight", transparent=True)
    return [png, pdf]
