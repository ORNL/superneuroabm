"""Shared IEEE print style for the SuperNeuroABM scaling figures.

Geometry and typography are ported from `GGap/SC2026/figures/scripts/_style.py` so the two
projects' figures sit together in a proceedings without looking like different papers. The
categorical palette is NOT ported: it stays the one `analyze_weak.py` / `analyze_strong.py`
already use, so a given in-degree is the same colour in a paper figure and in the diagnostic
figure it was checked against.

Unlike the GGap scaling scripts, every figure here calls `apply_rcparams()`. Those scripts set
all fifteen font sizes by hand on each artist, which is why their defaults drift.
"""

import matplotlib as mpl

# ── IEEE conference column widths (inches) ──────────────────────────────
# Reference only -- nothing imports these. They record the target medium so a new figure can be
# sized against it without going back to the IEEEtran class file.
COL_SINGLE = 3.487   # IEEEtran conference single-column text width
COL_DOUBLE = 7.16    # IEEEtran conference text width (full-page span)

# Standard panel: one figure per file, LaTeX subcaption composes them.
FIG_W, FIG_H = 3.3, 1.8
FIG_H_TALL = 2.1     # for panels carrying two series families plus a twin axis

# ── Font tiers (PRINT pt) ───────────────────────────────────────────────
F_TITLE = 8.0
F_LABEL = 7.0   # axis labels
F_TICK = 6.5    # tick labels
F_LEG = 6.0     # legend text
F_ANNOT = 5.5   # in-plot annotations and end-labels

# ── Categorical slots, used in fixed order and never cycled ─────────────
# Same three as analyze_weak.py: K=4000, K=2000, K=1000 in that order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]

# ── Reference / structural colours ──────────────────────────────────────
COLOR_IDEAL = "#374151"   # dark slate — ideal and baseline reference lines (from GGap)
COLOR_XRANK = "#FB8C00"   # orange — twin-axis overlay (from GGap)
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"       # white, not the diagnostic figures' cream: these go on paper

# ── Line weights ────────────────────────────────────────────────────────
LW_PRIMARY = 1.6     # the measured series that carries the claim
LW_IDEAL = 1.2
MS_PRIMARY = 3.5

MARK = dict(linewidth=LW_PRIMARY, markersize=MS_PRIMARY,
            markeredgecolor=SURFACE, markeredgewidth=0.6)


def apply_rcparams():
    """Set matplotlib defaults so per-figure code stays about the data."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": F_TICK,
        "axes.titlesize": F_TITLE,
        "axes.labelsize": F_LABEL,
        "xtick.labelsize": F_TICK,
        "ytick.labelsize": F_TICK,
        "legend.fontsize": F_LEG,
        "figure.titlesize": F_TITLE,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,   # embed TrueType, not Type 3 — IEEE requires it
        "ps.fonttype": 42,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
    })
