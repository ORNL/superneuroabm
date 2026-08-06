"""Weak-scaling analysis: raw run output in, final table + figures out.

Weak scaling holds per-GPU work constant (12,500 neurons x K synapses per worker) while the
worker count grows 1 -> 2048, so a perfectly weak-scaling code draws a flat line. Everything
interesting is how far from flat the curves sit and where they stop drifting.

This script is self-contained on purpose -- one file is the whole story for the weak campaign.
It does three things in one pass:

  1. CONSOLIDATE the append-only run summaries. Chunk jobs retry failed points and overlap, so
     the raw CSVs carry duplicate and stale rows; each (K, workers) point collapses to its
     newest job_id, and missing grid points are reported. That coverage check is the only thing
     standing between "36/36" and an eyeball judgement.
  2. DERIVE the timing metrics from the per-tick files. The runs record every tick truthfully;
     this script splits them into the first tick and the rest.
  3. REPORT: final CSV, printed table, figures.

FIRST TICK VS THE REST -- THE ONLY SPLIT
    The network is CONSTRUCTED during tick 1: the GPU buffers are built and the ghost topology
    is discovered. That costs roughly 200 s at K=1000 and 785 s at K=4000, against a ~5 ms tick
    afterwards -- so a single conflated `simulation_time` is ~99.9% construction, which is how a
    flat *construction* curve came to be published as a flat weak-scaling curve.

    Tick 1 is therefore never averaged into the step. It is reported on its own as
    `first_tick_s`, and the step is `step_s`. Both are real costs and both are published; what
    is not allowed is one number standing for both.

WHY THE STEP IS A MEAN OVER A WARM-UP WINDOW
    The field convention for a time-stepping code is the MEAN of a steady-state window, with
    setup and a warm-up both excluded -- GROMACS ships `-resetstep`/`-resethway` for exactly
    this, and its `-resethway` default discards half the steps. So the metric here is a mean;
    the only question is where the window starts.

    Excluding tick 1 alone is not enough. Construction spills past tick 1 at large w: K=4000
    w=2048 spends 111 s on tick 2, 50 s on tick 3 and 3.7 s on tick 4 before settling to an
    18.6 ms step, against <= 1.7 s at every other worker count. A mean over ticks 2..100 there
    reads 1,693 ms against an 18.6 ms step -- 91x wrong.

    WARMUP_TICKS is measured, not chosen by taste. Over the 44 runs in the two campaigns the
    mean and the median of the window agree to 0.09% (median across runs) once 10 ticks are
    dropped, against 5.59% at W=2; only three runs then exceed 1%. That agreement IS the test:
    when a mean and a median of the same samples coincide, the window has cleared the tail.

    Three runs settle later than 10 ticks and are reported rather than trimmed harder --
    weak K=4000 w=2048 (~30), strong P=2048 (~50), weak K=1000 w=32. Widening the window to
    cover them costs 30% of the samples to fix 3 points of 44, so the window stays at 10 and
    the exceptions are named. `step_median_s` is carried in the CSV beside `step_s` so the
    agreement is checkable per point rather than taken on trust.

WHY THE COMPONENTS USE THE SAME WINDOW
    A decomposition must describe the same ticks as the step it decomposes. Components taken
    over a different window (or with no window) let the settling tail back in through the
    component columns after it had been kept out of the step column, so a `comm %` ends up
    describing the settling ticks rather than a typical step. Same window, same statistic,
    all the way down; the small residual is what `other_s` carries.

Usage::

    python analyze_weak.py
    python analyze_weak.py --outdir figures
"""

import argparse
import csv
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend choice)

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"

# Validated categorical slots, used in fixed order and never cycled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, INK = "#fcfcfb", "#0b0b0b"
INK_SECONDARY, INK_MUTED = "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
MARK = dict(linewidth=2, markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5)

IN_DEGREES = [4000, 2000, 1000]
EXPECTED_WORKERS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

# Ticks dropped before the step window opens. Measured -- see the module docstring. Tick 1 is
# construction and is excluded regardless; this is the settling tail after it.
WARMUP_TICKS = 10

# Level 1 of the per-step decomposition: disjoint, sums to the step apart from `other`.
STEP_PARTS = [("comm", "ghost exchange"), ("compute", "GPU compute"),
              ("gpu_sync", "GPU sync"), ("write_back", "write back")]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", type=Path, nargs="+", default=None,
                   help="Run-summary CSVs (default: outputs/weak_3d_curve_*.csv)")
    p.add_argument("--ticks-dir", type=Path, default=OUTPUTS / "ticks",
                   help="Directory of per-tick CSVs written by weak_scaling.py")
    p.add_argument("--output", type=Path, default=OUTPUTS / "weak_3d_final.csv")
    p.add_argument("--outdir", type=Path, default=HERE / "figures")
    p.add_argument("--ticks", type=int, default=100,
                   help="Keep only run rows with this tick count (default 100)")
    p.add_argument("--warmup-ticks", type=int, default=WARMUP_TICKS,
                   help=f"Ticks dropped before the step window opens (default {WARMUP_TICKS}; "
                        "tick 1 is construction and is always excluded)")
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# 1. Consolidate
# ---------------------------------------------------------------------------

def read_summaries(paths, ticks):
    """Load run summaries, drop other tick counts, keep the newest row per (K, workers)."""
    rows, header = [], None
    for path in paths:
        if not path.exists():
            print(f"  WARNING: {path.name} missing -- skipped")
            continue
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if header is None:
                header = reader.fieldnames
            elif reader.fieldnames != header:
                raise SystemExit(f"header mismatch in {path.name}; refusing to merge")
            file_rows = list(reader)
        rows.extend(file_rows)
        print(f"  {path.name}: {len(file_rows)} rows")
    if header is None:
        raise SystemExit("no run-summary CSVs found -- run weak_3d_chunk.sh first")

    kept = [r for r in rows if int(r["ticks"]) == ticks]
    if len(kept) != len(rows):
        print(f"  dropped {len(rows) - len(kept)} row(s) with ticks != {ticks}")

    groups = defaultdict(list)
    for r in kept:
        groups[(int(r["in_degree"]), int(r["workers"]))].append(r)
    canonical, dupes = {}, 0
    for key, group in sorted(groups.items()):
        group.sort(key=lambda r: int(r["job_id"] or 0))
        canonical[key] = dict(group[-1])
        canonical[key]["n_runs"] = len(group)
        if len(group) > 1:
            dupes += 1
    print(f"  {len(canonical)} unique (K, workers) points; {dupes} had repeats")
    return canonical, header


def report_coverage(canonical):
    """Print the coverage matrix; return the missing grid points."""
    holes = []
    for K in IN_DEGREES:
        present = [w for w in EXPECTED_WORKERS if (K, w) in canonical]
        missing = [w for w in EXPECTED_WORKERS if (K, w) not in canonical]
        holes += [(K, w) for w in missing]
        status = "complete" if not missing else f"MISSING {missing}"
        print(f"  K={K:>4}: {len(present)}/{len(EXPECTED_WORKERS)} -- {status}")
    return holes


# ---------------------------------------------------------------------------
# 2. Derive from the per-tick files
# ---------------------------------------------------------------------------

def load_ticks(ticks_dir, K, npp, workers, job_id):
    """Per-tick rows for one run. Prefers the matching job_id, else the newest match."""
    stem = f"weak_K{K}_npp{npp}_w{workers}_"
    cands = sorted(Path(ticks_dir).glob(stem + "*.csv"))
    if not cands:
        return None
    exact = [c for c in cands if c.stem.endswith(f"_{job_id}")]
    with open(exact[-1] if exact else cands[-1], newline="") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def derive(tick_rows, warmup=WARMUP_TICKS):
    """Split the ticks into construction (tick 1) and a steady-state window.

    `warmup` counts from tick 1, so the window is ticks warmup+1 .. N. The step and every
    component are MEANS over that one window; `step_median_s` is the same window's median,
    carried so the mean/median agreement can be checked per point.
    """
    if not tick_rows:
        return None
    # A 1-tick file has no steady state. REFUSE it rather than falling back to tick 1: that
    # fallback reported a 188 s "step time" for the fused w=1 runs -- a number that is plausible
    # in shape, wrong by 4-5 orders of magnitude, and was silently used as the efficiency
    # baseline. Returning None makes the caller list the point as missing per-tick data, which
    # is visible.
    if len(tick_rows) < 2:
        return None
    # Short files still get a window rather than nothing: fall back to everything after tick 1.
    window = tick_rows[warmup:] if len(tick_rows) > warmup else tick_rows[1:]
    totals = [r["total_mean"] for r in window]
    out = {
        "first_tick_s": tick_rows[0]["total_mean"],
        "step_s": st.fmean(totals),
        "step_median_s": st.median(totals),
        "warmup_ticks": warmup if len(tick_rows) > warmup else 1,
        "n_step_ticks": len(window),
    }
    for name, _ in STEP_PARTS:
        out[f"{name}_s"] = st.fmean([r[f"{name}_mean"] for r in window])
    for name in ("pack", "exchange", "unpack", "wait"):
        out[f"{name}_s"] = st.fmean([r[f"{name}_mean"] for r in window])
    out["other_s"] = max(out["step_s"] - sum(out[f"{n}_s"] for n, _ in STEP_PARTS), 0.0)
    return out


def build_curves(canonical, ticks_dir, warmup=WARMUP_TICKS):
    """Merge summary facts with derived timings; group by in-degree, sorted by workers."""
    curves, missing_ticks = defaultdict(list), []
    for (K, w), row in sorted(canonical.items()):
        npp = int(row["neurons_per_worker"])
        tr = load_ticks(ticks_dir, K, npp, w, row.get("job_id", ""))
        d = derive(tr, warmup)
        if d is None:
            missing_ticks.append((K, w))
            continue
        rec = {"in_degree": K, "workers": w, "neurons_per_worker": npp,
               "nodes": int(row["nodes"] or 0),
               "total_neurons": int(row["total_neurons"]),
               "synapses": int(row["synapses"]), "ticks": int(row["ticks"]),
               "n_runs": row["n_runs"]}
        for k in ("generation_time", "model_creation_time", "gpu_setup_time",
                  "construction_time", "simulation_time", "total_time",
                  "peers_mean", "ghost_somas_mean", "ghost_local_ratio",
                  "send_bytes_mean", "bytes_per_peer"):
            rec[k] = float(row[k]) if row.get(k) else 0.0
        rec.update(d)
        curves[K].append(rec)
    if missing_ticks:
        print(f"  WARNING: no per-tick file for {len(missing_ticks)} point(s): {missing_ticks}")
    for K in curves:
        curves[K].sort(key=lambda r: r["workers"])
    return curves


# ---------------------------------------------------------------------------
# 3. Report
# ---------------------------------------------------------------------------

def style_axes(ax, workers):
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log", base=2)
    ax.set_xticks(workers)
    ax.set_xticklabels([str(w) for w in workers], fontsize=8)
    ax.minorticks_off()
    ax.set_xlabel("Workers (GPUs)", fontsize=10, color=INK_SECONDARY)
    ax.set_xlim(workers[0] / 1.4, workers[-1] * 2.6)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=3, width=0.8)


def label_end(ax, x, y, text, color, dy=0):
    ax.annotate(text, xy=(x, y), xytext=(8, dy), textcoords="offset points",
                ha="left", va="center", fontsize=8, color=color, fontweight="bold")


def plot_curves(curves, outdir, dpi):
    """Two panels: the steady-state step (the real metric) and the first tick, which is
    construction (what used to be reported as `simulation_time`). Both are flat for a
    weak-scaling code, but for different reasons, and conflating them is what made the old
    curve misleading."""
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for K in IN_DEGREES:
        curve = curves.get(K)
        if not curve:
            continue
        w = [r["workers"] for r in curve]
        fig, (ax_step, ax_build) = plt.subplots(1, 2, figsize=(11, 4.8), facecolor=SURFACE)

        step = [r["step_s"] * 1e3 for r in curve]
        ax_step.axhline(step[0], color=AXIS, ls="--", lw=1.2, label=f"flat at {step[0]:.2f} ms")
        ax_step.plot(w, step, "o-", color=SERIES[0],
                     label=f"mean of ticks {curve[0]['warmup_ticks'] + 1}..N", **MARK)
        label_end(ax_step, w[-1], step[-1], f"{step[-1]:.2f} ms", SERIES[0])
        style_axes(ax_step, w)
        ax_step.set_ylabel("Time per simulation step (ms)", fontsize=10, color=INK_SECONDARY)
        ax_step.set_ylim(bottom=0)
        ax_step.set_title("Steady-state step", fontsize=11, color=INK, pad=8)
        ax_step.legend(fontsize=8, frameon=False, labelcolor=INK_SECONDARY, loc="lower right")

        build = [r["first_tick_s"] for r in curve]
        ax_build.axhline(build[0], color=AXIS, ls="--", lw=1.2, label=f"flat at {build[0]:.0f} s")
        ax_build.plot(w, build, "o-", color=SERIES[1],
                      label="first tick (build + ghost discovery)", **MARK)
        label_end(ax_build, w[-1], build[-1], f"{build[-1]:.0f} s", SERIES[1])
        style_axes(ax_build, w)
        ax_build.set_ylabel("Construction time (s)", fontsize=10, color=INK_SECONDARY)
        ax_build.set_ylim(bottom=0)
        ax_build.set_title("First tick — network construction", fontsize=11, color=INK, pad=8)
        ax_build.legend(fontsize=8, frameon=False, labelcolor=INK_SECONDARY, loc="lower right")

        fig.suptitle(f"Weak scaling — 3D torus, in-degree K={K}", fontsize=13, color=INK,
                     x=0.02, ha="left", y=0.98)
        fig.text(0.02, 0.915,
                 f"{curve[0]['neurons_per_worker']:,} neurons + "
                 f"{curve[0]['synapses'] / 1e6:.1f} M synapses per GPU · {curve[0]['ticks']} ticks "
                 f"· constant work per worker",
                 fontsize=9, color=INK_MUTED, ha="left")
        # Read the two magnitudes off this K's own curve -- they differ ~4x across K, so a
        # hardcoded "~200 s against a ~5 ms step" is wrong on every panel but K=1000.
        build_med = st.median([r["first_tick_s"] for r in curve])
        step_med = st.median([r["step_s"] for r in curve])
        fig.text(0.02, 0.025,
                 "Left is the simulation step; right is the first tick, the one-time construction "
                 f"that dominates wall time (~{build_med:.0f} s against a ~{step_med * 1e3:.0f} ms "
                 "step).\n"
                 "Reported separately because a single 'simulation time' number conflates them "
                 f"and is ~{100 * build_med / (build_med + step_med * (curve[0]['ticks'] - 1)):.1f}% "
                 "construction.",
                 fontsize=7.5, color=INK_MUTED, ha="left", linespacing=1.5)
        fig.tight_layout(rect=[0, 0.06, 1, 0.89])
        path = outdir / f"weak_3d_K{K}.png"
        fig.savefig(path, dpi=dpi, facecolor=SURFACE)
        plt.close(fig)
        written.append(path)
    return written


def plot_comm(curves, outdir, dpi):
    """Peers, ghost volume and the exchange share -- why the step curve has the shape it does."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=SURFACE)
    ax_peers, ax_ghost, ax_frac = axes
    base = next((curves[K] for K in IN_DEGREES if curves.get(K)), None)
    if base is None:
        return []
    w = [r["workers"] for r in base]

    peers = [r["peers_mean"] for r in base]
    ax_peers.plot(w, peers, "o-", color=SERIES[0], **MARK)
    label_end(ax_peers, w[-1], peers[-1], f"{peers[-1]:.0f}", SERIES[0])
    ax_peers.set_title("MPI peers per rank", fontsize=11, color=INK, pad=8)
    ax_peers.set_ylabel("Peer ranks", fontsize=10, color=INK_SECONDARY)

    for color, K in zip(SERIES, IN_DEGREES):
        c = curves.get(K)
        if not c:
            continue
        ax_ghost.plot([r["workers"] for r in c], [r["ghost_somas_mean"] for r in c], "o-",
                      color=color, label=f"K={K}", **MARK)
        ax_frac.plot([r["workers"] for r in c],
                     [r["comm_s"] / r["step_s"] * 100 if r["step_s"] else 0
                      for r in c],
                     "o-", color=color, label=f"K={K}", **MARK)
    ax_ghost.set_title("Ghost somas per rank", fontsize=11, color=INK, pad=8)
    ax_ghost.set_ylabel("Ghost somas", fontsize=10, color=INK_SECONDARY)
    ax_ghost.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    ax_frac.set_title("Ghost exchange share of a step", fontsize=11, color=INK, pad=8)
    ax_frac.set_ylabel("Share of step time (%)", fontsize=10, color=INK_SECONDARY)
    for ax in (ax_ghost, ax_frac):
        ax.legend(fontsize=8, frameon=False, labelcolor=INK_SECONDARY, loc="upper left")
    for ax in axes:
        style_axes(ax, w)
        ax.set_ylim(bottom=0)

    fig.suptitle("Communication diagnostics — 3D torus weak scaling", fontsize=13, color=INK,
                 x=0.02, ha="left", y=0.98)
    fig.text(0.02, 0.915,
             f"Per-rank means over ticks {WARMUP_TICKS + 1}..N — the same window as the step, so "
             "the parts sum to it.",
             fontsize=9, color=INK_MUTED, ha="left")
    fig.tight_layout(rect=[0, 0.04, 1, 0.89])
    path = outdir / "weak_3d_comm.png"
    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)
    return [path]


def print_table(curves):
    """The table-view twin: every plotted value readable as text."""
    for K in IN_DEGREES:
        curve = curves.get(K)
        if not curve:
            continue
        print(f"\nK={K}  ({curve[0]['synapses']:,} local synapses/GPU, {curve[0]['ticks']} ticks)")
        hdr = (f"{'w':>5} {'nodes':>6} {'step(ms)':>10} {'median':>10} {'m-md%':>7} "
               f"{'first_tick(s)':>14} {'flat%':>7} {'peers':>6} {'ghost':>8} {'comm%':>7}")
        print(hdr)
        print("-" * len(hdr))
        base = curve[0]["step_s"]
        for r in curve:
            # The mean/median gap is the window check: a wide-enough window makes them agree.
            gap = (r["step_s"] - r["step_median_s"]) / r["step_median_s"] * 100
            print(f"{r['workers']:>5} {r['nodes']:>6} {r['step_s'] * 1e3:>10.3f} "
                  f"{r['step_median_s'] * 1e3:>10.3f} {gap:>+6.2f}% {r['first_tick_s']:>14.2f} "
                  f"{base / r['step_s'] * 100:>6.1f}% {r['peers_mean']:>6.0f} "
                  f"{r['ghost_somas_mean']:>8,.0f} "
                  f"{r['comm_s'] / r['step_s'] * 100 if r['step_s'] else 0:>6.1f}%")


def write_final(path, curves):
    fields = ["in_degree", "workers", "nodes", "total_neurons", "neurons_per_worker", "synapses",
              "ticks", "n_runs", "warmup_ticks", "n_step_ticks",
              "first_tick_s", "step_s", "step_median_s",
              "comm_s", "compute_s", "gpu_sync_s", "write_back_s", "other_s",
              "pack_s", "exchange_s", "unpack_s", "wait_s",
              "generation_time", "model_creation_time", "gpu_setup_time", "construction_time",
              "simulation_time", "total_time",
              "peers_mean", "ghost_somas_mean", "ghost_local_ratio", "send_bytes_mean",
              "bytes_per_peer"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        for K in IN_DEGREES:
            for r in curves.get(K, []):
                wr.writerow(r)
    return sum(len(curves.get(K, [])) for K in IN_DEGREES)


def main(argv=None):
    args = parse_args(argv)
    inputs = args.inputs or sorted(OUTPUTS.glob("weak_3d_curve_*.csv"))

    print("Consolidating run summaries (read-only; the raw files are append-only provenance):")
    canonical, _ = read_summaries(inputs, args.ticks)
    print("\nCoverage:")
    holes = report_coverage(canonical)

    print(f"\nDeriving timings from per-tick files ({args.ticks_dir}); tick 1 is construction, "
          f"the step is the mean of ticks {args.warmup_ticks + 1}..N:")
    curves = build_curves(canonical, args.ticks_dir, args.warmup_ticks)

    n = write_final(args.output, curves)
    print(f"\nWrote {n} rows to {args.output}")
    print_table(curves)

    written = plot_curves(curves, args.outdir, args.dpi)
    written += plot_comm(curves, args.outdir, args.dpi)
    print("\nFigures:")
    for p in written:
        print(f"  {p}")

    if holes:
        print(f"\nINCOMPLETE -- {len(holes)} missing grid point(s): {holes}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
