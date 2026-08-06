"""Strong-scaling analysis: raw run output in, final table + figures out.

Strong scaling holds the TOTAL problem fixed (204,800 neurons at in-degree K=1000) while the
worker count grows 16 -> 2048, so per-rank work falls as 1/P and a perfectly scaling code draws
the ideal diagonal. Everything interesting is where the measured curve leaves it, and why.

Self-contained on purpose -- one file is the whole story for the strong campaign. Three things
in one pass:

  1. CONSOLIDATE the append-only run summaries: drop other tick counts, collapse each
     (N, workers) point to its newest job_id, report missing grid points.
  2. DERIVE the timing metrics from the per-tick files. The runs record every tick truthfully;
     this script splits them into the first tick and the rest.
  3. REPORT: final CSV, printed table, three figures.

FIRST TICK VS THE REST -- THE ONLY SPLIT
    The network is CONSTRUCTED during tick 1: the GPU buffers are built and the ghost topology
    is discovered (~200 s against a ~5 ms tick afterwards). Tick 1 is therefore never averaged
    into the step. Two numbers are reported: `first_tick_s` (construction) and `step_s` (a
    simulation step). Both are real costs; what is not allowed is one standing for both.

WHY THE STEP IS A MEAN OVER A WARM-UP WINDOW
    The field convention for a time-stepping code is the MEAN of a steady-state window, with
    setup and a warm-up both excluded -- GROMACS ships `-resetstep`/`-resethway` for exactly
    this. So the metric is a mean; the only question is where the window starts.

    Excluding tick 1 alone is not enough. At P=256 exactly three ticks out of 99 (ticks 2-4, at
    864/945/628 ms against a 3.6 ms typical step) carry 87% of a mean taken over ticks 2..N, so
    that mean reports the settling ticks rather than a step -- it read `comm` at 94% where the
    steady-state share is ~57%.

    WARMUP_TICKS is measured. Over the 44 runs in the two campaigns the mean and the median of
    the window agree to 0.09% (median across runs) once 10 ticks are dropped, against 5.59% at
    W=2. That agreement IS the test: when a mean and a median of the same samples coincide, the
    window has cleared the tail. `step_median_s` is carried in the CSV beside `step_s` so it is
    checkable per point rather than taken on trust.

    P=2048 is the one point in this campaign that settles later than the window (~tick 50, on a
    bimodal 6.6/7.5 ms pattern that resolves at tick 51), and its +3.4% mean/median gap is
    reported rather than trimmed harder -- widening the window to cover it would cost 30% of the
    samples at every other point.

WHY THE COMPONENTS USE THE SAME WINDOW
    A decomposition must describe the same ticks as the step it decomposes. Components taken
    over a different window let the settling tail back in through the component columns after it
    had been kept out of the step column. Same window, same statistic, all the way down; the
    residual is what `other_s` carries (it ran a flat 0.105 ms at every worker count under the
    previous median-based derivation, so the decomposition is additive in practice).

BASELINE
    Speedup is read against the SMALLEST worker count present, which is the smallest the fixed
    problem fits on -- not 1 GPU, which never ran. The figures name the baseline explicitly.

Usage::

    python analyze_strong.py
    python analyze_strong.py --outdir figures
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

# Categorical slots 1-5 of the validated palette, in fixed order, never cycled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
# The exchange internals are a part-whole sequence in time, so they get an ordered one-hue ramp
# rather than more categorical hues; steps spread across the ramp so adjacent lines stay apart.
BLUE_RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
SURFACE, INK = "#fcfcfb", "#0b0b0b"
INK_SECONDARY, INK_MUTED = "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
MARK = dict(linewidth=2, markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5)

EXPECTED_WORKERS = [16, 32, 64, 128, 256, 512, 1024, 2048]
MOORE_PEERS = 26  # where the bounded-peer premise stops holding

# Ticks dropped before the step window opens. Measured -- see the module docstring. Tick 1 is
# construction and is excluded regardless; this is the settling tail after it.
WARMUP_TICKS = 10

STEP_PARTS = [("comm", "ghost exchange"), ("compute", "GPU compute"),
              ("gpu_sync", "GPU sync"), ("write_back", "write back")]
COMM_PARTS = [("pack", "pack"), ("exchange", "MPI exchange"), ("unpack", "unpack")]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", type=Path, nargs="+", default=None,
                   help="Run-summary CSVs (default: outputs/strong_3d_curve_*.csv)")
    p.add_argument("--ticks-dir", type=Path, default=OUTPUTS / "ticks")
    p.add_argument("--output", type=Path, default=OUTPUTS / "strong_3d_final.csv")
    p.add_argument("--outdir", type=Path, default=HERE / "figures")
    p.add_argument("--ticks", type=int, default=100)
    p.add_argument("--warmup-ticks", type=int, default=WARMUP_TICKS,
                   help=f"Ticks dropped before the step window opens (default {WARMUP_TICKS}; "
                        "tick 1 is construction and is always excluded)")
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# 1. Consolidate
# ---------------------------------------------------------------------------

def read_summaries(paths, ticks):
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
        raise SystemExit("no run-summary CSVs found -- run strong_3d_chunk.sh first")

    kept = [r for r in rows if int(r["ticks"]) == ticks]
    if len(kept) != len(rows):
        print(f"  dropped {len(rows) - len(kept)} row(s) with ticks != {ticks}")

    groups = defaultdict(list)
    for r in kept:
        groups[(int(r["total_neurons"]), int(r["in_degree"]), int(r["workers"]))].append(r)
    canonical, dupes = {}, 0
    for key, group in sorted(groups.items()):
        group.sort(key=lambda r: int(r["job_id"] or 0))
        canonical[key] = dict(group[-1])
        canonical[key]["n_runs"] = len(group)
        if len(group) > 1:
            dupes += 1
    print(f"  {len(canonical)} unique (N, K, workers) points; {dupes} had repeats")
    return canonical


# ---------------------------------------------------------------------------
# 2. Derive
# ---------------------------------------------------------------------------

def load_ticks(ticks_dir, N, K, workers, job_id):
    stem = f"strong_N{N}_K{K}_w{workers}_"
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
    out["comm_other_s"] = max(out["comm_s"] - out["pack_s"] - out["exchange_s"] - out["unpack_s"],
                              0.0)
    return out


def build_curve(canonical, ticks_dir, warmup=WARMUP_TICKS):
    curve, missing = [], []
    for (N, K, w), row in sorted(canonical.items()):
        tr = load_ticks(ticks_dir, N, K, w, row.get("job_id", ""))
        d = derive(tr, warmup)
        if d is None:
            missing.append((N, K, w))
            continue
        rec = {"total_neurons": N, "in_degree": K, "workers": w,
               "neurons_per_worker": int(row["neurons_per_worker"]),
               "nodes": int(row["nodes"] or 0), "synapses": int(row["synapses"]),
               "ticks": int(row["ticks"]), "n_runs": row["n_runs"]}
        for k in ("generation_time", "model_creation_time", "gpu_setup_time",
                  "construction_time", "simulation_time", "total_time",
                  "peers_mean", "ghost_somas_mean", "ghost_local_ratio",
                  "send_bytes_mean", "bytes_per_peer"):
            rec[k] = float(row[k]) if row.get(k) else 0.0
        rec.update(d)
        curve.append(rec)
    if missing:
        print(f"  WARNING: no per-tick file for {len(missing)} point(s): {missing}")
    curve.sort(key=lambda r: r["workers"])
    sizes = {r["total_neurons"] for r in curve}
    if len(sizes) > 1:
        raise SystemExit(f"mixed problem sizes {sorted(sizes)}; strong scaling needs one fixed N")
    return curve


def scaling_series(curve):
    """Speedup/efficiency, measured and with the exchange removed.

    The comm-removed variant is computed ENTIRELY in means and against its own baseline:
    `comm_s` is a mean, so subtracting it from a median step would be invalid (and past the
    midpoint the stall-inflated comm mean can exceed the median, producing nonsense). Its own
    baseline keeps both curves starting at 1.0, so they compare as scaling SHAPES rather than
    one appearing superlinear.
    """
    p0, t0 = curve[0]["workers"], curve[0]["step_s"]
    ideal = [r["workers"] / p0 for r in curve]
    speedup = [t0 / r["step_s"] for r in curve]
    nocomm = [max(r["step_s"] - r["comm_s"], 1e-9) for r in curve]
    sp_nc = [nocomm[0] / t for t in nocomm]
    return {
        "baseline": p0, "workers": [r["workers"] for r in curve], "ideal": ideal,
        "speedup": speedup, "speedup_nocomm": sp_nc,
        "eff": [s / i * 100 for s, i in zip(speedup, ideal)],
        "eff_nocomm": [s / i * 100 for s, i in zip(sp_nc, ideal)],
    }


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


def shade_breakdown(ax, curve):
    """Shade where the stencil outgrows the tile and the peer count leaves 26 -- a different
    regime, not more of the same curve."""
    broken = [r["workers"] for r in curve if r["peers_mean"] > MOORE_PEERS]
    if not broken:
        return
    ax.axvspan(min(broken) / 1.4, ax.get_xlim()[1], color=AXIS, alpha=0.13, zorder=0)
    ax.annotate("peers > 26\n(stencil > tile)", xy=(min(broken), 0.97),
                xycoords=("data", "axes fraction"), xytext=(4, -4), textcoords="offset points",
                ha="left", va="top", fontsize=7, color=INK_MUTED, linespacing=1.4)


def plot_speedup(curve, outdir, dpi):
    s = scaling_series(curve)
    w, p0 = s["workers"], s["baseline"]
    fig, (ax_sp, ax_eff) = plt.subplots(1, 2, figsize=(11, 4.8), facecolor=SURFACE)

    ax_sp.plot(w, s["ideal"], "--", color=AXIS, lw=1.2, label=f"ideal (linear from {p0} GPUs)")
    ax_sp.plot(w, s["speedup_nocomm"], "^-", color=SERIES[2], label="exchange removed", **MARK)
    ax_sp.plot(w, s["speedup"], "o-", color=SERIES[0], label="measured", **MARK)
    label_end(ax_sp, w[-1], s["speedup"][-1], f"{s['speedup'][-1]:.1f}x", SERIES[0])
    label_end(ax_sp, w[-1], s["speedup_nocomm"][-1], f"{s['speedup_nocomm'][-1]:.1f}x",
              SERIES[2], dy=11)
    style_axes(ax_sp, w)
    ax_sp.set_yscale("log", base=2)
    ax_sp.set_ylabel(f"Speedup vs {p0} GPUs", fontsize=10, color=INK_SECONDARY)
    ax_sp.set_title("Speedup (rest of the ticks)", fontsize=11, color=INK, pad=8)
    ax_sp.legend(fontsize=8, frameon=False, labelcolor=INK_SECONDARY, loc="upper left")
    shade_breakdown(ax_sp, curve)

    ax_eff.axhline(100, color=AXIS, ls="--", lw=1.2, label="ideal (100%)")
    ax_eff.plot(w, s["eff_nocomm"], "^-", color=SERIES[2], label="exchange removed", **MARK)
    ax_eff.plot(w, s["eff"], "o-", color=SERIES[0], label="measured", **MARK)
    label_end(ax_eff, w[-1], s["eff"][-1], f"{s['eff'][-1]:.0f}%", SERIES[0])
    label_end(ax_eff, w[-1], s["eff_nocomm"][-1], f"{s['eff_nocomm'][-1]:.0f}%", SERIES[2], dy=11)
    style_axes(ax_eff, w)
    ax_eff.set_ylabel("Parallel efficiency (%)", fontsize=10, color=INK_SECONDARY)
    ax_eff.set_ylim(0, 118)
    ax_eff.set_title(f"Parallel efficiency (baseline {p0} GPUs)", fontsize=11, color=INK, pad=8)
    ax_eff.legend(fontsize=8, frameon=False, labelcolor=INK_SECONDARY, loc="lower left")
    shade_breakdown(ax_eff, curve)

    r0 = curve[0]
    fig.suptitle("Strong scaling — 3D torus", fontsize=13, color=INK, x=0.02, ha="left", y=0.98)
    fig.text(0.02, 0.915,
             f"{r0['total_neurons']:,} neurons fixed · in-degree K={r0['in_degree']} · "
             f"{r0['ticks']} ticks · work per GPU falls "
             f"{r0['neurons_per_worker']:,} → {curve[-1]['neurons_per_worker']:,} neurons",
             fontsize=9, color=INK_MUTED, ha="left")
    fig.text(0.02, 0.025,
             f"Metric is the mean of ticks {WARMUP_TICKS + 1}..N. The first tick is excluded because it "
             "constructs the network (GPU buffer build + ghost discovery).\n"
             "“Exchange removed” uses each point's own measured compute time, against its own "
             "baseline.",
             fontsize=7.5, color=INK_MUTED, ha="left", linespacing=1.5)
    fig.tight_layout(rect=[0, 0.06, 1, 0.89])
    path = outdir / "strong_3d_speedup.png"
    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_breakdown(curve, outdir, dpi):
    w = [r["workers"] for r in curve]
    x = range(len(w))
    fig, (ax_share, ax_abs, ax_comm) = plt.subplots(1, 3, figsize=(15, 5.6), facecolor=SURFACE)

    def legend_below(ax, ncol):
        ax.legend(fontsize=8, frameon=False, labelcolor=INK_SECONDARY, ncol=ncol,
                  loc="upper center", bbox_to_anchor=(0.5, -0.15), handlelength=1.6,
                  columnspacing=1.4)

    # 100% stack: absolute times span decades, so the SHARE is what stays readable at every P.
    bottom = [0.0] * len(curve)
    parts = STEP_PARTS + [("other", "other")]
    for color, (key, lab) in zip(SERIES, parts):
        vals = [r[f"{key}_s"] / r["step_s"] * 100 if r["step_s"] else 0 for r in curve]
        ax_share.bar(x, vals, bottom=bottom, color=color, label=lab, width=0.72,
                     edgecolor=SURFACE, linewidth=1.5)   # 2px surface gap between segments
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax_share.set_xticks(list(x))
    ax_share.set_xticklabels([str(v) for v in w], fontsize=8)
    ax_share.set_ylim(0, 100)
    ax_share.set_ylabel("Share of step time (%)", fontsize=10, color=INK_SECONDARY)
    ax_share.set_xlabel("Workers (GPUs)", fontsize=10, color=INK_SECONDARY)
    ax_share.set_title("Where the step time goes", fontsize=11, color=INK, pad=8)
    legend_below(ax_share, 3)

    # Lines, not a stack: on a log axis stacked heights are not readable as values.
    step = [r["step_s"] for r in curve]
    floor = max(step) * 1e-4
    drawn = []
    for color, (key, lab) in zip(SERIES, parts):
        vals = [r[f"{key}_s"] for r in curve]
        if max(vals) < floor:
            continue
        ax_abs.plot(w, [max(v, floor) for v in vals], "o-", color=color, label=lab, **MARK)
        drawn += [v for v in vals if v > 0]
    ax_abs.plot(w, step, "--", color=AXIS, lw=1.2, label="step total")
    style_axes(ax_abs, w)
    ax_abs.set_yscale("log")
    ax_abs.set_ylim(min(drawn) / 3, max(step) * 3)
    ax_abs.set_ylabel("Time per step (s)", fontsize=10, color=INK_SECONDARY)
    ax_abs.set_title("Absolute cost per component", fontsize=11, color=INK, pad=8)
    legend_below(ax_abs, 3)

    cvals = []
    for color, (key, lab) in zip(BLUE_RAMP, COMM_PARTS + [("comm_other", "other")]):
        vals = [r[f"{key}_s"] for r in curve]
        ax_comm.plot(w, [max(v, 1e-9) for v in vals], "o-", color=color, label=lab, **MARK)
        cvals += [v for v in vals if v > 0]
    wait = [r["wait_s"] for r in curve]
    ax_comm.plot(w, wait, "s--", color=SERIES[1], label="wait (inside exchange)", **MARK)
    style_axes(ax_comm, w)
    ax_comm.set_yscale("log")
    ax_comm.set_ylim(min(cvals + wait) / 3, max(cvals + wait) * 3)
    ax_comm.set_ylabel("Time per step (s)", fontsize=10, color=INK_SECONDARY)
    ax_comm.set_title("Inside the ghost exchange", fontsize=11, color=INK, pad=8)
    legend_below(ax_comm, 3)

    for ax in (ax_abs, ax_comm):
        shade_breakdown(ax, curve)

    fig.suptitle("Strong scaling — per-step decomposition", fontsize=13, color=INK,
                 x=0.02, ha="left", y=0.98)
    fig.text(0.02, 0.915, f"Per-rank means over ticks {WARMUP_TICKS + 1}..N.",
             fontsize=9, color=INK_MUTED, ha="left")
    fig.text(0.02, 0.025,
             "Panels A and B show one disjoint level that sums to the step; panel C opens the "
             "ghost-exchange bar.\n"
             "Wait is a subset of MPI exchange — it separates latency from load imbalance — so "
             "it is never added into the stack.",
             fontsize=7.5, color=INK_MUTED, ha="left", linespacing=1.5)
    fig.tight_layout(rect=[0, 0.10, 1, 0.90])
    path = outdir / "strong_3d_breakdown.png"
    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_comm(curve, outdir, dpi):
    w = [r["workers"] for r in curve]
    fig, (ax_peers, ax_ratio, ax_bytes) = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=SURFACE)

    peers = [r["peers_mean"] for r in curve]
    ax_peers.axhline(MOORE_PEERS, color=AXIS, ls="--", lw=1.2, label="26 = 3³−1 (full Moore)")
    ax_peers.plot(w, peers, "o-", color=SERIES[0], **MARK)
    label_end(ax_peers, w[-1], peers[-1], f"{peers[-1]:.0f}", SERIES[0])
    ax_peers.set_ylabel("Peer ranks", fontsize=10, color=INK_SECONDARY)
    ax_peers.set_title("MPI peers per rank", fontsize=11, color=INK, pad=8)
    ax_peers.legend(fontsize=8, frameon=False, labelcolor=INK_SECONDARY, loc="upper left")
    ax_peers.set_ylim(bottom=0)

    ratio = [r["ghost_local_ratio"] for r in curve]
    ax_ratio.axhline(1.0, color=AXIS, ls="--", lw=1.2, label="halo = local (1:1)")
    ax_ratio.plot(w, ratio, "o-", color=SERIES[1], **MARK)
    label_end(ax_ratio, w[-1], ratio[-1], f"{ratio[-1]:,.0f}:1", SERIES[1])
    ax_ratio.set_yscale("log")
    ax_ratio.set_ylabel("Ghost somas per local soma", fontsize=10, color=INK_SECONDARY)
    ax_ratio.set_title("Surface-to-volume ratio", fontsize=11, color=INK, pad=8)
    ax_ratio.legend(fontsize=8, frameon=False, labelcolor=INK_SECONDARY, loc="upper left")

    per_peer = [r["bytes_per_peer"] for r in curve]
    ax_bytes.plot(w, per_peer, "o-", color=SERIES[2], **MARK)
    label_end(ax_bytes, w[-1], per_peer[-1], f"{per_peer[-1]:,.0f} B", SERIES[2])
    ax_bytes.set_yscale("log")
    ax_bytes.set_ylabel("Bytes per peer per step", fontsize=10, color=INK_SECONDARY)
    ax_bytes.set_title("Message size", fontsize=11, color=INK, pad=8)

    for ax in (ax_peers, ax_ratio, ax_bytes):
        style_axes(ax, w)
        shade_breakdown(ax, curve)

    fig.suptitle("Strong scaling — why it saturates", fontsize=13, color=INK,
                 x=0.02, ha="left", y=0.98)
    fig.text(0.02, 0.915, f"Per-rank means over ticks {WARMUP_TICKS + 1}..N.",
             fontsize=9, color=INK_MUTED, ha="left")
    fig.text(0.02, 0.025,
             "Local volume falls as 1/P but the halo surface only as P^(−2/3), so the "
             "ghost:local ratio climbs and the exchange takes a\n"
             "growing share of a shrinking step. Once a message is small enough its cost is "
             "latency, not bytes.",
             fontsize=7.5, color=INK_MUTED, ha="left", linespacing=1.5)
    fig.tight_layout(rect=[0, 0.06, 1, 0.89])
    path = outdir / "strong_3d_comm.png"
    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)
    return path


def print_table(curve):
    s = scaling_series(curve)
    r0 = curve[0]
    print(f"\nN={r0['total_neurons']:,} fixed, K={r0['in_degree']}, {r0['ticks']} ticks, "
          f"baseline {s['baseline']} GPUs")
    hdr = (f"{'P':>5} {'nodes':>6} {'n/GPU':>8} {'step(ms)':>10} {'median':>10} {'m-md%':>7} "
           f"{'first_tick(s)':>14} {'speedup':>8} {'eff':>7} {'peers':>6} "
           f"{'gh:loc':>8} {'comm%':>7} {'B/peer':>10}")
    print(hdr)
    print("-" * len(hdr))
    for r, sp, ef in zip(curve, s["speedup"], s["eff"]):
        comm_pct = r["comm_s"] / r["step_s"] * 100 if r["step_s"] else 0
        # The mean/median gap is the window check: a wide-enough window makes them agree.
        gap = (r["step_s"] - r["step_median_s"]) / r["step_median_s"] * 100
        print(f"{r['workers']:>5} {r['nodes']:>6} {r['neurons_per_worker']:>8,} "
              f"{r['step_s'] * 1e3:>10.3f} {r['step_median_s'] * 1e3:>10.3f} {gap:>+6.2f}% "
              f"{r['first_tick_s']:>14.2f} {sp:>7.2f}x {ef:>6.1f}% {r['peers_mean']:>6.0f} "
              f"{r['ghost_local_ratio']:>8.2f} {comm_pct:>6.1f}% {r['bytes_per_peer']:>10,.0f}")

    print("\nper-step decomposition (s), level 1 disjoint:")
    hdr2 = f"{'P':>5} " + " ".join(f"{lab:>15}" for _, lab in STEP_PARTS) + f"{'other':>15}"
    print(hdr2)
    print("-" * len(hdr2))
    for r in curve:
        print(f"{r['workers']:>5} " + " ".join(f"{r[f'{k}_s']:>15.6f}" for k, _ in STEP_PARTS)
              + f"{r['other_s']:>15.6f}")


def write_final(path, curve):
    fields = ["total_neurons", "in_degree", "workers", "nodes", "neurons_per_worker", "synapses",
              "ticks", "n_runs", "warmup_ticks", "n_step_ticks",
              "first_tick_s", "step_s", "step_median_s",
              "comm_s", "compute_s", "gpu_sync_s", "write_back_s", "other_s",
              "pack_s", "exchange_s", "unpack_s", "comm_other_s", "wait_s",
              "generation_time", "model_creation_time", "gpu_setup_time", "construction_time",
              "simulation_time", "total_time",
              "peers_mean", "ghost_somas_mean", "ghost_local_ratio", "send_bytes_mean",
              "bytes_per_peer"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(curve)
    return len(curve)


def main(argv=None):
    args = parse_args(argv)
    inputs = args.inputs or sorted(OUTPUTS.glob("strong_3d_curve_*.csv"))

    print("Consolidating run summaries (read-only; the raw files are append-only provenance):")
    canonical = read_summaries(inputs, args.ticks)

    print(f"\nDeriving timings from per-tick files ({args.ticks_dir}); tick 1 is construction, "
          f"the step is the mean of ticks {args.warmup_ticks + 1}..N:")
    curve = build_curve(canonical, args.ticks_dir, args.warmup_ticks)
    if not curve:
        raise SystemExit("no points with per-tick data -- nothing to analyse")

    present = [r["workers"] for r in curve]
    missing = [w for w in EXPECTED_WORKERS if w not in present]
    print("\nCoverage:")
    print(f"  {len(present)}/{len(EXPECTED_WORKERS)} points -- "
          + ("complete" if not missing else f"MISSING {missing}"))
    if present and present[0] != EXPECTED_WORKERS[0]:
        print(f"  NOTE: baseline P={EXPECTED_WORKERS[0]} absent; speedup is read against "
              f"P={present[0]} instead")

    n = write_final(args.output, curve)
    print(f"\nWrote {n} rows to {args.output}")
    print_table(curve)

    args.outdir.mkdir(parents=True, exist_ok=True)
    written = [plot_speedup(curve, args.outdir, args.dpi),
               plot_breakdown(curve, args.outdir, args.dpi),
               plot_comm(curve, args.outdir, args.dpi)]
    print("\nFigures:")
    for p in written:
        print(f"  {p}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
