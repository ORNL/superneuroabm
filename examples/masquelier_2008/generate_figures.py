"""
Generate all figures from the Masquelier 2008 replication,
matching the author's Hathway-Goodman-2018 figure style.

Figures generated (from a single simulation run):
  - Figure 3: Postsynaptic spike latency vs discharge number
  - Figure 4: Weighted spike raster during a pattern window
  - Figure 7AB: Average weight evolution over time
  - Figure 9 (sup): Membrane potential trace at 3 time windows
  - Figure 10 (sup): Input spike raster + firing rate histogram

Usage:
    python generate_figures.py              # HG-LIF (default)
    python generate_figures.py --model hg   # HG-LIF
    python generate_figures.py --model srm  # SRM
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiment_utils import (
    N_AFFERENTS, N_PATTERN, PATTERN_DURATION, PATTERN_FRACTION,
    TOTAL_SECONDS, SEGMENT_SECONDS, DT, TOTAL_TICKS, CHUNK_SIZE,
    SEED, JITTER_STD,
    generate_rate_profiles, generate_poisson_spikes,
    generate_pattern, embed_pattern, tile_spike_trains,
    get_spikes_in_range,
)

# Model variant configuration (set in main() from command-line args)
MODEL_VARIANT = "hg"  # default to hg

MODEL_CONFIGS = {
    "srm": {
        "output_dir": "output_srm",
        "theta": 731.21,
        "soma_breed": "srm_soma",
        "soma_config": "masquelier_srm_config_0",
        "synapse_breed": "double_exp_synapse",
        "synapse_config": "masquelier_srm_config_0",
        "dt_sim": 1e-3,        # SRM uses 1ms ticks
        "tick_scale": 1,
        "pattern_duration_ticks": 50,   # 50ms / 1ms
    },
    "hg": {
        "output_dir": "output_hg",
        "theta": 500.0,
        "soma_breed": "hg_lif_soma",
        "soma_config": "masquelier_hg_config_0",
        "synapse_breed": "single_exp_synapse",
        "synapse_config": "masquelier_config_0",
        "dt_sim": 1e-4,        # HG-LIF uses 0.1ms ticks
        "tick_scale": 10,
        "pattern_duration_ticks": 500,  # 50ms / 0.1ms
    },
}

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / MODEL_CONFIGS[MODEL_VARIANT]["output_dir"]
THETA = MODEL_CONFIGS[MODEL_VARIANT]["theta"]

# Tick resolution and pattern duration in native ticks (set in main())
DT_SIM = MODEL_CONFIGS[MODEL_VARIANT]["dt_sim"]
TICK_SCALE = MODEL_CONFIGS[MODEL_VARIANT]["tick_scale"]
PAT_DUR_TICKS = MODEL_CONFIGS[MODEL_VARIANT]["pattern_duration_ticks"]


def compute_find_t(spike_times_ticks, pattern_onsets_ticks, window=2.0):
    """Compute pattern finding time: first time FA rate drops to 0 Hz.

    Uses a sliding window to find the earliest time when the false alarm
    rate is zero and hit rate is above 80%.
    """
    spike_s = spike_times_ticks * DT_SIM
    onset_s = pattern_onsets_ticks * DT_SIM

    for t0 in np.arange(0, TOTAL_SECONDS, window):
        t1 = t0 + window
        seg_onsets = onset_s[(onset_s >= t0) & (onset_s < t1)]
        seg_spikes_s = spike_s[(spike_s >= t0) & (spike_s < t1)]
        seg_spikes_t = spike_times_ticks[(spike_s >= t0) & (spike_s < t1)]

        if len(seg_onsets) == 0:
            continue

        # Hit rate
        hits = 0
        for onset in seg_onsets:
            end = onset + PAT_DUR_TICKS * DT_SIM
            lo = np.searchsorted(seg_spikes_s, onset, side="left")
            hi = np.searchsorted(seg_spikes_s, end, side="left")
            if hi > lo:
                hits += 1
        hr = hits / len(seg_onsets)

        # False alarm rate
        spike_slots = seg_spikes_t // PAT_DUR_TICKS
        onset_slots = pattern_onsets_ticks[
            (onset_s >= t0) & (onset_s < t1)
        ] // PAT_DUR_TICKS
        in_pat = np.isin(spike_slots, onset_slots)
        n_fa = np.sum(~in_pat)
        non_pat_dur = window - len(seg_onsets) * PAT_DUR_TICKS * DT_SIM
        fa_hz = n_fa / non_pat_dur if non_pat_dur > 0 else 0

        if fa_hz < 0.5 and hr >= 0.8:
            return t0

    return 30.0  # fallback


def record_potential_windows(spike_trains, snap_times, snap_weights,
                             pattern_onsets_ticks, find_t):
    """Re-run 3 short (1s) simulation windows to record membrane potential.

    Returns dict of {label: (time_array_s, v_array)}.
    """
    cache_file = OUTPUT_DIR / "potential_traces.npz"
    if cache_file.exists():
        data = np.load(cache_file)
        return {
            "early": (data["early_t"], data["early_v"]),
            "mid": (data["mid_t"], data["mid_v"]),
            "late": (data["late_t"], data["late_v"]),
        }

    from superneuroabm.model import NeuromorphicModel

    ticks_per_sec = int(1.0 / DT_SIM)
    windows = [
        ("early", 0, ticks_per_sec),
        ("mid", int((find_t - 0.25) / DT_SIM), int((find_t + 0.75) / DT_SIM)),
        ("late", (TOTAL_SECONDS - 1) * ticks_per_sec, TOTAL_SECONDS * ticks_per_sec),
    ]

    potentials = {}
    save_data = {}

    for label, start_tick, end_tick in windows:
        chunk_ticks = end_tick - start_tick
        t_arr = np.arange(start_tick, end_tick) * DT_SIM  # time in seconds

        # Find the closest prior weight snapshot
        snap_idx = np.searchsorted(snap_times, start_tick, side="right") - 1
        if snap_idx < 0:
            weights = np.full(N_AFFERENTS, 0.475, dtype=np.float32)
        else:
            weights = snap_weights[snap_idx]

        print(f"  Recording {label} window: t={start_tick*DT:.2f}-{end_tick*DT:.2f}s "
              f"(weights from snap {snap_idx}) ...")

        cfg = MODEL_CONFIGS[MODEL_VARIANT]
        model = NeuromorphicModel(user_config=BASE_DIR / "masquelier_config.yaml", enable_internal_state_tracking=True)
        model.set_global_property_value("dt", DT_SIM)

        soma = model.create_soma(
            breed=cfg["soma_breed"], config_name=cfg["soma_config"]
        )
        synapses = []
        for i in range(N_AFFERENTS):
            syn = model.create_synapse(
                breed=cfg["synapse_breed"],
                pre_soma_id=-1,
                post_soma_id=soma,
                config_name=cfg["synapse_config"],
            )
            synapses.append(syn)

        model.setup(use_gpu=True)

        # Set weights
        for i, syn_id in enumerate(synapses):
            hp = model.get_agent_property_value(
                id=syn_id, property_name="hyperparameters"
            )
            hp[0] = float(weights[i])
            model.set_agent_property_value(syn_id, "hyperparameters", hp)

        # Load input spikes
        for i, syn_id in enumerate(synapses):
            chunk_spikes = get_spikes_in_range(
                spike_trains[i], start_tick, end_tick
            )
            if len(chunk_spikes) > 0:
                local_ticks = chunk_spikes - start_tick
                flat = np.empty(len(local_ticks) * 2)
                flat[0::2] = local_ticks
                flat[1::2] = 1.0
                existing = model.get_agent_property_value(
                    id=syn_id, property_name="input_spikes_tensor"
                )
                existing.extend(flat.tolist())
                model.set_agent_property_value(
                    syn_id, "input_spikes_tensor", existing
                )

        model.simulate(ticks=chunk_ticks, update_data_ticks=chunk_ticks)

        # Extract membrane potential
        states = np.array(model.get_internal_states_history(agent_id=soma))
        v_trace = states[:, 0]

        potentials[label] = (t_arr, v_trace)
        save_data[f"{label}_t"] = t_arr
        save_data[f"{label}_v"] = v_trace

        print(f"    v range: [{v_trace.min():.1f}, {v_trace.max():.1f}]")

        del model

    np.savez(cache_file, **save_data)
    return potentials


def load_simulation_data():
    """Load saved simulation results."""
    spike_times = np.load(OUTPUT_DIR / "spike_times.npy")
    data = np.load(OUTPUT_DIR / "weight_snapshots.npz")
    snap_times = data["times"]
    snap_weights = data["weights"]
    pattern_onsets = np.load(OUTPUT_DIR / "pattern_onsets.npy")
    return spike_times, snap_times, snap_weights, pattern_onsets


def regenerate_spike_trains():
    """Regenerate input spike trains using the same seed as the experiment."""
    rng = np.random.default_rng(SEED)
    segment_ticks = int(SEGMENT_SECONDS / DT)
    SUB_SEG = 30000

    spike_trains = [np.array([], dtype=np.int64) for _ in range(N_AFFERENTS)]
    for sub_start in range(0, segment_ticks, SUB_SEG):
        sub_end = min(sub_start + SUB_SEG, segment_ticks)
        sub_len = sub_end - sub_start
        rates_sub = generate_rate_profiles(N_AFFERENTS, sub_len, rng)
        trains_sub = generate_poisson_spikes(rates_sub, rng)
        for i in range(N_AFFERENTS):
            spike_trains[i] = np.concatenate([
                spike_trains[i], trains_sub[i] + sub_start
            ])
        del rates_sub, trains_sub

    pattern_rng = np.random.default_rng(SEED + 1000)
    pattern_spikes = generate_pattern(N_PATTERN, PATTERN_DURATION, pattern_rng)
    embed_rng = np.random.default_rng(SEED + 2000)
    spike_trains, pattern_onsets = embed_pattern(
        spike_trains, pattern_spikes, segment_ticks, embed_rng
    )
    spike_trains = tile_spike_trains(spike_trains, segment_ticks, TOTAL_SECONDS // SEGMENT_SECONDS)

    # Scale to native tick resolution if needed (HG-LIF uses 0.1ms ticks)
    if TICK_SCALE > 1:
        for i in range(len(spike_trains)):
            spike_trains[i] = spike_trains[i] * TICK_SCALE

    return spike_trains


def compute_latencies(spike_times_ticks, pattern_onsets_ticks):
    """Compute spike latency relative to pattern onset for each output spike."""
    spike_times_s = spike_times_ticks * DT_SIM
    pattern_onsets_s = pattern_onsets_ticks * DT_SIM
    pattern_end_s = pattern_onsets_s + PAT_DUR_TICKS * DT_SIM

    latencies = []  # (onset_time_s, latency_ms) for each pattern presentation
    for onset_s, end_s in zip(pattern_onsets_s, pattern_end_s):
        lo = np.searchsorted(spike_times_s, onset_s, side="left")
        hi = np.searchsorted(spike_times_s, end_s, side="left")
        if hi > lo:
            lat_ms = (spike_times_s[lo] - onset_s) * 1000
            latencies.append((onset_s, lat_ms))
        else:
            latencies.append((onset_s, np.nan))

    # Per-spike latency: for each output spike, find the nearest preceding pattern onset
    spike_latencies_ms = []
    for st in spike_times_s:
        idx = np.searchsorted(pattern_onsets_s, st, side="right") - 1
        if idx >= 0:
            onset = pattern_onsets_s[idx]
            end = onset + PAT_DUR_TICKS * DT_SIM
            if onset <= st < end:
                spike_latencies_ms.append((st - onset) * 1000)
            else:
                spike_latencies_ms.append(np.nan)  # false alarm
        else:
            spike_latencies_ms.append(np.nan)

    return latencies, np.array(spike_latencies_ms)


def fig_3(spike_latencies_ms):
    """Figure 3: Postsynaptic spike latency vs discharge number.

    Matches author's fig_3(): green dots, latency in ms on y-axis,
    discharge count on x-axis.
    """
    fig = plt.figure(figsize=(6, 5))
    plt.suptitle("Figure 3", fontsize=17)
    plt.title("Latency vs output neuron spikes", fontsize=17)

    # All spikes (including false alarms shown as NaN = off the chart)
    valid = ~np.isnan(spike_latencies_ms)
    plt.plot(np.arange(len(spike_latencies_ms)), spike_latencies_ms, "g.",
             markersize=3)
    plt.ylabel("Postsynaptic spike latency [ms]", fontsize=17)
    plt.xlabel("# discharge", fontsize=17)
    plt.yticks([0, 10, 20, 30, 40, 50])
    plt.ylim([-2, 52])
    plt.tick_params(labelsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(OUTPUT_DIR / "figure_3.png", dpi=150)
    plt.close()
    print("  Figure 3 saved.")


def fig_4(spike_trains, pattern_onsets_ticks, final_weights):
    """Figure 4: Weighted spike raster during a pattern window.

    Shows input spikes around one of the last pattern presentations,
    color-coded by synaptic weight (white = high weight, black = low).
    Black background, blue rectangle around pattern window.
    """
    # Use second-to-last pattern presentation (like the author)
    onset_tick = pattern_onsets_ticks[-2]
    onset_s = onset_tick * DT_SIM
    window_start = onset_s - 0.025  # 25ms before
    window_end = onset_s + 0.075    # 75ms after
    start_tick = int(window_start / DT_SIM)
    end_tick = int(window_end / DT_SIM)

    # Collect all input spikes in this window
    all_times = []
    all_afferents = []
    all_weights = []
    for i in range(N_AFFERENTS):
        spk = get_spikes_in_range(spike_trains[i], start_tick, end_tick)
        if len(spk) > 0:
            all_times.extend(spk * DT_SIM)
            all_afferents.extend([i] * len(spk))
            all_weights.extend([final_weights[i]] * len(spk))

    all_times = np.array(all_times)
    all_afferents = np.array(all_afferents)
    all_weights = np.array(all_weights)

    # Grayscale: high weight = white, low weight = black (invert for display)
    greyness = 1.0 - all_weights  # 0 = white (high w), 1 = black (low w)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor("k")
    sc = ax.scatter(all_times, all_afferents, c=greyness, s=3, lw=0,
                    cmap="Greys", vmin=0, vmax=1)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.ax.invert_yaxis()
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.set_ticklabels(["1.0", "0.8", "0.6", "0.4", "0.2", "0.0"])
    cbar.set_label("synaptic weight", fontsize=14)

    # Blue rectangle around pattern window for pattern afferents
    rect = mpatches.Rectangle(
        (onset_s - 0.001, 0), PAT_DUR_TICKS * DT_SIM, N_PATTERN,
        linewidth=2, edgecolor="b", facecolor="none"
    )
    ax.add_patch(rect)

    ax.set_title("Figure 4", fontsize=17)
    ax.set_xlabel("time [s]", fontsize=17)
    ax.set_ylabel("# afferent", fontsize=17)
    ax.set_xlim([window_start, window_end])
    ax.set_ylim([0, N_AFFERENTS])
    ax.tick_params(labelsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(OUTPUT_DIR / "figure_4.png", dpi=150)
    plt.close()
    print("  Figure 4 saved.")


def fig_7AB(snap_times, snap_weights):
    """Figure 7AB: Average weight evolution over time.

    Left panel: first 1 second. Right panel: whole simulation.
    Author shows RNN, NN, ATA lines; we only have RNN so single line.
    """
    times_s = snap_times * DT_SIM
    mean_weights = np.mean(snap_weights, axis=1)

    fig = plt.figure(figsize=(12, 5))
    plt.suptitle("Figure 7 A and B", fontsize=17)

    # Left: early period (adapt to snapshot resolution)
    early_end = min(TOTAL_SECONDS / 3, times_s[2]) if len(times_s) > 2 else TOTAL_SECONDS / 3
    ax1 = plt.subplot(121)
    ax1.set_title(f"first {early_end:.0f}s", fontsize=17)
    mask = times_s <= early_end
    ax1.plot(times_s[mask], mean_weights[mask], "k-o", linewidth=2, markersize=4, label="RNN")
    ax1.set_ylabel("average weight per synapse", fontsize=17)
    ax1.set_xlabel("time [s]", fontsize=17)
    ax1.set_ylim([0.15, 0.55])
    ax1.set_yticks([0.2, 0.3, 0.4, 0.5])
    ax1.set_xlim([0, early_end])
    ax1.tick_params(labelsize=14)

    # Right: whole simulation
    ax2 = plt.subplot(122)
    ax2.set_title("whole simulation", fontsize=17)
    ax2.plot(times_s, mean_weights, "k", linewidth=2, label="RNN")
    ax2.set_xlabel("time [s]", fontsize=17)
    ax2.set_ylim([0.15, 0.55])
    ax2.set_xticks(np.linspace(1, TOTAL_SECONDS, min(4, max(2, TOTAL_SECONDS // 30)), dtype=int))
    ax2.set_yticks([0.2, 0.3, 0.4, 0.5])
    ax2.set_xlim([1, TOTAL_SECONDS])
    ax2.legend(loc="upper right", fontsize=16)
    ax2.tick_params(labelsize=14)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(OUTPUT_DIR / "figure_7AB.png", dpi=150)
    plt.close()
    print("  Figure 7AB saved.")


def fig_9(potentials, pattern_onsets_ticks, find_t):
    """Figure 9 (supplemental): Membrane potential trace at 3 time windows.

    Matches the author's fig_9(): plots the actual membrane potential as a
    blue line, with threshold (red dashed) and resting potential (black dashed).
    Green shaded areas show pattern windows, red vertical dashed lines show
    pattern onsets.

    Three 1-second windows:
      Row 1: Early (0-1s)
      Row 2: Mid-simulation (around find_t)
      Row 3: Late (449-450s)
    """
    plt.rcParams["lines.linewidth"] = 1.0
    pattern_onsets_s = pattern_onsets_ticks * DT_SIM

    xlim_starts = [0, find_t - 0.25, TOTAL_SECONDS - 1]
    titles = [
        "Early (0 – 1 s)",
        f"Mid ({find_t-0.25:.1f} – {find_t+0.75:.1f} s)",
        f"Late ({TOTAL_SECONDS-1} – {TOTAL_SECONDS} s)",
    ]
    labels = ["early", "mid", "late"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=False)
    fig.suptitle("Figure 9 (supplemental)", fontsize=17)

    for idx, (ax, start, title, label) in enumerate(
        zip(axes, xlim_starts, titles, labels)
    ):
        end = start + 1.0
        t_arr, v_arr = potentials[label]

        # Plot membrane potential trace
        ax.plot(t_arr, v_arr, "b", label="potential")

        # Threshold and resting potential
        ax.axhline(y=THETA, color="r", linestyle="--", label="threshold")
        ax.axhline(y=0, color="k", linestyle="--", label="resting pot.")

        # Pattern windows and onsets
        onsets_in_window = pattern_onsets_s[
            (pattern_onsets_s >= start) & (pattern_onsets_s < end)
        ]
        for onset in onsets_in_window:
            ax.axvline(onset, ls="--", c="r", linewidth=2)
            ax.axvspan(onset, onset + PAT_DUR_TICKS * DT_SIM,
                       facecolor="g", alpha=0.3)

        ax.set_xlim([start, end])
        ax.set_yticks([0, int(THETA), int(THETA * 2)])
        ax.set_ylim([-THETA * 0.6, THETA * 2.2])
        ax.set_ylabel("Potential [a.u.]", fontsize=14)
        ax.set_title(title, fontsize=14)
        if idx == 2:
            ax.set_xlabel("t [s]", fontsize=17)

    axes[0].legend(loc="upper right", fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.savefig(OUTPUT_DIR / "figure_9_sup.png", dpi=150)
    plt.close()
    print("  Figure 9 (sup) saved.")


def fig_10(spike_trains, pattern_onsets_ticks):
    """Figure 10 (supplemental): Input spike raster + firing rate histogram.

    Shows 50 pattern afferents and 50 non-pattern afferents over a 0.5s window.
    Pattern spikes highlighted in red.
    """
    pattern_onsets_s = pattern_onsets_ticks * DT_SIM
    n_inds2plot = 50
    patternlength = PAT_DUR_TICKS * DT_SIM  # 0.05 s

    # Show 0.5s window starting just before first pattern
    starttime = pattern_onsets_s[0] - patternlength
    stoptime = starttime + 0.5
    start_tick = int(starttime / DT_SIM)
    end_tick = int(stoptime / DT_SIM)

    # Collect spikes for first 50 pattern afferents
    times_pat50 = []
    inds_pat50 = []
    for i in range(n_inds2plot):
        spk = get_spikes_in_range(spike_trains[i], start_tick, end_tick)
        if len(spk) > 0:
            times_pat50.extend(spk * DT_SIM)
            inds_pat50.extend([i] * len(spk))
    times_pat50 = np.array(times_pat50)
    inds_pat50 = np.array(inds_pat50)

    # Collect spikes for 50 non-pattern afferents (1000-1049)
    times_npat50 = []
    inds_npat50 = []
    for i in range(N_PATTERN, N_PATTERN + n_inds2plot):
        spk = get_spikes_in_range(spike_trains[i], start_tick, end_tick)
        if len(spk) > 0:
            times_npat50.extend(spk * DT_SIM)
            inds_npat50.extend([(i - N_PATTERN) + n_inds2plot] * len(spk))
    times_npat50 = np.array(times_npat50)
    inds_npat50 = np.array(inds_npat50)

    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(4, 4)
    fig.suptitle("Figure 10 (supplemental): Input spike raster", fontsize=17)

    # Main raster plot
    ax_main = fig.add_subplot(gs[0:3, 0:3])
    if len(times_pat50) > 0:
        ax_main.plot(times_pat50, inds_pat50, "b.", markersize=3)
    if len(times_npat50) > 0:
        ax_main.plot(times_npat50, inds_npat50, "b.", markersize=3)

    # Highlight pattern spikes in red
    onsets_in_window = pattern_onsets_s[
        (pattern_onsets_s >= starttime) & (pattern_onsets_s < stoptime)
    ]
    for onset in onsets_in_window:
        mask = ((times_pat50 >= onset) & (times_pat50 < onset + patternlength))
        if np.any(mask):
            ax_main.plot(times_pat50[mask], inds_pat50[mask], "r.", markersize=3)

    ax_main.set_ylabel("# afferent", fontsize=14)
    ax_main.set_xlim([starttime, stoptime])
    ax_main.set_ylim([-1, 2 * n_inds2plot])
    ax_main.set_xticks([])
    ax_main.tick_params(labelsize=12)

    # Right: firing rate bar chart
    ax_right = fig.add_subplot(gs[0:3, 3])
    if len(inds_pat50) > 0:
        uniq_p, counts_p = np.unique(inds_pat50, return_counts=True)
        rates_p = counts_p / 0.5  # Hz (0.5s window)
        ax_right.barh(uniq_p + 0.5, rates_p, height=0.8, color="r", alpha=0.7)
    if len(inds_npat50) > 0:
        uniq_n, counts_n = np.unique(inds_npat50, return_counts=True)
        rates_n = counts_n / 0.5
        ax_right.barh(uniq_n + 0.5, rates_n, height=0.8, color="b", alpha=0.7)
    ax_right.set_xlabel("Rate [Hz]", fontsize=12)
    ax_right.set_ylim([-1, 2 * n_inds2plot])
    ax_right.set_xticks([0, 50, 100])
    ax_right.set_yticks([])
    ax_right.tick_params(labelsize=12)

    # Bottom: time histogram
    ax_bot = fig.add_subplot(gs[3, 0:3])
    all_times = np.concatenate([times_pat50, times_npat50]) if len(times_npat50) > 0 else times_pat50
    if len(all_times) > 0:
        bin_width = 1e-3  # 1ms bins regardless of tick resolution
        counts_hist, bins = np.histogram(all_times, bins=int(0.5 / bin_width))
        bin_centers = (bins[:-1] + bins[1:]) / 2
        ax_bot.bar(bin_centers, counts_hist / (2 * n_inds2plot * bin_width),
                   width=bin_width * 0.9, color="b", alpha=0.7)
    ax_bot.set_xlabel("time [s]", fontsize=14)
    ax_bot.set_ylabel("Rate [Hz]", fontsize=12)
    ax_bot.set_xlim([starttime, stoptime])
    ax_bot.set_ylim([0, 100])
    ax_bot.tick_params(labelsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.savefig(OUTPUT_DIR / "figure_10_sup.png", dpi=150)
    plt.close()
    print("  Figure 10 (sup) saved.")


def fig_existing_weight_dist(snap_times, snap_weights):
    """Weight distribution over time (our original figure)."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Weight Distribution Over Time", fontsize=17)
    times_s = snap_times * DT_SIM
    snap_indices = np.linspace(0, len(times_s) - 1, 6, dtype=int)
    for ax, idx in zip(axes.flat, snap_indices):
        t_s = times_s[idx]
        w = snap_weights[idx]
        ax.hist(w, bins=50, range=(0, 1), color="steelblue", edgecolor="none")
        ax.set_title(f"t = {t_s:.0f}s", fontsize=14)
        ax.set_xlabel("Weight", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "weight_distribution.png", dpi=150)
    plt.close()
    print("  Weight distribution saved.")


def fig_existing_weight_map(final_weights):
    """Final weight map (our original figure)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(range(N_PATTERN), final_weights[:N_PATTERN], width=1.0,
                color="steelblue", edgecolor="none")
    axes[0].set_title(f"Pattern Afferents (0-{N_PATTERN-1})", fontsize=14)
    axes[0].set_xlabel("Afferent Index", fontsize=12)
    axes[0].set_ylabel("Weight", fontsize=12)
    axes[0].set_ylim(0, 1)
    axes[1].bar(range(N_PATTERN, N_AFFERENTS), final_weights[N_PATTERN:],
                width=1.0, color="coral", edgecolor="none")
    axes[1].set_title(f"Distractor Afferents ({N_PATTERN}-{N_AFFERENTS-1})", fontsize=14)
    axes[1].set_xlabel("Afferent Index", fontsize=12)
    axes[1].set_ylabel("Weight", fontsize=12)
    axes[1].set_ylim(0, 1)
    fig.suptitle("Final Weight Map", fontsize=17)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "weight_map.png", dpi=150)
    plt.close()
    print("  Weight map saved.")


def fig_existing_hit_fa(spike_times_ticks, pattern_onsets_ticks):
    """Hit rate and false alarm rate over time (our original figure)."""
    spike_times_s = spike_times_ticks * DT_SIM
    pattern_onsets_s = pattern_onsets_ticks * DT_SIM
    hit_window_ms = 10.0

    segment_boundaries = np.arange(0, TOTAL_SECONDS + 1, 30)
    hit_rates_over_time = []
    fa_rates_over_time = []

    for seg_i in range(len(segment_boundaries) - 1):
        t0_s = segment_boundaries[seg_i]
        t1_s = segment_boundaries[seg_i + 1]
        seg_onset_mask = (pattern_onsets_s >= t0_s) & (pattern_onsets_s < t1_s)
        seg_onsets = pattern_onsets_s[seg_onset_mask]

        seg_hits = 0
        for onset_s in seg_onsets:
            window_end = onset_s + PAT_DUR_TICKS * DT_SIM
            lo = np.searchsorted(spike_times_s, onset_s, side="left")
            hi = np.searchsorted(spike_times_s, window_end, side="left")
            if hi > lo and (spike_times_s[lo] - onset_s) * 1000 <= hit_window_ms:
                seg_hits += 1
        hr = seg_hits / len(seg_onsets) if len(seg_onsets) > 0 else 0
        hit_rates_over_time.append(hr)

        lo_s = np.searchsorted(spike_times_s, t0_s, side="left")
        hi_s = np.searchsorted(spike_times_s, t1_s, side="left")
        seg_ticks = spike_times_ticks[lo_s:hi_s]
        seg_spike_slots = seg_ticks // PAT_DUR_TICKS
        seg_onset_ticks = pattern_onsets_ticks[seg_onset_mask]
        seg_in_pat = np.isin(seg_spike_slots, seg_onset_ticks // PAT_DUR_TICKS)
        n_fa = np.sum(~seg_in_pat)
        non_pat_dur = (t1_s - t0_s) - len(seg_onsets) * PAT_DUR_TICKS * DT_SIM
        fa = n_fa / non_pat_dur if non_pat_dur > 0 else 0
        fa_rates_over_time.append(fa)

    fig, ax1 = plt.subplots(figsize=(12, 4))
    mid_times = (segment_boundaries[:-1] + segment_boundaries[1:]) / 2
    ax1.plot(mid_times, hit_rates_over_time, "b-", linewidth=2, label="Hit rate")
    ax1.set_xlabel("Time (s)", fontsize=14)
    ax1.set_ylabel("Hit rate", color="b", fontsize=14)
    ax1.set_ylim(0, 1.05)
    ax1.tick_params(axis="y", labelcolor="b", labelsize=12)
    ax1.tick_params(axis="x", labelsize=12)
    ax2 = ax1.twinx()
    ax2.plot(mid_times, fa_rates_over_time, "r-", linewidth=2, label="FA rate (Hz)")
    ax2.set_ylabel("False alarm rate (Hz)", color="r", fontsize=14)
    ax2.tick_params(axis="y", labelcolor="r", labelsize=12)
    fig.suptitle("Hit Rate and False Alarm Rate Over Time", fontsize=17)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "hit_fa_rate.png", dpi=150)
    plt.close()
    print("  Hit/FA rate saved.")


def fig_existing_latency(spike_times_ticks, pattern_onsets_ticks):
    """Figure 4-style: spike latency scatter over simulation time."""
    spike_times_s = spike_times_ticks * DT_SIM
    pattern_onsets_s = pattern_onsets_ticks * DT_SIM

    latencies = []
    for onset_s in pattern_onsets_s:
        window_end = onset_s + PAT_DUR_TICKS * DT_SIM
        lo = np.searchsorted(spike_times_s, onset_s, side="left")
        hi = np.searchsorted(spike_times_s, window_end, side="left")
        if hi > lo:
            lat = (spike_times_s[lo] - onset_s) * 1000
            latencies.append((onset_s, lat))
        else:
            latencies.append((onset_s, np.nan))

    fig, axes = plt.subplots(2, 1, figsize=(14, 7),
                             gridspec_kw={"height_ratios": [1, 2]})

    # Top panel: spike raster
    ax_top = axes[0]
    pat_dur_s = PAT_DUR_TICKS * DT_SIM
    for onset_s in pattern_onsets_s[::max(1, len(pattern_onsets_s) // 500)]:
        ax_top.axvspan(onset_s, onset_s + pat_dur_s,
                       alpha=0.15, color="dodgerblue", linewidth=0)
    if len(spike_times_s) > 0:
        spike_in_pat = np.zeros(len(spike_times_s), dtype=bool)
        for onset_s in pattern_onsets_s:
            lo = np.searchsorted(spike_times_s, onset_s, side="left")
            hi = np.searchsorted(spike_times_s, onset_s + pat_dur_s, side="left")
            spike_in_pat[lo:hi] = True
        fa_times = spike_times_s[~spike_in_pat]
        hit_times = spike_times_s[spike_in_pat]
        ax_top.eventplot([fa_times], lineoffsets=0.5, linelengths=0.8,
                         colors=["red"], linewidths=0.3, alpha=0.5)
        ax_top.eventplot([hit_times], lineoffsets=0.5, linelengths=0.8,
                         colors=["blue"], linewidths=0.3, alpha=0.7)
    ax_top.set_xlim(0, TOTAL_SECONDS)
    ax_top.set_yticks([])
    ax_top.set_ylabel("Spikes", fontsize=14)
    ax_top.set_title("Post-synaptic spikes (blue = pattern, red = false alarm)",
                     fontsize=14)

    # Bottom panel: latency scatter
    ax_bot = axes[1]
    lat_times = np.array([l[0] for l in latencies])
    lat_vals = np.array([l[1] for l in latencies])
    valid = ~np.isnan(lat_vals)
    miss = np.isnan(lat_vals)
    if np.any(miss):
        ax_bot.scatter(lat_times[miss],
                       np.full(np.sum(miss), pat_dur_s * 1000 - 1),
                       s=4, marker="x", color="lightgray", alpha=0.3, label="Miss")
    if np.any(valid):
        ax_bot.scatter(lat_times[valid], lat_vals[valid], s=6,
                       c=lat_vals[valid], cmap="coolwarm_r",
                       vmin=0, vmax=pat_dur_s * 1000,
                       alpha=0.6, edgecolors="none")
        window_s = 30.0
        med_times, med_vals = [], []
        for t0 in np.arange(0, TOTAL_SECONDS, window_s):
            m = valid & (lat_times >= t0) & (lat_times < t0 + window_s)
            if np.any(m):
                med_times.append(t0 + window_s / 2)
                med_vals.append(np.median(lat_vals[m]))
        if med_times:
            ax_bot.plot(med_times, med_vals, "k-", linewidth=1.5, alpha=0.8,
                        label="Median latency")
    ax_bot.set_xlim(0, TOTAL_SECONDS)
    ax_bot.set_ylim(-1, pat_dur_s * 1000 + 1)
    ax_bot.set_xlabel("Simulation time (s)", fontsize=14)
    ax_bot.set_ylabel("Spike latency [ms]", fontsize=14)
    ax_bot.axhline(y=0, color="green", linewidth=0.5, linestyle="--", alpha=0.5)
    ax_bot.legend(loc="upper right", fontsize=10)

    fig.suptitle("Spike Latency Relative to Pattern Onset", fontsize=17,
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figure4_spike_latency.png", dpi=200)
    plt.close()
    print("  Spike latency scatter saved.")


def main():
    global MODEL_VARIANT, OUTPUT_DIR, THETA, DT_SIM, TICK_SCALE, PAT_DUR_TICKS

    parser = argparse.ArgumentParser(description="Generate Masquelier 2008 figures")
    parser.add_argument(
        "--model", choices=["srm", "hg"], default="hg",
        help="Model variant: srm (SRM soma) or hg (HG-LIF soma, default)"
    )
    args = parser.parse_args()
    MODEL_VARIANT = args.model
    OUTPUT_DIR = BASE_DIR / MODEL_CONFIGS[MODEL_VARIANT]["output_dir"]
    THETA = MODEL_CONFIGS[MODEL_VARIANT]["theta"]
    DT_SIM = MODEL_CONFIGS[MODEL_VARIANT]["dt_sim"]
    TICK_SCALE = MODEL_CONFIGS[MODEL_VARIANT]["tick_scale"]
    PAT_DUR_TICKS = MODEL_CONFIGS[MODEL_VARIANT]["pattern_duration_ticks"]

    print(f"Model variant: {MODEL_VARIANT} (threshold={THETA}, dt={DT_SIM}, output={OUTPUT_DIR})")
    print("Loading simulation data ...")
    spike_times, snap_times, snap_weights, pattern_onsets = load_simulation_data()
    final_weights = snap_weights[-1]

    print(f"  {len(spike_times)} output spikes, {len(pattern_onsets)} pattern onsets")
    print(f"  {len(snap_times)} weight snapshots")

    print("Computing find_t ...")
    find_t = compute_find_t(spike_times, pattern_onsets)
    print(f"  find_t = {find_t:.1f}s")

    print("Regenerating input spike trains (same seed) ...")
    t0 = time.time()
    spike_trains = regenerate_spike_trains()
    print(f"  Done in {time.time()-t0:.1f}s")

    print("Recording membrane potential for Figure 9 ...")
    potentials = record_potential_windows(
        spike_trains, snap_times, snap_weights, pattern_onsets, find_t
    )

    print("Computing latencies ...")
    per_pattern_latencies, per_spike_latencies = compute_latencies(
        spike_times, pattern_onsets
    )

    print("Generating figures ...")

    # Author's figures
    fig_3(per_spike_latencies)
    fig_4(spike_trains, pattern_onsets, final_weights)
    fig_7AB(snap_times, snap_weights)
    fig_9(potentials, pattern_onsets, find_t)
    fig_10(spike_trains, pattern_onsets)

    # Our additional figures
    fig_existing_weight_dist(snap_times, snap_weights)
    fig_existing_weight_map(final_weights)
    fig_existing_hit_fa(spike_times, pattern_onsets)
    fig_existing_latency(spike_times, pattern_onsets)

    print(f"\nAll figures saved to {OUTPUT_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
