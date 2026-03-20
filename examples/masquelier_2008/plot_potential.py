"""
Plot membrane potential for arbitrary time windows.

Re-runs a short simulation with the correct learned weights to record
the membrane potential trace.

Usage:
    python plot_potential.py --model hg --window 20-30
    python plot_potential.py --model hg --window 0-1,20-30,449-450
    python plot_potential.py --model srm --window 10-11
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from superneuroabm.model import NeuromorphicModel

from experiment_utils import (
    N_AFFERENTS, N_PATTERN, SEED,
    SEGMENT_SECONDS, TOTAL_SECONDS,
    DT as DT_CREATE,
    generate_rate_profiles, generate_poisson_spikes,
    generate_pattern, embed_pattern, tile_spike_trains,
    get_spikes_in_range,
    PATTERN_DURATION as PATTERN_DURATION_1MS,
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_CONFIGS = {
    "srm": {
        "output_dir": "output_srm",
        "theta": 731.21,
        "soma_breed": "srm_soma",
        "soma_config": "masquelier_srm_config_0",
        "synapse_breed": "double_exp_synapse",
        "synapse_config": "masquelier_srm_config_0",
        "dt_sim": 1e-3,
        "tick_scale": 1,
        "pattern_duration_ticks": 50,
    },
    "hg": {
        "output_dir": "output_hg",
        "theta": 500.0,
        "soma_breed": "hg_lif_soma",
        "soma_config": "masquelier_hg_config_0",
        "synapse_breed": "single_exp_synapse",
        "synapse_config": "masquelier_config_0",
        "dt_sim": 1e-4,
        "tick_scale": 10,
        "pattern_duration_ticks": 500,
    },
}


def regenerate_spike_trains(tick_scale):
    """Regenerate input spike trains using the same seed as the experiment."""
    rng = np.random.default_rng(SEED)
    segment_ticks = int(SEGMENT_SECONDS / DT_CREATE)
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
    pattern_spikes = generate_pattern(N_PATTERN, PATTERN_DURATION_1MS, pattern_rng)
    embed_rng = np.random.default_rng(SEED + 2000)
    spike_trains, _ = embed_pattern(
        spike_trains, pattern_spikes, segment_ticks, embed_rng
    )
    spike_trains = tile_spike_trains(spike_trains, segment_ticks, TOTAL_SECONDS // SEGMENT_SECONDS)

    if tick_scale > 1:
        for i in range(len(spike_trains)):
            spike_trains[i] = spike_trains[i] * tick_scale

    return spike_trains


def record_window(cfg, spike_trains, snap_times, snap_weights,
                  start_s, end_s):
    """Run simulation to record membrane potential, chunked in 1s pieces.

    Large windows (>1s at 0.1ms) would OOM with internal state tracking
    for all 2001 agents, so we split into 1s sub-windows and concatenate.
    """
    dt_sim = cfg["dt_sim"]
    ticks_per_sec = int(1.0 / dt_sim)
    start_tick = int(start_s / dt_sim)
    end_tick = int(end_s / dt_sim)
    total_ticks = end_tick - start_tick

    # Find closest prior weight snapshot
    snap_idx = np.searchsorted(snap_times, start_tick, side="right") - 1
    if snap_idx < 0:
        weights = np.full(N_AFFERENTS, 0.475, dtype=np.float32)
    else:
        weights = snap_weights[snap_idx]

    # Split into 1s sub-windows
    sub_size = ticks_per_sec
    all_t = []
    all_v = []

    n_subs = (total_ticks + sub_size - 1) // sub_size
    print(f"  Simulating {start_s:.1f}-{end_s:.1f}s "
          f"({total_ticks} ticks, {n_subs} sub-chunks, weights from snap {snap_idx}) ...")

    model = NeuromorphicModel(user_config=BASE_DIR / "masquelier_config.yaml", enable_internal_state_tracking=True)
    model.set_global_property_value("dt", dt_sim)

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

    t0 = time.time()
    model.setup(use_gpu=True)  # Once — heavy init (code gen + JIT)

    # Set weights before first simulation
    for i, syn_id in enumerate(synapses):
        hp = model.get_agent_property_value(
            id=syn_id, property_name="hyperparameters"
        )
        hp[0] = float(weights[i])
        model.set_agent_property_value(syn_id, "hyperparameters", hp)

    for sub_idx in range(n_subs):
        sub_start = start_tick + sub_idx * sub_size
        sub_end = min(sub_start + sub_size, end_tick)
        sub_ticks = sub_end - sub_start

        # Load input spikes for this sub-chunk
        for i, syn_id in enumerate(synapses):
            chunk_spikes = get_spikes_in_range(
                spike_trains[i], sub_start, sub_end
            )
            if len(chunk_spikes) > 0:
                local_ticks = chunk_spikes - sub_start
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

        model.simulate(ticks=sub_ticks, update_data_ticks=sub_ticks)

        states = np.array(model.get_internal_states_history(agent_id=soma))
        v_sub = states[:, 0]
        t_sub = np.arange(sub_start, sub_end) * dt_sim
        all_t.append(t_sub)
        all_v.append(v_sub)

        model.reset(retain_parameters=True)

        if n_subs > 1:
            print(f"    Sub-chunk {sub_idx+1}/{n_subs} done")

    t_arr = np.concatenate(all_t)
    v_trace = np.concatenate(all_v)
    print(f"    Total: {time.time()-t0:.1f}s, v range: [{v_trace.min():.1f}, {v_trace.max():.1f}]")

    del model
    return t_arr, v_trace


def plot_windows(cfg, windows_data, pattern_onsets_ticks, output_path):
    """Plot membrane potential for one or more time windows."""
    dt_sim = cfg["dt_sim"]
    pat_dur_ticks = cfg["pattern_duration_ticks"]
    theta = cfg["theta"]
    pattern_onsets_s = pattern_onsets_ticks * dt_sim

    n_windows = len(windows_data)
    fig, axes = plt.subplots(n_windows, 1, figsize=(14, 3.5 * n_windows),
                             squeeze=False)
    fig.suptitle("Membrane Potential", fontsize=17)

    for idx, (label, t_arr, v_arr) in enumerate(windows_data):
        ax = axes[idx, 0]
        start = t_arr[0]
        end = t_arr[-1] + dt_sim

        ax.plot(t_arr, v_arr, "b", linewidth=0.8, label="potential")
        ax.axhline(y=theta, color="r", linestyle="--", label="threshold")
        ax.axhline(y=0, color="k", linestyle="--", label="resting pot.")

        # Pattern windows
        onsets_in_window = pattern_onsets_s[
            (pattern_onsets_s >= start) & (pattern_onsets_s < end)
        ]
        for onset in onsets_in_window:
            ax.axvline(onset, ls="--", c="r", linewidth=2)
            ax.axvspan(onset, onset + pat_dur_ticks * dt_sim,
                       facecolor="g", alpha=0.3)

        ax.set_xlim([start, end])
        ax.set_yticks([0, int(theta), int(theta * 2)])
        ax.set_ylim([-theta * 0.6, theta * 2.2])
        ax.set_ylabel("Potential [a.u.]", fontsize=14)
        ax.set_title(label, fontsize=14)

        if idx == n_windows - 1:
            ax.set_xlabel("t [s]", fontsize=14)
        if idx == 0:
            ax.legend(loc="upper right", fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot membrane potential for arbitrary time windows"
    )
    parser.add_argument(
        "--model", choices=["srm", "hg"], default="hg",
        help="Model variant (default: hg)"
    )
    parser.add_argument(
        "--window", required=True,
        help="Time window(s) in seconds, e.g. '20-30' or '0-1,20-30,449-450'"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output filename (default: potential_<window>.png in output dir)"
    )
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    output_dir = BASE_DIR / cfg["output_dir"]
    dt_sim = cfg["dt_sim"]
    tick_scale = cfg["tick_scale"]

    # Parse windows
    windows = []
    for w in args.window.split(","):
        parts = w.strip().split("-")
        start_s = float(parts[0])
        end_s = float(parts[1])
        windows.append((start_s, end_s))

    # Load simulation data
    print("Loading simulation data ...")
    data = np.load(output_dir / "weight_snapshots.npz")
    snap_times = data["times"]
    snap_weights = data["weights"]
    pattern_onsets = np.load(output_dir / "pattern_onsets.npy")

    # Regenerate spike trains
    print("Regenerating input spike trains ...")
    t0 = time.time()
    spike_trains = regenerate_spike_trains(tick_scale)
    print(f"  Done in {time.time()-t0:.1f}s")

    # Record each window
    windows_data = []
    for start_s, end_s in windows:
        label = f"{start_s:.1f} – {end_s:.1f} s"
        t_arr, v_trace = record_window(
            cfg, spike_trains, snap_times, snap_weights, start_s, end_s
        )
        windows_data.append((label, t_arr, v_trace))

    # Plot
    if args.output:
        output_path = output_dir / args.output
    else:
        window_str = args.window.replace(",", "_").replace("-", "to")
        output_path = output_dir / f"potential_{window_str}.png"

    plot_windows(cfg, windows_data, pattern_onsets, output_path)


if __name__ == "__main__":
    main()
