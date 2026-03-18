"""
Replication of Masquelier et al. (2008) using the Hathway-Goodman LIF model.

Uses the author's exact 3-variable LIF neuron with:
  du/dt = (A*a)/taus + (X*x - u)/taum
  da/dt = -a/taus
  x from single_exp synapses

Parameters match the author's Brian2 code exactly:
  T=500, K2=3, X=6.35, taus=tausyn=2.5ms, taum=10ms
  dt=0.1ms (matching Brian2's defaultclock.dt = 1e-4*second)

Usage:
    python run_experiment_hg.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from superneuroabm.model import NeuromorphicModel
from exp_pair_wise_stdp_bounded_nn import exp_pair_wise_stdp_bounded_nn

from experiment_utils import (
    N_AFFERENTS,
    N_PATTERN,
    PATTERN_DURATION as PATTERN_DURATION_1MS,
    PATTERN_FRACTION,
    TOTAL_SECONDS,
    SEGMENT_SECONDS,
    DT as DT_CREATE,
    SEED,
    generate_rate_profiles,
    generate_poisson_spikes,
    generate_pattern,
    embed_pattern,
    tile_spike_trains,
    get_spikes_in_range,
)

CONFIG_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = CONFIG_DIR / "output_hg"

# Simulation dt matches Brian2: defaultclock.dt = 1e-4 * second
DT_SIM = 1e-4  # 0.1ms
TICK_SCALE = int(DT_CREATE / DT_SIM)  # 10
TOTAL_TICKS = int(TOTAL_SECONDS / DT_SIM)
PATTERN_DURATION = PATTERN_DURATION_1MS * TICK_SCALE  # 500 ticks at 0.1ms


def run():
    rng = np.random.default_rng(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Generate input spikes at 1ms resolution (matching Brian2's
    # dt_createpattern = 0.001), then scale to 0.1ms simulation ticks.
    # ------------------------------------------------------------------
    print("Generating rate profiles for 150s segment ...")
    segment_ticks_1ms = int(SEGMENT_SECONDS / DT_CREATE)

    SUB_SEG = 30000
    spike_trains = [np.array([], dtype=np.int64) for _ in range(N_AFFERENTS)]

    for sub_start in range(0, segment_ticks_1ms, SUB_SEG):
        sub_end = min(sub_start + SUB_SEG, segment_ticks_1ms)
        sub_len = sub_end - sub_start
        t0 = time.time()
        rates_sub = generate_rate_profiles(N_AFFERENTS, sub_len, rng)
        trains_sub = generate_poisson_spikes(rates_sub, rng)
        for i in range(N_AFFERENTS):
            spike_trains[i] = np.concatenate([
                spike_trains[i],
                trains_sub[i] + sub_start
            ])
        del rates_sub, trains_sub
        print(f"  Sub-segment {sub_start//1000}-{sub_end//1000}s: {time.time()-t0:.1f}s")

    print("Generating pattern template ...")
    pattern_rng = np.random.default_rng(SEED + 1000)
    pattern_spikes = generate_pattern(N_PATTERN, PATTERN_DURATION_1MS, pattern_rng)

    t0 = time.time()
    print("Embedding pattern (25% of slots) ...")
    embed_rng = np.random.default_rng(SEED + 2000)
    spike_trains, pattern_onsets_seg_1ms = embed_pattern(
        spike_trains, pattern_spikes, segment_ticks_1ms, embed_rng
    )
    print(f"  Embedding done: {time.time()-t0:.1f}s")

    n_copies = TOTAL_SECONDS // SEGMENT_SECONDS
    print(f"Tiling to {TOTAL_SECONDS}s ({n_copies} copies) ...")
    spike_trains = tile_spike_trains(spike_trains, segment_ticks_1ms, n_copies)
    pattern_onsets_1ms = np.concatenate([
        pattern_onsets_seg_1ms + i * segment_ticks_1ms for i in range(n_copies)
    ])

    # Scale everything from 1ms ticks to 0.1ms ticks
    print(f"Scaling spike trains from 1ms to {DT_SIM*1000:.1f}ms resolution (x{TICK_SCALE}) ...")
    for i in range(N_AFFERENTS):
        spike_trains[i] = spike_trains[i] * TICK_SCALE
    pattern_onsets = pattern_onsets_1ms * TICK_SCALE
    segment_ticks = segment_ticks_1ms * TICK_SCALE

    np.save(OUTPUT_DIR / "pattern_onsets.npy", pattern_onsets)

    # ------------------------------------------------------------------
    # Build HG-LIF model
    # ------------------------------------------------------------------
    print("Building HG-LIF model ...")
    model = NeuromorphicModel(user_config=CONFIG_DIR / "masquelier_config.yaml", enable_internal_state_tracking=False)
    stdp_id = model.register_learning_rule(
        exp_pair_wise_stdp_bounded_nn,
        CONFIG_DIR / "exp_pair_wise_stdp_bounded_nn.py",
    )
    model.set_global_property_value("dt", DT_SIM)

    # Create HG-LIF output neuron
    soma = model.create_soma(
        breed="hg_lif_soma",
        config_name="masquelier_hg_config_0",
    )

    # Create 2000 single_exp input synapses (external -> soma)
    synapses = []
    for i in range(N_AFFERENTS):
        syn = model.create_synapse(
            breed="single_exp_synapse",
            pre_soma_id=-1,
            post_soma_id=soma,
            config_name="masquelier_stdp_config_0",
            learning_hyperparameters_overrides={"stdp_type": float(stdp_id)},
        )
        synapses.append(syn)

    print(f"  Soma ID: {soma}, Synapses: {synapses[0]}..{synapses[-1]}")

    # ------------------------------------------------------------------
    # Chunked simulation with reset() between chunks.
    # reset(retain_parameters=True) keeps weights but resets tick to 0,
    # so each chunk uses relative tick indices [0, CHUNK_TICKS).
    # ------------------------------------------------------------------
    CHUNK_TICKS = int(10.0 / DT_SIM)  # 100,000 ticks = 10s
    n_chunks = (TOTAL_TICKS + CHUNK_TICKS - 1) // CHUNK_TICKS

    print(f"Starting HG-LIF simulation: {TOTAL_TICKS} ticks, {n_chunks} chunks of {CHUNK_TICKS}")
    print(f"  dt={DT_SIM}s ({DT_SIM*1e6:.0f}us), total_time={TOTAL_SECONDS}s")
    t_start = time.time()

    all_spike_times = []
    weight_snapshots = []

    model.setup(use_gpu=True)  # Once — heavy init (code gen + JIT)

    for chunk_idx, chunk_start in enumerate(range(0, TOTAL_TICKS, CHUNK_TICKS)):
        chunk_t0 = time.time()
        chunk_end = min(chunk_start + CHUNK_TICKS, TOTAL_TICKS)
        chunk_ticks = chunk_end - chunk_start

        # Diagnostic on first chunk
        if chunk_idx == 0:
            dt_actual = model.get_global_property_value("dt")
            soma_hp = model.get_agent_property_value(id=soma, property_name="hyperparameters")
            syn_hp = model.get_agent_property_value(id=synapses[0], property_name="hyperparameters")
            syn_lp = model.get_agent_property_value(id=synapses[0], property_name="learning_hyperparameters")
            print(f"  [DIAG] dt={dt_actual}")
            print(f"  [DIAG] Soma hyperparams: {soma_hp}")
            print(f"  [DIAG] Synapse[0] hyperparams (weight first): {syn_hp}")
            print(f"  [DIAG] Synapse[0] learning params (stdp_type first): {syn_lp}")

        for i, syn_id in enumerate(synapses):
            chunk_spikes = get_spikes_in_range(
                spike_trains[i], chunk_start, chunk_end
            )
            if len(chunk_spikes) > 0:
                local_ticks = chunk_spikes - chunk_start
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

        model.simulate(ticks=chunk_ticks, update_data_ticks=1)

        soma_spikes = model.get_spike_times(soma)
        all_spike_times.extend([t + chunk_start for t in soma_spikes])

        weights = [
            model.get_agent_property_value(id=s, property_name="hyperparameters")[0]
            for s in synapses
        ]
        weight_snapshots.append((chunk_end, np.array(weights, dtype=np.float32)))

        model.reset(retain_parameters=True)

        n_patterns = np.sum((pattern_onsets >= chunk_start) & (pattern_onsets < chunk_end))
        sim_s = chunk_end * DT_SIM
        chunk_elapsed = time.time() - chunk_t0
        print(
            f"  Chunk {chunk_idx+1}: t={sim_s:.1f}s, "
            f"patterns={n_patterns}, spikes={len(soma_spikes)}, "
            f"w[0]={weights[0]:.4f}, w[-1]={weights[-1]:.4f}, "
            f"elapsed={chunk_elapsed:.1f}s"
        )

    all_spike_times = np.array(all_spike_times, dtype=np.int64)

    elapsed = time.time() - t_start
    final_weights = weight_snapshots[-1][1] if weight_snapshots else np.zeros(N_AFFERENTS)
    print(
        f"  Simulation complete: {len(all_spike_times)} spikes, "
        f"w[0]={final_weights[0]:.4f}, w[-1]={final_weights[-1]:.4f}, elapsed={elapsed:.1f}s"
    )

    np.save(OUTPUT_DIR / "spike_times.npy", all_spike_times)
    np.savez(
        OUTPUT_DIR / "weight_snapshots.npz",
        times=np.array([ws[0] for ws in weight_snapshots]),
        weights=np.stack([ws[1] for ws in weight_snapshots]),
    )

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------
    print("Analysing results ...")
    spike_times = all_spike_times * DT_SIM
    pattern_onsets_s = pattern_onsets * DT_SIM

    latencies = []
    hit_window_ms = 10.0
    hits = 0
    for onset_s in pattern_onsets_s:
        window_end = onset_s + PATTERN_DURATION * DT_SIM
        lo = np.searchsorted(spike_times, onset_s, side="left")
        hi = np.searchsorted(spike_times, window_end, side="left")
        if hi > lo:
            lat = (spike_times[lo] - onset_s) * 1000
            latencies.append((onset_s, lat))
            if lat <= hit_window_ms:
                hits += 1
        else:
            latencies.append((onset_s, np.nan))

    hit_rate = hits / len(pattern_onsets_s) if len(pattern_onsets_s) > 0 else 0

    spike_ticks = all_spike_times
    spike_slots = spike_ticks // PATTERN_DURATION
    in_pattern = np.isin(spike_slots, pattern_onsets // PATTERN_DURATION)
    n_false = np.sum(~in_pattern)
    total_non_pattern_s = TOTAL_SECONDS - len(pattern_onsets_s) * PATTERN_DURATION * DT_SIM
    false_alarm_rate = n_false / total_non_pattern_s if total_non_pattern_s > 0 else 0

    print(f"  Overall hit rate: {hit_rate*100:.1f}%")
    print(f"  False alarm rate: {false_alarm_rate:.2f} Hz")
    print(f"Plots and data saved to {OUTPUT_DIR}/")
    print("Done.")


if __name__ == "__main__":
    run()
