"""
Shared utilities for the Masquelier et al. (2008) STDP experiment.

Contains spike generation, pattern embedding, and model configuration
functions used by all experiment variants (HG-LIF, SRM, calibration, etc.).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_AFFERENTS = 2000          # Total number of input afferents
N_PATTERN = 1000            # First 1000 afferents carry the pattern
PATTERN_DURATION = 50       # Pattern length in ticks (50 ms at 1ms resolution)
PATTERN_FRACTION = 0.125    # 12.5% of time slots contain the pattern
TOTAL_SECONDS = 100         # Total simulation time (shortened for testing)
SEGMENT_SECONDS = 100       # Single segment
DT = 1e-3                   # 1 ms spike generation time step
TOTAL_TICKS = int(TOTAL_SECONDS / DT)
CHUNK_SIZE = 10000          # 10s chunks for memory management
SPONTANEOUS_RATE = 10.0     # 10 Hz spontaneous activity
MAX_SILENCE_MS = 50         # Force spike if silent > 50 ms
JITTER_STD = 1.0            # 1 ms Gaussian jitter for pattern embedding

SEED = 42

CONFIG_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Spike generation utilities (vectorized)
# ---------------------------------------------------------------------------

def generate_rate_profiles(n_afferents, n_ticks, rng):
    """Generate time-varying firing rate profiles for each afferent.

    Rate modulation following the paper (matching Hathway-Goodman 2018 code):
        r += s * dt
        s += 0.4 * uniform(-1, 1) * max_change_speed   (NOT multiplied by dt)
    with r clipped to [0, 90] Hz and s clipped to [-1800, 1800] Hz/s.

    The speed-of-change s gets a large random kick each tick (uniform(-720, 720)),
    causing rapid fluctuations that are smoothed by the rate integration.
    """
    MAX_CHANGE_SPEED = 1800.0  # Hz/s (= max_rate / max_time_wo_spike = 90/0.05)

    rates = np.zeros((n_afferents, n_ticks), dtype=np.float32)
    r = rng.uniform(0, 90, size=n_afferents).astype(np.float32)
    s = rng.uniform(-MAX_CHANGE_SPEED, MAX_CHANGE_SPEED, size=n_afferents).astype(np.float32)

    # Pre-generate all random increments at once
    # Author's code: rate_change += 1/5 * 2 * (rand()-0.5) * max_change_speed
    #              = rate_change += 0.4 * uniform(-1,1) * 1800
    #              = rate_change += uniform(-720, 720)  (per tick, NOT scaled by dt)
    all_ds = (rng.uniform(-1, 1, size=(n_ticks, n_afferents)) * 0.4 * MAX_CHANGE_SPEED).astype(np.float32)

    for t in range(n_ticks):
        rates[:, t] = r
        s = np.clip(s + all_ds[t], -MAX_CHANGE_SPEED, MAX_CHANGE_SPEED)
        r = np.clip(r + s * DT, 0, 90)

    return rates


def _enforce_max_silence(spikes, n_ticks, max_silence):
    """Add forced spikes to eliminate silence periods > max_silence.

    Operates on spike times (O(n_spikes)), not tick-by-tick (O(n_ticks)).
    """
    if len(spikes) == 0:
        return np.arange(0, n_ticks, max_silence, dtype=np.int64)

    forced = []

    # Gap before first spike
    if spikes[0] >= max_silence:
        forced.extend(range(max_silence, int(spikes[0]), max_silence))

    # Gaps between consecutive spikes
    isis = np.diff(spikes)
    long_gap_idx = np.where(isis >= max_silence)[0]
    for idx in long_gap_idx:
        t_start = int(spikes[idx])
        t_end = int(spikes[idx + 1])
        forced.extend(range(t_start + max_silence, t_end, max_silence))

    # Gap after last spike
    remaining = n_ticks - int(spikes[-1])
    if remaining >= max_silence:
        forced.extend(range(int(spikes[-1]) + max_silence, n_ticks, max_silence))

    if forced:
        return np.unique(np.concatenate([
            spikes, np.array(forced, dtype=np.int64)
        ]))
    return spikes


def generate_poisson_spikes(rates, rng):
    """Generate Poisson spike trains from rate profiles (vectorized).

    rates: (n_afferents, n_ticks) array of instantaneous rates in Hz.
    Returns list of arrays: spike_trains[i] = sorted array of spike ticks.
    """
    n_afferents, n_ticks = rates.shape

    # Vectorized spike generation
    prob = np.clip((rates + SPONTANEOUS_RATE) * DT, 0, 1)
    rand = rng.random((n_afferents, n_ticks), dtype=np.float32)
    spike_matrix = rand < prob
    del prob, rand  # free memory

    # Extract spike times per afferent and enforce max silence
    spike_trains = []
    for i in range(n_afferents):
        spikes = np.where(spike_matrix[i])[0].astype(np.int64)
        spikes = _enforce_max_silence(spikes, n_ticks, MAX_SILENCE_MS)
        spike_trains.append(spikes)

    return spike_trains


def generate_pattern(n_pattern_afferents, pattern_duration, rng):
    """Generate a fixed spike pattern (random 50ms segment)."""
    rates = generate_rate_profiles(n_pattern_afferents, pattern_duration, rng)
    pattern_spikes = generate_poisson_spikes(rates, rng)
    return pattern_spikes


def embed_pattern(spike_trains, pattern_spikes, n_ticks, rng):
    """Embed pattern into spike trains at random 25% of time slots (vectorized).

    Returns:
        spike_trains: modified spike trains (list of arrays)
        pattern_onsets: array of ticks where patterns were inserted
    """
    n_slots = n_ticks // PATTERN_DURATION
    n_pattern_slots = int(n_slots * PATTERN_FRACTION)

    # Select pattern slots with NO consecutive slots (matching author's code)
    # This ensures at least one non-pattern slot (50ms) between patterns,
    # giving the neuron time to recover from afterhyperpolarization + x=0 reset.
    slot_indices = np.zeros(n_slots, dtype=np.int32)
    placed = 0
    while placed < n_pattern_slots:
        idx = rng.integers(0, n_slots)
        if slot_indices[idx] == 0:
            # Check neighbors aren't pattern slots
            left_ok = (idx == 0) or (slot_indices[idx - 1] == 0)
            right_ok = (idx == n_slots - 1) or (slot_indices[idx + 1] == 0)
            if left_ok and right_ok:
                slot_indices[idx] = 1
                placed += 1
    slot_indices = np.where(slot_indices == 1)[0]
    slot_indices.sort()
    pattern_onsets = slot_indices * PATTERN_DURATION
    pattern_slot_set = set(slot_indices.tolist())

    # Step 1: Remove spikes in pattern windows for pattern afferents
    # Use slot-based membership test (vectorized per afferent)
    for i in range(N_PATTERN):
        train = spike_trains[i]
        if len(train) == 0:
            continue
        spike_slots = train // PATTERN_DURATION
        in_pattern = np.isin(spike_slots, slot_indices)
        spike_trains[i] = train[~in_pattern]

    # Step 2: Pre-flatten pattern into (afferent_index, relative_tick) pairs
    pat_aff_parts = []
    pat_tick_parts = []
    for i in range(N_PATTERN):
        n = len(pattern_spikes[i])
        if n > 0:
            pat_aff_parts.append(np.full(n, i, dtype=np.int64))
            pat_tick_parts.append(pattern_spikes[i].astype(np.int64))

    if not pat_aff_parts:
        return spike_trains, pattern_onsets

    pat_aff = np.concatenate(pat_aff_parts)
    pat_tick = np.concatenate(pat_tick_parts)
    n_pat = len(pat_aff)
    n_onsets = len(pattern_onsets)

    # Step 3: Generate ALL jittered pattern spikes for ALL onsets at once
    # Shape: n_onsets * n_pat total spikes
    all_aff = np.tile(pat_aff, n_onsets)
    onset_repeated = np.repeat(pattern_onsets, n_pat)
    tick_repeated = np.tile(pat_tick, n_onsets)
    all_jitter = np.round(
        rng.normal(0, JITTER_STD, size=n_onsets * n_pat)
    ).astype(np.int64)

    all_ticks = tick_repeated + onset_repeated + all_jitter
    all_ticks = np.clip(all_ticks, onset_repeated,
                        onset_repeated + PATTERN_DURATION - 1)

    # Step 4: Group by afferent and merge into spike trains
    sort_idx = np.argsort(all_aff, kind="mergesort")
    all_aff_sorted = all_aff[sort_idx]
    all_ticks_sorted = all_ticks[sort_idx]

    unique_aff, group_starts = np.unique(all_aff_sorted, return_index=True)
    group_ends = np.empty_like(group_starts)
    group_ends[:-1] = group_starts[1:]
    group_ends[-1] = len(all_aff_sorted)

    for j, aff_idx in enumerate(unique_aff):
        new_ticks = all_ticks_sorted[group_starts[j]:group_ends[j]]
        spike_trains[aff_idx] = np.unique(np.concatenate([
            spike_trains[aff_idx], new_ticks
        ]))

    return spike_trains, pattern_onsets


def tile_spike_trains(spike_trains, segment_ticks, n_copies):
    """Tile spike trains by pasting n_copies of the segment."""
    tiled = []
    for train in spike_trains:
        parts = [train + c * segment_ticks for c in range(n_copies)]
        tiled.append(np.concatenate(parts))
    return tiled


def get_spikes_in_range(train, start, end):
    """Return spike ticks from train that fall in [start, end). Uses binary search."""
    lo = np.searchsorted(train, start, side="left")
    hi = np.searchsorted(train, end, side="left")
    return train[lo:hi]


