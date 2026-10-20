"""
masquelier_input -- the input of Masquelier, Guyonneau & Thorpe (2008), generated the
way Hathway & Goodman (2018) do it, vectorised with numpy.

2,000 afferents fire for 450 s. Each afferent's rate performs a random walk between
0 and 90 Hz (it changes every 1 ms, which gives the irregular, non-stationary trains of
the paper), with a forced spike whenever an afferent has been silent for 50 ms and
10 Hz of Poisson noise added on top. Into this, a fixed 50 ms *pattern* -- the spikes
of the first 1,000 afferents in one 50 ms window, copied with 1 ms Gaussian jitter --
is pasted into a random quarter of the 50 ms slots, never two in a row. The pattern is
indistinguishable from the background by rate; only its spike *timing* repeats. The
other 1,000 afferents never take part in it. 150 s of input are generated and tiled
three times, as in the reference code.

The time grid is the simulation's 0.1 ms tick: rates are drawn per 1 ms step and each
spike lands on one of the ten ticks of its step, as in the reference (continuous times
rounded to the clock).

Entry point: ``generate_input(seed)`` -> (afferents, ticks, pattern_onsets, n_ticks).
"""

import numpy as np

N_AFFERENTS = 2000
N_PATTERN = 1000           # afferents 0..999 carry the pattern
SEGMENT_S = 150.0          # unique input generated, then tiled
N_COPIES = 3               # 3 x 150 s = 450 s
PATTERN_S = 0.05           # pattern length and slot length
PATTERN_FRACTION = 0.25    # fraction of slots that hold the pattern
DT_RATE = 1e-3             # the rate changes and spikes are drawn per 1 ms
DT_SIM = 1e-4              # simulation tick
MAX_RATE = 90.0            # Hz
MAX_SILENCE_S = 0.05       # a spike is forced after this long without one
NOISE_RATE = 10.0          # Hz of extra Poisson spikes on every afferent
JITTER_S = 1e-3            # SD of the jitter on each pasted pattern spike


def _background(rng, n_steps, ticks_per_step):
    """Rate random walk + forced spikes. Returns (afferents, ticks) of one segment."""
    max_speed = MAX_RATE / MAX_SILENCE_S          # Hz per second: 1800
    max_silence_steps = int(round(MAX_SILENCE_S / DT_RATE))
    rate = rng.uniform(0.0, MAX_RATE, N_AFFERENTS)
    speed = rng.uniform(-max_speed, max_speed, N_AFFERENTS)
    # a virtual spike before t = 0 so that the forced spikes do not all fall on step 50
    last = -rng.integers(0, max_silence_steps, N_AFFERENTS)
    aff_parts, tick_parts = [], []
    block = 10_000
    for start in range(0, n_steps, block):
        n = min(block, n_steps - start)
        u = rng.random((n, N_AFFERENTS), dtype=np.float32)
        kicks = rng.uniform(-1.0, 1.0, (n, N_AFFERENTS)).astype(np.float32) * (0.4 * max_speed)
        fired = np.zeros((n, N_AFFERENTS), dtype=bool)
        for i in range(n):
            t = start + i + 1                      # steps are 1..n_steps, as in the reference
            spk = (u[i] < rate * DT_RATE) | (t - last > max_silence_steps)
            last[spk] = t
            fired[i] = spk
            rate = np.clip(rate + speed * DT_RATE, 0.0, MAX_RATE)
            speed = np.clip(speed + kicks[i], -max_speed, max_speed)
        step_idx, aff = np.nonzero(fired)
        # a spike drawn in step t falls uniformly in (t-1, t] ms: one of the step's ticks
        ticks = (start + step_idx + 1) * ticks_per_step - rng.integers(0, ticks_per_step, len(aff))
        aff_parts.append(aff.astype(np.int32))
        tick_parts.append(ticks.astype(np.int64))
    return np.concatenate(aff_parts), np.concatenate(tick_parts)


def _pattern_slots(rng, n_slots):
    """Random quarter of the slots, no two adjacent (the reference's rejection loop)."""
    want = int(np.floor(n_slots * PATTERN_FRACTION))
    flags = np.zeros(n_slots, dtype=bool)
    while flags.sum() < want:
        i = rng.integers(0, n_slots)
        if not flags[i] and not (i > 0 and flags[i - 1]) and not (i < n_slots - 1 and flags[i + 1]):
            flags[i] = True
    return np.flatnonzero(flags)


def generate_input(seed=1):
    """Return ``(afferents, ticks, pattern_onsets, n_ticks)``: one entry per input spike
    (afferent index 0..1999, simulation tick), the tick of every pattern presentation,
    and the run length in ticks."""
    rng = np.random.default_rng(seed)
    ticks_per_step = int(round(DT_RATE / DT_SIM))                  # 10
    slot_ticks = int(round(PATTERN_S / DT_SIM))                    # 500
    segment_ticks = int(round(SEGMENT_S / DT_SIM))                 # 1,500,000
    n_steps = int(round(SEGMENT_S / DT_RATE))                      # 150,000

    aff, ticks = _background(rng, n_steps, ticks_per_step)

    # --- the pattern: the first pattern slot's spikes of afferents 0..N_PATTERN-1 ---
    slots = _pattern_slots(rng, segment_ticks // slot_ticks)
    first = slots[0]
    in_first = (ticks // slot_ticks == first) & (aff < N_PATTERN)
    template_aff = aff[in_first]
    template_rel = ticks[in_first] - first * slot_ticks

    # --- paste it, jittered, into every pattern slot (replacing those afferents' spikes) ---
    replaced = np.isin(ticks // slot_ticks, slots) & (aff < N_PATTERN)
    aff, ticks = aff[~replaced], ticks[~replaced]
    n_rep = len(slots)
    paste_aff = np.tile(template_aff, n_rep)
    paste_ticks = (np.tile(template_rel, n_rep)
                   + np.repeat(slots * slot_ticks, len(template_rel))
                   + np.rint(rng.normal(0.0, JITTER_S / DT_SIM, n_rep * len(template_rel))).astype(np.int64))
    paste_ticks = np.clip(paste_ticks, 1, segment_ticks - 1)

    # --- 10 Hz Poisson noise on every afferent, pattern windows included ---
    n_noise = rng.poisson(NOISE_RATE * SEGMENT_S, N_AFFERENTS)
    noise_aff = np.repeat(np.arange(N_AFFERENTS, dtype=np.int32), n_noise)
    noise_ticks = rng.integers(1, segment_ticks, n_noise.sum())

    aff = np.concatenate([aff, paste_aff.astype(np.int32), noise_aff])
    ticks = np.concatenate([ticks, paste_ticks, noise_ticks])

    # --- one spike per afferent per tick (the reference's remove_close_spikes); a sort
    # and a mask, because np.unique is 40x slower on 19 M keys ---
    key = np.sort(aff.astype(np.int64) * segment_ticks + ticks)
    distinct = np.empty(len(key), dtype=bool)
    distinct[0] = True
    np.not_equal(key[1:], key[:-1], out=distinct[1:])
    key = key[distinct]
    aff = (key // segment_ticks).astype(np.int32)
    ticks = key % segment_ticks

    # --- tile the segment: 3 x 150 s ---
    aff = np.tile(aff, N_COPIES)
    ticks = np.concatenate([ticks + c * segment_ticks for c in range(N_COPIES)])
    pattern_onsets = np.concatenate([slots * slot_ticks + c * segment_ticks for c in range(N_COPIES)])
    return aff, ticks, pattern_onsets, segment_ticks * N_COPIES
