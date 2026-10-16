"""Fixed per-tick cost of SuperNeuroABM on a silent network, decomposed by variant.

  variant  somas  synapses   isolates
  floor        1  0          grid barriers + tick loop + (per-tick launches) the host path
  somas    N      0          + the soma phase (N rows)
  ring     N      N          + the wake-expansion scan over N emitters (1 dependant each)

usage: fixed_cost_probe.py <variant> <N> <ticks> <fused|unfused>
fused: one launch for all ticks (the single-GPU user path); unfused: one launch per tick
(what multi-rank runs and verbose timing do). Prints ms/tick for the second simulate()
call (the first builds the device buffers). Laptop numbers: docs/EVENT_DRIVEN_NOTES.md
section 15.
"""
import sys
import time

from superneuroabm.model import NeuromorphicModel


def main(variant, N, ticks, mode):
    if variant == "floor":
        N = 1
    m = NeuromorphicModel(enable_internal_states_tracking=False, synapse_stepping="event",
                          learning_rule_info={})
    somas = [m.create_soma(breed="lif_soma", config_name="config_0") for _ in range(N)]
    n_syn = 0
    if variant == "ring":
        for i in range(N):
            m.create_synapse(breed="delivery_synapse", pre_soma_id=somas[i],
                             post_soma_id=somas[(i + 1) % N], config_name="config_0")
        n_syn = N
    m.setup()
    m.verbose_timing = (mode == "unfused")
    t0 = time.perf_counter()
    m.simulate(ticks=1, update_data_ticks=1)
    t_first = time.perf_counter() - t0
    t0 = time.perf_counter()
    m.simulate(ticks=ticks, update_data_ticks=1)
    t_run = time.perf_counter() - t0
    print(f"RESULT variant={variant} N={N} synapses={n_syn} ticks={ticks} mode={mode} "
          f"ms_per_tick={t_run / ticks * 1e3:.4f} first_tick_s={t_first:.3f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
