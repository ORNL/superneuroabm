"""SuperNeuroABM Brunel setup-time benchmark, single GPU.

Mirrors the w=1 point of ``scaling_analysis/weak_scaling.py`` (columnar
partition -> load_post_owned -> setup -> Poisson drive -> simulate) and reports
the wall time of every setup and first-tick sub-step, plus the per-property GPU
tensor allocation table.

Per-tick cost is reported as one aggregate number only; this round is about
where setup time goes.
"""

import argparse
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))          # scaling_analysis/
sys.path.insert(0, str(SCRIPT_DIR.parent.parent))   # repo root

import numpy as np

import bench_common as bc


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--neurons", type=int, default=12500)
    p.add_argument("--in-degree", type=int, default=1000)
    p.add_argument("--dt-ms", type=float, default=1.0)
    p.add_argument("--ticks", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--g", type=float, default=5.0)
    p.add_argument("--J-E", type=float, default=0.02581)
    p.add_argument("--delay", type=float, default=1.5)
    p.add_argument("--firing-rate", type=float, default=10.0)
    p.add_argument("--csv", default=str(SCRIPT_DIR / "results" / "brunel_sna.csv"))
    p.add_argument("--partition-dir", default=None)
    p.add_argument("--keep-partition", action="store_true",
                   help="Reuse a cached partition instead of regenerating it.")
    p.add_argument("--in-memory", action="store_true",
                   help="Skip the .npz round-trip: generate the partition arrays and "
                        "hand them straight to the columnar builder. Isolates what "
                        "serialization actually costs.")
    return p


def run_once(args, repeat):
    import cupy as cp
    from superneuroabm.model import NeuromorphicModel
    from superneuroabm.brunel import save_brunel_partition, brunel_partition
    import scaling_diagnostics as diagnostics

    log = bc.PhaseLog("sna", "columnar-inmem" if args.in_memory else "columnar",
                      args.neurons, args.in_degree, repeat=repeat)
    t_pipeline = time.time()

    # ---- 1. Generate the partition (rank-local, columnar .npz) ----
    part_dir = Path(args.partition_dir or (SCRIPT_DIR / "partitions" /
                    f"1w_{args.neurons}n_K{args.in_degree}_torus3d_rauto"))
    partition_file = part_dir / "partition_0.npz"
    C_E = round(0.8 * args.in_degree)
    C_I = args.in_degree - C_E

    gen_kwargs = dict(
        somas_per_rank=args.neurons,
        num_partitions=1,
        partition_rank=0,
        excitatory_in_degree=C_E,
        inhibitory_in_degree=C_I,
        topology="torus3d",
        inhibitory_weight_ratio=args.g,
        excitatory_weight=args.J_E,
        synaptic_delay_ms=args.delay,
        seed=args.seed,
        output_format="columns",
    )
    t0 = time.time()
    arrays = None
    if args.in_memory:
        arrays = brunel_partition(**gen_kwargs)
    elif not (args.keep_partition and partition_file.exists()):
        save_brunel_partition(output_dir=str(part_dir), **gen_kwargs)
    log.add("generation", time.time() - t0)

    # ---- 2. Load the partition ----
    t0 = time.time()
    model = NeuromorphicModel(enable_internal_states_tracking=False)
    if args.in_memory:
        # Same builder load_post_owned dispatches to for a .npz, minus the file.
        model._build_post_owned_columnar(arrays)
    else:
        model.load_post_owned(str(partition_file))
    # verbose_timing is required for the first-tick sub-timers. It also unfuses
    # the single-worker tick path, which inflates per-tick cost; that is why the
    # per-tick number here is reported as one aggregate and not analysed.
    model.verbose_timing = True
    load_wall = time.time() - t0
    lt = dict(getattr(model, "_load_timings", {}))
    log.add_all("model_load", lt)
    log.add("model_load.other", max(0.0, load_wall - sum(lt.values())))
    log.add("model_load.total", load_wall)

    n_syn = getattr(model, "_num_synapses", None) or len(model._synapse_ids)
    log.set_synapses(n_syn)

    # ---- 3. setup() ----
    t0 = time.time()
    model.setup()
    setup_wall = time.time() - t0
    st = dict(getattr(model, "_setup_timings", {}))
    # 'total' is SAGESim's own span and 'sagesim_setup' already covers it, so
    # neither counts toward the residual.
    inner = {k: v for k, v in st.items() if k not in ("total", "sagesim_setup")}
    log.add_all("setup", inner)
    log.add("setup.sagesim_total", st.get("total", 0.0))
    log.add("setup.other", max(0.0, setup_wall - sum(inner.values())))
    log.add("setup.total", setup_wall)

    # ---- 4. Poisson drive ----
    if getattr(model, "_input_synapse_ids", None) is not None:
        input_synapses = model._input_synapse_ids.tolist()
    else:
        input_synapses = [s for s in model._synapse_ids
                          if model.get_synapse_connectivity(s)[0] == -1]
    t0 = time.time()
    diagnostics.inject_poisson_drive(model, input_synapses, args.firing_rate,
                                     args.ticks, args.dt_ms, args.seed, 0)
    log.add("inject_drive", time.time() - t0)

    # ---- 5. Simulate ----
    t0 = time.time()
    model.simulate(ticks=args.ticks, update_data_ticks=1)
    sim_wall = time.time() - t0

    ticks = getattr(model, "_tick_timings", []) or []
    first = ticks[0] if ticks else {}
    log.add("first_tick.total", first.get("total", 0.0))
    log.add("first_tick.neighbor", first.get("neighbor", 0.0))
    log.add("first_tick.ghost_topo", first.get("contextualize", 0.0))
    log.add("first_tick.gpu_build", first.get("gpu_buffer_build", 0.0))
    log.add("first_tick.comm_init", first.get("comm_init", 0.0))
    sub = first.get("gpu_build_sub", {}) or {}
    for key in ("breed_ranges", "id_hashmap", "id_cpu_dict", "id_gpu_upload",
                "id_global_data", "id_gpu_hashmap", "combined_data", "mpi_sync",
                "prop_tensors", "write_bufs"):
        if key in sub:
            log.add(f"first_tick.gpu_build.{key}", sub[key])

    # Per-property allocation table (shape, bytes, seconds).
    idx2name = {i: n for n, i in
                model._agent_factory._property_name_2_index.items()}
    stats = getattr(model._gpu_buffers, "property_alloc_stats", []) or []
    prop_rows = []
    for s in stats:
        name = idx2name.get(s["prop_idx"], f"prop{s['prop_idx']}")
        log.add(f"first_tick.prop_tensors.{name}", s["seconds"],
                gpu=s["nbytes"] / 1e6)
        prop_rows.append((name, s["shape"], s["nbytes"], s["seconds"],
                          s.get("path") or "?"))

    # Steady-state cost, aggregate only.
    rest = sum(t.get("total", 0.0) for t in ticks[1:])
    log.add(f"ticks_2_{args.ticks}", rest)
    log.add("simulate_wall", sim_wall)
    log.add("total", time.time() - t_pipeline)

    pool_mb = cp.get_default_memory_pool().used_bytes() / 1e6
    log.add("gpu_pool_used", 0.0, gpu=pool_mb)

    bc.print_table(log.rows, f"SuperNeuroABM Brunel  N={args.neurons} "
                             f"K={args.in_degree} synapses={n_syn:,}")
    print(f"\n  Per-property GPU tensors (capacity x width, float32):")
    print(f"    {'property':<34}{'shape':>18}{'MB':>9}{'alloc s':>9}{'s/100MB':>9}  path")
    for name, shape, nbytes, secs, path in sorted(prop_rows, key=lambda r: -r[3]):
        per100 = secs / (nbytes / 1e8) if nbytes else 0.0
        print(f"    {name:<34}{str(shape):>18}{nbytes/1e6:>9.1f}{secs:>9.3f}"
              f"{per100:>9.2f}  {path}")
    print(f"\n  CuPy pool in use: {pool_mb:.1f} MB   "
          f"bytes/synapse (pool/synapses): {pool_mb*1e6/max(n_syn,1):.1f}")
    return log


def main():
    args = build_parser().parse_args()
    for r in range(args.repeat):
        print(f"\n########## repeat {r + 1}/{args.repeat} ##########")
        log = run_once(args, r)
        path = log.write(args.csv)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
