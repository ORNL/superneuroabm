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
    p.add_argument("--external-rate", type=float, default=14.53,
                   help="Per-external-input Poisson rate in Hz. "
                        "Default 14.53 is Brunel's eta=2: with --external-inputs 800 and "
                        "J_E=0.02581 it puts the mean depolarisation at 30 mV against a "
                        "15 mV threshold, i.e. twice the threshold rate.")
    p.add_argument("--external-inputs", type=int, default=None,
                   help="How many independent external afferents the single pre=-1 synapse "
                        "stands in for; the per-tick count is Poisson(inputs * rate * dt). "
                        "Defaults to C_E. Pass 1 to reproduce the historical sub-threshold "
                        "drive, which leaves the network silent.")
    p.add_argument("--external-weight", type=float, default=None,
                   help="Weight of the external input synapse. Default (None) is J_E, which "
                        "is correct when --external-inputs carries the aggregate in the spike "
                        "count. Setting it to C_E*J_E instead makes every single external "
                        "event supra-threshold, which is a Poisson relay, not an AI network.")
    p.add_argument("--csv", default=str(SCRIPT_DIR / "results" / "brunel_sna.csv"))
    p.add_argument("--partition-dir", default=None)
    p.add_argument("--keep-partition", action="store_true",
                   help="Reuse a cached partition instead of regenerating it.")
    p.add_argument("--no-learning-rules", action="store_true",
                   help="Build with learning_rule_info={} (no STDP kernels registered). Brunel "
                        "has static synapses, so this is the faithful configuration; it also "
                        "lets the framework intern `hyperparameters`, which STDP kernels "
                        "otherwise write.")
    p.add_argument("--synapse-stepping", choices=("clocked", "event"), default="event",
                   help="event (default): synapses stepped only after their pre fired; "
                        "clocked: every synapse every tick (the same kernels, for comparison).")
    p.add_argument("--fused", action="store_true",
                   help="Leave verbose_timing off so a single worker fuses all ticks into one "
                        "kernel launch (the path a user gets). The per-tick number is then "
                        "simulate_wall / ticks (per_tick_ms_fused); the first-tick sub-timers "
                        "are unavailable.")
    p.add_argument("--synapse-order", choices=("file", "pre"), default="file",
                   help="pre: reorder the partition's synapse rows by pre-soma so an emitter's "
                        "efferents are contiguous (coalesced wake-list access).")
    p.add_argument("--soma-breed", default="lif_soma")
    p.add_argument("--synapse-breed", default="delivery_synapse")
    p.add_argument("--in-memory", action="store_true",
                   help="Skip the .npz round-trip: generate the partition arrays and "
                        "hand them straight to the columnar builder. Isolates what "
                        "serialization actually costs.")
    return p


def run_once(args, repeat):
    import cupy as cp
    from superneuroabm.model import NeuromorphicModel
    from superneuroabm.brunel import save_brunel_partition, brunel_partition

    log = bc.PhaseLog("sna", "columnar-inmem" if args.in_memory else "columnar",
                      args.neurons, args.in_degree, repeat=repeat)
    t_pipeline = time.time()

    # ---- 1. Generate the partition (rank-local, columnar .npz) ----
    part_dir = Path(args.partition_dir or (SCRIPT_DIR / "partitions" /
                    f"1w_{args.neurons}n_K{args.in_degree}_global_rauto"))
    partition_file = part_dir / "partition_0.npz"
    C_E = round(0.8 * args.in_degree)
    C_I = args.in_degree - C_E

    gen_kwargs = dict(
        somas_per_rank=args.neurons,
        num_partitions=1,
        partition_rank=0,
        excitatory_in_degree=C_E,
        inhibitory_in_degree=C_I,
        # Random two-pool connectivity: C_E sources from the excitatory population and
        # C_I from the inhibitory one, drawn over the whole population. That is exactly
        # what brunel_nestgpu.py's two fixed_indegree calls build, so this file and the
        # NEST one measure the same network. The spatially-local torus3d belongs to the
        # scaling campaigns (weak_scaling.py --topology), where bounding the peer count is
        # the point; it is not comparable to NEST and at N=12500 it saturates the network
        # (162 Hz, CV 0.10) rather than settling into a balanced state (34.5 Hz, CV 0.34).
        topology="global",
        inhibitory_weight_ratio=args.g,
        excitatory_weight=args.J_E,
        external_weight=args.external_weight,
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
    model = NeuromorphicModel(enable_internal_states_tracking=False,
                              synapse_stepping=args.synapse_stepping,
                              **({"learning_rule_info": {}} if args.no_learning_rules else {}))
    # The partition carries its own breed names (brunel_partition writes lif_soma /
    # delivery_synapse) and the loader lets them win over its arguments, so a breed
    # override rewrites the fields in memory first.
    override = (args.soma_breed != "lif_soma" or args.synapse_breed != "delivery_synapse"
                or args.synapse_order != "file")
    if args.in_memory or override:
        if arrays is None:
            with np.load(str(partition_file), allow_pickle=True) as npz:
                arrays = {k: (v.item() if v.ndim == 0 else v) for k, v in npz.items()}
        arrays["soma_breed"] = args.soma_breed
        arrays["synapse_breed"] = args.synapse_breed
        if args.synapse_order == "pre":
            # external synapses (pre == -1) first, then efferent runs per pre-soma
            order = np.argsort(np.asarray(arrays["pre"]), kind="stable")
            for k in ("synapse_ids", "pre", "post", "syn_hp_vals"):
                arrays[k] = np.asarray(arrays[k])[order]
        # Same builder load_post_owned dispatches to for a .npz, minus the file.
        model.load_post_owned(arrays)
    else:
        model.load_post_owned(str(partition_file))
    # verbose_timing is required for the first-tick sub-timers. It also unfuses
    # the single-worker tick path, which inflates per-tick cost; that is why the
    # per-tick number here is reported as one aggregate and not analysed.
    model.verbose_timing = not args.fused
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
    drive_rate = args.external_rate
    n_ext = args.external_inputs if args.external_inputs is not None else C_E
    t0 = time.time()
    bc.inject_poisson_drive(model, input_synapses, drive_rate,
                            args.ticks, args.dt_ms, args.seed, 0,
                            inputs_per_soma=n_ext)
    log.add("inject_drive", time.time() - t0)
    lam = bc.drive_lambda(drive_rate, args.dt_ms, n_ext)
    print(f"  drive: {n_ext} external inputs x {drive_rate} Hz -> "
          f"lambda={lam:.3f} spikes/tick/soma")

    # ---- 5. Simulate ----
    t0 = time.time()
    model.simulate(ticks=args.ticks, update_data_ticks=1)
    sim_wall = time.time() - t0
    # Network activity (all somas are recorded by default): lets a run say whether the
    # per-tick numbers describe a silent or a firing network.
    spikes_by_soma = model.get_all_spike_times()
    n_out_spikes = sum(len(v) for v in spikes_by_soma.values())
    log.add("output_spikes", float(n_out_spikes))

    # Firing rate and ISI irregularity. A timing number measured on a silent network
    # describes the idle scan and nothing else, so both go in the CSV next to it --
    # `output_spikes: 0` was in the results for months without anyone reading it.
    duration_s = args.ticks * args.dt_ms / 1000.0
    rate_hz = n_out_spikes / args.neurons / duration_s if duration_s > 0 else 0.0

    # Steady state only, measured over the second half. The onset transient is large --
    # the population fires at several times its asymptotic rate for the first ~20 ticks
    # while recurrent inhibition builds up -- so a whole-run average of a short run
    # reports the transient, not the regime. Measured: 132 Hz whole-run vs 162 Hz
    # steady-state at N=12500, and 19.1 vs 18.5 at N=2500.
    half = args.ticks // 2
    steady_s = (args.ticks - half) * args.dt_ms / 1000.0
    n_steady, cvs = 0, []
    for ts in spikes_by_soma.values():
        t = np.sort(np.asarray(ts, dtype=np.float64))
        t = t[t >= half]
        n_steady += t.size
        if t.size >= 3:
            isi = np.diff(t)
            if isi.size and isi.mean() > 0:
                cvs.append(isi.std() / isi.mean())
    rate_steady = n_steady / args.neurons / steady_s if steady_s > 0 else 0.0
    cv_isi = float(np.mean(cvs)) if cvs else 0.0
    log.add("firing_rate_hz", rate_hz)
    log.add("firing_rate_steady_hz", rate_steady)
    log.add("cv_isi", cv_isi)
    log.add("cv_isi_n_somas", float(len(cvs)))
    print(f"  output spikes (all {len(model._soma_ids) or 'columnar'} somas, {args.ticks} ticks): {n_out_spikes:,}")
    print(f"  firing rate: {rate_hz:.2f} Hz/soma whole run, "
          f"{rate_steady:.2f} Hz steady state (last {args.ticks - half} ticks); "
          f"CV_ISI {cv_isi:.3f} over {len(cvs)} somas with >=3 spikes")
    if n_out_spikes == 0:
        print("  WARNING: network is SILENT -- per-tick timings describe the idle scan only.")

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
    if args.fused:
        log.add("per_tick_ms_fused", sim_wall / args.ticks * 1000.0)
        print(f"  fused: {sim_wall / args.ticks * 1000.0:.3f} ms/tick over {args.ticks} ticks "
              "(includes the first tick's buffer build)")
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
