"""
Weak scaling test - Brunel balanced random network (NEST hpc_benchmark style).

Weak scaling holds neurons-per-worker and in-degree K constant while growing the
worker count, so per-rank work (npp neurons x K synapses) stays constant and the
ideal runtime curve is flat. Connectivity is global fixed-in-degree (each neuron
draws K presynaptic sources from the whole population); cross-rank communication
emerges naturally from the partition -- there is no explicit cross-rank knob.

Phases:
  [1/4] Each rank generates its own partition file (Brunel, local-only).
  [2/4] Each rank loads its partition via model.load_post_owned().
  [3/4] GPU setup.
  [4/4] Inject Poisson external drive + simulate.
"""

import sys
import time
import argparse
import csv
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from superneuroabm.brunel import (save_brunel_partition, _grid_factorization,
                                   _grid_factorization_3d)
from superneuroabm.model import NeuromorphicModel
# Shared with strong_scaling.py on purpose: both campaigns must measure identically, or their
# numbers stop being comparable. No warm-up window is applied at collection -- see the module.
import scaling_diagnostics as diagnostics

# Run-summary schema. IDENTICAL in strong_scaling.py -- the two campaigns' summary CSVs must be
# interchangeable. Facts only: no derived per-tick quantity appears here (those live in the
# per-tick CSV, so the warm-up window stays an analysis-time choice).
SUMMARY_COLUMNS = [
    'job_id', 'nodes', 'workers', 'total_neurons', 'neurons_per_worker',
    'synapses', 'in_degree', 'topology', 'remote_ranks', 'connection_radius',
    'drive_rate', 'ticks',
    'generation_time', 'model_creation_time', 'gpu_setup_time',
    'construction_time', 'simulation_time', 'total_time',
    'peers_min', 'peers_mean', 'peers_max',
    'ghost_somas_mean', 'ghost_somas_max', 'ghost_local_ratio',
    'send_bytes_mean', 'bytes_per_peer',
]

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = int(os.environ.get('SLURM_STEP_NUM_TASKS', comm.Get_size()))
except ImportError:
    rank = 0
    size = 1
    comm = None



def main():
    parser = argparse.ArgumentParser(description="Weak scaling test - Brunel balanced network")
    parser.add_argument("--neurons-per-worker", type=int, default=12500,
                        help="Neurons per worker (constant for weak scaling; classic Brunel N)")
    parser.add_argument("--in-degree", type=int, default=4000,
                        help="Fixed in-degree K per neuron (Potjans/GPU standard; NEST=11250)")
    parser.add_argument("--topology",
                        choices=["global", "bounded", "torus2d", "torus3d"], default=None,
                        help="Recurrent wiring. 'torus3d' (3D weak-scaling stencil): ranks tile a "
                             "periodic a x b x c torus (near-cube, any worker count >= 1) and each "
                             "neuron draws its K sources within --connection-radius of its own 3D "
                             "position, so most edges are local (own tile) and only a bounded "
                             "surface halo crosses to neighbor tiles -- the peer count ramps to 26 "
                             "and plateaus (volume-local compute + surface comm). 'torus2d': the "
                             "2D analog, hard-constant 8 peers (needs an a,b>=3 grid). 'global': "
                             "whole population, the O(M) contrast baseline. 'bounded': own rank + "
                             "--remote-ranks random peers. Default infers from --global-uniform / "
                             "--remote-ranks.")
    parser.add_argument("--remote-ranks", type=int, default=None,
                        help="Bounded remote-rank fanout R (for --topology bounded): each rank "
                             "draws sources from its own rank + exactly R random remote ranks, "
                             "so its peer count is a hard constant == R. Omit (or "
                             "--global-uniform) for the whole-population contrast baseline.")
    parser.add_argument("--global-uniform", action="store_true",
                        help="Force the global-uniform draw (whole population); the "
                             "contrast baseline whose peer count tilts toward M-1. "
                             "Overrides --remote-ranks.")
    parser.add_argument("--connection-radius", type=float, default=None,
                        help="For --topology torus3d: Euclidean spatial radius (neuron-grid "
                             "units) of each neuron's presynaptic draw. Smaller radius = more "
                             "local (lighter halo). None auto-selects the smallest radius whose "
                             "ball holds >= 2*K neurons.")
    parser.add_argument("--g", type=float, default=5.0,
                        help="Relative inhibitory strength |J_I|/J_E (>1 = inhibition-dominated)")
    parser.add_argument("--J-E", dest="J_E", type=float, default=14.0,
                        help="Excitatory synaptic weight (J_I = -g*J_E). Use the value "
                             "calibrate_ai.py prints for the AI regime (config default 14 "
                             "is far too strong).")
    parser.add_argument("--delay", type=float, default=1.5,
                        help="Synaptic delay (ms)")
    parser.add_argument("--firing-rate", type=float, default=10.0,
                        help="External Poisson drive per neuron (Hz). Non-AI override; "
                             "prefer --external-rate from calibrate_ai.py for the AI regime.")
    parser.add_argument("--external-rate", type=float, default=None,
                        help="AI-calibrated external Poisson rate (Hz) from calibrate_ai.py. "
                             "Overrides --firing-rate when set.")
    parser.add_argument("--external-weight", type=float, default=None,
                        help="Aggregate external synapse weight (C_E*J) from calibrate_ai.py; "
                             "None inherits the excitatory weight.")
    parser.add_argument("--diagnostics", action="store_true",
                        help="Collect + log per-rank comm diagnostics (mpi_num_peers, "
                             "comm fraction, ghost volume). OFF by default so real timing "
                             "runs do not pay for metric collection.")
    parser.add_argument("--dump-per-rank-ticks", action="store_true",
                        help="Also write the full per-rank x per-tick array as .npz beside the "
                             "per-tick CSV (~12 MB compressed at 2048 ranks). The CSV keeps "
                             "across-rank mean/max/min, which is enough for every figure; this "
                             "preserves the per-rank distribution in case it is wanted later, "
                             "when re-running costs hundreds of node-hours.")
    parser.add_argument("--dt-ms", type=float, default=1.0,
                        help="Milliseconds per simulation tick (for Poisson rate)")
    parser.add_argument("--ticks", type=int, default=100,
                        help="Simulation ticks (100 matches the strong-scaling "
                             "campaign; the per-tick metric is tick-count "
                             "independent, but equal sample sizes keep the two "
                             "campaigns' statistics comparable)")
    parser.add_argument("--update-ticks", type=int, default=1,
                        help="Update data every N ticks")
    parser.add_argument("--partition-dir", type=str, default=None,
                        help="Directory for partition files. Defaults to ./partitions/{size}w_{n}n.")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to CSV file for appending timing results (rank 0 only)")
    args = parser.parse_args()

    neurons_per_worker = args.neurons_per_worker
    K = args.in_degree
    simulation_ticks = args.ticks
    update_ticks = args.update_ticks
    SEED = 42

    # Wiring topology. --topology wins; else infer from the legacy --global-uniform /
    # --remote-ranks flags (backward compatible). A single-worker run has no remote ranks,
    # so it always uses the trivial global path regardless of the requested topology.
    if args.topology is not None:
        topology = args.topology
    elif args.global_uniform:
        topology = "global"
    elif args.remote_ranks is not None:
        topology = "bounded"
    else:
        topology = "global"

    remote_rank_fanout = None
    grid_ab = None
    grid_abc = None
    if size == 1 and topology in ("bounded", "torus2d"):
        # These need >= 1 remote peer / an a,b>=3 grid; a single worker has neither, so
        # degrade to the trivial local draw. (global and torus3d run natively at size 1 --
        # torus3d M=1 is the 1-GPU weak-scaling baseline: all sources local, no MPI.)
        topology = "global"
    if topology == "bounded":
        # Clamp R to size-1 so a sweep that starts below R (e.g. 1-2 GPUs with R=4) still
        # runs instead of erroring -- the driver picks the largest valid fanout.
        R = args.remote_ranks if args.remote_ranks is not None else 1
        remote_rank_fanout = min(R, size - 1)
        if remote_rank_fanout != R and rank == 0:
            print(f"[warn] --remote-ranks {R} > size-1; "
                  f"clamped to {remote_rank_fanout} for {size} workers.")
    elif topology == "torus2d":
        # Fail fast (clearly) if the worker count can't form a valid a,b>=3 torus grid,
        # instead of dying deep inside partition generation.
        try:
            grid_ab = _grid_factorization(size)
        except ValueError as exc:
            if rank == 0:
                print(f"[error] --topology torus2d with {size} workers: {exc}")
            sys.exit(1)
    elif topology == "torus3d":
        # Any worker count >= 1 forms a valid near-cube grid (ramp-and-plateau); no fail-fast.
        grid_abc = _grid_factorization_3d(size)
    # Drive rate: AI-calibrated external rate if given, else the plain --firing-rate.
    drive_rate = args.external_rate if args.external_rate is not None else args.firing_rate

    if rank == 0:
        print("=" * 70)
        print("WEAK SCALING TEST - BRUNEL BALANCED NETWORK")
        print("=" * 70)
        if comm is not None:
            print(f"[DEBUG] MPI world size = {comm.Get_size()}, "
                  f"SLURM_STEP_NUM_TASKS = {os.environ.get('SLURM_STEP_NUM_TASKS', 'NOT SET')}, "
                  f"using size = {size}")
        print(f"Workers: {size}")
        print(f"Neurons per worker: {neurons_per_worker}")
        print(f"In-degree K: {K}")
        print(f"g (|J_I|/J_E): {args.g}   J_E: {args.J_E}   J_I: {-args.g * args.J_E}")
        wiring = {
            "global": "global-uniform (whole population)",
            "bounded": f"bounded fanout R={remote_rank_fanout}",
            "torus2d": (f"2D-torus tile-block ({grid_ab[0]}x{grid_ab[1]} grid, 8 neighbors)"
                        if grid_ab else "2D-torus tile-block"),
            "torus3d": (f"3D-torus spatial-radius stencil "
                        f"({grid_abc[0]}x{grid_abc[1]}x{grid_abc[2]} grid, radius="
                        f"{args.connection_radius if args.connection_radius else 'auto'}, "
                        f"<=26 neighbors)" if grid_abc else "3D-torus spatial-radius stencil"),
        }[topology]
        print(f"Wiring: {wiring}")
        print(f"Synaptic delay: {args.delay} ms")
        print(f"External drive: {drive_rate} Hz (Poisson)"
              + (f", external_weight={args.external_weight}"
                 if args.external_weight is not None else ""))
        print(f"Diagnostics: {'ON' if args.diagnostics else 'OFF'}")
        print(f"Total neurons: {size * neurons_per_worker:,}")
        print(f"Synapses per worker: ~{neurons_per_worker * K + neurons_per_worker:,} "
              f"(={neurons_per_worker}*{K} recurrent + {neurons_per_worker} external)")
        print(f"Simulation ticks: {simulation_ticks}")
        print("=" * 70)

    t_pipeline_start = time.time()

    # ---- [1/4] Generate partition file ----
    if rank == 0:
        print("\n[1/4] Generating partition files...")

    if args.partition_dir:
        partition_dir = args.partition_dir
    else:
        # Shared filesystem (not /tmp, which is node-local on Frontier). The wiring tag
        # keeps torus2d / bounded-R / global-uniform partitions from reusing each other's cache.
        radius_tag = (f"r{args.connection_radius:g}" if args.connection_radius else "rauto")
        wiring_tag = {"global": "global", "bounded": f"r{remote_rank_fanout}",
                      "torus2d": "torus2d",
                      "torus3d": f"torus3d_{radius_tag}"}[topology]
        script_dir = Path(__file__).parent
        # K is part of the cache key: auto-radius is a function of K, so a K-less name
        # would let a K=2000 run silently reuse K=4000 partitions (wrong network).
        partition_dir = str(script_dir / "partitions"
                            / f"{size}w_{neurons_per_worker}n_K{K}_{wiring_tag}")

    t0 = time.time()
    # Columnar (.npz) partitions: the row-oriented .pkl inflates ~30x in host RAM
    # during load and OOM-kills at >=4 ranks/node. The columnar encoding is read
    # straight into SAGESim's tensors + CSR with no per-synapse Python objects.
    partition_file = os.path.join(partition_dir, f"partition_{rank}.npz")
    if os.path.exists(partition_file):
        if rank == 0:
            print(f"    Partition files found in {partition_dir}, skipping generation.")
    else:
        # Map the driver's single in-degree K onto the two-pool Brunel split
        # (excitatory 4:1). C_E + C_I = K holds per soma.
        C_E = round(0.8 * K)
        C_I = K - C_E
        save_brunel_partition(
            output_dir=partition_dir,
            somas_per_rank=neurons_per_worker,
            num_partitions=size,
            partition_rank=rank,
            excitatory_in_degree=C_E,
            inhibitory_in_degree=C_I,
            topology=topology,
            remote_rank_fanout=remote_rank_fanout,
            connection_radius=args.connection_radius,
            inhibitory_weight_ratio=args.g,
            excitatory_weight=args.J_E,
            external_weight=args.external_weight,
            synaptic_delay_ms=args.delay,
            seed=SEED,
            output_format="columns",
        )
    generation_time = time.time() - t0
    if rank == 0:
        print(f"    Generation completed in {generation_time:.2f}s")

    # ---- [2/4] Load partition file ----
    if rank == 0:
        print("\n[2/4] Loading partition files...")

    t0 = time.time()
    model = NeuromorphicModel(enable_internal_states_tracking=False)
    model.load_post_owned(partition_file)
    # Diagnostics OFF by default: verbose_timing gates the engine's per-tick metric
    # collection (mpi_num_peers, ghost volume, comm seconds). Real timing runs leave it
    # off so the reported numbers don't include metric-collection overhead.
    model.verbose_timing = args.diagnostics
    model_creation_time = time.time() - t0
    if rank == 0:
        print(f"    Model loaded in {model_creation_time:.2f}s")

    # ---- [3/4] Setup GPU ----
    if rank == 0:
        print("\n[3/4] Setting up GPUs...")

    t0 = time.time()
    model.setup()
    gpu_setup_time = time.time() - t0
    if rank == 0:
        print(f"    GPU setup in {gpu_setup_time:.2f}s")

    # ---- [4/4] Poisson drive + simulate ----
    # Every neuron has one external (pre == -1) input synapse from the generator.
    # The columnar loader exposes the external-synapse ids directly (vectorized);
    # fall back to scanning the synapse-id set for the record (.pkl) path.
    if getattr(model, "_input_synapse_ids", None) is not None:
        input_synapses = model._input_synapse_ids.tolist()
    else:
        input_synapses = [sid for sid in model._synapse_ids
                          if model.get_synapse_connectivity(sid)[0] == -1]
    if rank == 0:
        print(f"\n    Injecting {drive_rate} Hz Poisson drive on "
              f"{len(input_synapses)} input synapses...")
    diagnostics.inject_poisson_drive(model, input_synapses, drive_rate,
                                     simulation_ticks, args.dt_ms, SEED, rank)

    if rank == 0:
        print("\n" + "=" * 70)
        print(f"RUNNING SIMULATION ({simulation_ticks} ticks)")
        print("=" * 70)

    start_time = time.time()
    model.simulate(ticks=simulation_ticks, update_data_ticks=update_ticks)
    wall_time = time.time() - start_time
    sim_time = getattr(model, '_simulation_time', wall_time)
    construction_time = getattr(model, '_construction_time', 0.0)

    total_time = time.time() - t_pipeline_start
    total_neurons = size * neurons_per_worker
    total_synapses = getattr(model, "_num_synapses", None)
    if total_synapses is None:
        total_synapses = len(model._synapse_ids)

    # ---- Diagnostics: record every tick truthfully; choose no window here ----
    # The per-tick CSV is the source of truth. Which leading ticks count as warm-up is an
    # ANALYSIS decision (analyze_weak.py --warmup-ticks). This driver previously averaged over
    # ALL ticks including the first, which is why its simulation_time was 99.9% buffer build.
    facts = None
    if args.diagnostics:
        # Both are COLLECTIVE -- every rank must call them, in this order, on every rank.
        tick_rows = diagnostics.collect_tick_records(model, comm, rank)
        facts = diagnostics.topology_facts(model, comm, rank, neurons_per_worker)

        outdir = Path(args.csv).parent if args.csv else Path(__file__).parent / "outputs"
        tick_path = diagnostics.tick_csv_path(outdir, "weak",
                                              f"K{K}_npp{neurons_per_worker}_w{size}")
        n_ticks = len(tick_rows) if (rank == 0 and tick_rows) else 0
        if comm is not None:
            n_ticks = comm.bcast(n_ticks, root=0)

        if rank == 0 and tick_rows:
            diagnostics.write_tick_csv(tick_path, tick_rows)
            print(f"\nPer-tick timing -> {tick_path}  ({len(tick_rows)} ticks)")
            for row in tick_rows[:3]:
                print(f"  tick {row['tick']}: total={row['total_mean']:.5f}s  "
                      f"comm={row['comm_mean']:.5f}s  compute={row['compute_mean']:.6f}s")

        if args.dump_per_rank_ticks and n_ticks:
            npz = diagnostics.dump_per_rank_ticks(   # collective
                str(tick_path)[:-4] + ".npz", model, comm, rank, n_ticks)
            if rank == 0 and npz:
                print(f"  per-rank array -> {npz}")

    if facts is not None and rank == 0:
        print("\n" + "-" * 70)
        print("TOPOLOGY (partition properties, constant across ticks)")
        print(f"  peers        : {facts['peers_min']}/{facts['peers_mean']:.1f}/{facts['peers_max']}"
              f" (min/mean/max)")
        print(f"  ghost somas  : {facts['ghost_somas_mean']:.0f}/{facts['ghost_somas_max']}"
              f" (mean/max)   ghost:local = {facts['ghost_local_ratio']:.2f}")
        print(f"  send bytes   : {facts['send_bytes_mean']:,.0f}/tick"
              f"   ({facts['bytes_per_peer']:,.0f} B per peer per tick)")
        print("-" * 70)

    if rank == 0:
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"Total neurons: {total_neurons:,} across {size} worker(s)")
        print(f"Local synapses: {total_synapses:,}")
        print(f"Generation time: {generation_time:.3f}s")
        print(f"Model load time: {model_creation_time:.3f}s")
        print(f"GPU setup time: {gpu_setup_time:.3f}s")
        print(f"Construction time: {construction_time:.3f}s")
        print(f"Simulation time: {sim_time:.3f}s")
        print(f"Total wall time: {total_time:.3f}s")
        print("=" * 70)
        print("SUCCESS - Brunel weak scaling run completed!")
        print("=" * 70)

    # Run summary: FACTS ONLY (identity, whole-run phases, partition properties). Every
    # per-tick and steady-state quantity is DERIVED and lives in the per-tick CSV, so the
    # warm-up window is revisited by re-analysis rather than by re-running. Schema is
    # identical to strong_scaling.py's -- see SUMMARY_COLUMNS.
    if args.csv and rank == 0:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not csv_path.exists()

        def f_(key, fmt="{}"):
            """Topology fact formatted, or '' when --diagnostics was off."""
            return fmt.format(facts[key]) if facts is not None else ''

        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(SUMMARY_COLUMNS)
            writer.writerow([
                os.environ.get('SLURM_JOB_ID', ''),
                os.environ.get('SLURM_NNODES', ''),
                size, total_neurons, neurons_per_worker, total_synapses, K,
                topology,
                ('' if remote_rank_fanout is None else remote_rank_fanout),
                ('' if args.connection_radius is None else f'{args.connection_radius:g}'),
                f'{drive_rate:.4f}', simulation_ticks,
                f'{generation_time:.4f}', f'{model_creation_time:.4f}',
                f'{gpu_setup_time:.4f}', f'{construction_time:.4f}',
                f'{sim_time:.4f}', f'{total_time:.4f}',
                f_('peers_min'), f_('peers_mean', '{:.2f}'), f_('peers_max'),
                f_('ghost_somas_mean', '{:.1f}'), f_('ghost_somas_max'),
                f_('ghost_local_ratio', '{:.4f}'),
                f_('send_bytes_mean', '{:.1f}'), f_('bytes_per_peer', '{:.1f}'),
            ])
        print(f"\nRun summary appended to {csv_path}")


if __name__ == "__main__":
    main()
