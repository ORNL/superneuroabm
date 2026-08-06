"""
Strong scaling test - Brunel balanced random network (NEST hpc_benchmark style).

Strong scaling holds the TOTAL problem size fixed (total neurons N and in-degree K)
while growing the worker count, so per-rank work (N/P neurons x K synapses) shrinks
and the ideal runtime curve is ~1/P. This is the companion to weak_scaling.py, run on
the same network family (3D-torus spatial stencil) so the two curves describe one code
under one wiring convention.

Why the same wiring convention. SAGESim's per-tick ghost exchange is POINT-TO-POINT
(one Isend/Irecv per peer, CommunicationManager.mpi_exchange), so per-tick cost is
    (distinct remote peers) x latency  +  (ghost volume) x 1/bandwidth
-- both connectivity-indexed, unlike NEST's connectivity-agnostic collective. That is
why the benchmark network is a spatial stencil (topology="torus3d") rather than the
global-uniform draw: with global wiring a rank's peer count grows toward P-1 and the
measurement reports the wiring rather than the machine.

The asymmetry with weak scaling, and the point of the experiment. Weak scaling holds
the per-rank tile size constant, so the halo/volume ratio and the peer count are both
constant. Strong scaling shrinks the tile against a FIXED connection radius, so

  * local volume falls as 1/P but the halo surface only as P^(-2/3): the comm/compute
    ratio grows as P^(1/3) -- the textbook reason neighbor-exchange codes saturate; and
  * once the tile edge drops below the connection radius the bounded-peer premise stops
    holding altogether and the peer count climbs past 26.

Both effects are measured, not avoided; --dry-run reports where the second one starts
for a given (N, P) before any allocation is spent.

Phases:
  [1/4] Each rank generates its own partition file (Brunel, local-only).
  [2/4] Each rank loads its partition via model.load_post_owned().
  [3/4] GPU setup.
  [4/4] Inject Poisson external drive + simulate.
"""

import sys
import time
import math
import argparse
import csv
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from superneuroabm.brunel import (save_brunel_partition, _grid_factorization,
                                  _grid_factorization_3d, _factor_near_cube,
                                  _ball_offsets, _radius_for_indegree)
# Shared with weak_scaling.py on purpose: both campaigns must measure identically, or their
# numbers stop being comparable. No warm-up window is applied at collection -- see the module.
import scaling_diagnostics as diagnostics

# Run-summary schema. IDENTICAL in weak_scaling.py -- the two campaigns' summary CSVs must be
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


def predicted_peer_count(num_workers, neurons_per_worker, connection_radius):
    """Distinct Moore-neighbour tiles a rank's stencil reaches on the a x b x c torus.

    The stencil spans ``ceil(radius / tile_edge)`` tiles either side along each axis, but
    a torus dimension smaller than that span wraps onto itself, so the reachable count per
    axis is ``min(2*span + 1, dim)``.

    ACCURACY: this counts a bounding BOX of tiles, while the stencil is a SPHERE, so it is
    **exact only while the span is one tile** and an **upper bound** beyond that (the corner
    tiles of the box lie outside the radius and are never reached). Exact for every point of
    the weak-scaling campaign (0, 1, 3, 7, 11, 17, 26, ... for w = 1, 2, 4, ...) and for the
    strong-scaling ramp and plateau; in the breakdown regime it over-predicts -- measured
    44 / 62 / 96 at P = 512 / 1024 / 2048 against 44 / 74 / 124 predicted. Use it to size and
    sanity-check a sweep, not as a substitute for the measured ``peers_mean``.

    The low-P ramp comes from the RANK grid still having a dimension < 3; the high-P climb
    past 26 from the TILE edge falling under the radius.
    """
    a, b, c = _grid_factorization_3d(num_workers)
    gx, gy, gz = _factor_near_cube(neurons_per_worker)
    span = [math.ceil(connection_radius / e) for e in (gx, gy, gz)]
    reach = [min(2 * s + 1, d) for s, d in zip(span, (a, b, c))]
    return reach[0] * reach[1] * reach[2] - 1


def preflight(num_workers, total_neurons, in_degree, connection_radius=None):
    """Geometry + sizing for one (N, P) point, or a dict with ``error`` set.

    Strong scaling changes neurons-per-worker at every point, so every point re-derives a
    different tile shape, global grid and peer count. This computes all of it from the real
    generator helpers with no MPI, GPU or disk, which is what makes ``--dry-run`` able to
    reject a bad ``--total-neurons`` for free.
    """
    if total_neurons % num_workers:
        return {"workers": num_workers,
                "error": f"{total_neurons} neurons is not divisible by {num_workers} workers"}
    npp = total_neurons // num_workers
    a, b, c = _grid_factorization_3d(num_workers)
    gx, gy, gz = _factor_near_cube(npp)
    A, B, C = a * gx, b * gy, c * gz
    radius = connection_radius if connection_radius else _radius_for_indegree(in_degree)
    ball = int(_ball_offsets(radius).shape[0])

    info = {
        "workers": num_workers, "neurons_per_worker": npp,
        "rank_grid": (a, b, c), "tile": (gx, gy, gz), "global_grid": (A, B, C),
        "radius": radius, "ball": ball,
        "synapses_per_rank": npp * in_degree + npp,
        "peers": predicted_peer_count(num_workers, npp, radius),
        "error": None,
    }
    if radius >= min(A, B, C):
        info["error"] = (f"connection_radius={radius:g} must be < the smallest global grid "
                         f"dimension min({A},{B},{C}); the torus would wrap a source back "
                         f"onto its own target")
    elif ball < in_degree:
        info["error"] = (f"radius={radius:g} yields only {ball} candidates in the ball, "
                         f"fewer than in_degree={in_degree}")
    return info


def print_preflight_table(worker_list, total_neurons, in_degree, connection_radius, ranks_per_node):
    """Print the sizing table for a whole sweep. Returns True if every point is valid."""
    infos = [preflight(w, total_neurons, in_degree, connection_radius) for w in worker_list]

    print("=" * 108)
    print("STRONG-SCALING PREFLIGHT  (dry run -- no MPI, no GPU, no disk)")
    print("=" * 108)
    print(f"Total neurons (fixed): {total_neurons:,}      in-degree K: {in_degree}      "
          f"topology: torus3d      ranks/node: {ranks_per_node}")
    print()
    header = (f"{'P':>6} {'nodes':>6} {'neurons/GPU':>12} {'syn/rank':>12} {'rank grid':>12} "
              f"{'tile':>14} {'global grid':>16} {'R':>5} {'ball':>6} {'peers':>6}")
    print(header)
    print("-" * len(header))
    all_ok = True
    for info in infos:
        if "neurons_per_worker" not in info:
            print(f"{info['workers']:>6}  SKIP -- {info['error']}")
            all_ok = False
            continue
        nodes = -(-info["workers"] // ranks_per_node)
        grid = "x".join(str(v) for v in info["rank_grid"])
        tile = "x".join(str(v) for v in info["tile"])
        gg = "x".join(str(v) for v in info["global_grid"])
        print(f"{info['workers']:>6} {nodes:>6} {info['neurons_per_worker']:>12,} "
              f"{info['synapses_per_rank']:>12,} {grid:>12} {tile:>14} {gg:>16} "
              f"{info['radius']:>5.0f} {info['ball']:>6,} {info['peers']:>6}")
        if info["error"]:
            print(f"       ^^ INVALID: {info['error']}")
            all_ok = False
    print("-" * len(header))

    # The three regimes are the headline result: name them rather than leaving them implicit
    # in the peer column. Ramp and plateau are set by the RANK grid (a dimension < 3 collapses
    # the torus wraparound); the climb past 26 is set by the TILE edge falling under the radius.
    valid = [i for i in infos if "neurons_per_worker" in i and not i["error"]]
    for label, chosen, why in (
        ("ramp", [i for i in valid if i["peers"] < 26], "rank grid still has a dimension < 3"),
        ("plateau at 26", [i for i in valid if i["peers"] == 26], "full 3D Moore neighbourhood"),
        ("breakdown", [i for i in valid if i["peers"] > 26],
         "tile edge < connection radius; bounded-peer premise no longer holds"),
    ):
        if chosen:
            points = ", ".join(f"{i['workers']}({i['peers']})" for i in chosen)
            print(f"  {label:<14} P = {points}   -- {why}")
    print("=" * 108)
    return all_ok



def main():
    parser = argparse.ArgumentParser(
        description="Strong scaling test - Brunel balanced network (fixed total problem)")
    parser.add_argument("--total-neurons", type=int, default=204800,
                        help="Total neurons N, held CONSTANT for strong scaling. Must be "
                             "divisible by the worker count. The default sweeps P=16..2048 "
                             "at 8 ranks/node with the P=16 baseline at 12,800 neurons/GPU "
                             "(12.8 M synapses/rank at K=1000 -- the per-rank load proven "
                             "safe at 8 ranks/node in the weak campaign).")
    parser.add_argument("--in-degree", type=int, default=1000,
                        help="Fixed in-degree K per neuron, held constant across the sweep")
    parser.add_argument("--topology",
                        choices=["global", "bounded", "torus2d", "torus3d"], default="torus3d",
                        help="Recurrent wiring. 'torus3d' (default, matches the weak-scaling "
                             "campaign): ranks tile a periodic a x b x c torus and each neuron "
                             "draws its K sources within --connection-radius of its own 3D "
                             "position. Under strong scaling the tile SHRINKS against that "
                             "fixed radius, so the halo/volume ratio grows as P^(1/3) and, "
                             "once the tile edge falls below the radius, the peer count climbs "
                             "past 26. 'global': whole population, the O(P)-peer contrast. "
                             "'bounded': own rank + --remote-ranks random peers. 'torus2d': "
                             "the 2D analog, hard-constant 8 peers.")
    parser.add_argument("--remote-ranks", type=int, default=None,
                        help="Bounded remote-rank fanout R (for --topology bounded): each rank "
                             "draws sources from its own rank + exactly R random remote ranks.")
    parser.add_argument("--global-uniform", action="store_true",
                        help="Force the global-uniform draw (whole population); the contrast "
                             "baseline whose peer count tilts toward P-1. Overrides "
                             "--remote-ranks and --topology.")
    parser.add_argument("--connection-radius", type=float, default=None,
                        help="For --topology torus3d: Euclidean spatial radius (neuron-grid "
                             "units) of each neuron's presynaptic draw. None auto-selects the "
                             "smallest radius whose ball holds >= 2*K neurons. Held FIXED "
                             "across a strong-scaling sweep -- it is the fixed stencil against "
                             "which the shrinking tile is measured.")
    parser.add_argument("--g", type=float, default=5.0,
                        help="Relative inhibitory strength |J_I|/J_E (>1 = inhibition-dominated)")
    parser.add_argument("--J-E", dest="J_E", type=float, default=0.02581,
                        help="Excitatory synaptic weight (J_I = -g*J_E). Default is the "
                             "AI-regime value from calibrate_ai.py used by the weak campaign; "
                             "the config default of 14 is far too strong.")
    parser.add_argument("--delay", type=float, default=1.5,
                        help="Synaptic delay (ms)")
    parser.add_argument("--firing-rate", type=float, default=10.0,
                        help="External Poisson drive per neuron (Hz)")
    parser.add_argument("--external-rate", type=float, default=None,
                        help="AI-calibrated external Poisson rate (Hz) from calibrate_ai.py. "
                             "Overrides --firing-rate when set.")
    parser.add_argument("--external-weight", type=float, default=None,
                        help="Aggregate external synapse weight (C_E*J) from calibrate_ai.py; "
                             "None inherits the excitatory weight.")
    parser.add_argument("--diagnostics", action="store_true",
                        help="Collect + log per-rank comm and per-step metrics. Strong-scaling "
                             "runs want this ON: the per-step decomposition and the "
                             "surface-to-volume ratio are what explain where the curve "
                             "saturates, and the comm-free bound is derived from them.")
    parser.add_argument("--dt-ms", type=float, default=1.0,
                        help="Milliseconds per simulation tick (for Poisson rate)")
    parser.add_argument("--ticks", type=int, default=30,
                        help="Simulation ticks (30 matches the weak-scaling campaign)")
    parser.add_argument("--update-ticks", type=int, default=1,
                        help="Update data every N ticks")
    parser.add_argument("--dump-per-rank-ticks", action="store_true",
                        help="Also write the full per-rank x per-tick array as .npz beside the "
                             "per-tick CSV (~12 MB compressed at 2048 ranks). The CSV keeps "
                             "across-rank mean/max/min, which is enough for every figure; this "
                             "preserves the per-rank distribution in case it is wanted later, "
                             "when re-running costs hundreds of node-hours.")
    parser.add_argument("--dry-run", type=int, nargs="*", default=None, metavar="P",
                        help="Print the geometry/sizing preflight and exit WITHOUT touching "
                             "MPI, the GPU or disk. With no values it reports the current "
                             "worker count; with values it reports that whole sweep, e.g. "
                             "--dry-run 16 32 64 128 256 512 1024 2048.")
    parser.add_argument("--ranks-per-node", type=int, default=8,
                        help="Ranks per node, used only to print the node column in --dry-run")
    parser.add_argument("--partition-dir", type=str, default=None,
                        help="Directory for partition files. Defaults to "
                             "./partitions/strong_{N}n_{P}w_K{K}_{wiring}.")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to CSV file for appending timing results (rank 0 only)")
    args = parser.parse_args()

    total_neurons = args.total_neurons
    K = args.in_degree
    simulation_ticks = args.ticks
    update_ticks = args.update_ticks
    SEED = 42

    # ---- Dry run: geometry only, before any allocation is spent ----
    if args.dry_run is not None:
        worker_list = args.dry_run if args.dry_run else [size]
        ok = print_preflight_table(worker_list, total_neurons, K,
                                   args.connection_radius, args.ranks_per_node)
        sys.exit(0 if ok else 1)

    # Wiring topology. --global-uniform still wins for backward compatibility; otherwise
    # --topology decides. A single-worker run has no remote ranks, so topologies that need
    # a peer degrade to the trivial local draw.
    topology = "global" if args.global_uniform else args.topology
    remote_rank_fanout = None
    grid_ab = None
    grid_abc = None
    if size == 1 and topology in ("bounded", "torus2d"):
        topology = "global"
    if topology == "bounded":
        R = args.remote_ranks if args.remote_ranks is not None else 1
        remote_rank_fanout = min(R, size - 1)
        if remote_rank_fanout != R and rank == 0:
            print(f"[warn] --remote-ranks {R} > size-1; "
                  f"clamped to {remote_rank_fanout} for {size} workers.")
    elif topology == "torus2d":
        try:
            grid_ab = _grid_factorization(size)
        except ValueError as exc:
            if rank == 0:
                print(f"[error] --topology torus2d with {size} workers: {exc}")
            sys.exit(1)
    elif topology == "torus3d":
        grid_abc = _grid_factorization_3d(size)

    # Strong scaling's defining constraint: N is fixed, so the worker count must divide it.
    if total_neurons % size != 0:
        if rank == 0:
            print(f"ERROR: --total-neurons ({total_neurons}) must be divisible by "
                  f"the worker count ({size}). Nearest usable totals: "
                  f"{total_neurons // size * size} or {(total_neurons // size + 1) * size}.")
        sys.exit(1)
    neurons_per_worker = total_neurons // size

    # Fail fast on an unbuildable geometry with the full sizing line, rather than dying
    # deep inside partition generation on one rank.
    if topology == "torus3d":
        info = preflight(size, total_neurons, K, args.connection_radius)
        if info["error"]:
            if rank == 0:
                print(f"[error] torus3d geometry invalid at P={size}, "
                      f"N={total_neurons:,}, K={K}: {info['error']}")
            sys.exit(1)

    drive_rate = args.external_rate if args.external_rate is not None else args.firing_rate

    if rank == 0:
        print("=" * 70)
        print("STRONG SCALING TEST - BRUNEL BALANCED NETWORK")
        print("=" * 70)
        if comm is not None:
            print(f"[DEBUG] MPI world size = {comm.Get_size()}, "
                  f"SLURM_STEP_NUM_TASKS = {os.environ.get('SLURM_STEP_NUM_TASKS', 'NOT SET')}, "
                  f"using size = {size}")
        print(f"Workers: {size}")
        print(f"Total neurons (FIXED): {total_neurons:,}")
        print(f"Neurons per worker: {neurons_per_worker:,}  (= N / {size})")
        print(f"In-degree K: {K}")
        print(f"g (|J_I|/J_E): {args.g}   J_E: {args.J_E}   J_I: {-args.g * args.J_E}")
        wiring = {
            "global": "global-uniform (whole population)",
            "bounded": f"bounded fanout R={remote_rank_fanout}",
            "torus2d": (f"2D-torus tile-block ({grid_ab[0]}x{grid_ab[1]} grid, 8 neighbors)"
                        if grid_ab else "2D-torus tile-block"),
            "torus3d": (f"3D-torus spatial-radius stencil "
                        f"({grid_abc[0]}x{grid_abc[1]}x{grid_abc[2]} rank grid, radius="
                        f"{args.connection_radius if args.connection_radius else 'auto'})"
                        if grid_abc else "3D-torus spatial-radius stencil"),
        }[topology]
        print(f"Wiring: {wiring}")
        if topology == "torus3d":
            print(f"  tile: {'x'.join(str(v) for v in info['tile'])}   "
                  f"global grid: {'x'.join(str(v) for v in info['global_grid'])}   "
                  f"radius: {info['radius']:g}   ball: {info['ball']:,}   "
                  f"predicted peers: {info['peers']}")
        print(f"Synaptic delay: {args.delay} ms")
        print(f"External drive: {drive_rate} Hz (Poisson)"
              + (f", external_weight={args.external_weight}"
                 if args.external_weight is not None else ""))
        print(f"Diagnostics: {'ON' if args.diagnostics else 'OFF'}")
        print(f"Synapses per worker: ~{neurons_per_worker * K + neurons_per_worker:,} "
              f"(={neurons_per_worker}*{K} recurrent + {neurons_per_worker} external)")
        print(f"Simulation ticks: {simulation_ticks}")
        print("=" * 70)

    # Imported here, not at module scope, so --dry-run works on a login node with no GPU.
    from superneuroabm.model import NeuromorphicModel

    t_pipeline_start = time.time()

    # ---- [1/4] Generate partition file ----
    if rank == 0:
        print("\n[1/4] Generating partition files...")

    if args.partition_dir:
        partition_dir = args.partition_dir
    else:
        # Shared filesystem (not /tmp, which is node-local on Frontier). N, P, K and the
        # wiring all go in the cache key: under strong scaling the per-rank content changes
        # with EVERY one of them, so a shorter name would silently reuse a wrong network.
        radius_tag = (f"r{args.connection_radius:g}" if args.connection_radius else "rauto")
        wiring_tag = {"global": "global", "bounded": f"r{remote_rank_fanout}",
                      "torus2d": "torus2d",
                      "torus3d": f"torus3d_{radius_tag}"}[topology]
        script_dir = Path(__file__).parent
        partition_dir = str(script_dir / "partitions"
                            / f"strong_{total_neurons}n_{size}w_K{K}_{wiring_tag}")

    t0 = time.time()
    # Columnar (.npz) partitions: the row-oriented .pkl inflates ~30x in host RAM during
    # load and OOM-kills at >=4 ranks/node. The columnar encoding is read straight into
    # SAGESim's tensors + CSR with no per-synapse Python objects.
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
    # The PROPERTY, not the private attribute: the setter also propagates the flag to the
    # agent factory, which is where half the per-tick timers live.
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
    # Every neuron has one external (pre == -1) input synapse from the generator. The
    # columnar loader exposes those ids directly (vectorized); fall back to scanning the
    # synapse-id set for the record (.pkl) path.
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
    total_synapses = getattr(model, "_num_synapses", None)
    if total_synapses is None:
        total_synapses = len(model._synapse_ids)

    # ---- Diagnostics: record every tick truthfully; choose no window here ----
    # The per-tick CSV is the source of truth. Which leading ticks count as warm-up is an
    # ANALYSIS decision (analyze_strong.py --warmup-ticks), because baking it in here once
    # hid a second warm-up tick behind an already-excluded first one.
    facts = None
    if args.diagnostics:
        # Both are COLLECTIVE -- every rank must call them, in this order, on every rank.
        tick_rows = diagnostics.collect_tick_records(model, comm, rank)
        facts = diagnostics.topology_facts(model, comm, rank, neurons_per_worker)

        # Path is derived identically on every rank (job id + run parameters), so the
        # collective .npz dump below does not need it broadcast.
        outdir = Path(args.csv).parent if args.csv else Path(__file__).parent / "outputs"
        tick_path = diagnostics.tick_csv_path(outdir, "strong",
                                              f"N{total_neurons}_K{K}_w{size}")
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
        print(f"Neurons per worker: {neurons_per_worker:,}")
        print(f"Local synapses: {total_synapses:,}")
        print(f"Generation time: {generation_time:.3f}s")
        print(f"Model load time: {model_creation_time:.3f}s")
        print(f"GPU setup time: {gpu_setup_time:.3f}s")
        print(f"Construction time: {construction_time:.3f}s")
        print(f"Simulation time: {sim_time:.3f}s")
        print(f"Total wall time: {total_time:.3f}s")
        print("=" * 70)
        print("SUCCESS - Brunel strong scaling run completed!")
        print("=" * 70)

    # Write CSV timing results (rank 0 only)
    if args.csv and rank == 0:
        # The run summary holds FACTS ONLY: identity, whole-run phase times, and partition
        # properties. Every per-tick and steady-state quantity is DERIVED and lives in the
        # per-tick CSV instead, so the warm-up window can be revisited by re-analysis rather
        # than by re-running. Keep this schema identical to weak_scaling.py's.
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
