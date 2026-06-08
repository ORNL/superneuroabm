"""
Strong scaling test - Brunel balanced random network (NEST hpc_benchmark style).

Strong scaling holds the TOTAL problem size fixed (total neurons N and in-degree
K) while growing the worker count, so per-rank work (N/P neurons x K synapses)
shrinks and the ideal runtime curve is ~1/P. Connectivity is global
fixed-in-degree (each neuron draws K presynaptic sources from the whole
population); cross-rank communication emerges from the partition.

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

from superneuroabm.synthetic_networks import generate_and_save_local_partition
from superneuroabm.model import NeuromorphicModel

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = int(os.environ.get('SLURM_STEP_NUM_TASKS', comm.Get_size()))
except ImportError:
    rank = 0
    size = 1
    comm = None


def inject_poisson_drive(model, input_synapses, rate_hz, ticks, dt_ms, seed):
    """Schedule a Poisson spike train on each external (pre == -1) input synapse."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, rank, 7]))
    p_spike = rate_hz * dt_ms / 1000.0
    for syn_id in input_synapses:
        fires = rng.random(ticks) < p_spike
        spike_ticks = np.nonzero(fires)[0] + 1  # ticks are 1-indexed
        if spike_ticks.size:
            model.add_spike_list(syn_id, [[int(t), 1.0] for t in spike_ticks])


def main():
    parser = argparse.ArgumentParser(description="Strong scaling test - Brunel balanced network")
    parser.add_argument("--total-neurons", type=int, default=40000,
                        help="Total neurons N, held CONSTANT for strong scaling")
    parser.add_argument("--in-degree", type=int, default=1000,
                        help="Fixed in-degree K per neuron")
    parser.add_argument("--g", type=float, default=5.0,
                        help="Relative inhibitory strength |J_I|/J_E (>1 = inhibition-dominated)")
    parser.add_argument("--J-E", dest="J_E", type=float, default=14.0,
                        help="Excitatory synaptic weight (J_I = -g*J_E)")
    parser.add_argument("--delay", type=float, default=1.5,
                        help="Synaptic delay (ms)")
    parser.add_argument("--firing-rate", type=float, default=10.0,
                        help="External Poisson drive per neuron (Hz)")
    parser.add_argument("--dt-ms", type=float, default=1.0,
                        help="Milliseconds per simulation tick (for Poisson rate)")
    parser.add_argument("--ticks", type=int, default=10,
                        help="Simulation ticks")
    parser.add_argument("--update-ticks", type=int, default=1,
                        help="Update data every N ticks")
    parser.add_argument("--partition-dir", type=str, default=None,
                        help="Directory for partition files. Defaults to ./partitions/strong_{N}n_{size}w.")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to CSV file for appending timing results (rank 0 only)")
    args = parser.parse_args()

    total_neurons = args.total_neurons
    K = args.in_degree
    simulation_ticks = args.ticks
    update_ticks = args.update_ticks
    SEED = 42

    if total_neurons % size != 0:
        if rank == 0:
            print(f"ERROR: --total-neurons ({total_neurons}) must be divisible by "
                  f"the worker count ({size}).")
        sys.exit(1)
    neurons_per_worker = total_neurons // size

    if rank == 0:
        print("=" * 70)
        print("STRONG SCALING TEST - BRUNEL BALANCED NETWORK")
        print("=" * 70)
        if comm is not None:
            print(f"[DEBUG] MPI world size = {comm.Get_size()}, "
                  f"SLURM_STEP_NUM_TASKS = {os.environ.get('SLURM_STEP_NUM_TASKS', 'NOT SET')}, "
                  f"using size = {size}")
        print(f"Workers: {size}")
        print(f"Total neurons (fixed): {total_neurons:,}")
        print(f"Neurons per worker: {neurons_per_worker:,}  (= N / {size})")
        print(f"In-degree K: {K}")
        print(f"g (|J_I|/J_E): {args.g}   J_E: {args.J_E}   J_I: {-args.g * args.J_E}")
        print(f"Synaptic delay: {args.delay} ms")
        print(f"External drive: {args.firing_rate} Hz (Poisson)")
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
        script_dir = Path(__file__).parent
        partition_dir = str(script_dir / "partitions" / f"strong_{total_neurons}n_{size}w")

    t0 = time.time()
    partition_file = os.path.join(partition_dir, f"partition_{rank}.pkl")
    if os.path.exists(partition_file):
        if rank == 0:
            print(f"    Partition files found in {partition_dir}, skipping generation.")
    else:
        generate_and_save_local_partition(
            output_dir=partition_dir,
            my_rank=rank,
            num_partitions=size,
            neurons_per_partition=neurons_per_worker,
            in_degree=K,
            g=args.g,
            J_E=args.J_E,
            delay=args.delay,
            external_rate_hz=args.firing_rate,
            seed=SEED,
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
    model._verbose_timing = True
    model_creation_time = time.time() - t0
    if rank == 0:
        print(f"    Model loaded in {model_creation_time:.2f}s")

    # ---- [3/4] Setup GPU ----
    if rank == 0:
        print("\n[3/4] Setting up GPUs...")

    t0 = time.time()
    model.setup(use_gpu=True)
    gpu_setup_time = time.time() - t0
    if rank == 0:
        print(f"    GPU setup in {gpu_setup_time:.2f}s")

    # ---- [4/4] Poisson drive + simulate ----
    input_synapses = [sid for sid in model._synapse_ids
                      if model.get_synapse_connectivity(sid)[0] == -1]
    if rank == 0:
        print(f"\n    Injecting {args.firing_rate} Hz Poisson drive on "
              f"{len(input_synapses)} input synapses...")
    inject_poisson_drive(model, input_synapses, args.firing_rate,
                         simulation_ticks, args.dt_ms, SEED)

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
    total_synapses = len(model._synapse_ids)

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
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not csv_path.exists()
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    'job_id', 'nodes', 'workers', 'total_neurons', 'neurons_per_worker',
                    'synapses', 'in_degree', 'ticks', 'generation_time',
                    'model_creation_time', 'gpu_setup_time', 'construction_time',
                    'simulation_time', 'total_time'
                ])
            writer.writerow([
                os.environ.get('SLURM_JOB_ID', ''),
                os.environ.get('SLURM_NNODES', ''),
                size, total_neurons, neurons_per_worker, total_synapses, K,
                simulation_ticks, f'{generation_time:.4f}', f'{model_creation_time:.4f}',
                f'{gpu_setup_time:.4f}', f'{construction_time:.4f}',
                f'{sim_time:.4f}', f'{total_time:.4f}'
            ])
        print(f"\nTiming results appended to {csv_path}")


if __name__ == "__main__":
    main()
