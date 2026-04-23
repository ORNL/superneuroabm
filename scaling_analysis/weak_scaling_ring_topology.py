"""
Weak scaling test with LIF neurons - PARTITION FILE FLOW

Phase 1: Each rank generates its partition file (or rank 0 generates all).
Phase 2: Each rank loads its partition file via model.load_partition().
Phase 3: Setup + simulate.
"""

import sys
import time
import argparse
import csv
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from superneuroabm.io.synthetic_networks import generate_and_save_local_partition
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


def main():
    parser = argparse.ArgumentParser(description="Weak scaling test with LIF neurons")
    parser.add_argument("--neurons-per-worker", type=int, default=5000,
                       help="Neurons per worker (constant for weak scaling)")
    parser.add_argument("--ticks", type=int, default=10,
                       help="Simulation ticks")
    parser.add_argument("--update-ticks", type=int, default=1,
                       help="Update data every N ticks")
    parser.add_argument("--intra-cluster-degree", type=int, default=10,
                       help="Average degree per neuron within cluster")
    parser.add_argument("--cross-cluster-edges", type=int, default=2000,
                       help="Cross-cluster edges per neighbor pair")
    parser.add_argument("--num-neighbor-clusters", type=int, default=1,
                       help="Number of neighbor clusters (1 for ring topology)")
    parser.add_argument("--partition-dir", type=str, default=None,
                       help="Directory with pre-generated partition files. If not set, generates in /tmp.")
    parser.add_argument("--csv", type=str, default=None,
                       help="Path to CSV file for appending timing results (rank 0 only)")
    args = parser.parse_args()

    neurons_per_worker = args.neurons_per_worker
    simulation_ticks = args.ticks
    update_ticks = args.update_ticks
    intra_cluster_degree = args.intra_cluster_degree
    cross_cluster_edges = args.cross_cluster_edges
    num_neighbor_clusters = args.num_neighbor_clusters

    if rank == 0:
        print("="*70)
        print(f"WEAK SCALING TEST - PARTITION FILE FLOW")
        print("="*70)
        if comm is not None:
            mpi_size = comm.Get_size()
            slurm_tasks = os.environ.get('SLURM_STEP_NUM_TASKS', 'NOT SET')
            print(f"[DEBUG] MPI.COMM_WORLD.Get_size() = {mpi_size}")
            print(f"[DEBUG] SLURM_STEP_NUM_TASKS = {slurm_tasks}")
            print(f"[DEBUG] Using size = {size}")
        print(f"Workers: {size}")
        print(f"Neurons per worker: {neurons_per_worker}")
        print(f"Intra-cluster degree: {intra_cluster_degree}")
        print(f"Neighbor clusters: {num_neighbor_clusters} (DIRECTED RING)")
        print(f"Cross-cluster edges per neighbor: {cross_cluster_edges}")
        total_cross = cross_cluster_edges * num_neighbor_clusters if size > 1 else 0
        print(f"Total neurons: {size * neurons_per_worker}")
        print(f"Expected edges per worker: ~{neurons_per_worker * intra_cluster_degree + total_cross:,}")
        print(f"Simulation ticks: {simulation_ticks}")
        print("="*70)

    SEED = 42
    EXTERNAL_INPUT_PROB = 0.1

    t_pipeline_start = time.time()

    # ---- [1/4] Generate partition files ----
    if rank == 0:
        print("\n[1/4] Generating partition files...")

    t0 = time.time()

    if args.partition_dir:
        partition_dir = args.partition_dir
    else:
        # Use shared filesystem (not /tmp which is node-local on Frontier)
        script_dir = Path(__file__).parent
        partition_dir = str(script_dir / "partitions" / f"{size}w_{neurons_per_worker}n")

    # Each rank generates and saves only its own partition file (skip if exists)
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
            intra_cluster_degree=intra_cluster_degree,
            cross_cluster_edges=cross_cluster_edges,
            num_neighbor_clusters=num_neighbor_clusters,
            topology_type="ring",
            external_input_prob=EXTERNAL_INPUT_PROB,
            seed=SEED,
        )

    t1 = time.time()
    generation_time = t1 - t0

    if rank == 0:
        print(f"    Generation completed in {generation_time:.2f}s")

    # ---- [2/4] Load partition file ----
    if rank == 0:
        print("\n[2/4] Loading partition files...")

    t0 = time.time()
    model = NeuromorphicModel(enable_internal_state_tracking=False)
    partition_file = os.path.join(partition_dir, f"partition_{rank}.pkl")
    model.load_partition(partition_file)
    model._verbose_timing = True
    t1 = time.time()
    model_creation_time = t1 - t0

    if rank == 0:
        print(f"    Model loaded in {model_creation_time:.2f}s")

    # ---- [3/4] Setup GPU ----
    if rank == 0:
        print("\n[3/4] Setting up GPUs...")

    t0 = time.time()
    model.setup(use_gpu=True)
    t1 = time.time()
    gpu_setup_time = t1 - t0

    if rank == 0:
        print(f"    GPU setup in {gpu_setup_time:.2f}s")

    # Add input spikes
    input_synapses = list(model.get_agents_with_tag("input_synapse"))
    if rank == 0:
        print(f"\n    Adding spikes to {len(input_synapses)} input synapses...")

    for synapse_id in input_synapses[:min(len(input_synapses), 50)]:
        model.add_spike(synapse_id=synapse_id, tick=1, value=1.0)

    # ---- [4/4] Simulate ----
    if rank == 0:
        print("\n" + "="*70)
        print(f"RUNNING SIMULATION ({simulation_ticks} ticks)")
        print("="*70)

    start_time = time.time()
    model.simulate(ticks=simulation_ticks, update_data_ticks=update_ticks)
    wall_time = time.time() - start_time
    sim_time = getattr(model, '_simulation_time', wall_time)
    construction_time = getattr(model, '_construction_time', 0.0)

    total_time = time.time() - t_pipeline_start
    total_neurons = size * neurons_per_worker
    total_edges = len(model.get_agents_with_tag("synapse"))

    if rank == 0:
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)
        print(f"Network Size:")
        print(f"  Total neurons: {total_neurons:,}")
        print(f"  Memory distributed across {size} worker(s)")
        print(f"\nGeneration time: {generation_time:.3f}s")
        print(f"Model load time: {model_creation_time:.3f}s")
        print(f"Construction time: {construction_time:.3f}s")
        print(f"Simulation time: {sim_time:.3f}s")
        print(f"Total wall time: {total_time:.3f}s")
        print("="*70)
        print("SUCCESS - Network simulation completed!")
        print("="*70)

    # Write CSV timing results
    if args.csv and rank == 0:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not csv_path.exists()
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    'job_id', 'nodes', 'workers', 'neurons', 'edges', 'ticks',
                    'generation_time', 'model_creation_time', 'gpu_setup_time',
                    'construction_time', 'simulation_time', 'total_time'
                ])
            writer.writerow([
                os.environ.get('SLURM_JOB_ID', ''),
                os.environ.get('SLURM_NNODES', ''),
                size, total_neurons, total_edges, simulation_ticks,
                f'{generation_time:.4f}', f'{model_creation_time:.4f}',
                f'{gpu_setup_time:.4f}', f'{construction_time:.4f}',
                f'{sim_time:.4f}', f'{total_time:.4f}'
            ])
        print(f"\nTiming results appended to {csv_path}")

if __name__ == "__main__":
    main()
