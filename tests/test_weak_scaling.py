"""
Weak scaling test for SuperNeuroABM.

This script demonstrates weak scaling by running networks of increasing size
with a proportionally increasing number of workers. The runtime should remain
relatively constant if the system scales well.

Usage:
    # Single GPU (baseline)
    python test_weak_scaling.py --num-workers 1 --neurons-per-worker 1000

    # Multiple GPUs/nodes with METIS partitioning
    mpirun -n 4 python test_weak_scaling.py --num-workers 4 --neurons-per-worker 1000

    # Large scale test
    mpirun -n 16 python test_weak_scaling.py --num-workers 16 --neurons-per-worker 2000
"""

import sys
import time
import argparse
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from superneuroabm.io.nx import model_from_nx_graph, generate_metis_partition
from superneuroabm.io.synthetic_networks import (
    generate_clustered_network,
    analyze_network_partition
)

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except ImportError:
    rank = 0
    size = 1
    comm = None


def run_weak_scaling_test(
    num_workers: int,
    neurons_per_worker: int,
    simulation_ticks: int = 1000,
    intra_cluster_prob: float = 0.3,
    inter_cluster_prob: float = 0.01,
    use_metis: bool = True,
    seed: int = 42
):
    """
    Run a weak scaling test.

    Args:
        num_workers: Number of workers (should match MPI size for multi-node)
        neurons_per_worker: Number of neurons per worker (constant for weak scaling)
        simulation_ticks: Number of simulation time steps
        intra_cluster_prob: Intra-cluster connection probability
        inter_cluster_prob: Inter-cluster connection probability
        use_metis: Whether to use METIS partitioning
        seed: Random seed
    """
    if rank == 0:
        print("=" * 70)
        print("WEAK SCALING TEST")
        print("=" * 70)
        print(f"Configuration:")
        print(f"  - MPI workers: {size}")
        print(f"  - Target workers for network: {num_workers}")
        print(f"  - Neurons per worker: {neurons_per_worker}")
        print(f"  - Total neurons: {num_workers * neurons_per_worker}")
        print(f"  - Simulation ticks: {simulation_ticks}")
        print(f"  - METIS partitioning: {use_metis}")
        print("=" * 70)

        if size != num_workers:
            print(f"WARNING: MPI size ({size}) != num_workers ({num_workers})")
            print(f"         This is OK for testing, but results may not be optimal.")
            print()

    # Generate network (only on rank 0, will be distributed)
    if rank == 0:
        print("\n[Step 1/5] Generating synthetic network...")
        start_time = time.time()

        graph = generate_clustered_network(
            num_clusters=num_workers,
            neurons_per_cluster=neurons_per_worker,
            intra_cluster_prob=intra_cluster_prob,
            inter_cluster_prob=inter_cluster_prob,
            external_input_prob=0.1,
            soma_breed="lif_soma",
            soma_config="config_0",
            synapse_breed="single_exp_synapse",
            synapse_config="no_learning_config_0",
            seed=seed
        )

        generation_time = time.time() - start_time
        print(f"[Step 1/5] Network generation completed in {generation_time:.2f}s")

        # Analyze theoretical partition quality
        if use_metis:
            print("\n[Step 2/5] Generating METIS partition...")
            start_time = time.time()
            partition_dict = generate_metis_partition(graph, num_workers)
            partition_time = time.time() - start_time
            print(f"[Step 2/5] METIS partition completed in {partition_time:.2f}s")

            # Analyze partition
            print("\n[Step 3/5] Analyzing partition quality...")
            stats = analyze_network_partition(graph, partition_dict)

            print(f"\nPartition Statistics:")
            print(f"  - Workers: {stats['num_workers']}")
            print(f"  - Nodes per worker: {stats['nodes_per_worker']}")
            print(f"  - Avg nodes: {stats['avg_nodes']:.1f} ± {stats['std_nodes']:.1f}")
            print(f"  - Node imbalance: {stats['node_imbalance']:.3f}")
            print(f"  - Avg edges per worker: {stats['avg_edges']:.1f} ± {stats['std_edges']:.1f}")
            print(f"  - Edge imbalance: {stats['edge_imbalance']:.3f}")
            print(f"  - Edge cut ratio: {stats['edge_cut_ratio']:.4f}")
            print(f"  - Inter-worker edges: {stats['inter_worker_edges']} / {stats['total_edges']}")
        else:
            print("\n[Step 2/5] Skipping METIS partition (using round-robin)")
            partition_dict = None
    else:
        graph = None
        partition_dict = None

    # Create model from graph
    if rank == 0:
        print("\n[Step 4/5] Creating model and distributing to workers...")

    start_time = time.time()

    model = model_from_nx_graph(
        graph,
        enable_internal_state_tracking=False,  # Disable for performance
        partition_method='metis' if use_metis else None,
        partition_dict=partition_dict if not use_metis else None
    )

    model_creation_time = time.time() - start_time

    if rank == 0:
        print(f"[Step 4/5] Model creation completed in {model_creation_time:.2f}s")

    # Setup GPU
    if rank == 0:
        print("\n[Step 5/5] Setting up simulation...")

    start_time = time.time()
    model.setup(use_gpu=True)
    setup_time = time.time() - start_time

    if rank == 0:
        print(f"[Step 5/5] Setup completed in {setup_time:.2f}s")

    # Add external input spikes
    if rank == 0:
        print("\nAdding input spikes...")

    input_synapses = model.get_agents_with_tag("input_synapse")
    num_inputs = len(input_synapses)

    if rank == 0:
        print(f"  - Found {num_inputs} input synapses")

    # Add random spikes to input synapses
    np.random.seed(seed + rank)
    spike_count = 0
    for synapse_id in input_synapses:
        # Add spikes at random times
        spike_times = np.random.randint(10, simulation_ticks - 10, size=5)
        for tick in spike_times:
            model.add_spike(synapse_id=synapse_id, tick=int(tick), value=1.0)
            spike_count += 1

    if rank == 0:
        print(f"  - Added {spike_count} input spikes")

    # Synchronize before simulation
    if comm is not None:
        comm.Barrier()

    # Run simulation
    if rank == 0:
        print("\n" + "=" * 70)
        print("RUNNING SIMULATION")
        print("=" * 70)

    start_time = time.time()
    model.simulate(ticks=simulation_ticks, update_data_ticks=simulation_ticks)
    simulation_time = time.time() - start_time

    if comm is not None:
        comm.Barrier()

    # Gather timing results
    if comm is not None:
        all_sim_times = comm.gather(simulation_time, root=0)
    else:
        all_sim_times = [simulation_time]

    # Print results
    if rank == 0:
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)

        print(f"\nTiming Summary:")
        print(f"  - Network generation: {generation_time:.2f}s")
        if use_metis:
            print(f"  - METIS partitioning: {partition_time:.2f}s")
        print(f"  - Model creation: {model_creation_time:.2f}s")
        print(f"  - GPU setup: {setup_time:.2f}s")
        print(f"  - Simulation: {np.mean(all_sim_times):.2f}s (avg across workers)")
        print(f"    - Min: {np.min(all_sim_times):.2f}s")
        print(f"    - Max: {np.max(all_sim_times):.2f}s")
        print(f"    - Std: {np.std(all_sim_times):.2f}s")
        print(f"  - Total time: {generation_time + model_creation_time + setup_time + np.mean(all_sim_times):.2f}s")

        # Calculate performance metrics
        total_neurons = num_workers * neurons_per_worker
        neurons_per_sec = total_neurons * simulation_ticks / np.mean(all_sim_times)
        synapses_per_sec = graph.number_of_edges() * simulation_ticks / np.mean(all_sim_times)

        print(f"\nPerformance Metrics:")
        print(f"  - Throughput: {neurons_per_sec:.2e} neuron-steps/sec")
        print(f"  - Throughput: {synapses_per_sec:.2e} synapse-steps/sec")
        print(f"  - Time per tick: {1000 * np.mean(all_sim_times) / simulation_ticks:.3f}ms")

        # Calculate load imbalance
        max_time = np.max(all_sim_times)
        min_time = np.min(all_sim_times)
        imbalance = (max_time - min_time) / np.mean(all_sim_times)

        print(f"\nLoad Balance:")
        print(f"  - Time imbalance: {100 * imbalance:.1f}%")
        print(f"  - Slowest worker overhead: {max_time - np.mean(all_sim_times):.2f}s")

        print("\n" + "=" * 70)
        print(f"Test completed successfully!")
        print("=" * 70)

        # Write results to file
        results_file = Path(__file__).parent / "output" / f"weak_scaling_n{num_workers}_p{neurons_per_worker}.txt"
        results_file.parent.mkdir(exist_ok=True)

        with open(results_file, "w") as f:
            f.write(f"Weak Scaling Test Results\n")
            f.write(f"========================\n\n")
            f.write(f"Configuration:\n")
            f.write(f"  Workers: {num_workers}\n")
            f.write(f"  Neurons per worker: {neurons_per_worker}\n")
            f.write(f"  Total neurons: {total_neurons}\n")
            f.write(f"  Total edges: {graph.number_of_edges()}\n")
            f.write(f"  Simulation ticks: {simulation_ticks}\n")
            f.write(f"  METIS: {use_metis}\n\n")
            f.write(f"Timing:\n")
            f.write(f"  Simulation time (avg): {np.mean(all_sim_times):.2f}s\n")
            f.write(f"  Simulation time (max): {np.max(all_sim_times):.2f}s\n")
            f.write(f"  Simulation time (min): {np.min(all_sim_times):.2f}s\n")
            f.write(f"  Load imbalance: {100 * imbalance:.1f}%\n\n")
            f.write(f"Performance:\n")
            f.write(f"  Throughput: {neurons_per_sec:.2e} neuron-steps/sec\n")
            f.write(f"  Time per tick: {1000 * np.mean(all_sim_times) / simulation_ticks:.3f}ms\n")
            if use_metis:
                f.write(f"\nPartition Quality:\n")
                f.write(f"  Edge cut ratio: {stats['edge_cut_ratio']:.4f}\n")
                f.write(f"  Node imbalance: {stats['node_imbalance']:.3f}\n")

        print(f"\nResults saved to: {results_file}")


def main():
    parser = argparse.ArgumentParser(description="Weak scaling test for SuperNeuroABM")

    parser.add_argument(
        "--num-workers",
        type=int,
        default=size,
        help="Number of workers (clusters) in the network (default: MPI size)"
    )

    parser.add_argument(
        "--neurons-per-worker",
        type=int,
        default=1000,
        help="Number of neurons per worker (constant for weak scaling) (default: 1000)"
    )

    parser.add_argument(
        "--ticks",
        type=int,
        default=1000,
        help="Number of simulation ticks (default: 1000)"
    )

    parser.add_argument(
        "--intra-prob",
        type=float,
        default=0.3,
        help="Intra-cluster connection probability (default: 0.3)"
    )

    parser.add_argument(
        "--inter-prob",
        type=float,
        default=0.01,
        help="Inter-cluster connection probability (default: 0.01)"
    )

    parser.add_argument(
        "--no-metis",
        action="store_true",
        help="Disable METIS partitioning (use round-robin instead)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    run_weak_scaling_test(
        num_workers=args.num_workers,
        neurons_per_worker=args.neurons_per_worker,
        simulation_ticks=args.ticks,
        intra_cluster_prob=args.intra_prob,
        inter_cluster_prob=args.inter_prob,
        use_metis=not args.no_metis,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
