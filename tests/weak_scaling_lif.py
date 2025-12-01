"""
Weak scaling test with LIF neurons only - focused on runtime comparison
"""

import sys
import time
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from superneuroabm.io.synthetic_networks import generate_clustered_network, analyze_network_partition
from superneuroabm.io.nx import model_from_nx_graph, generate_metis_partition

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
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
    parser.add_argument("--update-ticks", type=int, default=5,
                       help="Update data every N ticks")
    parser.add_argument("--intra-cluster-prob", type=float, default=0.01,
                       help="Intra-cluster connection probability (default: 0.01)")
    parser.add_argument("--inter-cluster-prob", type=float, default=0.001,
                       help="Inter-cluster connection probability (default: 0.001)")
    parser.add_argument("--no-metis", action="store_true",
                       help="Disable METIS partitioning")
    args = parser.parse_args()

    neurons_per_worker = args.neurons_per_worker
    simulation_ticks = args.ticks
    update_ticks = args.update_ticks
    intra_cluster_prob = args.intra_cluster_prob
    inter_cluster_prob = args.inter_cluster_prob
    use_metis = not args.no_metis and size > 1

    if rank == 0:
        print("="*70)
        print(f"WEAK SCALING TEST - Memory Scaling Demonstration")
        print("="*70)
        print(f"Goal: Show that more workers enable LARGER networks")
        print(f"      (not faster execution, but larger problem size)")
        print("="*70)
        print(f"Workers: {size}")
        print(f"Neurons per worker: {neurons_per_worker} (constant)")
        print(f"Total neurons: {size * neurons_per_worker}")
        print(f"Simulation ticks: {simulation_ticks}")
        print(f"Update ticks: {update_ticks}")
        print(f"Number of clusters: {size}")
        print(f"Intra-cluster prob: {intra_cluster_prob}")
        print(f"Inter-cluster prob: {inter_cluster_prob}")
        print(f"METIS partitioning: {use_metis}")
        print("="*70)

        # Generate network
        print("\n[1/4] Generating clustered network...")
        t0 = time.time()

        graph = generate_clustered_network(
            num_clusters=size,
            neurons_per_cluster=neurons_per_worker,
            intra_cluster_prob=intra_cluster_prob,
            inter_cluster_prob=inter_cluster_prob,
            external_input_prob=0.1,
            soma_breed="lif_soma",      # LIF only
            synapse_breed="single_exp_synapse",  # No learning
            synapse_config="no_learning_config_0",
            seed=42
        )
        t1 = time.time()
        print(f"    Network generated in {t1-t0:.2f}s")
        print(f"    Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")

        if use_metis:
            print("\n[2/4] Analyzing METIS partition...")
            partition_dict = generate_metis_partition(graph, size)
            stats = analyze_network_partition(graph, partition_dict)
            print(f"    Edge cut ratio: {stats['edge_cut_ratio']:.4f}")
            print(f"    Node imbalance: {stats['node_imbalance']:.3f}")
    else:
        graph = None

    # Create model
    if rank == 0:
        print("\n[3/4] Creating model and distributing...")

    t0 = time.time()
    model = model_from_nx_graph(
        graph,
        enable_internal_state_tracking=False,  # Disable for performance
        partition_method='metis' if use_metis else None
    )
    t1 = time.time()

    if rank == 0:
        print(f"    Model created in {t1-t0:.2f}s")

    # Setup GPU
    if rank == 0:
        print("\n[4/4] Setting up GPUs...")

    t0 = time.time()
    model.setup(use_gpu=True)
    t1 = time.time()

    if rank == 0:
        print(f"    GPU setup in {t1-t0:.2f}s")

    # Add input spikes
    input_synapses = list(model.get_agents_with_tag("input_synapse"))
    if rank == 0:
        print(f"\n    Adding spikes to {len(input_synapses)} input synapses...")

    # Add input spikes at the beginning to start simulation
    for synapse_id in input_synapses[:min(len(input_synapses), 50)]:
        model.add_spike(synapse_id=synapse_id, tick=1, value=1.0)

    # Run simulation
    if rank == 0:
        print("\n" + "="*70)
        print(f"RUNNING SIMULATION ({simulation_ticks} ticks)")
        print("="*70)

    start_time = time.time()
    model.simulate(ticks=simulation_ticks, update_data_ticks=update_ticks)
    sim_time = time.time() - start_time

    # Print results
    if rank == 0:
        total_neurons = size * neurons_per_worker
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)
        print(f"Network Size:")
        print(f"  Total neurons: {total_neurons:,}")
        print(f"  Total edges: {graph.number_of_edges():,}")
        print(f"  Memory distributed across {size} worker(s)")
        print(f"\nSimulation time: {sim_time:.3f}s")
        print(f"\nNote: With {size} worker(s), we can handle {total_neurons:,} neurons")
        print(f"      that would be {size}x larger than single GPU limit")
        print("="*70)
        print("SUCCESS - Network simulation completed!")
        print("="*70)

if __name__ == "__main__":
    main()
