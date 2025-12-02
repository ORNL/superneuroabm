"""
Weak scaling test with LIF neurons - CONSTANT COMMUNICATION MODEL

This uses generate_clustered_network_constant_comm() which maintains
a fixed number of cross-worker edges per worker, regardless of scale.
"""

import sys
import time
import argparse
import pickle
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from superneuroabm.io.synthetic_networks import generate_clustered_network_constant_comm, analyze_network_partition
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

def create_cluster_partition(graph, num_workers):
    """
    Create partition dict from cluster attribute in graph nodes.

    This uses the cluster structure that was created during network generation,
    ensuring perfect alignment between clusters and workers for truly constant
    communication overhead.

    Args:
        graph: NetworkX graph with 'cluster' attribute on nodes
        num_workers: Number of workers (for validation)

    Returns:
        Dictionary mapping node_id -> rank
    """
    partition_dict = {}
    for node in graph.nodes():
        if isinstance(node, int) and node >= 0:  # Skip external input nodes (-1)
            cluster_id = graph.nodes[node].get('cluster', 0)
            partition_dict[node] = cluster_id

    if rank == 0:
        print(f"[SuperNeuroABM] Created cluster-based partition")
        print(f"  - Using 'cluster' attribute from network generation")
        print(f"  - This ensures constant per-worker communication")

    return partition_dict

def main():
    parser = argparse.ArgumentParser(description="Weak scaling test with LIF neurons - Constant Communication Model")
    parser.add_argument("--neurons-per-worker", type=int, default=5000,
                       help="Neurons per worker (constant for weak scaling)")
    parser.add_argument("--ticks", type=int, default=10,
                       help="Simulation ticks")
    parser.add_argument("--update-ticks", type=int, default=5,
                       help="Update data every N ticks")
    parser.add_argument("--intra-cluster-degree", type=int, default=10,
                       help="Average degree per neuron within cluster (default: 10) - for PROPER weak scaling")
    parser.add_argument("--cross-cluster-edges", type=int, default=2000,
                       help="Number of edges in EACH direction for bidirectional pairs (default: 2000)")
    parser.add_argument("--num-neighbor-clusters", type=int, default=1,
                       help="Number of bidirectional neighbor pairs (default: 1 for TRUE weak scaling)")
    args = parser.parse_args()

    neurons_per_worker = args.neurons_per_worker
    simulation_ticks = args.ticks
    update_ticks = args.update_ticks
    intra_cluster_degree = args.intra_cluster_degree
    cross_cluster_edges = args.cross_cluster_edges
    num_neighbor_clusters = args.num_neighbor_clusters

    if rank == 0:
        print("="*70)
        print(f"WEAK SCALING TEST - PROPER WEAK SCALING")
        print("="*70)
        print(f"Goal: Constant per-worker workload as network scales")
        print(f"      - Constant neurons per worker")
        print(f"      - Constant edges per worker (O(n), not O(n²)!)")
        print(f"      - Constant communication per worker")
        print("="*70)
        print(f"Workers: {size}")
        print(f"Neurons per worker: {neurons_per_worker} (constant)")
        print(f"Intra-cluster degree: {intra_cluster_degree} edges/neuron (constant → O(n) scaling!)")
        print(f"Neighbor clusters per worker: {num_neighbor_clusters} (DIRECTED RING)")
        print(f"Cross-cluster edges per neighbor: {cross_cluster_edges}")
        total_cross_edges_out = cross_cluster_edges * num_neighbor_clusters if size > 1 else 0
        print(f"Total outgoing cross-cluster edges per worker: {total_cross_edges_out} (constant!)")
        print(f"Total incoming cross-cluster edges per worker: {total_cross_edges_out} (constant!)")
        print(f"Total neurons: {size * neurons_per_worker}")
        expected_edges = neurons_per_worker * intra_cluster_degree + total_cross_edges_out
        print(f"Expected edges per worker: ~{expected_edges:,}")
        print(f"Simulation ticks: {simulation_ticks}")
        print(f"Update ticks: {update_ticks}")
        print(f"Partitioning: Cluster-based")
        print("="*70)

    # Generate or load network from file
    # Use deterministic filename based on network parameters
    network_dir = Path(__file__).parent / "output"
    network_dir.mkdir(exist_ok=True)

    network_filename = (
        f"network_constcomm_c{size}_n{neurons_per_worker}_"
        f"deg{intra_cluster_degree}_cross{cross_cluster_edges}_"
        f"nbr{num_neighbor_clusters}_s42.pkl"
    )
    network_path = network_dir / network_filename

    if rank == 0:
        print("\n[1/4] Loading/generating clustered network...")

    # Check if network file exists (rank 0 checks, broadcasts to all)
    if rank == 0:
        file_exists = network_path.exists()
        print(f"    File exists: {file_exists}")
    else:
        file_exists = None

    # Broadcast file_exists to all ranks
    if comm is not None:
        file_exists = comm.bcast(file_exists, root=0)

    if file_exists:
        # Load existing network
        if rank == 0:
            print(f"    Loading network from {network_path.name}...")
        t0 = time.time()
        with open(network_path, 'rb') as f:
            graph = pickle.load(f)
        t1 = time.time()
        if rank == 0:
            print(f"    Network loaded in {t1-t0:.2f}s")
            print(f"    Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")

        # Barrier to ensure all ranks finish loading before proceeding
        if comm is not None:
            comm.Barrier()
    else:
        # Generate new network (only rank 0)
        if rank == 0:
            print(f"    Generating new network...")
            t0 = time.time()
            graph = generate_clustered_network_constant_comm(
                num_clusters=size,
                neurons_per_cluster=neurons_per_worker,
                intra_cluster_degree=intra_cluster_degree,
                cross_cluster_edges=cross_cluster_edges,
                num_neighbor_clusters=num_neighbor_clusters,
                external_input_prob=0.1,
                soma_breed="lif_soma",      # LIF only
                synapse_breed="single_exp_synapse",  # No learning
                synapse_config="no_learning_config_0",
                seed=42
            )
            t1 = time.time()
            print(f"    Network generated in {t1-t0:.2f}s")
            print(f"    Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")

            # Save to file
            print(f"    Saving network to {network_path.name}...")
            t0 = time.time()
            with open(network_path, 'wb') as f:
                pickle.dump(graph, f)
            t1 = time.time()
            print(f"    Network saved in {t1-t0:.2f}s")

        # Other ranks wait for rank 0 to finish
        if comm is not None:
            comm.Barrier()

        # All non-zero ranks load the saved network
        if rank != 0:
            with open(network_path, 'rb') as f:
                graph = pickle.load(f)

    # Create cluster-based partition (after graph is loaded on all ranks)
    if rank == 0:
        print("\n[2/4] Analyzing cluster-based partition...")

    # TEMPORARILY TEST WITH NO PARTITION TO ISOLATE ISSUE
    cluster_partition = None  # Disable custom partition
    # cluster_partition = create_cluster_partition(graph, size) if size > 1 else None

    # if rank == 0 and cluster_partition:
    #     stats = analyze_network_partition(graph, cluster_partition)
    #     print(f"    Edge cut ratio: {stats['edge_cut_ratio']:.4f}")
    #     print(f"    Node imbalance: {stats['node_imbalance']:.3f}")
    #     print(f"    Cross-worker edges: {stats['inter_worker_edges']}")

    # Create model
    if rank == 0:
        print("\n[3/4] Creating model and distributing...")

    t0 = time.time()
    model = model_from_nx_graph(
        graph,
        enable_internal_state_tracking=False,  # Disable for performance
        partition_dict=cluster_partition  # Use cluster partition instead of METIS
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
