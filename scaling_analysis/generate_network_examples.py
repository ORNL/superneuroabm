"""
Generate example synthetic networks and analyze their properties.

This script creates various synthetic networks and analyzes their suitability
for METIS partitioning and weak scaling tests.

Usage:
    python generate_network_examples.py
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from superneuroabm.io.synthetic_networks import (
    generate_clustered_network,
    generate_grid_network,
    generate_ring_of_clusters,
    analyze_network_partition
)
from superneuroabm.io.nx import generate_metis_partition


def analyze_network_topology(graph, name, num_workers=4):
    """
    Analyze network topology and partition quality.
    """
    print("\n" + "=" * 70)
    print(f"Network: {name}")
    print("=" * 70)

    print(f"\nBasic Statistics:")
    print(f"  - Nodes: {graph.number_of_nodes()}")
    print(f"  - Edges: {graph.number_of_edges()}")

    # Calculate degree statistics
    degrees = [d for n, d in graph.degree()]
    print(f"  - Avg degree: {np.mean(degrees):.2f} ± {np.std(degrees):.2f}")
    print(f"  - Min/Max degree: {np.min(degrees)} / {np.max(degrees)}")

    # Generate METIS partition
    print(f"\nGenerating METIS partition for {num_workers} workers...")
    partition_dict = generate_metis_partition(graph, num_workers)

    # Analyze partition
    stats = analyze_network_partition(graph, partition_dict)

    print(f"\nPartition Quality:")
    print(f"  - Nodes per worker: {stats['nodes_per_worker']}")
    print(f"  - Node balance: {stats['avg_nodes']:.1f} ± {stats['std_nodes']:.1f} (imbalance: {100*stats['node_imbalance']:.1f}%)")
    print(f"  - Edge balance: {stats['avg_edges']:.1f} ± {stats['std_edges']:.1f} (imbalance: {100*stats['edge_imbalance']:.1f}%)")
    print(f"  - Edge cut ratio: {stats['edge_cut_ratio']:.4f}")
    print(f"  - Cross-worker edges: {stats['inter_worker_edges']} / {stats['total_edges']} ({100*stats['edge_cut_ratio']:.1f}%)")

    return stats


def visualize_network_structure(graph, partition_dict, name, output_dir):
    """
    Visualize network structure and partition (for small networks).
    """
    if graph.number_of_nodes() > 500:
        print(f"  - Skipping visualization (too large: {graph.number_of_nodes()} nodes)")
        return

    try:
        import networkx as nx

        print(f"  - Creating visualization...")

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Filter out NaN nodes for visualization
        valid_nodes = [n for n in graph.nodes() if not (isinstance(n, float) and np.isnan(n))]
        subgraph = graph.subgraph(valid_nodes)

        # Layout
        if 'position' in list(graph.nodes(data=True))[0][1]:
            # Grid network - use position attribute
            pos = {n: graph.nodes[n]['position'] for n in valid_nodes}
        elif 'cluster' in list(graph.nodes(data=True))[0][1]:
            # Clustered network - use spring layout within clusters
            pos = nx.spring_layout(subgraph, k=0.5, iterations=50)
        else:
            pos = nx.spring_layout(subgraph, k=0.5, iterations=50)

        # Plot 1: By cluster/type
        ax1.set_title(f"{name} - Network Structure", fontsize=14)

        if 'cluster' in list(graph.nodes(data=True))[0][1]:
            # Color by cluster
            clusters = [graph.nodes[n].get('cluster', 0) for n in valid_nodes]
            node_colors = plt.cm.tab10(np.array(clusters) % 10)
        else:
            node_colors = 'lightblue'

        nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors,
                              node_size=30, alpha=0.8, ax=ax1)
        nx.draw_networkx_edges(subgraph, pos, alpha=0.2, width=0.5, ax=ax1)
        ax1.axis('off')

        # Plot 2: By partition
        ax2.set_title(f"{name} - METIS Partition", fontsize=14)

        partitions = [partition_dict.get(n, 0) for n in valid_nodes]
        partition_colors = plt.cm.Set3(np.array(partitions) % 12)

        nx.draw_networkx_nodes(subgraph, pos, node_color=partition_colors,
                              node_size=30, alpha=0.8, ax=ax2)
        nx.draw_networkx_edges(subgraph, pos, alpha=0.2, width=0.5, ax=ax2)
        ax2.axis('off')

        plt.tight_layout()

        output_file = output_dir / f"{name.lower().replace(' ', '_')}_visualization.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  - Visualization saved to: {output_file}")

    except Exception as e:
        print(f"  - Visualization failed: {e}")


def main():
    """
    Generate and analyze example networks.
    """
    print("=" * 70)
    print("SuperNeuroABM Synthetic Network Examples")
    print("=" * 70)

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    num_workers = 4
    seed = 42

    # Example 1: Clustered network (best for METIS)
    print("\n[Example 1] Clustered Network (Recommended for weak scaling)")
    graph1 = generate_clustered_network(
        num_clusters=num_workers,
        neurons_per_cluster=100,
        intra_cluster_prob=0.3,
        inter_cluster_prob=0.01,
        seed=seed
    )
    stats1 = analyze_network_topology(graph1, "Clustered Network", num_workers)
    partition1 = generate_metis_partition(graph1, num_workers)
    visualize_network_structure(graph1, partition1, "Clustered Network", output_dir)

    # Example 2: Grid network
    print("\n[Example 2] Grid Network")
    graph2 = generate_grid_network(
        grid_size=(20, 20),
        connection_radius=2,
        connection_prob=0.5,
        seed=seed
    )
    stats2 = analyze_network_topology(graph2, "Grid Network", num_workers)
    partition2 = generate_metis_partition(graph2, num_workers)
    visualize_network_structure(graph2, partition2, "Grid Network", output_dir)

    # Example 3: Ring of clusters
    print("\n[Example 3] Ring of Clusters")
    graph3 = generate_ring_of_clusters(
        num_clusters=num_workers,
        neurons_per_cluster=100,
        intra_cluster_prob=0.3,
        adjacent_cluster_prob=0.05,
        seed=seed
    )
    stats3 = analyze_network_topology(graph3, "Ring of Clusters", num_workers)
    partition3 = generate_metis_partition(graph3, num_workers)
    visualize_network_structure(graph3, partition3, "Ring of Clusters", output_dir)

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)

    summary_data = [
        ("Clustered Network", graph1.number_of_nodes(), graph1.number_of_edges(), stats1),
        ("Grid Network", graph2.number_of_nodes(), graph2.number_of_edges(), stats2),
        ("Ring of Clusters", graph3.number_of_nodes(), graph3.number_of_edges(), stats3),
    ]

    print(f"\n{'Network':<25} {'Nodes':<8} {'Edges':<8} {'Edge Cut':<12} {'Node Imb':<12} {'Edge Imb':<12}")
    print("-" * 85)
    for name, nodes, edges, stats in summary_data:
        print(f"{name:<25} {nodes:<8} {edges:<8} {stats['edge_cut_ratio']:<12.4f} {stats['node_imbalance']:<12.3f} {stats['edge_imbalance']:<12.3f}")

    print("\n" + "=" * 70)
    print("RECOMMENDATIONS FOR WEAK SCALING")
    print("=" * 70)
    print("""
1. CLUSTERED NETWORK (Best Choice):
   - Lowest edge cut ratio → minimal cross-worker communication
   - Perfect load balance by design
   - Scales easily by adding more clusters
   - Best for demonstrating weak scaling properties

2. GRID NETWORK (Good for spatial problems):
   - Natural partitioning for spatial domains
   - Good load balance
   - Suitable for problems with spatial structure

3. RING OF CLUSTERS (Good for circular dependencies):
   - Controlled cross-cluster communication
   - Good for testing worst-case scenarios
   - Useful for pipeline-like architectures

For weak scaling tests, use CLUSTERED NETWORK with:
- num_clusters = number of workers
- neurons_per_cluster = constant (e.g., 1000-5000)
- intra_cluster_prob = 0.2-0.4 (adjust for desired density)
- inter_cluster_prob = 0.001-0.01 (keep low!)
""")

    print(f"\nResults saved to: {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
