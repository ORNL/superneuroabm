"""
Synthetic network generation for weak scaling tests.

This module provides functions to generate spiking neural networks that are
optimized for METIS partitioning, ensuring:
1. Balanced agent distribution across workers
2. Minimal cross-worker communication
3. Similar computational load per worker
"""

import networkx as nx
import numpy as np
from typing import Optional, Dict, Tuple, List


def generate_clustered_network(
    num_clusters: int,
    neurons_per_cluster: int,
    intra_cluster_prob: float = 0.3,
    inter_cluster_prob: float = 0.01,
    external_input_prob: float = 0.2,
    soma_breed: str = "lif_soma",
    soma_config: str = "config_0",
    synapse_breed: str = "single_exp_synapse",
    synapse_config: str = "no_learning_config_0",
    excitatory_ratio: float = 0.8,
    weight_exc: float = 14.0,
    weight_inh: float = -10.0,
    seed: Optional[int] = None
) -> nx.DiGraph:
    """
    Generate a clustered spiking neural network optimized for METIS partitioning.

    This creates multiple clusters with high intra-cluster connectivity and
    low inter-cluster connectivity, which METIS can efficiently partition.

    Args:
        num_clusters: Number of clusters (should match number of workers)
        neurons_per_cluster: Number of neurons in each cluster
        intra_cluster_prob: Connection probability within cluster (default: 0.3)
        inter_cluster_prob: Connection probability between clusters (default: 0.01)
        external_input_prob: Probability of external input per neuron (default: 0.2)
        soma_breed: Neuron type ("lif_soma" or "izh_soma")
        soma_config: Configuration name for somas
        synapse_breed: Synapse type
        synapse_config: Configuration name for synapses
        excitatory_ratio: Ratio of excitatory neurons (default: 0.8)
        weight_exc: Weight for excitatory synapses (default: 14.0)
        weight_inh: Weight for inhibitory synapses (default: -10.0)
        seed: Random seed for reproducibility

    Returns:
        NetworkX DiGraph with neuron and synapse attributes

    Example:
        >>> # Create network for 4 workers with 1000 neurons each
        >>> graph = generate_clustered_network(
        ...     num_clusters=4,
        ...     neurons_per_cluster=1000,
        ...     seed=42
        ... )
        >>> # Total neurons: 4000, optimized for 4 workers
    """
    if seed is not None:
        np.random.seed(seed)

    graph = nx.DiGraph()
    total_neurons = num_clusters * neurons_per_cluster

    print(f"[SyntheticNet] Generating clustered network:")
    print(f"  - Clusters: {num_clusters}")
    print(f"  - Neurons per cluster: {neurons_per_cluster}")
    print(f"  - Total neurons: {total_neurons}")
    print(f"  - Intra-cluster p: {intra_cluster_prob}")
    print(f"  - Inter-cluster p: {inter_cluster_prob}")

    # Create neurons organized by cluster
    neuron_ids = []
    for cluster_id in range(num_clusters):
        cluster_neurons = []
        for i in range(neurons_per_cluster):
            neuron_id = cluster_id * neurons_per_cluster + i

            # Determine if excitatory or inhibitory
            is_excitatory = i < int(neurons_per_cluster * excitatory_ratio)

            # Add neuron node
            graph.add_node(
                neuron_id,
                soma_breed=soma_breed,
                config=soma_config,
                cluster=cluster_id,  # Store cluster ID for analysis
                type="excitatory" if is_excitatory else "inhibitory",
                tags=[f"cluster_{cluster_id}"]
            )
            cluster_neurons.append(neuron_id)

        neuron_ids.append(cluster_neurons)

    # Add intra-cluster connections
    intra_edges = 0
    for cluster_id, cluster_neurons in enumerate(neuron_ids):
        for pre in cluster_neurons:
            for post in cluster_neurons:
                if pre != post and np.random.random() < intra_cluster_prob:
                    # Get neuron types
                    pre_type = graph.nodes[pre]["type"]
                    weight = weight_exc if pre_type == "excitatory" else weight_inh

                    graph.add_edge(
                        pre,
                        post,
                        synapse_breed=synapse_breed,
                        config=synapse_config,
                        overrides={"hyperparameters": {"weight": weight}},
                        connection_type="intra_cluster"
                    )
                    intra_edges += 1

    # Add inter-cluster connections
    inter_edges = 0
    for cluster_i in range(num_clusters):
        for cluster_j in range(num_clusters):
            if cluster_i != cluster_j:
                for pre in neuron_ids[cluster_i]:
                    for post in neuron_ids[cluster_j]:
                        if np.random.random() < inter_cluster_prob:
                            pre_type = graph.nodes[pre]["type"]
                            weight = weight_exc if pre_type == "excitatory" else weight_inh

                            graph.add_edge(
                                pre,
                                post,
                                synapse_breed=synapse_breed,
                                config=synapse_config,
                                overrides={"hyperparameters": {"weight": weight}},
                                connection_type="inter_cluster"
                            )
                            inter_edges += 1

    # Add external inputs
    external_inputs = 0
    for cluster_neurons in neuron_ids:
        for post in cluster_neurons:
            if np.random.random() < external_input_prob:
                graph.add_edge(
                    -1,  # External input
                    post,
                    synapse_breed=synapse_breed,
                    config=synapse_config,
                    overrides={"hyperparameters": {"weight": weight_exc}},
                    connection_type="external",
                    tags=["input_synapse"]
                )
                external_inputs += 1

    # Print statistics
    total_edges = intra_edges + inter_edges
    theoretical_edge_cut = inter_edges / total_edges if total_edges > 0 else 0

    print(f"[SyntheticNet] Network statistics:")
    print(f"  - Total edges: {total_edges}")
    print(f"  - Intra-cluster edges: {intra_edges} ({100*intra_edges/total_edges:.1f}%)")
    print(f"  - Inter-cluster edges: {inter_edges} ({100*inter_edges/total_edges:.1f}%)")
    print(f"  - External inputs: {external_inputs}")
    print(f"  - Theoretical edge cut (with perfect partition): {theoretical_edge_cut:.4f}")

    return graph


def generate_clustered_network_constant_comm(
    num_clusters: int,
    neurons_per_cluster: int,
    intra_cluster_prob: Optional[float] = None,
    intra_cluster_degree: Optional[int] = None,
    cross_cluster_edges: int = 2000,
    external_input_prob: float = 0.2,
    soma_breed: str = "lif_soma",
    soma_config: str = "config_0",
    synapse_breed: str = "single_exp_synapse",
    synapse_config: str = "no_learning_config_0",
    excitatory_ratio: float = 0.8,
    weight_exc: float = 14.0,
    weight_inh: float = -10.0,
    seed: Optional[int] = None
) -> nx.DiGraph:
    """
    Generate a clustered network for PROPER WEAK SCALING with constant per-worker work.

    This function creates networks suitable for weak scaling tests where:
    1. Per-worker workload remains constant as workers scale (linear scaling)
    2. Per-worker communication remains constant (truly constant communication)

    IMPORTANT: For proper weak scaling, use intra_cluster_degree (NOT intra_cluster_prob):
    - intra_cluster_degree: Each neuron connects to a FIXED number of neurons (O(n) edges per worker)
    - intra_cluster_prob: Each neuron connects with probability p (O(n²) edges per worker - NOT weak scaling!)

    Args:
        num_clusters: Number of clusters (should match number of workers)
        neurons_per_cluster: Number of neurons in each cluster
        intra_cluster_prob: Connection probability within cluster (creates O(n²) edges - NOT for weak scaling!)
        intra_cluster_degree: Average degree per neuron (creates O(n) edges - PROPER weak scaling!)
        cross_cluster_edges: Total edges each cluster sends to ALL other clusters (default: 2000)
        external_input_prob: Probability of external input per neuron (default: 0.2)
        soma_breed: Neuron type ("lif_soma" or "izh_soma")
        soma_config: Configuration name for somas
        synapse_breed: Synapse type
        synapse_config: Configuration name for synapses
        excitatory_ratio: Ratio of excitatory neurons (default: 0.8)
        weight_exc: Weight for excitatory synapses (default: 14.0)
        weight_inh: Weight for inhibitory synapses (default: -10.0)
        seed: Random seed for reproducibility

    Returns:
        NetworkX DiGraph with neuron and synapse attributes

    Example (PROPER weak scaling):
        >>> # Create network with constant degree (linear edge count)
        >>> graph = generate_clustered_network_constant_comm(
        ...     num_clusters=4,
        ...     neurons_per_cluster=10000,
        ...     intra_cluster_degree=10,  # Each neuron connects to 10 others
        ...     cross_cluster_edges=2000,
        ...     seed=42
        ... )
        >>> # Per worker: 10,000 neurons × 10 degree = 100,000 edges (linear!)
    """
    # Validate parameters
    if intra_cluster_prob is None and intra_cluster_degree is None:
        raise ValueError("Must specify either intra_cluster_prob or intra_cluster_degree")
    if intra_cluster_prob is not None and intra_cluster_degree is not None:
        raise ValueError("Cannot specify both intra_cluster_prob and intra_cluster_degree")

    if seed is not None:
        np.random.seed(seed)

    graph = nx.DiGraph()
    total_neurons = num_clusters * neurons_per_cluster

    print(f"[SyntheticNet] Generating clustered network for WEAK SCALING:")
    print(f"  - Clusters: {num_clusters}")
    print(f"  - Neurons per cluster: {neurons_per_cluster}")
    print(f"  - Total neurons: {total_neurons}")

    if intra_cluster_degree is not None:
        print(f"  - Intra-cluster degree: {intra_cluster_degree} edges/neuron (O(n) edges - PROPER weak scaling!)")
        expected_intra_edges = num_clusters * neurons_per_cluster * intra_cluster_degree
        print(f"  - Expected intra-cluster edges: {expected_intra_edges:,}")
    else:
        print(f"  - Intra-cluster prob: {intra_cluster_prob} (O(n²) edges - NOT proper weak scaling!)")
        expected_intra_edges = int(num_clusters * neurons_per_cluster * (neurons_per_cluster - 1) * intra_cluster_prob)
        print(f"  - Expected intra-cluster edges: {expected_intra_edges:,}")

    print(f"  - Cross-cluster edges per worker: {cross_cluster_edges} (constant)")

    # Create neurons organized by cluster
    neuron_ids = []
    for cluster_id in range(num_clusters):
        cluster_neurons = []
        for i in range(neurons_per_cluster):
            neuron_id = cluster_id * neurons_per_cluster + i

            # Determine if excitatory or inhibitory
            is_excitatory = i < int(neurons_per_cluster * excitatory_ratio)

            # Add neuron node
            graph.add_node(
                neuron_id,
                soma_breed=soma_breed,
                config=soma_config,
                cluster=cluster_id,
                type="excitatory" if is_excitatory else "inhibitory",
                tags=[f"cluster_{cluster_id}"]
            )
            cluster_neurons.append(neuron_id)

        neuron_ids.append(cluster_neurons)

    # Add intra-cluster connections
    intra_edges = 0
    for cluster_id, cluster_neurons in enumerate(neuron_ids):
        if intra_cluster_degree is not None:
            # CONSTANT DEGREE approach (proper weak scaling!)
            # Each neuron connects to exactly intra_cluster_degree random targets
            for pre in cluster_neurons:
                # Sample random targets (excluding self)
                possible_targets = [n for n in cluster_neurons if n != pre]
                num_targets = min(intra_cluster_degree, len(possible_targets))

                if num_targets > 0:
                    targets = np.random.choice(possible_targets, size=num_targets, replace=False)

                    for post in targets:
                        pre_type = graph.nodes[pre]["type"]
                        weight = weight_exc if pre_type == "excitatory" else weight_inh

                        graph.add_edge(
                            pre,
                            post,
                            synapse_breed=synapse_breed,
                            config=synapse_config,
                            overrides={"hyperparameters": {"weight": weight}},
                            connection_type="intra_cluster"
                        )
                        intra_edges += 1
        else:
            # PROBABILITY approach (creates O(n²) edges - NOT proper weak scaling!)
            for pre in cluster_neurons:
                for post in cluster_neurons:
                    if pre != post and np.random.random() < intra_cluster_prob:
                        pre_type = graph.nodes[pre]["type"]
                        weight = weight_exc if pre_type == "excitatory" else weight_inh

                        graph.add_edge(
                            pre,
                            post,
                            synapse_breed=synapse_breed,
                            config=synapse_config,
                            overrides={"hyperparameters": {"weight": weight}},
                            connection_type="intra_cluster"
                        )
                        intra_edges += 1

    # Add inter-cluster connections with TRULY CONSTANT per-worker communication
    # Each cluster sends exactly cross_cluster_edges TOTAL, distributed among all other clusters
    inter_edges = 0
    for cluster_i in range(num_clusters):
        if num_clusters == 1:
            continue  # No other clusters to connect to

        # Distribute cross_cluster_edges among (num_clusters - 1) target clusters
        edges_per_target = cross_cluster_edges // (num_clusters - 1)
        remainder = cross_cluster_edges % (num_clusters - 1)

        # Build list of target clusters
        target_clusters = [j for j in range(num_clusters) if j != cluster_i]

        for idx, cluster_j in enumerate(target_clusters):
            # Give remainder edges to first few targets to reach exact total
            edges_to_add = edges_per_target + (1 if idx < remainder else 0)

            if edges_to_add == 0:
                continue

            # Randomly sample exactly edges_to_add connections
            max_possible = neurons_per_cluster * neurons_per_cluster
            edges_to_add = min(edges_to_add, max_possible)

            # Generate all possible edges and sample
            all_possible_edges = [
                (pre, post)
                for pre in neuron_ids[cluster_i]
                for post in neuron_ids[cluster_j]
            ]

            # Randomly select edges_to_add edges without replacement
            selected_edges = np.random.choice(
                len(all_possible_edges),
                size=edges_to_add,
                replace=False
            )

            for edge_idx in selected_edges:
                pre, post = all_possible_edges[edge_idx]
                pre_type = graph.nodes[pre]["type"]
                weight = weight_exc if pre_type == "excitatory" else weight_inh

                graph.add_edge(
                    pre,
                    post,
                    synapse_breed=synapse_breed,
                    config=synapse_config,
                    overrides={"hyperparameters": {"weight": weight}},
                    connection_type="inter_cluster"
                )
                inter_edges += 1

    # Add external inputs
    external_inputs = 0
    for cluster_neurons in neuron_ids:
        for post in cluster_neurons:
            if np.random.random() < external_input_prob:
                graph.add_edge(
                    -1,  # External input
                    post,
                    synapse_breed=synapse_breed,
                    config=synapse_config,
                    overrides={"hyperparameters": {"weight": weight_exc}},
                    connection_type="external",
                    tags=["input_synapse"]
                )
                external_inputs += 1

    # Print statistics
    total_edges = intra_edges + inter_edges
    theoretical_edge_cut = inter_edges / total_edges if total_edges > 0 else 0

    # Calculate total agent count (neurons + synapses)
    total_neurons = num_clusters * neurons_per_cluster
    total_synapses = total_edges + external_inputs  # All edges become synapse agents
    total_agents = total_neurons + total_synapses
    agents_per_worker = total_agents / num_clusters if num_clusters > 0 else 0

    print(f"[SyntheticNet] Network statistics:")
    print(f"  - Total neurons: {total_neurons:,}")
    print(f"  - Total edges: {total_edges:,}")
    print(f"    - Intra-cluster: {intra_edges:,} ({100*intra_edges/total_edges:.1f}%)")
    print(f"    - Inter-cluster: {inter_edges:,} ({100*inter_edges/total_edges:.1f}%)")
    print(f"  - External inputs: {external_inputs:,}")
    print(f"  - Theoretical edge cut (with perfect partition): {theoretical_edge_cut:.4f}")
    print(f"\n[SyntheticNet] AGENT COUNT (for weak scaling verification):")
    print(f"  - Total agents: {total_agents:,} ({total_neurons:,} neurons + {total_synapses:,} synapses)")
    print(f"  - Agents per worker: {agents_per_worker:,.0f}")
    print(f"  - Scaling: {agents_per_worker:,.0f} agents/worker × {num_clusters} workers = {total_agents:,} total")

    return graph


def generate_grid_network(
    grid_size: Tuple[int, int],
    connection_radius: int = 1,
    connection_prob: float = 0.5,
    external_input_prob: float = 0.1,
    soma_breed: str = "lif_soma",
    soma_config: str = "config_0",
    synapse_breed: str = "single_exp_synapse",
    synapse_config: str = "no_learning_config_0",
    excitatory_ratio: float = 0.8,
    weight_exc: float = 14.0,
    weight_inh: float = -10.0,
    seed: Optional[int] = None
) -> nx.DiGraph:
    """
    Generate a grid-structured network with local connectivity.

    This creates a 2D grid of neurons where each neuron connects primarily
    to its neighbors within a given radius. This topology is naturally
    partitionable and METIS-friendly.

    Args:
        grid_size: (rows, cols) dimensions of the grid
        connection_radius: Euclidean distance for connectivity (default: 1)
        connection_prob: Probability of connection within radius (default: 0.5)
        external_input_prob: Probability of external input per neuron
        soma_breed: Neuron type
        soma_config: Configuration name for somas
        synapse_breed: Synapse type
        synapse_config: Configuration name for synapses
        excitatory_ratio: Ratio of excitatory neurons
        weight_exc: Weight for excitatory synapses
        weight_inh: Weight for inhibitory synapses
        seed: Random seed for reproducibility

    Returns:
        NetworkX DiGraph with grid structure

    Example:
        >>> # Create 100x100 grid (10,000 neurons)
        >>> graph = generate_grid_network(
        ...     grid_size=(100, 100),
        ...     connection_radius=2,
        ...     seed=42
        ... )
    """
    if seed is not None:
        np.random.seed(seed)

    rows, cols = grid_size
    total_neurons = rows * cols
    graph = nx.DiGraph()

    print(f"[SyntheticNet] Generating grid network:")
    print(f"  - Grid size: {rows}x{cols} ({total_neurons} neurons)")
    print(f"  - Connection radius: {connection_radius}")
    print(f"  - Connection probability: {connection_prob}")

    # Create neurons in grid layout
    neuron_positions = {}
    for i in range(rows):
        for j in range(cols):
            neuron_id = i * cols + j
            is_excitatory = neuron_id < int(total_neurons * excitatory_ratio)

            graph.add_node(
                neuron_id,
                soma_breed=soma_breed,
                config=soma_config,
                position=(i, j),
                type="excitatory" if is_excitatory else "inhibitory"
            )
            neuron_positions[neuron_id] = (i, j)

    # Add connections based on distance
    edge_count = 0
    for pre_id, (pre_i, pre_j) in neuron_positions.items():
        for post_id, (post_i, post_j) in neuron_positions.items():
            if pre_id != post_id:
                # Calculate Euclidean distance
                dist = np.sqrt((pre_i - post_i)**2 + (pre_j - post_j)**2)

                if dist <= connection_radius and np.random.random() < connection_prob:
                    pre_type = graph.nodes[pre_id]["type"]
                    weight = weight_exc if pre_type == "excitatory" else weight_inh

                    graph.add_edge(
                        pre_id,
                        post_id,
                        synapse_breed=synapse_breed,
                        config=synapse_config,
                        overrides={"hyperparameters": {"weight": weight}}
                    )
                    edge_count += 1

    # Add external inputs
    external_inputs = 0
    for neuron_id in neuron_positions.keys():
        if np.random.random() < external_input_prob:
            graph.add_edge(
                -1,  # External input
                neuron_id,
                synapse_breed=synapse_breed,
                config=synapse_config,
                overrides={"hyperparameters": {"weight": weight_exc}},
                tags=["input_synapse"]
            )
            external_inputs += 1

    print(f"[SyntheticNet] Network statistics:")
    print(f"  - Total edges: {edge_count}")
    print(f"  - External inputs: {external_inputs}")
    print(f"  - Avg degree: {2*edge_count/total_neurons:.2f}")

    return graph


def generate_ring_of_clusters(
    num_clusters: int,
    neurons_per_cluster: int,
    intra_cluster_prob: float = 0.3,
    adjacent_cluster_prob: float = 0.05,
    external_input_prob: float = 0.2,
    soma_breed: str = "lif_soma",
    soma_config: str = "config_0",
    synapse_breed: str = "single_exp_synapse",
    synapse_config: str = "no_learning_config_0",
    excitatory_ratio: float = 0.8,
    weight_exc: float = 14.0,
    weight_inh: float = -10.0,
    seed: Optional[int] = None
) -> nx.DiGraph:
    """
    Generate a ring of clusters network.

    This creates clusters arranged in a ring topology, where each cluster
    connects primarily to itself and its immediate neighbors in the ring.
    This provides a balanced structure with predictable cross-cluster
    communication patterns.

    Args:
        num_clusters: Number of clusters in the ring
        neurons_per_cluster: Number of neurons per cluster
        intra_cluster_prob: Connection probability within cluster
        adjacent_cluster_prob: Connection probability to adjacent clusters
        external_input_prob: Probability of external input per neuron
        soma_breed: Neuron type
        soma_config: Configuration name for somas
        synapse_breed: Synapse type
        synapse_config: Configuration name for synapses
        excitatory_ratio: Ratio of excitatory neurons
        weight_exc: Weight for excitatory synapses
        weight_inh: Weight for inhibitory synapses
        seed: Random seed for reproducibility

    Returns:
        NetworkX DiGraph with ring-of-clusters structure

    Example:
        >>> # Create ring of 8 clusters with 500 neurons each
        >>> graph = generate_ring_of_clusters(
        ...     num_clusters=8,
        ...     neurons_per_cluster=500,
        ...     seed=42
        ... )
    """
    if seed is not None:
        np.random.seed(seed)

    graph = nx.DiGraph()
    total_neurons = num_clusters * neurons_per_cluster

    print(f"[SyntheticNet] Generating ring-of-clusters network:")
    print(f"  - Clusters: {num_clusters}")
    print(f"  - Neurons per cluster: {neurons_per_cluster}")
    print(f"  - Total neurons: {total_neurons}")

    # Create neurons organized by cluster
    neuron_ids = []
    for cluster_id in range(num_clusters):
        cluster_neurons = []
        for i in range(neurons_per_cluster):
            neuron_id = cluster_id * neurons_per_cluster + i
            is_excitatory = i < int(neurons_per_cluster * excitatory_ratio)

            graph.add_node(
                neuron_id,
                soma_breed=soma_breed,
                config=soma_config,
                cluster=cluster_id,
                type="excitatory" if is_excitatory else "inhibitory",
                tags=[f"cluster_{cluster_id}"]
            )
            cluster_neurons.append(neuron_id)

        neuron_ids.append(cluster_neurons)

    # Add intra-cluster connections
    intra_edges = 0
    for cluster_neurons in neuron_ids:
        for pre in cluster_neurons:
            for post in cluster_neurons:
                if pre != post and np.random.random() < intra_cluster_prob:
                    pre_type = graph.nodes[pre]["type"]
                    weight = weight_exc if pre_type == "excitatory" else weight_inh

                    graph.add_edge(
                        pre, post,
                        synapse_breed=synapse_breed,
                        config=synapse_config,
                        overrides={"hyperparameters": {"weight": weight}}
                    )
                    intra_edges += 1

    # Add connections to adjacent clusters in ring
    inter_edges = 0
    for cluster_id in range(num_clusters):
        # Connect to next cluster in ring
        next_cluster = (cluster_id + 1) % num_clusters

        for pre in neuron_ids[cluster_id]:
            for post in neuron_ids[next_cluster]:
                if np.random.random() < adjacent_cluster_prob:
                    pre_type = graph.nodes[pre]["type"]
                    weight = weight_exc if pre_type == "excitatory" else weight_inh

                    graph.add_edge(
                        pre, post,
                        synapse_breed=synapse_breed,
                        config=synapse_config,
                        overrides={"hyperparameters": {"weight": weight}}
                    )
                    inter_edges += 1

    # Add external inputs
    external_inputs = 0
    for cluster_neurons in neuron_ids:
        for post in cluster_neurons:
            if np.random.random() < external_input_prob:
                graph.add_edge(
                    -1,  # External input
                    post,
                    synapse_breed=synapse_breed,
                    config=synapse_config,
                    overrides={"hyperparameters": {"weight": weight_exc}},
                    tags=["input_synapse"]
                )
                external_inputs += 1

    total_edges = intra_edges + inter_edges
    print(f"[SyntheticNet] Network statistics:")
    print(f"  - Total edges: {total_edges}")
    print(f"  - Intra-cluster edges: {intra_edges}")
    print(f"  - Inter-cluster edges: {inter_edges}")
    print(f"  - External inputs: {external_inputs}")

    return graph


def analyze_network_partition(
    graph: nx.DiGraph,
    partition_dict: Dict[int, int]
) -> Dict[str, any]:
    """
    Analyze network partition quality.

    Args:
        graph: NetworkX graph
        partition_dict: Mapping from node_id to worker_rank

    Returns:
        Dictionary with partition statistics
    """
    num_workers = max(partition_dict.values()) + 1

    # Count nodes per worker
    nodes_per_worker = [0] * num_workers
    for rank in partition_dict.values():
        nodes_per_worker[rank] += 1

    # Count edges within and between workers
    intra_worker_edges = 0
    inter_worker_edges = 0
    edges_per_worker = [0] * num_workers

    for u, v in graph.edges():
        if u in partition_dict and v in partition_dict:
            u_rank = partition_dict[u]
            v_rank = partition_dict[v]

            if u_rank == v_rank:
                intra_worker_edges += 1
                edges_per_worker[u_rank] += 1
            else:
                inter_worker_edges += 1

    total_edges = intra_worker_edges + inter_worker_edges
    edge_cut_ratio = inter_worker_edges / total_edges if total_edges > 0 else 0

    # Calculate balance metrics
    avg_nodes = np.mean(nodes_per_worker)
    std_nodes = np.std(nodes_per_worker)
    avg_edges = np.mean(edges_per_worker)
    std_edges = np.std(edges_per_worker)

    return {
        "num_workers": num_workers,
        "nodes_per_worker": nodes_per_worker,
        "avg_nodes": avg_nodes,
        "std_nodes": std_nodes,
        "node_imbalance": std_nodes / avg_nodes if avg_nodes > 0 else 0,
        "edges_per_worker": edges_per_worker,
        "avg_edges": avg_edges,
        "std_edges": std_edges,
        "edge_imbalance": std_edges / avg_edges if avg_edges > 0 else 0,
        "total_edges": total_edges,
        "intra_worker_edges": intra_worker_edges,
        "inter_worker_edges": inter_worker_edges,
        "edge_cut_ratio": edge_cut_ratio
    }
