"""
Synthetic network generation for weak scaling tests.

This module provides functions to generate spiking neural networks that are
optimized for METIS partitioning, ensuring:
1. Balanced agent distribution across workers
2. Minimal cross-worker communication
3. Similar computational load per worker

NOTE: the ``metadata=[...]`` graph attributes set below (e.g. ``"input_synapse"``,
``"cluster_<n>"``) are application-side annotations and are NO LONGER consumed by
superneuroabm — the framework no longer tracks labels (it is a pure id->property
store). TODO: input synapses (pre == -1) should be specified EXPLICITLY by this
generator and surfaced to the caller (e.g. returned as an id list) instead of
relying on a framework label lookup.
"""

from collections import defaultdict

import networkx as nx
import numpy as np
import random
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
    synapse_config: str = "config_0",
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
                metadata=[f"cluster_{cluster_id}"]
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
    for cluster_id, cluster_neurons in enumerate(neuron_ids):
        for post in cluster_neurons:
            if np.random.random() < external_input_prob:
                graph.add_edge(
                    -1,  # External input
                    post,
                    synapse_breed=synapse_breed,
                    config=synapse_config,
                    overrides={"hyperparameters": {"weight": weight_exc}},
                    connection_type="external",
                    metadata=["input_synapse"],
                    cluster=cluster_id  # Assign to post-synaptic neuron's cluster
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
    num_neighbor_clusters: Optional[int] = None,
    external_input_prob: float = 0.2,
    soma_breed: str = "lif_soma",
    soma_config: str = "config_0",
    synapse_breed: str = "single_exp_synapse",
    synapse_config: str = "config_0",
    excitatory_ratio: float = 0.8,
    weight_exc: float = 14.0,
    weight_inh: float = -10.0,
    seed: Optional[int] = None,
    topology_type: str = "ring"
) -> nx.DiGraph:
    """
    Generate a clustered network for PROPER WEAK SCALING with constant per-worker work.

    This function creates networks suitable for weak scaling tests where:
    1. Per-worker workload remains constant as workers scale (linear scaling)
    2. Per-worker communication remains constant (truly constant communication)
    3. Per-worker contextualization overhead remains constant

    IMPORTANT: For proper weak scaling, use intra_cluster_degree (NOT intra_cluster_prob):
    - intra_cluster_degree: Each neuron connects to a FIXED number of neurons (O(n) edges per worker)
    - intra_cluster_prob: Each neuron connects with probability p (O(n²) edges per worker - NOT weak scaling!)

    CRITICAL: Use num_neighbor_clusters to control contextualization overhead:
    - If None (default), each cluster connects to ALL other clusters (NOT TRUE weak scaling!)
    - If specified, each cluster forms BIDIRECTIONAL pairs with K neighbors
    - Recommended: Set to 1 for minimal constant cross-cluster communication

    Args:
        num_clusters: Number of clusters (should match number of workers)
        neurons_per_cluster: Number of neurons in each cluster
        intra_cluster_prob: Connection probability within cluster (creates O(n²) edges - NOT for weak scaling!)
        intra_cluster_degree: Average degree per neuron (creates O(n) edges - PROPER weak scaling!)
        cross_cluster_edges: Edges in EACH direction for bidirectional pairs (default: 2000)
        num_neighbor_clusters: Number of bidirectional neighbor pairs (None = all neighbors)
        external_input_prob: Probability of external input per neuron (default: 0.2)
        soma_breed: Neuron type ("lif_soma" or "izh_soma")
        soma_config: Configuration name for somas
        synapse_breed: Synapse type
        synapse_config: Configuration name for synapses
        excitatory_ratio: Ratio of excitatory neurons (default: 0.8)
        weight_exc: Weight for excitatory synapses (default: 14.0)
        weight_inh: Weight for inhibitory synapses (default: -10.0)
        seed: Random seed for reproducibility
        topology_type: Connection topology - "ring" (sequential neighbors) or "random" (random selection)

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

    # Set default for num_neighbor_clusters if not specified
    if num_neighbor_clusters is None:
        # Default: connect to all other clusters (backward compatibility but NOT true weak scaling)
        num_neighbor_clusters = num_clusters - 1 if num_clusters > 1 else 0
        print(f"[Warning] num_neighbor_clusters not specified, using all-to-all ({num_neighbor_clusters} neighbors). For TRUE weak scaling, set num_neighbor_clusters=1.")
    else:
        # Validate num_neighbor_clusters
        max_neighbors = num_clusters - 1
        if num_neighbor_clusters > max_neighbors:
            print(f"[Warning] num_neighbor_clusters ({num_neighbor_clusters}) > max possible ({max_neighbors}). Using {max_neighbors}.")
            num_neighbor_clusters = max_neighbors

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

    print(f"  - Neighbor clusters per worker: {num_neighbor_clusters} (DIRECTED RING)")
    print(f"  - Cross-cluster edges per neighbor: {cross_cluster_edges}")
    # With directed ring, each cluster sends cross_cluster_edges to K neighbors
    total_cross_edges_per_worker = cross_cluster_edges * num_neighbor_clusters if num_clusters > 1 else 0
    print(f"  - Total cross-cluster edges per worker (outgoing): {total_cross_edges_per_worker} (constant!)")
    if num_clusters > 1:
        print(f"  - Each worker also receives {total_cross_edges_per_worker} edges from {num_neighbor_clusters} sender(s)")

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
                metadata=[f"cluster_{cluster_id}"]
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
                # performing a list comprehension here makes this an O(N^2) block
                # so instead we compute how many targets we need, and test until we get them
                targets_desired = min(intra_cluster_degree, len(cluster_neurons)-1)
                num_targets = 0
                seen_indices = set()
                seen_indices.add(pre)

                # keep trying to find a non-self post neuron
                while num_targets < targets_desired:
                    # np.array needs to convert the list to a numpy array, built-in random does not
                    post = random.choice(cluster_neurons)

                    if post not in seen_indices:
                        seen_indices.add(post)

                        pre_type = graph.nodes[pre]["type"]
                        weight = weight_exc if pre_type == "excitatory" else weight_inh

                        graph.add_edge(
                            pre,
                            post,
                            synapse_breed=synapse_breed,
                            config=synapse_config,
                            overrides={"hyperparameters": {"weight": weight}},
                            connection_type="intra_cluster",
                            cluster=cluster_id
                        )
                        intra_edges += 1
                        num_targets += 1
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
                            connection_type="intra_cluster",
                            cluster=cluster_id
                        )
                        intra_edges += 1

    # Add inter-cluster connections with CONSTANT per-worker communication
    # Using BIDIRECTIONAL PAIRS topology: each cluster pairs with K neighbors bidirectionally
    # This ensures each cluster has exactly K unique communication partners (not 2K)
    inter_edges = 0
    for cluster_i in range(num_clusters):
        if num_clusters == 1 or num_neighbor_clusters == 0:
            continue  # No other clusters to connect to

        # Select K neighbor clusters based on topology type
        target_clusters = []

        if topology_type == "ring":
            # DIRECTED RING topology (original behavior)
            # Strategy: cluster i connects to next K clusters in ring (unidirectional)
            # For K=1: 0→1→2→3→0 (directed ring, keeps network connected)
            # Each cluster sends to K neighbors and receives from K neighbors
            # With K=1: each cluster has 2 unique communication partners (send-to, receive-from)
            #   - 2 workers: same neighbor for send/receive → 1 unique partner
            #   - 4+ workers: different neighbors for send/receive → 2 unique partners (constant!)
            for offset in range(1, num_neighbor_clusters + 1):
                cluster_j = (cluster_i + offset) % num_clusters
                target_clusters.append(cluster_j)

        elif topology_type == "random":
            # RANDOM topology: randomly select K unique neighbors (excluding self)
            # Each cluster randomly selects num_neighbor_clusters from all other clusters
            # This tests realistic all-to-all communication patterns
            possible_neighbors = [c for c in range(num_clusters) if c != cluster_i]
            num_to_select = min(num_neighbor_clusters, len(possible_neighbors))
            target_clusters = np.random.choice(
                possible_neighbors,
                size=num_to_select,
                replace=False
            ).tolist()

        else:
            raise ValueError(f"Unknown topology_type: {topology_type}. Must be 'ring' or 'random'.")

        pre_ids = neuron_ids[cluster_i]

        # Send cross_cluster_edges to each of the K target clusters
        N = neurons_per_cluster
        exc_boundary = int(N * excitatory_ratio)
        pre_base = cluster_i * N

        for cluster_j in target_clusters:
            total_possible = N * N
            edges_to_add = min(cross_cluster_edges, total_possible)

            if edges_to_add > 0:
                # Deterministic strided pattern: evenly spaced across [0, N*N)
                pair_offset = ((cluster_i * 7919 + cluster_j * 6271) ^ (seed if seed else 0)) % total_possible
                flat_indices = (np.arange(edges_to_add, dtype=np.int64)
                                * (total_possible // edges_to_add)
                                + pair_offset) % total_possible
                pre_offsets = flat_indices // N
                post_offsets = flat_indices % N
                pre_ids = pre_base + pre_offsets
                post_ids = cluster_j * N + post_offsets
                weights = np.where(pre_offsets < exc_boundary, weight_exc, weight_inh)

                for idx in range(edges_to_add):
                    graph.add_edge(
                        int(pre_ids[idx]),
                        int(post_ids[idx]),
                        synapse_breed=synapse_breed,
                        config=synapse_config,
                        overrides={"hyperparameters": {"weight": float(weights[idx])}},
                        connection_type="inter_cluster",
                        cluster=cluster_i
                    )
                    inter_edges += 1

    # Add external inputs
    external_inputs = 0
    for cluster_id, cluster_neurons in enumerate(neuron_ids):
        for post in cluster_neurons:
            if np.random.random() < external_input_prob:
                graph.add_edge(
                    -1,  # External input
                    post,
                    synapse_breed=synapse_breed,
                    config=synapse_config,
                    overrides={"hyperparameters": {"weight": weight_exc}},
                    connection_type="external",
                    metadata=["input_synapse"],
                    cluster=cluster_id  # Assign to post-synaptic neuron's cluster
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


###############################################################################
# Distributed network generation helpers
# Each rank generates only its local cluster's edges independently.
###############################################################################

# Seed offsets to ensure independent RNG streams for different components
_TOPOLOGY_SEED_OFFSET = 1_000_000
_INTRA_SEED_OFFSET = 2_000_000
_CROSS_SEED_OFFSET = 3_000_000
_EXTERNAL_SEED_OFFSET = 4_000_000


def compute_cluster_topology(
    num_clusters: int,
    num_neighbor_clusters: int,
    topology_type: str = "ring",
    seed: int = 42,
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """
    Compute the full cluster connectivity topology deterministically.

    All ranks call this independently and get identical results.

    Args:
        num_clusters: Total number of clusters.
        num_neighbor_clusters: K outgoing neighbors per cluster.
        topology_type: "ring" or "random".
        seed: Base random seed.

    Returns:
        (outgoing, incoming): dicts mapping cluster_id -> sorted list of
        target/source cluster_ids.
    """
    num_neighbor_clusters = min(num_neighbor_clusters, num_clusters - 1)
    outgoing: Dict[int, List[int]] = {}
    incoming: Dict[int, List[int]] = defaultdict(list)

    if num_clusters <= 1 or num_neighbor_clusters == 0:
        for c in range(num_clusters):
            outgoing[c] = []
        return outgoing, dict(incoming)

    rng = np.random.RandomState(seed + _TOPOLOGY_SEED_OFFSET)

    for c in range(num_clusters):
        if topology_type == "ring":
            targets = [(c + offset) % num_clusters
                       for offset in range(1, num_neighbor_clusters + 1)]
        elif topology_type == "random":
            possible = [x for x in range(num_clusters) if x != c]
            targets = rng.choice(
                possible,
                size=min(num_neighbor_clusters, len(possible)),
                replace=False,
            ).tolist()
        else:
            raise ValueError(f"Unknown topology_type: {topology_type}")

        outgoing[c] = sorted(targets)
        for t in targets:
            incoming[t].append(c)

    # Sort incoming lists for deterministic ordering
    for c in incoming:
        incoming[c] = sorted(incoming[c])

    return outgoing, dict(incoming)


def compute_edge_counts_per_cluster(
    num_clusters: int,
    neurons_per_cluster: int,
    intra_cluster_degree: int,
    cross_cluster_edges: int,
    outgoing: Dict[int, List[int]],
    external_input_prob: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-cluster edge counts deterministically (no MPI needed).

    Args:
        num_clusters: Total number of clusters.
        neurons_per_cluster: Neurons per cluster.
        intra_cluster_degree: Edges per neuron within cluster.
        cross_cluster_edges: Edges per outgoing neighbor pair.
        outgoing: Outgoing topology map from compute_cluster_topology().
        external_input_prob: Probability of external input per neuron.
        seed: Base random seed.

    Returns:
        (intra_counts, cross_counts, ext_counts): int arrays of length num_clusters.
    """
    N = neurons_per_cluster
    intra_counts = np.full(num_clusters, N * intra_cluster_degree, dtype=np.int64)
    cross_counts = np.array(
        [len(outgoing[c]) * cross_cluster_edges for c in range(num_clusters)],
        dtype=np.int64,
    )
    ext_counts = np.empty(num_clusters, dtype=np.int64)
    for c in range(num_clusters):
        rng = np.random.RandomState(seed + _EXTERNAL_SEED_OFFSET + c)
        ext_counts[c] = int(np.sum(rng.random(N) < external_input_prob))

    return intra_counts, cross_counts, ext_counts


def build_agent_id_to_rank(
    num_clusters: int,
    neurons_per_cluster: int,
    intra_counts: np.ndarray,
    cross_counts: np.ndarray,
    ext_counts: np.ndarray,
) -> Tuple[np.ndarray, int, int]:
    """
    Build the global agent_id_to_rank array deterministically.

    Agent ID layout:
      [0, num_somas):  soma agent IDs  (soma i -> cluster i//N)
      [num_somas, total_agents):  synapse agent IDs in canonical order:
        for each cluster c:
          intra_counts[c] synapses owned by c
          cross_counts[c] synapses owned by c
          ext_counts[c]   synapses owned by c

    Returns:
        (agent_id_to_rank, num_somas, total_agents)
    """
    num_somas = num_clusters * neurons_per_cluster
    total_synapses = int(intra_counts.sum() + cross_counts.sum() + ext_counts.sum())
    total_agents = num_somas + total_synapses

    agent_id_to_rank = np.empty(total_agents, dtype=np.int32)

    # Soma assignments
    for c in range(num_clusters):
        start = c * neurons_per_cluster
        end = start + neurons_per_cluster
        agent_id_to_rank[start:end] = c

    # Synapse assignments in canonical order
    offset = num_somas
    for c in range(num_clusters):
        n_intra = int(intra_counts[c])
        n_cross = int(cross_counts[c])
        n_ext = int(ext_counts[c])
        total_c = n_intra + n_cross + n_ext
        agent_id_to_rank[offset:offset + total_c] = c
        offset += total_c

    return agent_id_to_rank, num_somas, total_agents


def compute_synapse_id_ranges(
    num_clusters: int,
    num_somas: int,
    intra_counts: np.ndarray,
    cross_counts: np.ndarray,
    ext_counts: np.ndarray,
) -> List[Tuple[int, int, int, int, int, int]]:
    """
    Compute per-cluster synapse agent ID ranges.

    Returns:
        List of (intra_start, intra_end, cross_start, cross_end, ext_start, ext_end)
        for each cluster.
    """
    ranges = []
    offset = num_somas
    for c in range(num_clusters):
        n_intra = int(intra_counts[c])
        n_cross = int(cross_counts[c])
        n_ext = int(ext_counts[c])

        intra_start = offset
        intra_end = intra_start + n_intra
        cross_start = intra_end
        cross_end = cross_start + n_cross
        ext_start = cross_end
        ext_end = ext_start + n_ext

        ranges.append((intra_start, intra_end, cross_start, cross_end, ext_start, ext_end))
        offset = ext_end

    return ranges


def _generate_cross_edges_vectorized(
    source_cluster: int,
    target_cluster: int,
    neurons_per_cluster: int,
    cross_cluster_edges: int,
    excitatory_ratio: float,
    weight_exc: float,
    weight_inh: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate cross-cluster edges for one (source, target) pair.

    Uses deterministic strided connectivity: evenly space edges across
    the N*N possible pairs. This is O(E) with no random sampling overhead.
    """
    N = neurons_per_cluster
    total_possible = N * N
    edges_to_add = min(cross_cluster_edges, total_possible)

    # Deterministic strided pattern: evenly spaced indices across [0, N*N)
    # Offset by (source_cluster ^ target_cluster) to vary pattern per pair
    pair_offset = ((source_cluster * 7919 + target_cluster * 6271) ^ seed) % total_possible
    flat_indices = (np.arange(edges_to_add, dtype=np.int64)
                    * (total_possible // edges_to_add)
                    + pair_offset) % total_possible

    pre_offsets = flat_indices // N
    post_offsets = flat_indices % N

    pre_ids = source_cluster * N + pre_offsets
    post_ids = target_cluster * N + post_offsets

    exc_boundary = int(N * excitatory_ratio)
    weights = np.where(pre_offsets < exc_boundary, weight_exc, weight_inh)

    return pre_ids.astype(np.int64), post_ids.astype(np.int64), weights


def generate_local_cluster_edges(
    cluster_id: int,
    neurons_per_cluster: int,
    intra_cluster_degree: int,
    cross_cluster_edges: int,
    outgoing_targets: List[int],
    incoming_sources: List[int],
    excitatory_ratio: float = 0.8,
    weight_exc: float = 14.0,
    weight_inh: float = -10.0,
    external_input_prob: float = 0.1,
    seed: int = 42,
) -> dict:
    """
    Generate all edges for one cluster's local view (vectorized).

    Each rank calls this for its own cluster. Returns edge data as numpy arrays.

    Returns:
        dict with keys:
        - 'intra': (pre_ids, post_ids, weights)  -- intra-cluster edges
        - 'cross_out': (pre_ids, post_ids, weights, target_clusters)  -- outgoing cross-cluster
        - 'cross_in': (pre_ids, post_ids, weights, source_clusters)   -- incoming cross-cluster
        - 'external': (post_ids,)  -- external input edges
    """
    N = neurons_per_cluster
    base = cluster_id * N
    exc_boundary = int(N * excitatory_ratio)

    # --- Intra-cluster edges (strided pattern) ---
    degree = min(intra_cluster_degree, N - 1)
    total_intra = N * degree

    # For each neuron i, connect to (i + stride*k) % N for k=1..degree
    # where stride ensures good spread. Skip self by starting at offset 1.
    neuron_indices = np.arange(N, dtype=np.int64)
    intra_pre = np.repeat(neuron_indices, degree)
    offsets = np.tile(np.arange(1, degree + 1, dtype=np.int64), N)
    # Use a stride that is coprime with N for good distribution
    stride = max(1, (N // (degree + 1)))
    intra_post = (intra_pre + offsets * stride) % N
    intra_pre = base + intra_pre
    intra_post = base + intra_post

    # Vectorized weight assignment
    pre_offsets = intra_pre - base
    intra_weight = np.where(pre_offsets < exc_boundary, weight_exc, weight_inh)

    # --- Outgoing cross-cluster edges (vectorized) ---
    if outgoing_targets:
        cross_out_parts_pre = []
        cross_out_parts_post = []
        cross_out_parts_weight = []
        cross_out_parts_target = []
        for target_c in outgoing_targets:
            pre, post, w = _generate_cross_edges_vectorized(
                cluster_id, target_c, N, cross_cluster_edges,
                excitatory_ratio, weight_exc, weight_inh, seed,
            )
            cross_out_parts_pre.append(pre)
            cross_out_parts_post.append(post)
            cross_out_parts_weight.append(w)
            cross_out_parts_target.append(np.full(len(pre), target_c, dtype=np.int64))
        cross_out_pre = np.concatenate(cross_out_parts_pre)
        cross_out_post = np.concatenate(cross_out_parts_post)
        cross_out_weight = np.concatenate(cross_out_parts_weight)
        cross_out_target = np.concatenate(cross_out_parts_target)
    else:
        cross_out_pre = np.empty(0, dtype=np.int64)
        cross_out_post = np.empty(0, dtype=np.int64)
        cross_out_weight = np.empty(0, dtype=np.float64)
        cross_out_target = np.empty(0, dtype=np.int64)

    # --- Incoming cross-cluster edges (vectorized) ---
    if incoming_sources:
        cross_in_parts_pre = []
        cross_in_parts_post = []
        cross_in_parts_weight = []
        cross_in_parts_source = []
        for source_c in incoming_sources:
            # Use the SAME seed as the source cluster used for this pair
            pre, post, w = _generate_cross_edges_vectorized(
                source_c, cluster_id, N, cross_cluster_edges,
                excitatory_ratio, weight_exc, weight_inh, seed,
            )
            cross_in_parts_pre.append(pre)
            cross_in_parts_post.append(post)
            cross_in_parts_weight.append(w)
            cross_in_parts_source.append(np.full(len(pre), source_c, dtype=np.int64))
        cross_in_pre = np.concatenate(cross_in_parts_pre)
        cross_in_post = np.concatenate(cross_in_parts_post)
        cross_in_weight = np.concatenate(cross_in_parts_weight)
        cross_in_source = np.concatenate(cross_in_parts_source)
    else:
        cross_in_pre = np.empty(0, dtype=np.int64)
        cross_in_post = np.empty(0, dtype=np.int64)
        cross_in_weight = np.empty(0, dtype=np.float64)
        cross_in_source = np.empty(0, dtype=np.int64)

    # --- External inputs ---
    rng_ext = np.random.RandomState(seed + _EXTERNAL_SEED_OFFSET + cluster_id)
    mask = rng_ext.random(N) < external_input_prob
    ext_post_ids = base + np.where(mask)[0]

    return {
        'intra': (intra_pre, intra_post, intra_weight),
        'cross_out': (cross_out_pre, cross_out_post, cross_out_weight, cross_out_target),
        'cross_in': (cross_in_pre, cross_in_post, cross_in_weight, cross_in_source),
        'external': (ext_post_ids,),
    }


def generate_grid_network(
    grid_size: Tuple[int, int],
    connection_radius: int = 1,
    connection_prob: float = 0.5,
    external_input_prob: float = 0.1,
    soma_breed: str = "lif_soma",
    soma_config: str = "config_0",
    synapse_breed: str = "single_exp_synapse",
    synapse_config: str = "config_0",
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
                metadata=["input_synapse"]
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
    synapse_config: str = "config_0",
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
                metadata=[f"cluster_{cluster_id}"]
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
                    metadata=["input_synapse"]
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


def generate_and_save_partitions(
    output_dir: str,
    num_partitions: int,
    neurons_per_partition: int,
    intra_cluster_degree: int = 10,
    cross_cluster_edges: int = 2000,
    num_neighbor_clusters: int = 1,
    topology_type: str = "ring",
    soma_breed: str = "lif_soma",
    soma_config: str = "config_0",
    synapse_breed: str = "single_exp_synapse",
    synapse_config: str = "config_0",
    excitatory_ratio: float = 0.8,
    weight_exc: float = 14.0,
    weight_inh: float = -10.0,
    external_input_prob: float = 0.1,
    seed: int = 42,
) -> None:
    """Generate synthetic partitioned network and save as partition files.

    Produces one partition file per rank in the format expected by
    NeuromorphicModel.load_from_file(). Can run on a single node (no MPI).

    Args:
        output_dir: Directory to write partition_{rank}.pkl files.
        num_partitions: Number of partitions (should match MPI world size).
        neurons_per_partition: Number of neurons per partition.
        intra_cluster_degree: Edges per neuron within cluster.
        cross_cluster_edges: Cross-cluster edges per neighbor pair.
        num_neighbor_clusters: Number of neighbor clusters per partition.
        topology_type: "ring" or "random".
        soma_breed: Breed name for somas.
        soma_config: Config name for somas.
        synapse_breed: Breed name for synapses.
        synapse_config: Config name for synapses.
        excitatory_ratio: Fraction of excitatory neurons.
        weight_exc: Excitatory synapse weight.
        weight_inh: Inhibitory synapse weight.
        external_input_prob: Probability of external input per neuron.
        seed: Random seed.
    """
    import pickle
    from pathlib import Path

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    size = num_partitions

    # Deterministic topology (identical results regardless of caller)
    outgoing, incoming = compute_cluster_topology(
        num_clusters=size,
        num_neighbor_clusters=num_neighbor_clusters,
        topology_type=topology_type,
        seed=seed,
    )

    intra_counts, cross_counts, ext_counts = compute_edge_counts_per_cluster(
        num_clusters=size,
        neurons_per_cluster=neurons_per_partition,
        intra_cluster_degree=intra_cluster_degree,
        cross_cluster_edges=cross_cluster_edges,
        outgoing=outgoing,
        external_input_prob=external_input_prob,
        seed=seed,
    )

    agent_id_to_rank, num_somas, total_agents = build_agent_id_to_rank(
        num_clusters=size,
        neurons_per_cluster=neurons_per_partition,
        intra_counts=intra_counts,
        cross_counts=cross_counts,
        ext_counts=ext_counts,
    )

    syn_ranges = compute_synapse_id_ranges(
        num_clusters=size,
        num_somas=num_somas,
        intra_counts=intra_counts,
        cross_counts=cross_counts,
        ext_counts=ext_counts,
    )

    print(f"[generate_and_save_partitions] Generating {size} partitions...")
    print(f"  Neurons/partition: {neurons_per_partition}, Total somas: {num_somas}")
    print(f"  Total agents: {total_agents}")

    for r in range(size):
        edges_data = generate_local_cluster_edges(
            cluster_id=r,
            neurons_per_cluster=neurons_per_partition,
            intra_cluster_degree=intra_cluster_degree,
            cross_cluster_edges=cross_cluster_edges,
            outgoing_targets=outgoing.get(r, []),
            incoming_sources=incoming.get(r, []),
            excitatory_ratio=excitatory_ratio,
            weight_exc=weight_exc,
            weight_inh=weight_inh,
            external_input_prob=external_input_prob,
            seed=seed,
        )

        intra_start, intra_end, cross_start, cross_end, ext_start, ext_end = syn_ranges[r]

        local_soma_ids = list(range(
            r * neurons_per_partition,
            (r + 1) * neurons_per_partition,
        ))
        local_id_set = set(local_soma_ids)

        nodes = [{'id': soma_id} for soma_id in local_soma_ids]
        graph_edges = []
        remote_node_ranks = {}

        def add_edge(syn_id, pre_id, post_id, weight):
            graph_edges.append({
                'source': int(pre_id), 'target': int(post_id),
                'synapse_id': syn_id,
                'attributes': {'weight': float(weight)},
            })
            if int(pre_id) >= 0 and int(pre_id) not in local_id_set:
                remote_node_ranks[int(pre_id)] = int(agent_id_to_rank[int(pre_id)])
            if int(post_id) >= 0 and int(post_id) not in local_id_set:
                remote_node_ranks[int(post_id)] = int(agent_id_to_rank[int(post_id)])

        intra_pre, intra_post, intra_weight = edges_data['intra']
        for i in range(len(intra_pre)):
            add_edge(intra_start + i, int(intra_pre[i]), int(intra_post[i]), float(intra_weight[i]))

        cross_out_pre, cross_out_post, cross_out_weight, _ = edges_data['cross_out']
        for i in range(len(cross_out_pre)):
            add_edge(cross_start + i, int(cross_out_pre[i]), int(cross_out_post[i]), float(cross_out_weight[i]))

        ext_post_ids = edges_data['external'][0]
        for i in range(len(ext_post_ids)):
            add_edge(ext_start + i, -1, int(ext_post_ids[i]), float(weight_exc))

        cross_in_pre, cross_in_post, cross_in_weight, cross_in_source = edges_data['cross_in']
        for i in range(len(cross_in_pre)):
            source_c = int(cross_in_source[i])
            source_cross_start = syn_ranges[source_c][2]
            source_targets = outgoing[source_c]
            offset_to_r = source_targets.index(r) * cross_cluster_edges
            ghost_syn_id = source_cross_start + offset_to_r + i % cross_cluster_edges
            graph_edges.append({
                'source': int(cross_in_pre[i]), 'target': int(cross_in_post[i]),
                'synapse_id': ghost_syn_id,
            })
            remote_node_ranks[int(cross_in_pre[i])] = source_c

        partition = {
            'nodes': nodes,
            'edges': graph_edges,
            'remote_node_ranks': remote_node_ranks,
        }

        out_file = out_dir / f"partition_{r}.pkl"
        with open(out_file, 'wb') as f:
            pickle.dump(partition, f)

    print(f"[generate_and_save_partitions] Saved {size} partition files to {out_dir}")


def generate_and_save_local_partition(
    output_dir: str,
    my_rank: int,
    num_partitions: int,
    neurons_per_partition: int,
    intra_cluster_degree: int = 10,
    cross_cluster_edges: int = 2000,
    num_neighbor_clusters: int = 1,
    topology_type: str = "ring",
    soma_breed: str = "lif_soma",
    soma_config: str = "config_0",
    synapse_breed: str = "single_exp_synapse",
    synapse_config: str = "config_0",
    excitatory_ratio: float = 0.8,
    weight_exc: float = 14.0,
    weight_inh: float = -10.0,
    external_input_prob: float = 0.1,
    seed: int = 42,
) -> str:
    """Generate and save only this rank's partition file (distributed).

    Each rank calls this independently — no MPI needed. All ranks compute
    identical deterministic topology, then each generates only its own edges.

    Returns:
        Path to the saved partition file.
    """
    import pickle
    from pathlib import Path

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    size = num_partitions
    r = my_rank

    # Deterministic topology (identical on all ranks)
    outgoing, incoming = compute_cluster_topology(
        num_clusters=size,
        num_neighbor_clusters=num_neighbor_clusters,
        topology_type=topology_type,
        seed=seed,
    )

    intra_counts, cross_counts, ext_counts = compute_edge_counts_per_cluster(
        num_clusters=size,
        neurons_per_cluster=neurons_per_partition,
        intra_cluster_degree=intra_cluster_degree,
        cross_cluster_edges=cross_cluster_edges,
        outgoing=outgoing,
        external_input_prob=external_input_prob,
        seed=seed,
    )

    agent_id_to_rank, num_somas, total_agents = build_agent_id_to_rank(
        num_clusters=size,
        neurons_per_cluster=neurons_per_partition,
        intra_counts=intra_counts,
        cross_counts=cross_counts,
        ext_counts=ext_counts,
    )

    syn_ranges = compute_synapse_id_ranges(
        num_clusters=size,
        num_somas=num_somas,
        intra_counts=intra_counts,
        cross_counts=cross_counts,
        ext_counts=ext_counts,
    )

    # Generate only this rank's edges
    edges_data = generate_local_cluster_edges(
        cluster_id=r,
        neurons_per_cluster=neurons_per_partition,
        intra_cluster_degree=intra_cluster_degree,
        cross_cluster_edges=cross_cluster_edges,
        outgoing_targets=outgoing.get(r, []),
        incoming_sources=incoming.get(r, []),
        excitatory_ratio=excitatory_ratio,
        weight_exc=weight_exc,
        weight_inh=weight_inh,
        external_input_prob=external_input_prob,
        seed=seed,
    )

    intra_start, intra_end, cross_start, cross_end, ext_start, ext_end = syn_ranges[r]

    local_soma_ids = list(range(r * neurons_per_partition, (r + 1) * neurons_per_partition))
    local_id_set = set(local_soma_ids)

    # Graph format: nodes = neurons, edges = synapses
    nodes = [{'id': soma_id} for soma_id in local_soma_ids]
    graph_edges = []
    remote_node_ranks = {}

    def add_edge(syn_id, pre_id, post_id, weight):
        graph_edges.append({
            'source': int(pre_id),
            'target': int(post_id),
            'synapse_id': syn_id,
            'attributes': {'weight': float(weight)},
        })
        # Track remote node ranks
        if int(pre_id) >= 0 and int(pre_id) not in local_id_set:
            remote_node_ranks[int(pre_id)] = int(agent_id_to_rank[int(pre_id)])
        if int(post_id) >= 0 and int(post_id) not in local_id_set:
            remote_node_ranks[int(post_id)] = int(agent_id_to_rank[int(post_id)])

    # Intra-cluster
    intra_pre, intra_post, intra_weight = edges_data['intra']
    for i in range(len(intra_pre)):
        add_edge(intra_start + i, int(intra_pre[i]), int(intra_post[i]), float(intra_weight[i]))

    # Outgoing cross-cluster
    cross_out_pre, cross_out_post, cross_out_weight, _ = edges_data['cross_out']
    for i in range(len(cross_out_pre)):
        add_edge(cross_start + i, int(cross_out_pre[i]), int(cross_out_post[i]), float(cross_out_weight[i]))

    # External input
    ext_post_ids = edges_data['external'][0]
    for i in range(len(ext_post_ids)):
        add_edge(ext_start + i, -1, int(ext_post_ids[i]), float(weight_exc))

    # Ghost edges (incoming cross-cluster: remote source → local target)
    cross_in_pre, cross_in_post, cross_in_weight, cross_in_source = edges_data['cross_in']
    for i in range(len(cross_in_pre)):
        source_c = int(cross_in_source[i])
        source_cross_start = syn_ranges[source_c][2]
        source_targets = outgoing[source_c]
        offset_to_r = source_targets.index(r) * cross_cluster_edges
        ghost_syn_id = source_cross_start + offset_to_r + i % cross_cluster_edges
        graph_edges.append({
            'source': int(cross_in_pre[i]),
            'target': int(cross_in_post[i]),
            'synapse_id': ghost_syn_id,
        })
        remote_node_ranks[int(cross_in_pre[i])] = source_c

    partition = {
        'nodes': nodes,
        'edges': graph_edges,
        'remote_node_ranks': remote_node_ranks,
    }

    out_file = out_dir / f"partition_{r}.pkl"
    with open(out_file, 'wb') as f:
        pickle.dump(partition, f)

    return str(out_file)
