"""
Saves and loads NetworkX graphs and parses them into SuperNeuroABM networks.

"""

import warnings
import pickle
from typing import Dict, Optional

import networkx as nx
import numpy as np

from superneuroabm.model import NeuromorphicModel
from superneuroabm.util import load_component_configurations


def generate_metis_partition(graph: nx.DiGraph, num_workers: int) -> Dict[int, int]:
    """
    Generate network partition using METIS for optimal agent-to-worker assignment.

    This function creates a partition that minimizes cross-worker communication by grouping
    connected nodes together. The partition can significantly improve multi-worker performance
    (10-20× reduction in MPI overhead).

    Args:
        graph: NetworkX graph to partition
        num_workers: Number of MPI workers

    Returns:
        Dictionary mapping node -> rank

    Raises:
        ImportError: If metis is not installed
    """
    try:
        import metis
    except ImportError:
        raise ImportError(
            "METIS not installed. Install with: pip install metis\n"
            "Note: Requires METIS library installed on system"
        )

    # Convert to undirected graph for METIS
    if graph.is_directed():
        G_undirected = graph.to_undirected()
    else:
        G_undirected = graph

    # Filter out NaN nodes
    nodes_to_remove = [n for n in G_undirected.nodes() if isinstance(n, float) and np.isnan(n)]
    G_undirected = G_undirected.copy()
    G_undirected.remove_nodes_from(nodes_to_remove)

    # Create adjacency list (METIS format)
    node_list = list(G_undirected.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}

    adjacency = []
    for node in node_list:
        neighbors = [node_to_idx[neighbor] for neighbor in G_undirected.neighbors(node)]
        adjacency.append(neighbors)

    # Run METIS
    print(f"[SuperNeuroABM] Running METIS partition with {num_workers} partitions...")
    _, partition_array = metis.part_graph(adjacency, nparts=num_workers, recursive=True)

    # Normalize partition indices to start from 0
    # METIS may return partitions starting from 1 when nparts=1
    unique_partitions = sorted(set(partition_array))
    partition_remap = {old_id: new_id for new_id, old_id in enumerate(unique_partitions)}
    partition_array = [partition_remap[p] for p in partition_array]

    # Convert to dict mapping original node -> rank
    partition_dict = {}
    for idx, rank in enumerate(partition_array):
        original_node = node_list[idx]
        partition_dict[original_node] = int(rank)

    # Calculate partition quality
    total_edges = 0
    cross_worker_edges = 0
    for u, v in graph.edges():
        if u in partition_dict and v in partition_dict:
            total_edges += 1
            if partition_dict[u] != partition_dict[v]:
                cross_worker_edges += 1

    edge_cut_ratio = cross_worker_edges / total_edges if total_edges > 0 else 0

    print(f"[SuperNeuroABM] Partition quality:")
    print(f"  - Edge cut ratio (P_cross): {edge_cut_ratio:.4f}")
    print(f"  - Total edges: {total_edges}, Cross-worker edges: {cross_worker_edges}")

    return partition_dict


def model_from_nx_graph(
    graph: nx.DiGraph,
    enable_internal_state_tracking: bool = True,
    partition_method: Optional[str] = None,
    partition_dict: Optional[Dict[int, int]] = None
) -> NeuromorphicModel:
    """
    Load a NetworkX graph and create a NeuromorphicModel.

    Args:
        graph: A NetworkX DiGraph object.
        Vertices should have 'soma_breed' and 'config' attributes, and optionally
            'overrides' and 'tags attributes.
        Edges should have 'synapse_breed' and 'config' attributes, and optionally
            'overrides' and 'tags attributes.
        enable_internal_state_tracking: If True (default), tracks internal states history
            during simulation. If False, disables tracking to save memory and improve performance.
        partition_method: Partition method to use. Options:
            - None: No partitioning (default, round-robin assignment)
            - 'metis': Generate METIS partition (requires multiple MPI workers)
        partition_dict: Pre-computed partition dictionary mapping node_id -> rank.
            If provided, this overrides partition_method.

    Returns:
        A NeuromorphicModel object constructed from the graph.
    """
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    component_configurations = load_component_configurations()

    model = NeuromorphicModel(enable_internal_state_tracking=enable_internal_state_tracking)

    # Handle partitioning based on method
    node_to_rank = None

    if partition_dict is not None:
        # Use provided partition dictionary
        node_to_rank = partition_dict
        if rank == 0:
            print(f"[SuperNeuroABM] Using provided partition dictionary")

    elif partition_method == 'metis':
        if size == 1:
            if rank == 0:
                print("[SuperNeuroABM] Warning: METIS partition requested but running with single worker. Skipping partition.")
        else:
            if rank == 0:
                print(f"\n{'='*60}")
                print(f"Multi-worker mode: {size} workers")
                print(f"Generating METIS partition for optimal performance...")
                print(f"{'='*60}\n")

                # Generate METIS partition (only rank 0)
                node_to_rank = generate_metis_partition(graph, size)

            # Broadcast partition to all ranks
            node_to_rank = comm.bcast(node_to_rank, root=0)

            if rank == 0:
                print(f"[SuperNeuroABM] Partition generated and broadcast to all workers")

    elif partition_method is None:
        if size > 1 and rank == 0:
            print(f"[SuperNeuroABM] Multi-worker mode ({size} workers) - using round-robin assignment")
            print(f"[SuperNeuroABM] Tip: Use partition_method='metis' for better performance")
    else:
        raise ValueError(f"Unknown partition_method: {partition_method}. Use None or 'metis'")

    name2id = {}
    id2name = {}

    # If we have a partition, we need to convert node->rank to agent_id->rank
    # We do this by creating agents in a specific order based on the partition
    if node_to_rank:
        # Create mapping from agent_id (creation order) to rank
        # We'll create nodes in sorted order to ensure consistency across all ranks
        sorted_nodes = sorted([n for n in graph.nodes() if not (isinstance(n, float) and np.isnan(n))])
        agent_id_to_rank = {}
        agent_id = 0
        for node in sorted_nodes:
            if node in node_to_rank:
                agent_id_to_rank[agent_id] = node_to_rank[node]
                agent_id += 1

        # Load this mapping directly into the model (no file needed)
        model._agent_factory._partition_mapping = agent_id_to_rank
        model._agent_factory._partition_loaded = True

        if rank == 0:
            print(f"[SuperNeuroABM] Converted node partition to agent_id partition")

        # Create somas in sorted order to match partition
        for node in sorted_nodes:
            data = graph.nodes[node]
            soma_breed = data.get("soma_breed")
            config_name = data.get("config", "config_0")
            overrides = data.get("overrides", {})
            tags = set(data.get("tags", []))
            tags.add(f"nx_node:{node}")

            soma_id = model.create_soma(
                breed=soma_breed,
                config_name=config_name,
                hyperparameters_overrides=overrides.get("hyperparameters"),
                default_internal_state_overrides=overrides.get("internal_state"),
                tags=tags,
            )
            name2id[node] = soma_id
            id2name[soma_id] = node
    else:
        # Create somas from graph nodes (original behavior)
        for node, data in graph.nodes(data=True):
            if type(node) == float and np.isnan(node):
                continue
            soma_breed = data.get("soma_breed")
            config_name = data.get("config", "config_0")
            overrides = data.get("overrides", {})
            tags = set(data.get("tags", []))
            tags.add(f"nx_node:{node}")

            soma_id = model.create_soma(
                breed=soma_breed,
                config_name=config_name,
                hyperparameters_overrides=overrides.get("hyperparameters"),
                default_internal_state_overrides=overrides.get("internal_state"),
                tags=tags,
            )
            name2id[node] = soma_id
            id2name[soma_id] = node

    # Create synapses from graph edges
    synapse_count = 0
    for u, v, data in graph.edges(data=True):
        synapse_breed = data.get("synapse_breed")
        config_name = data.get("config", "config_0")
        overrides = data.get("overrides", {})
        tags = set(data.get("tags", []))
        tags.add(f"nx_edge:{u}_to_{v}")

        pre_soma_id = name2id.get(u, np.nan)  # External input if not found
        post_soma_id = name2id[v]
        model.create_synapse(
            breed=synapse_breed,
            pre_soma_id=pre_soma_id,
            post_soma_id=post_soma_id,
            config_name=config_name,
            hyperparameters_overrides=overrides.get("hyperparameters"),
            default_internal_state_overrides=overrides.get("internal_state"),
            learning_hyperparameters_overrides=overrides.get(
                "learning_hyperparameters"
            ),
            default_internal_learning_state_overrides=overrides.get(
                "default_internal_learning_state"
            ),
            tags=tags,
        )
        synapse_count += 1

    return model


def nx_graph_from_model(
    model: NeuromorphicModel, override_internal_state: bool = True
) -> nx.DiGraph:
    """
    Convert a NeuromorphicModel to a NetworkX graph.

    Args:
        model: A NeuromorphicModel object.
        override_internal_state: If True, adds overrides of internal_state
        and internal_learning_state with post simulation internal_state
        and internal_learning_state.

    Returns:
        A NetworkX DiGraph representing the model.
    """
    graph = nx.DiGraph()

    # Add nodes for somas
    for soma_id in model.get_agents_with_tag("soma"):
        soma_breed = model.get_agent_breed(soma_id).name
        config = model.get_agent_config_name(soma_id)
        overrides = model.get_agent_config_diff(soma_id)
        # TODO: implement tags writing
        tags = model.get_agent_tags(soma_id)

        if not override_internal_state:
            # Remove internal state overrides if not needed
            overrides.pop("internal_state", None)
            overrides.pop("internal_learning_state", None)

        graph.add_node(
            soma_id,
            soma_breed=soma_breed,
            config=config,
            overrides=overrides,
        )

    # Add edges for synapses
    for synapse_id in model.get_agents_with_tag("synapse"):
        pre_soma_id, post_soma_id = model.get_synapse_connectivity(synapse_id)
        synapse_breed = model.get_agent_breed(synapse_id).name
        config = model.get_agent_config_name(synapse_id)
        overrides = model.get_agent_config_diff(synapse_id)

        if not override_internal_state:
            # Remove internal state overrides if not needed
            overrides.pop("internal_state", None)
            overrides.pop("internal_learning_state", None)

        graph.add_edge(
            pre_soma_id,
            post_soma_id,
            synapse_breed=synapse_breed,
            config=config,
            overrides=overrides,
        )

    return graph


if __name__ == "__main__":
    # Example usage
    graph = nx.DiGraph()
    graph.add_node(
        "A",
        soma_breed="lif_soma",
        config="config_0",
        overrides={
            "hyperparameters": {"R": 1.1e6},
            "internal_state": {"v": -60.01},
        },
    )
    graph.add_node(
        "B",
        soma_breed="izh_soma",
        config="config_0",
        overrides={
            "hyperparameters": {"a": 0.0102, "b": 5.001},
            "internal_state": {"v": -75.002},
        },
    )
    # np.nan indicates external synapse
    graph.add_edge(
        np.nan,
        "A",
        synapse_breed="single_exp_synapse",
        config="no_learning_config_0",
        overrides={"hyperparameters": {"weight": 13.5}},
    )
    graph.add_edge(
        "A",
        "B",
        synapse_breed="single_exp_synapse",
        config="no_learning_config_0",
        overrides={"hyperparameters": {"weight": 13.5}},
    )

    model = model_from_nx_graph(graph)
    model.setup(use_gpu=True)

    # Add spikes to the synapse connected to the first soma
    input_synapses = model.get_agents_with_tag("input_synapse")
    model.add_spike(synapse_id=input_synapses.pop(), tick=10, value=1.0)

    model.simulate(ticks=200, update_data_ticks=200)

    # Retrieve and print soma spikes
    for soma_id in model.get_agents_with_tag("soma"):
        spikes = model.get_spike_times(soma_id)
        print(f"Soma {soma_id} spikes: {spikes}")

    # Print the graph structure with all attributes
    graph_out = nx_graph_from_model(model)
    print("---------------------------------------------------------------")
    print("Graph structure (override internal states):")
    for node, data in graph_out.nodes(data=True):
        print(f"Node {node}: {data}")
    for u, v, data in graph_out.edges(data=True):
        print(f"Edge {u} -> {v}: {data}")

    print("---------------------------------------------------------------")
    print("\n")
    print("---------------------------------------------------------------")
    # Print the graph structure with all attributes
    graph_out = nx_graph_from_model(model, override_internal_state=False)
    print("Graph structure (do not override internal states):")
    for node, data in graph_out.nodes(data=True):
        print(f"Node {node}: {data}")
    for u, v, data in graph_out.edges(data=True):
        print(f"Edge {u} -> {v}: {data}")
    print("---------------------------------------------------------------")
