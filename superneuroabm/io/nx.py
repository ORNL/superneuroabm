"""
Saves and loads NetworkX graphs and parses them into SuperNeuroABM networks.

"""

import warnings
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
            "METIS not installed."
        )

    # Filter out external input nodes (-1)
    nodes_to_remove = [n for n in graph.nodes() if n == -1]
    G_filtered = graph.copy()
    G_filtered.remove_nodes_from(nodes_to_remove)

    # Convert to undirected graph for METIS
    if G_filtered.is_directed():
        G_undirected = G_filtered.to_undirected()
    else:
        G_undirected = G_filtered

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
    partition_dict: Optional[Dict[int, int]] = None,
    skip_remote_bookkeeping: bool = False
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

    # If we have a partition, use bulk registration for O(N_local) creation
    if node_to_rank:
        from sagesim.space import NetworkSpace

        sorted_nodes = sorted([n for n in graph.nodes() if n != -1])
        num_somas = len(sorted_nodes)
        num_synapses = graph.number_of_edges()
        total_agents = num_somas + num_synapses

        # Phase 1: Build agent_id_to_rank array and optionally collect local data
        agent_id_to_rank = np.empty(total_agents, dtype=np.int32)

        if skip_remote_bookkeeping:
            # Optimized: collect local/ghost data during the same O(N+E) iteration
            local_soma_data = {}
            for agent_id, node in enumerate(sorted_nodes):
                name2id[node] = agent_id
                soma_rank = node_to_rank.get(node, agent_id % size)
                agent_id_to_rank[agent_id] = soma_rank
                if soma_rank == rank:
                    local_soma_data[agent_id] = (node, graph.nodes[node])

            local_edge_data = {}
            ghost_edge_data = {}
            synapse_agent_id = num_somas
            for u, v, data in graph.edges(data=True):
                if u in node_to_rank and u >= 0:
                    syn_rank = node_to_rank[u]
                elif v in node_to_rank:
                    syn_rank = node_to_rank[v]
                else:
                    syn_rank = synapse_agent_id % size
                agent_id_to_rank[synapse_agent_id] = syn_rank

                if syn_rank == rank:
                    local_edge_data[synapse_agent_id] = (u, v, data)
                else:
                    post_soma_id = name2id[v]
                    if agent_id_to_rank[post_soma_id] == rank:
                        ghost_edge_data[synapse_agent_id] = (u, v, data)

                synapse_agent_id += 1

            # Free graph memory after extracting all needed data
            del graph
        else:
            for agent_id, node in enumerate(sorted_nodes):
                name2id[node] = agent_id
                agent_id_to_rank[agent_id] = node_to_rank.get(node, agent_id % size)

            edges_list = list(graph.edges(data=True))
            synapse_agent_id = num_somas
            for u, v, data in edges_list:
                if u in node_to_rank and u >= 0:
                    agent_id_to_rank[synapse_agent_id] = node_to_rank[u]
                elif v in node_to_rank:
                    agent_id_to_rank[synapse_agent_id] = node_to_rank[v]
                else:
                    agent_id_to_rank[synapse_agent_id] = synapse_agent_id % size
                synapse_agent_id += 1

        # Phase 2: Bulk register agents and space
        model._agent_factory.bulk_register_agents(total_agents, agent_id_to_rank)
        network_space = model.get_space()
        network_space.bulk_add_agents(total_agents)

        local_agent_map = model._agent_factory._rank2agentid2agentidx.get(rank, {})

        if rank == 0:
            print(f"[SuperNeuroABM] Bulk registration: {num_somas} neurons, {num_synapses} synapses, {total_agents} total")
            print(f"[SuperNeuroABM] Local agents on rank 0: {len(local_agent_map)}")

        if skip_remote_bookkeeping:
            # Phase 3 (optimized): Only create local somas — no remote bookkeeping
            for agent_id, (node, node_data) in local_soma_data.items():
                soma_breed = node_data.get("soma_breed")
                config_name = node_data.get("config", "config_0")
                overrides = node_data.get("overrides", {})
                tags = set(node_data.get("tags", []))
                tags.add(f"nx_node:{node}")

                model.create_soma_at_index(
                    agent_id=agent_id,
                    local_idx=local_agent_map[agent_id],
                    breed=soma_breed,
                    config_name=config_name,
                    overrides=overrides,
                    tags=tags,
                )

            # Phase 4 (optimized): Only create local synapses + ghost connections
            for synapse_agent_id, (u, v, data) in local_edge_data.items():
                synapse_breed = data.get("synapse_breed")
                config_name = data.get("config", "config_0")
                overrides = data.get("overrides", {})
                tags = set(data.get("tags", []))
                tags.add(f"nx_edge:{u}_to_{v}")

                pre_soma_id = name2id.get(u, -1)
                post_soma_id = name2id[v]

                model.create_synapse_at_index(
                    agent_id=synapse_agent_id,
                    local_idx=local_agent_map[synapse_agent_id],
                    breed=synapse_breed,
                    pre_soma_id=pre_soma_id,
                    post_soma_id=post_soma_id,
                    config_name=config_name,
                    learning_rule=data.get("learning_rule"),
                    learning_rule_config=data.get("learning_rule_config", "default"),
                    overrides=overrides,
                    tags=tags,
                )

            # Ghost synapses: only the bidirectional connection (post_soma -> synapse)
            for synapse_agent_id, (u, v, data) in ghost_edge_data.items():
                post_soma_id = name2id[v]
                network_space.connect_agents(post_soma_id, synapse_agent_id, directed=True)

        else:
            # Phase 3: Create somas — full creation for local, lightweight bookkeeping for remote
            for agent_id, node in enumerate(sorted_nodes):
                node_data = graph.nodes[node]
                soma_breed = node_data.get("soma_breed")
                config_name = node_data.get("config", "config_0")
                overrides = node_data.get("overrides", {})
                tags = set(node_data.get("tags", []))
                tags.add(f"nx_node:{node}")

                if agent_id in local_agent_map:
                    model.create_soma_at_index(
                        agent_id=agent_id,
                        local_idx=local_agent_map[agent_id],
                        breed=soma_breed,
                        config_name=config_name,
                        overrides=overrides,
                        tags=tags,
                    )
                else:
                    # Remote soma: lightweight bookkeeping only
                    model._soma_ids.append(agent_id)
                    model.agentid2config[agent_id] = config_name
                    tags.update({"soma", soma_breed})
                    for tag in tags:
                        model.tag2component[tag].add(agent_id)

            # Phase 4: Create synapses — full creation for local, connections for remote
            synapse_agent_id = num_somas
            for u, v, data in edges_list:
                synapse_breed = data.get("synapse_breed")
                config_name = data.get("config", "config_0")
                overrides = data.get("overrides", {})
                tags = set(data.get("tags", []))
                tags.add(f"nx_edge:{u}_to_{v}")

                pre_soma_id = name2id.get(u, -1)
                post_soma_id = name2id[v]

                if synapse_agent_id in local_agent_map:
                    # Local synapse: full creation with property data + connections
                    model.create_synapse_at_index(
                        agent_id=synapse_agent_id,
                        local_idx=local_agent_map[synapse_agent_id],
                        breed=synapse_breed,
                        pre_soma_id=pre_soma_id,
                        post_soma_id=post_soma_id,
                        config_name=config_name,
                        learning_rule=data.get("learning_rule"),
                        learning_rule_config=data.get("learning_rule_config", "default"),
                        overrides=overrides,
                        tags=tags,
                    )
                else:
                    # Remote synapse: only handle connections that affect local somas
                    # Post-soma needs synapse in its neighbor list for STDP bidirectional link
                    if post_soma_id != -1 and post_soma_id in local_agent_map:
                        network_space.connect_agents(post_soma_id, synapse_agent_id, directed=True)

                    # Lightweight bookkeeping for remote synapse
                    model._synapse_ids.append(synapse_agent_id)
                    model.agentid2config[synapse_agent_id] = config_name
                    model.synapse2soma_map[synapse_agent_id]["pre"] = pre_soma_id
                    model.synapse2soma_map[synapse_agent_id]["post"] = post_soma_id
                    if pre_soma_id != -1:
                        model.soma2synapse_map[pre_soma_id]["post"].add(synapse_agent_id)
                    else:
                        tags.add("input_synapse")
                    if post_soma_id != -1:
                        model.soma2synapse_map[post_soma_id]["pre"].add(synapse_agent_id)
                    tags.update({"synapse", synapse_breed})
                    for tag in tags:
                        model.tag2component[tag].add(synapse_agent_id)

                synapse_agent_id += 1

    else:
        # Non-partitioned path: original behavior
        for node, data in graph.nodes(data=True):
            if node == -1:
                continue
            soma_breed = data.get("soma_breed")
            config_name = data.get("config", "config_0")
            overrides = data.get("overrides", {})
            tags = set(data.get("tags", []))
            tags.add(f"nx_node:{node}")

            soma_id = model.create_soma(
                breed=soma_breed,
                config_name=config_name,
                overrides=overrides,
                tags=tags,
            )
            name2id[node] = soma_id

        # Create synapses from graph edges
        for u, v, data in graph.edges(data=True):
            synapse_breed = data.get("synapse_breed")
            config_name = data.get("config", "config_0")
            overrides = data.get("overrides", {})
            tags = set(data.get("tags", []))
            tags.add(f"nx_edge:{u}_to_{v}")

            pre_soma_id = name2id.get(u, -1)
            post_soma_id = name2id[v]
            model.create_synapse(
                breed=synapse_breed,
                pre_soma_id=pre_soma_id,
                post_soma_id=post_soma_id,
                config_name=config_name,
                learning_rule=data.get("learning_rule"),
                learning_rule_config=data.get("learning_rule_config", "default"),
                overrides=overrides,
                tags=tags,
            )

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

        lr_info = model.agentid2learning_rule.get(synapse_id)
        edge_data = dict(
            synapse_breed=synapse_breed,
            config=config,
            overrides=overrides,
        )
        if lr_info is not None:
            edge_data["learning_rule"] = lr_info[0]
            edge_data["learning_rule_config"] = lr_info[1]
        graph.add_edge(pre_soma_id, post_soma_id, **edge_data)

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
    # -1 indicates external synapse
    graph.add_edge(
        -1,
        "A",
        synapse_breed="single_exp_synapse",
        config="config_0",
        overrides={"hyperparameters": {"weight": 13.5}},
    )
    graph.add_edge(
        "A",
        "B",
        synapse_breed="single_exp_synapse",
        config="config_0",
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
