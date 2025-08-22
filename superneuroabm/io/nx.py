"""
Saves and loads NetworkX graphs and parses them into SuperNeuroABM networks.

"""

import warnings

import networkx as nx
import numpy as np

from superneuroabm.model import NeuromorphicModel
from superneuroabm.util import load_component_configurations


def model_from_nx_graph(graph: nx.DiGraph) -> NeuromorphicModel:
    """
    Load a NetworkX graph from a file.

    Args:
        graph: A NetworkX DiGraph object.
        Vertices should have 'soma_breed', 'config', and 'overrides' attributes.
        Edges should have 'synapse_breed', 'config', and 'overrides' attributes.

    Returns:
        A NeuromorphicModel object constructed from the graph.
    """
    component_configurations = load_component_configurations()

    model = NeuromorphicModel()
    name2id = {}
    id2name = {}

    # Create somas from graph nodes
    for node, data in graph.nodes(data=True):
        if type(node) == float and np.isnan(node):
            continue
        soma_breed = data.get("soma_breed")
        config_name = data.get("config", "config_0")
        overrides = data.get("overrides", {})

        soma_id = model.create_soma(
            breed=soma_breed,
            config_name=config_name,
            hyperparameters_overrides=overrides.get("hyperparameters"),
            default_internal_state_overrides=overrides.get("internal_state"),
        )
        name2id[node] = soma_id
        id2name[soma_id] = node

    # Create synapses from graph edges
    for u, v, data in graph.edges(data=True):
        synapse_breed = data.get("synapse_breed")
        config_name = data.get("config", "config_0")
        overrides = data.get("overrides", {})

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
