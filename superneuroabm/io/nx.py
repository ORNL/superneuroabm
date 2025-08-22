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
        config = component_configurations["soma"][soma_breed][config_name]
        overrides = data.get("overrides", {})

        if not soma_breed:
            raise ValueError(f"Node {node} is missing 'soma_breed' attribute.")

        # Apply overrides to soma configuration
        for override_category in overrides:
            if override_category in config:
                category_defaults = config[override_category]
                for parameter_name, parameter_value in overrides[
                    override_category
                ].items():
                    if parameter_name in category_defaults:
                        category_defaults[parameter_name] = parameter_value
                    else:
                        warnings.warn(
                            f"Parameter '{parameter_name}' not found in '{override_category}' for node {node}.",
                            UserWarning,
                        )
        soma_id = model.create_soma(
            breed=soma_breed,
            parameters=[float(val) for val in config["hyperparameters"].values()],
            default_internal_state=[
                float(val) for val in config["default_internal_state"].values()
            ],
        )
        name2id[node] = soma_id
        id2name[soma_id] = node

    # Create synapses from graph edges
    for u, v, data in graph.edges(data=True):
        synapse_breed = data.get("synapse_breed")
        config_name = data.get("config", "config_0")
        config = component_configurations["synapse"][synapse_breed][config_name]
        overrides = data.get("overrides", {})

        if not synapse_breed:
            raise ValueError(f"Edge ({u}, {v}) is missing 'synapse_breed' attribute.")

        # Apply overrides to soma configuration
        for override_category in overrides:
            if override_category in config:
                category_defaults = config[override_category]
                for parameter_name, parameter_value in overrides[
                    override_category
                ].items():
                    if parameter_name in category_defaults:
                        category_defaults[parameter_name] = parameter_value
                    else:
                        warnings.warn(
                            f"Parameter '{parameter_name}' not found in '{override_category}' for edge {u} -> {v}.",
                            UserWarning,
                        )

        pre_soma_id = name2id.get(u, np.nan)  # External input if not found
        post_soma_id = name2id[v]
        model.create_synapse(
            breed=synapse_breed,
            pre_soma_id=pre_soma_id,
            post_soma_id=post_soma_id,
            parameters=[float(val) for val in config["hyperparameters"].values()],
            default_internal_state=[
                float(val) for val in config["default_internal_state"].values()
            ],
            learning_parameters=[
                float(val) for val in config["learning_hyperparameters"].values()
            ],
            default_internal_learning_state=[
                float(val) for val in config.get("default_learning_state", {}).values()
            ],
        )

    return model


if __name__ == "__main__":
    # Example usage
    graph = nx.DiGraph()
    graph.add_node(
        "A",
        soma_breed="lif_soma",
        config="config_0",
        overrides={
            "hyperparameters": {"R": 1.1e6},
            "default_internal_state": {"v": -60.01},
        },
    )
    graph.add_node(
        "B",
        soma_breed="izh_soma",
        config="config_0",
        overrides={
            "hyperparameters": {"a": 0.0102, "b": 5.001},
            "default_internal_state": {"v": -75.002},
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
