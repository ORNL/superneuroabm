"""
SuperNeuroABM utilities

"""

from pathlib import Path

import yaml

try:
    import networkx as nx
except ImportError:
    nx = None


current_dir = Path(__file__).parent
base_config_fpath = current_dir / "component_base_config.yaml"


def load_component_configurations(config_file: str = base_config_fpath) -> dict:
    """
    Load component configurations from a YAML file.

    Args:
        config_file: Path to the YAML configuration file.

    Returns:
        A dictionary containing the component configurations.
    """
    with open(config_file, "r", encoding="utf-8") as f:
        configurations = yaml.safe_load(f)
        # Make sure all end values are floats
        for component_class in configurations:
            for breed in configurations[component_class]:
                for config_name in configurations[component_class][breed]:
                    for type in configurations[component_class][breed][config_name]:
                        for key, value in configurations[component_class][breed][
                            config_name
                        ][type].items():
                            configurations[component_class][breed][config_name][type][
                                key
                            ] = float(value)
    return configurations


def _none_safe(value):
    """Treat the string 'None' (from GraphML round-trip) as None."""
    return None if value is None or value == "None" else value


def nx_graph_from_model(model, override_internal_states: bool = True):
    """Convert a NeuromorphicModel to a NetworkX graph.

    Args:
        model: A NeuromorphicModel object.
        override_internal_states: If True, adds overrides of internal_states
            and learning_internal_states with post-simulation values.

    Returns:
        A NetworkX DiGraph representing the model.
    """
    if nx is None:
        raise ImportError("NetworkX is required. Install with: pip install networkx")

    graph = nx.DiGraph()

    for soma_id in model._soma_ids:
        soma_breed = model.get_agent_breed(soma_id)
        config = model.get_agent_config_name(soma_id)
        overrides = model.get_agent_config_diff(soma_id)

        if not override_internal_states:
            overrides.pop("internal_states", None)
            overrides.pop("learning_internal_states", None)

        graph.add_node(
            soma_id,
            soma_breed=soma_breed,
            config=config,
            overrides=overrides,
        )

    for synapse_id in model._synapse_ids:
        pre_soma_id, post_soma_id = model.get_synapse_connectivity(synapse_id)
        synapse_breed = model.get_agent_breed(synapse_id)
        config = model.get_agent_config_name(synapse_id)
        overrides = model.get_agent_config_diff(synapse_id)

        if not override_internal_states:
            overrides.pop("internal_states", None)
            overrides.pop("learning_internal_states", None)

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
