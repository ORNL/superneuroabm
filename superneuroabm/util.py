"""
SuperNeuroABM utilities

"""

from pathlib import Path

import yaml


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
    with open(config_file, "r") as f:
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
