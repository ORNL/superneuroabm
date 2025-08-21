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
    return configurations
