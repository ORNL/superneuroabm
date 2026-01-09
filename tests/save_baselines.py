#!/usr/bin/env python
"""
Convenience script to save all test baselines.

This script runs all tests in test_synapse_and_soma_models.py and saves
their spike times as baselines for future comparison.

Usage:
    cd /home/xxz/superneuroabm/tests
    python save_baselines.py
"""

import sys
from pathlib import Path

# Add parent directory to path
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR.parent))

from tests.test_synapse_and_soma_models import save_all_baselines

if __name__ == "__main__":
    save_all_baselines()
