"""
Find the single-GPU memory limit by testing progressively larger networks.

This helps determine the right network size for weak scaling demonstrations.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from superneuroabm.io.synthetic_networks import generate_clustered_network
from superneuroabm.io.nx import model_from_nx_graph

print("="*70)
print("Finding Single-GPU Memory Limit")
print("="*70)
print("Testing progressively larger networks until memory limit is reached...")
print("="*70)

# Test different network sizes
test_sizes = [1000, 2000, 5000, 10000, 20000, 50000]

for neurons in test_sizes:
    print(f"\nTesting {neurons:,} neurons...")

    try:
        # Generate network
        graph = generate_clustered_network(
            num_clusters=1,
            neurons_per_cluster=neurons,
            intra_cluster_prob=0.3,
            inter_cluster_prob=0.0,
            external_input_prob=0.1,
            soma_breed="lif_soma",
            synapse_breed="single_exp_synapse",
            synapse_config="no_learning_config_0",
            seed=42
        )

        print(f"  Network created: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

        # Create model
        model = model_from_nx_graph(graph, enable_internal_state_tracking=False)
        print(f"  Model created")

        # Setup GPU - this is where memory allocation happens
        model.setup(use_gpu=True)
        print(f"  ✓ SUCCESS: {neurons:,} neurons fit in GPU memory")

        # Run tiny simulation to verify it works
        model.simulate(ticks=10, update_data_ticks=10)
        print(f"  ✓ Simulation works")

    except Exception as e:
        print(f"  ✗ FAILED: {type(e).__name__}: {str(e)[:100]}")
        if "memory" in str(e).lower() or "alloc" in str(e).lower():
            print(f"\n  Memory limit found at ~{neurons:,} neurons")
            print(f"  Recommended neurons-per-worker: {neurons // 2:,} (half of limit)")
            break
        else:
            print(f"  Non-memory error, continuing...")

    print("-"*70)

print("\n" + "="*70)
print("Memory limit search complete!")
print("="*70)
