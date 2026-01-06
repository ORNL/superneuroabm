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
print("Using sparse networks (1% density) with LIF neurons")
print("="*70)

# Test different network sizes - start at 5000 and go larger
test_sizes = [5000, 10000, 15000, 20000, 30000, 40000, 50000, 60000, 70000, 80000]

last_success = 0

for neurons in test_sizes:
    print(f"\nTesting {neurons:,} neurons...")

    try:
        # Generate sparse network
        graph = generate_clustered_network(
            num_clusters=1,
            neurons_per_cluster=neurons,
            intra_cluster_prob=0.01,   # Sparse: 1% density
            inter_cluster_prob=0.0,    # Single cluster
            external_input_prob=0.1,
            soma_breed="lif_soma",
            synapse_breed="single_exp_synapse",
            synapse_config="no_learning_config_0",
            seed=42
        )

        print(f"  Network: {graph.number_of_nodes()} nodes, {graph.number_of_edges():,} edges")

        # Create model
        model = model_from_nx_graph(graph, enable_internal_state_tracking=False)
        print(f"  Model created")

        # Setup GPU - this is where memory allocation happens
        model.setup(use_gpu=True)
        print(f"  ✓ GPU setup successful")

        # Run tiny simulation to verify it works
        model.simulate(ticks=5, update_data_ticks=1)  # Update every tick
        print(f"  ✓ Simulation works: {neurons:,} neurons fit in GPU memory")

        last_success = neurons

    except Exception as e:
        error_msg = str(e)
        print(f"  ✗ FAILED: {type(e).__name__}")

        # Check if it's a memory error
        is_memory_error = any(keyword in error_msg.lower()
                             for keyword in ['memory', 'alloc', 'oom', 'out of memory'])

        if is_memory_error:
            print(f"\n" + "="*70)
            print(f"GPU MEMORY LIMIT FOUND")
            print("="*70)
            print(f"Maximum successful: {last_success:,} neurons")
            print(f"Failed at: {neurons:,} neurons")
            print(f"\nRecommended neurons-per-worker for weak scaling:")
            print(f"  Conservative: {last_success // 2:,} neurons")
            print(f"  Aggressive: {int(last_success * 0.8):,} neurons")
            print("="*70)
            break
        else:
            # Non-memory error, print and continue
            print(f"  Error: {error_msg[:200]}")
            print(f"  (Not a memory error, continuing...)")

    print("-"*70)

if last_success == test_sizes[-1]:
    print("\n" + "="*70)
    print(f"All tests passed! GPU can handle at least {last_success:,} neurons")
    print(f"Try larger sizes if needed")
    print("="*70)
