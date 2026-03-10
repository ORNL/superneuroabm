"""
Simple weak scaling test - no visualization, just basic functionality test
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from superneuroabm.io.synthetic_networks import generate_clustered_network
from superneuroabm.io.nx import model_from_nx_graph

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except ImportError:
    rank = 0
    size = 1
    comm = None

# Test parameters
neurons_per_worker = 100
simulation_ticks = 100

if rank == 0:
    print("="*60)
    print(f"Simple Weak Scaling Test")
    print("="*60)
    print(f"MPI size: {size}")
    print(f"Neurons per worker: {neurons_per_worker}")
    print(f"Total neurons: {size * neurons_per_worker}")
    print(f"Simulation ticks: {simulation_ticks}")
    print("="*60)

    # Generate network
    print("\nGenerating network...")
    graph = generate_clustered_network(
        num_clusters=size,
        neurons_per_cluster=neurons_per_worker,
        intra_cluster_prob=0.2,
        inter_cluster_prob=0.01,
        external_input_prob=0.3,  # Higher prob to ensure some inputs
        seed=42
    )
    print(f"Network generated: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
else:
    graph = None

# Create model with METIS partitioning
if rank == 0:
    print("\nCreating model...")

use_metis = size > 1
model = model_from_nx_graph(
    graph,
    enable_internal_state_tracking=False,
    partition_method='metis' if use_metis else None
)

if rank == 0:
    print("Model created successfully")
    print("\nSetting up GPU...")

model.setup(use_gpu=True)

if rank == 0:
    print("GPU setup complete")

# Add some input spikes
input_synapses = model.get_agents_with_tag("input_synapse")
if rank == 0:
    print(f"\nAdding spikes to {len(input_synapses)} input synapses...")

for synapse_id in list(input_synapses)[:10]:  # Just first 10
    model.add_spike(synapse_id=synapse_id, tick=10, value=1.0)

if comm:
    comm.Barrier()

# Run simulation
if rank == 0:
    print(f"\nRunning simulation for {simulation_ticks} ticks...")

start_time = time.time()
model.simulate(ticks=simulation_ticks, update_data_ticks=simulation_ticks)
sim_time = time.time() - start_time

if comm:
    comm.Barrier()
    all_times = comm.gather(sim_time, root=0)
else:
    all_times = [sim_time]

if rank == 0:
    import numpy as np
    print(f"\nSimulation complete!")
    print(f"Time: {np.mean(all_times):.3f}s (avg), {np.max(all_times):.3f}s (max)")
    print("\nSUCCESS!")
