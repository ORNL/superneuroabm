# Large-Scale Distributed Simulation with SuperNeuroABM

## When Do You Need Distributed Simulation?

Single-GPU memory is often limited (typically 16-80 GB). When your spiking neural network exceeds this capacity, you need **distributed simulation** across multiple GPUs and compute nodes.

**Typical Memory Requirements:**
- **10K neurons + 100K synapses**: ~500 MB (single GPU ✓)
- **100K neurons + 1M synapses**: ~5 GB (single GPU ✓)
- **1M neurons + 10M synapses**: ~50 GB (single GPU ✓, high-end GPUs only)
- **10M neurons + 100M synapses**: ~500 GB (**requires distributed simulation**)

When your network exceeds single-GPU capacity, SuperNeuroABM seamlessly scales to **multi-node, multi-GPU execution** using MPI (Message Passing Interface).

---

## How Distributed Simulation Works

### The Big Picture

```
┌─────────────────────────────────────────────────────────────┐
│               Large Spiking Neural Network                  │
│         (Too large to fit on a single GPU)                  │
│                                                             │
│  10M neurons, 100M synapses, ~500 GB memory needed          │
└─────────────────────────────────────────────────────────────┘
                            ↓
              ┌─────────────────────────────┐
              │   Network Partitioning      │
              │  (Assign agents to workers) │
              └─────────────────────────────┘
                            ↓
    ┌──────────────┬──────────────┬──────────────┬──────────────┐
    │   Worker 0   │   Worker 1   │   Worker 2   │   Worker 3   │
    │   Node 0     │   Node 0     │   Node 1     │   Node 1     │
    │   GPU 0      │   GPU 1      │   GPU 0      │   GPU 1      │
    ├──────────────┼──────────────┼──────────────┼──────────────┤
    │ 2.5M neurons │ 2.5M neurons │ 2.5M neurons │ 2.5M neurons │
    │ 25M synapses │ 25M synapses │ 25M synapses │ 25M synapses │
    │   ~125 GB    │   ~125 GB    │   ~125 GB    │   ~125 GB    │
    └──────────────┴──────────────┴──────────────┴──────────────┘
                            ↓
              ┌─────────────────────────────┐
              │   Synchronized Simulation   │
              │  (MPI manages communication)│
              └─────────────────────────────┘
```

### Key Concept: Agent Partitioning and Ghost Agents

**Distributed simulation = Partitioning agents across workers**

Each **worker** (MPI rank) contains:
- **Local agents**: The agents owned by this worker (neurons and synapses)
- **Ghost agents**: Copies of agents from other workers that are neighbors of local agents

**How it works:**
- Each worker has its own **GPU** for parallel computation
- Workers simulate **independently** for `sync_n_ticks` timesteps using their local agents
- State changes of agents in the same rank are reflected **immediately** (no communication needed)
- Workers **communicate** at synchronization points to update the state of all ghost agents across all ranks
- Then continue simulating for another `sync_n_ticks` timesteps

**Example:**
If Worker 0 has Neuron A, and Worker 1 has Neuron B, and they are connected:
- Worker 0 owns Neuron A (local agent) and has a ghost copy of Neuron B
- Worker 1 owns Neuron B (local agent) and has a ghost copy of Neuron A
- During simulation, each worker uses its ghost copies to compute interactions
- At synchronization, workers exchange state updates to keep ghost copies current

---

## SAGESim Handles Everything Automatically

**Good news:** You don't need to worry about the complex details of distributed simulation!

**SAGESim** (SuperNeuroABM's backend ABM framework) handles all of this automatically:

✅ **Ghost agent discovery** and management for cross-worker synapses
✅ **MPI communication** for cross-worker synapses
✅ **State synchronization** between workers
✅ **GPU memory management** on each worker

❌ **Not** automatic: **deciding which rank owns which agent.** SuperNeuroABM reads the
ownership you encode in the partition files — it never infers placement. Partitioning and
load balancing are yours to do up front.

**What you need to do:**
1. Partition the network yourself and write one file per rank (see
   [PARTITION_LOADING.md](PARTITION_LOADING.md))
2. Specify how many nodes and GPUs to use (via SLURM job script) — the file count must
   match the rank count
3. Launch your simulation with `mpirun` or `srun`

From there SAGESim handles the rest, including creating and managing ghost agents.

---

## Setting Up Distributed Simulation

### Step 1: Configure Your SLURM Job Script

Here's a typical setup for **2 nodes, 4 GPUs total**:

```bash
#!/bin/bash
#SBATCH -A your_project       # Project allocation
#SBATCH -J snn_simulation     # Job name
#SBATCH -N 2                  # Number of nodes
#SBATCH --gpus=4              # Total GPUs (2 per node)
#SBATCH -t 01:00:00           # Time limit
#SBATCH -p batch              # Partition

# Load necessary modules
module load PrgEnv-gnu/8.6.0
module load miniforge3/23.11.0-0
module load rocm/6.4.1                    # For AMD GPUs
module load craype-accel-amd-gfx90a       # GPU architecture
module load metis/5.1.0                   # Graph partitioning library

# Activate conda environment
source activate /path/to/your/superneuroabm_env

# Run with 4 MPI workers (1 per GPU)
# -n 4: 4 MPI ranks total
# -c 7: 7 CPU cores per task (adjust based on your system)
# --gpus-per-task=1: Each rank gets 1 GPU
# --gpu-bind=closest: Bind rank to physically closest GPU (NUMA-aware)
srun -n 4 -c 7 --gpus-per-task=1 --gpu-bind=closest \
    python -u my_simulation.py \
    --partition_dir ./partitions

echo "Simulation complete"
```

**Key Parameters:**

| Parameter | Meaning | Example |
|-----------|---------|---------|
| `-N` | Number of compute nodes | `2` |
| `--gpus` | Total number of GPUs | `4` (2 per node) |
| `-n` | Number of MPI workers | `4` (1 per GPU) |
| `--gpus-per-task` | GPUs per MPI worker | `1` (ideal) |
| `--gpu-bind=closest` | NUMA-aware GPU binding | Automatic |
| `-c` | CPU cores per worker | `7` (system-dependent) |

**Best Practice:** Use **1 MPI worker per GPU** for optimal performance.

---

### Step 2: Modify Your Python Script

Your simulation code needs minimal changes for distributed execution:

Each rank loads **its own** partition file. Partitioning happens ahead of time, in your
own producer — see [PARTITION_LOADING.md](PARTITION_LOADING.md) for the file format and
the ownership invariant.

```python
import pickle

from mpi4py import MPI
from superneuroabm.model import NeuromorphicModel

# Initialize MPI (required)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Which worker am I? (0, 1, 2, ...)
size = comm.Get_size()  # How many workers total?

if rank == 0:
    print(f"Running distributed simulation on {size} MPI workers")

# Each rank builds only the agents it owns. The number of partition files must
# equal the launched rank count.
model = NeuromorphicModel(enable_internal_states_tracking=False)  # save memory
model.load_post_owned(f"partition_{rank}.pkl")

# Keep the partition around: it is this rank's source of truth for *which* agents it
# owns. SuperNeuroABM stores id -> properties and nothing else, so any app-level
# labelling ("these are my input synapses") lives in your data, not the model.

# Setup and simulate (SAGESim handles distribution automatically!)
model.setup()

# Add external inputs. Under post-owns, an input synapse lives on whichever rank owns
# its post soma, so every rank injects into its own -- pre == -1 marks an external input.
with open(f"partition_{rank}.pkl", "rb") as f:
    partition = pickle.load(f)
input_synapses = [syn["id"] for syn in partition["synapses"] if syn["pre"] == -1]

for syn in input_synapses:
    model.add_local_spike(synapse_id=syn, tick=10, value=1.0)  # owner-only, no MPI

# Run simulation (all workers participate)
model.simulate(ticks=10000, update_data_ticks=100)

# Analyze results (each worker has its local results)
if rank == 0:
    print("Simulation complete!")
    # Gather results from all workers if needed
```

---

## Network Partitioning Methods

**Network partitioning** determines which agents are assigned to which workers. Different networks may perform better with different partitioning methods.

> **SuperNeuroABM does not partition for you.** Deciding which rank owns which soma,
> assigning globally-unique agent IDs, and guaranteeing the partition is complete are all
> the *producer's* job — use METIS, a generator, or a rule, then write one file per rank
> and load it with `load_post_owned()`. See
> [PARTITION_LOADING.md](PARTITION_LOADING.md) §1 for the full responsibility boundary.
> The two methods below are guidance for writing that producer, not framework options.

### Method 1: Round-Robin (Default)

**How it works:**
- Agents assigned in sequence: Agent 0 → Worker 0, Agent 1 → Worker 1, ..., Agent N → Worker 0, ...
- Simple and deterministic assignment
- Does not consider network topology

**Example** (in your producer, before writing the per-rank files):
```python
owner = {soma_id: soma_id % num_ranks for soma_id in soma_ids}
```

**Characteristics:**
- Guarantees equal number of agents per worker (perfect load balance)
- Connected agents may be distributed across different workers
- Deterministic assignment (same result every time)

---

### Method 2: METIS Graph Partitioning

**How it works:**
- Uses the **METIS algorithm** to partition the network graph
- Attempts to minimize edge cuts (connections between workers)
- Groups connected agents together when possible
- Topology-aware partitioning

**Example** (in your producer, before writing the per-rank files):
```python
import networkx as nx
import nxmetis                                  # or pymetis / the METIS CLI

_, parts = nxmetis.partition(graph, num_ranks)  # parts[r] = soma ids for rank r
owner = {sid: r for r, ids in enumerate(parts) for sid in ids}
```

**Characteristics:**
- Attempts to keep connected agents on the same worker
- May have slight load imbalance between workers
- Requires METIS library (usually pre-installed on HPC systems)
- Performance depends on network structure

**Partition Quality:**
Report the edge cut ratio from your producer, where the whole network is in one process:

```python
cut = sum(1 for u, v in graph.edges() if owner[u] != owner[v])
print(f"Edge cut ratio: {cut / graph.number_of_edges():.4f} "
      f"({cut:,} of {graph.number_of_edges():,} edges cross workers)")
```

The edge cut ratio indicates what fraction of connections cross worker boundaries. Lower values mean fewer ghost agents and less communication overhead.

---

## How Workers Communicate: Ghost Agents and Synchronized Simulation

### Simulation Cycle

Each MPI worker follows this cycle:

```
┌─────────────────────────────────────────────────┐
│  STEP 1: Simulate Independently                 │
│  Each worker runs simulation for sync_n_ticks   │
│                                                 │
│  Worker 0: Tick 0 → 100                         │
│    - Updates local agents (owned by Worker 0)   │
│    - Reads ghost agents (copies from other      │
│      workers) as needed                         │
│    - State changes to local agents reflected    │
│      immediately (no communication)             │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│  STEP 2: MPI Communication                      │
│  Workers exchange state updates for all         │
│  ghost agents across all ranks                  │
│                                                 │
│  Each worker sends:                             │
│  - Updated states of its local agents           │
│                                                 │
│  Each worker receives:                          │
│  - Updated states for its ghost agents          │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│  STEP 3: Update Ghost Agents                    │
│  Apply received updates to ghost agent copies   │
│                                                 │
│  Worker 0 updates its ghost copies of agents    │
│  that belong to Workers 1, 2, 3                 │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│  STEP 4: Repeat                                 │
│  Continue simulation for next sync_n_ticks      │
│                                                 │
│  Worker 0: Tick 100 → 200                       │
│  (using updated ghost agents)                   │
└─────────────────────────────────────────────────┘
```

### Ghost Agents Example

Consider two neurons on different workers that are connected:

```
Worker 0:                           Worker 1:
┌────────────────────┐              ┌────────────────────┐
│ Neuron A (LOCAL)   │              │ Neuron A (GHOST)   │
│ - Owned by Worker 0│              │ - Copy from Wkr 0  │
│ - v = -65 mV       │              │ - v = -65 mV       │
│                    │              │                    │
│ Neuron B (GHOST)   │              │ Neuron B (LOCAL)   │
│ - Copy from Wkr 1  │              │ - Owned by Worker 1│
│ - v = -70 mV       │              │ - v = -70 mV       │
└────────────────────┘              └────────────────────┘

During simulation (Ticks 0-100):
Worker 0:                           Worker 1:
- Updates Neuron A (local)          - Updates Neuron B (local)
- Reads Neuron B state from         - Reads Neuron A state from
  ghost copy (no communication)       ghost copy (no communication)
- Neuron A spikes at tick 50        - Neuron B receives input
  → state change immediate            from its ghost copy of A
                                      (but ghost is stale!)

At synchronization point (Tick 100):
┌────────────────────────────────────────┐
│  MPI Communication                     │
│                                        │
│  Worker 0 → Worker 1:                  │
│    Neuron A state (v, spikes, etc.)    │
│                                        │
│  Worker 1 → Worker 0:                  │
│    Neuron B state (v, spikes, etc.)    │
└────────────────────────────────────────┘

After synchronization:
Worker 0:                           Worker 1:
- Ghost copy of Neuron B updated    - Ghost copy of Neuron A updated
- Now has current state from Wkr 1  - Now has current state from Wkr 0
```

**Key Points:**
- **Local agents**: State changes are immediate (no communication delay)
- **Ghost agents**: State updates occur only at synchronization points (every `sync_n_ticks`)
- **Communication content**: States of all agents that are ghost copies on other workers
- SAGESim automatically manages which agents need ghost copies based on network connectivity

**You don't need to write any communication code!**

---

## Synchronization Interval: `sync_n_ticks`

The `sync_n_ticks` parameter controls how often workers communicate to update ghost agents.

**How it works:**
- Workers simulate independently for `sync_n_ticks` timesteps
- During this period, ghost agents become stale (not updated)
- At synchronization points, ghost agents are updated with current states from owning workers
- Then simulation continues for another `sync_n_ticks`

**Trade-offs:**

**Smaller `sync_n_ticks` (e.g., 10-50 ticks):**
- Ghost agents updated more frequently
- More MPI communication overhead
- Better for networks with fast dynamics where staleness matters

**Larger `sync_n_ticks` (e.g., 500-1000 ticks):**
- Ghost agents updated less frequently
- Less MPI communication overhead
- Ghost agents may be significantly stale between updates

**Default**: `sync_n_ticks = 100`

The optimal value depends on your specific network structure and dynamics

---

## Complete Example: Large Network Simulation

### Python Script: `large_network_sim.py`

```python
#!/usr/bin/env python3
"""
Large-scale distributed spiking neural network simulation
"""
import time
import pickle
import argparse
from pathlib import Path

from mpi4py import MPI

from superneuroabm.model import NeuromorphicModel


def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--partition_dir', type=str, required=True,
                        help='Directory of per-rank partition files '
                             '(partition_0.pkl, partition_1.pkl, ...)')
    parser.add_argument('--ticks', type=int, default=10000,
                        help='Number of simulation ticks')
    parser.add_argument('--sync_ticks', type=int, default=100,
                        help='Synchronization interval (ticks between MPI communication)')
    args = parser.parse_args()

    # Load network
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Distributed Simulation Configuration")
        print(f"{'='*60}")
        print(f"MPI Workers: {size}")
        print(f"Partitions: {args.partition_dir}")
        print(f"Simulation Ticks: {args.ticks}")
        print(f"Sync Interval: {args.sync_ticks}")
        print(f"{'='*60}\n")

    # Each rank builds only the agents it owns. Partitioning already happened in
    # the producer that wrote these files -- see PARTITION_LOADING.md. The number of
    # files must equal the launched rank count.
    if rank == 0:
        print(f"\nLoading rank-local partitions...")

    t_start = time.time()

    partition_path = Path(args.partition_dir) / f"partition_{rank}.pkl"
    model = NeuromorphicModel(enable_internal_states_tracking=False)  # save memory
    model.load_post_owned(str(partition_path))

    # This rank's own record of what it owns -- the model itself only stores
    # id -> properties, so app-level labelling stays in your data.
    with open(partition_path, "rb") as f:
        partition = pickle.load(f)

    t_partition = time.time() - t_start

    if rank == 0:
        print(f"Local build complete in {t_partition:.2f} sec")

    # Setup model
    if rank == 0:
        print(f"\nSetting up model (allocating GPU memory, compiling kernels)...")

    t_start = time.time()
    model.setup()
    t_setup = time.time() - t_start

    if rank == 0:
        print(f"Setup complete in {t_setup:.2f} sec")

    # Add external inputs. pre == -1 marks an external input synapse; each rank
    # injects only into the ones it owns, so this needs no MPI.
    input_synapses = [syn["id"] for syn in partition["synapses"] if syn["pre"] == -1]
    if rank == 0:
        print(f"\nAdding external inputs...")
    print(f"  rank {rank}: {len(input_synapses)} input synapses")
    for syn in input_synapses[:10]:  # Add spikes to first 10 inputs
        model.add_local_spike(synapse_id=syn, tick=10, value=1.0)

    # Run simulation
    if rank == 0:
        print(f"\nStarting simulation...")
        print(f"{'='*60}")

    t_start = time.time()

    model.simulate(
        ticks=args.ticks,
        update_data_ticks=args.sync_ticks  # Sync interval
    )

    t_sim = time.time() - t_start

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Simulation complete!")
        print(f"{'='*60}")
        print(f"Total simulation time: {t_sim:.2f} sec")
        print(f"Simulation speed: {args.ticks/t_sim:.1f} ticks/sec")
        print(f"Biological time simulated: {args.ticks * 1e-3:.2f} sec")
        print(f"Speedup vs. real-time: {(args.ticks * 1e-3) / t_sim:.2f}×")

    # Analyze results (each worker has local results)
    local_soma_count = len(partition["somas"])
    local_synapse_count = len(partition["synapses"])

    # Gather stats from all workers
    all_soma_counts = comm.gather(local_soma_count, root=0)
    all_synapse_counts = comm.gather(local_synapse_count, root=0)

    if rank == 0:
        print(f"\nLoad Balance:")
        print(f"{'='*60}")
        for i in range(size):
            print(f"Worker {i}: {all_soma_counts[i]} neurons, {all_synapse_counts[i]} synapses")

        avg_neurons = sum(all_soma_counts) / size
        max_imbalance = max(abs(c - avg_neurons) / avg_neurons for c in all_soma_counts) * 100
        print(f"\nMax load imbalance: {max_imbalance:.1f}%")


if __name__ == "__main__":
    main()
```

---

### SLURM Job Script: `run_large_sim.sh`

```bash
#!/bin/bash
#SBATCH -A your_project
#SBATCH -J large_snn_sim
#SBATCH -o logs/snn_%j.out
#SBATCH -e logs/snn_%j.err
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 4                    # 4 compute nodes
#SBATCH --gpus=16               # 16 GPUs total (4 per node)

# Prevent SLURM from resetting environment
unset SLURM_EXPORT_ENV

# Load modules
module load PrgEnv-gnu/8.6.0
module load miniforge3/23.11.0-0
module load rocm/6.4.1
module load craype-accel-amd-gfx90a
module load metis/5.1.0

# Activate environment
source activate /path/to/superneuroabm_env

# Create log directory
mkdir -p logs

# Run with 16 MPI workers (1 per GPU)
srun -n 16 -c 7 --gpus-per-task=1 --gpu-bind=closest \
    python -u large_network_sim.py \
    --partition_dir ./partitions \
    --ticks 10000 \
    --sync_ticks 100

echo "Job complete"
date
```



## Summary

### Key Takeaways

1. **SAGESim handles all MPI complexity** - You just specify worker count and partition method

2. **Ghost agents** enable distributed simulation - Each worker has local agents and ghost copies of neighbors from other workers

3. **State changes to local agents are immediate** - No communication needed within the same rank

4. **Ghost agents are synchronized** every `sync_n_ticks` via MPI communication

5. **Two partition methods available** - Round-robin (default) and METIS (graph-based). Performance depends on your specific network

6. **Distributed simulation is transparent** - Same Python code works for single-GPU and multi-GPU

### Quick Reference

**Enable distributed simulation:**
```python
from mpi4py import MPI
from superneuroabm.model import NeuromorphicModel

rank = MPI.COMM_WORLD.Get_rank()
model = NeuromorphicModel()
model.load_post_owned(f"partition_{rank}.pkl")   # partition offline; see PARTITION_LOADING.md
```

**Launch with MPI:**
```bash
srun -n 4 --gpus-per-task=1 --gpu-bind=closest python my_sim.py
```

**That's it!** SAGESim and SuperNeuroABM handle the rest automatically.

---

## Further Reading

- **METIS Documentation**: [http://glaros.dtc.umn.edu/gkhome/metis/metis/overview](http://glaros.dtc.umn.edu/gkhome/metis/metis/overview)
- **MPI4PY Documentation**: [https://mpi4py.readthedocs.io/](https://mpi4py.readthedocs.io/)
- **SLURM GPU Binding**: [https://slurm.schedmd.com/gres.html](https://slurm.schedmd.com/gres.html)
- **SAGESim Repository**: [https://github.com/ORNL/SAGESim](https://github.com/ORNL/SAGESim)

---

**Version**: 1.0
**Last Updated**: November 2025
**Authors**: SuperNeuroABM Development Team, Oak Ridge National Laboratory
