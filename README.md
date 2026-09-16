# SuperNeuroABM

**SuperNeuroABM** is a GPU-based multi-agent simulation framework for neuromorphic computing. Built on top of [SAGESim](https://github.com/ORNL/SAGESim), it enables fast and scalable simulation of spiking neural networks on both NVIDIA and AMD GPUs.

## Key Features

- **GPU Acceleration**: Leverages CUDA (NVIDIA) or ROCm (AMD) for high-performance simulation
- **Scalable**: From single GPU to multi-GPU HPC clusters via MPI
- **Flexible Neuron Models**: LIF, adaptive-threshold LIF, higher-order LIF, and Izhikevich somas; single-exponential and weighted synapses
- **STDP Support**: Built-in pair-wise, bounded, quantized and memristive spike-timing-dependent plasticity, plus user-registered learning rules
- **Train/Eval Switch**: `model.train()` / `model.eval()` and `set_learning_enabled()` toggle plasticity globally or per synapse
- **Named Parameters**: `get_hyperparameters()` / `set_hyperparameters()` (and their learning counterparts) address parameters by name rather than by position
- **Bulk and Distributed Construction**: build a whole network in one call with `create_from_lists()`, or have each MPI rank build only its own partition with `load_post_owned()` / `load_from_adjacency()` — no global graph is ever materialized
- **Network Generation**: Brunel balanced random networks via `brunel_partition()` with
  `topology="global" | "bounded" | "torus2d" | "torus3d"`, plus a spatially embedded
  economical small-world variant in `spatial_smallworld_partition()` (`superneuroabm/brunel.py`)

## Requirements

- Python 3.11+
- NVIDIA GPU with CUDA drivers **or** AMD GPU with ROCm
- MPI implementation (OpenMPI, MPICH, Cray MPICH, ...) for multi-GPU execution

Validated stack: ROCm 7.2.0 with CuPy 14.0.1 on AMD MI250X (see SAGESim's
`docs/frontier_setup_rocm720_cupy1401.md`).

## Installation

Your system might require specific steps to install `mpi4py` and/or `cupy` depending on your hardware. In that case, use your system's recommended instructions to install these dependencies first.

```bash
pip install superneuroabm
```

This pulls in `sagesim>=0.7.0`.

## Quick Start

```python
from superneuroabm.model import NeuromorphicModel

model = NeuromorphicModel()

# Two LIF somas
pre = model.create_soma(breed="lif_soma", config_name="config_0")
post = model.create_soma(breed="lif_soma", config_name="config_0")

# External drive into `pre` (pre_soma_id=-1 means "external input"), and pre -> post
drive = model.create_synapse(
    breed="single_exp_synapse", pre_soma_id=-1, post_soma_id=pre, config_name="config_0"
)
model.create_synapse(
    breed="single_exp_synapse", pre_soma_id=pre, post_soma_id=post, config_name="config_0"
)

# Compile step functions and allocate GPU buffers
model.setup()

for tick in (2, 20, 40):
    model.add_spike(synapse_id=drive, tick=tick, value=1)

model.simulate(ticks=200)

print("pre  spikes:", model.get_spike_times(soma_id=pre))
print("post spikes:", model.get_spike_times(soma_id=post))
```

Breed and config names come from `superneuroabm/component_base_config.yaml`; pass your own YAML
with `NeuromorphicModel(user_config=...)` to override or add parameter sets.

## Tutorials

| notebook | what it covers |
|---|---|
| [`tutorials/00_simple_heterogenous_network.ipynb`](tutorials/00_simple_heterogenous_network.ipynb) | building a heterogeneous network by hand, injecting spikes, reading spike times and internal state histories |
| [`tutorials/01_superneuroabm_digits.ipynb`](tutorials/01_superneuroabm_digits.ipynb) | a two-layer feedforward SNN on the sklearn 8x8 digits, trained with semi-supervised bounded STDP and evaluated with spike-count readout |

Tutorial 01 brings its own components — [`tutorials/user_customized_lif.py`](tutorials/user_customized_lif.py)
and [`tutorials/user_customized_stdp.py`](tutorials/user_customized_stdp.py) — registered on the
model at runtime, with no changes to the installed package. See
[`docs/CUSTOM_COMPONENTS.md`](docs/CUSTOM_COMPONENTS.md) for the full pattern.

Runnable experiment scripts — including the Brunel network and the Masquelier 2008 STDP
replication — live in the `ns-applications` repository rather than here.

## Unit Tests

```bash
python -m pytest tests/
```

That runs single-rank only. The multi-rank tests are in the same files and self-skip there, so a
green run says nothing about correctness above one rank. Sweep the rank counts with:

```bash
tests/run_mpi_tests.sh          # ranks 1 2 4
tests/run_mpi_tests.sh 2        # only 2 ranks
```

One GPU is enough — the ranks share it via `mpirun --oversubscribe`. On a cluster, the MPI
consistency test compares multi-rank spike times against the single-rank baseline with one GPU
per rank:

```bash
srun -A <account> -q debug -N1 -n2 -c7 --gpu-bind=closest \
     python -m pytest tests/test_mpi_comparison.py
```

## Performance

Measured on Frontier (OLCF), one MPI rank per MI250X GCD, using a Brunel network on a periodic 3D
lattice with a bounded connection radius:

- **Weak scaling** — at 12,500 neurons per GPU held constant, per-step parallel efficiency stays
  within **99–100 %** from 64 to 2048 GPUs (a 32x span) at in-degrees K = 1000, 2000 and 4000.
  The largest point is 25.6 M neurons on 2048 GPUs.
- **Strong scaling** — a fixed 204,800-neuron problem reaches **13.4x** speedup in wall time on
  64x the GPUs (16 to 2048).

Methodology, the complete measured record, and the caveats are in
[`scaling_analysis/paper_figures/README.md`](scaling_analysis/paper_figures/README.md); the design
discussion behind the wiring convention is in [`docs/BRUNEL_SCALING.md`](docs/BRUNEL_SCALING.md).
Reproduce with `scaling_analysis/weak_3d_chunk.sh` and `scaling_analysis/strong_3d_chunk.sh`.

## Documentation

| | |
|---|---|
| [`docs/FUNCTIONALITY_GUIDE.md`](docs/FUNCTIONALITY_GUIDE.md) | the API surface, end to end |
| [`docs/CUSTOM_COMPONENTS.md`](docs/CUSTOM_COMPONENTS.md) | bringing your own soma, synapse and learning rule |
| [`docs/DATA_FORMAT.md`](docs/DATA_FORMAT.md) | how agent properties are laid out |
| [`docs/DISTRIBUTED_SIMULATION.md`](docs/DISTRIBUTED_SIMULATION.md) | running across ranks |
| [`docs/PARTITION_LOADING.md`](docs/PARTITION_LOADING.md) | distributed construction from partition files |
| [`docs/SINGLE_GPU_NETWORK_CONSTRUCTION.md`](docs/SINGLE_GPU_NETWORK_CONSTRUCTION.md) | bulk construction on one GPU |
| [`docs/BRUNEL_SCALING.md`](docs/BRUNEL_SCALING.md) | Brunel network generation and the scaling study |
| [`docs/SPIKE_RECORDING_NOTES.md`](docs/SPIKE_RECORDING_NOTES.md) | what is recorded, and when |
| [`docs/CPU_GPU_DATA_FLOW.md`](docs/CPU_GPU_DATA_FLOW.md), [`docs/CPU_GPU_SYNC_DESIGN_NOTES.md`](docs/CPU_GPU_SYNC_DESIGN_NOTES.md) | host/device transfer and synchronization design |

## Publications

[Date, Prasanna, Chathika Gunaratne, Shruti R. Kulkarni, Robert Patton, Mark Coletti, and Thomas Potok. "SuperNeuro: A fast and scalable simulator for neuromorphic computing." In Proceedings of the 2023 International Conference on Neuromorphic Systems, pp. 1-4. 2023.](https://dl.acm.org/doi/abs/10.1145/3589737.3606000)

## License

BSD-3-Clause License - Oak Ridge National Laboratory
