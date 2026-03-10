# SuperNeuroABM

**SuperNeuroABM** is a GPU-based multi-agent simulation framework for neuromorphic computing. Built on top of [SAGESim](https://github.com/ORNL/SAGESim), it enables fast and scalable simulation of spiking neural networks on both NVIDIA and AMD GPUs.

## Key Features

- **GPU Acceleration**: Leverages CUDA (NVIDIA) or ROCm (AMD) for high-performance simulation
- **Scalable**: From single GPU to multi-GPU HPC clusters via MPI
- **Flexible Neuron Models**: Support for various soma and synapse step functions
- **STDP Support**: Built-in spike-timing-dependent plasticity mechanisms
- **Network I/O**: Import/export neural network topologies

## Requirements

- Python 3.11+
- NVIDIA GPU with CUDA drivers **or** AMD GPU with ROCm 5.7.1+
- MPI implementation (OpenMPI, MPICH, etc.) for multi-GPU execution

## Installation

Your system might require specific steps to install `mpi4py` and/or `cupy` depending on your hardware. In that case, use your system's recommended instructions to install these dependencies first.

```bash
pip install superneuroabm
```

## Quick Start

```python
from superneuroabm.model import SuperNeuroModel

# Create model
model = SuperNeuroModel()

# Create neurons
n1 = model.create_neuron()
n2 = model.create_neuron()

# Connect with synapse
model.create_synapse(n1, n2, weight=1.0)

# Setup and run
model.setup(use_gpu=True)
model.simulate(ticks=100)
```

## Unit Tests

To run unit tests:

```bash
python -m unittest tests.test_synapse_and_soma_models
```

## Publications

[Date, Prasanna, Chathika Gunaratne, Shruti R. Kulkarni, Robert Patton, Mark Coletti, and Thomas Potok. "SuperNeuro: A fast and scalable simulator for neuromorphic computing." In Proceedings of the 2023 International Conference on Neuromorphic Systems, pp. 1-4. 2023.](https://dl.acm.org/doi/abs/10.1145/3589737.3606000)

## License

BSD-3-Clause License - Oak Ridge National Laboratory
