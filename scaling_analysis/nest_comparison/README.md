# SuperNeuroABM vs NEST GPU: setup-time comparison

First measured comparison between SuperNeuroABM and NEST GPU. Every NEST claim in
`docs/BRUNEL_SCALING.md` before this was architectural or cited from the literature;
no NEST of any kind had ever been run on this hardware.

This round measures **setup**, not steady-state ticks. Per-tick cost is already small
(1.16 s for 99 ticks at the point below); setup is 99.6 % of wall time, so that is what
is broken down.

## Hardware

One NVIDIA RTX 3000 Ada Generation Laptop GPU (sm_89), 8 GiB, WSL2, driver 572.83,
31 GiB host RAM. The published Frontier numbers in `docs/BRUNEL_SCALING.md` are from
2048 MI250X GCDs and are **not** comparable; both simulators are re-measured here.

## Network

The w=1 point of `scaling_analysis/weak_scaling.py`: 12,500 neurons (80 % excitatory),
exact in-degree K=1000, one external input per neuron at 10 Hz, 100 ticks of 1 ms,
static weights, seed 42. See the table in the plan file for the parameter mapping.

Three differences between the two simulators cannot be removed and are disclosed
rather than papered over:

| | SuperNeuroABM | NEST GPU |
|---|---|---|
| wiring | sources drawn in a radius-8 ball on a periodic 3D lattice | global `fixed_indegree` |
| refractory period | integrates during refractoriness | `iaf_psc_exp` does not |
| delay | 1.5 ms truncated to one 1 ms tick | 1.0 ms |

NEST GPU's connection cost is wiring-agnostic, so the first row favours neither side on
setup time. The other two are irrelevant at this drive level: the network is silent.

**The drive is sub-threshold by design.** `scaling_analysis/paper_figures/README.md`
documents that the published campaign runs at ~0 Hz, so this comparison inherits a silent
network. That is like-for-like for setup, and it removes NEST GPU's event-driven advantage
from the per-tick number, which is one more reason not to read anything into per-tick here.

## Scripts

| script | what it measures |
|---|---|
| `bench_common.py` | shared long-format CSV schema (one row per phase) and RSS/GPU probes |
| `brunel_sna_local.py` | SuperNeuroABM Brunel: generation, load, each `setup()` sub-step, each first-tick sub-step, per-property GPU tensor allocation |
| `brunel_nestgpu.py` | NEST GPU Brunel: create, connect, calibrate, warm-up, timed simulate |
| `masquelier_sna.py` | SuperNeuroABM Masquelier, one 10 s chunk (`--fused` for the single-launch tick path) |
| `masquelier_nestgpu.py` | NEST GPU Masquelier: `spike_generator` afferents, `iaf_psc_exp` soma, `stdp` synapse group |
| `masquelier_brian2.py` | Brian2 Masquelier using the Hathway-Goodman reference equations verbatim |

The three Masquelier scripts share the experiment's own spike-train generator
(`ns-applications/duplicates/masquelier_2008/experiment_utils.py`) and the same seed, so all
three deliver the identical 1,212,948 input spikes. Model equivalence is approximate: Brian2
uses the reference equations, SuperNeuroABM its Hathway-Goodman kernels, and NEST GPU a
stock `iaf_psc_exp` plus its built-in STDP. Output spike counts differ and are reported;
this is a wall-clock comparison at matched size and input, not a replication.

**Measure one simulator at a time.** A NEST GPU Masquelier run that overlapped a Brunel run
reported 21.5 s instead of 12.6 s.

Both write to `results/*.csv` with columns
`simulator, variant, neurons, in_degree, synapses, repeat, phase, seconds, host_rss_mb, gpu_mb`.

## Reproducing

```sh
# SuperNeuroABM
/home/xxz/miniforge3/envs/sna-dev/bin/python brunel_sna_local.py \
    --neurons 12500 --in-degree 1000 --ticks 100 --csv results/brunel_sna_after_fix.csv

# NEST GPU (build first, see below)
source ~/software/nest-gpu/install/bin/nestgpu_vars.sh
/home/xxz/miniforge3/envs/sna-dev/bin/python brunel_nestgpu.py \
    --neurons 12500 --in-degree 1000 --ticks 100 --csv results/brunel_nestgpu.csv
```

K=2000 does not fit: SuperNeuroABM already holds 3.56 GB of GPU pool and 16.6 GB of host
RSS at K=1000, so doubling the synapse count exceeds both the 8 GiB device and comfortable
host headroom.

### Building NEST GPU on this machine

```sh
mamba create -y -n nestgpu -c conda-forge python=3.12 "cuda-toolkit=12.9" \
    cmake make "gxx_linux-64=13" numpy scipy matplotlib mpi4py openmpi
git clone https://github.com/nest/nest-gpu ~/software/nest-gpu/src
export PATH=$HOME/miniforge3/envs/nestgpu/bin:$PATH
cmake -DCMAKE_INSTALL_PREFIX=$HOME/software/nest-gpu/install \
      -Dwith-gpu-arch=89 -Dwith-mpi=ON ~/software/nest-gpu/src
make -j6 && make install
```

`-Dwith-mpi=OFF` **does not build**: `src/connect.h:1053-1055` references `MPI_Comm` and
`mpi_comm_vect_` unconditionally, so the no-MPI configuration fails to compile. Build with
MPI on and run single-process.

`with-gpu-arch` is the GPU architecture flag (not `CMAKE_CUDA_ARCHITECTURES`); it defaults
to 80 and must be 89 for an Ada card.

## A caveat the Brunel numbers need

Brunel connectivity is *generated from a rule*, which NEST GPU can execute entirely on the
GPU without ever materialising an edge list. A network that comes from a file (Cora, a
connectome) cannot use that path. Measured separately, the rule path is nearly flat at
~0.05 s for 12.5 M edges while an explicit edge list is linear at ~0.18 us/edge, so 12.5 M
explicit edges cost 2.26 s. Quote 105x, not 5,400x, when the network is data-derived.

## Findings

See `docs/PERFORMANCE_NOTES.md` for the analysis and the ranked list of levers. In short:
Brunel construction is 4,200-5,400x slower than NEST GPU and 99.6 % of our wall time, with
over half of it spent building two history buffers that tracking had switched off; Masquelier
is a different story, 2.7x off NEST GPU with construction negligible, and CPU Brian2 beats
both GPU simulators at that size.
