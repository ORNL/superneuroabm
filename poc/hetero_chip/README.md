# hetero_chip — hardware-aware heterogeneous chip experiments

The two experiments behind `docs/WHY_SUPERNEUROABM.md` (short summary: who can do what, and why).
Source excerpts, probe results and full result tables: `docs/WHY_SUPERNEUROABM_EVIDENCE.md`.

| File | What |
|---|---|
| `chip_devices.py`, `chip_config.yaml` | Experiment A (programmed inference chip) kernels: Poisson encoder, digital and analog LIF circuits, SRAM / RRAM (1T1R, passive, selector) / PCM devices |
| `run_chip_poc.py` | Experiment A: 9 hardware designs of a digit-classifier chip in one run; `--static-only` ablation; `--v-swing` sweep |
| `compare_ablation.py` | full device model vs static-only, paired per design |
| `learning_devices.py`, `learning_config.yaml` | Experiment B (chip that learns by itself) kernels: relays, LIF with activity trace, memristive device that is read and programmed by its spikes |
| `run_learning_poc.py` | Experiment B: on-chip learning, 7 device designs x N chip instances; `--plot` saves learned conductances |
| `coupling_devices.py`, `coupling_config.yaml`, `run_coupling_probe.py` | row 9: a device reads a device of another neuron on another rank (heater/sensor pair); bitwise on 1, 2, 4 ranks |
| `probe_brian2.py` | what Brian2 2.9 can and cannot express (run in an env with Brian2) |
| `competitor_probes/` | what Arbor 0.12 and GeNN 5.5 can and cannot express (own README) |
| `results/` | JSON per run (accuracies, per-digit predictions, spike digest), learned-conductance figure |

Run from this directory (the framework writes `step_func_code.py` to the working directory and
imports the kernels by module name):

```bash
PY=/home/xxz/miniforge3/envs/sna-dev/bin/python
$PY run_chip_poc.py                                   # ~15 s, 2.1 M devices
mpirun --oversubscribe -n 2 $PY run_chip_poc.py       # same spike digest
$PY run_chip_poc.py --static-only && $PY compare_ablation.py
$PY run_learning_poc.py --plot                        # ~20 s
mpirun --oversubscribe -n 2 $PY run_learning_poc.py --instances 4
$PY run_coupling_probe.py                              # ~10 s; same digest on -n 2, -n 4
```

Multi-rank runs need `simulate(..., update_data_ticks=1)` (the MPI exchange interval); both drivers
do this. Device parameters are illustrative orders of magnitude, not fitted to one technology.
