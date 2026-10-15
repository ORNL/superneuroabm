# competitor_probes — what Arbor and GeNN can express

Probes behind `docs/WHY_SUPERNEUROABM_EVIDENCE.md` §3.4 (2026-10-13). Each tries one capability of
the device-per-synapse table on a tiny network. `probe_brian2.py` (one directory up) does the same for
Brian2.

| File | Simulator | What it tests |
|---|---|---|
| `arbor_probe.py` + `arbor_mod/` | Arbor 0.12.2 (pip, CPU, no MPI) | per-synapse state and parameters; every-step `g*sinh(v/v0)` current; three mechanism types on one single-CV cable cell; event-time randomness via the `WHITE_NOISE` workaround (`memdev_ev.mod`); same result across cell-group decompositions and thread counts |
| `arbor_noise_only.py` | Arbor | `WHITE_NOISE` read in `NET_RECEIVE` *without* a stochastic ODE (`memdev_noise.mod`): segfaults |
| `arbor_heat.py` + `arbor_mod_heat/` | Arbor | devices on the same compartment sharing state through a pseudo-ion ("heater" writes, "sensor" reads) |
| `genn_probe.py` | GeNN 5.5.0 (`single_threaded_cpu` backend) | per-synapse state/parameters, per-step synapse dynamics with `V_post`, mixed models on one neuron, `gennrand_*` in synapse code (fails: undefined identifier), devices on one presynaptic row coupled through a custom connectivity update |
| `genn_probe2.py` | GeNN | workaround: a custom connectivity update fills per-synapse random variables each step |
| `genn_probe3.py` | GeNN | synapse-to-synapse reads through a shared array indexed by pre/post id |

Run (Arbor; a venv with `pip install arbor`, whose `arbor-build-catalogue` needs that venv's python):

```bash
python $(which arbor-build-catalogue) memcat arbor_mod      # -> memcat-catalogue.so
python arbor_probe.py ./memcat-catalogue.so all              # or: hetero
python $(which arbor-build-catalogue) heat arbor_mod_heat && python arbor_heat.py ./heat-catalogue.so
```

GeNN 5.5.0 was built from source (`pip install .` in a clone); no CUDA was available, so every GeNN
result is CPU-backend only and GPU behaviour is from its source. Arbor GPU and MPI were not run.
