# Why SuperNeuroABM — detailed evidence

The full evidence behind the short summary in `WHY_SUPERNEUROABM.md`: source excerpts, probe results,
experiment setups and result tables, open items. Draft, 2026-10-01. Every claim is tagged with how it
was established: **[source]** read in the other simulator's source code, **[probe]** tested with a
script in this repo, **[run]** measured with SuperNeuroABM in this repo, **[docs]** taken from the
other project's documentation and *not* tested here. Code and raw results: `poc/hetero_chip/`.

---

## 1. Thesis

Designing neuromorphic hardware needs a simulator in which **the device is the unit of simulation**:
every synapse (and every neuron circuit) carries its own state, runs its own technology's model while
spikes cross it, can see the neuron it is wired to, draws its own randomness, and can differ from its
neighbour. And it must do this at the size of a chip or a system of chips, on many GPUs, without the
answer depending on how the network was split across them.

SuperNeuroABM is built that way because it is an agent-based simulator: a synapse is an agent with a
state vector and a kernel, chosen per agent. The argument rests on two experiments, both a small
digit classifier mapped onto simulated hardware (64 inputs -> 640 synaptic devices -> 10 neurons per
core, many cores per run):

- **Experiment A — a programmed inference chip** (section 4). The weights come from a classifier
  trained offline and are written into simulated devices. Nine hardware designs (SRAM, RRAM in three
  array styles, PCM with and without drift compensation, two SRAM/RRAM mixes) run side by side on
  the same input spikes. Question: how much accuracy does each technology lose?
- **Experiment B — a chip that learns by itself** (section 5). The devices start blank and the chip
  learns the digits during the run: every input spike sends a programming pulse, and what the pulse
  does to the conductance is the device model. Seven device behaviours, 16 chip instances each.
  Question: which device non-idealities break on-chip learning?

The capabilities, and where each experiment demonstrates them:

| | Capability | Why hardware-aware design needs it | Shown in Experiment A | Shown in Experiment B |
|---|---|---|---|---|
| C1 | Per-device state and per-device code, run when spikes cross the device and/or every tick | programming, drift, read noise, wear, learning are things a device *does* | each cell computes its own current from its own programming error, fault and read noise (`chip_rram_synapse`, `chip_pcm_synapse`); its conductance is recomputed, not changed | each device holds a conductance that its own programming pulses change during the run (`learn_device`) |
| C2 | Access to the connected neuron's state from the device | crossbar columns sit at the membrane voltage; learning pulses depend on post activity | RRAM cells read the neuron's membrane voltage every tick: read current saturates, passive cells leak (`chip_devices.py:277`) | each device reads its neuron's activity trace to choose SET or RESET (`learning_devices.py:175`) |
| C3 | Device-granular heterogeneity: any synapse can run any model, mixed onto the same neuron | hybrid digital/analog chips; technology mapping is a per-synapse decision | `hetero_*` designs put SRAM and RRAM cells in the same column, chosen synapse by synapse (`run_chip_poc.py:121`); 3 neuron and 3 synapse kernels in one network | weak: fixed digital teacher wires and learning devices feed the same neuron; the seven device behaviours are one kernel with seven parameter sets |
| C4 | Randomness keyed on the device, not on the GPU | a seed is a chip instance; the same chip must come out of 1 GPU and 64 GPUs | frozen per-device draws (programming error, faults, neuron mismatch) and per-read noise; identical spikes on 1, 2 and 4 ranks | per-device step sizes, write noise, stochastic switching; identical spikes on 1, 2 and 4 ranks; instances differ only in device ids |
| C5 | Multi-GPU / multi-node with C1-C4 intact | device-level models of real chips do not fit one GPU | **only the code path**: 1, 2 and 4 MPI ranks, all sharing one laptop GPU | same |
| C6 | Models written in Python, no simulator rebuild | design-space exploration means many device models | 6 kernels in `chip_devices.py`, registered at run time | 4 kernels in `learning_devices.py`, registered at run time |

Which simulator fails which experiment, in short (details in sections 3-5):

| | NEST GPU | NEST (CPU) | Brian2 |
|---|---|---|---|
| Experiment A | 8 of 9 designs reproducible by folding device constants into weights; the selector crossbar is not (needs C1+C2 every tick) [run, source] | same; the selector crossbar cannot be a NEST synapse [docs] | can, on one GPU, with one synapse group and one neuron variable per technology [probe] |
| Experiment B | **cannot** run 5 of the 7 devices without rewriting NEST GPU: no per-synapse parameters and no random numbers in its plasticity hook (C1, C4) [source] | most likely can, as an event-driven NESTML synapse, on CPUs; loses GPU and partition-invariant chip instances [docs] | can, on one GPU [probe] |

So the clean "cannot" today is Experiment B against NEST GPU. What is **not** yet shown is a run that
is impossible in Brian2: both experiments fit one GPU, and Brian2's limit is C5 (one node or one GPU),
which neither experiment has exercised beyond one GPU. Section 7 says what experiment closes that gap.

---

## 2. The problem: static versus dynamic device effects

Hardware non-idealities fall into two classes, and the distinction decides which simulators can model
them:

- **Static** — a constant per device once the chip is programmed: weight quantization, programming
  error, stuck-at faults, conductance drift evaluated at a fixed deployment age, drift compensation,
  neuron-circuit mismatch. These can be *folded into the weights* (and per-neuron parameters) before
  the simulation starts. Any simulator with per-connection weights can then run them, including
  NEST GPU, at full speed.
- **Dynamic** — happen during the run and depend on activity or time: per-read noise; read current
  that depends on the membrane voltage on the column; leakage through unselected cells when the
  cell's I-V is nonlinear (a *linear* leak sums to a constant per-neuron leak and is static —
  section 4.1 confirms it); drift, retention loss or read disturb over a long run; endurance wear; and, above all, **on-chip
  learning**, where the weight *is* the device state and is produced by programming pulses whose effect
  is device physics (nonlinear, noisy, device-dependent, quantized). These need code that runs per
  device, per event or per tick, with that device's own state and randomness.

Hardware-aware *DNN* tools (IBM AIHWKit, CrossSim, MemTorch, NeuroSim) model a crossbar as a noisy
matrix-vector multiply. They have no spiking dynamics or time, so they are a different class of tool,
not a competitor here. The spiking simulators compared here do not use weight matrices either; NEST
GPU, for example, delivers each spike through per-connection records (section 3.1).

---

## 3. What a synapse *is* in each simulator

### 3.1 NEST GPU 2.0.0 — a synapse is a 12-byte record and one `atomicAdd` [source]

Source read: `~/software/nest-gpu/src/src/` (version file: 2.0.0).

- **Storage.** `conn12b.h:29`: `struct conn12b_struct { uint target_port_syn; float weight; };`
  (target, receptor port and synapse group packed into one int), plus a 32-bit key `conn12b_key`
  holding source and delay (`conn12b.h:35`) — 12 bytes per connection; `conn16b` uses a 64-bit key
  and a full 32-bit target, 16 bytes. One `float` is the whole per-synapse state.
- **Spike delivery.** `input_spike_buffer.h:329`, inside `deliverSpikesKernel`: for every emitted
  spike, loop over the source's outgoing connections and do
  ```cpp
  atomicAdd(&input_spike_buffer_[i_target][i_port][i_slot], weight * mul);
  ```
  Nothing else runs on a synapse when a spike crosses it, and there is no per-step loop over synapses.
- **Plasticity hook.** `syn_model.h:46`: `SynapseUpdate(int syn_group, float* w, float Dt)`, a
  compiled `switch` over two cases (`test_syn_model`, `stdp`). It is called only at spike times
  (`get_spike.h:95` for pre, `rev_spike.h:92` for post). It receives the weight, the pre/post spike
  time difference, and `SynGroupParamMap[syn_group - 1]` — parameters **shared by the whole group**.
  It does not receive per-synapse parameters, neuron state, the clock, or a random-number state.
- **Model registry.** `syn_model.h:61`: `syn_model_name = { "", "test_syn_model", "stdp" }`;
  `syn_model.cu:129-143` (`CreateSynGroup`) is an `if/else` over those names that throws
  `Unknown synapse model name` for anything else. NEST GPU's documentation has a guide for new
  *neuron* models (copy a `user_m1_*` template over `user_m1.*`, edit, rebuild; or NESTML) and none
  for synapse models (`doc/htmldoc/guides/implement_new_neuron_models.rst`).

**Time-driven neurons, event-driven synapses.** Like NEST, NEST GPU is hybrid: `NESTGPU::SimulationStep`
(`nestgpu.cu:751`) calls every neuron group's `Update` on every step (`nestgpu.cu:788`), while a
synapse is touched only when a spike crosses it (delivery, and the plasticity hook at pre/post spike
times). Being event-driven is therefore *not* what separates NEST GPU from NEST CPU for Experiment B — both
deliver at the same moments. The difference is what code may run at that moment and what state a
synapse may hold: arbitrary NESTML/C++ with per-synapse state in NEST, a fixed `atomicAdd` plus a
`(w, Dt, group params)` hook in NEST GPU. Neither can run device code between spikes (Experiment A's
selector crossbar).

This is a deliberate trade, and a good one for its purpose: 12 bytes per synapse (16 in the wide layout) and work
proportional to spikes. It is why NEST GPU is fast. (SuperNeuroABM's synapses have been event-woken too
since 2026-10-15 — stepped only when their pre fires, with the synaptic kinetics on the soma — and the
per-tick gap on our Brunel benchmark went from ~40x to ~1.4x, 0.28 vs 0.20 ms per tick on one GPU
with all ticks in one launch (NEST GPU's own printed time for that run is 0.52 ms per step; an earlier
~7x figure, 1.28 ms, was measured with one launch per tick, which the benchmark's timers force);
what remains is the cost of a synapse being an agent with its own row and learning rule rather than
a 12-byte record, `docs/EVENT_DRIVEN_NOTES.md` sections 7 and 15.) It is also why it cannot
be hardware-aware beyond static effects: there is nowhere to keep device state and no place where
device code runs. Adding it means widening the connection struct (used by connection building,
delivery, remote/MPI connection code), adding a per-step synapse kernel and per-synapse RNG state, and
rebuilding — a different simulator, not a new model.

### 3.1a NEST GPU's published scaling [docs]

- **Tiddia et al., Front. Neuroinform. 2022** — multi-area macaque model (4.13 M neurons, 24.2 G
  synapses) on 32 V100 GPUs: real-time factor ~15-16 in the metastable state, 3.1x faster than NEST on
  32 CPU nodes. A fixed-size model, not a weak-scaling study.
- **Golosio, Tiddia, Villamar, ..., Senk, "Scalable construction of spiking neural networks using up
  to thousands of GPUs"** (arXiv 2512.09502; Neuromorph. Comput. Eng. 6, 024012, 2026):
  construction moved onto the GPU (multi-area model 686 s -> 55.5 s on 32 GPUs); weak scaling of a
  balanced random network at 2.25e5 neurons and 2.53e9 synapses per GPU over 128 -> 1,024 A100 GPUs
  on Leonardo Booster (2.59e12 synapses at 1,024 GPUs; dt 0.1 ms; ~8 spikes/s). **The scaling is not
  flat:** construction time grows linearly with the number of GPUs (Fig. 4a, below 100 s), and
  simulation time per model second, T_wall/T_model, grows sublinearly from 128 to 1,024 GPUs
  (Fig. 4b). The authors name spike communication between nodes as the main bottleneck — in a random
  network every GPU exchanges spikes with every other — and do not claim ideal efficiency. The full machine
  (3,456 nodes) is an extrapolation: 3.5e13 connections, ~50 s construction + 42 s preparation.
- **Unverified:** a Jülich record, "NEST GPU simulations scale up to networks of billions of spiking
  neurons and trillions of synapses", reportedly ran 12,000 GPUs on Leonardo (3.37 G neurons,
  37.96 T synapses, sub-second construction). The record could not be opened; numbers are from
  search snippets only, and no simulation speed or efficiency was seen.

**Reading the metric.** T_wall/T_model is the real-time factor (RTF): wall-clock time of the
simulation run (construction excluded) divided by the biological time simulated. RTF = 1 is real time,
< 1 faster, 10 is ten times slower; dividing by T_model makes runs of different length comparable. It
is the same quantity as our per-step time in other units: RTF = (wall time per step) / dt. Our
weak-scaling runs use dt = 1 ms (`weak_scaling.py --dt-ms`, default 1.0), so a 6.4 ms step is RTF
6.4; NEST GPU's runs use dt = 0.1 ms, where RTF 1 means 0.1 ms per step. Because RTF scales with
1/dt, comparisons need the same dt — a further reason the two curves below are not comparable as is.

**Against SuperNeuroABM's own weak scaling** (`docs/BRUNEL_SCALING.md` §5.5a, Frontier, 2026-07-30):
step time flat from 64 to 2,048 GPUs (within 1-4 %), at 65-83 % of the 1-GPU step (the loss is in the
1 -> 64 ramp as the halo grows to 26 neighbours); largest run 25.6 M neurons and 102 G synapses. Not
the same workload: ~50-200x fewer synapses per GPU (12.5-50 M vs 2.5 G), spatially local
connectivity with at most 26 peer GPUs (theirs: random, every GPU talks to all), and a silent network
(our exchange is a fixed 20 B per ghost neuron per step). Our step time is flat past 64 GPUs; theirs
grows sublinearly — but on a far larger and harder workload, so this is *not* evidence that
SuperNeuroABM scales better. A fair comparison needs the same network on both simulators at several
GPU counts (our local-connectivity network in NEST GPU, or a random network in SuperNeuroABM). Until
then our scale claim is only "custom per-device models with flat weak scaling over many GPUs in a
local-connectivity regime", which NEST GPU cannot run at any scale.

Sources: [Tiddia et al. 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9289599/);
[arXiv 2512.09502](https://arxiv.org/abs/2512.09502) /
[NCE article](https://iopscience.iop.org/article/10.1088/2634-4386/ae65d2);
[Jülich record (unverified)](https://juser.fz-juelich.de/record/1044379).

### 3.2 NEST 3 (CPU) [docs, not tested here]

Synapses are objects with their own state and can be user-defined in NESTML (code-generated C++,
compiled into a module). They are **event-driven**: a synapse is updated only when a spike is
delivered through it. Anything that must evolve between spikes as a function of a continuously
changing signal — e.g. a passive-crossbar cell leaking the membrane through its own nonlinear I-V on
every step — cannot be a NEST synapse; it would have to be folded into a custom neuron model that
carries every attached device's parameters. CPU only (MPI + threads), but it scales to the largest CPU
supercomputers. Random streams are per virtual process, so results are reproducible for a fixed number
of virtual processes.

Experiment B's learning devices change only when a presynaptic spike arrives and need the post-neuron's recent
activity, which a NEST synapse can reconstruct from the post-neuron's spike history. So a custom NESTML
synapse with per-synapse parameters and random draws should express Experiment B on CPUs (to be confirmed by
building it). What it would not give: GPU execution; the same chip instance at a different number of
virtual processes (per-pulse noise follows the process's random stream, unless a counter-based
generator keyed on connection identity is hand-written in C++); and a new model without a
code-generate-and-compile step.

**Does NEST on a CPU supercomputer cover SuperNeuroABM?** Not entirely, but the gap is narrower than
for NEST GPU, and it should be stated precisely:

- *Expressiveness.* NEST covers static device effects, device effects at spike times, and (most
  likely) Experiment B's on-chip device learning, at very large scale. It does not cover device code
  that runs **between** spikes coupled to the neuron's state (Experiment A's selector crossbar;
  likewise volatile memristors driven by the membrane voltage, or sneak currents), interactions
  beyond pre/post (synapse-level pooling, synapses as input ports), or the same chip instance at a
  different number of processes without a hand-written counter-based generator.
- *Speed.* Not measured against NEST on CPU on these workloads, so no claim there. Against the
  others, measured (`docs/EVENT_DRIVEN_NOTES.md`, 2026-10-15): on the Masquelier 2008 experiment
  (2,000 plastic afferents, one neuron, 100,000 steps) SuperNeuroABM takes 1.07 s, Brian2 on CPU
  5.2 s and NEST GPU 11.0 s; on Brunel (12.5 M synapses) it is ~7x slower per step than NEST GPU.
  Both simulators do work proportional to spikes now; the earlier "steps every synapse every tick"
  was true and has been removed.
- *Where it runs.* Current leadership HPC systems put most of their compute in GPUs, which NEST on
  CPU leaves idle. That is an argument about cost and access, not capability.

Against NEST on CPU, the defensible "cannot" is therefore device dynamics between spikes coupled to
the neuron, and interactions beyond pre/post — today carried by one design (the selector crossbar).

### 3.3 Brian2 2.9.0 [probe: `poc/hetero_chip/probe_brian2.py`; multi-GPU: docs]

| Probe | Result |
|---|---|
| Per-step (`clock-driven`) synapse ODE that reads `v_post` | **works** |
| Two `Synapses` objects (two device technologies) summing into the same neuron variable | **`NotImplementedError`: "Multiple 'summed variables' target the variable 'Isyn' … Use multiple variables in the target group instead"** |
| `if/else` in synaptic model code | **error** |
| `for` loop in synaptic model code | **error** |

Brian2's equation language can express per-device dynamics, including neuron coupling and per-step
noise, on one device. Heterogeneity costs one `Synapses` object per technology *and* one input variable
per technology in the neuron equations (so the neuron model is rewritten whenever a technology is
added); branches must be rewritten as arithmetic (`int(cond)`, `clip`), and loops are unavailable
without hand-written target-language functions.

**HPC options [docs; source].** Brian2's own parallelism is OpenMP multithreading in C++ standalone
mode (`prefs.devices.cpp_standalone.openmp_threads`), i.e. one shared-memory node; its documentation
says the OpenMP code "has not yet been well tested". Brian2's documentation describes no MPI or
multi-node mode, and Brian2 2.9's `cpp_standalone` device source contains no MPI code (checked:
OpenMP referenced 20 times, MPI 0). Brian2CUDA generates code for a single GPU; all benchmarks in its
paper (Alevi et al., Front. Neuroinform. 2022) run on one GPU and the paper describes no multi-GPU or
MPI mode. Brian2GeNN runs through GeNN, also a single-GPU code generator (not checked here). So
Brian2's ceiling is one node or one GPU — the hard limit (C5).

### 3.4 GeNN 5.5 and Arbor 0.12 [probe: `poc/hetero_chip/competitor_probes/`; source; docs]

Checked 2026-10-13. Arbor: probes on the 0.12.2 pip wheel (CPU, no MPI), source at main `a2af7ac`.
GeNN: built 5.5.0 from source, probes on its CPU backend only (no CUDA compiler here); GPU behaviour
is from its source. **Arbor covers most of the table; GeNN covers it on one GPU.** This narrows the
argument (see "What survives" below).

**Arbor** — a synapse is an NMODL point mechanism on a cable cell.
- *New device model without rebuild* ✅ [probe]: NMODL -> `arbor-build-catalogue` -> a `.so` loaded
  at run time (~5 s); Arbor itself is not rebuilt.
- *Per-synapse state, per-synapse parameters, every-step current from the membrane* ✅ [probe]:
  `g*sinh(v/v0)` injected every step, with `g` changed at events and per-instance step sizes.
- *Different device models on one neuron* ✅ [probe]: three mechanism types on one cell.
- *Randomness at spike time* ⚠️ [probe, source]: NMODL's only randomness is `WHITE_NOISE` for a
  `DERIVATIVE` block solved with `METHOD stochastic` (`modcc/module.cpp:455-460`); there is no
  uniform or random function (`modcc/token.cpp:70-87`). Reading `W` in `NET_RECEIVE` alone compiles
  but allocates no random buffer and segfaults (`arbor_noise_only.py`). Workaround that works
  (`memdev_ev.mod`): also use `W` in a dummy stochastic ODE; `NET_RECEIVE` then reads a per-synapse
  N(0,1) draw, enough for stochastic switching and write noise. Undocumented, Gaussian only, and
  the draw is the previous step's (events are delivered before the random numbers are refreshed,
  `fvm_lowered_cell_impl.hpp:185-219`).
- *Same chip on any decomposition* ⚠️ [docs, source, probe]: documented — "the same random numbers
  are generated regardless of concurrency, domain decomposition, and computational backend"
  (`doc/dev/sde.rst:102-103`); the stream is keyed on seed, mechanism id, variable, step, cell id and
  per-cell index (`multicore/rand.cpp:19-20`). But the mechanism id is the position in each cell
  group's sorted mechanism list (`shared_state.cpp:378`), so a cell grouped with different mechanism
  types gets different noise (probe `hetero`). With the same types everywhere, device states were
  bitwise identical over five decompositions, membrane voltage differed by <= 1.8e-15
  (`arbor_probe.py all`; summation order, not pinned down). On GPU, per-compartment current sums end
  in an `atomicAdd` (`include/arbor/gpu/reduce_by_key.hpp:78-81`), so repeat GPU runs may differ.
- *Many GPUs* ✅ [docs, not run]: MPI with one GPU per rank.
- *Point neurons* ⚠️ [source, probe]: `lif_cell` has fixed parameters and adds each event's weight to
  V_m (`lif_cell.hpp:15-26`), so it cannot host mechanisms; a single-compartment cable cell works but
  runs the cable solver every step (cost not measured).
- *A device reading another device or a third neuron* ⚠️/❌ [probe, source, docs]: devices on the
  **same compartment** can share state through an ion species (`arbor_heat.py`). Across neurons the
  only channels are spikes (`NET_RECEIVE` takes one weight, `module.cpp:566-567`) and gap junctions,
  which see only `v` and `v_peer` (`doc/fileformat/nmodl.rst:229-232`) and whose cells must sit in
  the same cell group, i.e. the same rank (`doc/cpp/domdec.rst:93-95`). NMODL has no arrays or loops
  (`nmodl.rst:142`); hand-written `--raw` C++ mechanisms were not tested.

**GeNN** — synapse code strings, compiled at `model.build()`.
- *New device model, per-synapse state and parameters, every-step synapse dynamics reading `V_post`
  and adding to the post current, mixed models on one neuron* ✅ [probe].
- *Randomness in synapse code* ⚠️ [probe, source]: `gennrand_*` in spike-time and per-step synapse
  code fails ("Undefined identifier"); a maintainer, in GeNN issue #497 (open): "GeNN doesn't currently
  support random number generation in any type of per-synapse code". Workaround that works
  (`genn_probe2.py`): a custom connectivity update has a per-row generator
  (`backendSIMT.cc:1294-1300`) and can refill per-synapse random variables every step.
  The same issue says GPU spikes are processed "in unpredictable order", so GPU runs do not repeat
  exactly.
- *A device reading another device* ⚠️ [probe, CPU]: a custom connectivity update can walk one
  presynaptic row (same synapse group, SPARSE only, `customConnectivityUpdate.cc:138,153`); a
  writable shared array indexed by pre/post id gives arbitrary device-to-device reads
  (`genn_probe3.py`; on GPU races are likely unless double-buffered, untested).
- *Many GPUs* ❌ [source]: no MPI or model partitioning in GeNN 5; its NCCL code only reduces across
  batch copies of one model (`backendCUDAHIP.cc:805-822`); multi-GPU is a wish-list item (issue #210).

**What survives.** Against GeNN: one hard "cannot", one network across several GPUs; spike-time
randomness is "not native", with a working workaround. Against Arbor: rows 1-5, 7 and 8 hold in
Arbor too (GPU/MPI from docs). What remains is softer — spike-time randomness only through an
undocumented Gaussian workaround, point neurons only as cable cells, partition invariance documented
but not bitwise in the probe — plus one hard gap: **no coupling between devices on different neurons**.
The one capability found in neither is therefore *a device that reads other devices (or any agent
other than its two neurons) in a network spread over many GPUs*: GeNN can couple devices but only on
one GPU; Arbor scales but cannot couple devices across cells. SuperNeuroABM does it [run]:
`load_from_adjacency()` takes neighbour lists verbatim, so a synapse's slots may name another synapse
on another rank; SAGESim discovers ghosts from any neighbour list and synapse `internal_states` are
neighbour-visible (`superneuroabm/model.py:224`). Probe `poc/hetero_chip/run_coupling_probe.py`: 16
neurons, each with a heater device and a sensor device; sensor j reads heater j+1 (another neuron,
another rank whenever ranks > 1); the heat is random per pulse and double-buffered by tick parity, so
the read is race-free on one GPU and equal to the ghost copy on several. Digest `7d387425…`, 67 spikes
and identical sensor conductances on 1, 2 and 4 ranks. Limit: a synapse has at most two neighbour
slots (`model.py:2423`), so a device needing its input, its neuron and several neighbours must be a
soma-kind agent or use a helper agent (not tried).

### 3.5 SuperNeuroABM — a synapse is an agent [run]

- Its own state vector (`internal_states`, `learning_internal_states`) and its own parameters.
- Its own kernel, chosen **per agent** by breed; the fused kernel dispatches on each agent's breed, so
  SRAM cells, RRAM cells and PCM pairs can sit in the same column and feed the same neuron.
- It runs every tick (with an exact idle fast path), reads its pre-neuron's spike and — because the
  partition rule is *post-owns* (every synapse lives on its post-neuron's rank) — its post-neuron's
  state, on any number of GPUs.
- `rand_normal(tick, agent_index, salt)` is rewritten at code generation to a Philox draw keyed on
  (seed, tick, **global agent id**, salt). A draw at tick 0 is a frozen per-device constant (programming
  error, stuck fault, mismatch) that is recomputed rather than stored and is the same number on every
  rank layout; a draw at `tick` is per-event noise with the same property.
- Kernels are ~30-line Python functions compiled with `cupyx.jit`, registered at runtime
  (`docs/CUSTOM_COMPONENTS.md`). No rebuild.

The price: 147 bytes per agent on the GPU in Experiment A (CuPy memory-pool bytes in use after the run)
[run], about 10x NEST GPU's connection record, and per-tick work proportional to the number of
synapses rather than spikes.

### 3.6 Is it the agent model that matters? (What if synapses were folded into the soma?)

NEST and NEST GPU store a neuron's incoming synapses with the neuron (or its thread/GPU) rather than as
independent objects. Would SuperNeuroABM lose these capabilities if it did the same?

Not in principle. What Experiment B needs is per-device state, per-device code, randomness keyed on the
device, the presynaptic spike and the post-neuron's state. A folded soma that loops over its incoming
devices with those ingredients can do it, and post-neuron state even becomes local for free. What a
folded layout gives up, or makes the user rebuild:

1. **Per-synapse model choice.** One soma kernel serves all its inputs, so SRAM, RRAM and PCM cells on
   one neuron become a type switch inside the soma kernel — a hand-written union of every device model
   — instead of a breed per agent.
2. **Ragged fan-in.** SAGESim pads property vectors to the widest agent; a soma carrying
   fan-in x device-state values pads every soma to the largest fan-in in the network.
3. **GPU parallelism.** One thread per soma walks its synapses serially; with thousands of inputs per
   neuron that is poor occupancy and load imbalance, against one thread per synapse.
4. **Inputs that are not pre/post.** Synapse-level pooling and synapses used as input ports (needed by
   the S-GATs work) are lost.

Folding is also not what limits the others: NEST CPU's synapses are separate objects, yet cannot run
per-tick code; NEST GPU's limit is its 12-byte record and fixed hook. The property that matters is
*user code and user state per synapse, device-keyed randomness, on GPUs*. Agents are how
SuperNeuroABM provides it without a union model or padding. (Project decision 2026-09-10: synapses stay
agents; a folded kind for classic point-neuron benchmarks is at most a later, optional addition.)

### 3.7 Summary matrix

| | NEST GPU 2.0 | NEST 3 | Brian2 2.9 | GeNN 5.5 | Arbor 0.12 | SuperNeuroABM |
|---|---|---|---|---|---|---|
| C1 per-device state + code | no: one float, fixed `atomicAdd` [source] | yes, at spike times (NESTML) [docs] | yes (equations) [probe] | yes [probe] | yes (NMODL) [probe] | yes [run] |
| C1 per-tick, neuron-coupled device | no [source] | no: event-driven [docs] | yes [probe] | yes [probe] | yes [probe] | yes [run] |
| C3 technologies mixed onto one neuron | static weights only | yes (per-connection model) [docs] | needs one neuron variable per technology [probe] | yes [probe] | yes [probe] | yes [run] |
| Branches / loops in device code | n/a | yes (NESTML) [docs] | no [probe] | C-like code (not checked) | branches yes, no loops or arrays [docs] | yes [run] |
| C4 randomness in device code, partition-invariant | no RNG in the hook [source] | per virtual process [docs] | single process | not in synapse code; workaround [probe]; one GPU | Gaussian only, spike time via workaround; invariant by design, not bitwise in probe [probe, docs] | yes, bitwise [run] |
| Device reads another device / a third neuron | no [source] | no [docs] | not between synapses (`linked_var` refuses synapse variables) [probe]; only by making every device a neuron, one GPU (untested) | yes, workaround, one GPU [probe] | same compartment only [probe, source] | yes, across ranks, bitwise [run] (probe; two neighbour slots per synapse) |
| C5 multi-GPU | yes | no GPU | no: OpenMP on one node, or one GPU (Brian2CUDA) [docs; source] | no: one GPU [source] | yes, MPI [docs] | yes: MPI, run here on one GPU [run]; multi-node scaling in `docs/BRUNEL_SCALING.md` |
| C6 no rebuild for a new device | no: C++ registry [source] | NESTML compile [docs] | yes | yes (code generated at build) [probe] | catalogue compile, no Arbor rebuild [probe] | yes |
| Point neurons | yes | yes | yes | yes | only as single-compartment cable cells [source] | yes |

---

## 4. Experiment A — a programmed inference chip (`run_chip_poc.py`)

**Demonstrates:** C1 and C2 (each cell's current from its own constants, read noise and the
neuron's membrane voltage, every tick), C3 (SRAM and RRAM mixed in one column), C4 (identical on 1, 2
and 4 ranks), C6. **Against other simulators:** mostly reproducible in NEST GPU by folding device
constants into weights (section 4.1); the exception is the selector crossbar.

**Setup.** A logistic-regression digit classifier (sklearn 8x8 digits; float accuracy 95.8 % on 360
test digits) is deployed as cores of 64 on-chip Poisson encoders -> 640 synaptic devices -> 10 LIF
neurons. Every test digit is presented, on identical input spike trains, to one core of each of nine
hardware designs, all in one simulation: 55 440 somas and 2 073 600 synaptic devices, 150 ticks
(dt = 1 ms). Device kernels: `chip_devices.py` (6 kernels, under 400 lines with comments);
parameters: `chip_config.yaml` (illustrative orders of magnitude, not fitted to one device).

| Design | What it is | Accuracy [95 % CI] |
|---|---|---|
| ideal | float SRAM, float digital neurons | 95.0 % [92.2, 96.8] |
| sram_4bit | 4-bit SRAM, 8-fractional-bit fixed-point neurons | 94.7 % [91.9, 96.6] |
| rram_1t1r | RRAM 1T1R: programming error, 1 %+1 % stuck cells, read noise, read current vs membrane; analog neurons with mismatch and noise | 80.0 % [75.6, 83.8] |
| rram_passive | same cells in a selector-less crossbar: every cell leaks the membrane every tick | 66.9 % [61.9, 71.6] |
| rram_passive_sel | passive crossbar with a sinh selector (per-device nonlinearity) | 72.2 % [67.4, 76.6] |
| pcm_1month | differential PCM, power-law drift with per-device exponent, one month after programming | 85.8 % [81.9, 89.1] |
| pcm_1month_gdc | the same with global drift compensation | 92.2 % [89.0, 94.6] |
| hetero_aware | the 20 % most important cells on 4-bit SRAM, the rest RRAM 1T1R, mixed in every column | 85.0 % [80.9, 88.3] |
| hetero_random | the same 20 % SRAM budget on random cells | 82.5 % [78.2, 86.1] |

Paired McNemar tests (same digits, same input spikes): drift compensation p = 0.001 (it reproduces
the effect reported by Joshi et al., Nat. Commun. 2020); hetero_aware vs hetero_random p = 0.31 and
vs all-RRAM p = 0.06 — not significant at this size.

**Membrane-to-read-voltage ratio** (how far the column voltage swings relative to the read voltage)
decides how much the passive-array effects matter:

| ratio | rram_1t1r | rram_passive | rram_passive_sel |
|---|---|---|---|
| 0.002 | 80.6 % | 80.3 % | 78.6 % |
| 0.01 (default) | 80.0 % | 66.9 % | 72.2 % |
| 0.05 | 78.6 % | 20.0 % | 14.4 % |

At 0.05 the selector makes things worse: its exponential I-V turns the membrane voltage into a large
read error. That is a real design trade-off and only appears when the cell's I-V is evaluated against
the membrane voltage every tick.

**Partition invariance [run].** The same chip on 1, 2 and 4 MPI ranks (one GPU, `--oversubscribe`),
with every core placed off its encoders' rank so every device reads its input across a rank boundary:
spike digest `93a677e0…` on all three — bitwise identical spike trains.

### 4.1 Ablation: how much of Experiment A could NEST GPU do? (`--static-only`, `compare_ablation.py`)

The static-only mode keeps exactly what a fixed-weight simulator can represent — a constant weight
per synapse and a constant extra leak per neuron (the passive-array leak linearized at 0 V) — from the
**same device draws**, and drops every in-run effect (read noise, membrane-dependent read current,
selector nonlinearity). Same chip, same input spikes; paired test per design:

| Design | Full device model | Static-only | Gap | McNemar p |
|---|---|---|---|---|
| ideal | 95.0 % | 95.0 % | 0.0 | 1 |
| sram_4bit | 94.7 % | 94.7 % | 0.0 | 1 |
| rram_1t1r | 80.0 % | 80.6 % | -0.6 | 0.63 |
| rram_passive | 66.9 % | 67.2 % | -0.3 | 1 |
| **rram_passive_sel** | **72.2 %** | **79.7 %** | **-7.5** | **7e-6** |
| pcm_1month | 85.8 % | 86.1 % | -0.3 | 1 |
| pcm_1month_gdc | 92.2 % | 91.9 % | +0.3 | 1 |
| hetero_aware | 85.0 % | 84.4 % | +0.6 | 0.5 |
| hetero_random | 82.5 % | 82.5 % | 0.0 | 1 |

**Honest reading: 8 of 9 inference designs are reproducible in NEST GPU by baking weights** (the linear
passive crossbar is algebraically a per-neuron leak plus weights, apart from read noise, and the
ablation confirms it). Only the
selector crossbar needs per-tick device code: its static approximation over-estimates accuracy by
7.5 points. Inference-only hardware studies dominated by static effects do not, by themselves, need
SuperNeuroABM. That is why Experiment B exists.

---

## 5. Experiment B — a chip that learns by itself (`run_learning_poc.py`)

**Demonstrates:** C1 with state that changes during the run (each device's conductance, moved by its
own programming pulses), C2 (the device reads its neuron's activity trace), C4 (per-device step sizes,
write noise and stochastic switching, identical on 1, 2 and 4 ranks; Monte Carlo over 16 chip
instances), C6. Weak on C3. **Against other simulators:** NEST GPU cannot run 5 of the 7 devices; NEST
on CPU most likely can; Brian2 can on one GPU.

**Setup.** Same 64 -> 10 task, but every core starts blank and learns on-chip in one pass over the 1 437
training digits, then classifies the 360 test digits. 64 input relays and 10 teacher relays are shared;
during training the teacher clamps the correct output on and the others off. Each presynaptic spike
reads its device against a reference column and applies one programming pulse; the post-neuron's
activity trace picks SET or RESET. What the pulse does is the device model (`learning_devices.py`):

- `ideal`: analog, deterministic, conductance-dependent (soft-bounded) steps;
- `linear`: constant steps with hard bounds (abrupt, state-independent switching);
- `d2d`: every device its own SET and RESET step size (30 % spread);
- `c2c`: write noise as large as the step;
- `levels16`, `levels4`: finite conductance states with stochastic switching;
- `rram_full`: d2d + c2c + 16 levels + 2 % stuck cells.

Seven designs x 16 chip instances (1 194 somas, 82 954 synapses), 125 790 ticks, 9 s training + 4 s
test on one laptop GPU. Instances differ only in their device draws.

| Device | Mean accuracy | sd | min | max | What NEST GPU would need |
|---|---|---|---|---|---|
| ideal | 74.2 % | 0.0 | 74.2 % | 74.2 % | a new C++ rule (fits its hook) |
| linear | 78.1 % | 0.0 | 78.1 % | 78.1 % | a new C++ rule (fits its hook) |
| d2d | 49.1 % | 10.8 | 24.2 % | 64.4 % | per-synapse parameters: not passed to the hook |
| c2c | 73.6 % | 1.9 | 70.8 % | 77.5 % | a random number per pulse: not passed to the hook |
| levels16 | 35.9 % | 6.2 | 25.3 % | 47.2 % | a random number per pulse: not passed to the hook |
| levels4 | 19.6 % | 6.2 | 10.3 % | 29.2 % | a random number per pulse: not passed to the hook |
| rram_full | 25.0 % | 4.4 | 18.3 % | 33.1 % | both |

Learned conductances (g - g_ref, instance 0 of each design):
`poc/hetero_chip/results/learning_conductances_seed1.png`. The ideal and write-noise devices learn
recognizable digit templates; constant steps binarize them; device-to-device step variation scrambles
them; 16- and 4-level devices with stochastic switching reduce them to a few noisy levels.

**Findings.** The step-size asymmetry between devices (d2d) and few-level stochastic switching are what
break on-chip learning here; write noise as large as the step is nearly harmless, because thousands
of small pulses average it out. Instance-to-instance spread is large (d2d: 24-64 %), so a single
simulated chip says little about yield — the Monte Carlo over chip instances is part of the result.
The ideal device reaches 74 %, below the offline classifier (95.8 %): the rule is a local, one-pass
Hebbian rule, and the step size trades speed against noise (smaller steps were more accurate in an
offline check). Accuracy is not the point of the experiment; the comparison between devices is.

**Why this cannot be pre-baked or added to NEST GPU.** (NEST on CPU is different: see section 3.2.) The weight is the device's state, produced
during the run by activity, so there is nothing to fold into weights. A new rule could be added to
NEST GPU in C++ for the two deterministic devices, because they fit `SynapseUpdate(syn_group, &w, Dt)`.
The other five need a random number per pulse or per-synapse step sizes, and the hook passes neither.
Per-device step sizes could be approximated coarsely by spreading synapses over several synapse
groups with different parameters, but the group index shares a 32-bit word with the target-node index
(`conn12b.cu:52`: `max_port_syn_nbits_ = 32 - max_node_nbits_`), so the number of groups shrinks as
the network grows. Per-pulse randomness has no such workaround; supporting it means changing the hook
signature, the connection struct and the delivery kernels.

**Partition invariance [run].** 4 instances per design, full training (125 790 ticks with on-chip
learning, stochastic switching and per-device step variation), on 1, 2 and 4 ranks: test-phase spike
digest `e49a12c8…` on all three and identical per-instance accuracies (wall time 7 s / 3.4 min /
8 min — the multi-rank runs share one GPU and exchange spikes every tick). A reduced 200-digit run gave
`f4cc22ee…` on 1, 2 and 4 ranks.

### 5.1 "Every simulator can do STDP" — learning rule versus learning device

STDP is a learning **rule**: a function that says how much a weight should change given spike timing.
NEST, NEST GPU and Brian2 all run it the same way: compute Δw, add it to a float weight, exactly.
NEST GPU's built-in rule (`stdp.h:95`) is a deterministic function of the weight, the spike-time
difference and group-shared parameters; two synapses with the same history get the same update.

On a memristive chip the rule only decides *which pulse to send*. What the pulse does is device physics:

| | Rule in a simulator | Rule executed by a device |
|---|---|---|
| Step vs state | as written in the rule | SET/RESET steps shrink near the bounds, SET and RESET differ |
| Same pulse twice | same Δw | different Δg each time (write noise) |
| Two devices, same pulse | same Δw | different step sizes (device-to-device variation) |
| Resolution | float | a few levels; a sub-level request switches only with some probability |
| Faults | none | stuck cells ignore the pulse |

So the question is not whether a simulator can run STDP, but whether it can run *STDP as executed by
device X*: per-synapse state, per-synapse parameters and a random number per pulse inside the
update. All seven rows of the table above use the same rule; only the device under it changes, and the
result moves from 74 % (ideal device — any simulator can produce this once the rule exists) to 49 %
(device-to-device variation) and 20 % (4 levels, stochastic switching). Those last numbers are the ones
a hardware designer needs.

Strictly, Experiment B's rule is not pair-based STDP: it is a teacher-supervised, spike-driven Hebbian rule
(SET on a presynaptic spike while the post-neuron is active, RESET otherwise). The device argument is
independent of the rule. SuperNeuroABM also ships a device-level STDP rule,
`superneuroabm/step_functions/synapse/stdp/exp_pair_wise_stdp_memristive.py` (nonlinear update with
per-pulse write and read noise via `rand_normal`), which NEST GPU's hook cannot host for the same
reason. Repeating Experiment B with timing-based STDP is a straightforward variant if needed.

---

## 6. What hardware-aware SNN users do today without SuperNeuroABM

1. **Bake static effects into weights** and run NEST GPU / GeNN / Brian2 at full speed. Works for
   quantization, programming error, faults, drift at a fixed age, mismatch — section 4.1 shows this
   covers most inference-only studies.
2. **Model dynamic effects on one GPU** in Brian2 or GeNN, with one synapse group per technology.
3. **Fork the simulator** (NEST GPU: widen the connection struct, add per-step kernels and RNG,
   rebuild).
4. **Use a vendor's simulator** for one chip and one technology, which does not support comparing
   technologies.

None of these gives dynamic, learning, heterogeneous devices across many GPUs with reproducible chip
instances.

---

## 7. What is still missing before this is a paper

1. **A run that does not fit one GPU.** This is the experiment that makes Brian2/GeNN a "cannot" rather
   than "possible with workarounds". At the measured 147 B per agent, an 80 GB GPU holds about 550 M
   devices. A device-level model of a billion-synapse system needs several GPUs, and one of a
   system-scale platform (Intel's Hala Point is quoted at 128 billion synapses) needs hundreds. Plan:
   add a core-replication knob to Experiment B, run it on an HPC system past one GPU's memory, report weak
   scaling and the digest across GPU counts.
2. **One experiment that is a "cannot" for all three at once.** Today each "cannot" comes from a
   different design. Combine them: learning devices with per-pulse randomness and per-device
   parameters (rules out NEST GPU), which also evolve every tick with the membrane voltage — a passive
   crossbar with selectors, or volatile memristors (rules out NEST on CPU, whose synapses run only at
   spikes) — at a size beyond one GPU (rules out Brian2).
3. **Test the remaining simulators.** GeNN and Arbor: done on CPU (section 3.4) — Arbor covers most
   rows, so the argument now rests on device-to-device coupling across neurons at multi-GPU scale.
   Still open: run Arbor on GPUs with MPI and measure its cost for point neurons on cable cells;
   build Experiment B's device as a NESTML synapse (section 3.2).
3a. **A hardware experiment built on device-to-device coupling.** The coupling itself is shown
   (section 3.4 probe). Still to do: a device model that reads its physical neighbours (thermal
   crosstalk or write disturb between adjacent cells of different neurons) during on-chip learning,
   run past one GPU.
4. **Speed context:** run the static-only Experiment A in NEST GPU and report the price of generality on
   the same workload (expected: NEST GPU faster per tick).
5. **Device parameters from one real technology** (and a citation per number) in place of
   illustrative values.
6. **A stronger learning experiment** (more epochs, signed differential-pair devices with refresh —
   which needs imperative device code) so the ideal-device accuracy is closer to offline training.

---

## 8. Reproduce

```bash
cd poc/hetero_chip
PY=/home/xxz/miniforge3/envs/sna-dev/bin/python

# Experiment A: inference chip, 9 designs; invariance across ranks; static-only ablation
$PY run_chip_poc.py
mpirun --oversubscribe -n 2 $PY run_chip_poc.py      # same digest
$PY run_chip_poc.py --static-only && $PY compare_ablation.py
$PY run_chip_poc.py --v-swing 0.05 --out /tmp/sweep  # passive-array sweep

# Experiment B: on-chip learning, 7 devices x 16 chip instances
$PY run_learning_poc.py --plot
mpirun --oversubscribe -n 2 $PY run_learning_poc.py --instances 4   # same digest as 1 rank

# Brian2 probes (env with Brian2 2.9)
/home/xxz/miniforge3/envs/diehlcook/bin/python probe_brian2.py
```

Things found about SuperNeuroABM's own API while building this: `docs/API_DESIGN_NOTES.md`
(the `simulate()` sync-interval argument; the columnar loader's single-breed assumption) and
`docs/SPIKE_RECORDING_NOTES.md` Finding 4 (record buffer sizing).
