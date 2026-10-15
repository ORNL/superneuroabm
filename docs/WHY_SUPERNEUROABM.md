# Why SuperNeuroABM

**What we can do that the others cannot:** simulate a neuromorphic chip *device by device* — every
synapse running its own technology's model, with its own changing state and its own randomness,
updated every time step together with the neuron it feeds — mixed freely in one network, on many
GPUs, with the same result whatever the number of GPUs.

**Caveat (2026-10-13):** Arbor can do most of this too (rows 1-5, 7, 8 below, for cable cells). The
one capability found in no other simulator is row 9 — a device that interacts with other devices, not
only with its own two neurons — on many GPUs. SuperNeuroABM does it through `load_from_adjacency()`
(shown with a small probe; a hardware experiment that uses it is the next step).

Compared here: NEST GPU 2.0, NEST 3 (CPU), Brian2 2.9, GeNN 5.5, Arbor 0.12. Full evidence:
[`WHY_SUPERNEUROABM_EVIDENCE.md`](WHY_SUPERNEUROABM_EVIDENCE.md); code: `poc/hetero_chip/`.

## Who can do what

| # | Capability | NEST GPU | NEST (CPU) | Brian2 | GeNN | Arbor | SuperNeuroABM |
|---|---|---|---|---|---|---|---|
| 1 | Write a new synapse / device model without rebuilding the simulator | ❌ 3 synapse models hard-coded in C++ (s) | ⚠️ NESTML, which generates and compiles C++ (d) | ✅ equations (p) | ✅ code strings (p) | ⚠️ NMODL compiled into a catalogue (p) | ✅ Python kernel (r) |
| 2 | Device state that changes during the run (e.g. a conductance moved by programming pulses) | ❌ one float weight, changed only by its built-in STDP (s) | ✅ at spike times (d) | ✅ (p) | ✅ (p) | ✅ (p) | ✅ (r) |
| 3 | Per-device randomness and parameters in that update (write noise, stochastic switching, device-to-device variation) | ❌ the update receives only weight, spike-time difference and group-shared parameters (s) | ✅ (d) | ✅ (p) | ⚠️ no random numbers in synapse code; workaround (p) | ⚠️ Gaussian only; at spike time via an undocumented workaround (p) | ✅ (r) |
| 4 | Device code that runs every time step using the neuron's membrane voltage (passive-crossbar leak, read current vs membrane) | ❌ synapses run only when a spike crosses them (s) | ❌ same (d) | ✅ (p) | ✅ (p) | ✅ (p) | ✅ (r) |
| 5 | Different device models on the same neuron, chosen per synapse | ❌ only as plain weights (s) | ✅ per-connection model (d) | ⚠️ one synapse group *and* one neuron input variable per model; no `if`/`for` in model code (p) | ✅ (p) | ✅ (p) | ✅ (r) |
| 6 | Same simulated chip (same device randomness) on any number of GPUs / processes | n/a: no device randomness | ❌ random streams follow the process count (d) | n/a: one process | n/a: one GPU | ⚠️ designed for it; noise depends on which mechanisms share a cell group, not bitwise in probe (p, s) | ✅ bitwise identical on 1, 2, 4 ranks (r) |
| 7 | Runs on GPUs | ✅ | ❌ | ✅ one GPU (Brian2CUDA) (d) | ✅ one GPU | ✅ (d) | ✅ (r) |
| 8 | Scales across many GPUs / nodes | ✅ | ✅ CPU nodes | ❌ one node (OpenMP) or one GPU (d, s) | ❌ one GPU (s) | ✅ MPI (d) | ✅ MPI (r, so far on one GPU) |
| 9 | A device that reads other devices or a third neuron (thermal crosstalk, write disturb between neighbouring cells of different neurons) | ❌ (s) | ❌ (d) | ❌ between synapses; only by making every device a neuron (p) | ⚠️ workaround, one GPU (p) | ❌ across neurons; same compartment only (p, s) | ✅ `load_from_adjacency()`: a device reads a device of another neuron on another rank, bitwise on 1, 2, 4 ranks (r) |

(s) read in the simulator's source · (p) tested with `poc/hetero_chip/probe_brian2.py` ·
(r) run here · (d) from documentation, not tested. ❌ means *not possible through the simulator's
model interfaces*; with enough changes to a simulator's own source code anything is possible, and that
is a fork, not a model.

**Each of the others misses something:**
- **NEST GPU** misses 1-5: it has no device models at all, only a fixed weight.
- **NEST (CPU)** misses 4, 6 and 7: no device code between spikes, no GPU, and the simulated chip changes with the process count.
- **Brian2** misses 8 and needs workarounds for 5: everything runs on one node or one GPU.
- **GeNN** misses 8 and needs workarounds for 3: everything runs on one GPU.
- **Arbor** misses 9 and needs workarounds for 3 and 6; point neurons must be cable cells (cost not measured).

Arbor comes closest. What no other column has is row 9 together with row 8; SuperNeuroABM is the only
column with both.

## Static vs dynamic device effects — which rows they need

Not every hardware effect needs a special simulator. What matters is *when* the effect happens:

| Class | What it means | Examples | Rows needed | Who can |
|---|---|---|---|---|
| **Static** | a constant per device once the chip is programmed, independent of activity | weight quantization, programming error, stuck-at cells, drift evaluated at a fixed age, neuron-circuit mismatch, a linear crossbar leak (a constant extra leak per neuron) | **none of 1-6**: compute it in Python before the run and fold it into each synapse's weight and each neuron's parameters | every simulator, including NEST GPU |
| **Dynamic, at spike time** | the device changes or is read only when a spike crosses it, and the change depends on its own state, its own parameters or fresh randomness | on-chip learning by programming pulses, write noise, stochastic switching between levels, device-to-device variation of the update step, per-read noise, wear-out counting | **2, 3** (plus 1 and 5 to write and mix device models; 6 to keep the randomness consistent across GPUs) | NEST (CPU), Brian2, GeNN and Arbor (randomness through workarounds), SuperNeuroABM — **not NEST GPU** |
| **Dynamic, every time step** | the device acts *between* spikes, driven by the neuron's changing membrane voltage | passive-crossbar leak through a nonlinear selector, read current that falls as the membrane charges, volatile memristors relaxing and driven by the membrane | **4** (plus 1-3, 5, 6) | Brian2, GeNN, Arbor, SuperNeuroABM — **not NEST GPU, not NEST (CPU)** |
| **Dynamic, coupled between devices** | a device changes because of *other devices* — neighbouring cells that belong to other neurons | thermal crosstalk when a neighbouring cell is programmed, write disturb on shared lines | **9** (plus 2-4) | GeNN through workarounds on one GPU; SuperNeuroABM — **not NEST GPU, NEST (CPU), Brian2 or Arbor** |

Rows 7-8 (GPUs, many GPUs) apply to every class and are where Brian2 and GeNN drop out.

So every "cannot" in this document is about a dynamic effect. In the experiments below, 8 of the 9
inference designs of Experiment A turned out to be static (folding them into weights changed nothing),
the selector crossbar is dynamic-every-step, and Experiment B (learning in the device) is dynamic at
spike time. An inference study made only of static effects does not need SuperNeuroABM; learning and
anything that happens between spikes does.

## Why

### NEST GPU — a synapse is a weight, and a spike is an addition

**How it works.** Each connection is a 12-byte record (default layout): a 32-bit key packing the
source neuron and the delay, a 32-bit word packing the target neuron, the receptor port and a
synapse-group index, and one `float weight` (`conn12b.h:29`, `:35`). Every time step,
NEST GPU updates every *neuron* (`nestgpu.cu:751`, `:788`). Synapses are touched only when a neuron
spikes: `deliverSpikesKernel` walks the spiking neuron's outgoing connections and adds `weight` to
each target's input buffer with one `atomicAdd` (`input_spike_buffer.h:306`, `:329`). Plasticity is a
single function, `SynapseUpdate(syn_group, &weight, Dt)` (`syn_model.h:46`), called when the pre- or
post-neuron spikes (`get_spike.h:95`, `rev_spike.h:92`).

- **Row 1 ❌.** The synapse models are a fixed list in C++ — `{"", "test_syn_model", "stdp"}`
  (`syn_model.h:61`), i.e. a static synapse and two plastic ones — and `CreateSynGroup`
  (`syn_model.cu:129-142`) throws `Unknown synapse model name` for anything else. A new model means
  editing those files, adding a CUDA function to the `switch` in `SynapseUpdate`, and rebuilding the
  library. NEST GPU's documentation explains how to add new *neuron* models (copy a `user_m1`
  template, edit, rebuild) and has no equivalent for synapses.
- **Row 2 ❌.** The record has room for exactly one number, the weight, and the only code that writes
  it is the built-in STDP (or the test model). A device needs more state — a conductance pair, a
  conductance level, a write count — and there is nowhere to keep it without changing the connection
  record, which the connection builder, the spike-delivery kernels and the multi-GPU connection code
  all depend on.
- **Row 3 ❌.** `SynapseUpdate` receives the weight, the spike-time difference and
  `SynGroupParamMap[syn_group - 1]`: one parameter set *shared by every synapse in the group*. It
  receives no random-number state, so there can be no write noise and no stochastic switching, and no
  per-synapse parameters, so no device-to-device variation. Splitting synapses into several groups
  could fake a few device classes, but the group index shares one 32-bit word with the target-neuron
  index (`conn12b.cu:52`: `max_port_syn_nbits_ = 32 - max_node_nbits_`), so the bigger the network,
  the fewer groups are available.
- **Row 4 ❌.** The per-step loop runs over neurons only. A synapse whose pre-neuron is silent is never
  visited, and the delivery kernel only adds the weight — it never reads the target's membrane
  voltage. So a cell that conducts every step depending on the membrane (a passive-crossbar cell
  leaking the column, a read current that saturates as the membrane charges) cannot be expressed.
- **Row 5 ❌.** There are no device models to mix. A "hybrid SRAM + RRAM" network can only exist as
  different static weight values.
- **Row 6 n/a.** There is no randomness inside synapse updates to keep consistent.
- **Rows 7-8 ✅.** It runs on GPUs and across GPUs, and it is fast: 12 bytes per synapse (16 in its wide layout) and work
  proportional to spikes. That is the reason for the design above, not an oversight.

**Connectivity limits.** Any directed graph between neurons can be built — rules `one_to_one`,
`all_to_all`, `fixed_total_number`, `fixed_indegree`, `fixed_outdegree`, and arbitrary edge lists by
passing index lists to `one_to_one` — with no fan-in or fan-out cap other than memory. Checked: the
full SGNN Cora network (31 836 connections, fan-in from 1 to 1520, per-edge weights, STDP readout
group) builds in NEST GPU with every edge and every fan-in identical to the SuperNeuroABM file
(`scaling_analysis/nest_comparison/cora_graph_nestgpu.py`). What did not carry over is per-synapse
behaviour: synaptic time constants become receptor ports of the neuron (fine for Cora's three values,
not for a different one per synapse), and the neuron and STDP rule are not identical without new C++.
The limits are elsewhere:
- *Every connection is neuron to neuron.* Nothing can connect to a connection, and there is no
  third-party (tripartite) rule.
- *Missing rules* (per its docs): `pairwise_bernoulli`, `pairwise_poisson`, the tripartite rule and
  spatial (distance-dependent) rules. Autapses and multapses are always allowed by the probabilistic
  rules and cannot be switched off.
- *A fixed bit budget per connection.* In the default 12-byte layout, node index, delay, receptor
  port and synapse group share 64 bits: by default 2^20 (~1 M) neurons per GPU (`conn12b.cu:156`;
  `CreateNodeGroup` throws beyond that), delays up to 4095 steps (the remaining 12 bits of the key),
  64 synapse groups (`conn12b.cu:158`) and 32 receptor ports. Allowing more neurons per GPU takes bits
  from delays, ports and groups — 2^24 neurons leaves 255-step delays and 8 bits for port and group
  together. The 16-byte layout (`conn16b.cu:136-138`) gives a 32-bit node index, 16383-step delays,
  1024 groups and 128 ports. For this document the group count matters most: faking device-to-device
  variation with synapse groups (row 3) gets at most 64 device classes by default.

**Couldn't a custom neuron model do the device work instead?** NEST GPU lets users write new neuron
models (from the `user_m1` templates, then rebuild), so why not read the synapse weights there and add
read and write noise? Three facts from the source:

1. *The neuron sees only a sum.* Arriving spikes reach the neuron through `GetSpikes`
   (`get_spike.cu`) as one number per receptor port per step — the sum of `weight x mul` over every
   spike that arrived. Which synapse contributed what is gone. A neuron model can add noise to that
   sum, an approximation of read noise, but per-device read noise needs each device's contribution
   separately (the right noise size depends on the squared weights of the synapses that fired, which
   the sum does not reveal).
2. *The neuron-model interface does not give a neuron its synapses.* A neuron kernel gets its own
   variables and its summed input. NEST GPU does keep an internal reverse index from a neuron to its
   incoming connections (`TargetRevConnection`, `rev_spike.cu:50`), but only for plastic connections
   (`connect.h:2641-2643`) and only as an internal device global. A custom kernel *can* reach into it
   and rewrite `weight` with noise — but that is editing NEST GPU's internal CUDA, relying on
   undocumented data structures, limited to plastic connections, and still with one float per device:
   any further device state needs new arrays indexed by connection number, kept consistent with NEST
   GPU's connection sorting and multi-GPU layout.
3. *Doing it per step means a serial loop per neuron.* Every-step device code (row 4) would have the
   neuron kernel walk all its incoming devices through that index on every step — the "neuron and
   synapses as one agent" design of the section below, built by hand inside NEST GPU, with its costs.

So the ❌ marks for NEST GPU mean *not possible through its model interfaces*; anything is possible by
rewriting its CUDA, which is a fork of the simulator. What a custom NEST GPU neuron model can do
legitimately is neuron-level effects: circuit mismatch, thermal noise, approximate noise on the
summed input.

### NEST (CPU) — programmable synapses, but only when a spike arrives

**How it works.** A synapse is a C++ object stored with its target neuron's thread. When a spike is
delivered through it, its `send()` runs: it can update its own state (decaying it analytically over
the time since its last spike), read the post-neuron's spike history, and pass a weighted event on.
Neurons are updated every time step; synapses are not. New synapse models are written in NESTML,
which generates C++ that is compiled into a module and loaded. It runs on CPUs with MPI and threads.

- **Row 1 ⚠️.** Possible, but every new or changed device model is a code-generate-compile-load cycle,
  not a Python function.
- **Rows 2, 3, 5 ✅.** A NESTML synapse can carry device state, draw random numbers and have its own
  parameters, and each connection can use a different synapse model. So on-chip device learning,
  where the device changes only when a spike crosses it, should be expressible (not tested here).
- **Row 4 ❌.** A synapse only runs when a spike arrives. A device whose state or current depends on
  the membrane voltage *between* spikes — a passive-crossbar leak through a nonlinear selector, a
  volatile memristor driven by the membrane — would need the whole membrane trajectory, which a synapse
  never sees. The only workaround is to move every attached device's physics into a custom neuron
  model, which gives up per-synapse device models.
- **Row 6 ❌.** Random numbers come from per-thread ("virtual process") streams, so results repeat only
  for the same number of virtual processes. Run the same network on more processes and every device
  gets different noise: a different chip. Avoiding that needs a hand-written counter-based generator
  keyed on connection identity, in C++.
- **Row 7 ❌.** CPU only. On today's GPU-heavy supercomputers it leaves most of the machine idle.
- **Row 8 ✅.** It scales to the largest CPU supercomputers.

### Brian2 — expressive equations, one model per group, one machine

**How it works.** A `Synapses` group is described by one set of equations and statements, compiled to
code. Equations can be `clock-driven` (integrated every step, and able to read the post-neuron's
`v_post`) and statements can draw random numbers. The group's output reaches the neuron through a
`(summed)` variable.

- **Rows 1-4 ✅** (all probed): equations, per-synapse state and parameters, `randn()` inside
  updates, and per-step synapse equations that read `v_post`.
- **Row 5 ⚠️.** One `Synapses` group has one model, so three technologies need three groups. Brian2
  then refuses to let two groups sum into the same neuron variable (`NotImplementedError: Multiple
  'summed variables' target the variable 'Isyn'`, probed), so the neuron equations need one input
  variable per technology and must be rewritten whenever a technology is added. Inside one model there
  is no `if` or `for` (probed): branches such as stuck-at faults or quantization have to be rewritten as
  arithmetic (`int(cond)`, `clip`), and loops (an iterative device solve) need hand-written C++/CUDA
  functions.
- **Row 6 n/a.** A Brian2 run is one process, so there is no decomposition to stay consistent across.
- **Row 7 ✅, one GPU.** Brian2CUDA generates code for a single GPU; every benchmark in its paper runs
  on one GPU.
- **Row 8 ❌.** Brian2's own parallelism is OpenMP threads in C++ standalone mode — one shared-memory
  node, and its documentation says that code "has not yet been well tested". There is no MPI or
  multi-node mode in its documentation, and none in its code generator (Brian2 2.9: 20 OpenMP
  references, 0 MPI). The largest model is what fits one node or one GPU.

### GeNN and Arbor — checked 2026-10-13 (details: evidence doc §3.4)

- **GeNN** generates GPU code from per-synapse code strings: per-step synapse code reading `V_post`,
  per-synapse state and parameters, and mixed models on one neuron all work (probed on its CPU
  backend). It has no random numbers in synapse code (a custom connectivity update can refill random
  variables each step) and runs on **one GPU**: no MPI or model partitioning.
- **Arbor** places NMODL mechanisms on cable cells and runs on many GPUs with MPI. A mechanism has
  its own state and parameters, runs every step against the membrane voltage, and several kinds can
  sit on one cell. Its only randomness is Gaussian white noise in a stochastic ODE (reaching it at
  spike time needs an undocumented trick), point neurons must be single-compartment cable cells, and
  a mechanism sees only its own compartment: **no coupling between devices on different neurons**
  except gap junctions, which see only two voltages and must sit on the same rank.

### SuperNeuroABM — every synapse is an agent

- **Row 1.** A device is a ~30-line Python function compiled with `cupyx.jit`, registered at run time
  (`register_synapse_type`). Nothing is rebuilt.
- **Rows 2-3.** Each synapse agent has its own state vector and parameters, read and written by its
  own kernel. `rand_normal(tick, agent_index, salt)` is rewritten to a Philox draw keyed on the
  synapse's *global id*: a draw at tick 0 is a fixed per-device constant (programming error, fault),
  a draw at `tick` is per-event noise.
- **Row 9.** `load_from_adjacency()` takes each agent's neighbour list verbatim: a synapse's two slots
  and a soma's input list may name *any* agent — another synapse included — on any rank, and SAGESim's
  ghost exchange delivers synapse state across GPUs every tick (the same path S-GATs uses).
- **Row 4.** Every agent's kernel runs every time step (idle synapses return early), and each synapse
  is placed on the same GPU as its post-neuron (post-owns), so it can read the neuron's membrane
  voltage directly.
- **Row 5.** The model is chosen per agent (its breed); the generated kernel dispatches per agent, so
  any mix of device models can feed one neuron.
- **Row 6.** Random numbers depend on the device's global id, not on which GPU it is on: the same chip
  on 1, 2 or 4 ranks gives bitwise-identical spikes.
- **Rows 7-8.** CuPy kernels on GPUs; MPI between GPUs, exchanging spikes and synapse outputs.

**The price.** About 10x more GPU memory per synapse than NEST GPU (147 vs 12-16 bytes) and, on a
Brunel network, about 7x slower per time step (1.28 vs 0.19 ms with 12.5 M synapses), the cost of
each synapse being an agent with its own row and learning rule rather than a 12-byte record. The
work is proportional to spikes in both: synapses are stepped only when their presynaptic neuron
fires. On a single-neuron STDP experiment (Masquelier 2008, 2,000 plastic synapses) SuperNeuroABM
is faster than both Brian2 on CPU and NEST GPU (1.07 s vs 5.2 s vs 11.0 s). On scaling, NEST GPU has run far larger weak-scaling tests — 128 to 1,024 GPUs at 2.5 billion
synapses per GPU, random connectivity, firing — with simulation time per model second growing
sublinearly and construction time linearly. Ours is flat from 64 to 2,048 GPUs, but at 12.5-50
million synapses per GPU on a local, silent network; the workloads are too different to say who
scales better (details: evidence doc §3.1a). Our claim is what each synapse can do, not size or speed.

## What if a neuron and its synapses were one agent?

NEST and NEST GPU keep a neuron's incoming synapses *with* the neuron. Could SuperNeuroABM do the same
— one agent per neuron, holding its own state plus the state and parameters of every incoming
synapse, with one kernel that updates the neuron and loops over its synapses — and keep the table?

| # | Capability | Kept? | What changes |
|---|---|---|---|
| 1 | New device model without rebuild | ✅ | Still Python, but a device model is now code *inside* the neuron kernel, not a kernel of its own. |
| 2 | Changing device state | ✅ | Stored as per-synapse slots in the neuron agent's state. |
| 3 | Per-device randomness and parameters | ✅ | Key the random numbers on (neuron id, slot); store parameters per slot. |
| 4 | Every-step device code using the membrane | ✅ | Easier: the membrane voltage is local to the same agent. |
| 5 | Different device models on one neuron | ⚠️ | One kernel serves every synapse of the neuron, so it must contain every device model and switch on a per-slot type code — the same union-model workaround Brian2 forces. |
| 6 | Same chip on any number of GPUs | ✅ | As long as slot order is fixed. |
| 7-8 | GPUs, many GPUs | ✅ | Neurons partition naturally; only spikes cross GPUs. |

**What it would gain**
- **Memory.** No agent bookkeeping per synapse (id, breed, neighbour lists, spike buffers, padding to
  the widest breed). Simple synapses could approach NEST GPU's few bytes; device-rich ones still need
  their own state.
- **Setup time.** Building millions of synapse agents is today's slowest step; it would mostly go.
- **One pass instead of two.** Synapses would no longer write a current that the neuron then reads back.

**What it would cost**
- **Row 5, as above:** device models stop being separate, registrable kernels.
- **GPU parallelism.** One GPU thread per neuron walks its synapses one by one. With thousands of
  inputs per neuron, and fan-in that varies from neuron to neuron, that is slow and badly balanced;
  today there is one thread per synapse. Fixing it means teaching SAGESim to split one agent over many
  threads.
- **Uneven fan-in.** SAGESim pads every agent's property vector to the largest one. A neuron agent
  holding *fan-in x (state + parameters)* would pad every neuron to the largest fan-in in the network,
  unless SAGESim gains ragged per-agent arrays.
- **Synapses as things in their own right.** Other agents could no longer read or connect to a
  synapse — synapse-level pooling and synapses used as input ports (needed by the S-GATs work) go away —
  and the per-synapse choice of learning rule becomes another switch inside the neuron kernel.
- **Speed is not automatic.** Merging removes per-synapse overhead, but the neuron would still visit
  every synapse every step. NEST GPU's per-step speed comes from *pushing* spikes only to the
  synapses of neurons that fired. Adopting that is exactly what loses row 4 — device code that must run
  on silent synapses every step.

**Conclusion.** Merging would keep most of the table and save memory and setup time, but it turns
row 5 into a union model, gives up synapse-level parallelism and synapse-level structure, and does not
by itself deliver NEST GPU's speed. The capability comes from *user code and user state per synapse,
on GPUs*, not from synapses being separate agents as such; separate agents are how SuperNeuroABM gets
it without union models or padding. This matches the project decision of 2026-09-10: synapses stay
agents, and a merged neuron+synapse kind would at most be an optional fast path for networks whose
synapses are simple (static weights or standard STDP), not for device-level studies.

## Evidence: two experiments

Both map a small digit classifier onto simulated hardware (64 inputs -> 640 devices -> 10 neurons per
core, many cores per run).

**Experiment A — programmed inference chip** (`run_chip_poc.py`; uses rows 1, 4, 5, 6). Weights from
an offline classifier are written into SRAM, RRAM and PCM devices; nine hardware designs, 2.1 M
devices, one run. Most device effects here are constants that could be folded into plain weights, so 8
of the 9 designs could also be run in NEST GPU. The exception needs row 4: a passive crossbar with
selectors, whose cells conduct every time step against the membrane voltage. Approximating it with
plain weights misses its accuracy by 7.5 points (p = 7e-6).

**Experiment B — a chip that learns by itself** (`run_learning_poc.py`; uses rows 2, 3, 6). Devices
start blank; each input spike sends a programming pulse whose effect is the device model. Same
learning rule, seven device behaviours, 16 chip instances each:

| Device | Accuracy | NEST GPU can run it? |
|---|---|---|
| ideal | 74.2 % | yes, with a new C++ rule |
| constant-step switching | 78.1 % | yes, with a new C++ rule |
| write noise as large as the step | 73.6 % | no (row 3) |
| 30 % device-to-device step variation | 49.1 % | no (row 3) |
| 16 / 4 levels, stochastic switching | 35.9 % / 19.6 % | no (row 3) |
| all of the above + 2 % stuck cells | 25.0 % | no (row 3) |

Both experiments give bitwise-identical spikes on 1, 2 and 4 MPI ranks (row 6).

## Not shown yet

- **A hardware experiment that needs row 9.** Row 9 is shown only by a probe (`run_coupling_probe.py`:
  each device reads one device of another neuron). Next: thermal crosstalk or write disturb between
  neighbouring cells during on-chip learning, run past one GPU. It is now the claim that separates
  SuperNeuroABM from Arbor. A synapse agent has two neighbour slots (any agents), so a device that
  needs its input, its neuron *and* several neighbours must be a soma-kind agent or read them through
  a helper agent (not tried yet).

- **Row 8 beyond one GPU.** Every multi-rank run here shared one laptop GPU. A run larger than one
  GPU's memory is what turns Brian2 and GeNN from "needs workarounds" into "cannot".
- **One experiment that needs every row.** Today rows 2-3 (rules out NEST GPU) and row 4 (rules out
  NEST on CPU) come from different experiments. Next: learning devices that also evolve every time step
  with the membrane voltage, run past one GPU.
- **Not run:** NEST on CPU (its cells come from documentation); GeNN on a GPU and Arbor on GPUs with
  MPI (their cells come from CPU probes plus source and documentation). No speed comparison on these
  workloads.

## Is continued development worth funding?

**Yes, for a narrow purpose; NEST is enough for most other work.**

- **Use NEST / NEST GPU** for standard point-neuron networks and standard plasticity (NEST GPU is
  ~40x faster per step, ~10x leaner, mature), for hardware studies with only static effects, and for
  raw scale.
- **SuperNeuroABM is the tool for co-designing learning neuromorphic hardware:** learning inside
  memristive devices with per-device randomness and parameters, devices that act every time step
  with the neuron, both mixed per synapse, on many GPUs, reproducible per chip instance — plus
  synapse-level computation (e.g. S-GATs). Pitch it as that, not as a faster or bigger NEST.

**Deliverables that would settle the question** (and the gaps a reviewer would find today):
1. One experiment that needs every row of the table, including row 9, run past one GPU's memory
   (the decisive "cannot" for NEST GPU, NEST on CPU, Brian2, GeNN and Arbor together).
2. One hardware-calibrated study: real device parameters, a prediction checked against measurements.
3. Run Arbor on GPUs with point-like cells (speed and memory against SuperNeuroABM), test
   NEST+NESTML, and verify agent ids beyond 2^24 (stored as float32 on the GPU,
   which may make neighbouring devices share random numbers in very large networks — unverified).
4. Reduce construction time and memory per synapse without losing per-synapse generality.

**Decision rule:** if item 3 shows that Arbor or NESTML already covers this niche, putting the device
model work into that tool is cheaper than maintaining a separate simulator. **Status (2026-10-13):**
Arbor covers rows 1-5, 7 and 8 for cable cells (CPU probes; evidence doc §3.4). What it does not cover
is row 9, native device randomness at spike time and point neurons without a cable solver. So the case
for SuperNeuroABM now rests on row 9 at scale (probe shown; hardware experiment at scale to come) and on point-neuron cost against Arbor (to
be measured on GPUs).
