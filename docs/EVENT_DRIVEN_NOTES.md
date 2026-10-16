# Event-driven SuperNeuroABM: design notes and measurements

Branch `perf/event-driven` in both repos (created 2026-10-14 from `dev`). Working notes; the
numbers are from the laptop GPU (RTX 3000 Ada) unless stated.

## 1. Why: the per-tick cost is O(synapses), and one soma thread can be the whole tick

Two measurements decided the direction.

**Brunel, N = 12,500, K = 1,000, 30 Hz, dt = 1 ms (2026-09-17).** NEST GPU does ~343,000
spike deliveries per tick; we step 12,512,500 synapse agents. 36.5x more work for 40.7x the
time, so per unit of work we are within ~12 % of NEST and the whole gap is O(synapses) vs
O(spikes).

**Masquelier, 2,000 afferents onto one neuron, dt = 0.1 ms (2026-10-14).** Not launch-bound,
as `PERFORMANCE_NOTES.md` assumed: the tick is linear in the afferent count because the single
soma thread pulls all K afferent currents serially (~60-65 ns per load).

| afferents | as is | soma gather stubbed to one read | gather's share |
|---|---|---|---|
| 2,000 | 143.7 us/tick | 24.0 us/tick | 83 % |
| 20,000 | 1,338 us/tick | 45.0 us/tick | 97 % |

The remaining 24 us is three grid barriers, event stamping, one soma step and two
2,000-thread synapse phases (one GPU wave).

## 2. What "event-driven" has to mean here

Two separate changes, not one:

1. **Delivery is pushed, not pulled** (round 1). A synapse deposits its current into the post
   soma's inbox; the soma reads one value instead of looping over K afferents. Fixes the hub
   problem above and is the prerequisite for everything else.
2. **Synapses with no per-tick state are stepped only when woken** (round 2). An active set on
   its own gives ~nothing with the built-in synapses: Brunel's `tau_fall` is 10 ms against a
   33 ms ISI, so every synapse's current is non-zero on every tick (a floor of 1e-3 needs
   70 ms). The event path needs a delivery-only synapse kind with the kinetics on the soma
   (one per-receptor variable, the NEST layout, and what a pulse-deposits-charge device model
   is anyway). Synapses that genuinely carry per-tick state (drift, thermal, write disturb)
   stay agents, woken on events or on a slow clock.

The coworker's `new-step-functions` kernels (AdEx, Izh, triplet, suppression) are all
compatible: both somas are pull-gather somas (one loop to replace), both rules change weight
only at pre/post spikes and only decay traces in between.

## 3. Push prototype (no framework change) and the one bug it found

Scratchpad copies of the Masquelier kernels: the synapse does
`jit.atomic_add(internal_states, (post_soma_index, 4), I_synapse)` into a fifth soma slot, the
soma consumes and clears it.

| Masquelier, 2,000 afferents | pull | push | speedup | spikes |
|---|---|---|---|---|
| 10 s (100k ticks) | 14.2-14.4 s | 1.54-1.57 s (15.5 us/tick, 3 repeats) | 9.2x | 414 = 414 |
| 100 s (1M ticks) | 141 s | 13.96 s | 10.1x | 639 = 639 |
| 20,000 afferents, 10 s | 133.8 s | 7.3 s | 18x | |

Brian2 CPU: 5.2 s / 16.5 s; NEST GPU 11.0 s (10 s). Results are **bitwise identical** to the
pull run (spike times and weights). Two push runs were bitwise identical to each other.

**The bug.** The first push run fired 1-2 ticks early from the third spike on. Not float
order: re-associating the pull sum (even/odd partial sums) was bitwise identical. The
Masquelier STDP kernel (priority 101) zeroes the synapse current at a post spike *after* the
synapse kernel (priority 100) had already pushed it. Rule: **a pushed value must be the tick's
final value.** The EPSP reset now lives in the synapse kernel, where it belongs physically.
Diagnostic that found it: a soma computing both sums, driving with the pull sum, recording the
first tick where they differ.

## 4. Round 1 design: a reduce inbox as a breed-local array

Why not just let kernels write neighbour rows: an agent's state must stay written only by
itself, or step functions stop being trivially parallel. The sanctioned exception is an
**inbox**: other agents may deposit into it through commutative atomic reductions, into a
buffer the framework owns and swaps. This is the design every parallel ABM/graph engine
converges on (FLAME GPU 2 macro properties with layer boundaries; HASH next-step messages;
NEST GPU / GeNN `atomicAdd` delivery; PyG/DGL scatter-reduce).

SAGESim already has the two halves: breed-local arrays with per-breed double buffering and an
end-of-tick write-back. The inbox is a BLA with `reduce="sum"`:

```python
model.register_breed_local_array("synaptic_input", breed=[lif, izh, ...], shape_per_agent=(1,),
                                 reduce="sum")
# sender kernel (any breed that declares the array):
deposit_to_agent(synaptic_input, synaptic_input_idx, post_soma_index, 0, I_synapse)
# receiver kernel, next tick:
I = synaptic_input[synaptic_input_idx[agent_index]][0]
```

- deposits are `atomic_add` into the **write buffer**; the receiver reads the **read buffer**;
  the end-of-tick write-back becomes *copy then zero* (drain). Reader and writers never touch
  the same buffer, whatever the priorities.
- `deposit_to_agent` is the only sanctioned depositor; the write analysis counts it as a write
  of the array, and the transform rewrites its first argument to the write buffer.
- a BLA may span several breeds (all soma kinds share one inbox).
- a breed's step function declares which BLAs it takes (`breed_local_arrays=[...]`), so
  kernels that do not use the inbox keep their signature. Default None = all, as before.
- the host-side post-launch copy skips reduce BLAs (it would overwrite the read buffer with
  the drained zeros).

Known limits in round 1, each its own later round:
- `neighbor_visible=True` with `reduce` raises `NotImplementedError`: depositing into a ghost
  needs a reduce-back-to-owner exchange (the mirror of `exchange_ghost_data`). Deposits to a
  non-local target are dropped (`idx_map == -1`); SNN post-owns partitioning never produces
  one. Detection needs an injected error-flag argument.
- float atomics sum in arrival order; bit-exactness across runs is observed, not guaranteed.
  Deterministic mode = integer/fixed-point accumulation or a segmented ordered reduction.
- `sum` only; `max`/`min`/`or` are the same code path with a different atomic.

## 5. Round 1 in SuperNeuroABM (implemented 2026-10-14)

- `NeuromorphicModel(..., synaptic_input="pull" | "push")`; default `"pull"` in this round so
  no existing app changes behaviour. The inbox (`synaptic_input`, shape (2,) per soma, over
  every soma breed) is registered in `setup()` either way, so kernel signatures do not depend
  on the mode; the mode is a compile-time constant (`step_functions/synaptic_input.PUSH`, set
  in `setup()`, read like `single_exp.I_FLOOR`).
- A soma or synapse kernel takes part by ending its parameter list with
  `synaptic_input, synaptic_input_idx` (detected from the signature, so it works for
  built-ins, `register_*_type` and `soma_breed_info` alike). Somas call
  `synaptic_input_pushed()` / `read_synaptic_input(...)`, synapses
  `deliver_synaptic_current(...)` after their last write to the current. A kernel without
  the pair keeps its signature and pulls; in push mode a *synapse* kernel without it is
  refused at `setup()` (the soma would receive nothing). `load_from_adjacency` with more than
  one rank is refused in push mode (it releases post-owns, so a deposit could target a
  ghost).
- Converted: lif, izh, hg_lif, lif_soma_adaptive_thr; single_exp, double_exp (slot 0 fast,
  slot 1 slow), weighted_synapse. The generated `learning_rule_selector` declares no arrays.
  The coworker's AdEx/Izh (`new-step-functions`) need the same two-line change on merge.
- Benchmarks: `brunel_sna_local.py --synaptic-input push`; `dump_tensors_for_diff.py` with
  `SNA_SYNAPTIC_INPUT=push`.

### Verification so far

| check | result |
|---|---|
| SAGESim `tests/test_reduce_bla.py` (inbox: next-tick sum, resume, bystander breed, multi-breed, 4096 senders, validation) | 10 passed |
| SAGESim suite / MPI tests at 2 and 4 ranks | 87 passed / all ok |
| SNA suite, pull default | 216 passed, 1 skipped, 1 xfailed |
| SNA `tests/test_synaptic_input_push.py` (chain bit-exact, fan-in to round-off, weighted, resume, validation) | 6 passed |
| Masquelier 2,000 afferents, 10 s, framework push vs pull (app kernels with the two params and the EPSP reset in the synapse) | **bitwise identical**: 414 spikes, all weights |
| Brunel 2,000 / K=100 / 30 ticks tensor dump, push vs pull | every tensor and the spike counts bitwise identical except soma `internal_states`: 1,188 of 612,000 entries differ, max relative 3.3e-7 (float32 order of 100-term sums) |

| Brunel 12.5k / K=1000 / 100 ticks, `brunel_sna_local.py` | push 38,026 spikes / 36.69 Hz steady = pull; ticks 2-100 0.998 s -> 0.860 s (10.1 -> 8.7 ms/tick, -14 %) |
| 8-soma recurrent network, 1 vs 2 vs 4 ranks, push vs pull | bitwise identical after the placement fix below |

### A partitioning bug the two-rank run exposed

The incremental `create_synapse` path placed agents round-robin by id under MPI, not
post-owns as its comment said. In pull mode that is invisible (ghost synapse rows are
exchanged every tick); in push mode one recurrent synapse per soma sat on the other rank,
its post soma was a ghost there, and the deposit was dropped. The spike trains did not show
it (the test network fires refractory-locked); the membrane potentials did, by ~20 mV.
`create_synapse` now passes `rank=<post soma's rank>` to `create_agent_of_breed`, so the
incremental path is post-owns like the loaders. Side effect for pull mode: no ghost synapse
rows any more, only ghost pre-somas.

Lesson for the framework: a dropped deposit must become an error, not a silent zero. That
needs an error-flag argument injected into the kernels (same mechanism as `_seed`); it is
the first item of the ghost-reduce round.

### ... and a second one: a rank with no agents

Post-owns placement of a 2-soma test network on 4 ranks leaves ranks 2 and 3 empty, which
SAGESim never supported: `worker_coroutine` raised `KeyError` (no `_rank2agentid2agentidx`
entry), the launch had zero blocks, and the empty rank's property tensors were 1-D where
the shared kernel expected 2-D/3-D, so the other ranks hung in a collective. Fixed in
SAGESim: every rank gets a dict entry at construction, a launch is at least one block, and
the ranks agree each property's full trailing shape (`Allreduce` MAX) so an empty rank
allocates what the kernel was compiled for. `tests/test_empty_rank.py` runs 2 agents with
scalar, 2-D, 3-D and breed-local tensors at 1/2/4 ranks.

Brunel reading: the soma gather is a small share of a 12.5 M-synapse tick, so push alone
gives 14 %; the remaining 8.7 ms is the synapse phase, which is what round 2 removes.

## 6a. Round-2 go/no-go: two measurements on the 12.5 M-synapse Brunel network

Event-driven means two things, and they were measured separately so each half's payoff is
known before any framework work (scratchpad `delivery_floor.py`, `brunel_delivery/`):

| variant | per-synapse state | every synapse stepped | ms/tick |
|---|---|---|---|
| today, pull | yes | yes | 10.1 (100 ticks) / 9.85 (400 ticks) |
| round 1, push | yes | yes | 8.7 / 8.2 |
| **B. delivery-only synapses, current filtered on the soma** (SNA kernels, push, no wake list) | no | yes: each idle synapse reads its pre's spike slot and returns | **4.9 / 4.66** |
| **A. true event delivery** (standalone CuPy kernel on the real CSR: spiking neurons add `w` to each target's inbox, one LIF step per neuron) | no | no | **0.28** = delivery 0.074 + LIF 0.008 + spike compaction 0.197 (823 k deliveries/tick at 63.6 Hz; NEST GPU's whole tick is 0.19) |

So removing per-synapse state is worth 1.8-2.1x and not visiting idle synapses another
~10-15x: 20-30x in total is attainable, and the larger share is the wake list.

Equivalence of B: spike totals identical to pull through 60 ticks (8,631 and 21,555 at 12.5k)
and per-neuron identical on the 2,000-neuron dump; later the trajectories split (a running
soma-side sum rounds differently from 1,000 separately decayed currents). The 400-tick steady
rate is 29.7 Hz vs pull 31.4 Hz; a 1e-5 weight perturbation of pull alone moves it to
33.7 Hz, so that gap is trajectory-to-trajectory variation of this balanced network, not a
modelling difference. The soma-side current is NEST's `iaf_psc_exp` layout, i.e. closer to the
reference model than the per-synapse copy.

Diehl-Cook is not a showcase for the wake list: its STDP traces are per-synapse state and
its inputs are external synapses (a pre-trace has no agent to live on), so only the push
share (~10 %) applies without restructuring the app.

### A correction to round 1: the mode must live in data, not in a constant

The first version carried `synaptic_input="push"` as a module constant
(`synaptic_input.PUSH`) read by the kernels at compile time, like `single_exp.I_FLOOR`.
That is unsafe here: CuPy compiles a kernel at its *first launch*, and SAGESim reuses an
identical generated kernel across models, so in one process a pull model's compilation
served a later push model and its synapses never deposited (the full SNA suite caught it;
the earlier in-process pull-vs-push test could not, because the push model silently ran
as pull). The mode is now carried by the inbox's index map: in push mode every soma breed
is a member of the inbox, in pull mode it is registered with no members, so a soma tests
`synaptic_input_idx[agent_index] >= 0` (`has_inbox`) and a deposit into a row-less target
is a no-op. No constant, no signature change, any mix of models in one process.
`lif_psc_soma` (inbox-only) is not registered in pull mode.

## 7. Round 2: event-woken agents (implemented 2026-10-14)

### SAGESim: wake lists

- `Model.configure_wake(event_property)`: a width-2 property; an agent raises an event at
  tick t by writing > 0 into slot `t % 2` of its own row (SNA uses `output_spikes_tensor`).
- `Breed.register_step_func(..., wake_on_roles=[0])`: the breed's agents at that priority are
  stepped only on the tick after a neighbour in one of those slots raised an event, or after
  an explicit wake. A priority is woken as a whole (every breed stepping at it declares
  roles), so its loop runs over a queue instead of the index range.
- At setup the reverse adjacency is built once from the CSR: emitter row -> dependants
  through the declared roles, over local *and ghost* emitter rows (a ghost's event arrives
  through the ordinary property exchange, same one-tick delay as a local one).
- Per tick, in the pre-tick phase: every thread scans emitter rows; a raised slot reserves
  a queue segment with one atomic (`atomic_add(count, out_degree)`) and copies the row's
  dependants. The priority loop walks the queue; `wake_stamp[row]` (last tick the row ran,
  -1 never) makes a row queued twice run once and lets a kernel catch up state in closed
  form after sleeping. Thread 0 zeroes the counts at tick end.
- Explicit wakes from injected code: `base = atomic_add(wake_count_<p>, 0, 1);
  wake_queue_<p>[base] = row` (SNA does this for input events in its pre-tick code).
- Queue capacity = all role wakes + one explicit wake per agent (worst case); the reverse
  adjacency costs one int32 per (woken agent, role).
- Also fixed on the way: one generated kernel file per distinct generated text
  (`step_func_code_<hash>.py`). CuPy compiles a kernel at its first launch from its
  function's source lines, so two models with different generated code set up before
  either ran would compile the first from the second's file.
- `tests/test_wake_list.py` (emitters with periods, followers woken through role 0 / both
  roles, twins, resume, validation), in the MPI runner at 1/2/4 ranks.

### SuperNeuroABM: `synapse_stepping="event"`

- Needs `synaptic_input="push"`. Two new built-in kinds carry the model that makes a synapse
  stateless: `delivery_synapse` (deposits `spike * scale * weight` when its pre fired, else
  returns; event-capable) and `lif_psc_soma` (lif_soma with the exponential synaptic current
  filtered on the soma, `tau_syn` hyperparameter, `I_syn` state -- NEST's `iaf_psc_exp`
  layout). In event mode the default synapse kinds are restricted to event-capable ones and
  `register_synapse_type` needs `event_capable=True`.
- Synapse breeds are registered with `wake_on_roles=[0]` (the pre-soma); injected input
  events wake their synapse from the pre-tick stamping loop. Learning rules (priority 101)
  still run clocked in this version: a rule with per-synapse traces needs decay-on-wake
  (round 2b) before it can be woken.
- `tests/test_event_stepping.py`: clocked vs event on an 8-soma recurrent network with
  external drive is bit-identical (spikes and state); resume; validation. In the MPI runner.
- Benchmark: `brunel_sna_local.py --synaptic-input push --synapse-stepping event
  --soma-breed lif_psc_soma --synapse-breed delivery_synapse --no-learning-rules`.

### Round 2 measured (Brunel 12.5k / K=1000 / 400 ticks, no learning rules, laptop GPU)

| mode | ms/tick | vs pull |
|---|---|---|
| pull (lif_soma + single_exp) | 7.66 | 1 |
| round 1: push, same kinds | 5.56 | 1.4x |
| push, clocked delivery_synapse + lif_psc_soma | 2.93 | 2.6x |
| push, event-woken, synapse rows in file order | 2.35 -> 2.11 with block-cooperative expansion | 3.6x |
| **push, event-woken, synapse rows ordered by pre-soma** (`--synapse-order pre`) | **0.92** | **8.3x** |
| push, event-woken, loader lays rows out by pre (now the default in event mode, no flag) | 1.10 (0.95 with the file pre-sorted; the difference is run noise) | 7-8x |
| event mode, silent network (fixed cost: grid barriers, soma, stamping, inbox write-back, queue reset) | 0.69 -> 0.63 after merging two barriers (4 per tick) | |
| standalone delivery kernel (section 6a) | 0.28 | |
| NEST GPU | 0.19 | |

Spike totals identical across pull / clocked delivery / event at 30 and 60 ticks (8,631 /
21,555); the 400-tick totals differ by trajectory divergence only (section 6a).

What the last two rows taught: with rows in file order each woken synapse touched ~10
scattered 64-byte sectors (queue entry, stamp, breed, CSR, parameters, inbox index, atomic) =
~256 MB/tick of random traffic, i.e. the 1.4 ms activity cost was DRAM-bound, not
latency-bound (4 blocks/SM instead of 2 changed nothing; 8 deadlocks the grid barrier as
expected). Ordering synapse rows by pre makes an emitter's woken synapses a contiguous run,
so the queue loop's reads coalesce and the activity cost falls to ~0.23 ms, at the floor.
The fused single-launch path is slower than per-tick launches here (3.5 vs 2.35 ms), so host
launch overhead is not the remaining gap; the fixed 0.69 ms is.

Done since: (1) the post-owns loaders (columnar and record) lay synapse rows out by pre in
event mode; (2) barriers merged, 6 -> 4 per tick (expansion needs none unless the first
priority is woken; the queue reset sits under the write-back barrier); (3) **decay-on-wake
learning rules**: the four built-in pair-wise rules keep the tick they last ran in
`learning_internal_states[3]` and decay their traces with `trace_decay(dt, tau, n)` =
`(1-dt/tau)^n` (`n == 1` is the single multiply they always did, so clocked stepping is
bit-identical); in event mode the learning-rule selector is woken on roles 0 and 1 (pre or
post fired) and by input events, and `register_learning_rule(..., event_capable=True)` is
required for custom rules. Clocked vs event with STDP on an 8-soma recurrent network: same
spikes, weights equal to 1e-5 (power vs repeated multiply). Found on the way: the wake
dedupe stamp must be per priority, not per row -- the same synapse legitimately runs at
priority 100 (delivery) and 101 (rule) in one tick. The coworker's triplet and suppression
rules need the same two-line change (last-tick slot, `trace_decay`) to be event-capable.
(3b) **Wake eligibility**: `Model.set_wake_eligibility(priority, fn)` lets a model exclude
agents from a woken priority's reverse adjacency; SuperNeuroABM uses it so only synapses
created with a learning rule are woken for the rule phase (Brunel's synapses carry none, and
with the four built-in rules registered every pre/post spike used to queue ~800 k rule
no-ops per tick). Explicit wakes from input events are not filtered, so a wake queue is
sized for every agent of the woken breeds plus all role wakes. Brunel with the four
built-in rules registered: event 1.92 -> **0.96 ms/tick**, i.e. the same as with no rules.
Also fixed: the synapse kernels now tolerate a synapse with no post slot (an adjacency
file may list only the pre); round 1 had made them read slot 1 unconditionally, which
read past the CSR segment, and `deposit_to_agent` indexed `idx_map[target]` before
checking the sign, so the -1 sentinel read the element before the map (undefined; it
happened to be benign until it faulted in a 2-rank `load_from_adjacency` test). The
sentinel is now tested in SAGESim. Note for kernel authors: the CSR transform rewrites
exactly one alias of `locations[agent_index]` per kernel.
Still to do: (4) the reverse ghost exchange for partition-independent push.

### Why NEST GPU is still ~3-5x faster per tick, and what of it is structural

The algorithm is now the same: spiking neuron -> efferent list -> atomic add into the
target's input -> one neuron update per step. The remaining difference is the execution
model and what an "agent" carries:

1. **Execution model.** SAGESim runs one persistent generated kernel per launch with
   *software* grid barriers (an atomic counter every block spins on) between phases, four
   per tick, on only 2-4 blocks per SM (co-residency). NEST GPU launches a few small
   ordinary kernels per step, each sized to its work, and lets kernel boundaries do the
   synchronisation. Our silent-network cost of 0.63 ms/tick is this: barriers plus the
   clocked phases (soma, input stamping, inbox write-back, queue reset) run by ~9-18k
   persistent threads, plus the per-launch host path.
2. **What a synapse is.** NEST's synapse is a 12-byte record read once by the delivery
   kernel; it has no identity of its own. Ours is an agent: the queue holds its row index,
   the loop stamps it (`atomic_exch`) so a doubly-woken agent runs once, dispatches on its
   breed, resolves pre/post through the CSR and parameters through the interning table,
   then deposits. That is ~6 memory touches per delivery against NEST's ~2, and it is the
   price of synapses that can carry arbitrary per-synapse models and learning rules --
   the capability the positioning rests on. With rows laid out by pre those touches
   coalesce, which is why the activity cost dropped from 1.4 to ~0.3 ms.
3. **What a neuron step writes.** Our soma writes three state slots, a history slot and the
   spike slot through generic padded property rows; NEST updates a struct-of-arrays with
   exactly its state.

So: the activity-dependent part is within ~1.5x of the floor; the fixed 0.63 ms is the
agent framework's execution model. Closing that would mean launching per-phase kernels
(or CUDA graphs) instead of the fused persistent kernel -- a different SAGESim execution
backend, not a tuning item -- and is a decision, not a next step.

**Correction (2026-10-15, section 15):** the paragraph above is wrong about where the
0.63 ms goes. Measured apart, the fused kernel's own fixed cost on a silent Brunel tick is
0.12 ms (barriers 0.007, soma phase 0.013, wake-expansion scan 0.08); the other ~0.5 ms
is the host path of launching once per tick, which the benchmark's verbose timers force
and the single-GPU user path (all ticks in one launch) does not pay. "Fused is slower
than per-tick launches" in the round-2 table came from dividing a wall time that
included the 4-6 s buffer build by the tick count. Brunel firing, fused: 0.28 ms/tick.

## 6. Expected gains (estimates, to be replaced by measurements)

| network | before | after round 1 (push), measured | after round 2 (wake lists), estimate |
|---|---|---|---|
| Masquelier 2,000, 10 s | 143.7 us/tick (14.3 s) | 15.5 us/tick (1.54-1.57 s, 3 repeats; 9.2x), bitwise identical | ~10-15 us: fixed cost dominates |
| Masquelier 2,000, 100 s | 141 s | 13.96 s (10.1x), same 639 spikes | |
| Masquelier 20,000, 10 s | 1,338 us/tick | 73 us/tick (18x) | |
| Brunel 12.5k / K=1000 / 100 ticks | 10.1 ms/tick (this run) | 8.7 ms/tick (-14 %), same spikes | 0.4-0.8 ms (10-20x); NEST GPU 0.19 ms |
| Diehl-Cook | 186 ms/presentation | not measured | needs soma-side STDP traces |

## 8. Could push/event be the only mode? (2026-10-15)

Synapses are agents in every mode: a synapse is a row of the synapse breed with its own id,
weight, learning state, step function and rule. Event mode only changes which rows SAGESim
visits each tick (the wake list). What the synapse gives up in event mode is its own current
filter, which moves to the soma (`lif_psc_soma`, NEST's `iaf_psc_exp` layout).

Everything in the repo and the coworker branches is expressible in push/event: the four
built-in rules (decay-on-wake), the Daffodil memristive rule and the low-resolution synapse
on `origin/memristor_step_function` (both change state only at pre/post spikes), AdEx/Izh
somas (inbox read), Masquelier and DC apps (per-synapse traces with wake-on-input). The
`synaptic_delay` parameter is unused in every mode today (fixed 1-tick delay in
`get_soma_spike`), so delays are not a pull advantage either; in push they are a delay ring
on the inbox, as in NEST.

What pull/clocked still buys, measured or by construction:

| property | pull/clocked | push (clocked or event) |
|---|---|---|
| run-to-run reproducibility | bitwise (fixed summation order) | **not bitwise**: atomic-add order varies. Brunel 2000/K=100, 400 ticks, two runs: push/clocked 1,381 of 2,000 somas differ in `v` (max 2e-3); event identical with Brunel's two uniform weights, but with per-synapse jittered weights 1,541 somas differ (max 1e-3). Spike counts matched in all cases at this length; a chaotic net will diverge later. NEST GPU and GeNN have the same property. Fix if wanted: ordered segmented reduction instead of atomics (SAGESim `reduce="sum", ordered=True`). |
| partition freedom on multi-rank | any (synapse reads ghost spike; soma reads ghost synapse) | any, since step 3 (section 11): a deposit into a ghost soma is summed back to its owner |
| soma-side integration | soma iterates its afferents: any function of them | sum per inbox channel only (2 channels today; should be configurable, Cora needs 2 taus, AMPA/NMDA/GABA 3-4) |
| per-synapse current readout | `I_synapse` per synapse recorded | per-soma `I_syn` per channel |
| synapse types with continuous per-tick state not driven by spikes | stepped every tick | refused in event mode (`event_capable=False`); would need a per-type "always wake" or a separate clocked breed |

Script: `$S/repro_check.py <pull|push|event> out.npz [jitter]` (scratchpad).

### Masquelier on the event path (measured 2026-10-15)

Scratchpad port (`$S/masq_fw_event/`, ns-applications untouched): built-in
`delivery_synapse` woken by the injected input events, the summed EPSP `x` on the soma
with the post-spike reset applied there (`masq_psc_hg_lif_soma.py`: the per-synapse
recursion summed over afferents), and the nearest-neighbour STDP rule with
decay-on-wake (`exp(-n dt/tau)`, `last_tick` slot). 2000 afferents, dt 0.1 ms, fused
single-worker path, laptop GPU:

| run | pull/clocked | push/clocked | push/event | Brian2 CPU | NEST GPU |
|---|---|---|---|---|---|
| 10 s (100k ticks) | 14.3 s | 1.66-1.68 s | **1.07-1.08 s** | 5.2 s | 11.0 s |
| 100 s (1M ticks) | 141 s | 15.4 s | **9.46 s** | 16.5 s | - |

Same 414 (10 s) / 639 (100 s) output spikes at identical ticks as push; final weights
within 7e-7 (float32, different summation path). The first run after an idle GPU is
~2x slower (clocks ramp); the table is the steady state. Per tick the event path is
~10 µs: at 2000 synapses the clocked synapse pass was already one wave, so the gain
over push is the skipped idle rows and the smaller rule phase, not O(synapses).

## 9. Plan: push/event as the single SNN path (proposed 2026-10-15)

Decisions this plan assumes (user to confirm): synapses stay agents; push/event becomes
the only SuperNeuroABM path and pull stays in SAGESim as the general neighbour read;
atomic-add nondeterminism is accepted and documented (fixed-point inbox = later option).

| step | repo | work | effort | gate |
|---|---|---|---|---|
| 1 | SNA | N-channel psc soma (tau per channel, N from config) + `channel` hyperparameter on `delivery_synapse`, carried by `create_synapse` and the loaders | 1 d | two-tau test net: clocked single_exp == event psc spikes |
| 2 | SGNN, ns-apps | SGNN builder sets channel per group, two-tau soma config; Masquelier: drop the three scratch kernels into the duplicate (back up `duplicates/` first) | 1 d | Cora 699/221 (small diffs allowed, not bitwise); Masquelier 414/639 spikes |
| 3 | SAGESim | reverse ghost exchange for reduce arrays: deposits into ghost rows reduced back to the owner; lift the `neighbor_visible` restriction; drop the `load_from_adjacency` refusal | 2-3 d | push == pull bitwise at 1/2/3/4 ranks with synapses on the PRE rank; 106 + MPI |
| 4 | SNA | push/event default; remove soma pull loops, kinetic synapses (single/double_exp, weighted -> delivery + channels), the two switches and `has_inbox` branches; tests and docs | 2 d | 224-equivalent suite + MPI 1/2/3/4; Brunel 0.96 ms/tick |
| 5 | ns-apps | Diehl-Cook: ge/gi as two soma channels, per-synapse traces woken by input events | 1-2 d | 73.03% on the control subset (never the full 40k) |
| 6 | SNA | coworker kernels on merge: AdEx/Izh read the inbox, triplet/suppression rules decay-on-wake, memristor kernels off the `globals` signature | 1 d | their configs run |
| 7 | SNA/SAGESim | in-place reset (keep GPU buffers, restore initial columns on device) | 1 d | SGNN per-paper 45 -> ~20 ms |
| 8 | docs | re-time Brunel/Masquelier/SGNN vs NEST GPU/Brian2; PERFORMANCE_NOTES Masquelier correction; WHY evidence (FLAME GPU probe) | 1 d | tables updated |

About 10-12 working days. Steps 1-2 first (they unblock the apps on one GPU and are
measurable within two days); step 3 before step 4 so nothing is deleted while push is
still partition-dependent; 5-8 in any order. Every step ends with the standard gate
(both suites, MPI 1/2/3/4, bitwise dump where applicable) and a commit by the user.

## 10. Step 1 done: N-channel synaptic input (2026-10-15)

`NeuromorphicModel(synaptic_channels=K)` sets the inbox width (default 2, unchanged).
`delivery_synapse` gained a `channel` hyperparameter (the slot it deposits into);
`lif_psc_soma` filters each slot with its own `tau_syn_<c>` into `I_syn_<c>` and sums
them (NEST `iaf_psc_exp_multisynapse`, GeNN's per-synapse-group accumulator). Setup
refuses a psc config missing a channel's tau/state and a channel outside 0..K-1
(`_check_synaptic_channels`, also per-agent overrides and loader columns via
`_intern_combo`). Tests: `tests/test_synaptic_channels.py` (9: two-tau net equals the
per-synapse single_exp model's spikes with v to 1e-5; clocked == event bitwise; swapped
channels differ; the four refusals; one-channel model), in the MPI runner at 1/2/3/4.
`test_event_stepping.test_reset_then_rerun` pins that reset() clears wake state.

Verification: SNA 233 passed, MPI 1/2/3/4 ok. Brunel 12.5k event 400 ticks: 162,561
spikes with the old and the new kernels; per tick old 1.10-1.25 vs new 1.19-1.30 ms
(same GPU state; the 0.96 earlier was a faster clock state). Masquelier event 10 s:
same 414 spikes, weights bitwise identical. Brunel 2000/K=100 400 ticks: spike counts and
I_syn identical to the pre-change dump; v differs in 477/2000 somas by <=1e-3 (2.6e-6
relative): the compiler contracts the unchanged membrane arithmetic differently around
the new loop; the new kernel is itself run-to-run bitwise.

### Step 2a: SGNN Cora on push/event (2026-10-15)

`~/superneuroabm_sgnn` (branch distributed_model_creation): `configs/cora/sna_cora_event_config.yaml`
(same numbers; `lif_psc_soma`, `delivery_synapse`, `synaptic_input: push`,
`synapse_stepping: event`, `synaptic_channels: 2`, `input_synapse_breed`),
`build_snn_from_data.assign_channels` (per post soma, distinct taus in first-appearance
order -> channels; soma gets `tau_syn_<c>`; refuses more taus than channels; YAML 1.1
gives `1e-3` as a string, so taus are compared as floats), runner reads the three mode
keys from the config. Partition `partitions/cora_event_np1`. Result 699/1000 correct,
221 ties == clocked. Time 1000 papers back to back: clocked 58.5 s, event 59.5 s (both
~50 ms/paper of which the post-reset rebuild dominates; the 44.7 s earlier was a faster
GPU state). As predicted: event mode is not where SGNN's time goes (in-place reset is).

### Step 2b: Masquelier duplicate on push/event (2026-10-15)

`~/ns-applications/duplicates/masquelier_2008`: `run_experiment_hg.py --stepping event`
(default stays clocked), kernels `masq_psc_hg_lif_soma.py` and
`exp_pair_wise_stdp_bounded_nn_ev.py`, event breeds added to `masquelier_config.yaml`
(sources backed up to the scratchpad first). Full 100 s run, continuous mode, both
steppings back to back: 671 spikes at identical ticks, hit rate 96.0 %, false alarms
4.56 Hz, final weights within 2.2e-6; elapsed 147.1 s clocked vs 13.7 s event (10.7x).

## 11. Step 3 done: reduce-back-to-owner exchange (2026-10-15)

SAGESim: a reduce breed-local array may now be `neighbor_visible` (and multi-breed).
Ghost rows get rows in both buffers (a ghost of still-unknown breed too: loader-built
partitions learn remote breeds only when the communication maps are built); the
end-of-tick drain already covered ghost rows. `CommunicationManager._reverse_exchange`
sends every ghost row of a reduce array to its owner and the owner adds the arrivals
into its own row before its agents read, using the forward request maps with the roles
swapped (one extra message per peer, tag 2, no new topology discovery). Reduce arrays
are excluded from the forward owner->ghost copy. Two latent bugs fixed on the way: the
breed-local exchange maps were only built on ranks that needed ghosts themselves (a
serving-only rank never packed its arrays: wrong values or a shorter message), and a
model whose only visible data is a breed-local array never exchanged at all.
`tests/test_reduce_bla.py` (15: cross-rank senders, mixed local+remote senders, two
receiver breeds, chunked launches) passes at 1/2/3/4 ranks.

SuperNeuroABM: the soma inbox is registered `neighbor_visible` in push mode and
`load_from_adjacency` no longer refuses push at >1 rank. `test_load_from_adjacency`
gained the split-synapse case in push/clocked (vs the pull reference) and push/event
(delivery synapse on the pre's rank, psc soma on the other): same spike trains at 1 and
2 ranks. Correctness no longer depends on where a synapse lives; post-owns stays the
default placement because it keeps deposits local (no reverse traffic).

Found by the new tests: an adjacency file may list a synapse with its pre only (the
external-input synapse `[-1]`), which pull mode never needed, but a synapse must know
its post to deposit (push) and to see post spikes (rules). `load_from_adjacency` now
fills the missing post slot from the local soma that names the synapse as incoming; a
synapse whose post is on another rank must list it. Also: a whole-network file loaded
on every rank of an MPI run makes every rank an owner, so a collective getter reports
each spike once per rank; a reference model in an MPI test is built with create_*.

## 12. Step 4 done: one path (2026-10-15)

`NeuromorphicModel(synapse_stepping="event")` is the default and the only input path:
`synaptic_input` is gone (synapses always deposit into the soma inbox), the clocked
kinetic synapses `single_exp_synapse`, `double_exp_synapse` and `weighted_synapse` are
deleted, and every built-in soma (`lif_soma`, `izh_soma`, `hg_lif_soma`,
`lif_soma_adaptive_thr`) filters its synaptic input on the soma with one exponential
per inbox channel (`step_functions/synaptic_input.synaptic_current`; `tau_syn_<c>` /
`I_syn_<c>` in every soma config). `lif_psc_soma` became `lif_soma`. The one built-in
synapse is `delivery_synapse` (`config_0` weight 14, `config_1` weight 100, the former
weighted_synapse's). `synapse_stepping="clocked"` stays as the verification arm (same
kernels, every synapse every tick) and for a custom synapse type with continuous
per-tick state; a custom learning rule must be registered `event_capable=True`
(decay-on-wake, see tests/step_funcs/exp_pair_wise_stdp_bounded_nn.py for the
conversion). A kernel that reads `has_inbox` still compiles (always true for a soma).

Equivalences kept: the former weighted_synapse == delivery into a channel whose
`tau_syn_c == dt` (decay factor 0); the former single_exp pairing == `tau_syn_c` set to
its `tau_fall`; the stored spike-time baselines of test_synapse_and_soma_models (lif and
izh with the old single_exp synapse) still match to the tick.

Verification: SNA 236 passed; MPI 1/2/3/4 ok (test_synaptic_channels now checks each
channel against the closed form w(1-dt/tau)^n and superposition, since the per-synapse
reference is gone). SGNN Cora: both configs rebuilt on lif_soma + delivery_synapse
(`assign_channels` now always runs), 699/1000 correct, 221 ties, main config 49.9 s,
event config 55.4 s (GPU clock noise). Masquelier duplicate event run: 671 spikes
identical to the step-2 run, 13.4 s. Brunel: see the table below.

Partition files written before this change name `single_exp_synapse` and must be
regenerated (the Brunel benchmark cache, `brunel_partition` now writes
`delivery_synapse`). Still pending, application-side: the tutorials
(`tutorials/00_simple_heterogenous_network.ipynb`, `01_superneuroabm_digits.ipynb`,
`digits_config.yaml`) and the Diehl-Cook duplicate still name the old synapse kinds.

| Brunel 12.5k / K=1000, 400 ticks, rules registered, regenerated partition | spikes | ms/tick |
|---|---|---|
| default (`synapse_stepping="event"`) | 162,561 (same as steps 1-3) | 1.28 |
| `--synapse-stepping clocked` (same kernels, every synapse every tick) | 153,609 | 4.91 |

The two steppings deposit in a different thread order, so after a few hundred ticks the
chaotic network has diverged (section 8; they agree to the bit on the test networks and
for the first tens of ticks here). For reference the removed pull path was 7.66 ms/tick.

### Step 5 done: tutorials and Diehl-Cook (2026-10-15)

Tutorials 00 and 01 run on `delivery_synapse` with the time constant on the soma and were
re-executed in place (nbconvert + ipykernel are now in the dev env). Tutorial 01's custom
LIF reads the inbox through `synaptic_current`; its custom STDP rule decays traces on wake
and is registered `event_capable=True`. One semantic detail mattered: the rule cuts a trace
to zero once it is below `a/5`, and stepped every tick that cut happens on the first tick
below the window. Decaying on wake and cutting afterwards changed the learned weights (max
0.64 -> 0.70); applying the cut to the trace's value on the tick before the wake, then
again after the update, reproduces 0.64 exactly. A custom event-capable rule must preserve
per-tick side effects of this kind, not only the exponential decay.

Diehl-Cook duplicate (ns-applications): conductances `ge`/`gi` are two channels on
`dc_cond_soma`, `dc_cond_synapse` deposits its weight and keeps only `pre_arrived`. It
runs `synapse_stepping="clocked"` on purpose: its kernels read the pre spike of the same
tick to reproduce Brian 1's one-tick recurrent hop, and an event-woken synapse runs the
tick after its pre fires (the exc -> inh -> exc loop would take 2 ms instead of 1 ms). A
SAGESim option to wake dependants right after the emitting priority of the same tick would
lift that; noted, not built. Sanity run 300/300: 36.33 %, retries 26/25/46, identical to
before; 2.6x faster (65 s whole run; the exc soma no longer gathers 883 afferents serially).
`test_brian1_semantics.py` passes against its CPU reference (|dv| 5e-5 mV).

## 13. Step 7: in-place reset (2026-10-15)

`reset()` used to free the device buffers; the next `simulate()` rebuilt them (index map,
CSR, interning, uploads): on Cora that rebuild was the largest single item per paper.
SAGESim's `reset(release_buffers=False)` now keeps the buffers and marks them
`pending_reset`; `_flush_pending_reset()` uploads the (reset) host columns over the local
rows and their write buffers, zeroes every breed-local array (the inbox starts empty) and
clears the wake stamps and queue counts. Interned properties are untouched (read-only on
the device). It runs before the next launch, before the next device read, and at the end
of `NeuromorphicModel.reset()` once `_reset_agents` has rewritten the host columns, so
`reset(retain_parameters=...)` is in place by default and `release_buffers=True` keeps
the old path. Alongside, `_device_dirty` (set at every launch, cleared by a flush) makes
the host/device authority explicit: a host write while the device is dirty first syncs
the written properties device->host (`_ensure_host_current`), lands, and schedules an
in-place refresh, so the `set_hyperparameters` guard that forced a `reset()` first is
gone (docs/CPU_GPU_DATA_FLOW.md section 5). Tests: SAGESim `tests/test_reset_in_place.py`
(5, in the MPI runner), SNA `test_event_stepping.test_in_place_reset_equals_rebuild`
(rebuild and in-place paths give the same second run to the bit, learned weights kept).

Measured on Cora (2,715 somas, 32,836 synapses, 20 ticks per paper, profile over 20
papers): the in-place reset alone changed nothing (53.1 s per 1000 papers vs 51.3 s with
`release_buffers=True`), because the rebuild was not where the time went. The profile
showed `_prepare_kernel_extras` at 33 ms per launch: the spike-record mask was rebuilt
after every reset with one device write per soma. Replaced by one scatter. Per paper now:
simulate 2.6 ms (20 ticks), reset 3.5 ms, readout 0.4 ms, i.e. ~6.5 ms against ~45 ms
before (the 1000-paper total is in the gate log below). Lesson re-learned: profile the
loop, not the step one expects to be slow.
Gate (2026-10-15): SAGESim 115 passed + MPI 1/2/3/4; SNA 238 passed + MPI 1/2/3/4; Cora
1000 papers **7.7 s** (was 50-58 s), 699/1000 correct, 221 ties; Diehl-Cook sanity 36.33 %.

## 14. Step 6: the coworker's step functions merged (2026-10-15)

`origin/new-step-functions` (27e2003: canonical Izhikevich rewrite with the 20 figure-1
panel configs, AdEx soma with 7 firing-pattern configs, Pfister-Gerstner triplet STDP,
Froemke-Dan suppression STDP) merged into perf/event-driven and brought onto the single
input path:

* `izh_soma` is now the canonical `0.04 v^2 + 5 v + 140 - u + I` model in the paper's
  units (the old `k, vthr, C, ...` form is gone; configs that used it must move to the
  panel configs, e.g. tutorial 00 drives panel A with `I_in: 14` instead of the old 500).
  Both somas take their input from the inbox (`synaptic_current`, `tau_syn_<c>` /
  `I_syn_<c>` added to every config; the soma records the summed current at [2]).
* Both rules decay their traces on wake (`trace_decay`, `last_tick` slot) and are
  built-in event-capable rules with ids 4 (`triplet_stdp`) and 5 (`suppression_stdp`);
  the first user rule is therefore 6 (tests, tutorial 01 and CUSTOM_COMPONENTS updated).
  The suppression factors use stored spike ticks, so waking changes nothing there.
* Tests: `tests/test_new_step_functions.py` (10: each soma fires under its documented
  drive and on delivered input, each rule learns, clocked == event spikes with weights to
  1e-4, rules on both new somas, stable ids). The Izhikevich spike-time baseline of
  test_synapse_and_soma_models was regenerated for the new model (lif baselines unchanged).
  Tutorials re-executed: 00 (Izhikevich panel A, 21 spikes, STDP weight 0.50 -> 0.75),
  01 unchanged (weight range 0.64, custom rule id 6).

### Masquelier 450 s on the authors' original input, event stepping (2026-10-15)

`ns-applications/duplicates/masquelier_2008/run_hg_input.py --stepping event` (the runner
gained the switch; its clocked branch now sets `synapse_stepping="clocked"` explicitly):
57,721,830 input spikes, 4,500,000 ticks. Simulate **71.7 s** (16 us/tick; 12.7 us/tick in
steady state, the first 50 s chunk carries the JIT and the buffer build), build + inject
5.8 s, 80.0 s wall for the whole script including loading the 920 MB input dump.
Previously 647 s (144 us/tick) on the clocked kernels: 9.0x. The authors' own Brian2
`cpp_standalone` run of the same experiment takes 79.7 s. Same 2,982 output spikes.

### Brunel, ours vs NEST GPU, back to back (2026-10-15, laptop GPU, 12.5k / K=1000 / 400 ticks)

| | SuperNeuroABM (event) | NEST GPU 2.0 |
|---|---|---|
| per tick, steady state | 1.22 ms with one launch per tick (ticks 2-400: 0.487 s; the benchmark's verbose timers force this path); **0.28 ms fused**, the path a single-GPU user gets (section 15, measured with the build excluded) | 0.20 ms (ticks 11-400: 0.0786 s by the script's timer, GPU idle at return; NEST GPU's own printed "Simulation time" of 0.201 s is not this: it is wall time since the end of `Calibrate()`, and ~0.13 s of it was the benchmark script's own work between the calls, section 15) |
| first tick / calibrate | 6.25 s (device buffer build) | 0.05 s |
| construction | 25.2 s load + 3.8 s setup | 0.03 s connect |
| whole run | 46.9 s | 1.85 s |
| firing rate | 32.5 Hz (33.5 steady) | 28.1 Hz |

## 15. The fixed per-tick cost, decomposed (2026-10-15)

The question was where the "fixed" ~0.6 ms of a Brunel tick goes and what of it can be
cut. Measured on the laptop GPU, build excluded (`simulate(1)` first, then the timed call),
event stepping, no learning rules. Probes in the session scratchpad (`fixed_cost/`).

| probe | one launch for all ticks (fused) | one launch per tick |
|---|---|---|
| 1 soma, no synapses: barriers, tick loop, inbox write-back | 0.007 ms | 0.525 ms |
| 12,500 `lif_soma`, no synapses: + soma phase | 0.020 ms | 0.521 ms |
| 12,500 somas + a 12,500-synapse ring: + wake-expansion scan over every emitter | 0.102 ms | 0.708 ms |
| Brunel K=100, silent | 0.108 ms | |
| Brunel K=1000, silent | 0.122 ms | 0.672 ms |
| Brunel K=1000, firing 33 Hz (162,561 spikes, same as every earlier run) | **0.279 ms** | 0.882 ms |

What the table says:

1. **The kernel's own fixed cost is 0.12 ms per tick, not 0.6**, and it does not depend on
   K. Of it, ~0.08 ms is the wake-expansion scan: every emitter row (12,500 somas) is
   tested for a raised event by thread 0 of a block, serially, ~174 per block on 72
   blocks, every tick, silent or not. The soma phase is 0.013 ms, the four software grid
   barriers plus the tick loop 0.007, the inbox write-back and queue reset the rest.
2. **The other ~0.5 ms is the host path of one launch**, paid once per `simulate()`
   window: cupyx.jit's `__call__` re-derives a type signature for all ~60 kernel
   arguments on every call (0.26 ms; the cupy driver launch underneath is 0.022 ms), the
   spike-count read-back is 0.044 ms, three device scalar resets ~0.03, and the
   per-launch bookkeeping and timing prints ~0.15. On one GPU the user path fuses all
   ticks into one launch and never pays it; a multi-rank run pays it once per exchange
   window (every tick today).
3. **The earlier headline, 1.22-1.28 ms/tick, was the per-tick-launch number**: the
   benchmark turns `verbose_timing` on for its first-tick sub-timers, and that forces one
   launch per tick. Fused, Brunel runs at 0.28 ms/tick: 1.4x NEST GPU's 0.20. The
   round-2 statement that the fused path was *slower* divided a wall time that included
   the 4-6 s buffer build by the tick count; `brunel_sna_local.py --fused` now times the
   first tick apart.
   The earlier "0.52 ms/tick by NEST GPU's own print" was a misreading of NEST GPU's
   `Simulation time:` line. Instrumented run (unbuffered C stdout, timestamps around
   each call, `deviceSynchronize` after each return): that line is `end_real_time_ -
   build_real_time_`, i.e. wall time since the END OF `Calibrate()`, printed cumulatively
   after every `Simulate()`, so it also counts whatever the host did between the calls.
   In the benchmark that is ~0.13 s of `PhaseLog.add` (an `nvidia-smi` call per phase)
   against 0.077 s inside the two `Simulate()` calls; the device is idle when `Simulate()`
   returns (sync 0.1 ms). The script's timer around `Simulate(390 ms)` is the number:
   0.075-0.080 s, 0.19-0.20 ms/tick.
4. Activity cost at 33 Hz, fused: 0.28 - 0.12 = 0.16 ms for ~420k deliveries per tick.

Where it can be cut, in order of payoff:

- **Compaction of fired emitters (SAGESim, general).** The emitter breed's step already
  tests its event slot in the generated post-step code (spike recording); append the row
  to a fired list there and let the expansion stride over the fired list instead of every
  emitter. O(fired) instead of O(emitters): ~0.08 -> ~0.01 ms on Brunel (tick 0.28 ->
  ~0.21), and it removes a term that is 6 ms/tick at a million emitters per GPU. Any ABM
  with event wake benefits. About one day with the bitwise gate (same spikes).
- **Launch path for the per-window case (SAGESim, general).** Keep the compiled cupy
  `Function` from the first call and launch it directly with the arguments already
  typed (-0.24 ms); fold the three scalar resets into one small kernel and read the spike
  count with the same copy (-0.05); skip the timing bookkeeping unless verbose (-0.1).
  ~0.5 -> ~0.15 ms per launch; matters for multi-GPU runs, where a launch per exchange
  window is unavoidable. NEST's answer to the same cost is to exchange every min-delay
  steps rather than every step; ours would be a buffered-delay exchange, a larger change.
- **Spike record sizing (SuperNeuroABM, done here).** The record was sized by agents x
  ticks / 100, and agents include synapses: a fused 400-tick Brunel launch allocated and
  NaN-filled 400 MB for 162k spikes. Now somas x ticks / 4 (10 MB).

Found on the way: `--no-learning-rules`, which lets the framework intern the soma
parameters, had been broken since step 4 made every soma read its parameters through the
`synaptic_current` helper: the generator's seed / logical_ids injection handed a helper
without an `agent_ids` parameter one argument more than its rewritten definition took
("takes 10 positional arguments but 11 were given"). Fixed in SAGESim (`_inject_seed`
adds `logical_ids` at a call site only when the call forwards `agent_ids`, which is the
only case the definition gets the parameter) with a regression test in
`tests/test_interning_codegen.py`.

## 16. The multi-rank tick, measured (2026-10-15, laptop GPU shared by N ranks)

Step 1 of the next round: where a one-launch-per-tick multi-rank tick goes. Same 12.5k /
K=1000 Brunel network at every rank count (the single-rank columnar partition split by post
into contiguous soma blocks, `scratchpad/multirank/split_partition.py`), a partition-invariant
Poisson drive (one global draw indexed by post-soma id), event stepping, no learning rules,
400 ticks, SAGESim's per-tick timers on (so N=1 is the per-tick-launch path, not the fused
one). Per-tick means over ticks 50-400, in ms, rank 0 (ranks agree to ~1 %); two runs each.

| N | total | host launch | GPU (sync) | extras | pack | MPI wait | unpack | reverse | bytes/peer | peers |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.96 / 1.02 | 0.37 | 0.32 | 0.14 | - | - | - | - | - | 0 |
| 2 | 2.06 / 2.31 | 0.36 / 0.46 | 0.35 | 0.12 | 0.29 | 0.16 | 0.18 | 0.48 | 175 kB | 1 |
| 3 | 3.86 / 3.46 | 0.55 / 0.49 | 0.32 | 0.16 | 0.45 | 0.61 | 0.45 | 1.14 | 117 kB | 2 |
| 4 | 6.26 / 5.55 | 0.74 / 0.56 | 0.36 | 0.16 | 0.70 | 1.26 | 0.68 | 2.14 | 88 kB | 3 |

Gate: 2, 3 and 4 ranks produce the SAME 156,183 spikes, spike for spike (`compare_spikes.py`).
The 1-rank run matches them spike for spike through tick 108 (39,925 spikes) and then
diverges (156,951 in total): one accumulation order for a soma's 1000 deposits (all
atomics) against local atomics plus peer chunks added afterwards, a last-bit difference
the chaotic network amplifies after ~100 ticks. That is the float-order nondeterminism
accepted in section 8, not a timing difference: a one-tick shift anywhere would show at
tick 13.

What the table says:

1. **The exchange grows with the number of peers, not with bytes.** Bytes per peer fall
   from 175 kB to 88 kB while pack + unpack + reverse go 0.95 -> 2.0 -> 3.5 ms: about
   1 ms per peer. A cProfile of rank 0 (`mr_profile.py`) shows why: `_reverse_exchange`
   spends most of its time in its own Python (per-peer slicing of cupy arrays, a `.get()`
   per peer, a `cp.asarray` per peer, a gather-add-scatter per peer and per array);
   pack and unpack loop over peers x visible properties the same way. The code does
   O(peers x properties) small cupy operations and 2 x peers device syncs per tick.
2. **MPI wait grows the same way** (0.16 -> 0.61 -> 1.26 ms) because the ranks share one
   GPU: a rank's `Waitall` waits for its peers' packs, which serialise on the device.
   Not a property of the machine the code will run on.
3. **The host launch path is the same 0.37 ms as on one rank** (plus contention at N=4).
4. **GPU time per tick is flat at 0.32-0.36 ms** at every N: the kernel is not the cost.

Caveats of the laptop: one GPU time-sliced between N processes, no GPU-aware MPI (every
peer buffer is staged through the host with a `.get()` / `cp.asarray`), 22 cores. The
per-peer op structure and the launch path are properties of the code and carry over; the
wait and the staging do not.

Frontier context (pull-era kernel, weak scaling 2026-07-30, `weak_single_4213362.out`,
rank 37 at tick 50): total 5.1 ms = launch 0.4 + exchange 0.6 + kernel 1.1 + rest. The
event path cuts the kernel to ~0.3 ms, so on that machine launch + exchange would be
roughly three quarters of the tick as the code stands.

Where to cut (step 2 of the round):

- **Launch path (SAGESim, general):** keep the compiled cupy `Function` from the first
  call and launch it with pre-typed arguments; fold the three scalar resets; skip the
  bookkeeping unless verbose. 0.37 + ~0.15 -> ~0.1 ms per launch.
- **Exchange in O(1) operations per tick (SAGESim, general):** build at setup one send
  index array in peer order with per-peer offsets, so a tick is one gather per property
  (or one for all, with a fixed [property, agent] layout), one device-to-host copy, MPI
  from host views, one host-to-device copy, one scatter per property; the reverse the
  same way with one segmented add. Expected: pack + unpack + reverse from ~1 ms per peer
  to ~0.2 ms per tick regardless of peers. Matters on Frontier's torus (26 peers).
- **Ship only the spike slot (SuperNeuroABM, decision needed):** 28 of the bytes per
  ghost soma are 7 floats, of which the event path reads 2 (`output_spikes_tensor`).
  The other 5 are the soma's `internal_states`, neighbour-visible only because the
  synapse `internal_states` is flagged visible for the pull-era soma gather and
  SAGESim's visibility is per property name across breeds. Turning it off halves the
  per-property work in pack/unpack; a custom kernel that reads a neighbour's
  `internal_states` (poc/hetero_chip does) would set
  `set_property_neighbor_visible("internal_states", True)`.
- Later, bigger: exchange spikes as events (O(spikes) instead of O(ghost rows)); exchange
  every min-delay ticks instead of every tick.
