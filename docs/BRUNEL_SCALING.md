# Brunel network for SuperNeuroABM scaling tests — discussion doc

> **Status: for group discussion, not implementation.** This doc lays out *what a
> Brunel-network API for scaling tests needs*, *why*, how NEST/NEST-GPU do it, and the
> **open decisions** (with a proposed default to react to). Nothing here is built yet.
> The goal of the meeting is to agree on the feature set, the defaults, what we leave
> out, and — importantly — whether our **synapse-as-agent** architecture changes the
> scaling story versus NEST.

---

## 1. Why we want this

We want SuperNeuroABM to run **weak- and strong-scaling tests on a Brunel network**, the
same network NEST and NEST-GPU use for their scaling benchmarks. Brunel is *the* canonical
SNN scaling benchmark (Brunel 2000; NEST `hpc_benchmark`; Kunkel et al. 2014; Jordan et al.
2018; van Albada et al. 2018). Using it makes SuperNeuroABM's scaling numbers **directly
comparable** to the rest of the field instead of being a one-off.

The ask: an **API that builds a Brunel network of a requested size and feeds it to
SuperNeuroABM to simulate**, so a user can sweep across GPUs/ranks and get meaningful timing.

---

## 2. What a Brunel network is (so we agree on the target)

> **Terminology — "population" = somas.** In NEST a *population* is a group of
> **neurons**, and a NEST neuron = a SuperNeuroABM **soma**. So the excitatory population
> `N_E` and inhibitory population `N_I` are just sets of somas; "population" never means
> synapses. **Synapses are not in a population** — in NEST a synapse is a *connection*
> between two neurons (an edge property), and in SuperNeuroABM it is a separate agent that
> wires two somas. A neuron's excitatory/inhibitory **identity belongs to the soma** (which
> population it's in); the synaptic **weight sign** (`+J` vs `−g·J`) is carried by the
> synapse and is set by *which population its presynaptic soma belongs to* (every synapse
> out of an inhibitory soma carries `−g·J`). So "draw `C_E` sources from the E-pop" means
> "pick `C_E` excitatory **somas** as presynaptic partners; their synapses get `+J`."
>
> **A soma's E/I type is fixed and is about its *outgoing* synapses, not its incoming ones.**
> Type is a structural label assigned once at construction (first `N_E` ids excitatory, next
> `N_I` inhibitory) and never changes during simulation — by Dale's principle a soma has one
> sign on everything it *sends*. So a synapse `pre=A, post=B` gets its sign from **A** (the
> source); **B's** type is irrelevant to it. Every soma's *incoming* synapses are therefore a
> fixed **mix** — exactly `C_E` excitatory (`+J`) and `C_I` inhibitory (`−g·J`) — regardless
> of the target soma's own type. Receiving lots of inhibition just makes a soma *fire less*;
> it does **not** turn it inhibitory. Type (fixed, sets outgoing sign) and activity (dynamic,
> driven by incoming synapses) are separate.

A sparse, balanced, random recurrent LIF network:

- **Two populations, 4:1** — `N_E` excitatory + `N_I` inhibitory, `N_E = 4·N_I`. Sized by
  `order`: `N_E = 4·order`, `N_I = order`.
- **Fixed in-degree** — each neuron receives **exactly** `C_E` excitatory + `C_I` inhibitory
  recurrent synapses (`C_E = ε·N_E`, `C_I = ε·N_I`, ε≈0.1). *This is the property that makes
  Brunel a clean scaling benchmark* (§5).
- **Balanced weights** — excitatory `+J`, inhibitory `−g·J`; `g=4` exact balance, `g=5`
  (inhibition-dominated) → the asynchronous-irregular (AI), cortex-like regime.
- **External Poisson drive** — each neuron driven at rate `ν_ext`, written `η = ν_ext/ν_thr`,
  `ν_thr = θ/(J·C_E·τ_m)`.
- **Homogeneous delay** `D = 1.5 ms`; LIF: `τ_m=20, V_th=20, V_reset=0, t_ref=2`.

NEST reference numbers (our correctness anchor): `order=2500` → `N_E=10000, N_I=2500`;
`ε=0.1` → `C_E=1000, C_I=250`; `g=5, J=0.1 mV, η=2, D=1.5 ms`.

---

## 3. How NEST represents a synapse — and why it matters to us

**This is the section to read carefully**, because SuperNeuroABM and NEST make the
*opposite* architectural choice, and that choice is what the scaling argument hinges on.

| | **NEST / NEST-GPU** | **SuperNeuroABM** |
|---|---|---|
| Neuron | time-driven object, `update()` **every tick** | agent, stepped **every tick** |
| Synapse | passive **connection record**, **event-driven** — touched only when a presynaptic spike arrives | **first-class agent**, stepped **every tick** like a neuron |
| Synapse storage | on the **postsynaptic** rank (the one owning the target neuron) | the synapse agent is co-located with its **post-soma** (post-owns; see [PARTITION_LOADING.md](PARTITION_LOADING.md)) |
| Between spikes | synapse does **nothing** (no work, no update) | synapse agent's step function **still runs** each tick |

In NEST a synapse is **not** an independently-scheduled agent. Quoting the NEST architects
(Stapmanns et al. 2021): *"All synapses implemented in NEST are so far purely event-driven"*
— a synapse's state changes *only* at incoming-spike times (Morrison et al. 2005's
**hybrid scheme**: neurons time-driven, synapses event-driven). Delivery: presynaptic neuron
fires → spike routed to the rank owning the target → the connection writes a weighted event
into a **ring buffer on the target neuron** → the target reads its ring buffer during its own
per-tick integration. Even **STDP** weights are recomputed *lazily on the next presynaptic
spike* (reading the post-neuron's stored spike history via `Archiving_Node`), never per tick.
NEST-GPU (Golosio et al. 2021) keeps the same event-driven model: delay-grouped adjacency
arrays, spikes delivered by a kernel only when a delay bucket comes due.

**Why this matters for us.** The synapse-as-agent design has **two** consequences, and the
second is the one that actually threatens scaling:

1. *Per-tick compute* scales with the **number of synapses, not the number of spikes.** NEST's
   per-tick synapse work scales with spikes delivered (≈ rate × K × neurons), which at ~5 Hz is
   far less than touching every synapse every ms. So our **absolute** per-tick cost is higher,
   and our curve reflects stepping all synapses, not just active ones. This is a constant-factor
   cost, *not* a scaling break: if synapses/rank is held constant (which fixed in-degree gives),
   per-rank synapse-agent compute stays constant as `N` grows.
2. *Per-tick communication* (the real problem). Because synapses are agents that read their
   pre-soma's state every tick, SAGESim ghost-exchanges **dense remote soma state, per peer rank,
   every tick** — connectivity-indexed, not spike-driven (§5.2). NEST never pays this: its spike
   exchange is collective and connectivity-agnostic. **This is what makes "fixed in-degree → flat
   weak scaling" true for NEST but NOT automatically true for us** — see §5.2 and decision D4.

So the agent model costs us a constant compute factor (1) *and* introduces a connectivity-driven
communication term (2) that NEST is immune to. Don't conflate them: (1) is "we're slower per
tick," (2) is "our weak-scaling curve can tilt." This doc treats (1) as the honest price of the
agent model (disclose it) and elevates (2) to the central open decision (D4).

One thing we already share with NEST: **post-owns storage**. NEST stores a synapse on the rank
owning its target neuron; SuperNeuroABM's `load_post_owned` co-locates each synapse with its
post-soma, so the synapse→post-soma current read is always local — only the pre-soma→synapse
spike read can cross a rank. Our partitioning convention matches NEST's; what differs is that our
cross-rank read happens **per tick over point-to-point messages** instead of inside a collective
(§5.2). Post-owns bounds our *memory*, but not our *peer count* — that's the gap.

---

## 4. What the API needs to expose (features to discuss)

Each row is a candidate feature. The meeting decides: keep / drop / default.

| # | Feature | Why it matters | Proposed |
|---|---|---|---|
| F1 | Take **Brunel/NEST parameters** — size (`order`/`N`), connectivity (`ε` or `C_E,C_I`), `g`, `J`, `η`, `delay`. | The vocabulary the benchmark is defined in and every simulator publishes in → comparability. | **Keep** |
| F2 | **Two sizing modes**: ε-mode (`N`+`ε` → derive `C_E,C_I`) and fixed-in-degree mode (`C_E,C_I` fixed, `N` grows). | The two modes *are* strong vs weak scaling (§5). | **Keep** |
| F3 | **Faithful E/I topology** — exactly `C_E` sources from the E-pop, `C_I` from the I-pop; weight sign follows source pop. | Holds the network in one dynamical regime as it scales. Alternative is a single-pool random split. | **Decision D1** |
| F4 | **Rank-local generation** — each rank builds only its own neurons' incoming edges, `O(neurons/rank · K)`, never the global graph; closed-form unique IDs + a remote-owner map. | A 10⁶–10⁹-neuron benchmark can't materialize the global graph anywhere; this is what lets *construction* scale. | **Keep** |
| F5 | **External drive** — create the input topology and provide the Poisson **rate** from `η`/`ν_thr`. | Topology alone is silent; the drive is what we time. | **Keep** |
| F6 | **Two delivery paths** — per-rank partition file (distributed runs) and in-memory lists (tests / small runs / oracle). | Distributed = the real runs; single-process = tests + correctness oracle. | **Keep** |
| F7 | **Brunel-faithful LIF preset** — the neuron params + a `J` calibrated to ~0.1 mV PSP. | The published AI/SI regimes only appear with the right neuron + drive. | **Decision D2** |

---

## 5. How weak & strong scaling work — Brunel, NEST, and us

Per-neuron compute and memory are dominated by a neuron's **in-degree** (number of incoming
synapses). That one fact drives everything.

### 5.1 The two modes are the two tests

- **Weak scaling** ⇐ fixed-in-degree mode (F2). Hold **neurons/rank** and **`C_E,C_I`**
  constant; grow rank count `M` so `N = M·neurons/rank` grows with the machine. Per-rank
  *compute* = `neurons/rank × (C_E+C_I)` → **constant** → flat *if compute-bound*.
  *Why fixed in-degree, not fixed ε:* if ε were fixed, `C = ε·N` grows with `N` → per-neuron
  work grows → not weak scaling. So **ε floats** (network gets sparser as it grows).
  ⚠️ **This holds the *compute* flat but not necessarily the *communication*** — for us, with
  global-uniform wiring, per-rank peer count grows with `M` (§5.2/D4). Fixed in-degree is
  *necessary but not sufficient* for flat weak scaling on SuperNeuroABM.
- **Strong scaling** ⇐ ε-mode (F2). Hold **total `N`** (or `order`) and **in-degree** fixed;
  grow `M` so neurons/rank `= N/M` shrinks. Per-rank compute ~`1/M` → ideal curve ~`1/M`;
  communication (§5.2) is again the floor that flattens the speedup at high `M`.

### 5.2 What NEST holds constant, and the three mechanisms behind its scaling

NEST's weak/strong scaling comes from three things working together — useful as a checklist
for what we'll need too:

1. **Held constant for weak scaling:** neurons-per-rank, in-degree `K`, and firing rate.
   (Kunkel 2014: 2,000 neurons/core, `K=11,250`, → 1.86×10⁹ neurons on the K computer.
   Jordan 2018: 18,000 neurons/node, `K=11,250`, → 1.5×10⁹ neurons.) Because `K` and rate are
   fixed, both per-rank **work** and per-rank **spike count** are invariant as `N` grows.
2. **Connection storage that is per-rank-bounded.** NEST's 4g redesign (Kunkel 2014) made
   per-rank synapse memory depend on **local neurons `N/M`, not total `N`** — empty target
   lists are never instantiated, so a rank pays only for synapses whose target it owns. 5g
   (Jordan 2018) split this into a postsynaptic *connection table* + a presynaptic *routing
   index*. **Our equivalent:** post-owns partitioning already bounds a rank's synapses by
   `(local neurons × K)` — structurally the same property. We need to confirm our per-rank
   *memory* (agent buffers for somas + synapses + ghosts) actually stays flat in a weak sweep.
3. **Min-delay-batched, directed *collective* spike exchange.** NEST exchanges spikes once per
   **min-delay interval** (1.5 ms), not per timestep, and (5g) via directed `MPI_Alltoall`
   rather than `MPI_Allgather`. Crucially the payload is **spikes**, and per-rank spike volume
   is set by the (held-constant) firing rate — **not by connectivity**. A spike is emitted
   *once* by its source; whichever ranks own targets pick it out of the collective buffer and
   resolve fan-out **locally** against their post-local synapses. So "a neuron's `K` presynaptic
   partners are scattered across many ranks" creates **zero** extra messages — NEST's exchange
   is **connectivity-agnostic and peer-count-agnostic**. This is the main thing that lets its
   scaling continue past thousands of ranks.

   **Our equivalent is architecturally the OPPOSITE — and this is the section to argue over.**
   SAGESim's per-tick ghost exchange (`CommunicationManager.mpi_exchange`,
   `SAGESim/sagesim/gpu_kernels.py:908`) is **point-to-point**: a loop of non-blocking `Isend`/
   `Irecv` over a precomputed peer list (`_send_peer_order` / `_recv_peer_order`,
   gpu_kernels.py:920–937), **one dense message per peer rank**, closed by `Waitall`. It ships
   the **soma state of every distinct remote pre-soma every sync** (no spike-sparsity). The
   collectives we *do* use are all **off the per-tick hot path**: an `Alltoall` of request counts
   runs **once** at setup in `build_communication_maps` (gpu_kernels.py:627) to discover the
   topology; `allgather` appears only in owner-resolution accessors and end-of-run result
   gather. So our per-tick cost ∝ **(distinct remote peers)** [latency, = length of the
   peer-order loop, one message each] **+ (ghost volume)** [bandwidth]. Both terms are
   **connectivity-indexed** — exactly what NEST's collective makes irrelevant.

   **Consequence (D0/D4): we cannot brainlessly copy NEST's global-uniform fixed-in-degree
   benchmark.** With global-uniform wiring, distinct remote peers per rank grows toward
   `num_ranks−1` as `N` grows, so the `mpi_exchange` loop lengthens with the machine → a
   communication/latency tilt. NEST sees none of this because its exchange is collective and
   spike-driven; we see it because ours is point-to-point and connectivity-driven. **Fixed
   in-degree keeps per-rank *compute* flat for us, but NOT per-rank *communication*** — which is
   precisely the gap NEST never has to confront. Two levers to close it: **(1)** keep the
   point-to-point exchange but make wiring **spatial/local** so each rank's peer list stays a
   small constant (network-generation change only, no engine change — the spatial-Brunel
   effort); or **(2)** re-architect the per-tick exchange toward a NEST-style **collective
   spike-broadcast** with post-local fan-out (engine change in `mpi_exchange`, removes the
   constraint entirely). Lever 1 is far cheaper near-term; lever 2 is the deeper fix. This is
   likely our strong-*and*-weak-scaling limiter, exactly as MPI spike-exchange is NEST's
   (*"deviations from perfect scaling can mainly be traced back to MPI communication"* — Jordan
   2018).

### 5.2b At a glance — NEST vs us (the whole argument in pictures)

**The one table to remember.** Same Brunel network, same fixed in-degree. What differs is
*how the engine moves cross-rank data each tick.*

| Per-tick cross-rank exchange | **NEST / NEST-GPU** | **SuperNeuroABM (SAGESim)** |
|---|---|---|
| What is sent | **spikes** (sparse, only neurons that fired) | **dense soma state** of every distinct remote pre-soma |
| How it is sent | one **collective** per min-delay window | **point-to-point**, one `Isend`/`Irecv` **per peer rank** (`gpu_kernels.py:908`) |
| Cost scales with | **firing rate** (held constant) | **# distinct remote peers** (latency) **+ ghost volume** (bandwidth) |
| Effect of "K partners scattered across all ranks" | **zero** extra messages | **more** messages — one per newly-touched peer rank |
| Under global-uniform wiring, as ranks `M` grow | flat | **peer count → `M−1` → tilt** |
| Verdict for weak scaling | fixed in-degree **sufficient** | fixed in-degree **necessary but NOT sufficient** |

**Why a spike-broadcast doesn't care about wiring, but per-peer messaging does:**

```
NEST  — collective spike exchange (connectivity-AGNOSTIC)
  rank0 ─┐
  rank1 ─┤            ┌──────────────┐         every rank reads the shared buffer
  rank2 ─┼──spikes──▶ │ ONE collective│ ──────▶ and resolves its own targets locally
  rank3 ─┤            │   per window  │         (fan-out is LOCAL, free)
  rank4 ─┘            └──────────────┘
  cost ∝ #spikes (firing rate) — NOT who-connects-to-whom.  Adding ranks: cost flat.

SuperNeuroABM — point-to-point ghost exchange (connectivity-INDEXED)
  rankR ──Isend/Irecv──▶ peer A   ┐
        ──Isend/Irecv──▶ peer B   │  ONE message PER distinct remote peer rank,
        ──Isend/Irecv──▶ peer C   │  every tick (gpu_kernels.py:920–930)
        ──Isend/Irecv──▶ ...      ┘
  cost ∝ #peers.  Global-uniform wiring scatters a neuron's K sources across ALL ranks,
  so #peers climbs toward (M−1) as the machine grows  →  latency tilt.
```

**Make it concrete — distinct remote peers per rank (global-uniform Brunel).** Each neuron
draws `K` sources uniformly from the whole population; with enough ranks essentially every
*other* rank holds at least one source a local neuron needs, so a rank ends up messaging
almost all `M−1` others:

| Ranks `M` | NEST peers/rank (collective) | Our peers/rank (point-to-point, global-uniform) | Our peers/rank (spatial/local, lever 1) |
|---:|:---:|:---:|:---:|
| 2   | n/a (1 collective) | 1 | 1 |
| 8   | n/a | ~7 | ~8 (fixed halo) |
| 64  | n/a | ~63 | ~8 |
| 1024 | n/a | ~1023 | ~8 |
| `M` | **O(1)** | **O(M)** ❌ | **O(1)** ✅ |

(Numbers illustrative; exact values come from the `n_peers` we'd log per rank.)

**What the weak-scaling curve looks like (sketch).**

```
 time/tick
   │                                         ✗ us, global-uniform wiring
   │                                    ✗      (peer count ∝ M → comm tilt)
   │                              ✗
   │                        ✗
   │                  ✗
   │      ✗ ✗ ✗
   │ ─────────────────────────────────────  ✓ NEST  &  ✓ us with spatial/local wiring (lever 1)
   │                                            (flat — peer count constant)
   └──────────────────────────────────────────────────▶  ranks M  (log scale)
       ideal weak scaling = flat line
```

**The decision, as a fork (this is D4):**

```
  Want flat weak scaling on SuperNeuroABM?
        │
        ├─ Lever 1: change the WIRING → spatial/local (bounded peer set)
        │     • network-generation change only, NO engine change
        │     • cheap, near-term;  keeps point-to-point exchange
        │     • caveat: "local-only" wiring can look too simple (the GGap critique)
        │
        └─ Lever 2: change the ENGINE → collective spike-broadcast in mpi_exchange()
              • connectivity-agnostic like NEST → NEST's verbatim setup then weak-scales
              • removes the constraint entirely;  bigger effort (SAGESim change)
              •  (or Lever 3: keep as-is and DISCLOSE the tilt as honest agent-model cost)
```

### 5.3 Regime stability while scaling (F3, F7)

The balanced-network mean field depends on `C_E, C_I, g, η` — **not on `N`**. Fixing
in-degree + the E/I split keeps firing statistics constant as `N` grows, so a sweep measures
*performance*, not a drifting dynamical target.

**One line:** *fix neurons/rank + in-degree → weak; fix total N + in-degree → strong; ε
floats; rank-local generation keeps construction scalable; fixed E/I split + faithful neuron
keep the regime stable. **But for us, flat weak scaling also needs constant per-rank PEER
COUNT — which fixed in-degree alone does NOT give under global-uniform wiring (§5.2/D4).***

---

## 6. Sketch of the API surface (to react to, not final)

A module `superneuroabm/brunel.py` with roughly:

- `resolve_brunel(mode, order/N, epsilon | C_E,C_I, neurons_per_partition, num_partitions)`
  → integer counts `(N_E, N_I, C_E, C_I, neurons_per_partition)`. The only place mode logic lives.
- `brunel_partition(my_rank, num_partitions, N_E, N_I, C_E, C_I, …, J, g, delay, seed)`
  → one rank's `{"somas", "synapses", "remote_ranks"}` (the core random draw).
- `save_brunel_partition(...)` → writes `partition_{rank}.pkl` (distributed path).
- `build_brunel_lists(...)` → `(somas, synapses)` for an in-memory build (single-process path).
- `brunel_external_rate(eta, J, C_E, theta, tau_m)` → Poisson rate (Hz), NEST's `p_rate`.

Plus two thin driver scripts — `weak_scaling.py` (knobs: neurons/rank, `C_E,C_I`) and
`strong_scaling.py` (knobs: `order`/`N`, `ε`) — that generate → load → setup → drive →
simulate → log timing.

The output (`{somas, synapses, remote_ranks}`, synapse `pre=−1` = external input) is the
shape SuperNeuroABM's loaders already consume (see [PARTITION_LOADING.md](PARTITION_LOADING.md)),
so the API is a **producer of that contract** — no change to the model engine itself.

---

## 7. Proposed defaults (to confirm or change)

| Knob | Proposed default | Note |
|---|---|---|
| E:I ratio | 4:1 (`N_E=4·order`) | Brunel standard |
| `ε` | 0.1 | strong-scaling / classic |
| `g` (`\|J_I\|/J_E`) | 5.0 | inhibition-dominated AI regime |
| `J` | 0.1 mV (or model-unit equiv) | tied to D2 (units) |
| `η` | 2.0 | external drive |
| `delay` | 1.5 ms | homogeneous; also the comm-batching window if we batch |
| in-degree (weak) | `C_E=1000, C_I=250` | NEST `hpc_benchmark`-ish (NEST's big runs use K=11,250) |
| neurons/rank (weak) | TBD (e.g. 5k–18k) | machine-dependent; NEST used 2k (CPU core) – 18k (node) |
| external synapses/neuron | 1 (vs literal `C_E`) | **Decision D3** |
| seed | fixed (e.g. 42) + rank | reproducible |

---

## 8. What we do NOT need (scope boundaries — confirm)

- **No plasticity / STDP** — static-weight benchmark; learning rules off.
- **No spatial / distance-dependent connectivity** — pure random Brunel (the spatial variant
  is a separate effort).
- **No multiple neuron/synapse types** — one LIF population type, one synapse type.
- **No structured cortical microcircuit** (Potjans–Diesmann) — a different benchmark.
- **No changes to the SuperNeuroABM model/engine** — the API only *produces* the partition
  contract the existing loaders consume.
- **No event-driven synapse rewrite** (for now) — we step synapses every tick as agents; we
  *note* the contrast with NEST rather than re-architect (see D0).
- **No detailed recording / analysis** — minimal spike output for a sanity/regime check;
  timing CSV is the deliverable.
- **No graph-library dependency** (NetworkX etc.) — closed-form rank-local draw only.

---

## 9. Open decisions for the meeting

- **D0 — Synapse-as-agent vs event-driven (architectural).** We step every synapse every tick;
  NEST touches a synapse only on a spike (§3). This is a per-tick **compute** constant-factor
  cost (we're slower per tick, but it does not by itself break weak scaling — synapses/rank is
  constant). Do we (a) accept it as the honest price of the agent model and **disclose** it,
  (b) add an event-driven synapse fast-path later, or (c) something else? *Note:* the per-tick
  **communication** consequence of the agent model is the bigger issue and is tracked separately
  in **D4** — don't conflate the two.
- **D1 — Topology fidelity.** Exact two-pool `C_E`/`C_I` draw (faithful Brunel/NEST, regime
  stable, recommended) **vs** single-pool uniform draw + 80/20 split (simpler, per-neuron E/I
  counts random → drifts from textbook). *Headline call.*
- **D2 — Neuron units.** NEST-normalized preset (`V_th=20, V_reset=0, τ_m=20ms, J=0.1mV` —
  matches literature 1:1) **vs** biophysical preset (mV/nF/MΩ — matches SuperNeuroABM's
  existing LIF style) **vs** topology-only (user supplies neuron config, defer dynamics).
  Either preset needs one PSP-amplitude calibration to hit the AI regime.
- **D3 — External drive fidelity.** One external input synapse per neuron (simple, enough for
  timing — recommended) **vs** literal `C_E` external synapses per neuron (closer to Brunel's
  text, more synapses/work).
- **D4 — Communication model (the crux — see the pictures in §5.2b and §5.2 mechanism 3).** Our per-tick exchange is
  **point-to-point and connectivity-indexed** (one `Isend`/`Irecv` per peer rank,
  gpu_kernels.py:908,920–937); NEST's is a **collective, spike-driven** exchange whose per-rank
  volume tracks firing rate, not connectivity. Under global-uniform Brunel wiring our peer count
  grows toward `num_ranks−1` → communication tilt that NEST does not have → **we must not copy
  NEST's wiring verbatim and expect NEST's curves.** Decide between: **(a)** keep point-to-point,
  bound peer count with **spatial/local connectivity** (network-gen change only; near-term
  weak-scaling path); **(b)** re-architect toward a **collective spike-broadcast** with
  post-local fan-out and min-delay batching (engine change in `mpi_exchange`; removes the
  constraint, larger effort); **(c)** keep current per-tick exchange and **disclose** the
  connectivity-driven tilt as the honest cost of the agent model. Likely our dominant
  scaling limiter regardless.
- **D5 — Scope of first build.** Full (API + topology + neuron preset + both scaling scripts)
  **vs** API + scripts first (timing sweeps work; neuron calibration as follow-up) **vs** just
  the `brunel.py` module + tests.
- **D6 — Validation bar.** Timing-only for the first milestone, or require a firing-rate/CV
  regime check (AI for `g=5, η=2`) before we trust the numbers?

### Figures to produce from the runs (so the meeting agrees what we'll plot)

The point of D4 is that the **diagnostic** plot is not the usual time-vs-ranks — it's
**peer-count / comm-fraction vs ranks**, which shows *why* scaling holds or breaks.

| Plot | x → y | What it shows |
|---|---|---|
| Weak scaling | ranks `M` → time/tick | flat = ideal; a rising tail = our comm tilt |
| Strong scaling | ranks `M` → speedup | vs. ideal diagonal; saturation = comm floor |
| **Peer count (the diagnostic)** | ranks `M` → distinct peers/rank (`n_peers`) | **O(1) vs O(M)** — the whole D4 argument in one line |
| Comm fraction | ranks `M` → % time in `mpi_exchange` | when communication overtakes compute |
| Time breakdown | per rank-count → stacked bar (compute \| pack \| exchange \| wait \| unpack) | where the time goes |
| Throughput | ranks `M` → synapses/sec or spikes/sec | headline performance number |

We already have the hooks to log `n_ghost_somas`, `n_peers`, `cross_rank_fraction` per rank;
the peer-count plot falls straight out of `_send_peer_order` length.

---

## 10. References

- **Brunel, N. (2000).** Dynamics of sparsely connected networks of excitatory and inhibitory
  spiking neurons. *J. Comput. Neurosci.* 8(3):183–208.
- **Morrison, A., et al. (2005).** Advancing the boundaries of high-connectivity network
  simulation with distributed computing. *Neural Comput.* 17(8):1776–1801. (Event-driven
  synapses, ring-buffer delivery, min-delay decoupling, postsynaptic storage.)
- **Kunkel, S., et al. (2014).** Spiking network simulation code for petascale computers.
  *Front. Neuroinform.* 8:78. (4g sparse-table connections; per-rank memory ∝ N/M, not N.)
- **Jordan, J., et al. (2018).** Extremely scalable spiking neuronal network simulation code.
  *Front. Neuroinform.* 12:2. (5g two-tier infra; Allgather→Alltoall; weak/strong numbers:
  K=11,250; 18k neurons/node; N=1,152,000 strong.)
- **Golosio, B., et al. (2021).** Fast simulations of highly-connected spiking cortical models
  using GPUs. *Front. Comput. Neurosci.* 15:627620. (NEST-GPU: delay-grouped adjacency arrays,
  spike-driven GPU delivery.)
- **Stapmanns, J., et al. (2021).** Event-based update of synapses in voltage-based learning
  rules. (Authoritative restatement of NEST's event-driven synapse / min-delay mechanisms.)
- **van Albada, S.J., et al. (2018).** Performance comparison of GPU, HPC, and neuromorphic
  hardware. *Front. Neurosci.* 12:941.
