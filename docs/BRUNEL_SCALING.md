# Brunel network for SuperNeuroABM scaling tests — discussion doc

> **Status: partially implemented.** The core generator API is now built —
> `superneuroabm/brunel.py` provides `brunel_partition` (per-rank sizing, selectable
> `topology`), `build_brunel_network` (in-memory), `save_brunel_partition` (per-rank file), and
> `brunel_external_rate`. The **faithful two-pool `C_E`/`C_I` topology** (D1) is what
> `topology="global"`, `"bounded"` and `"torus2d"` do; the **topology the scaling campaign
> actually measured is `"torus3d"`** (§5.5a), which draws a *single* pool of `C_E + C_I` sources
> **uniformly within a hard-cutoff spatial ball** and reads E/I identity off each source's id —
> so its 4:1 balance holds over the population, not per soma. Distance-weighted (Gaussian)
> connectivity exists only in the separate `spatial_smallworld` generator (§5.5b) and is **not**
> part of the campaign. The LIF preset is biophysical (D2 deferred). This doc still captures
> *why*, how NEST/NEST-GPU differ, and the **open decisions** (D0/D3–D6) — notably the
> communication model (D4), which is a separate effort and not addressed by the generator alone.

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

### 3.1 Where do synaptic *current* dynamics live? — a concrete divergence

The Brunel benchmark uses **static** synapses, so nothing below affects the scaling runs. But it
is the clearest illustration of the architectural choice in §3, and it matters the moment we care
about anything richer than a delta-current synapse — so it belongs here as a reference point for
D0.

**Question that surfaces it:** *can one postsynaptic neuron have two input synapses with
different exponential-decay time constants for their current?*

**Where the decay `τ` lives is the whole answer.** The exp-decay of the postsynaptic current is a
property of the **receptor/conductance dynamics**, and the two architectures put those dynamics in
different objects:

| | **NEST / NEST-GPU** | **SuperNeuroABM** |
|---|---|---|
| Who owns the current-decay `τ` | the **soma** (neuron model), not the connection | the **synapse agent** (it has its own per-tick state) |
| A connection carries | weight, delay, plasticity state — **no `τ`** | whatever state it wants, including its own `τ` |
| How current is integrated | synapse drops a weighted impulse into a **ring buffer**; the soma's per-tick `update()` applies **one decay kernel per channel** | synapse agent integrates **its own** decaying contribution each tick, then writes the result to the post-soma |

**On a plain neuron (`iaf_psc_exp`, one `tau_syn`): NEST answer is no.** Every excitatory input
decays with the *same* `tau_syn_ex` — all inputs land in one ring buffer integrated by one kernel
in the soma. The synapse has no `τ` to differ on; the soma offers only one. This is a direct
consequence of §3: the synapse is passive, the soma owns the decay.

**With a `*_multisynapse` neuron model: NEST answer is yes — via receptor ports.** Models like
`iaf_psc_exp_multisynapse` take a **vector** `tau_syn = [τ₁, τ₂, …]` — `n` receptor ports, each
with its own `τ` and its own ring buffer. Each connection carries a `receptor_type` index
selecting a port, so `synapse A → port 0 (τ=2 ms)` and `synapse B → port 1 (τ=20 ms)` coexist on
one neuron (the standard way to model AMPA/NMDA/GABA kinetics). **But the `τ` set is pre-declared
on the neuron and fixed** — a synapse *selects* one of `n` enumerated `τ`; it cannot carry a free,
continuous per-connection `τ`. Genuinely per-synapse `τ` would need one port per synapse, which is
not how it's used. It is a **routing-index** scheme, not a per-synapse-state scheme.

**What you actually have to do in NEST — the recipe.** Two inputs with `τ=2 ms` and `τ=20 ms` onto
one post neuron takes three coordinated steps, and none of them touch the synapse's *dynamics*
— they only pick a **neuron model with the right receptor set** and **route** each connection to a
port:

1. **Pick a multisynapse neuron model and declare the `τ` set on the *neuron*.** Create the post
   neuron as e.g. `iaf_psc_exp_multisynapse` with `tau_syn = [2.0, 20.0]` (ms). The two decay
   constants are now **ports 1 and 2** on that neuron (NEST `receptor_type` is **1-indexed**;
   `receptor_type = 0` is the "no receptor" default). The `τ` values live entirely on the neuron —
   the synapses will not carry them.
2. **Route each connection to the port whose `τ` it wants** by setting `receptor_type` at
   `Connect` time. In `nest.Connect(pre, post, syn_spec={...})` put `receptor_type = 1` for the
   `τ=2 ms` input and `receptor_type = 2` for the `τ=20 ms` input. Same synapse *model* (e.g.
   `static_synapse`) both times; only the port index differs.
3. **The soma does the rest.** Each port has its own ring buffer and its own decay kernel; the
   neuron's `update()` integrates buffer 1 with `τ=2 ms` and buffer 2 with `τ=20 ms`, then **sums**
   the two currents into `I_syn`. The synapse never decays anything — it only chose which buffer to
   drop its weighted impulse into.

So "two `τ` on one neuron" is achieved by **enumerating both `τ` up front on the neuron** and
**tagging each synapse with a port index** — *not* by giving the synapses different dynamics. If
later you need a third `τ`, you must **redeclare the neuron's `tau_syn` vector** (add a port); you
cannot just make a new synapse with a new `τ`. This is the exact limitation the routing-index scheme
imposes.

**In the synapse-as-agent model: yes, natively, with no special machinery.** Because the synapse
agent already runs every tick and holds its own state, it can carry its **own `τ`** and integrate
its **own** decaying current locally, then hand the result to the post-soma — which just sums what
its synapses give it. Every synapse having a **distinct, continuous** `τ` is native: no receptor
ports, no `tau_syn` vector declared on the neuron, no upper bound on the number of distinct `τ`.

**The trade (feeds D0).** NEST pushes current dynamics **into the soma** and makes synapses passive
selectors → cheap, but heterogeneous kinetics are limited to what the neuron model **pre-declares**
(`multisynapse`). The agent model lets each synapse be a **full stateful dynamical object** →
arbitrary per-synapse kinetics for free (an expressiveness *upside*), at the per-tick cost D0/D4
already flag. The same pattern recurs for plasticity: NEST's event-driven STDP is only possible
because synapses are passive and post-neurons archive their own history (§3), whereas an agent
synapse can read its post-soma's live state each tick and update eagerly. **Net: NEST buys speed by
constraining what a synapse can be; the agent model buys expressiveness by paying per-tick
unconditionally.** Heterogeneous synaptic time constants are a concrete case where "synapse as
first-class agent" is *strictly more expressive* than NEST without a special neuron model — worth
weighing in D0 alongside the cost.

### 3.2 How NEST does STDP without ever running the synapse at post-spike time

Brunel is a **static-weight** benchmark (§8: plasticity off), so this changes nothing in the
scaling runs. It is here because it resolves an apparent paradox in §3 and, like §3.1, sharpens the
D0 trade — so it belongs as a reference point.

**The apparent paradox.** If a synapse is **event-driven** — touched only when a *presynaptic*
spike arrives (§3) — and the postsynaptic current is integrated in the **soma's** per-tick
`update()`, then STDP needs the *post* neuron's spike timing, but the synapse never runs at
post-spike time. So how does the weight update ever see the post-spike?

**The resolution: `Archiving_Node` + lazy, pre-spike-triggered update** (Morrison et al. 2005).
Delivery and weight update happen at **different times, in different places**:

1. **Current delivery (per incoming spike, in the soma).** A pre-spike makes the synapse write a
   weighted event into the **post neuron's ring buffer**; the soma's time-driven `update()` reads
   that buffer every tick and integrates it. The synapse is *not* scheduled tick-by-tick — it just
   deposited a value the soma later consumes.
2. **Weight update (STDP) — deferred and *pre*-spike-triggered.** The post neuron does **not**
   notify its synapses when it fires. Instead every plastic-synapse target is an **`Archiving_Node`**
   that silently records its **own** spike history with timestamps. Weights are recomputed **lazily,
   only when the next presynaptic spike arrives** — never at post-spike time, never per tick. On
   pre-spike #k at time `t_k` the synapse:
   - looks back over `(t_{k-1}, t_k]` — the interval since its own previous pre-spike,
   - **reads the post neuron's archived spikes** in that interval,
   - applies **all** accumulated pairings at once: each archived post-spike in the window
     contributes **potentiation** (pre-before-post), and the incoming pre-spike contributes
     **depression** against the post neuron's trace (post-before-pre),
   - writes the new weight back, then delivers the event.

   So both the causal (LTP) and acausal (LTD) sides are handled at pre-spike time by *replaying* the
   post neuron's recorded spikes — the synapse is never scheduled at the instant the post fires.

**Three invariants make it consistent:**

- **Exponential STDP kernels** reduce a whole window of pairings to a running trace (`K_+` on the
  pre side, `K_−` read from the post archive), so a batch of post-spikes applies in one shot without
  visiting each pairing when it occurred.
- **The archive keeps just enough post-spike history** — bounded by the max delay — so a
  late-arriving pre-spike can still see the post-spikes it should pair with; older history is
  trimmed.
- **The weight the soma uses for delivery is always the last-written one.** Between two pre-spikes
  the weight is "stale," but no spike traverses that synapse in that interval, so no current is ever
  delivered with a wrong weight — it is guaranteed fresh *at the moment it is used*.

**The event-replayable constraint (and its limit).** This scheme requires each synapse model's
update to be **replayable at the next pre-spike from its own state + the post neuron's *archived*
history**. Plain STDP needs only spike times, so a spike archive suffices. Rules that need a
*continuous* postsynaptic signal every tick — **voltage-based** plasticity (Clopath;
Urbanczik–Senn) — do **not** fit the plain scheme: `V_m(t)` at every tick is neither sparse nor a
spike list. Keeping those event-driven is exactly why NEST had to **extend the archive to a
compressed continuous trace** (Stapmanns et al. 2021, our ref) — that paper exists because the naive
spike archive was not enough.

**The trade (feeds D0), same shape as §3.1.** NEST's lazy STDP is **only possible because** synapses
are passive/event-driven and post-neurons archive their own history. In the **synapse-as-agent**
model a synapse already runs every tick, so it can read its post-soma's live state and update the
weight **eagerly and incrementally** — no `Archiving_Node`, no closed-form fast-forward, no replay,
and voltage-based rules are trivial (just read `V_m`). That is *simpler to express*, at the per-tick
cost D0/D4 flag. **Net: NEST defers to avoid per-tick work but is constrained to event-replayable
rules; the agent model updates eagerly and handles arbitrary continuous rules, paying per-tick
unconditionally.** If plasticity ever re-enters scope, this is the trade to write down.

### 3.3 The ABM advantage vs. NEST *and* Brian2 — an open agent ontology on a distributed engine

§3.1 and §3.2 are two instances of a single, more fundamental property, and it's worth stating it
directly because it is the reason the ABM design exists. **The advantage is not "richer synapses" —
that's a symptom. The advantage is an *open agent ontology*: soma and synapse are the current agent
types, but the architecture privileges no fixed number of them.** An agent is just *a stateful
object with a step function and neighbors*, so adding a **new entity type — a dendrite, an axon, a
glial cell, a neuromodulatory pool** — is the same act as everything already there: define its
state, its step function, its neighbors. You extend the *model*, not the *engine*.

This lands differently against NEST and against Brian2, so the comparison must be **three-way** —
lumping them together hides the actual argument. NEST and Brian2 sit at opposite ends of a spectrum;
the ABM is a third axis (open ontology **on** a distributed engine).

| | **NEST / NEST-GPU** | **Brian2** | **Our ABM (SuperNeuroABM)** |
|---|---|---|---|
| Core abstraction | fixed **2 kinds**: time-driven neuron + event-driven connection | **equations + code-generation** — user writes differential eqs / reset rules as strings; Brian2 generates C++/GPU code | **agents** — any number of independently-stepped stateful types |
| Synapse | passive connection record, **event-driven** | a `Synapses` **group** (arrays of state) with user `on_pre`/`on_post`, vectorized | a **first-class agent**, stepped every tick |
| Where synaptic dynamics live | in the **soma** (neuron model); synapse selects a receptor port (§3.1) | in the `Synapses` group's **equations** (expressive) | in the **synapse agent** itself (§3.1) |
| Add a **new entity type** (dendrite, glia) | extend a **compartmental neuron model** in engine C++; it's *internal* to a neuron | add another `NeuronGroup`/`Synapses` with equations; no true peer-object system | **define a new agent type** — a *peer* of soma/synapse |
| Scaling target | **multi-node MPI**, 10⁶–10⁹ neurons (its whole point) | **single node** — OpenMP threads or **one GPU** (Brian2GeNN); **no native multi-node MPI** | **multi-node / multi-GPU** (SAGESim) — the design goal |
| Per-tick cost model | synapse work ∝ **spikes** (collective, connectivity-agnostic exchange) | single-node — no cross-rank exchange to pay | work ∝ **synapses**; per-tick **point-to-point** ghost exchange (D0/D4) |

**Read the table as two different wins:**

- **vs. NEST — expressiveness / open ontology.** NEST's synapse is passive and its entity set is
  closed at two kinds. We get for free what NEST needs special machinery for: per-synapse current-τ
  (no `multisynapse` model — §3.1), eager voltage-based plasticity (no `Archiving_Node` — §3.2),
  and — the general form — **new first-class agent types added by writing a step function, not by
  extending the engine.** NEST buys scale by constraining what an entity can be; we don't accept
  that constraint.
- **vs. Brian2 — distributed scale + a true peer-object model.** Brian2 is *already* expressive
  (arbitrary equations, user `on_pre`/`on_post`), so the τ and plasticity points are **not**
  advantages over Brian2 — it can do those. Our two real advantages over Brian2 are: **(1)** native
  **multi-node distribution** — Brian2 runs on a single node (OpenMP or one GPU via Brian2GeNN) with
  **no native MPI**, while SAGESim is multi-rank/multi-GPU by design (the whole §5 story); and
  **(2)** a true **peer-object model** — Brian2 is still two containers (`NeuronGroup`, `Synapses`)
  over vectorized state arrays, so a dendrite is another group you bolt on with equations, not an
  independently-scheduled, individually-addressable entity with its own neighbor graph the way our
  agent is.

**The synthesis.** *NEST = scalable but closed (2 fixed kinds); Brian2 = expressive but single-node
(no native MPI); our ABM = an open agent ontology **on** a distributed multi-node engine — the one
combination neither competitor offers.* Adding a **dendrite agent** is the concrete proof point: a
new peer agent for us, an engine-model extension for NEST, a single-node group-of-equations for
Brian2.

**Two honest caveats (so this doesn't overclaim):**

1. **The cost is the same coin.** Every new agent type is another population stepped **every tick**,
   and if it reads a *remote* neighbor each tick it feeds the **point-to-point, connectivity-indexed
   ghost exchange** (D4). "Easy to add agent types" and "per-tick D0/D4 cost" are the *same*
   property from two sides: the ABM makes new entities cheap to *express* and pays for them in
   *runtime*. The niche we occupy (open ontology **at** distributed scale) is exactly what makes D4
   ours to solve and unavoidable — NEST dodges it with collective spike exchange; Brian2 dodges it
   by not distributing at all.
2. **"Easily" is a claim to verify, not assert.** The architecture *permits* arbitrary agent types;
   whether adding a *third* type (dendrite) is actually a clean "write a step function" today
   depends on how generic SAGESim's agent registration, neighbor wiring, and ghost exchange really
   are beyond the soma+synapse pair. Two types working does not prove N types work frictionlessly —
   worth confirming the engine treats agent-type count as a free parameter before we lean on this in
   a paper.

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

### 5.4 Implemented: spatial 2D-torus (tile-block) — the flat-scaling wiring (milestone 1)

The D4 fix we chose is **Lever 1 (spatial/local connectivity)**, implemented in
`brunel.py` as `topology="torus2d"`. It gives constant per-rank peer count the conventional
way point-to-point codes weak-scale (HPC stencils; distance-dependent cortical models,
Billeh 2020 / Schmidt 2018).

- **Geometry.** The `num_partitions` ranks form an `a×b` **torus grid** of tiles
  (`_grid_factorization` picks a near-square `a,b ≥ 3`); each rank owns one tile = its
  contiguous soma-id block `[r·npp,(r+1)·npp)` (post-owns for free). Every soma draws its
  `C_E`+`C_I` recurrent sources uniformly over its **9 tiles** — its own + the 8 Moore
  neighbors (`_select_neighbor_tiles`, torus-wrapped, so no special boundary ranks).
- **Why it's flat.** A rank reads from — and, by symmetry of the grid, sends to — **exactly
  its 8 neighbor tiles**, independent of `M`. So both the message count (peers) and the
  ghost volume (border somas) are **O(1) in M**. This is the "tile-block" *connectivity
  kernel* (not a GPU kernel): locality at tile granularity, drawn from whole neighbor tiles.
  A continuous distance kernel — Gaussian, position-dependent, border-strip-only — was a
  later realism upgrade, and it landed in **`spatial_smallworld` (§5.5b), not here and not in
  the measured `torus3d` topology**, both of which draw *uniformly* with a hard cutoff.
- **Peer count vs M (the headline figure).** `torus2d` is **flat at 8**; the global-uniform
  baseline climbs toward `M−1` (O(M)); the random `remote_rank_fanout=R` wiring sits between
  (mean `R`, max ~`log M`). Sweep `M = 16, 32, 64, 128, 256, 512, 1024` (powers of 2,
  aspect-bounded grids; the 8-peer plateau begins at M=16 — M=9 is degenerate, 8 = M−1).
  Second axis: connection **density** `K` = a *family* of flat curves (fixed K per curve).
- **Status.** Flat local-only. Verified by `tests/test_brunel_spatial_torus.py` (exactly-8
  Moore-neighbor peers, contiguous blocks, fixed in-degree, E/I, no autapses,
  grid-factorization). **Superseded for 3D by §5.5** (the 2D tile-block's uniform-over-9 draw put
  only ~1/9 of edges on the own rank — comm-heavy; the 3D work uses a proper spatial radius).

### 5.5a Implemented: 3D spatial-radius stencil — the point-to-point weak-scaling convention

The 3D successor to §5.4, and the one that follows how point-to-point HPC codes (PDE/stencil,
MD, lattice QCD) actually weak-scale: **volume-local compute + surface-halo communication**. In
`brunel.py` as `topology="torus3d"`.

- **Geometry + coordinates.** The `num_partitions` ranks tile a **periodic `a×b×c` torus**
  (`_grid_factorization_3d`, near-cube, **relaxed to any `M ≥ 1`** so the sweep starts at 1 GPU).
  Every neuron has a **3D position**: each rank's contiguous id block is a `gx×gy×gz` neuron
  sub-block (`_factor_near_cube`), so `id = rank·npp + intra` is a bijection to the global grid
  and *a rank's id block == one spatial tile* (post-owns for free). Helpers `_soma_positions` /
  `_positions_to_soma`.
- **Spatial-radius draw.** Each neuron draws its fixed `K` sources **uniformly (without
  replacement) within Euclidean `connection_radius`** of its own position on the torus
  (`_draw_sources_radius`; `connection_radius=None` auto-picks the smallest ball holding `≥ 2K`).
  Because the radius is smaller than a tile, **interior neurons draw entirely from their own tile
  (fully local, no MPI)**; only neurons within the radius of a tile face reach a neighbor → the
  cross-rank traffic is a bounded **surface halo**. The own-rank (local) fraction **rises with
  tile size** (surface/volume): npp = 1k/8k/27k → 0.62/0.80/0.86 local. E/I identity is read off
  each drawn source's id for the weight (no exact `C_E`/`C_I` split — fine for a comm benchmark;
  the bio-realistic two-pool / distance-dependent E/I is §5.5b).
- **The 4:1 E/I balance is a population average, not a per-soma property.** Since the draw is a
  single pool over the ball and the E-portion of a rank's id block maps to a contiguous *slab* of
  the tile (`ix ∈ [0,16)` of 20 at npp=12500), a soma's own excitatory in-degree fraction depends
  on where it sits. Measured at npp=12500, w=27, K=1000, R=8: per-neuron E-fraction spans
  **0.601 – 1.000** (median 0.780) around a population mean of exactly **0.8000**, so
  slab-interior somas receive **almost no inhibition**. Disclose this wherever the network is
  described; do not call it balanced per soma.
- **The measured exchange is activity-independent.** Across all 36 weak-scaling points,
  `send_bytes_mean / ghost_somas_mean = 20.00` exactly — a fixed 20 B of ghost soma state per
  ghost per tick, sent whether or not anything spiked. That is why the sub-threshold external
  drive the campaign actually ran (§7 D3 / `paper_figures/README.md` §1.4) leaves every reported
  number intact: these curves measure partition geometry, not a firing regime.
- **Baseline + ramp-and-plateau.** The **1-GPU run is the baseline `T(1)`** (a `1×1×1` grid: all
  sources wrap onto the own tile, no MPI). As the grid fills out the Moore-neighbor peer count
  **ramps** (M=1/2/4/8/27 → peers 0/1/3/7/26) and **plateaus at 26** once every dim ≥ 3. The
  weak-scaling claim is the **flat plateau** (`T(M) ≈ T(1)`, parallel efficiency `T(1)/T(M)`); the
  small-M ramp is the expected surface-dominated regime, shown not hidden. On Frontier (8 GCD/node)
  the ramp+baseline (1→32 GPUs) fits on ≤ 4 nodes (debug-cheap); the plateau (64→1024 = 8→128
  nodes) is the batch sweep. Contrast curves: `global` O(M), `bounded` O(log M).
- **Status.** Implemented, unit-verified and **measured — campaign complete 2026-07-30, 36/36
  points** (results below). Unit tests `tests/test_brunel_spatial_torus3d.py` (grid
  factorization, coordinate bijection, peer ramp→plateau, surface-halo locality, fixed K, no
  autapses, negative inhibitory weight, 1-GPU baseline). Driver `scaling_analysis/weak_scaling.py`,
  runner `weak_3d_chunk.sh`, analysis `analyze_weak.py`.

#### Results (campaign complete, 2026-07-30 — 36/36 points)

12,500 neurons/GPU held constant, `K` ∈ {1000, 2000, 4000}, `w` = 1 → 2048 GPUs (1 → 512 nodes),
100 ticks. `outputs/weak_3d_final.csv`; figures `figures/weak_3d_K{1000,2000,4000}.png`.
`K`=1000 ran 8 ranks/node; `K`=2000 and 4000 need 4 (memory ceiling, see `MEMORY_ANALYSIS.md`).

As in §5.5c, two timings, never conflated. **Construction** is tick 1; **step** is the mean of
ticks 11…100 (see *The step metric* below). Construction is ~99.8 % of a 100-tick run, so a
single "simulation time" number is a construction benchmark. `flat %` is the weak-scaling
efficiency `T(1)/T(w)` — 100 % is a flat curve.

| `w` | nodes | step, `K`=1000 | flat % | step, `K`=2000 | flat % | step, `K`=4000 | flat % | peers |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 4.175 ms | 100.0 % | 7.491 ms | 100.0 % | 15.338 ms | 100.0 % | 0 |
| 2 | 1 | 4.604 ms | 90.7 % | 7.788 ms | 96.2 % | 16.460 ms | 93.2 % | 1 |
| 8 | 1–2 | 5.326 ms | 78.4 % | 8.752 ms | 85.6 % | 17.400 ms | 88.1 % | 7 |
| 32 | 4–8 | 6.049 ms | 69.0 % | 9.313 ms | 80.4 % | 18.135 ms | 84.6 % | 17 |
| **64** | 8–16 | 6.343 ms | **65.8 %** | 9.729 ms | **77.0 %** | 18.522 ms | **82.8 %** | **26** |
| 128 | 16–32 | 6.345 ms | 65.8 % | 9.690 ms | 77.3 % | 18.596 ms | 82.5 % | 26 |
| 256 | 32–64 | 6.398 ms | 65.3 % | 9.698 ms | 77.2 % | 18.574 ms | 82.6 % | 26 |
| 512 | 64–128 | 6.386 ms | 65.4 % | 9.691 ms | 77.3 % | 18.655 ms | 82.2 % | 26 |
| 1024 | 128–256 | 6.364 ms | 65.6 % | 9.604 ms | 78.0 % | 18.635 ms | 82.3 % | 26 |
| 2048 | 256–512 | 6.392 ms | 65.3 % | 9.633 ms | 77.8 % | 19.325 ms | 79.4 %† | 26 |

† `K`=4000 `w`=2048 sits 4.0 % above its own median — construction settles late there (see
*The step metric*). Every other point agrees with its median to ≤0.6 %.

**The plateau is flat, and that is the result.** Past `w`=64 — where the stencil saturates at 26
Moore neighbours — the step time stops moving: it varies by **0.9 % (`K`=1000) and 1.3 %
(`K`=2000)** across `w`=64 → 2048, a **32× span of GPUs** and of problem size. `K`=4000 varies by
4.3 %, carried entirely by its `w`=2048 point; over `w`=64 → 1024 it is 0.7 %. The largest point
simulates **25.6 M neurons and 102 G synapses on 2048 GCDs** at the same ~18.6 ms step as
64 GCDs. This is the claim §5.4 and §5.5a were built to test, and it holds on **steady-state step
time**, not merely on construction — the distinction that invalidated the previous campaign's
headline.

**The step metric: mean of a steady-state window.** The field convention for a time-stepping code
is the mean of a window with setup *and* a warm-up excluded — GROMACS ships `-resetstep` /
`-resethway` for exactly this. Tick 1 is construction and is never averaged in. But excluding
tick 1 *alone* is not enough here: construction spills past it at large `w`, and a mean over
ticks 2…100 reads **1,693 ms against an 18.6 ms step** at `K`=4000 `w`=2048 — 91× wrong.

`WARMUP_TICKS`=10 is measured, not chosen. Across the 44 runs in both campaigns the mean and the
median of the window agree to **0.09 %** (median across runs) once 10 ticks are dropped, against
5.59 % at W=2; only three runs then exceed 1 %. That agreement is the test — when a mean and a
median of the same samples coincide, the window has cleared the tail. The three exceptions
(`K`=4000 `w`=2048 at +4.0 %, strong `P`=2048 at +3.4 %, `K`=1000 `w`=32 at +1.6 %) settle later
than tick 10 and are **reported rather than trimmed harder**: widening the window to cover them
would cost 30 % of the samples at every other point. `step_median_s` sits beside `step_s` in the
final CSVs so the gap is checkable per point.

**All the loss is in the ramp, and it is geometric, not a scaling failure.** Efficiency falls from
100 % to the plateau over `w`=1 → 64 while the peer count climbs 0 → 26. A 1-GPU run has no halo
at all; each added dimension of the rank grid adds neighbours until every dimension is ≥ 3 and the
26-neighbour stencil closes. Once it closes there is nothing left to add, which is precisely why
the curve goes flat and stays flat — the cost is bounded by the stencil, not by `w`.

**Efficiency rises with `K`: 66 % → 77 % → 83 %.** This is the surface-to-volume argument showing
up in the measurement. Raising `K` at fixed neurons/GPU adds *volume* work (more synapses to
integrate) while the halo — the set of remote somas whose spikes must cross a rank boundary — is
set by geometry and grows far more slowly: ghost somas run 37 k / 53 k / 81 k for a 4× span of `K`.
So communication falls from 33 % of the step at `K`=1000 to 15 % at `K`=4000, and the plateau sits
correspondingly higher. **The harder the science, the better this scales** — the opposite of the
strong-scaling regime in §5.5c, where shrinking per-rank work drives comm to 80 % and the curve
reverses.

**Two honest caveats.**

1. **Construction does not scale as cleanly as the step.** It is roughly flat over most of the
   sweep (~207 s / ~414 s / ~826 s for `K`=1000/2000/4000) but excursions appear at the top end —
   `K`=4000 hits 1481 s at `w`=1024 against 905 s at `w`=512 and 1069 s at `w`=2048, and `K`=1000
   reaches 259 s at `w`=2048 against ~210 s typical. It is non-monotonic, so it is run-to-run
   variance in an MPI-heavy phase rather than a trend. The step time is unaffected at the same
   points (18.614 vs 18.585 ms), which is itself the argument for reporting the two separately.
2. **At `w`=2048 construction spills past tick 1.** `K`=4000 `w`=2048 spends 111 s on tick 2, 50 s
   on tick 3 and 3.7 s on tick 4 before settling to ~18.6 ms, against ≤ 1.7 s at every other
   worker count; `K`=1000 shows the same shape at ~125 s per tick. This is what the 10-tick
   warm-up window exists to clear, and at that one point it does not fully clear it — the residual
   is the 4.0 % mean-over-median gap flagged in the table. The tail is reported rather than
   trimmed, since it is a real cost of starting a 2048-rank run.

**Cross-check against the strong campaign.** `K`=1000 at `w`=16 (12,500 neurons/GPU) steps in
5.622 ms; §5.5c at `P`=16 (12,800 neurons/GPU) steps in 5.229 ms — **7.5 % apart** despite the
weak point carrying 2.4 % *fewer* neurons. Both reproduce their own archived baselines to ~1 %, so
this is a stable configuration difference, not noise: 12,500 and 12,800 factor into different
near-cube neuron tiles, giving the two runs different halo surface-to-volume ratios at the same
rank count. Worth knowing when quoting a single "step time at 16 GPUs".

**A measurement note on the 1-GPU baseline.** SAGESim fuses all ticks into one kernel launch when
`num_workers`==1, and its timing record is written once per launch — so the original `w`=1 runs
produced a single row covering construction plus all 100 ticks, with no step time to read.
`sagesim/model.py` now skips fusion when `verbose_timing` is on, and the three `w`=1 points were
re-run (job 5120512). Unfusing also turned out to be **faster**, not a handicap: 4.18 vs 57.8 ms
per tick at `K`=1000 and 7.49 vs 64.3 ms at `K`=2000, with construction matching to 1 % at
`K`=2000 and `gpu_sync` totals to within 3 %, so the GPU does the same work either way. The
fused fast path is left in place for production, but it is not earning its keep at these sizes.

### 5.5b Implemented: spatial economical small-world — the realistic-scenario software test

The **orthogonal** deliverable (Deliverable 2): *not* weak scaling, but a **bio-realistic** network
to stress-test the software under realistic connectivity and confirm realistic dynamics. In
`brunel.py` as `spatial_smallworld_partition` (a peer of `brunel_partition`; same post-owns
contract, reuses the §5.5a coordinates). Follows Bassett & Bullmore 2017 (*Small-World Brain
Networks Revisited*) — the **economical small-world**: `P(i,j) ~ f(distance)` local + Watts-Strogatz
shortcuts.

- **Distance-dependent, probabilistic, variable in-degree.** Each pair connects with probability
  `amplitude · f(distance)` over a truncation ball (`_draw_smallworld_edges`); `f` = Gaussian
  (default) / exponential / power-law (`_kernel_weights`). In-degree is **variable** (degree
  heterogeneity / hubs), mean ≈ `mean_in_degree` (subject to kernel feasibility — a very tight
  inhibitory kernel caps how many inhibitory partners fit locally).
- **Distance-dependent E/I.** Tighter inhibitory kernel (`kernel_width_inh < kernel_width_exc`) →
  inhibition is more local (verified: I-sources sit closer than E-sources). Needs a **fine E/I
  spatial pattern** (`_is_excitatory_spatial`, a hash of the id giving ~`excitatory_fraction` E,
  finely interleaved so any small ball has both) — a pure function of the id, so every rank agrees
  on a remote source's sign.
- **Watts-Strogatz shortcuts.** A fraction `longrange_fraction` (β) of edges are `Poisson`-count
  random long-range shortcuts (uniform-global) — the short-path-length term. **This is why it does
  NOT weak-scale:** β=0 → sources stay in the 26-neighbor halo (peers ≤ 26); β>0 → out-of-halo
  shortcuts appear and the peer count **climbs toward M−1** (verified: at M=64, β=0/0.05/0.3 →
  peers 26/63/63, long-range edges 0/3/18%). Growing peers are expected — the realistic story, not
  a failure. Variable-in-degree synapse ids use a `max_in_degree` stride.
- **Validation (the deliverable, not a flat curve):** (a) **small-world index** σ = (C/C_rand)/
  (L/L_rand) via `validate_smallworld.py` — measured σ ≈ 4 at β=0 (small-world; a 3D high-degree
  lattice already has short paths, so σ is maximal at low β and declines toward random as β→1);
  (b) **AI firing regime** (mean rate few Hz, CV_ISI ≈ 1) via `realistic_run.py --validate-ai`;
  (c) a representative **at-scale** distributed run showing the software handles distance-dependent
  + long-range connectivity end-to-end (report the emergent, growing peer/ghost profile).
- **Status.** Implemented + unit-verified: `tests/test_brunel_spatial_smallworld.py` (variable
  in-degree, inhibition-more-local, β=0 lattice vs β>0 shortcuts + peer growth, fine E/I, no
  autapses, negative inhibitory weight, unique ids, kernel forms). Driver `realistic_run.py`;
  σ-validator `validate_smallworld.py`.

---

### 5.5c Implemented: strong scaling on the same 3D stencil — fixed problem, growing `P`

The companion to §5.5a, run on the **same wiring convention** so the two curves describe one
code rather than two experiments. Driver `scaling_analysis/strong_scaling.py`, runner
`strong_3d_chunk.sh`, analysis `analyze_strong.py` (consolidation + figures in one file).

**Both campaigns share one measurement path** (`scaling_analysis/scaling_diagnostics.py`) so
their numbers are comparable by construction rather than by coincidence. The standard file set
is six: `weak_scaling.py` / `strong_scaling.py` (drivers), `weak_3d_chunk.sh` /
`strong_3d_chunk.sh` (runners, `-p batch`), `analyze_weak.py` / `analyze_strong.py` (analysis),
plus the shared `scaling_diagnostics.py`. Everything superseded is under
`scaling_analysis/archive/2026-07-28_pre_unification/`.

**Timing is recorded, not decided.** Every run writes a per-tick CSV to `outputs/ticks/` — one
row per tick, across-rank mean/max/min of every timer — and applies **no warm-up window**. The
window is a `--warmup-ticks` flag on the *analysis* scripts (default 2). This matters because a
window fixed at collection time cannot be revisited: the first strong-scaling campaign excluded
tick 1 only, left tick 2's warm-up inside every mean, and reported the ghost exchange at 94.5 %
of the step where the steady-state value is ~57 %. That is now a re-analysis, not a re-run.

- **What is held fixed.** Total `N = 204,800` neurons, `K = 1000`, `topology="torus3d"`, auto
  radius `R = 8`, and the weak campaign's dynamics verbatim (`g=5.0`, `J_E=0.02581`,
  `delay=1.5`, 10 Hz Poisson, seed 42, `ticks=30`). Only `P` varies, `16 → 2048`, at a uniform
  **8 ranks/node**. Per-rank work falls as `1/P`; the ideal curve is the linear diagonal.
- **Why `torus3d` and not `global`.** The same D4 argument as §5.5a: our exchange is
  point-to-point, so per-tick cost is indexed by *peers* and *ghost volume*. With global wiring
  a rank's peer count grows toward `P−1` and the measurement reports the wiring rather than the
  machine. Same convention ⇒ the two curves compose.
- **Baseline is `P = 16`, not 1 GPU.** 16 is the smallest worker count whose per-rank load
  (12,800 neurons ≈ 12.8 M synapses) is the load already proven safe at 8 ranks/node in the weak
  campaign. A 1-GPU baseline for this `N` would need ~205 M synapses on one GCD, which does not
  fit; reporting speedup against a point that never ran would be fiction. The figures name the
  baseline on the axis label.

| `P` | nodes | neurons/GPU | rank grid | tile | peers | regime |
|---|---|---|---|---|---|---|
| 16 | 2 | 12,800 | 2×2×4 | 20×20×32 | 11 | ramp |
| 32 | 4 | 6,400 | 2×4×4 | 16×20×20 | 17 | ramp |
| 64 | 8 | 3,200 | 4×4×4 | 10×16×20 | 26 | plateau |
| 128 | 16 | 1,600 | 4×4×8 | 10×10×16 | 26 | plateau |
| 256 | 32 | 800 | 4×8×8 | 8×10×10 | 26 | plateau |
| 512 | 64 | 400 | 8×8×8 | 5×8×10 | 44 | breakdown |
| 1024 | 128 | 200 | 8×8×16 | 5×5×8 | 62 | breakdown |
| 2048 | 256 | 100 | 8×16×16 | 4×5×5 | 96 | breakdown |

(Peer column is **measured**. `predicted_peer_count` counts a bounding box of tiles while the
stencil is a sphere, so it is exact while the span is one tile — the whole weak campaign, and
this curve's ramp and plateau — and an upper bound past that: it predicted 74 / 124 at
P = 1024 / 2048 against the measured 62 / 96.)

- **Three regimes, and the one that is new.** The low-`P` **ramp** (11, 17) is the same effect as
  §5.5a's small-M ramp — a *rank-grid* dimension below 3 collapses the torus wraparound — so the
  baseline itself sits in it. The **plateau at 26** is the healthy stencil. The **breakdown** at
  `P ≥ 512` is what strong scaling adds and weak scaling cannot show: holding `N` fixed shrinks
  the tile against a *fixed* radius until the tile edge falls under it, at which point the
  bounded-peer premise the convention rests on stops holding and the peer count climbs. Those
  points are reported as a distinct regime (shaded on every figure), not as more of the curve.
  And the regime boundary is not just geometric — **the speedup curve turns at exactly the same
  place** (see the results below).
- **Metric: the MEDIAN steady-state step time.** Not `simulation_time`, and not the mean.

  *Not `simulation_time`*: tick 1 carries ghost discovery, the GPU buffer build and `comm_init`,
  and it dominates that phase completely — measured at **99.5–99.9 % of `simulation_time` at
  every worker count** (at `P=16`, 209.6 s of a 211.7 s simulate, against 0.52 s for all 100
  steps). A curve built on it reports buffer allocation, not simulation. `tick_first_s` is
  reported separately rather than discarded.

  *Two warm-up ticks, not one*: tick 2 is the first tick to actually exchange ghost data and
  touch the lazily allocated buffers, and it still runs 3×–739× the typical tick (2.73 s against
  a 0.0037 s median at `P`=256). `--warmup-ticks` (default 2) sets the excluded window; both
  excluded ticks are reported as `tick_first_s` and `tick_warmup_s` rather than dropped. This is
  a *deterministic* second warm-up, not network jitter — an early reading of the campaign
  mistook it for a stall tail.

  *Not the mean*: with warm-up left in, the mean tracks it rather than the code (the comm share
  at `P`=256 read 0.945 of the step where the steady-state value is ~0.57). The median is immune
  — one extra sample cannot move a median of ~100 — which is why the speedup curve was correct
  even before the window was widened. The median is the headline, with p10, mean and the worst
  steady tick beside it. `ticks=100`, which costs under a second per run because tick 1
  dominates wall time anyway.
- **Per-step decomposition.** The SAGESim timers are **nested**, so they are recorded at two
  levels and never summed flat. On a steady-state tick the `data_prep` window contains exactly
  one thing — `exchange_ghost_data()` (`SAGESim/sagesim/model.py:1692`) — so `data_prep` *is* the
  ghost exchange and is reported as `comm`. Level 1 (`comm | compute | gpu_sync | write_back |
  other`) is disjoint and sums to the step time; level 2 (`pack | exchange | unpack |
  comm_other`) opens `comm`; `wait` is a subset of `exchange` and is reported alongside to split
  latency from load imbalance. The **"if the exchange were free"** overlay on the speedup figure
  is derived from these timers — no comm-off ablation run is needed.
- **Two caveats, both accepted.** (1) The global neuron grid is derived from `(P, npp)`, so its
  *shape* drifts across `P` — volume stays exactly `N` and every neuron keeps exactly `K` sources
  within radius 8, so the workload is statistically identical at every point, but the specific
  edge set is not. (2) At `P ≥ 512` essentially every source is remote (ghost:local reaches ~88:1
  at `P = 2048`); those points measure the latency-bound floor, which is why they are included.
- **Preflight.** `python strong_scaling.py --dry-run 16 32 … 2048` prints the table above from the
  real generator helpers with no MPI, GPU or disk, and exits non-zero if any point violates
  `radius < min(A,B,C)` or leaves the ball short of `K`. Strong scaling re-derives the tile at
  every point, so this is where a bad `--total-neurons` is caught for free.

#### Results (campaign complete, re-run 2026-07-30 — 8/8 points, no failures or retries)

Jobs 5114878 (32 nodes, 9m46s, `P`=16…256) and 5114882 (256 nodes, 1m52s, `P`=512…2048).
`outputs/strong_3d_final.csv`; figures `figures/strong_3d_{speedup,breakdown,comm}.png`.

Two timings are reported, never one conflated number. **Construction** is tick 1 — the GPU
buffer build plus ghost-topology discovery. **Step** is the mean of ticks 11…100, a simulation
step (window justified in §5.5a). At `P`=16 construction is 213 s against a 5.2 ms step, so a
single "simulation time" is ~99.8 % construction.

| `P` | n/GPU | construction | step (mean) | step speedup | efficiency | end-to-end speedup | peers | ghost:local | comm % | B/peer |
|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 12,800 | 213.33 s | 0.005246 s | 1.00× | 100 % | 1.00× | 11 | 2.98 | 27 % | 69,450 |
| 32 | 6,400 | 112.66 s | 0.004539 s | 1.16× | 58 % | 1.92× | 17 | 4.07 | 42 % | 30,643 |
| 64 | 3,200 | 59.03 s | 0.004301 s | 1.22× | 30 % | 3.67× | 26 | 5.95 | 52 % | 14,643 |
| 128 | 1,600 | 35.41 s | 0.004034 s | 1.30× | 16 % | 6.19× | 26 | 8.45 | 53 % | 10,404 |
| 256 | 800 | 19.77 s | 0.003642 s | **1.44×** | 9.0 % | 10.01× | 26 | 12.30 | 57 % | 7,566 |
| 512 | 400 | 16.56 s | 0.004405 s | 1.19× | 3.7 % | 13.45× | 44 | 19.19 | 65 % | 3,490 |
| 1024 | 200 | 15.68 s | 0.005120 s | 1.02× | 1.6 % | **13.88×** | 62 | 29.46 | 71 % | 1,901 |
| 2048 | 100 | 14.97 s | 0.006895 s | **0.76×**† | 0.6 % | 13.40× | 95 | 46.46 | 80 % | 972 |

† `P`=2048 settles at ~tick 51, later than any other point, and reads 3.4 % above its own median.
Reported rather than trimmed; see *The step metric* in §5.5a.

The earlier ⚠️ note on this table is **resolved**: the contaminated `comm %` column (44 % → 95 %)
is replaced by the steady-state share above, **27 % → 80 %**, matching the 27 → 81 % the warning
predicted. The speedup peak is 1.44× at `P`=256 whichever statistic is used.

The fix was the window, not the statistic. At `P`=256 exactly three ticks out of 99 (ticks 2–4 at
864/945/628 ms against a ~3.6 ms step) carry 87 % of a mean taken over ticks 2…N, so that mean
described the settling ticks rather than a step. Opening the window at tick 11 removes them, and
the mean and median of what remains then agree to 0.03 % at this point — which is the test that
the window is wide enough (§5.5a). The components use the same window and sum to the step with a
~0.105 ms residual at every `P`, so the decomposition stays additive.

**The curve peaks and reverses.** Step speedup tops out at **1.44× at `P`=256**, then falls to
**0.76× at `P`=2048 — slower than the 16-GPU baseline**. The turn lands exactly where the peer
count leaves 26, so the geometric breakdown predicted before the runs shows up in the timing.

**Why, in one number.** Per-tick `GPU compute` is **0.370 ms at all eight points**, across a
**128× span of per-rank work** (12,800 → 100 neurons/GPU) — it does not move in the third
decimal. The kernel is launch-bound *at these problem sizes*: there is no compute left to
parallelize. `gpu_sync` is the only shrinking component (3.34 → 0.87 ms, 3.8× over 128×), while
the ghost exchange grows from 1.41 ms to 5.32 ms — 27 % → 80 % of the step — even as
bytes-per-peer falls 69 KB → 972 B. Communication grows while the data shrinks, so the exchange
ends **latency-bound, not bandwidth-bound**.

**Report both headline numbers, labelled.** **Time to solution** reaches **13.9× on 64× the
GPUs** (`P`=16 → 1024, 257.4 s → 18.5 s) and is still 13.4× at `P`=2048 — but that is measuring
**construction**, which is 99.8 % of a 100-tick run. The **simulation step** peaks at 1.44× (9 %)
and then regresses. Quoting the 13.9× alone would be a construction benchmark wearing a
simulation label; quoting the 1.44× alone would understate a code whose dominant cost does
parallelise. Both go on the same figure (`paper_figures/strong_speedup.py`).

**The limitation, plainly.** Network construction (load + GPU buffer build) dominates the run,
and per-tick GPU compute is fixed-cost-bound from the first point of the sweep, so the
strong-scaling window is already closed at the baseline. This is the current software's
behaviour, reported as such. **Weak scaling (§5.5a) remains the headline scaling result**;
this section exists for completeness.

**No K=4000 strong curve was run, deliberately.** `K` is coupled to stencil width — a radius-8
ball holds 2,108 candidates, so `K`=4000 forces radius 13, and the peer count would run
11 → 99 with no 26-plateau at all. A second curve would therefore differ in *two coupled
variables at once* (4× work **and** a wider stencil), leaving any divergence unattributable.
`K`-dependence is already the weak campaign's headline (`K` ∈ {1000, 2000, 4000}). The
launch-bound claim is scoped to the 128× work span actually measured; a single `K`=4000 point
at `P`=16 (~1.2 node-hours) would extend it if ever challenged.

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
- ~~**No spatial / distance-dependent connectivity**~~ — **superseded by the D4 discussion.**
  Milestone 1 *is* spatial: the measured weak-scaling setup is the `topology="torus3d"` 3D
  spatial-radius stencil (§5.5a), which draws **uniformly within a hard-cutoff ball**. (The
  `torus2d` tile-block of §5.4 was the 2D predecessor and is superseded.) A *continuous
  distance* connectivity kernel (Gaussian) and *long-range* connectivity are **not** part of
  the scaling campaign; they were since implemented in the separate `spatial_smallworld`
  generator (§5.5b), which is a bio-realism deliverable and is not measured by these curves.
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

### Figures produced from the runs (§5.5d)

Two layers, deliberately separate. `analyze_weak.py` / `analyze_strong.py` write **diagnostic**
figures to `scaling_analysis/figures/` — screen-sized, multi-panel, for reading while a campaign
is being debugged. `scaling_analysis/paper_figures/` writes the **print** set: IEEE
single-column, 600 dpi PNG + Type-42 PDF pairs, one figure per file so LaTeX `subcaption` composes
them. The print layer reads the two final CSVs and derives nothing the analysis layer already
derived; `make_all.py` exits non-zero if any grid point is missing, and `--print` dumps every
plotted value as text.

Style is ported from `GGap/SC2026/figures/scripts/` so the two projects' figures sit together in
a proceedings. Run: `cd scaling_analysis/paper_figures && python make_all.py`.

| | script | source | the claim it supports |
|---|---|---|---|
| **F1** | `weak_efficiency.py` | `weak_3d_final.csv` | efficiency 1→2048 GPUs; ramp shaded, plateau flat, peers on the twin axis |
| **F2** | `weak_step_time.py` | `weak_3d_final.csv` | the same in absolute ms — the number a reader quotes |
| **F3** | `phase_bars.py` | `weak_3d_final.csv` | a 100-tick run is 99.8 % one-time cost; licenses F1's metric |
| **F4** | `strong_speedup.py` | `strong_3d_final.csv` | step peaks 1.44× and reverses; time to solution reaches 13.9× |
| **F5** | `strong_breakdown.py` | `strong_3d_final.csv` | why: ghost exchange 27 %→80 %, compute pinned at 0.370 ms |
| **F6** | `setup_amortization.py` | `weak_3d_final.csv` | ~39–55k ticks before setup stops dominating |

**The peer count is the diagnostic, and it rides on the primary figures rather than getting its
own.** D4's argument — O(1) peers, not O(M) — is what makes the plateau a plateau, so it belongs
on the same axes as the curve it explains: F1 and F4 both carry `peers_mean` on a twin y-axis,
and both shade the regime where the bounded-peer premise does not hold (`w` < 64 on the weak
sweep, `P` ≥ 512 on the strong one). A standalone peer plot would separate the claim from its
evidence.

**Why the weak sweep starts at `w`=1 and the efficiency curve falls before it flattens.** This is
the published convention for point-to-point codes, not a workaround: WOMBAT (Mendygral et al.
2017, *ApJS* 228:23) reports off-node communication saturating between 3 and 27 nodes for the same
geometric reason — at ≥3 ranks per dimension every rank acquires its full set of unique 3D
neighbours — with update times "nearly flat for larger node counts" past it. What the convention
does not permit is plotting the ramp unmarked, so F1 shades it and rules the axis at `w`=64, and
every caption quotes **both** efficiency numbers (against `T(1)` and against `T(64)`).

**Why F1's efficiency is built on the step and not on `simulation_time`, though GGap's is.**
GGap's `_scaling_common.py:derive_metrics` uses `simulation_time = first_tick + steady_state`,
defending it as more honest than a bare steady-state tick. That holds at GGap's ratio — their
first tick is **55–58 %** of `simulate()`, a genuine blend. Ours is **99.70–99.88 %**, so the same
curve is construction with a rounding error of simulation: non-monotonic, no knee at `w`=64, no
plateau (`K`=4000 reads 100, 75.8, …, 66.5, **40.6**, 56.3 %, where the 40.6 % is the construction
variance of caveat 1, not a scaling signal). Publishing that shape as weak scaling is exactly what
invalidated the previous campaign. F1 therefore draws the end-to-end curve **too** — thin, dashed,
named, visibly subordinate — and F3/F6 show the split it comes from.

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
