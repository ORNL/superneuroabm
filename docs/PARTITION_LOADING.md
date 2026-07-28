# SuperNeuroABM: Loading & Distributing Networks

How SuperNeuroABM ingests a partitioned network, what file format it expects, the
**ownership rule** that makes a network distributable, and the **local-load-save**
pattern that keeps building large networks cheap.

This is the *framework* contract — dataset-agnostic. For a concrete end-to-end example
(the Cora citation SNN, built with METIS), see
[`../../superneuroabm_sgnn/SGNN_CORA.md`](../../superneuroabm_sgnn/SGNN_CORA.md).

## Two loaders — pick by how you describe connectivity

There are two whole-model loaders. They build the **same** kind of model and share all
property/bookkeeping logic; they differ only in **what each rank's file says about
connectivity**, and consequently in the placement constraint they impose:

| | `load_post_owned()` (Method 1) | `load_from_adjacency()` (Method 2) |
|---|---|---|
| You describe a synapse by | its **endpoints** `pre`/`post` | its **neighbor list** (and each soma's too) |
| The loader | **derives** all connectivity, incl. each post-soma's incoming list | **reads** the neighbor lists verbatim |
| Placement constraint | **post-owns**: every synapse must be co-located with its post-soma (§3) | **none** — a synapse may be on `rank(pre)` while its post-soma is on another rank |
| Cross-rank reach | a synapse's remote **pre** soma | **any** neighbor (remote pre soma *or* remote incoming synapse), named in `remote_ranks` |
| Use when | the natural SNN/NEST convention fits and the partitioner can honor post-owns | you need to place synapses freely (e.g. partition the synapse network itself) |

**Why two.** Method 1's file names a synapse only by `pre`/`post`, so a post-soma
discovers its incoming synapses by *scanning the synapses in its own file* — it can only
find ones that are local. There is **no way to name a remote incoming synapse** in that
schema, which is exactly why post-owns is *required*, not just recommended. Method 2
lets the post-soma's `neighbors` name its incoming synapses **explicitly**, so a remote
one can be declared in `remote_ranks` and fetched by ghost exchange — the same machinery
Method 1 already uses for a remote pre soma. Method 2 therefore **releases** the
constraint at the cost of the producer writing connectivity in full.

§§1–6 below describe Method 1 (post-owns) in detail — the common case — then §7 covers
Method 2 as the constraint-releasing alternative.

---

## 1. Responsibility boundary

| Responsibility | Owner |
|---|---|
| Define the network: which somas/synapses exist, their breed/config/overrides | **User** |
| Decide which rank owns which soma; assign globally-unique agent IDs | **User** (METIS, a rule, a generator — any tool) |
| Produce each rank's local data (the three dicts below) | **User** |
| **Guarantee the partition is correct and complete** (post-owns honored, every synapse present exactly once) | **User** |
| Read local data, compute property tensors, build connections, set up ghosts | **SuperNeuroABM** `load_post_owned()` |
| Manage agents on GPU, MPI ghost exchange each tick | **SAGESim** |

SuperNeuroABM does **not** partition graphs, assign IDs, or decide placement. It
**reads** the ownership you encode and builds the model — it never infers ownership.

> **The producer owns correctness.** `load_post_owned()` trusts the files. It builds
> exactly what each file lists and **does not validate** that your partition is correct
> or complete (see §3.3 for what is and isn't detectable, and why). If the files are
> wrong, you get a silently miswired model, not an error. Verifying the partition is
> **your** responsibility, and the only place with enough information to do it fully is
> your producer (where the whole network is in one process).

---

## 2. The format: three dicts per rank

A rank's network is a single dict with **exactly three keys**:

```python
{
  "somas":    [ {"id", "breed", "config", "overrides"}, ... ],   # somas this rank OWNS
  "synapses": [ {"id", "pre", "post", "breed", "config",         # synapses this rank OWNS
                 "learning_rule", "learning_rule_config", "overrides"}, ... ],
  "remote_ranks": { agent_id: owner_rank, ... },                 # every non-local neighbor
}
```

- **Soma** — `id` (globally unique int), `breed` (e.g. `"lif_soma"`), `config` (e.g.
  `"config_0"`), optional `overrides` (`{"hyperparameters": {...}, "internal_states":
  {...}}`).
- **Synapse** — `id`, `pre` (presynaptic soma id, or `-1` for external input), `post`
  (postsynaptic soma id), `breed`, `config`, optional `learning_rule` /
  `learning_rule_config` / `overrides` (e.g. `{"hyperparameters": {"weight": 14.0}}`).
- **`remote_ranks`** — maps every neighbor agent that is **not local** to the rank that
  owns it. Under the ownership rule (§3) the only non-local neighbor is a synapse's
  remote **pre** soma.

IDs are a single flat space: somas and synapses are both SAGESim agents and must not
collide (a common scheme: synapse ids start at `num_somas`).

Only **pickle** (`.pkl`/`.pickle`) is read. A legacy graph-centric schema
(`nodes`/`edges`/`source`/`target`) is rejected with a clear error — there is no
translation layer.

> A single-rank model is just the `K = 1` case: one file, `remote_ranks = {}`.

---

## 3. The ownership invariant: **post-owns**

> **Every synapse is owned by the rank of its post (postsynaptic) soma.**
> A synapse appears in **exactly one** rank's `synapses` list. Its pre soma may be
> remote; if so it is named in `remote_ranks`. Nothing is listed twice.

### Why post, not pre

Creating a synapse wires three neighbor relationships:

```
(synapse → pre_soma)    synapse reads the pre-soma's spike   (slot 0)
(synapse → post_soma)   synapse reads the post-soma's spike  (slot 1, for STDP)
(post_soma → synapse)   post-soma sums the synapse's current (the soma's input)
```

The third one mutates a **third agent's** neighbor list — the post-soma's. SAGESim's
`connect_agents(A, B)` only ever writes `A`'s neighbor list, so **A must be local**.
The post-soma is the `A` in `(post_soma → synapse)`, so the post-soma must be local
when the synapse is created. Owning the synapse on `rank(post)` makes all three writes
local — `synapse` and `post_soma` are both on this rank; only `pre_soma` may be remote
(and it only ever appears as a *read target*, never as a writer). This is the
NEST/Brunel convention and it removes the need to duplicate any synapse across ranks.

### What `load_post_owned()` does with it

`load_post_owned()` (`superneuroabm/model.py`) **reads, never infers**: it creates every
soma and **every** synapse listed (no locality test, no ghost split), building one
adjacency map handed to SAGESim in bulk:

```python
for syn in synapses:                          # each listed synapse is OWNED here
    adjacency[syn_id] = [pre_id]              # slot 0 = pre  (-1 for input)
    if post_id != -1:
        adjacency[syn_id].append(post_id)    # slot 1 = post
        adjacency[post_id].append(syn_id)    # post-soma claims its incoming synapse
```

The slot order is positional and meaningful: the synapse kernel reads slot 0 = pre,
slot 1 = post. (Self-loops, `pre == post`, collapse the two slots and are therefore
not supported — producers must avoid them.) The local post-soma's neighbor list is the
set of synapses flowing into it. A remote `pre` only ever appears as a *value* in the
adjacency, never as a key, so it gets no local container — its rank comes from
`remote_ranks`, passed straight to SAGESim as `remote_agent_ranks`.

The payoff: each rank's file already contains exactly the synapses it simulates, and
the only cross-boundary reference is each synapse's remote pre soma, fetched at run
time via ghost exchange. No global agent→rank table, no duplicated synapses.

### 3.3 Validating the invariant is **the producer's job** — the loader cannot

`load_post_owned()` builds a post-soma's incoming-synapse list **purely from the synapses
in that rank's file** (`adjacency[post_id].append(syn_id)` above). This is correct
*only because* post-owns guarantees that **all** of a soma's incoming synapses live in
the same file as that soma. The loader relies on this; it does **not** check it. If the
contract is broken, there are two failure modes — and they are not equally detectable:

Note one asymmetry that the table below leans on: **soma ownership is declared
explicitly** in each rank's `somas[]` — the loader never *infers* a soma's location from
its synapses. That explicit declaration is exactly what gives the loader a local
reference to check synapse *placement* against (Bug A). What it does **not** give is a
reference for synapse *existence* — no list says "soma X should have N incoming
synapses" — which is why a dropped synapse (Bug B) stays invisible.

| | Bug A — synapse in the **wrong** rank | Bug B — synapse **missing** from every rank |
|---|---|---|
| What | A file lists a synapse whose post-soma is not owned by that rank | A synapse that should exist was never written to any rank's file |
| Symptom | A post-soma's neighbor list gets an entry on a rank that doesn't own it; the real owner's list is short one | A post-soma is built with an **incomplete** neighbor list — it silently integrates fewer inputs |
| Detectable by the **loader**? | **Yes, locally** — the synapse's `post` is not in this rank's somas. `load_post_owned()` **enforces** this: a synapse whose `post` is not a local soma raises immediately. | **No, ever** — absence leaves no local trace. A rank cannot know a soma "should have had" more incoming synapses; nothing in its file says so. |
| Must be guaranteed by | the producer (the loader also re-checks it) | **the producer, only** |

**Why the loader can't catch Bug B.** A partitioned rank has, by design, lost the global
view — that is the entire point of partitioning. It sees only its own file. "A synapse
is missing" is a statement about the *global* network, and no amount of local inspection
can recover it. Detecting it requires comparing against the complete network, which
exists in exactly one place: **your producer, at partition time, before the files are
split.** Therefore:

> **You must verify partition completeness in your producer.** After you assign every
> synapse to a rank and before writing the files, assert that (1) every synapse in the
> global network was written to exactly one rank's bucket — a simple conservation count —
> and (2) each synapse's bucket is `rank(post)`. Neither check is something
> `load_post_owned()` can do for you. See `SGNN_CORA.md` for a worked producer that does
> this.

The loader's Bug-A guard (cheap, catches misplacement immediately) is a convenience, not
a substitute: it narrows the window, it does not close it. Completeness (Bug B) is
unconditionally yours.

---

## 4. Building large networks: local load & save

For a network too large to hold globally, the scaling principle is:

> **Each rank produces only its own local data** (its three dicts), from its own slice
> of the problem — never materializing the global network anywhere.

A synthetic example ships with the framework:
`superneuroabm/synthetic_networks.py::generate_and_save_local_partition` builds a
Brunel balanced random network **one rank at a time** — each rank draws only the
in-edges of *its* neurons (so generation is `O(neurons_per_rank × in_degree)`, never
`O(N_global)`), and writes its own `partition_{r}.pkl`. No MPI is needed during
generation; ranks are independent.

Everything after "produce local data" is **post-processing, and the user's choice**:

- **Save to files**, then launch a distributed run that loads them — best when the
  build is expensive or reused across runs, or when build and run happen on different
  resources. This is what `load_post_owned()` consumes.
- **Gather directly** — build each rank's dicts in the same process that will simulate,
  and feed them to the model without a disk round-trip — best for one-shot runs.

Either way the *contract* is the same three dicts with the post-owns invariant. The
framework does not care whether they came from disk or from memory.

> **The one global thing each rank needs is its `remote_ranks` map** — the owner rank
> of each remote pre soma it references. A generator computes this locally from its own
> ownership rule (e.g. `rank = soma_id // neurons_per_rank`); a partitioner emits it
> while writing each rank's file. It is **boundary-sized**, not `N_global`-sized.

---

## 5. Distributed run

Every rank runs the same code on its own file; `setup` and `simulate` are collective.

```python
from mpi4py import MPI
from superneuroabm.model import NeuromorphicModel

rank = MPI.COMM_WORLD.Get_rank()

model = NeuromorphicModel()
model.load_post_owned(f"partition_{rank}.pkl")   # rank-local somas + synapses
model.setup()                                    # GPU buffers + ghost topology discovery
model.simulate(ticks=100)                        # GPU kernels + MPI ghost exchange / tick
```

- `setup()` allocates GPU buffers and, on the first tick, discovers ghost topology from
  `remote_ranks` — **boundary-scoped** (proportional to distinct boundary neighbours,
  not `N_global`) — and caches it.
- `simulate()` runs the kernels and, every `update_data_ticks`, exchanges only the
  boundary-visible agent state across ranks. A synapse's remote pre-soma spike arrives
  before the synapse kernel reads it; no per-read communication.
- The number of `partition_{r}.pkl` files **must** equal the launched rank count.

### Injecting input and reading results: collective vs local

Both the input you inject and the results you read come in two flavours — a
**collective** form (every rank calls with the same id; convenient) and a **local**
form (only the owning rank acts; scalable):

| | Collective | Local |
|---|------------|-------|
| inject input | `add_spike(id, tick, value)` — **every** rank calls with the same id; one allgather under the hood, owner stores the spike | `add_local_spike(id, tick, value)` — only the **owning** rank calls; `KeyError` on a non-local id; no MPI |
| read property | `get_agent_property_value(id, prop)` (and `get_spike_times`) — **any** rank reads **any** agent (owner supplies via `comm.allgather`) | `get_local_agent_property_value(id, prop)` — only the **owning** rank; `KeyError` otherwise; no MPI |
| cost | one allgather per call | zero communication |
| use | convenient for small runs / any-rank access | the scalable path — caller knows the agent is local |

`simulate`/`reset` are collective regardless. Choosing local inject + local reads + a
single final `comm.gather` (instead of per-call allgathers) is what scales to large rank
counts — the caller resolves ownership from its own app metadata and touches only what
it owns.

> **Worked example.** The SGNN runners show both styles on the *same* partitions:
> `run_sgnn_sna.py` uses the collective accessors (`add_spike` /
> `get_agent_property_value` / `get_spike_times` — every rank calls with the same id),
> while `run_sgnn_sna_mpi_local.py` uses the local accessors (`add_local_spike` /
> `get_local_agent_property_value`) and assembles results with one final
> `comm.gather(root=0)`. Same model, same result; the local variant drops the per-call
> allgather and is the one to reach for as rank counts grow.

---

## 6. Construction paths are mutually exclusive

A model is built by **exactly one** path: `load_post_owned()`, `load_from_adjacency()`
(both whole-model, one shot), **or** incremental `create_soma`/`create_synapse`. Any
loader raises if the model already has agents, and the incremental calls raise after a
load — so you cannot mix them. The incremental path follows the post-owns rule (a
synapse's post is local on its rank, so the `(post_soma → synapse)` connection is always
made).

---

## 7. Method 2: `load_from_adjacency()` — explicit neighbors, no post-owns

Use this when you need to place synapses freely — e.g. to own a synapse on `rank(pre)`,
or to partition the **synapse network** itself rather than the soma graph. Instead of
`pre`/`post` endpoints, **every** soma and synapse carries its own `neighbors` list, and
the loader builds exactly what is listed.

### Schema (three dicts, like §2, but neighbor-based)

```python
{
  "somas":    [ {"id", "breed", "config", "overrides",
                 "neighbors": [incoming_syn_id, ...]}, ... ],   # the soma's incoming synapses
  "synapses": [ {"id", "breed", "config", "learning_rule",
                 "learning_rule_config", "overrides",
                 "neighbors": [pre[, post]]}, ... ],            # POSITIONAL slots
  "remote_ranks": { agent_id: owner_rank, ... },               # any cross-rank neighbor
}
```

- **Synapse `neighbors` is positional and load-bearing**: slot 0 = `pre`, slot 1 =
  `post` (the synapse kernel reads slot 0 for the incoming spike, slot 1 for STDP).
  `[-1]` is an external-input synapse (pre = -1, no post). The loader copies the list
  **verbatim** — it never sorts or dedups it.
- **Soma `neighbors`** is its set of incoming synapse ids, order-free (the soma sums over
  them). This is the list Method 1 *derives*; here you state it.
- **`remote_ranks`** now names **any** non-local neighbor — a synapse's remote pre soma
  (as before) **or** a soma's remote incoming synapse (the newly-allowed case). Ghost
  exchange fetches that neighbor's visible state each tick exactly as for a remote pre.

### How the constraint is released

A post-soma's incoming list is **given**, not derived from local synapses, so an incoming
synapse may live on another rank: list it in the soma's `neighbors` and declare it in
`remote_ranks`. The soma reads the remote synapse's current via ghost exchange — verified
end-to-end (a synapse owned by `rank(pre)` feeding a post-soma on another rank produces
the same result as the co-located build). SAGESim is unchanged; this uses the same
borrow-neighbor-state mechanism Method 1 relies on.

### What the loader validates (cheap, local) vs. what it cannot

`load_from_adjacency()` checks, before building: each synapse `neighbors` has length 1 or
2; every neighbor id is `-1`, a local agent, or named in `remote_ranks` (a dangling
reference raises). It **cannot** check completeness — that a soma's `neighbors` lists
*all* its incoming synapses — for the same reason as §3.3 (Bug B is invisible locally).
Completeness is the producer's job. With Method 2 the producer also owns slot order and
the full neighbor lists, so it carries *more* correctness responsibility than Method 1,
not less.

---

## What SuperNeuroABM does **not** do

- Partition a graph or decide placement — your tool (METIS, a generator, a rule) does.
- Assign agent IDs — encode them in the files.
- Infer ownership or connectivity — it reads the layout you provide (the post-owns
  `pre`/`post` of Method 1, or the explicit `neighbors` of Method 2).
- **Validate that your partition is correct or complete** — it trusts the files. A
  *missing* synapse is undetectable locally (§3.3); guaranteeing completeness is the
  producer's job. The loader may guard against a *misplaced* synapse locally, but that
  does not make the partition correct — only your producer can.
- Parse arbitrary formats — pickle only, the three-dict schema.
