# SuperNeuroABM: Loading & Distributing Networks

How SuperNeuroABM ingests a network with `load_from_file()`, what file format it
expects, the **ownership invariant** that makes a network distributable, and the
**local-load-save** pattern that keeps building large networks cheap.

This is the *framework* contract — dataset-agnostic. For a concrete end-to-end example
(the Cora citation SNN, built with METIS), see
[`../superneuroabm_sgnn/SGNN_CORA.md`](../superneuroabm_sgnn/SGNN_CORA.md).

---

## 1. Responsibility boundary

| Responsibility | Owner |
|---|---|
| Define the network: which somas/synapses exist, their breed/config/overrides | **User** |
| Decide which rank owns which soma; assign globally-unique agent IDs | **User** (METIS, a rule, a generator — any tool) |
| Produce each rank's local data (the three dicts below) | **User** |
| Read local data, compute property tensors, build connections, set up ghosts | **SuperNeuroABM** `load_from_file()` |
| Manage agents on GPU, MPI ghost exchange each tick | **SAGESim** |

SuperNeuroABM does **not** partition graphs, assign IDs, or decide placement. It
**reads** the ownership you encode and builds the model — it never infers ownership.

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

### What `load_from_file()` does with it

`load_from_file()` (`superneuroabm/model.py`) **reads, never infers**: it creates every
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
  resources. This is what `load_from_file()` consumes.
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
model.load_from_file(f"partition_{rank}.pkl")   # rank-local somas + synapses
model.setup(use_gpu=True)                        # GPU buffers + ghost topology discovery
model.simulate(ticks=100)                        # GPU kernels + MPI ghost exchange / tick
```

- `setup()` allocates GPU buffers and, on the first tick, discovers ghost topology from
  `remote_ranks` — **boundary-scoped** (proportional to distinct boundary neighbours,
  not `N_global`) — and caches it.
- `simulate()` runs the kernels and, every `update_data_ticks`, exchanges only the
  boundary-visible agent state across ranks. A synapse's remote pre-soma spike arrives
  before the synapse kernel reads it; no per-read communication.
- The number of `partition_{r}.pkl` files **must** equal the launched rank count.

### Reading results: collective vs local

Two ways to read an agent property after simulation:

| | Collective | Local |
|---|------------|-------|
| call | `get_agent_property_value(id, prop)` | `get_local_agent_property_value(id, prop)` |
| who can read | **any** rank reads **any** agent (owner supplies via `comm.allgather`) | only the **owning** rank; `KeyError` otherwise; no MPI |
| cost | one allgather per call | zero communication |
| use | convenient for small runs / any-rank reads | the scalable path — caller knows the agent is local |

`simulate`/`reset` are collective regardless. Choosing local reads + a single final
`comm.gather` (instead of per-read allgathers) is what scales to large rank counts —
the caller resolves ownership from its own app metadata and reads only what it owns.

---

## 6. Construction paths are mutually exclusive

A model is built **either** by `load_from_file()` (whole model, one shot) **or** by
incremental `create_soma`/`create_synapse` — not both. The incremental path follows the
same post-owns rule (a synapse's post is local on its rank, so the
`(post_soma → synapse)` connection is always made). `load_from_file()` raises if the
model already has agents, and the incremental calls raise after a file load.

---

## What SuperNeuroABM does **not** do

- Partition a graph or decide placement — your tool (METIS, a generator, a rule) does.
- Assign agent IDs — encode them in the files.
- Infer ownership — it reads the post-owns layout you provide.
- Parse arbitrary formats — pickle only, the three-dict schema.
