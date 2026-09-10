# Performance notes: where SuperNeuroABM's time and memory go

**Status: investigation. Nothing here is implemented beyond measurement hooks.**
The only code changes made for this document are instrumentation and the benchmark scripts
in `scaling_analysis/nest_comparison/`: SAGESim records its setup sub-step timings, its
per-property GPU allocation cost, and which conversion path each property took;
SuperNeuroABM records its `setup()` and columnar-loader phases. No allocation, padding, or
setup behaviour was changed.

This answers three questions asked in September 2026:

1. Does every agent storing its own copy of identical parameters explain the memory use?
2. Does it explain setup dominating wall time?
3. Why is Masquelier 2008 chunked, and why are we slower than NEST?

## TL;DR

The parameter-duplication hypothesis is **half right, and it is not the biggest problem.**

Measured at the single-GPU Brunel point (12,500 neurons, in-degree 1000, 12.5 M synapses,
100 ticks) on one RTX 3000 Ada:

| | SuperNeuroABM | NEST GPU | ratio |
|---|---|---|---|
| network construction (rule-generated) | 228-293 s, **now 76 s** | 0.054 s | 4,200-5,400x, now 1,400x |
| network construction (explicit edge list) | 228-293 s, **now 76 s** | 2.26 s | 101-130x, now 34x |
| ticks (99 steps) | 1.16 s | 0.014 s | 83x |
| end-to-end wall clock | 229-305 s, **now 77 s** | 4.5 s | 51-68x, now 17x |
| peak host RSS | 16,584 MB, **now 9,877 MB** | 206 MB | 81x, now 48x |
| GPU memory | 3,558 MB | 367 MB | 9.7x |
| bytes/synapse (GPU) | 284 B | ~12 B | 24x |

Both networks are silent (0 Hz), identical in size, and hold the same 12,512,500 synapses.
The second row is the honest one for a network that comes from a file rather than from a
generative rule; see the dedicated section below. SuperNeuroABM ranges are two runs; NEST GPU
is the median of three. The bold "now" figures are after the two first-tick fixes below,
median of three runs. SuperNeuroABM's
construction time is not stable run to run, and the variance sits almost entirely in the
first-tick buffer build discussed below.

That is the Brunel answer. Masquelier 2008 has a *different* answer, measured separately
below: there SuperNeuroABM is 2.7x slower than NEST GPU, construction is negligible, and
CPU Brian2 beats both GPU simulators outright.

**99.6 % of SuperNeuroABM's wall time is construction**, which confirms the long-standing
claim in `docs/BRUNEL_SCALING.md`. But the dominant cost is not parameter duplication and
not the padded parameter columns. It is this:

> **Building the two internal-state history buffers takes 107.6 s, 45 % of the entire run,
> for 150 MB of GPU data each, while history tracking is switched off.**

They are allocated even though history tracking is switched *off*. They are the only
depth-3 columns, so they alone fall into the `awkward.from_iter` path in
`convert_to_padded_gpu_tensor`, which walks 12.5 M Python rows one at a time. Every other
property, including the 801 MB spike tensor, converts 10-50x faster per byte.

Parameter duplication is real and worth fixing, but it is second order: the padded
parameter columns (`learning_hyperparameters` at width 11, `hyperparameters` at width 9,
`synapse_delay_reg` which no kernel reads) account for about 1.05 GB of GPU memory and
20 s of setup, against 102-150 s for two buffers nobody asked for.

## Method

`scaling_analysis/nest_comparison/` holds the scripts, the raw CSVs, the exact commands,
and the hardware and network specification, including the three model differences between
the two simulators that could not be removed. Read that README before quoting any number.

Two facts frame the comparison. Both simulators were re-measured on this laptop GPU, because
the published SuperNeuroABM figures are from 2048 MI250X GCDs on Frontier. And the drive is
sub-threshold, so no spikes flow; this is like-for-like for construction and is the reason
the per-tick column above should not be read as a synapse-stepping result.

## Where SuperNeuroABM's construction time goes

Third run, with the finer instrumentation described below. Every line is measured, not
inferred; the residual `.other` rows are what the timers do not yet cover.

| phase | s | note |
|---|---|---|
| `model_load` | **41.8** | |
| ├ `build_from_local_columns` | 31.2 | SAGESim, unprofiled below this line |
| ├ `property_columns` | 9.2 | SuperNeuroABM template + dedup |
| ├ `neighbor_csr` | 0.6 | |
| └ everything else | 0.9 | decode, constraint check, assemble, bookkeeping |
| `setup()` | **20.8** | |
| ├ `snn_shrink_buffers` | **18.8** | the per-agent history-buffer loop |
| ├ `sort_by_breed` | 1.0 | |
| ├ `analysis` | 0.4 | AST write-analysis |
| ├ codegen + JIT | 0.1 | source hash cache hit |
| └ other | 0.5 | |
| first tick | **162.8** | |
| ├ GPU buffer build | 149.2 | of which property tensors 141.3 |
| └ ghost topology | 12.8 | on a run with zero ghosts |
| ticks 2-100 | 1.0 | |
| **total** | **237.6** | |

`setup()`'s 18.8 s residual is now measured rather than guessed: it is the loop at
`superneuroabm/model.py:940-955` that gives every one of the 12.5 M agents its own
single-slot history buffer. SAGESim's own six setup phases total 2.0 s.

`model_load` splits three to one in SAGESim's favour: 31.2 s inside
`build_from_local_columns` against 9.2 s of SuperNeuroABM's own column building. That
31.2 s has no finer breakdown yet and is the largest single unprofiled item left.

### Earlier runs (coarser timers)

Two runs, 12.5 M synapses. Run 2 reused the cached partition, hence no generation cost:

| phase | run 1 (s) | run 2 (s) | share of run 2 |
|---|---|---|---|
| partition generation | 2.2 | cached | - |
| `load_post_owned` | 44.2 | 43.1 | 18.8 % |
| `setup()` | 20.3 | 19.9 | 8.7 % |
| first tick (GPU buffer build) | 219.1 | 156.7 | 68.5 % |
| ticks 2-100 | 1.2 | 1.2 | 0.5 % |
| **total** | **294.0** | **228.9** | |

Inside `setup()`, the six phases SAGESim itself times account for only 2.0 s. The other
**18.8 s is the per-agent Python loop in `NeuromorphicModel.setup()`**
(`superneuroabm/model.py:940-949`) that shrinks the two history buffers when tracking is
off. It touches all 12.5 M agents to write two small lists each.

Inside the first tick, `prop_tensors` is the bulk of the GPU build (190.6 s of 203.6 s
in run 1, 132.9 s of 142.8 s in run 2). Per property:

| property | shape | MB | alloc s | s/100 MB | µs/row | conversion path |
|---|---|---|---|---|---|---|
| `learning_internal_states_buffer` | (12525000, 1, 3) | 150.3 | **54.5** | 36.3 | 4.35 | **depth3_awkward** |
| `internal_states_buffer` | (12525000, 1, 3) | 150.3 | **53.1** | 35.3 | 4.24 | **depth3_awkward** |
| `hyperparameters` | (12525000, 9) | 450.9 | 6.5 | 1.4 | 0.52 | depth2_ragged |
| `learning_hyperparameters` | (12525000, 11) | 551.1 | 5.9 | 1.1 | 0.47 | depth2_ragged |
| `input_spikes_tensor` | (12525000, 16) | 801.6 | 5.7 | 0.7 | 0.46 | depth2_ragged |
| `synapse_delay_reg` | (12525000, 1) | 50.1 | 5.4 | 10.8 | 0.43 | depth2_ragged |
| `internal_states` | (12525000, 3) | 150.3 | 5.3 | 3.5 | 0.42 | depth2_ragged |
| `output_spikes_tensor` | (12525000, 2) | 100.2 | 2.3 | 2.3 | 0.19 | depth2_ragged |
| `learning_internal_states` | (12525000, 3) | 150.3 | 2.2 | 1.5 | 0.18 | depth2_ragged |
| `breed` | (12525000,) | 50.1 | 0.3 | 0.7 | 0.03 | depth1 |

The `conversion path` column is recorded by the converter itself, so this is not an
inference. The two buffers are the only depth-3 columns in the model and the only ones that
reach `awkward.from_iter`.

Read the µs/row column, not the MB column: this cost is per Python row, not per byte. The
801 MB spike tensor converts in 5.7 s while the 50 MB delay register takes 5.4 s, because
both have 12.5 M rows. Depth-2 rows cost about 0.45 µs each; depth-3 rows cost about 4.3 µs,
roughly ten times more. The two buffers are 107.6 s of the 141.3 s spent on property tensors.

That gap is the depth-3 `awkward` fallback (`SAGESim/sagesim/internal_utils.py:245-252`);
depth-1 and depth-2 columns take the numpy paths above it.

### Why the first tick costs 149 s

The first tick is the largest single item in the run, and `prop_tensors` is 141.3 s of its
149.2 s. Profiling the converter's internals at full scale gives the mechanism, not just the
attribution. There are two slow paths and they share one root cause.

**One property column spans every breed, so the fast path never fires (~33 s).**
`convert_to_padded_gpu_tensor` opens with a uniform fast path that converts a whole column in
a single numpy assignment, and it is taken only when every row has the same length. But a
soma's `hyperparameters` is 9 wide and a synapse's is 5, and they share one column, so
`set(row_lengths)` is `{5, 9}` and the column falls through to the ragged path. That path does
one Python-level `result[i, :len(row)] = row` per row. Measured on a 12,512,500-row column:

| stage | s |
|---|---|
| `[len(r) for r in col]` | 0.16 |
| `set(row_lengths)` | 0.05 |
| `max(len(r) for r in col)` | 0.30 |
| `np.full((12.5M, 9), nan)` | 0.12 |
| the per-row assignment loop | **3.76** |
| total | **4.38** (0.35 µs/row) |

Eight such columns is about 33 s, which matches the 33.7 s measured for the eight non-buffer
columns.

**The tracking-off buffer shape is three-dimensional (~107 s).** One slot per agent makes the
row `[[0, 0, 0]]`, so `_detect_depth` returns 3 and the column goes to `ak.from_iter`, which
walks all 12.5 M nested Python lists. Measured at 28.5 s for `from_iter` alone (2.28 µs/row);
the full path measured 53-54 s per column in the real run, 107.6 s for the two.

**What that column is actually doing, and what each part of it costs.** The whole job of
`hyperparameters` at the first tick is to become a dense `float32` rectangle in device
memory. Three stages, measured on the 12,512,500-row column with the CUDA context already
warm:

| stage | s | note |
|---|---|---|
| walk the Python rows into a numpy rectangle | 3.85 | |
| `cp.array(...)`, 450 MB host to device | 0.061 | 7.4 GB/s |
| the same transfer from pinned host memory | 0.051 | 9.0 GB/s |

**The transfer is free; building the thing to transfer is the entire cost.** Marshalling is
63x the DMA. Across all ten columns the traffic is about 2.6 GB, roughly 0.35 s of the
141.3 s, under 0.3 %. Pinning host buffers is not a lever here: measured steady-state it is
1.2x on a cost that is already negligible. (A first transfer in a fresh process does look
dramatically slower, about 0.28 s, but that is CUDA context and memory-pool warm-up being
charged to whichever transfer happens to be first, not a property of pageable memory.)

Worth stating plainly what is being moved: **450 MB to represent 72 bytes of distinct
information**, two rows of nine `float32`. Every synapse carries a private copy of the same
five numbers because the kernel indexes `neuron_params[agent_index][k]`. Uploading the two
distinct rows plus a per-agent `int32` index instead moves 50 MB rather than 450 MB, and
takes 0.038 s against 3.91 s. That is the parameter-interning idea (D6), and it is the one
change here that also cuts GPU memory rather than only setup time. It needs the kernel
signature change, which is why it stays deferred.

**Are the two problems fixed by one change? Partly, and the detail matters.**

These columns are Python object graphs read one row at a time, while the underlying data is
nearly constant. The columnar builder deduplicates rows: `[syn_row] * 12_500_000` is 12.5 M
pointers to one list object. So a converter that deduplicated by `id()`, padded only the
distinct rows and gathered with `table[inverse]` would replace 12.5 M Python statements with
one vectorized gather.

Counting the distinct row objects in a real model says exactly where that applies:

| column | rows | distinct objects | dedup applies |
|---|---|---|---|
| `hyperparameters` and the other 7 parameter columns | 44,000 | 2-3 | yes |
| `internal_states_buffer` | 44,000 | **44,000** | **no** |
| `learning_internal_states_buffer` | 44,000 | **44,000** | **no** |
| any column, record path (`create_soma`/`create_synapse`) | 550 | 550 | no |

So identity dedup fixes the eight parameter columns and **does not touch the two buffers**,
which are the expensive ones. `setup()` writes `[state[::]]` per agent, and `state[::]` is a
fresh copy, so every one of the 12.5 M buffer rows is a distinct object.

Fixing the buffers takes a second, separate change: have `setup()` assign one shared row
object to every agent instead of a per-agent copy. That is safe because the kernels only ever
write to those slots and `get_internal_states_history` returns nothing while tracking is off,
so the initial contents are never read. Measured on a 2 M-row buffer column:

| | s |
|---|---|
| today (`ak.from_iter`, distinct objects) | 4.42 |
| shared object + identity dedup | 0.14 |

**Neither change alone is enough for the buffers.** Sharing without the dedup still hands
`ak.from_iter` 12.5 M entries to iterate. Dedup without the sharing finds 12.5 M distinct ids
and falls back. Together they are 32x.

Revised projection for the 141.3 s of property tensors:

| change | property tensors |
|---|---|
| today | 141.3 s |
| identity dedup only | ~115 s (the eight parameter columns, ~26 s saved) |
| identity dedup + shared buffer rows | **~9 s** |

Two caveats. Identity dedup assumes shared rows are never mutated in place, the same
assumption the builder's dedup already makes and that `internal_utils.py:189-206` exists to
catch when it is violated. And a failed dedup is not free: probing `map(id, col)` over
distinct objects measured 0.66 µs/row, about 8 s per column at this scale, so the record path
would pay for a probe that never pays off. The fast path needs a cheap early-out, such as
testing whether the first handful of rows are the same object, before scanning the column.

The remaining 12.8 s of the first tick is ghost-topology discovery, on a single-GPU run that
has no ghosts at all.

### Why the history buffers exist at all when tracking is off

Turning tracking off does not remove them. Every synapse and learning kernel writes to them
on every tick, unguarded:

```python
buffer_idx = t_current % len(internal_states_buffer[agent_index])
internal_states_buffer[agent_index][buffer_idx][0] = I_synapse
```

(`step_functions/synapse/single_exp.py:68-71`, `double_exp.py:74-76`, and all four STDP
rules.) Because the kernel always writes, the column cannot be empty. So
`NeuromorphicModel.setup()` shrinks each agent's row to a single slot instead, `t_current % 1`
is always 0, and the kernel writes into a scratch slot that nothing ever reads.

The cost follows from that shape. One slot per agent makes the column `(N, 1, W)`, three
dimensional, and these are the only three-dimensional properties in the model, so they alone
take the `awkward` fallback. Two switched-off buffers cost 107.6 s because of how their
off state is represented, not because anyone wanted the data.

Removing them cleanly needs one of: a shared row object across agents (all rows are
identical when tracking is off, and only the GPU copy is ever mutated), a uniform-shape fast
path in the converter, or a kernel-contract change so the off state is a 2-D scratch column.
The first two are non-breaking.

**A latent bug found while checking this.** `step_functions/synapse/weighted_synapse.py:60-63`
indexes the buffer with the raw tick, `internal_states_buffer[agent_index][t_current][0]`,
with no modulo, unlike every sibling kernel. With tracking off the row holds one slot, so from
tick 2 that write lands outside the agent's row. Every test for that kernel passes
`enable_internal_states_tracking=True`, so nothing exercises it. Not changed here.

Three of these columns are pure waste at this configuration, and the table prices each one:

- `learning_hyperparameters` is 11 wide because `_compute_max_property_sizes`
  (`superneuroabm/model.py:66-78`) runs at line 126, *before* the learning-rule configs are
  popped at line 129. Brunel has no plasticity, so all 551 MB and 6.9 s buy nothing.
- `synapse_delay_reg` is allocated for every agent and read by no kernel in the repository.
- Both history buffers exist only to be indexed with `t_current % 1` while tracking is off.

## Memory

Peak host RSS is 16.6 GB for 12.5 M synapses, about 1.3 KB per synapse. The columnar
loader avoids per-agent property objects, so this is dominated by the id bookkeeping
described in `scaling_analysis/MEMORY_ANALYSIS.md` plus the numpy staging inside
`convert_to_padded_gpu_tensor`, which builds the full `capacity x width` array on the host
before the device copy.

On the GPU, 3,558 MB for 12.5 M synapses is 284 B per synapse, against roughly 12 B for
NEST GPU. Of our 284 B, about 52 B is genuine per-synapse state (weight, synaptic current,
three STDP traces, the CSR entry). The rest is duplicated constants and cross-breed padding:
every property tensor is one dense array spanning *all* breeds, padded to the widest row any
breed needs, so a soma-only column is still materialised for every synapse and vice versa.

NEST GPU fits in-degree 4000 (50 M synapses, 751 MB, 0.14 s of setup) on the same card,
where SuperNeuroABM cannot fit in-degree 2000.

## Verdict on the three questions

**1. Does per-agent parameter duplication explain the memory use?** Partly. It is real:
identical parameters are materialised per agent on both host and device, and at Brunel
scale the padded parameter columns are gigabytes. But the digits tutorial (788 agents) and
Masquelier (2001 agents) hold a handful of distinct parameter sets across a few thousand
agents, so interning would save kilobytes there. It is a large-network fix, not a fix for
the examples that feel slow.

**2. Does it explain setup dominating wall time?** No. Setup does dominate, overwhelmingly.
But the cost is Python-per-agent work in the conversion and setup loops, and the single
largest item is two buffers that are switched off. Parameter columns are about 20 s of the
293 s.

**3. Masquelier and the NEST gap.** Two separate causes, neither about memory:

- Masquelier is chunked because `get_soma_spike`
  (`superneuroabm/step_functions/synapse/util.py:40-52`) rescans each synapse's spike row
  from index 0 on every tick, so work is O(ticks x spikes-in-row) and grows quadratically
  with chunk length. Diehl-Cook already carries a cursor fix
  (`ns-applications/duplicates/diehl_cook_2015/dc_spike_io.py:44-62`). Chunking is also
  physically wrong: `_reset_agents` zeroes soma state and STDP traces at every boundary.
- The NEST gap at construction is what this document measures: 5,400x, entirely on our side
  of the fence, in Python loops and a mis-selected conversion path. The *architectural* gap
  that no layout change removes is separate and applies to steady state: every synapse is an
  agent stepped every tick, while NEST GPU delivers spikes event-wise. That is decision D0 in
  `docs/BRUNEL_SCALING.md:948-954` and it is not what makes these runs slow today.

## How NEST GPU builds 12.5 M synapses in 54 ms

230 M synapses per second is worth explaining, so this is read out of the NEST GPU source
rather than assumed. It is three separate decisions, not one trick.

**1. Connecting is a kernel launch, not a loop.** `connectFixedIndegree`
(`src/connect.h:3860-3928`) issues one `curandGenerate` call that fills all 12.5 M random
source indices directly in device memory, then about six kernels of 1024 threads per block
over those 12.5 M elements: `setSource`, `setIndegreeTarget`, `setConnectionWeights`,
`setConnectionDelays`, `setPort`, `setSynGroup`. No connection data crosses the PCIe bus and
no connection exists as a host object at any point. Seven kernels over 12.5 M elements in
7 ms is ordinary GPU throughput.

**2. Targets are computed, not stored.** `setIndegreeTarget` (`connect.h:2164-2178`) does
`i_target = getNodeIndex(target, i_conn / indegree)`. Under a fixed in-degree the target is
implied by the connection's own index, so it costs no memory and no generation work.

**3. A connection is 12 bytes of bit-packed integers.** From `src/conn12b.h`, the key is one
32-bit word holding source id and delay behind `SourceMask`/`DelayMask`, and the struct is one
32-bit word holding target, port and synapse group plus a 4-byte float weight. No pointers, no
per-object header, no padding. That is exactly the ~12 B/synapse measured here.

The one step that scales linearly is `Calibrate`, and it is a sort: `organizeConnections`
runs a GPU radix sort (`copass_sort::sort`, `connect.h:3105`) to group connections by source
and delay, which is what makes event delivery cheap later. It measured 0.044 s, 0.081 s and
0.126 s at in-degree 1000, 2000 and 4000, linear in synapse count as a sort should be.

### The 5,400x is a best case NEST GPU cannot always use

`fixed_indegree` is a *generative* rule: NEST GPU never materialises an edge list, it
synthesises one on the GPU. A measured network cannot be expressed that way. Cora is 2,715
papers and 31,836 citations that exist in a file; a connectome is the same. Those have to be
handed over edge by edge, through `Connect` with explicit node lists, which marshals Python
lists into ctypes arrays on the host.

Measured on this GPU, one process per configuration, `Connect` plus `Calibrate`:

| edges | generative rule | explicit edge list | ratio |
|---|---|---|---|
| 31,836 (Cora) | 0.026 s | 0.033 s | 1.3x |
| 1,000,000 | 0.027 s | 0.196 s | 7.3x |
| 12,500,000 | 0.050 s | 2.263 s | 45x |

The rule path is nearly flat across a 340x range of edge counts, because almost all of its
cost is the `Calibrate` sort; per edge it approaches zero. The explicit path is linear at
about 0.18 us/edge and is dominated by `Connect` itself, which is host-side list marshalling,
not GPU work.

**So the fair headline for a data-derived network is 2.3 s against our 237.6 s, roughly
105x, not 5,400x.** That is the number to quote when the network comes from a file.

The conclusion does not flip, and here is the part that matters most for us: our own
topology generation was **2.2 s**, essentially identical to NEST GPU's 2.26 s of explicit
edge ingestion at the same scale. We are already competitive at the step that actually
depends on the graph. The remaining 235 s is serialize, deserialize, rebuild as Python rows,
convert row by row into padded tensors, and none of it cares where the topology came from.

At Cora's actual size the question is moot for both. SuperNeuroABM builds a 2,715-node,
31,836-edge network (34,551 agents, via the `create_soma`/`create_synapse` path an
application really uses) in 1.53 s: 0.36 s creating agents, 0.40 s in `setup()`, 0.78 s in
the first tick. NEST GPU does the equivalent in 0.033 s. Forty-six times faster, and both
irrelevant, because a real Cora run spends its time on the inference presentations rather
than on setup.

### Why our path cannot be that fast as written

SuperNeuroABM generates connectivity on the host in numpy, writes it to an `.npz`, reads it
back, turns it into Python list columns, and then walks those columns row by row into dense
padded float32 tensors that are copied to the device. Every synapse is a host-side object at
some point in that chain, and the chain runs four times over the same 12.5 M items.

The deeper difference is what is being stored. NEST GPU stores a *connection*: 12 bytes,
bit-packed, write-once, with a fixed meaning. SuperNeuroABM stores an *agent*: ten named
properties in dense columns spanning every breed, 284 B/synapse, because a synapse is a
first-class agent that runs arbitrary user code every tick. That expressiveness is the
product's premise, not an accident, and it is not free.

But the 5,400x gap is not the price of that premise. The topology generation this comparison
is arguably unfair about is 2.2 s of the 237.6 s. The rest is layout: serialize, deserialize,
rebuild as Python rows, convert row by row. Nothing about first-class synapse agents requires
building a `(N, 1, 3)` ragged Python structure for a buffer that tracking has switched off, or
walking 12.5 M rows in the interpreter to pad a column that is already rectangular. Those are
the levers below, and none of them touches the agent model.

## Masquelier 2008: a second, different answer

The same 10 s experiment (2000 afferents, one output neuron, 2000 plastic synapses,
1,212,948 input spikes, 100,000 ticks at dt = 0.1 ms), same spike trains from the
experiment's own generator, on this machine:

| | total | simulate | output spikes |
|---|---|---|---|
| Brian2 2.9, CPU, cython | **11.1 s** | 6.5 s | 20 (*) |
| NEST GPU | 12.6 s | 11.0 s | 8,241 |
| SuperNeuroABM, fused ticks | 33.8 s | 30.8 s | 414 |
| SuperNeuroABM, unfused ticks | 84.5 s | 81.4 s | 414 |

Every row was measured with the GPU otherwise idle. An earlier NEST GPU run that overlapped
with a Brunel run reported 21.5 s rather than 12.6 s, so these are not numbers to collect
concurrently.

Three things follow, and none of them is the Brunel story.

**Construction is not the problem here.** SuperNeuroABM builds this network in 0.27 s of
`setup()` plus 0.11 s of buffer construction. At 2001 agents the Python-per-agent costs that
dominate Brunel are invisible. The gap is entirely in the 100,000 tick loop.

**The single-worker fused path is worth 2.5x on a small network,** and it is off by default
whenever `verbose_timing` is on (`SAGESim/sagesim/model.py:1251`). Both rows above produce
byte-identical results: 414 output spikes and the same weights. The comment at
`model.py:1245-1250` records fusing being 8-14x *slower* at Brunel scale, so the right rule
is size-dependent, and nothing currently chooses between them for the user.

**A CPU simulator wins outright at this size.** Brian2 is 3x faster than SuperNeuroABM and
slightly faster than NEST GPU. With 2001 nodes there is nothing for a GPU to parallelise, and
both GPU simulators pay launch and barrier overhead 100,000 times. Masquelier-scale work does
not belong on a GPU; that is a property of the problem, not a defect in SuperNeuroABM.

So SuperNeuroABM's remaining gap to NEST GPU here is 2.7x, not the 5,400x seen on Brunel
construction. The two questions have genuinely different answers.

Spike counts differ across the three because the models are only approximately matched, and
deliberately so: NEST GPU gets `iaf_psc_exp` plus its built-in `stdp` synapse group in place
of the Hathway-Goodman 3-variable LIF, so it is the least faithful of the three and its 8,241
spikes should not be read as a replication. (*) The Brian2 script was meant to use the
reference equations verbatim but transcribed the reference's un-restricted `_nn` STDP variant
instead of its default RNN rule; that neuron dies after ~20 discharges (the original's own
`_nn` control gives 21), so its 6.5 s is for a silent network and the 20 is not a result.
Corrected on 2026-09-10; see "Implemented: tick-major input-spike events".
This is a wall-clock comparison at matched network size and matched input, not a numerical
replication. Note also that the chunking analysis below still stands: this measurement is one
10 s chunk, which is the regime chunking was introduced to keep fast.

## Implemented: the two first-tick fixes

Both landed on branch `perf/nestgpu-comparison`. Median of three runs at the same
12,500-neuron, 12.5 M-synapse point, GPU otherwise idle:

| phase | before | after | change |
|---|---|---|---|
| `setup.snn_shrink_buffers` | 18.8 s | 0.60 s | 31x |
| `setup()` total | 20.8 s | 2.31 s | 9.0x |
| first-tick property tensors | 141.3 s | **5.46 s** | **26x** |
| first-tick GPU build | 149.2 s | 12.72 s | 11.7x |
| first tick total | 162.8 s | 27.41 s | 5.9x |
| `load_post_owned` | 41.8 s | 39.64 s | unchanged |
| ghost topology | 12.8 s | 13.90 s | unchanged |
| ticks 2-100 | 0.99 s | 1.16 s | unchanged |
| **end to end** | **237.6 s** | **77.37 s** | **3.1x** |
| peak host RSS | 16,584 MB | 9,877 MB | -6.7 GB |
| GPU pool | 3,558 MB | 3,558 MB | unchanged, as intended |

The GPU figure is the control: the device tensors are byte-for-byte what they were, so
nothing about what the kernels see has changed. Masquelier at 10 s reproduces exactly, 414
output spikes and the same five weights, which is the end-to-end check that per-agent STDP
state still diverges correctly.

**Change A** (`superneuroabm/model.py`, `_share_history_buffers`) gives every agent the same
history-buffer row object instead of a private copy, when tracking is off. Same value, same
shape, same tensor; 12.5 M Python objects become one.

**Change B** (`SAGESim/sagesim/internal_utils.py`) detects rows that are the same object,
converts only the distinct ones, and gathers on the device. Nine of the ten columns now report
the `identity_dedup` path; only `breed` does not, and it is a depth-1 scalar column that
already converted in 0.3 s.

### Simulate time specifically, before and after

`simulate()` includes the first tick, so it is worth splitting.

**Brunel, 12,500 neurons / 12.5 M synapses / 100 ticks:**

| | before | after | change |
|---|---|---|---|
| `simulate()` wall | 164.37 s | 29.03 s | 5.7x |
| of which first tick (GPU buffer build) | 162.80 s | 27.41 s | 5.9x |
| of which ticks 2-100 | 0.99 s | 1.16 s | unchanged |
| per steady-state tick | 10.0 ms | 11.7 ms | unchanged |

**Masquelier, 2000 afferents / 10 s / 100,000 ticks, fused:**

| | before | after |
|---|---|---|
| `setup()` | 0.27 s | 0.31 s |
| `simulate()` | 30.83 s | 30.85 s |
| end to end | 33.79 s | 35.11 s |

Masquelier is unchanged, and that is the expected result rather than a disappointment. Its
construction is 0.11 s: at 2001 agents there was nothing for these fixes to remove. Its 30.8 s
is the 100,000-tick loop, which neither change touches. The two experiments continue to have
different answers, exactly as the NEST GPU comparison found.

The steady-state tick figures on both sides are noise, not signal: nothing about the kernels or
the device tensors changed, and the Brunel per-tick number is measured on the unfused path that
`verbose_timing` forces.

### Equivalence: what is bitwise identical, and the one thing that is not

Dumping every device tensor before and after, for the same seeded model (2,000 neurons,
in-degree 100, 30 ticks, Poisson drive), and comparing raw bytes:

| array | identical |
|---|---|
| spike output | yes |
| `hyperparameters` (the STDP weights) | yes |
| `internal_states`, `learning_internal_states` | yes |
| `input_spikes_tensor`, `output_spikes_tensor` | yes |
| `learning_hyperparameters`, `synapse_delay_reg`, `breed` | yes |
| neighbour CSR offsets and values | yes |
| `internal_states_buffer` | yes |
| `learning_internal_states_buffer` | **no** |

The one exception is the write-only scratch buffer, and the difference is padding. A Brunel
synapse has no learning rule, so its `learning_internal_states` is an empty list; seeding the
buffer from it produced a row that padded out to `NaN, NaN, NaN`, while a soma's produced
zeros. The shared row is explicit zeros for everyone, so previously-NaN slots now read 0.0.

This is unobservable. No kernel reads these slots (every access is a write, plus `len()`),
`get_learning_internal_states_history` returns `[]` while tracking is off, and in Brunel the
column is never written at all because the learning-rule selector takes its no-op branch.
The proof is the rest of the table: the spike output and every state tensor match byte for
byte. Masquelier, which does have a learning rule and therefore does write this buffer,
reproduces its 414 spikes and weights exactly.

Recorded rather than papered over, because "bitwise identical except one array" is the kind
of claim that should name the array.

**Masquelier, compared in full.** Five weights out of 2000 is weak evidence, so the whole
result was dumped under both code versions and compared byte for byte:

| | n | bitwise identical |
|---|---|---|
| output spike times | 414 | yes |
| final synapse weights | 2000 | yes |

1,949 of those 2000 weights are distinct values, spanning 0.0000 to 0.8839. That is the
point worth keeping: 2000 synapses created from one config, sharing one host parameter row,
still learn 1,949 different weights. Sharing on the host never touched per-agent state on the
device, which is what the whole change rests on.

Two things learned while implementing:

- **Slice-assign, never rebind.** `Model.setup()` caches
  `__rank_local_agent_data_tensors` as references to the column list objects, so replacing a
  dict entry left the GPU build reading a stale empty column and compiling a 2-D tensor against
  kernels that index three deep. `test_internal_states_tracking` caught it.
- **The dedup cap has to be a ratio, not a count.** `input_spikes_tensor` holds ~12,501
  distinct rows among 12.5 M, because each of the 12,500 input synapses owns a Poisson train
  while every recurrent synapse still holds the untouched sentinel. An absolute cap of 1024
  rejected a 1001:1 compression and left that column on the slow path at 6.3 s; a ratio test
  takes it to 0.58 s.

### What is left, re-ranked

`load_post_owned` at 39.6 s is now **51 % of the run** and the clear next target, with
`build_from_local_columns` about 31 s of it. Ghost-topology discovery is 13.9 s on a run with
zero ghosts. Between them they are 69 % of what remains.

`input_spikes_tensor` is still the largest tensor at 801 MB, of which roughly 700 MB is padding:
the column is padded to the widest row (16 floats) while 12.5 M recurrent synapses only ever
hold the 2-float sentinel. Dedup fixes its conversion time but not its footprint, because the
gather expands back to full size. The structural fix is a global event CSR, the same migration
`locations` already went through, and it needs a kernel change.

## Implemented: tick-major input-spike events (2026-09-10)

**Why.** Masquelier had to be run in 10 s chunks with `reset()` between them. The cause was
speed, not memory: `get_soma_spike` rescanned each synapse's padded row
`[-1, 0, tick, val, ...]` from index 0 on every call, twice per tick (synapse step and STDP
rule), so a run of T ticks cost O(T x spikes/row). Measured before the change: 100 s in one
`simulate()` had not finished after 31.5 min (ten 10 s chunks take ~5 min); host RSS 1.6 GB
and 417 MB GPU, so it was never memory.

**What.** Injected spikes no longer live in the property store. `add_spike*` append to a
host-side event store; `add_spikes(ids, ticks, values)` takes flat arrays (Brian2
`SpikeGeneratorGroup` shape); `set_input_events(rows, ticks, values)` does the same from CuPy
arrays without a host round trip (Diehl-Cook). When the kernel launches the store is compiled
into `ev_offsets[tick]`, `ev_syn`, `ev_val` (duplicates summed, sorted by tick) and passed as
extra kernel arrays. A new SAGESim hook, `pre_tick_code` in `_get_extra_kernel_config`, emits
code at the top of every tick: all threads stride over that tick's events and stamp
`input_spikes_tensor[row] = [tick, value]`, followed by a grid barrier only on ticks that have
events. `get_soma_spike` reads the stamp with one comparison. The row is a fixed 2 floats, the
per-`simulate()` Python tuple sort is gone, injecting between `simulate()` calls never touches
the property buffers (nothing learned is lost), and a launch that sees the clock rewound
(`model.tick = 0`, Diehl-Cook) clears stale stamps itself. Kernel signatures and
`get_soma_spike`'s signature are unchanged; every existing kernel works unmodified.

### Equivalence (bitwise, not asserted)

| check | result |
|---|---|
| Masquelier 10 s, `masquelier_sna.py --fused --dump-dir`, before vs after | all 414 spike times and all 2000 weights identical |
| Brunel 2000 / K=100 / 30 ticks with 500 Hz Poisson drive, `dump_tensors_for_diff.py` | every device tensor identical (soma state, currents, STDP state, spike counts); `input_spikes_tensor` shape (204000, 46) -> (204000, 2) |
| Diehl-Cook `test_brian1_semantics.py` (tick-level vs the Brian 1 model) | Part A and B pass, spike trains identical, max dv 6e-5 mV |
| Diehl-Cook 300/300/300 sanity run, seed 0 | accuracy 36.33 % reproduced; final weights, theta, assignments, label responses bitwise identical |
| `superneuroabm/tests` (146 + the two rewritten injection files), `SAGESim/tests`, 2-rank injection test | pass |

### Timing (same laptop GPU, one simulator at a time)

| network | ours before | ours after | NEST GPU | Brian2 |
|---|---|---|---|---|
| Masquelier 10 s, 2000 afferents, 1.21 M spikes, fused: simulate | 32.2 s (322 us/tick) | **15.5 s** (155 us/tick) | 11.0 s | 5.2 s (RNN rule, 475 spikes vs our 414) |
| Masquelier 100 s, 12.1 M spikes, one run: simulate | >31.5 min, killed | **141 s** (141 us/tick), inject 0.41 s, 399 MB GPU | not run | 16.5 s (RNN rule, 891 spikes vs our 639; per 25 s: 713/69/46/63 vs 463/67/46/63) |
| Masquelier 450 s on the ORIGINAL authors' input, 57.7 M spikes: simulate | not feasible | **647 s** (144 us/tick), inject 5.9 s, 685 MB GPU | — | original code, cpp_standalone: 79.7 s |
| Brunel 12,500 / 12.5 M / 100 ticks: total / ticks 2-100 / GPU | 80.9 s / 1.16 s / 3597 MB | 83.1-86.0 s / 1.02-1.19 s / **2927 MB** (two clean runs; a third at 101 s overlapped another GPU process) | 2.1 s / 0.014 s / 367 MB | — |
| Diehl-Cook n_exc=100, ms per presentation (1000 ticks) | ~200-220 (NOTES §5) | **186** (n_exc=400: 705) | — | faithful transcription 2.2-4.5 digits/s incl. ~1.25 attempts/digit, i.e. ~180-360 ms/presentation |

Per-tick cost is now flat in run length (155 -> 141 -> 144 us/tick from 10 s to 450 s). The
remaining 140 us/tick on a 2001-agent network is the fused kernel's grid barriers, not the
input path; a CPU simulator wins at this size, as recorded above. Brunel's tick loop is
unchanged within noise (10.2-11.9 ms/tick vs 11.6), so the delivery barrier is not visible
at 12,500 Poisson inputs; the 670 MB GPU saving is the padded rectangle gone.

### Masquelier against the original code on identical input (450 s, seed 1)

`~/Hathway-Goodman-2018/reference/run_hg_dump.py` (a copy of the authors' `run_simulation.py`
that saves its input and results) and `ns-applications/duplicates/masquelier_2008/run_hg_input.py`
+ `compare_hg.py`:

| | original (Brian2 2.1.2, float64) | SuperNeuroABM (float32) |
|---|---|---|
| output spikes | 3318 | 2982 |
| spikes in 0-50 s / 50-100 s / 100-150 s ... | 1304 / 258 / 256 / 236 / 258 / 256 / 236 / 258 / 256 | 968 / **258 / 256 / 236 / 258 / 256 / 236 / 258 / 256** |
| authors' metrics: hit rate / false alarms / avg latency | 1.000 / 0 / 3.76 ms | 1.000 / 0 / 4.09 ms |
| find_t (pattern found) | 23.3 s, spike #1176 | **17.0 s, spike #811** |
| success (authors' criterion) | 1 | 1 |
| final weights > 0.5 | 365, all pattern afferents | 355, all pattern afferents; 98.1 % same set, corr 0.967 |
| spike-by-spike | first spike tick 92; ISIs 159, 163, 156 | 93; 162, 166, 157 — 2 % identical, 60 % within 5 ticks |

Same outcome by the paper's criteria and identical steady-state firing (one spike per
presentation), but not identical trajectories: ours converges earlier (fewer discharges before
the pattern is found) and the early inter-spike intervals are ~2 % longer. This is the "faster
convergence" seen before in the chunked runs, reproduced here in a continuous run on the
authors' own input, so it is a property of the implementation (float32; the order in which the
synaptic current is decayed and incremented; `>=` vs `>` at threshold), not of the chunk
resets. Exact spike-time identity is not reachable with float32 state.

### Chunked-with-reset vs continuous (our trains, 100 s)

`run_experiment_hg.py --mode chunked` (the historical loop: inject each 10 s with relative
ticks, `simulate`, `reset(retain_parameters=True)`) against `--mode continuous` (inject once,
ten `simulate()` calls without reset), same seed and trains, metrics from `compare_runs.py`:

| run | spikes | < 50 s | find_t | discharges to find | hit rate | FA | latency | w > 0.5 pattern/other |
|---|---|---|---|---|---|---|---|---|
| chunked, reset every 10 s | 672 | 539 | 29.10 s | 486 | 0.994 | 0 | 1.580 ms | 397 / 0 |
| continuous | 671 | 538 | 29.10 s | 484 | 1.000 | 0 | 1.566 ms | 398 / 1 |
| historical `output_hg/` (old code, chunked) | 672 | 539 | 29.10 s | 486 | 0.994 | 0 | 1.580 ms | 397 / 0 |

The first 436 spikes — the entire first chunk — are identical between the two arms; they
diverge only after the first `reset()`, and end at the same place. Clearing soma state,
synaptic current and STDP traces every 10 s therefore neither helps nor hurts convergence:
the state it clears has a memory of tens of milliseconds. The faster-than-Brian2 convergence
in the section above is present in the continuous run too, so it is the implementation, not
the resets. (The new code's chunked arm also reproduces the old `output_hg/` run exactly.)
Chunking is now unnecessary and the continuous mode is the default.

## Ranked levers

Each is measured on the 12.5 M synapse point above and tagged with its deferred item below.

| # | lever | measured gain | risk | item |
|---|---|---|---|---|
| ~~1~~ | ~~Share one buffer row object across agents~~ | **done**, see above | | |
| 2 | Replace the per-agent buffer-shrink loop in `NeuromorphicModel.setup()` | **-18.8 s** | low | D3 |
| **1** | Profile and vectorise `build_from_local_columns` | up to -31 s, now the largest single item | unknown, unprofiled | new |
| ~~4~~ | ~~Identity fast path in `convert_to_padded_gpu_tensor`~~ | **done**: with lever 1, property tensors 141.3 s -> 5.46 s | | |
| **2** | Skip ghost-topology discovery when there are no ghosts | -13.9 s single-GPU | low | new |
| 6 | `property_columns` in the columnar loader | up to -9.2 s | low | new |
| 7 | Per-class `_compute_max_property_sizes`: soma `learning_hyperparameters` -> `[-1.0]` | -551 MB GPU, -5.9 s | low | D3 |
| 8 | Drop `synapse_delay_reg` payload (no kernel reads it) | -50 MB GPU, -5.4 s | low | D3 |
| ~~9~~ | ~~Spike-read cursor (Masquelier)~~ | **done differently**: tick-major event list, see below. The "no effect on the 10 s chunk" guess was wrong: 32.2 s -> 15.5 s | | |
| 10 | Choose fused vs unfused ticks by network size instead of by `verbose_timing` | 2.5x on Masquelier, and it is currently accidental | low | new |
| 11 | Typed id maps instead of the four bookkeeping dicts | host RSS | medium | D5 |
| 12 | Parameter interning into shared tables | ~1 GB GPU at 50 M synapses | high, breaks kernel signature | D6 |

Items 1, 2, 7 and 8 are all inside `superneuroabm`, all low risk, and together account for
137.7 s of the 237.6 s. They should be measured again after each change with the same
scripts.

Four items came out of the measurement rather than the original plan: the two unprofiled
loader phases (items 3 and 6, together 40.4 s), ghost-topology discovery costing 12.8 s on a
run with zero ghosts, and the fused-tick path being chosen by a debug flag.

## Deferred work

Recorded in September 2026, not implemented. The full plan, including the design detail for
each item, is in `~/.claude/plans/one-thing-about-superneuroabm-sprightly-allen.md`.

**D2 (superseded by the event list, 2026-09-10). Masquelier O(1)-amortised spike read.** Cursor in header slot `[1]` of
`input_spikes_tensor` (element index of the first pair with `tick >= t_current`), advancing
only past strictly older pairs so the synapse step at priority 100 and the STDP rule at 101
read the same value within a tick. Safe because every SuperNeuroABM property is in
`no_double_buffer` and the tensor is not neighbor-visible. The cursor is a lower bound, so
`simulate()` zeroes it when re-sorting and `reset()` zeroes the device column. Host side:
accept numpy arrays in `add_spike_list`, replace the per-synapse tuple sort in `simulate()`
with a sortedness check.

**D3. Width fixes, non-breaking.** Per-component-class `_compute_max_property_sizes`; soma
`learning_hyperparameters` default `[-1.0]` and `learning_internal_states` `[]`;
`synapse_delay_reg` default `[]` while staying registered so positional argument 12 is
unchanged; history buffers as one shared row object per column instead of the per-agent loop.

**D4. SAGESim first-tick vectorisation.** Identity fast path keyed on row `id()`,
`fromiter`/`chain` depth-2 path, `target_width=` so the MPI width sync pads at allocation,
skip the `combined` copy when there are no ghosts. Same treatment for
`build_csr_from_ragged`. `sort_by_breed` via `np.argsort` plus `itemgetter`, keeping the
`OrderedDict` for `_rank2agentid2agentidx[worker]` whose key order is load-bearing. Skip
`copy(default)` in `create_agent` when the kwarg is present. Dead code to remove:
`_download_local_data_to_cpu`, `convert_to_equal_side_tensor`, the `type(x) == Iterable`
branch in `agent.py:387-391`, `Breed._prop2maxdims`.

**D5. Typed id maps.** Accessors `rank_of / breed_of / local_index / local_ids`, then freeze
the four bookkeeping dicts into sorted-array maps at `setup()`. MPI tests are the gate.

**D6. Parameter interning, design only.** No clean non-breaking variant exists: globals are
positional kernel arguments. Would need a v2 18-argument signature with `hp_table` and
`lhp_table`, per-agent rows reduced to `[weight, synaptic_delay, set_id]`, touching every
`@jit.rawkernel` in both repositories and in ns-applications. (The `input_spikes_tensor`
event list landed on its own on 2026-09-10 without a signature change.)

## Reproduction

```sh
cd scaling_analysis/nest_comparison
/home/xxz/miniforge3/envs/sna-dev/bin/python brunel_sna_local.py \
    --neurons 12500 --in-degree 1000 --ticks 100 --csv results/brunel_sna.csv
source ~/software/nest-gpu/install/bin/nestgpu_vars.sh
/home/xxz/miniforge3/envs/sna-dev/bin/python brunel_nestgpu.py \
    --neurons 12500 --in-degree 1000 --ticks 100 --csv results/brunel_nestgpu.csv

# Masquelier, 10 s, all three simulators
/home/xxz/miniforge3/envs/sna-dev/bin/python masquelier_sna.py --seconds 10 --fused
/home/xxz/miniforge3/envs/sna-dev/bin/python masquelier_nestgpu.py --seconds 10
/home/xxz/miniforge3/envs/diehlcook/bin/python masquelier_brian2.py --seconds 10
```

Raw per-phase CSVs are in `scaling_analysis/nest_comparison/results/`. The instrumentation
they rely on is in SAGESim on branch `perf/nestgpu-comparison`; without it `setup()` reports
nothing and `prop_tensors` is a single aggregate number.
