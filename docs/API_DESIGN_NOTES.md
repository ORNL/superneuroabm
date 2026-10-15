# API design notes

Smells found while using the public API for something new. Recorded, not fixed; each entry says where
it bit and what the workaround was. Related: `SPIKE_RECORDING_NOTES.md` (Finding 4).

## 1. `simulate(ticks, update_data_ticks)` — the second argument is the MPI sync interval (2026-10-01)

`NeuromorphicModel.simulate(ticks, update_data_ticks=1)` passes the value straight to SAGESim's
`simulate(ticks, sync_workers_every_n_ticks)`. The SuperNeuroABM name reads like "how often to refresh
recorded data", so `simulate(ticks=150, update_data_ticks=150)` looks harmless — and on one rank it is.
On 2+ ranks it means ghost somas exchange spikes once per call, so every cross-rank synapse sees no
input for the whole run. `poc/hetero_chip/run_chip_poc.py` did exactly this: chance accuracy on 2 and 4
ranks with no error. Rename (or alias) to `sync_every_n_ticks`, and/or warn when it exceeds the minimum
synaptic delay on a multi-rank run, since anything larger silently drops spikes' timing.

## 2. The columnar post-owned reader assumes one breed per class (2026-10-01)

`_spec_from_post_owned_columns` reads `soma_breed` / `synapse_breed` / `*_config` as scalars ("the
network it describes is uniform"; the docstring already says a per-agent producer needs a new reader).
A heterogeneous network — `poc/hetero_chip` mixes 3 soma and 3 synapse breeds, and per-synapse device
technology is its point — must use the row-oriented record path. Fine at 2 M synapses (records ->
spec in a few seconds); it is the path that inflates ~30x in host RAM at Brunel scale. A categorical
breed/config column (codes + table, as interning already does for property rows) would remove the
restriction without a second builder.

## 3. The spike record is sized from the agent count, not the recorded somas (2026-10-13)

`_prepare_kernel_extras` sizes the per-call spike record as `max(10000, num_local_agents * ticks / 100)`.
Synapse agents dominate that count in the usual build, so the bound is loose. Fold the synapses into
the somas (Brunel prototype: 25,000 agents instead of 12.5 M) and it binds: the onset burst of a
12,500-neuron network emits 11,396 spikes in ONE tick, against room for 10,000. Workaround: shorter
`simulate()` calls plus `set_recorded_somas` on a subset. A soma spikes at most once per tick, so
`recorded_somas * ticks` is an exact upper bound; it is also the natural thing to size by.

## 4. `load_from_adjacency` rejects a repeated id in a soma's neighbour list (2026-10-13)

The duplicate check assumes a soma's neighbours are its incoming synapses, where a repeat would
double count. When a soma reads presynaptic somas directly (synapses folded into the soma, as in
the onchip-learning-crosstalk column design), a repeat is a multapse, which Brunel's
`allow_multapses=True` default produces (~4 % of edges at K = 1000). Workaround: generate without
multapses, or dedupe and carry a multiplicity per edge.

## Generated learning-rule selector is one shared file per package (2026-10-15)

`_generate_learning_rule_selector()` writes `superneuroabm/_generated/learning_rule_selector.py`
and imports it. Two SuperNeuroABM processes on one machine (a pytest run and a benchmark,
or two benchmarks) overwrite each other's file: a model that registered a rule named
`my_rule` in the test suite made a concurrently starting benchmark fail at JIT with
`NameError: Unbound name: my_rule`. MPI ranks are safe only because they generate the
same text. Fix later: name the file by a hash of its content, as SAGESim now does for the
fused kernel (`step_func_code_<sha1[:12]>.py`), or generate it into the model's own
temp directory. Workaround: do not run two SNA processes with different rule sets at once.
