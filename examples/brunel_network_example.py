"""
Example: Build and run a Brunel balanced random network with ``brunel.py``.

A walk-through of the general-purpose Brunel generator
(``superneuroabm/brunel.py``) and its three entry points. Read top-to-bottom; each
section is self-contained and prints what it does.

What a Brunel network is
    A sparse, balanced, random recurrent LIF network: an excitatory and an inhibitory
    population (4:1), each neuron receiving a FIXED number of excitatory + inhibitory
    recurrent inputs, inhibition-dominated weights, plus one external Poisson-driven
    input per neuron. It's the canonical SNN scaling benchmark (Brunel 2000; NEST
    ``hpc_benchmark``).

The three entry points
    1. ``build_brunel_network(...)``  -> ``(somas, synapses)`` in memory, for a
       single-process build via ``model.create_from_lists(...)``. Use for tests,
       interactive work, a single-GPU run.
    2. ``save_brunel_partition(...)`` -> writes one rank's ``partition_{rank}.pkl`` for
       a distributed run consumed by ``model.load_post_owned(...)``. Each MPI rank calls
       it for its own ``partition_rank`` — no communication during generation.
    3. ``brunel_partition(...)``      -> the core: returns one partition's
       ``{somas, synapses[, remote_ranks]}`` dict directly (what the two wrappers build
       on). Use when you want the dict without touching disk or the model.

Key ideas the API encodes
    * **Per-rank sizing.** You give ``somas_per_rank``; the global total is DERIVED
      (``somas_per_rank * num_partitions``). You never hand-compute the total.
    * **Fixed in-degree, emergent out-degree.** Each soma gets EXACTLY
      ``excitatory_in_degree`` + ``inhibitory_in_degree`` recurrent inputs. (You can
      instead pass ``connection_probability`` and let the in-degrees be derived.)
    * **Weights come from the config; you override only what you want.** Leave
      ``excitatory_weight=None`` to inherit ``component_base_config.yaml``; the
      inhibitory sign (``-g * weight``) is always written by the generator because the
      config can't know a synapse is inhibitory.

Requirements
    * Sections 1–2 (build the dict / write files) are pure NumPy — no GPU needed.
    * Section 3 (actually simulate) needs a CUDA/ROCm GPU and ``superneuroabm``
      installed (``pip install -e .``).

Run:  python examples/brunel_network_example.py
      python examples/brunel_network_example.py --simulate   # also runs on GPU
"""

import argparse
import tempfile
from pathlib import Path

import numpy as np

from superneuroabm.brunel import (
    brunel_partition,
    build_brunel_network,
    save_brunel_partition,
    brunel_external_rate,
)


def section_1_in_memory():
    """Entry point 1: build a whole network in memory (single process)."""
    print("\n" + "=" * 70)
    print("1) build_brunel_network(...) -> (somas, synapses) in memory")
    print("=" * 70)

    # A small network: 1000 neurons, each with 80 excitatory + 20 inhibitory
    # recurrent inputs and 1 external input. Weights inherit the synapse config
    # (excitatory), and inhibitory synapses are -g * that. Everything is keyword-only.
    somas, synapses = build_brunel_network(
        somas_per_rank=1000,          # whole network (num_partitions defaults to 1)
        excitatory_in_degree=80,      # C_E : exactly 80 excitatory inputs per soma
        inhibitory_in_degree=20,      # C_I : exactly 20 inhibitory inputs per soma
        inhibitory_weight_ratio=5.0,  # g   : inhibitory weight = -5 * excitatory weight
        # excitatory_weight=None      -> inherit component_base_config.yaml (14.0)
        # synaptic_delay_ms=None      -> inherit config delay
        excitatory_fraction=0.8,      # 800 excitatory somas, 200 inhibitory
        seed=42,
    )
    n_ext = sum(1 for s in synapses if s["pre"] == -1)
    n_rec = len(synapses) - n_ext
    print(f"  somas       = {len(somas)}")
    print(f"  synapses    = {len(synapses)}  ({n_rec} recurrent + {n_ext} external)")
    print(f"  per soma    = {n_rec // len(somas)} recurrent (= 80 + 20)  + "
          f"{n_ext // len(somas)} external")
    print("  -> feed these straight to model.create_from_lists(somas, synapses)")

    # The alternative connectivity input: density instead of exact counts.
    somas2, synapses2 = build_brunel_network(
        somas_per_rank=1000,
        excitatory_in_degree=None,        # leave the counts unset...
        inhibitory_in_degree=None,
        connection_probability=0.1,       # ...and derive them: 0.1*800=80, 0.1*200=20
    )
    print(f"  (connection_probability=0.1 gives the same {len(synapses2)} synapses)")
    return somas, synapses


def section_2_partition_files():
    """Entry point 2/3: write per-rank files, and inspect the raw partition dict."""
    print("\n" + "=" * 70)
    print("2) save_brunel_partition(...) -> per-rank partition_{rank}.pkl")
    print("=" * 70)

    out_dir = Path(tempfile.mkdtemp(prefix="brunel_example_"))
    num_partitions = 2  # pretend we have 2 MPI ranks / 2 GPUs

    # In a real run each MPI rank calls this once for its OWN partition_rank.
    # Here we loop over both to show what a 2-rank build produces.
    for partition_rank in range(num_partitions):
        path = save_brunel_partition(
            out_dir,
            somas_per_rank=500,           # 500 per rank -> 1000 total (DERIVED)
            num_partitions=num_partitions,
            partition_rank=partition_rank,
            excitatory_in_degree=40,
            inhibitory_in_degree=10,
        )
        print(f"  rank {partition_rank}: wrote {path}")

    print("  -> each rank then: model.load_post_owned(f'partition_{rank}.pkl')")

    print("\n" + "=" * 70)
    print("3) brunel_partition(...) -> the raw dict (what the wrappers build on)")
    print("=" * 70)

    # Rank 0 of the 2-rank build. Note the recurrent sources are drawn from the WHOLE
    # population, so some presynaptic somas live on rank 1 -> they appear in
    # remote_ranks (the cross-rank / ghost boundary).
    p0 = brunel_partition(
        somas_per_rank=500, num_partitions=2, partition_rank=0,
        excitatory_in_degree=40, inhibitory_in_degree=10,
    )
    print(f"  rank 0 keys        : {sorted(p0.keys())}")
    print(f"  local somas        : {len(p0['somas'])} (ids 0..499)")
    print(f"  local synapses     : {len(p0['synapses'])}")
    print(f"  remote pre-somas   : {len(p0['remote_ranks'])} "
          f"(presynaptic somas owned by rank 1)")

    # A single-partition build has no remote somas, so remote_ranks is OMITTED.
    p_single = brunel_partition(somas_per_rank=100, num_partitions=1,
                                excitatory_in_degree=8, inhibitory_in_degree=2)
    print(f"  single-rank keys   : {sorted(p_single.keys())} "
          f"(no 'remote_ranks' when num_partitions=1)")


def section_4_external_rate():
    """Optional helper: NEST-style analytic external drive rate."""
    print("\n" + "=" * 70)
    print("4) brunel_external_rate(...) -> external Poisson rate (Hz), optional")
    print("=" * 70)
    rate = brunel_external_rate(
        relative_rate=2.0,          # eta: 2x threshold rate
        excitatory_weight=14.0,
        excitatory_in_degree=80,
        threshold=20.0,
        membrane_time_constant_ms=20.0,
    )
    print(f"  analytic external rate = {rate:.1f} Hz")
    print("  (optional — with the biophysical LIF preset you may pass an explicit rate)")


def section_5_simulate(somas, synapses):
    """Actually build the network on a GPU and simulate a few ticks."""
    print("\n" + "=" * 70)
    print("5) Simulate on GPU (needs a CUDA/ROCm device)")
    print("=" * 70)
    from superneuroabm.model import NeuromorphicModel

    model = NeuromorphicModel(enable_internal_states_tracking=False)
    model.create_from_lists(somas, synapses)     # the (somas, synapses) from section 1
    model.setup(use_gpu=True)
    print("  setup(use_gpu=True) OK")

    # Drive the external (pre == -1) input synapses with a Poisson spike train.
    ticks, rate_hz, dt_ms = 10, 10.0, 1.0
    input_synapses = [sid for sid in model._synapse_ids
                      if model.get_synapse_connectivity(sid)[0] == -1]
    rng = np.random.default_rng(np.random.SeedSequence([42, 7]))
    p_spike = rate_hz * dt_ms / 1000.0
    for syn_id in input_synapses:
        spike_ticks = np.nonzero(rng.random(ticks) < p_spike)[0] + 1  # 1-indexed
        if spike_ticks.size:
            model.add_spike_list(syn_id, [[int(t), 1.0] for t in spike_ticks])

    model.simulate(ticks=ticks, update_data_ticks=1)
    print(f"  simulate({ticks} ticks) OK")
    print("  SUCCESS")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--simulate", action="store_true",
                        help="also build on GPU and run a few ticks (needs a GPU)")
    args = parser.parse_args()

    somas, synapses = section_1_in_memory()
    section_2_partition_files()
    section_4_external_rate()
    if args.simulate:
        section_5_simulate(somas, synapses)
    else:
        print("\n(Re-run with --simulate to also build on a GPU and step the network.)")


if __name__ == "__main__":
    main()
