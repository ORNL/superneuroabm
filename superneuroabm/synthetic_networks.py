"""
Synthetic network generation for distributed SNN scaling tests.

Generates a **Brunel balanced random network** (the standard SNN scaling
benchmark, following NEST's ``hpc_benchmark``) one MPI rank at a time, writing
each rank's partition directly to a file in the schema consumed by
``NeuromorphicModel.load_from_file()``::

    {
      "somas":    [{"id", "breed", "config", "overrides"}, ...],
      "synapses": [{"id", "pre", "post", "breed", "config",
                    "learning_rule", "learning_rule_config", "overrides"}, ...],
      "remote_ranks": {agent_id: rank},   # any neighbor soma not owned locally
    }

Design notes
------------
* **Local-only, vectorized generation.** Each rank materializes connectivity for
  *only its own* neurons (a per-target draw), so generation is ``O(npp * K)`` and
  never builds the global graph. This is how large-scale SNN simulators (NEST,
  Jordan et al. 2018) keep weak scaling flat.
* **Fixed in-degree K** (default 1000) drawn uniformly from the global
  population. Synapses scale ``O(N)`` -> correct weak scaling. No explicit
  "cross-rank" knob: cross-rank traffic emerges from the partition (remote
  fraction ~ ``(P-1)/P``).
* **Inhibition-dominated balance** (``g = |J_I|/J_E = 5``), E:I = 4:1, synaptic
  delay 1.5 ms, and one external (``pre = -1``) Poisson-driven input synapse per
  neuron -- the asynchronous-irregular regime used in benchmarks.

References: Brunel (2000); Potjans & Diesmann (2014); Kunkel et al. (2014);
Jordan et al. (2018); NEST ``hpc_benchmark.py``.
"""

import pickle
from pathlib import Path

import numpy as np


def generate_and_save_local_partition(
    output_dir: str,
    my_rank: int,
    num_partitions: int,
    neurons_per_partition: int,
    in_degree: int = 1000,
    g: float = 5.0,
    J_E: float = 14.0,
    delay: float = 1.5,
    excitatory_ratio: float = 0.8,
    external_rate_hz: float = 10.0,
    soma_breed: str = "lif_soma",
    soma_config: str = "config_0",
    synapse_breed: str = "single_exp_synapse",
    synapse_config: str = "config_0",
    seed: int = 42,
) -> str:
    """Generate and save this rank's partition of a Brunel balanced network.

    Builds the connectivity for this rank's local neurons only (no global graph
    materialization) and writes ``partition_{my_rank}.pkl`` in the SNN-native
    schema consumed by ``NeuromorphicModel.load_from_file()``. Each rank can call
    this independently -- no MPI / communication is needed during generation.

    Each local neuron is a postsynaptic target that receives exactly ``in_degree``
    recurrent synapses whose presynaptic sources are drawn uniformly at random
    from the whole population ``[0, N)`` (``N = num_partitions *
    neurons_per_partition``). A source is excitatory iff its global id is in the
    first ``excitatory_ratio`` fraction of ids; excitatory synapses get weight
    ``J_E``, inhibitory get ``J_I = -g * J_E``. Every neuron additionally gets one
    external input synapse (``pre = -1``) driven by the harness at
    ``external_rate_hz``.

    Synapse ids are a closed-form, globally-unique function of (target, in-edge
    index), so no cross-rank coordination is required::

        soma id        = t                  (owned by rank t // neurons_per_partition)
        recurrent syn  = N + t*K + j         (j-th input of target t)
        external syn   = N + N*K + t         (external drive for target t)

    .. note::
        Sources are drawn with ``np.random.Generator.integers``. Multapses
        (repeated source for the same target) are allowed, matching NEST's
        default benchmark draw. Autapses (self-connections, ``pre == post``)
        are resampled away: a synapse stores its endpoints in a positional
        2-slot neighbor list ``[pre, post]`` that SAGESim deduplicates, so
        ``pre == post`` would collapse to one slot and lose the post endpoint.
        They are irrelevant to a static-weight scaling benchmark (no STDP), so
        excluding them keeps the draw faithful while staying loadable.

    :param output_dir: Directory to write ``partition_{my_rank}.pkl``.
    :param my_rank: This rank's id in ``[0, num_partitions)``.
    :param num_partitions: Total number of partitions (MPI world size).
    :param neurons_per_partition: Neurons owned by each partition.
    :param in_degree: Fixed in-degree K per neuron (recurrent sources).
    :param g: Relative inhibitory strength ``|J_I| / J_E`` (>1 => inhibition-dominated).
    :param J_E: Excitatory synaptic weight; ``J_I = -g * J_E``.
    :param delay: Synaptic delay in ms (written to ``overrides.hyperparameters``).
    :param excitatory_ratio: Fraction of excitatory neurons (E:I = 4:1 => 0.8).
    :param external_rate_hz: External Poisson drive per neuron (consumed by the harness).
    :param soma_breed: Breed name for somas.
    :param soma_config: Config name for somas.
    :param synapse_breed: Breed name for synapses.
    :param synapse_config: Config name for synapses.
    :param seed: Base random seed (combined with ``my_rank`` for the local draw).
    :return: Path to the saved partition file.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    P = num_partitions
    npp = neurons_per_partition
    r = my_rank
    K = in_degree
    N = P * npp

    lo, hi = r * npp, (r + 1) * npp
    exc_boundary = int(excitatory_ratio * N)
    J_I = -g * J_E

    SYN_BASE = N
    EXT_BASE = N + N * K

    rng = np.random.default_rng(np.random.SeedSequence([seed, r]))

    # --- Somas owned by this rank ---
    local_ids = np.arange(lo, hi, dtype=np.int64)
    somas = [
        {"id": int(t), "breed": soma_breed, "config": soma_config, "overrides": {}}
        for t in local_ids
    ]

    # --- Recurrent synapses: K random presynaptic sources per local target ---
    # pre[t, j] = uniform source in [0, N); synapse for (target t, in-edge j).
    pre = rng.integers(0, N, size=(npp, K), dtype=np.int64)          # (npp, K)
    # Forbid autapses (pre == post == this target). Self-loops are valid in the
    # NEST/Brunel statistical draw, but our synapse stores its endpoints in a
    # positional 2-slot neighbor list [pre, post] and SAGESim deduplicates
    # neighbors, so pre == post would collapse to one slot and drop the post
    # endpoint. They contribute nothing to a static-weight scaling benchmark
    # (no STDP), so resample any target-self draw to a different source.
    self_mask = pre == local_ids[:, None]                            # (npp, K)
    while self_mask.any():
        pre[self_mask] = rng.integers(0, N, size=int(self_mask.sum()), dtype=np.int64)
        self_mask = pre == local_ids[:, None]
    post = np.repeat(local_ids, K)                                    # (npp*K,)
    j_idx = np.tile(np.arange(K, dtype=np.int64), npp)                # (npp*K,)
    pre_flat = pre.ravel()                                            # (npp*K,)
    syn_ids = SYN_BASE + post * K + j_idx                             # closed-form ids
    weights = np.where(pre_flat < exc_boundary, J_E, J_I)             # signed weights

    synapses = [
        {
            "id": int(sid),
            "pre": int(p),
            "post": int(q),
            "breed": synapse_breed,
            "config": synapse_config,
            "learning_rule": None,
            "learning_rule_config": "default",
            "overrides": {"hyperparameters": {"weight": float(w),
                                              "synaptic_delay": float(delay)}},
        }
        for sid, p, q, w in zip(syn_ids, pre_flat, post, weights)
    ]

    # --- External input synapses: one per local neuron (pre = -1) ---
    ext_ids = EXT_BASE + local_ids
    synapses.extend(
        {
            "id": int(sid),
            "pre": -1,
            "post": int(t),
            "breed": synapse_breed,
            "config": synapse_config,
            "learning_rule": None,
            "learning_rule_config": "default",
            "overrides": {"hyperparameters": {"weight": float(J_E),
                                              "synaptic_delay": float(delay)}},
        }
        for sid, t in zip(ext_ids, local_ids)
    )

    # --- remote_ranks: any presynaptic soma not owned locally ---
    # (External pre = -1 is excluded.) SAGESim uses this to fetch ghosts.
    remote_pre = np.unique(pre_flat[(pre_flat < lo) | (pre_flat >= hi)])
    remote_ranks = {int(p): int(p) // npp for p in remote_pre}

    partition = {"somas": somas, "synapses": synapses, "remote_ranks": remote_ranks}

    out_file = out_dir / f"partition_{r}.pkl"
    with open(out_file, "wb") as f:
        pickle.dump(partition, f)

    return str(out_file)
