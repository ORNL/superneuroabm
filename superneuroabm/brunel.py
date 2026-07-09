"""
General-purpose **Brunel balanced random network** generator for SuperNeuroABM.

The Brunel network (Brunel 2000; NEST ``hpc_benchmark``) is the canonical SNN
benchmark: a sparse, balanced, random recurrent LIF network with an excitatory and
an inhibitory population (4:1), fixed in-degree, inhibition-dominated weights, and
external Poisson drive. This module builds it and emits the ``{somas, synapses,
remote_ranks}`` post-owns partition contract that ``NeuromorphicModel`` already
consumes -- see ``docs/PARTITION_LOADING.md`` for the format, and
``docs/BRUNEL_SCALING.md`` for the design rationale.

This is a **general-purpose generator**, not a scaling-test helper: it takes Brunel
parameters and produces a network for whatever use -- an interactive build, a unit
test, a single-GPU correctness oracle, or a distributed scaling run. Scaling scripts
are just one caller.

Three entry points:

* :func:`brunel_partition` -- the core. Builds **one partition's** slice (set
  ``num_partitions=1`` for a whole single-process network). Returns the partition dict.
* :func:`build_brunel_network` -- convenience for the single-process / in-memory path.
  Returns ``(somas, synapses)`` ready for ``NeuromorphicModel.create_from_lists``.
* :func:`save_brunel_partition` -- writes one rank's ``partition_{rank}.pkl`` for a
  distributed run consumed by ``NeuromorphicModel.load_post_owned``.

Plus :func:`brunel_external_rate`, an optional helper for NEST's analytic drive rate.

Design notes
------------
* **Per-rank sizing, global draw.** The user gives ``somas_per_rank``; the global
  total is *derived* (``somas_per_rank * num_partitions``). Each rank owns a
  contiguous soma-id block, but recurrent presynaptic sources are drawn from the
  **whole** population -- that global draw is what produces the cross-rank
  ``remote_ranks`` boundary.
* **Faithful two-pool topology.** Each soma receives *exactly* ``excitatory_in_degree``
  sources from the E-population and ``inhibitory_in_degree`` from the I-population
  (NEST/Brunel convention). In-degree is fixed; out-degree is emergent.
* **Local-only, vectorized.** A rank materializes connectivity for *only its own*
  neurons (``O(somas_per_rank * in_degree)``), never the global graph. No MPI during
  generation -- ranks are independent.
* **The inhibitory sign lives in the weight.** The ``single_exp_synapse`` kernel uses
  the stored weight directly (no sign logic), so an inhibitory synapse must store a
  *negative* weight. See :func:`brunel_partition` for details.

References: Brunel (2000); Kunkel et al. (2014); Jordan et al. (2018);
NEST ``hpc_benchmark.py``.
"""

import pickle
from pathlib import Path

import numpy as np

from superneuroabm.util import load_component_configurations


def _config_synapse_weight(synapse_breed, synapse_config):
    """Read the default synaptic weight from the component base config (as float)."""
    configs = load_component_configurations()
    try:
        return float(configs["synapse"][synapse_breed][synapse_config]
                     ["hyperparameters"]["weight"])
    except KeyError as exc:
        raise KeyError(
            f"No hyperparameters.weight for synapse breed '{synapse_breed}' config "
            f"'{synapse_config}' in component_base_config.yaml."
        ) from exc


def _draw_sources(rng, num_targets, in_degree, pool_lo, pool_hi,
                  target_ids, allow_multapses, allow_autapses):
    """Draw an (num_targets, in_degree) array of presynaptic ids from [pool_lo, pool_hi).

    Vectorized uniform draw. ``target_ids`` (shape (num_targets,)) is the global id of
    each local target, used to forbid autapses when the target lies inside this pool.
    ``allow_multapses=False`` resamples until each target's row has unique sources.
    """
    if in_degree == 0:
        return np.empty((num_targets, 0), dtype=np.int64)
    pool_size = pool_hi - pool_lo
    if not allow_multapses and in_degree > pool_size:
        raise ValueError(
            f"in_degree {in_degree} exceeds pool size {pool_size} with "
            "allow_multapses=False (cannot draw that many unique sources)."
        )
    src = rng.integers(pool_lo, pool_hi, size=(num_targets, in_degree), dtype=np.int64)

    # Forbid autapses: a source equal to its own target. Only relevant when the
    # target's id falls inside [pool_lo, pool_hi). Resample offending entries.
    if not allow_autapses:
        self_mask = (src == target_ids[:, None]) & \
                    (target_ids[:, None] >= pool_lo) & (target_ids[:, None] < pool_hi)
        while self_mask.any():
            src[self_mask] = rng.integers(pool_lo, pool_hi,
                                          size=int(self_mask.sum()), dtype=np.int64)
            self_mask = (src == target_ids[:, None]) & \
                        (target_ids[:, None] >= pool_lo) & (target_ids[:, None] < pool_hi)

    # Forbid multapses: repeated source within a target's row. Resample per-row
    # duplicates until every row is unique (also re-checks autapses via the loop above
    # being applied first; here we only dedup, and autapse-free is preserved because a
    # resampled value is re-checked next iteration).
    if not allow_multapses:
        for _ in range(1000):  # generous cap; uniqueness converges fast for K << pool
            done = True
            for t in range(num_targets):
                row = src[t]
                # find duplicates (and, if forbidden, autapses) to resample
                _, first_idx, counts = np.unique(row, return_index=True, return_counts=True)
                dup_positions = np.setdiff1d(np.arange(in_degree), first_idx)
                bad = list(dup_positions)
                if not allow_autapses and pool_lo <= target_ids[t] < pool_hi:
                    bad.extend(np.nonzero(row == target_ids[t])[0].tolist())
                bad = np.unique(np.asarray(bad, dtype=np.int64))
                if bad.size:
                    done = False
                    row[bad] = rng.integers(pool_lo, pool_hi, size=bad.size, dtype=np.int64)
            if done:
                break
        else:
            raise RuntimeError("multapse/autapse resampling did not converge.")
    return src


def brunel_partition(
    *,
    # --- size: per-rank primary, global total DERIVED ---
    somas_per_rank: int = 12500,
    num_partitions: int = 1,
    partition_rank: int = 0,
    excitatory_fraction: float = 0.8,
    # --- connectivity: in-degree primary, probability derives it ---
    excitatory_in_degree: int | None = 1000,
    inhibitory_in_degree: int | None = 250,
    connection_probability: float | None = None,
    # --- weights / dynamics: None = inherit the named config; a value overrides ---
    excitatory_weight: float | None = None,
    inhibitory_weight_ratio: float = 5.0,
    synaptic_delay_ms: float | None = None,
    # --- external drive topology ---
    external_synapses_per_soma: int = 1,
    external_weight: float | None = None,
    # --- multapse / autapse toggles ---
    allow_multapses: bool = True,
    allow_autapses: bool = False,
    # --- breeds / configs (SuperNeuroABM component names) ---
    soma_breed: str = "lif_soma",
    soma_config: str = "config_0",
    synapse_breed: str = "single_exp_synapse",
    synapse_config: str = "config_0",
    # --- reproducibility ---
    seed: int = 42,
) -> dict:
    """Build one partition's slice of a Brunel balanced random network.

    Returns the post-owns partition contract consumed by
    ``NeuromorphicModel.load_post_owned`` / ``create_from_lists``::

        {"somas": [...], "synapses": [...], "remote_ranks": {id: rank}}

    For a single-partition network (``num_partitions == 1``) there are no remote
    somas, so ``remote_ranks`` is **omitted** and the return is just
    ``{"somas": [...], "synapses": [...]}`` (the loader defaults a missing
    ``remote_ranks`` to ``{}``).

    Sizing
        ``somas_per_rank`` is the per-partition soma count (equal for every
        partition); the global ``total_somas = somas_per_rank * num_partitions`` is
        derived. This partition owns the contiguous soma-id block
        ``[partition_rank * somas_per_rank, (partition_rank + 1) * somas_per_rank)``.

    Topology (faithful two-pool)
        The first ``round(excitatory_fraction * total_somas)`` ids are excitatory,
        the rest inhibitory. Each local soma receives **exactly**
        ``excitatory_in_degree`` recurrent synapses whose presynaptic sources are
        drawn uniformly from the E-pool, and **exactly** ``inhibitory_in_degree`` from
        the I-pool. In-degree is fixed; out-degree is emergent (≈ in-degree on
        average, Poisson-distributed). Multapses follow ``allow_multapses`` (NEST
        default: allowed); autapses (``pre == post``) are resampled away by default
        because SAGESim's positional ``[pre, post]`` neighbor slot dedups a self-loop.

    Weights (config-driven; the inhibitory sign is topological)
        The *effective excitatory weight* is ``excitatory_weight`` if given, else the
        value in ``component_base_config.yaml`` for the named synapse config. An
        excitatory synapse under ``excitatory_weight=None`` carries **no weight
        override** (it inherits the config verbatim); if a value is given it is written
        as an override. An **inhibitory synapse always** carries a weight override
        equal to ``-inhibitory_weight_ratio * effective_excitatory_weight`` -- its sign
        is decided by the draw (source pool), which the config cannot know.

        .. important::
            The ``single_exp_synapse`` kernel integrates ``spike * scale * weight``
            **directly, with no sign logic**, and the presynaptic spike is a
            non-negative magnitude. So an inhibitory synapse **must** store a negative
            weight; passing a positive one would make it excitatory and unbalance the
            network.

    External drive
        Each local soma gets ``external_synapses_per_soma`` input synapses with
        ``pre = -1`` (driven by the harness at run time). Their weight is
        ``external_weight`` if given, else the effective excitatory weight.

    Ids
        Soma ids are ``[0, total_somas)``. Synapse ids are a closed-form,
        globally-unique function of (target, in-edge index) starting at
        ``total_somas``, so soma and synapse id spaces never collide and no cross-rank
        coordination is needed.

    :param somas_per_rank: Somas owned by each partition (equal across partitions).
    :param num_partitions: Total number of partitions (e.g. MPI world size).
    :param partition_rank: This partition's index in ``[0, num_partitions)``.
    :param excitatory_fraction: Fraction of somas that are excitatory (Brunel: 0.8).
    :param excitatory_in_degree: Exactly this many excitatory recurrent inputs per
        soma (``None`` -> derive from ``connection_probability``).
    :param inhibitory_in_degree: Exactly this many inhibitory recurrent inputs per
        soma (``None`` -> derive from ``connection_probability``).
    :param connection_probability: Connection density; derives an in-degree left
        ``None`` (``in_degree = round(p * pool_size)``).
    :param excitatory_weight: Excitatory synaptic weight; ``None`` inherits the synapse
        config's weight.
    :param inhibitory_weight_ratio: ``g``; inhibitory weight is
        ``-g * effective_excitatory_weight``.
    :param synaptic_delay_ms: Synaptic delay override in ms; ``None`` inherits the
        synapse config's delay.
    :param external_synapses_per_soma: Number of ``pre = -1`` input synapses per soma.
    :param external_weight: External-input synaptic weight; ``None`` uses the effective
        excitatory weight.
    :param allow_multapses: Allow a repeated source for the same target (NEST default).
    :param allow_autapses: Allow ``pre == post`` self-connections (default ``False``;
        resampled away).
    :param soma_breed: Breed name for somas.
    :param soma_config: Config name for somas.
    :param synapse_breed: Breed name for synapses.
    :param synapse_config: Config name for synapses.
    :param seed: Base random seed (combined with ``partition_rank`` for the draw).
    :return: The partition dict (see above).
    """
    if somas_per_rank <= 0:
        raise ValueError(f"somas_per_rank must be positive (got {somas_per_rank}).")
    if num_partitions <= 0:
        raise ValueError(f"num_partitions must be positive (got {num_partitions}).")
    if not (0 <= partition_rank < num_partitions):
        raise ValueError(
            f"partition_rank {partition_rank} out of range [0, {num_partitions})."
        )
    if not (0.0 < excitatory_fraction < 1.0):
        raise ValueError(
            f"excitatory_fraction must be in (0, 1) (got {excitatory_fraction})."
        )

    npp = somas_per_rank
    P = num_partitions
    r = partition_rank
    total_somas = npp * P

    excitatory_somas = round(excitatory_fraction * total_somas)
    inhibitory_somas = total_somas - excitatory_somas
    if excitatory_somas <= 0 or inhibitory_somas <= 0:
        raise ValueError(
            f"E/I split degenerate: total={total_somas}, E={excitatory_somas}, "
            f"I={inhibitory_somas}. Increase total_somas or adjust excitatory_fraction."
        )

    C_E, C_I = excitatory_in_degree, inhibitory_in_degree
    if C_E is None or C_I is None:
        if connection_probability is None:
            raise ValueError(
                "Connectivity underspecified: give excitatory_in_degree and "
                "inhibitory_in_degree, or a connection_probability to derive them."
            )
        if C_E is None:
            C_E = round(connection_probability * excitatory_somas)
        if C_I is None:
            C_I = round(connection_probability * inhibitory_somas)
    if C_E < 0 or C_I < 0:
        raise ValueError(f"in-degrees must be non-negative (got C_E={C_E}, C_I={C_I}).")

    K = C_E + C_I  # recurrent in-degree per soma
    ext_per = external_synapses_per_soma

    # Effective excitatory weight: explicit value, else the synapse config's weight.
    eff_exc_w = (float(excitatory_weight) if excitatory_weight is not None
                 else _config_synapse_weight(synapse_breed, synapse_config))
    inh_w = -float(inhibitory_weight_ratio) * eff_exc_w
    ext_w = float(external_weight) if external_weight is not None else eff_exc_w

    lo, hi = r * npp, (r + 1) * npp
    # Closed-form id bases (global, collision-free with soma ids [0, total)).
    SYN_BASE = total_somas
    EXT_BASE = total_somas + total_somas * K

    rng = np.random.default_rng(np.random.SeedSequence([seed, r]))

    local_ids = np.arange(lo, hi, dtype=np.int64)

    # --- Somas owned by this partition ---
    somas = [
        {"id": int(t), "breed": soma_breed, "config": soma_config, "overrides": {}}
        for t in local_ids
    ]

    # --- Recurrent synapses: exactly C_E from E-pool + C_I from I-pool per target ---
    pre_E = _draw_sources(rng, npp, C_E, 0, excitatory_somas,
                          local_ids, allow_multapses, allow_autapses)          # (npp, C_E)
    pre_I = _draw_sources(rng, npp, C_I, excitatory_somas, total_somas,
                          local_ids, allow_multapses, allow_autapses)          # (npp, C_I)
    pre = np.concatenate([pre_E, pre_I], axis=1)                                # (npp, K)

    post = np.repeat(local_ids, K)                                             # (npp*K,)
    j_idx = np.tile(np.arange(K, dtype=np.int64), npp)                         # (npp*K,)
    pre_flat = pre.ravel()                                                     # (npp*K,)
    syn_ids = SYN_BASE + post * K + j_idx                                      # closed-form
    # Source is excitatory iff its id < excitatory_somas (id-based E/I identity).
    is_exc = pre_flat < excitatory_somas
    weights = np.where(is_exc, eff_exc_w, inh_w)

    synapses = []
    for sid, p, q, w, exc in zip(syn_ids, pre_flat, post, weights, is_exc):
        # Excitatory: only override weight if the user set an explicit value (else
        # inherit config). Inhibitory: always override (the sign is not in the config).
        hyper = {}
        if (not exc) or (excitatory_weight is not None):
            hyper["weight"] = float(w)
        if synaptic_delay_ms is not None:
            hyper["synaptic_delay"] = float(synaptic_delay_ms)
        synapses.append({
            "id": int(sid),
            "pre": int(p),
            "post": int(q),
            "breed": synapse_breed,
            "config": synapse_config,
            "learning_rule": None,
            "learning_rule_config": "default",
            "overrides": {"hyperparameters": hyper} if hyper else {},
        })

    # --- External input synapses: external_synapses_per_soma per local soma (pre=-1) ---
    for k in range(ext_per):
        ext_ids = EXT_BASE + local_ids * ext_per + k
        for sid, t in zip(ext_ids, local_ids):
            hyper = {}
            # Write the external weight override only when it differs from the config
            # default (i.e. the user set excitatory_weight or external_weight); an
            # all-defaults external synapse inherits the config weight.
            if external_weight is not None or excitatory_weight is not None:
                hyper["weight"] = float(ext_w)
            if synaptic_delay_ms is not None:
                hyper["synaptic_delay"] = float(synaptic_delay_ms)
            synapses.append({
                "id": int(sid),
                "pre": -1,
                "post": int(t),
                "breed": synapse_breed,
                "config": synapse_config,
                "learning_rule": None,
                "learning_rule_config": "default",
                "overrides": {"hyperparameters": hyper} if hyper else {},
            })

    partition = {"somas": somas, "synapses": synapses}

    # --- remote_ranks: any presynaptic soma not owned locally (omit when single-rank) ---
    if P > 1:
        remote_pre = np.unique(pre_flat[(pre_flat < lo) | (pre_flat >= hi)])
        partition["remote_ranks"] = {int(p): int(p) // npp for p in remote_pre}

    return partition


def build_brunel_network(**kwargs):
    """Build a whole Brunel network in memory (single process).

    Convenience wrapper over :func:`brunel_partition` with ``num_partitions=1`` (so
    ``somas_per_rank`` *is* the whole network), returning ``(somas, synapses)`` ready
    for ``NeuromorphicModel.create_from_lists(somas, synapses, ...)``. No disk, no
    rank/remote concept. Ideal for tests, interactive builds, and a single-GPU oracle.

    Accepts every keyword of :func:`brunel_partition` except ``num_partitions`` and
    ``partition_rank`` (both forced to the single-process values).

    :return: ``(somas, synapses)`` lists.
    """
    for reserved in ("num_partitions", "partition_rank"):
        if reserved in kwargs:
            raise TypeError(
                f"build_brunel_network() does not accept '{reserved}' "
                "(it builds a single-process network). Use brunel_partition() for "
                "multi-partition builds."
            )
    partition = brunel_partition(num_partitions=1, partition_rank=0, **kwargs)
    return partition["somas"], partition["synapses"]


def save_brunel_partition(output_dir, *, filename=None, **kwargs):
    """Generate one partition and pickle it to ``output_dir/partition_{rank}.pkl``.

    Convenience wrapper over :func:`brunel_partition` that writes the partition dict to
    disk in the schema ``NeuromorphicModel.load_post_owned`` reads. Each MPI rank calls
    this for its own ``partition_rank`` independently -- no communication is needed
    during generation.

    :param output_dir: Directory to write into (created if missing).
    :param filename: Output filename; defaults to ``partition_{partition_rank}.pkl``.
    :param kwargs: Every keyword of :func:`brunel_partition`.
    :return: Path to the saved partition file (str).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    partition_rank = kwargs.get("partition_rank", 0)
    partition = brunel_partition(**kwargs)
    name = filename if filename is not None else f"partition_{partition_rank}.pkl"
    out_file = out_dir / name
    with open(out_file, "wb") as f:
        pickle.dump(partition, f)
    return str(out_file)


def brunel_external_rate(*, relative_rate=2.0, excitatory_weight=14.0,
                         excitatory_in_degree=1000, threshold=20.0,
                         membrane_time_constant_ms=20.0):
    """NEST-style external Poisson drive rate (Hz), the benchmark's ``p_rate``.

    Computes the threshold rate ``nu_thr = threshold / (J * C_E * tau_m)`` and returns
    ``relative_rate * nu_thr * C_E`` (Hz), the total external Poisson rate a neuron must
    receive to sit at ``relative_rate`` (``eta``) times threshold.

    Provided for parity with the NEST ``hpc_benchmark`` definition. With SuperNeuroABM's
    biophysical LIF preset the analytic rate is a calibration item, so callers may
    instead pass an explicit rate; this helper is optional.

    :param relative_rate: ``eta`` = external rate relative to threshold rate.
    :param excitatory_weight: ``J`` (same units as threshold).
    :param excitatory_in_degree: ``C_E``.
    :param threshold: Firing threshold ``theta``.
    :param membrane_time_constant_ms: ``tau_m`` in ms.
    :return: External Poisson rate in Hz.
    """
    tau_m_s = membrane_time_constant_ms / 1000.0
    nu_thr = threshold / (excitatory_weight * excitatory_in_degree * tau_m_s)
    return relative_rate * nu_thr * excitatory_in_degree
