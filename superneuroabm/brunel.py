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
* **Per-rank sizing.** The user gives ``somas_per_rank``; the global total is *derived*
  (``somas_per_rank * num_partitions``). Each rank owns a contiguous soma-id block, and
  whichever presynaptic sources fall outside it are what produce the cross-rank
  ``remote_ranks`` boundary. How wide that draw reaches is set by ``topology``: the whole
  population (``"global"``), own rank + a chosen peer set (``"bounded"``/``"torus2d"``), or
  a spatial ball around each target (``"torus3d"``).
* **Two-pool topology -- for every topology EXCEPT ``"torus3d"``.** Under ``"global"``,
  ``"bounded"`` and ``"torus2d"`` each soma receives *exactly* ``excitatory_in_degree``
  sources from the E-population and ``inhibitory_in_degree`` from the I-population
  (NEST/Brunel convention). ``"torus3d"`` instead does a **single-pool** draw of
  ``excitatory_in_degree + inhibitory_in_degree`` sources from the ball and reads each drawn
  source's E/I identity off its id, so its per-soma E/I mix is 4:1 only *on average over the
  population*, not per soma. In-degree is fixed either way; out-degree is emergent.
* **Local-only, vectorized.** A rank materializes connectivity for *only its own*
  neurons (``O(somas_per_rank * in_degree)``), never the global graph. No MPI during
  generation -- ranks are independent.
* **The inhibitory sign lives in the weight.** The ``single_exp_synapse`` kernel uses
  the stored weight directly (no sign logic), so an inhibitory synapse must store a
  *negative* weight. See :func:`brunel_partition` for details.

References: Brunel (2000); Kunkel et al. (2014); Jordan et al. (2018);
NEST ``hpc_benchmark.py``.
"""

import math
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


def _normalize_ranges(ranges):
    """Sort, drop empties, and merge adjacent/overlapping ``[lo, hi)`` id ranges.

    Returns ``(los, his, offsets, pool_size)`` where ``los``/``his`` are int64 arrays of
    the merged half-open ranges, ``offsets`` is the exclusive prefix-sum of range sizes
    (so range ``k`` occupies flat indices ``[offsets[k], offsets[k+1])``), and
    ``pool_size`` is the total number of ids across all ranges.
    """
    clean = sorted((int(lo), int(hi)) for lo, hi in ranges if hi > lo)
    if not clean:
        return (np.empty(0, np.int64), np.empty(0, np.int64),
                np.zeros(1, np.int64), 0)
    merged = [list(clean[0])]
    for lo, hi in clean[1:]:
        if lo <= merged[-1][1]:            # overlaps/abuts the running range → extend
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    los = np.array([lo for lo, _ in merged], dtype=np.int64)
    his = np.array([hi for _, hi in merged], dtype=np.int64)
    sizes = his - los
    offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    return los, his, offsets, int(offsets[-1])


def _draw_from_ranges(rng, los, offsets, pool_size, shape):
    """Uniformly draw ids from the union of merged ranges (``shape`` many).

    Draws a flat index into ``[0, pool_size)`` and maps it back to a real id: find which
    range the flat index falls in (``searchsorted`` on the exclusive prefix sums) and add
    the in-range remainder to that range's ``lo``. Every id in the union is equiprobable.
    """
    flat = rng.integers(0, pool_size, size=shape, dtype=np.int64)
    k = np.searchsorted(offsets, flat, side="right") - 1   # range index per draw
    return los[k] + (flat - offsets[k])


def _draw_sources(rng, num_targets, in_degree, source_ranges,
                  target_ids, allow_multapses, allow_autapses):
    """Draw an (num_targets, in_degree) array of presynaptic ids from ``source_ranges``.

    ``source_ranges`` is a list of half-open ``[lo, hi)`` global-id ranges; sources are
    drawn uniformly from their **union** (a single ``[lo, hi)`` pool is just a one-element
    list). This generalization is what lets a target draw from an arbitrary set of source
    ranks (own rank + the chosen remote ranks) instead of one contiguous pool.

    Vectorized uniform draw. ``target_ids`` (shape (num_targets,)) is the global id of
    each local target, used to forbid autapses when the target lies inside the pool.
    ``allow_multapses=False`` resamples until each target's row has unique sources.
    """
    if in_degree == 0:
        return np.empty((num_targets, 0), dtype=np.int64)
    los, his, offsets, pool_size = _normalize_ranges(source_ranges)
    if pool_size == 0:
        raise ValueError("source_ranges is empty (no ids to draw sources from).")
    if not allow_multapses and in_degree > pool_size:
        raise ValueError(
            f"in_degree {in_degree} exceeds pool size {pool_size} with "
            "allow_multapses=False (cannot draw that many unique sources)."
        )

    def _in_pool(ids):
        """Boolean mask: which ids fall inside the union of ranges."""
        j = np.searchsorted(his, ids, side="right")        # first range with hi > id
        inside = j < los.size
        j_clipped = np.where(inside, j, 0)
        return inside & (ids >= los[j_clipped])

    src = _draw_from_ranges(rng, los, offsets, pool_size, (num_targets, in_degree))

    # Forbid autapses: a source equal to its own target. Only relevant when the target's
    # id falls inside the pool. Resample offending entries.
    if not allow_autapses:
        target_in_pool = _in_pool(target_ids)
        self_mask = (src == target_ids[:, None]) & target_in_pool[:, None]
        while self_mask.any():
            n_bad = int(self_mask.sum())
            src[self_mask] = _draw_from_ranges(rng, los, offsets, pool_size, (n_bad,))
            self_mask = (src == target_ids[:, None]) & target_in_pool[:, None]

    # Forbid multapses: repeated source within a target's row. Resample per-row
    # duplicates until every row is unique (autapse-free is preserved because a resampled
    # value is re-checked next iteration).
    if not allow_multapses:
        target_in_pool = _in_pool(target_ids)
        for _ in range(1000):  # generous cap; uniqueness converges fast for K << pool
            done = True
            for t in range(num_targets):
                row = src[t]
                # find duplicates (and, if forbidden, autapses) to resample
                _, first_idx, counts = np.unique(row, return_index=True, return_counts=True)
                dup_positions = np.setdiff1d(np.arange(in_degree), first_idx)
                bad = list(dup_positions)
                if not allow_autapses and target_in_pool[t]:
                    bad.extend(np.nonzero(row == target_ids[t])[0].tolist())
                bad = np.unique(np.asarray(bad, dtype=np.int64))
                if bad.size:
                    done = False
                    row[bad] = _draw_from_ranges(rng, los, offsets, pool_size, (bad.size,))
            if done:
                break
        else:
            raise RuntimeError("multapse/autapse resampling did not converge.")
    return src


def _select_remote_ranks(seed, partition_rank, num_partitions, remote_rank_fanout):
    """Pick exactly ``remote_rank_fanout`` distinct remote ranks for this partition.

    Seeded per partition (independent of the connectivity draw's seed stream) so the
    choice is reproducible and every neuron the partition owns shares the *same* remote
    peer set -- this is what bounds the per-rank peer count to ``remote_rank_fanout``
    (see the module docstring / BRUNEL_SCALING D4). Returns a sorted int list.
    """
    others = np.array([q for q in range(num_partitions) if q != partition_rank],
                      dtype=np.int64)
    rng = np.random.default_rng(np.random.SeedSequence([seed, partition_rank, 20240607]))
    chosen = rng.choice(others, size=remote_rank_fanout, replace=False)
    return sorted(int(q) for q in chosen)


def _grid_factorization(num_partitions):
    """Factor ``num_partitions`` into a near-square ``(a, b)`` torus grid, ``a <= b``.

    Picks ``a`` = the largest divisor of ``M`` not exceeding ``sqrt(M)`` (so the grid is as
    square as possible), ``b = M // a``. Requires ``a >= 3`` (hence ``b >= 3``): a 2D-torus
    partition needs *both* grid dimensions >= 3 so that a tile's eight Moore neighbors are
    eight *distinct* tiles (with a dimension of 1 or 2 the wraparound collapses neighbors and
    the peer count is < 8). Primes and ``M < 9`` therefore cannot form a valid grid.
    """
    M = num_partitions
    a = 1
    for d in range(1, math.isqrt(M) + 1):
        if M % d == 0:
            a = d
    b = M // a
    if a < 3:
        raise ValueError(
            f"num_partitions={M} cannot form a 2D-torus grid with both dimensions >= 3 "
            f"(best factorization {a}x{b}). torus2d needs each grid dimension >= 3 for a full "
            "8-neighbor stencil -- e.g. M = 9, 12, 16, 32, 64, ... work; primes and M < 9 do not."
        )
    return a, b


def _select_neighbor_tiles(partition_rank, a, b):
    """The 8 Moore-neighbor tile ranks of ``partition_rank`` on the ``a x b`` torus (wrapped).

    Rank ``r`` sits at grid position ``(r // b, r % b)``; its neighbors are the eight
    ``(dr, dc) != (0, 0)`` offsets, each wrapped modulo the grid dimension (torus, so edge
    tiles are not special). With ``a, b >= 3`` these are eight distinct ranks, none equal to
    ``r`` -- this is what pins the recurrent peer count to a hard constant 8, independent of
    ``num_partitions`` (the weak-scaling wiring invariant for a geometric partition). Returns a
    sorted int list.
    """
    tr, tc = divmod(partition_rank, b)
    neighbors = set()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            neighbors.add(((tr + dr) % a) * b + ((tc + dc) % b))
    neighbors.discard(partition_rank)
    return sorted(neighbors)


def _factor_near_cube(n):
    """Factor ``n`` into ``(gx, gy, gz)``, ``gx <= gy <= gz``, product ``n``, most cube-like.

    Used for both the **rank grid** (factor ``num_partitions`` into an ``a x b x c`` torus of
    tiles) and the **intra-tile grid** (factor ``somas_per_rank`` into a ``gx x gy x gz`` block of
    neurons). Unlike the strict 2D ``_grid_factorization``, dimensions of 1 or 2 are *allowed*:
    a small rank count (or a prime) simply yields a flatter grid, and the nearest-neighbor peer
    count then ramps up with the machine size and plateaus once every dimension is >= 3 (the
    weak-scaling "ramp-and-plateau"). Returns ``(gx, gy, gz)``.
    """
    if n <= 0:
        raise ValueError(f"cannot factor non-positive n={n}.")
    best = None
    for gx in range(1, round(n ** (1 / 3)) + 2):
        if n % gx:
            continue
        m = n // gx
        for gy in range(gx, math.isqrt(m) + 1):
            if m % gy:
                continue
            gz = m // gy
            if gz < gy:
                continue
            spread = gz - gx
            if best is None or spread < best[0]:
                best = (spread, (gx, gy, gz))
    return best[1] if best is not None else (1, 1, n)


def _grid_factorization_3d(num_partitions):
    """Factor ``num_partitions`` into a near-cube ``(a, b, c)`` torus grid, ``a <= b <= c``.

    The 3D analogue of ``_grid_factorization``, but **relaxed**: any ``num_partitions >= 1`` is
    accepted (dimensions may be 1 or 2). This is what lets the weak-scaling sweep start at a
    single GPU (``1x1x1``) and grow -- the per-rank Moore-neighbor peer count ramps (0, 1, ... )
    and saturates at 26 once ``a, b, c >= 3``. Returns ``(a, b, c)``.
    """
    return _factor_near_cube(num_partitions)


def _select_neighbor_tiles_3d(partition_rank, a, b, c):
    """The distinct Moore-neighbor tile ranks of ``partition_rank`` on the ``a x b x c`` torus.

    Rank ``r`` sits at grid position ``(tx, ty, tz)`` with
    ``r = tx*(b*c) + ty*c + tz``; its neighbors are the ``(dx, dy, dz) != 0`` offsets in
    ``{-1,0,1}^3`` (26 of them), each wrapped modulo the grid dimension (torus). With
    ``a, b, c >= 3`` these are **26 distinct** ranks; when a dimension is 1 or 2 the wraparound
    collapses neighbors, so fewer than 26 are returned (the ramp regime). Returns a sorted int list.
    """
    tx, rem = divmod(partition_rank, b * c)
    ty, tz = divmod(rem, c)
    neighbors = set()
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                neighbors.add(((tx + dx) % a) * (b * c)
                              + ((ty + dy) % b) * c + ((tz + dz) % c))
    neighbors.discard(partition_rank)
    return sorted(neighbors)


def _soma_positions(soma_ids, num_partitions, somas_per_rank):
    """Map soma id(s) to their ``(X, Y, Z)`` positions on the global 3D neuron grid.

    The bijection underlying the spatial wirings. The ``num_partitions`` ranks tile the grid as an
    ``a x b x c`` torus (``_grid_factorization_3d``) and each rank's contiguous id block is a
    ``gx x gy x gz`` neuron sub-block (``_factor_near_cube``), so ``id = rank*npp + intra`` maps to
    a unique lattice point **and** a rank's id block is exactly one spatial tile (post-owns holds,
    "spatially near == same/adjacent rank"). Accepts a scalar or array; returns ``(X, Y, Z)``
    int64 arrays of matching shape.
    """
    a, b, c = _grid_factorization_3d(num_partitions)
    gx, gy, gz = _factor_near_cube(somas_per_rank)
    ids = np.asarray(soma_ids, dtype=np.int64)
    rank, intra = np.divmod(ids, somas_per_rank)
    tx, trem = np.divmod(rank, b * c)
    ty, tz = np.divmod(trem, c)
    ix, irem = np.divmod(intra, gy * gz)
    iy, iz = np.divmod(irem, gz)
    return tx * gx + ix, ty * gy + iy, tz * gz + iz


def _positions_to_soma(X, Y, Z, num_partitions, somas_per_rank):
    """Inverse of :func:`_soma_positions`: global ``(X, Y, Z)`` -> soma id (matching shape)."""
    a, b, c = _grid_factorization_3d(num_partitions)
    gx, gy, gz = _factor_near_cube(somas_per_rank)
    tx, ix = np.divmod(np.asarray(X, np.int64), gx)
    ty, iy = np.divmod(np.asarray(Y, np.int64), gy)
    tz, iz = np.divmod(np.asarray(Z, np.int64), gz)
    rank = tx * (b * c) + ty * c + tz
    intra = ix * (gy * gz) + iy * gz + iz
    return rank * somas_per_rank + intra


def _ball_offsets(radius):
    """Integer ``(dx, dy, dz)`` offsets within Euclidean ``radius`` (excluding the origin).

    The spatial stencil: the set of grid displacements a source may sit at relative to a target,
    ``0 < dx^2+dy^2+dz^2 <= radius^2``. Returns an ``(n_off, 3)`` int64 array.
    """
    R = int(math.floor(radius))
    ax = np.arange(-R, R + 1, dtype=np.int64)
    dx, dy, dz = np.meshgrid(ax, ax, ax, indexing="ij")
    off = np.stack([dx.ravel(), dy.ravel(), dz.ravel()], axis=1)
    d2 = (off ** 2).sum(axis=1)
    keep = (d2 > 0) & (d2 <= radius * radius)
    return off[keep]


def _radius_for_indegree(in_degree, headroom=2.0):
    """Smallest radius whose ball holds ``>= headroom * in_degree`` neurons (room to sample K)."""
    R = 1.0
    while _ball_offsets(R).shape[0] < headroom * in_degree:
        R += 1.0
    return R


def _draw_sources_radius(rng, partition_rank, num_partitions, somas_per_rank,
                         in_degree, connection_radius, allow_autapses,
                         target_chunk=2048):
    """Draw an ``(somas_per_rank, in_degree)`` array of presynaptic ids within a spatial radius.

    The **3D spatial-radius stencil** (``topology="torus3d"``). Neurons live on a periodic 3D grid:
    the ``num_partitions`` ranks tile the grid as an ``a x b x c`` torus of tiles, and each rank's
    contiguous id block ``[r*npp, (r+1)*npp)`` *is* a ``gx x gy x gz`` spatial sub-block (so
    post-owns holds and "spatially near == same/adjacent rank" for free). Each target draws its
    ``in_degree`` sources uniformly **without replacement** from the neurons within Euclidean
    ``connection_radius`` of it on the torus.

    Because ``connection_radius`` is smaller than a tile, an *interior* neuron draws entirely from
    its own tile (fully local, no MPI); only neurons within ``connection_radius`` of a tile face
    reach a neighbor tile. So the cross-rank traffic is a bounded **surface halo** -- volume-local
    compute + surface communication, constant in ``num_partitions`` -> weak scaling. The realized
    remote peer set is a subset of the 26 Moore neighbors, ramping to exactly 26 once the tiles are
    large relative to the radius and every grid dimension is >= 3.
    """
    a, b, c = _grid_factorization_3d(num_partitions)      # rank (tile) grid
    gx, gy, gz = _factor_near_cube(somas_per_rank)         # intra-tile neuron grid
    A, B, C = a * gx, b * gy, c * gz                       # global neuron grid
    npp = somas_per_rank

    if connection_radius >= min(A, B, C):
        raise ValueError(
            f"connection_radius={connection_radius} must be < the smallest global grid dimension "
            f"min({A},{B},{C}) so the torus does not wrap a source back onto its own target."
        )

    # Global positions of this rank's npp targets.
    local_ids = np.arange(partition_rank * npp, (partition_rank + 1) * npp, dtype=np.int64)
    tX, tY, tZ = _soma_positions(local_ids, num_partitions, npp)

    offsets = _ball_offsets(connection_radius)
    n_off = offsets.shape[0]
    if n_off < in_degree:
        raise ValueError(
            f"connection_radius={connection_radius} yields only {n_off} candidate neurons in the "
            f"ball, fewer than in_degree={in_degree}. Increase connection_radius (or lower "
            "in_degree / raise somas_per_rank)."
        )
    ox, oy, oz = offsets[:, 0], offsets[:, 1], offsets[:, 2]

    pre = np.empty((npp, in_degree), dtype=np.int64)
    for start in range(0, npp, target_chunk):
        end = min(start + target_chunk, npp)
        rows = np.arange(end - start)[:, None]
        # Candidate source positions on the periodic grid, then position -> global id.
        cX = (tX[start:end, None] + ox[None, :]) % A
        cY = (tY[start:end, None] + oy[None, :]) % B
        cZ = (tZ[start:end, None] + oz[None, :]) % C
        cand_id = _positions_to_soma(cX, cY, cZ, num_partitions, npp)   # (chunk, n_off)
        # Sample in_degree distinct offsets per target (the in_degree smallest random keys).
        keys = rng.random((end - start, n_off))
        sel = np.argpartition(keys, in_degree - 1, axis=1)[:, :in_degree]
        chosen = cand_id[rows, sel]                            # (chunk, in_degree)
        # Forbid autapses (only reachable via a full torus wrap on a degenerate tiny grid).
        if not allow_autapses:
            self_id = local_ids[start:end]
            bad = chosen == self_id[:, None]
            while bad.any():
                bi = np.nonzero(bad)
                chosen[bad] = cand_id[bi[0], rng.integers(0, n_off, size=bi[0].size)]
                bad = chosen == self_id[:, None]
        pre[start:end] = chosen
    return pre


def _is_excitatory_spatial(soma_ids, excitatory_fraction):
    """Fine-grained deterministic E/I identity for the spatial small-world (``spatial_smallworld``).

    ``~excitatory_fraction`` of all neurons are excitatory, chosen by a hash of the soma id so
    that spatially-adjacent neurons are *independently* E or I -- i.e. any small distance ball
    contains both populations (unlike the coarse per-rank block split, which a small kernel could
    land entirely inside). It is a **pure function of the id**, so every rank agrees on a remote
    source's E/I (needed to set the post-owned synapse's sign). Returns a bool array.
    """
    ids = np.asarray(soma_ids, dtype=np.uint64)
    h = (ids * np.uint64(2654435761)) & np.uint64(0xFFFFFFFF)
    return (h.astype(np.float64) / 2.0 ** 32) < excitatory_fraction


def _kernel_weights(dist, width, form):
    """Distance-decay kernel ``f(dist)`` for the spatial small-world's local connectivity.

    ``form`` selects the functional shape (``width`` is ``sigma`` / ``lambda`` / ``alpha``):
      * ``"gaussian"``    -- ``exp(-dist^2 / (2*width^2))`` (cortical-model standard).
      * ``"exponential"`` -- ``exp(-dist / width)`` (Bassett & Bullmore economical small-world).
      * ``"powerlaw"``    -- ``dist^(-width)`` (heavier tail; ``dist >= 1`` in the ball).
    """
    dist = np.asarray(dist, dtype=np.float64)
    if form == "gaussian":
        return np.exp(-(dist ** 2) / (2.0 * width * width))
    if form == "exponential":
        return np.exp(-dist / width)
    if form == "powerlaw":
        return np.power(np.maximum(dist, 1.0), -float(width))
    raise ValueError(f"kernel_form must be 'gaussian', 'exponential', or 'powerlaw' (got {form!r}).")


def _draw_smallworld_edges(rng, partition_rank, num_partitions, somas_per_rank,
                           excitatory_fraction, mean_in_degree, longrange_fraction,
                           kernel_form, kernel_width_exc, kernel_width_inh,
                           truncation_radius, longrange_form, allow_autapses,
                           target_chunk=1024):
    """Draw the spatial economical small-world recurrent edges for one rank's neurons.

    For each local target: a **local tier** where every neuron within ``truncation_radius`` connects
    with probability ``amplitude * f(distance)`` (distance-decay kernel, separate widths for E and I
    sources -- inhibition is more local), plus a **long-range tier** of ``Poisson(beta * mean)``
    shortcuts to uniformly-random neurons anywhere in the population (the Watts-Strogatz rewiring
    that gives short path length). In-degree is **variable** (realistic degree heterogeneity / hubs),
    with mean ``mean_in_degree``. Returns ``(pre_concat, counts)`` where ``pre_concat`` is the flat
    source-id array grouped in local-target order and ``counts[i]`` is target ``i``'s in-degree.
    """
    P, npp = num_partitions, somas_per_rank
    total = npp * P
    a, b, c = _grid_factorization_3d(P)
    gx, gy, gz = _factor_near_cube(npp)
    A, B, C = a * gx, b * gy, c * gz
    if truncation_radius >= min(A, B, C):
        raise ValueError(
            f"truncation_radius={truncation_radius} must be < the smallest grid dimension "
            f"min({A},{B},{C})."
        )
    local_ids = np.arange(partition_rank * npp, (partition_rank + 1) * npp, dtype=np.int64)
    tX, tY, tZ = _soma_positions(local_ids, P, npp)

    # Local distance tier: per-offset connection probability, split by source E/I width.
    offsets = _ball_offsets(truncation_radius)
    dist = np.sqrt((offsets ** 2).sum(axis=1))
    kE = _kernel_weights(dist, kernel_width_exc, kernel_form)
    kI = _kernel_weights(dist, kernel_width_inh, kernel_form)
    S_E, S_I = kE.sum(), kI.sum()
    local_mean = (1.0 - longrange_fraction) * mean_in_degree
    # amplitude so E-sources contribute exc_fraction*local_mean edges, I-sources the rest
    A_E = min(1.0, local_mean / S_E) if S_E > 0 else 0.0
    A_I = min(1.0, local_mean / S_I) if S_I > 0 else 0.0
    fE, fI = A_E * kE, A_I * kI
    ox, oy, oz = offsets[:, 0], offsets[:, 1], offsets[:, 2]
    long_mean = longrange_fraction * mean_in_degree

    # Draw both tiers as flat (post, pre) edge arrays (no per-target Python loop), then group by
    # post so the caller can assign per-post synapse-id offsets.
    post_parts, pre_parts = [], []
    for start in range(0, npp, target_chunk):
        end = min(start + target_chunk, npp)
        tgt = local_ids[start:end]
        # --- local distance tier: Bernoulli over the ball, vectorized ---
        cX = (tX[start:end, None] + ox[None, :]) % A
        cY = (tY[start:end, None] + oy[None, :]) % B
        cZ = (tZ[start:end, None] + oz[None, :]) % C
        cand = _positions_to_soma(cX, cY, cZ, P, npp)                 # (m, n_off)
        prob = np.where(_is_excitatory_spatial(cand, excitatory_fraction),
                        fE[None, :], fI[None, :])
        connect = rng.random(cand.shape) < prob
        if not allow_autapses:
            connect &= cand != tgt[:, None]
        ti, oi = np.nonzero(connect)                                  # target-row / offset idx
        post_parts.append(tgt[ti])
        pre_parts.append(cand[ti, oi])
        # --- long-range tier: Poisson(beta*mean) uniform-global shortcuts per target ---
        if long_mean > 0:
            n_long = rng.poisson(long_mean, size=end - start)
            long_post = np.repeat(tgt, n_long)
            long_pre = rng.integers(0, total, size=int(n_long.sum()), dtype=np.int64)
            if not allow_autapses:
                keep = long_pre != long_post
                long_post, long_pre = long_post[keep], long_pre[keep]
            post_parts.append(long_post)
            pre_parts.append(long_pre)

    post_all = np.concatenate(post_parts) if post_parts else np.empty(0, dtype=np.int64)
    pre_all = np.concatenate(pre_parts) if pre_parts else np.empty(0, dtype=np.int64)
    # Group edges by post (stable sort keeps a target's local+long edges together).
    order = np.argsort(post_all, kind="stable")
    pre_sorted = pre_all[order]
    counts = np.bincount(post_all - partition_rank * npp, minlength=npp).astype(np.int64)
    return pre_sorted, counts


def _rank_ei_blocks(ranks, somas_per_rank, exc_per_rank, want_excitatory):
    """E- or I-portion id ranges for the given ranks (E/I interleaved *within* each rank).

    Every rank ``q`` owns ``somas_per_rank`` somas laid out as ``exc_per_rank`` excitatory
    ids ``[q*npp, q*npp+exc_per_rank)`` followed by the rest inhibitory
    ``[q*npp+exc_per_rank, (q+1)*npp)``. So *every* rank holds both populations -- which is
    what lets an arbitrary set of remote ranks supply both E and I sources (a global,
    block-contiguous E/I split would concentrate inhibitory somas in the last ranks and
    starve a random peer set of inhibition). Returns a list of ``(lo, hi)`` tuples.
    """
    out = []
    for q in ranks:
        base = q * somas_per_rank
        if want_excitatory:
            lo, hi = base, base + exc_per_rank
        else:
            lo, hi = base + exc_per_rank, base + somas_per_rank
        if hi > lo:
            out.append((lo, hi))
    return out


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
    # --- weak-scaling wiring: how each soma's recurrent source ranks are chosen ---
    topology: str | None = None,
    remote_rank_fanout: int | None = None,
    connection_radius: float | None = None,
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
    # --- on-disk encoding ---
    output_format: str = "records",
) -> dict:
    """Build one partition's slice of a Brunel balanced random network.

    Returns the post-owns partition contract consumed by
    ``NeuromorphicModel.load_post_owned`` / ``create_from_lists``::

        {"somas": [...], "synapses": [...], "remote_ranks": {id: rank}}

    ``output_format`` selects the ENCODING of that same post-owns network, not
    a different network:

    - ``"records"`` (default): the row-oriented list-of-dicts above. Flexible
      (arbitrary per-entry overrides) and self-documenting; used by SGNN, the
      examples, and small single-process builds. Nothing about this format
      changes when the parameter is omitted.
    - ``"columns"``: a columnar array dict for very large partitions, where the
      row-oriented staging is the host-RAM ceiling. Every per-synapse field is
      one typed array and the uniform categoricals (breed/config/learning rule)
      are stored once; ``NeuromorphicModel.load_post_owned`` reads it via a
      columnar build that never materializes a per-synapse Python object. The
      schema is::

        {"schema": "columnar_post_owned_v1",
         "soma_ids": int64[N], "soma_breed": str, "soma_config": str,
         "synapse_ids": int64[M], "pre": int64[M], "post": int64[M],
         "synapse_breed": str, "synapse_config": str,
         "learning_rule": str ('' == None), "learning_rule_config": str,
         "syn_hp_keys": str[k], "syn_hp_vals": float64[M, k],  # per-synapse hp overrides
         "remote_ids": int64[R], "remote_rank_of": int64[R]}

      ``pre == -1`` marks an external-input synapse (same sentinel as records).

    For a single-partition network (``num_partitions == 1``) there are no remote
    somas, so ``remote_ranks`` is **omitted** and the return is just
    ``{"somas": [...], "synapses": [...]}`` (the loader defaults a missing
    ``remote_ranks`` to ``{}``).

    Sizing
        ``somas_per_rank`` is the per-partition soma count (equal for every
        partition); the global ``total_somas = somas_per_rank * num_partitions`` is
        derived. This partition owns the contiguous soma-id block
        ``[partition_rank * somas_per_rank, (partition_rank + 1) * somas_per_rank)``.

    Topology (two-pool, E/I interleaved per rank -- ``topology != "torus3d"``)
        E/I identity is assigned **within each rank**: the first
        ``round(excitatory_fraction * somas_per_rank)`` somas of every rank's block are
        excitatory, the rest inhibitory (so a soma ``t`` is excitatory iff
        ``(t % somas_per_rank) < exc_per_rank``). Every rank therefore owns both
        populations -- which is what lets ``remote_rank_fanout`` restrict sources to an
        arbitrary set of ranks and still find both E and I sources (a global
        block-contiguous split would concentrate inhibition in the last ranks). Each
        local soma receives **exactly** ``excitatory_in_degree`` recurrent synapses drawn
        uniformly from the E-pool (of the allowed ranks) and **exactly**
        ``inhibitory_in_degree`` from the I-pool. In-degree is fixed; out-degree is
        emergent. Multapses follow ``allow_multapses`` (NEST default: allowed); autapses
        (``pre == post``) are resampled away by default because SAGESim's positional
        ``[pre, post]`` neighbor slot dedups a self-loop.

        ``topology="torus3d"`` does **not** use this two-pool draw -- see its bullet under
        the ``topology`` parameter. It keeps the same per-rank E/I identity rule (and hence
        the same weights) but draws a single pool of ``C_E + C_I`` sources from the spatial
        ball, so a soma's realized E/I mix depends on where it sits: because the E-portion of
        a rank's id block maps to a contiguous *slab* of the tile, a soma deep in the E-slab
        can draw almost no inhibitory sources at all. The 4:1 ratio holds over the population,
        not per soma. This is acceptable for a communication benchmark (the ghost exchange is
        activity-independent) but is not a balanced-network guarantee.

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
        soma (``None`` -> derive from ``connection_probability``). **``topology="torus3d"``
        consumes only the sum** ``excitatory_in_degree + inhibitory_in_degree``; it does not
        honour the split.
    :param inhibitory_in_degree: Exactly this many inhibitory recurrent inputs per
        soma (``None`` -> derive from ``connection_probability``). Same ``"torus3d"``
        caveat as ``excitatory_in_degree``.
    :param connection_probability: Connection density; derives an in-degree left
        ``None`` (``in_degree = round(p * pool_size)``).
    :param topology: How each soma's recurrent **source ranks** are chosen -- the wiring
        that decides the per-rank cross-rank peer count (BRUNEL_SCALING D4). One of:

        * ``"global"`` -- sources drawn from the **whole** population (faithful global-uniform
          Brunel); a rank's distinct remote peer count grows toward ``num_partitions - 1`` as
          the machine grows, tilting our point-to-point ghost exchange. The contrast baseline.
        * ``"bounded"`` -- own rank + exactly ``remote_rank_fanout`` **randomly chosen** remote
          ranks (peer count ``== R``, but *incoming* only -- send-to fanout is emergent).
        * ``"torus2d"`` -- own rank + its **8 Moore-neighbor tiles** on a 2D torus, the ranks
          forming an ``a x b`` grid (``a*b == num_partitions``, near-square, both >= 3). Sources
          are drawn uniformly over these 9 tiles, so the per-rank peer count is a hard constant
          ``== 8`` independent of ``num_partitions`` -- the geometric-partition weak-scaling
          invariant (spatial locality; the halo reaches only adjacent tiles). Requires
          ``num_partitions`` to factor into an ``a, b >= 3`` grid (>= 9, non-prime).
        * ``"torus3d"`` -- the **3D spatial-radius stencil** (point-to-point weak-scaling
          convention). The ranks tile a periodic ``a x b x c`` torus (near-cube, any
          ``num_partitions >= 1``) and every neuron carries a 3D position; each target draws its
          ``K`` sources uniformly within Euclidean ``connection_radius`` of itself on the grid.
          Interior neurons draw entirely from their own tile (local); only neurons within the
          radius of a tile face reach a neighbor -> a bounded **surface halo** (volume-local
          compute + surface communication). The realized remote peer set is a subset of the 26
          Moore neighbors that ramps to 26 as the grid grows and plateaus there -- weak scaling.
          Unlike the two-pool topologies, E/I identity is read off each drawn source's id for the
          weight (no exact ``C_E``/``C_I`` split), which is fine for a communication benchmark.

        ``None`` (default) infers ``"bounded"`` when ``remote_rank_fanout`` is set, else
        ``"global"`` -- backward compatible with callers that only pass ``remote_rank_fanout``.
    :param remote_rank_fanout: The ``R`` for ``topology="bounded"``: each partition draws its
        sources only from **its own rank plus exactly ``R`` randomly chosen remote ranks**
        (the same ``R`` for every soma it owns). The draw is validated to *realize* all ``R``
        peers (each chosen rank receives >= 1 edge) and re-drawn/raised otherwise. Requires
        ``R < num_partitions``. Ignored unless ``topology`` resolves to ``"bounded"``.
    :param connection_radius: For ``topology="torus3d"``: the Euclidean spatial radius (in
        neuron-grid units) of each soma's presynaptic draw. ``None`` (default) auto-selects the
        smallest radius whose ball holds ``>= 2 * K`` neurons (sampling headroom). Must be smaller
        than the smallest global grid dimension. Ignored for other topologies.
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

    # E/I identity is interleaved *within* each rank: the first ``exc_per_rank`` somas of
    # every rank's block are excitatory, the rest inhibitory. So every rank owns both
    # populations (essential for bounded remote_rank_fanout -- see _rank_ei_blocks). A
    # soma id ``t`` is excitatory iff ``(t % npp) < exc_per_rank``.
    exc_per_rank = round(excitatory_fraction * npp)
    inh_per_rank = npp - exc_per_rank
    excitatory_somas = exc_per_rank * P     # global E count (for the p-derivation below)
    inhibitory_somas = inh_per_rank * P
    if exc_per_rank <= 0 or inh_per_rank <= 0:
        raise ValueError(
            f"E/I split degenerate per rank: somas_per_rank={npp}, E/rank={exc_per_rank}, "
            f"I/rank={inh_per_rank}. Increase somas_per_rank or adjust excitatory_fraction."
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

    # Resolve the wiring topology (see the ``topology`` parameter). Default: infer
    # "bounded" when remote_rank_fanout is set, else "global" -- backward compatible with
    # callers that only pass remote_rank_fanout.
    if topology is None:
        topology = "bounded" if remote_rank_fanout is not None else "global"
    if topology not in ("global", "bounded", "torus2d", "torus3d"):
        raise ValueError(
            f"topology must be 'global', 'bounded', 'torus2d', or 'torus3d' (got {topology!r})."
        )
    if topology == "bounded":
        if remote_rank_fanout is None:
            raise ValueError("topology='bounded' requires remote_rank_fanout to be set.")
        if remote_rank_fanout < 0:
            raise ValueError(
                f"remote_rank_fanout must be non-negative (got {remote_rank_fanout})."
            )
        if remote_rank_fanout >= P:
            raise ValueError(
                f"remote_rank_fanout {remote_rank_fanout} must be < num_partitions {P} "
                "(a rank cannot pick itself or more remote ranks than exist)."
            )

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

    local_ids = np.arange(lo, hi, dtype=np.int64)

    # --- Somas owned by this partition ---
    somas = [
        {"id": int(t), "breed": soma_breed, "config": soma_config, "overrides": {}}
        for t in local_ids
    ]

    # --- Recurrent presynaptic draw: pre[t, j] = source of target t's j-th in-edge (npp, K) ---
    if topology == "torus3d":
        # 3D spatial-radius stencil (weak-scaling / point-to-point convention). Neurons sit on a
        # periodic 3D grid; each target draws its K sources within ``connection_radius`` of its own
        # position, so most sources are local (own tile) and only a bounded surface halo crosses to
        # neighbor tiles. Peer count is emergent from geometry (subset of the 26 Moore neighbors,
        # ramps to 26) -- no enforce loop, no E/I two-pool split (E/I identity is read off each
        # drawn source's id for the weight, exactly as below).
        radius = (connection_radius if connection_radius is not None
                  else _radius_for_indegree(K))
        rng = np.random.default_rng(np.random.SeedSequence([seed, r, 0]))
        pre = _draw_sources_radius(rng, r, P, npp, K, radius, allow_autapses)   # (npp, K)
    else:
        # --- Source ranks: whole population / {r} u S_r random / {r} u 8 torus neighbors ---
        # E/I identity is per-rank interleaved, so we build the E-pool / I-pool from the E- and
        # I-blocks of the allowed ranks (every rank contributes both). ``S_r`` is the set of remote
        # peer ranks the draw must realize exactly (None for the un-enforced global case).
        if topology == "global":
            S_r = None
            source_ranks = list(range(P))             # global-uniform: whole population
        elif topology == "bounded":
            S_r = _select_remote_ranks(seed, r, P, remote_rank_fanout)
            source_ranks = [r] + S_r                  # own rank + exactly R random remote ranks
        else:  # torus2d
            a, b = _grid_factorization(P)
            S_r = _select_neighbor_tiles(r, a, b)
            source_ranks = [r] + S_r                  # own rank + its 8 Moore-neighbor tiles
        e_ranges = _rank_ei_blocks(source_ranks, npp, exc_per_rank, want_excitatory=True)
        i_ranges = _rank_ei_blocks(source_ranks, npp, exc_per_rank, want_excitatory=False)

        # --- Recurrent synapses: exactly C_E from E-pool + C_I from I-pool per target ---
        # For a bounded/torus2d peer set, re-draw (bumped seed) if some chosen peer rank got zero
        # edges, so the realized peer set is *exactly* S_r (== R for bounded, == 8 for torus2d),
        # not merely a subset -- a hard-constant peer count is what makes weak scaling hold.
        enforce_peers = topology in ("bounded", "torus2d")
        max_attempts = 100 if enforce_peers else 1
        for attempt in range(max_attempts):
            rng = np.random.default_rng(np.random.SeedSequence([seed, r, attempt]))
            pre_E = _draw_sources(rng, npp, C_E, e_ranges,
                                  local_ids, allow_multapses, allow_autapses)   # (npp, C_E)
            pre_I = _draw_sources(rng, npp, C_I, i_ranges,
                                  local_ids, allow_multapses, allow_autapses)   # (npp, C_I)
            pre = np.concatenate([pre_E, pre_I], axis=1)                         # (npp, K)
            if not enforce_peers:
                break
            realized = set((pre.ravel() // npp).tolist()) - {r}
            if realized == set(S_r):
                break
        else:
            raise RuntimeError(
                f"rank {r}: could not realize all {len(S_r)} {topology} peers in "
                f"{max_attempts} attempts (chosen {S_r}, realized {sorted(realized)}). "
                "Increase somas_per_rank/in-degree (or, for bounded, lower remote_rank_fanout)."
            )

    post = np.repeat(local_ids, K)                                             # (npp*K,)
    j_idx = np.tile(np.arange(K, dtype=np.int64), npp)                         # (npp*K,)
    pre_flat = pre.ravel()                                                     # (npp*K,)
    syn_ids = SYN_BASE + post * K + j_idx                                      # closed-form
    # Source is excitatory iff it sits in the E-portion of its own rank's block, i.e.
    # (id % npp) < exc_per_rank (per-rank interleaved E/I identity).
    is_exc = (pre_flat % npp) < exc_per_rank
    weights = np.where(is_exc, eff_exc_w, inh_w)

    # --- External input synapses: external_synapses_per_soma per local soma (pre=-1).
    # k-major, soma-minor order (matches the k-outer / soma-inner record loop below),
    # so recurrent-then-external is one consistent synapse order shared by both formats.
    ext_k = np.arange(ext_per, dtype=np.int64)
    ext_ids_all = (EXT_BASE + local_ids[None, :] * ext_per + ext_k[:, None]).ravel()
    ext_post_all = np.tile(local_ids, ext_per)
    ext_pre_all = np.full(ext_ids_all.shape, -1, dtype=np.int64)
    ext_weight_all = np.full(ext_ids_all.shape, ext_w, dtype=np.float64)

    if output_format == "columns":
        # Columnar encoding: one typed array per per-synapse field, no per-object
        # staging. The resolved weight is stored for EVERY synapse (excitatory
        # "inherit config" == the config default == eff_exc_w, so writing it as an
        # explicit override yields the identical hyperparameter value the record
        # path computes). synaptic_delay is uniform, included only when set.
        synapse_ids = np.concatenate([syn_ids, ext_ids_all]).astype(np.int64)
        pre_col = np.concatenate([pre_flat, ext_pre_all]).astype(np.int64)
        post_col = np.concatenate([post, ext_post_all]).astype(np.int64)
        weight_col = np.concatenate([weights, ext_weight_all]).astype(np.float64)

        hp_keys = ["weight"]
        hp_cols = [weight_col]
        if synaptic_delay_ms is not None:
            hp_keys.append("synaptic_delay")
            hp_cols.append(np.full(synapse_ids.shape, float(synaptic_delay_ms),
                                   dtype=np.float64))
        syn_hp_vals = np.stack(hp_cols, axis=1) if hp_cols else np.zeros((len(synapse_ids), 0))

        partition = {
            "schema": "columnar_post_owned_v1",
            "soma_ids": local_ids.astype(np.int64),
            "soma_breed": soma_breed,
            "soma_config": soma_config,
            "synapse_ids": synapse_ids,
            "pre": pre_col,
            "post": post_col,
            "synapse_breed": synapse_breed,
            "synapse_config": synapse_config,
            "learning_rule": "",  # '' == None (npz has no None)
            "learning_rule_config": "default",
            # plain unicode array (NOT dtype=object) so np.load(mmap_mode='r') can
            # memory-map the file — object arrays would force an eager pickle load.
            "syn_hp_keys": np.array(hp_keys),
            "syn_hp_vals": syn_hp_vals,
        }
        if P > 1:
            remote_pre = np.unique(pre_flat[(pre_flat < lo) | (pre_flat >= hi)])
            partition["remote_ids"] = remote_pre.astype(np.int64)
            partition["remote_rank_of"] = (remote_pre // npp).astype(np.int64)
        else:
            partition["remote_ids"] = np.empty(0, dtype=np.int64)
            partition["remote_rank_of"] = np.empty(0, dtype=np.int64)
        return partition

    if output_format != "records":
        raise ValueError(
            f"output_format must be 'records' or 'columns', got {output_format!r}")

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

    # --- External input synapses (record encoding): same k-outer / soma-inner order ---
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


def spatial_smallworld_partition(
    *,
    somas_per_rank: int = 12500,
    num_partitions: int = 1,
    partition_rank: int = 0,
    excitatory_fraction: float = 0.8,
    # --- distance-dependent connectivity (variable in-degree; the paper's P(i,j) ~ f(d)) ---
    mean_in_degree: int = 1000,
    kernel_form: str = "gaussian",
    kernel_width_exc: float = 4.0,
    kernel_width_inh: float = 2.0,
    truncation_radius: float | None = None,
    # --- Watts-Strogatz long-range shortcuts (the small-world path-length term) ---
    longrange_fraction: float = 0.05,
    longrange_form: str = "uniform",
    max_in_degree: int | None = None,
    # --- weights / dynamics (as brunel_partition) ---
    excitatory_weight: float | None = None,
    inhibitory_weight_ratio: float = 5.0,
    synaptic_delay_ms: float | None = None,
    external_synapses_per_soma: int = 1,
    external_weight: float | None = None,
    allow_autapses: bool = False,
    soma_breed: str = "lif_soma",
    soma_config: str = "config_0",
    synapse_breed: str = "single_exp_synapse",
    synapse_config: str = "config_0",
    seed: int = 42,
) -> dict:
    """Build one partition of a **spatial economical small-world** brain network (Deliverable 2).

    A bio-realistic network for **stress-testing the software under realistic connectivity** --
    *not* a weak-scaling benchmark (its long-range shortcuts make the peer count grow with the
    machine, by design). Follows Bassett & Bullmore 2017 (*Small-World Brain Networks Revisited*):
    neurons are embedded in a 3D torus (same coordinate bijection as ``topology="torus3d"``), and

    * **local connectivity is distance-dependent** -- each pair connects with probability
      ``amplitude * f(distance)`` (``kernel_form``: gaussian / exponential / power-law), with a
      **tighter kernel for inhibition** (``kernel_width_inh < kernel_width_exc``). This yields
      **variable in-degree** (degree heterogeneity / hubs), mean ``mean_in_degree``;
    * **a fraction ``longrange_fraction`` of edges are long-range shortcuts** (Watts-Strogatz
      rewiring) to random neurons anywhere -- the short-path-length term that makes it *small-world*.

    Returns the same ``{"somas", "synapses", "remote_ranks"}`` post-owns contract as
    :func:`brunel_partition` (schema unchanged; coordinates are generation-time only). Synapse ids
    use a ``max_in_degree`` stride (the variable-in-degree analogue of the closed-form fixed-K id).

    :param mean_in_degree: Target mean recurrent in-degree (per neuron; the draw is stochastic).
    :param kernel_form: Distance-decay shape: ``"gaussian"`` | ``"exponential"`` | ``"powerlaw"``.
    :param kernel_width_exc: Excitatory kernel width (sigma / lambda / alpha, per ``kernel_form``).
    :param kernel_width_inh: Inhibitory kernel width (typically < ``kernel_width_exc``).
    :param truncation_radius: Ball radius the local kernel is evaluated over; ``None`` -> ``3 *
        max(kernel_width_exc, kernel_width_inh)`` (captures ~all of a Gaussian's mass).
    :param longrange_fraction: ``beta`` -- fraction of the mean in-degree drawn as random long-range
        shortcuts (0 = pure distance-local lattice; larger = more small-world / more remote traffic).
    :param longrange_form: ``"uniform"`` (distance-independent; implemented). ``"powerlaw"`` is
        reserved (raises) -- a distance-biased long-range tail is a future extension.
    :param max_in_degree: Synapse-id stride / hard in-degree cap; ``None`` -> ``2*mean+100``.
    :return: The partition dict.
    """
    if not (0.0 <= longrange_fraction <= 1.0):
        raise ValueError(f"longrange_fraction must be in [0, 1] (got {longrange_fraction}).")
    if longrange_form != "uniform":
        raise ValueError(
            f"longrange_form={longrange_form!r} not implemented; only 'uniform' is available "
            "(a 'powerlaw' distance-biased long-range tail is a planned extension)."
        )
    P, npp, r = num_partitions, somas_per_rank, partition_rank
    total_somas = npp * P
    exc_per_rank = round(excitatory_fraction * npp)
    trunc = (truncation_radius if truncation_radius is not None
             else 3.0 * max(kernel_width_exc, kernel_width_inh))
    Kmax = max_in_degree if max_in_degree is not None else int(2 * mean_in_degree + 100)

    eff_exc_w = (float(excitatory_weight) if excitatory_weight is not None
                 else _config_synapse_weight(synapse_breed, synapse_config))
    inh_w = -float(inhibitory_weight_ratio) * eff_exc_w
    ext_w = float(external_weight) if external_weight is not None else eff_exc_w

    lo, hi = r * npp, (r + 1) * npp
    local_ids = np.arange(lo, hi, dtype=np.int64)
    SYN_BASE = total_somas
    EXT_BASE = total_somas + total_somas * Kmax

    somas = [{"id": int(t), "breed": soma_breed, "config": soma_config, "overrides": {}}
             for t in local_ids]

    rng = np.random.default_rng(np.random.SeedSequence([seed, r, 0]))
    pre_flat, counts = _draw_smallworld_edges(
        rng, r, P, npp, excitatory_fraction, mean_in_degree, longrange_fraction,
        kernel_form, kernel_width_exc, kernel_width_inh, trunc, longrange_form, allow_autapses)
    if counts.size and counts.max() > Kmax:
        raise ValueError(
            f"rank {r}: a neuron drew {int(counts.max())} inputs > max_in_degree {Kmax}; "
            "raise max_in_degree (it is the synapse-id stride and the hard in-degree cap)."
        )

    # Variable-in-degree synapse ids: source's j-th in-edge to post t -> SYN_BASE + t*Kmax + j.
    post = np.repeat(local_ids, counts)
    within = np.arange(int(counts.sum()), dtype=np.int64) \
        - np.repeat(np.cumsum(counts) - counts, counts)
    syn_ids = SYN_BASE + post * Kmax + within
    is_exc = _is_excitatory_spatial(pre_flat, excitatory_fraction)
    weights = np.where(is_exc, eff_exc_w, inh_w)

    synapses = []
    for sid, p, q, w, exc in zip(syn_ids, pre_flat, post, weights, is_exc):
        hyper = {}
        if (not exc) or (excitatory_weight is not None):
            hyper["weight"] = float(w)
        if synaptic_delay_ms is not None:
            hyper["synaptic_delay"] = float(synaptic_delay_ms)
        synapses.append({
            "id": int(sid), "pre": int(p), "post": int(q),
            "breed": synapse_breed, "config": synapse_config,
            "learning_rule": None, "learning_rule_config": "default",
            "overrides": {"hyperparameters": hyper} if hyper else {},
        })

    for k in range(external_synapses_per_soma):
        ext_ids = EXT_BASE + local_ids * external_synapses_per_soma + k
        for sid, t in zip(ext_ids, local_ids):
            hyper = {}
            if external_weight is not None or excitatory_weight is not None:
                hyper["weight"] = float(ext_w)
            if synaptic_delay_ms is not None:
                hyper["synaptic_delay"] = float(synaptic_delay_ms)
            synapses.append({
                "id": int(sid), "pre": -1, "post": int(t),
                "breed": synapse_breed, "config": synapse_config,
                "learning_rule": None, "learning_rule_config": "default",
                "overrides": {"hyperparameters": hyper} if hyper else {},
            })

    partition = {"somas": somas, "synapses": synapses}
    if P > 1:
        remote_pre = np.unique(pre_flat[(pre_flat < lo) | (pre_flat >= hi)])
        partition["remote_ranks"] = {int(p): int(p) // npp for p in remote_pre}
    return partition


def save_spatial_smallworld_partition(output_dir, *, filename=None, **kwargs):
    """Generate one :func:`spatial_smallworld_partition` and pickle it to ``partition_{rank}.pkl``.

    The Deliverable-2 analogue of :func:`save_brunel_partition`: writes the post-owns partition dict
    to disk in the schema ``NeuromorphicModel.load_post_owned`` reads. Each MPI rank calls this for
    its own ``partition_rank`` independently.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    partition_rank = kwargs.get("partition_rank", 0)
    partition = spatial_smallworld_partition(**kwargs)
    name = filename if filename is not None else f"partition_{partition_rank}.pkl"
    out_file = out_dir / name
    with open(out_file, "wb") as f:
        pickle.dump(partition, f)
    return str(out_file)


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
    """Generate one partition and write it to ``output_dir/partition_{rank}.{ext}``.

    Convenience wrapper over :func:`brunel_partition` that writes the partition to
    disk in the schema ``NeuromorphicModel.load_post_owned`` reads. Each MPI rank calls
    this for its own ``partition_rank`` independently -- no communication is needed
    during generation.

    The on-disk format follows ``output_format`` (a :func:`brunel_partition` kwarg):

    - ``"records"`` (default) -> pickle at ``partition_{rank}.pkl`` (row-oriented).
    - ``"columns"``  -> ``np.savez`` at ``partition_{rank}.npz`` (columnar arrays;
      ``np.load(mmap_mode='r')``-friendly, so the loader never eagerly copies it
      into host RAM).

    :param output_dir: Directory to write into (created if missing).
    :param filename: Output filename; defaults to ``partition_{rank}.{pkl|npz}``.
    :param kwargs: Every keyword of :func:`brunel_partition`.
    :return: Path to the saved partition file (str).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    partition_rank = kwargs.get("partition_rank", 0)
    partition = brunel_partition(**kwargs)

    if kwargs.get("output_format", "records") == "columns":
        name = filename if filename is not None else f"partition_{partition_rank}.npz"
        out_file = out_dir / name
        # object-dtype arrays (syn_hp_keys) require allow_pickle at save+load.
        np.savez(out_file, **partition)
        return str(out_file)

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
