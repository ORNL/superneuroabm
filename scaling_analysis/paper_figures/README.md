# Scaling study — figures, methodology and reproduction

Complete source material for the scaling section of the paper: what was run, why the network is
shaped the way it is, and how to reproduce it. Two figures, in `figs/` as 600-dpi PNG and
Type-42-embedded PDF pairs.

All numbers here come from `outputs/weak_3d_final.csv` and `outputs/strong_3d_final.csv`.
`docs/BRUNEL_SCALING.md` is the longer design-discussion document; where the two disagree the
CSVs win (its §5.5c still records `ticks=30`, superseded by the `ticks=100` re-run).

| | script | claim |
|---|---|---|
| **F1** | `weak_efficiency.py` | Per-step parallel efficiency is flat across the constant-peer regime, w = 64 → 2048 (a 32× span), at three in-degrees. |
| **F2** | `strong_speedup.py` | Wall-clock time to solution on a fixed 204,800-neuron problem improves 13.4× on 64× the GPUs, against the ideal diagonal. |

```sh
python make_all.py            # both figures -> figs/
python make_all.py --print    # plus every plotted value as text
```

`make_all.py` exits non-zero if any expected grid point is missing, so a figure drawn from a
truncated sweep cannot be mistaken for a finished one.

**Symbols.** **K** is the recurrent in-degree — incoming synapses per neuron — fixed per curve.
**w** is the worker count; one worker is one GPU (one MI250X GCD). **Peers** are the *distinct*
ranks a rank exchanges with per tick.

---

## 1. Network model

The benchmark network is a **Brunel balanced random network** whose *source selection* has been
replaced by a spatial rule. Both halves of that sentence matter, so they are separated below.

### 1.1 Inherited from Brunel

| property | value | note |
|---|---|---|
| soma model | LIF (`lif_soma`) | current-based, single-exponential synapse |
| E/I ratio | **80 / 20** (`excitatory_fraction = 0.8`) | every rank owns both populations |
| excitatory weight `J_E` | 0.02581 | |
| inhibitory weight `J_I` | **−`g`·`J_E` = −0.12905**, `g` = 5.0 | inhibition-dominated; the sign is carried in the weight, the kernel has no sign logic |
| recurrent in-degree | **fixed K per neuron**; `C_E = 0.8K`, `C_I = K − C_E` are supplied but the spatial draw uses only their **sum** | E/I mix is a population average, **not** per soma — see §1.3 |
| external drive | 1 Poisson input synapse per soma (`pre = −1`) at 10 Hz, weight `J_E` | far below the AI threshold rate — see §1.4 |
| autapses | disallowed | |
| seed | 42 | |

Fixed in-degree and the inhibition-dominated 4:1 balance are the two properties that make this a
Brunel network rather than a generic random graph. **Fixed in-degree is preserved exactly; the
4:1 balance is preserved only in population average** (§1.3).

### 1.2 What is replaced, and why

Classic Brunel draws each neuron's `C_E` sources uniformly from **all** excitatory neurons and
`C_I` from **all** inhibitory neurons — global uniform connectivity. We instead place every neuron
on a periodic 3D lattice and draw its K sources uniformly, without replacement, from the neurons
within a Euclidean **connection radius** of its own position (`topology="torus3d"` in
`superneuroabm/brunel.py`).

**The reason is the communication pattern, not biology.** Our ghost exchange is **point-to-point**:
per-tick cost is indexed by how many peers a rank talks to and how much ghost volume crosses, not
by a global collective. Under global uniform wiring every rank draws sources from every other
rank, so a rank's peer count grows toward `w − 1` and its halo grows with the machine. A
weak-scaling curve measured that way reports the *wiring* rather than the machine, and degrades by
construction no matter how good the implementation is.

A spatial radius bounds it. Because the radius is smaller than a rank's tile, an **interior**
neuron draws entirely from its own tile and generates no MPI traffic at all; only neurons within
one radius of a tile face reach a neighbouring tile. Cross-rank traffic is therefore a bounded
**surface halo** against volume-local compute — the same surface-to-volume structure by which
PDE/stencil, molecular-dynamics and lattice-QCD codes weak-scale, and the reason a bounded-peer
stencil is the standard shape for a point-to-point weak-scaling benchmark. Precedent: WOMBAT
(Mendygral et al. 2017, ApJS 228:23) reports off-node communication saturating between 3 and 27
nodes for exactly this geometric reason, with update times "nearly flat for larger node counts"
past that point.

### 1.3 What this network does and does not claim

Stated explicitly, because the substitution has consequences a reviewer will look for:

- **Preserved exactly:** E/I identity and weights, fixed recurrent in-degree K per neuron,
  external Poisson drive, LIF dynamics.
- **Statistical, not exact:** the per-neuron E/I *mix*, and with it the 4:1 balance. `C_E` and
  `C_I` are supplied, but the spatial draw uses only their sum (K = `C_E` + `C_I`,
  `brunel.py:812` → `_draw_sources_radius`); each drawn source's E/I identity is read off its id
  to set the weight. A given neuron therefore receives roughly, not exactly, an 800/200 split.
  Measured over one tile (npp=12500, w=27, K=1000, R=8), the per-neuron excitatory fraction of
  in-degree spans **0.601 – 1.000** (p25 0.668, median 0.780, p75 0.936) against a population
  mean of exactly **0.8000** — i.e. per-neuron `C_E` ranges 601 … 1000 where the nominal value
  is 800, and the most excitatory-slab-interior neurons receive **essentially no inhibition at
  all**. This is a communication benchmark, so that is a disclosure, not a defect — but it means
  the network must not be described as balanced per soma.
- **Spatially segregated E and I.** E/I identity is assigned by id (a soma is excitatory iff
  `id % npp < 0.8·npp`) and id maps bijectively to lattice position, so within every tile the
  excitatory population occupies a contiguous slab — **the first 16 of 20 x-slices**, for both the
  12,500- and 12,800-neuron tiles used here — rather than being interspersed. Local E/I
  composition therefore varies with position inside a tile.

**These figures are a communication and throughput benchmark, not a claim about Brunel network
dynamics.** The bio-realistic variant with randomised spatial E/I, a distance kernel and
distance-dependent E/I is a separate topology (`spatial_smallworld`, `docs/BRUNEL_SCALING.md`
§5.5b) and is not what was measured here.

### 1.4 The drive is sub-threshold, and why that does not matter

The campaign passes only `--firing-rate 10.0`; neither `--external-rate` nor `--external-weight`
is passed, so the single `pre = −1` synapse carries weight `J_E` and fires at 10 Hz. But
`calibrate_ai.py:120-129` defines the intended convention: that one synapse **stands in for the
soma's `C_E` external inputs**, so a driven network needs an *aggregate* weight `≈ C_E·J` at
`eta·nu_thr`. With this preset (θ = `vthr − vrest` = 15 mV, J = 0.1 mV, τ_m = `R·C` = 10 ms,
`C_E` = 800) the drive that ran is ~10³× weaker than that. Mean depolarisation is
`rate·J·τ_m` = 10 × 0.1 mV × 0.01 s = **0.01 mV** against a 15 mV threshold distance, so **the
network does not fire**. (The injector is also per-tick Bernoulli at `p = rate·dt/1000`,
`scaling_diagnostics.py:80` — the usual discrete-time approximation to Poisson.)

This does not affect any reported number, because **the ghost exchange is activity-independent**:
across all 36 weak points, `send_bytes_mean / ghost_somas_mean = 20.00` **exactly** — a fixed
20 B of ghost soma state per ghost per tick, sent whether or not anything spiked. Weak scaling
here is therefore a property of the partition geometry, and these curves would be unchanged by a
calibrated drive. Any text describing this network must say what it measured (structural halo
exchange) rather than implying an asynchronous-irregular regime that was never reached.

## 2. The 3D torus decomposition

The `w` ranks tile a periodic `a × b × c` torus of tiles (`_grid_factorization_3d` →
`_factor_near_cube`: the most cube-like factorisation, relaxed to accept any `w ≥ 1` so the sweep
can start at one GPU). Each rank's contiguous id block *is* a `gx × gy × gz` spatial sub-block, so
`id = rank·npp + intra` is a bijection onto the global neuron grid and a rank's id block is
exactly one spatial tile. Post-owns-synapse therefore holds for free, and "spatially near" implies
"same or adjacent rank".

The connection radius is chosen automatically as the smallest ball holding ≥ 2K neurons — **R = 8
for K = 1000** — and is held fixed across a sweep, so weak scaling grows the machine at constant
communication geometry and strong scaling shrinks the tile against a constant radius.

## 3. Peer count is geometry, not measurement

A rank's Moore stencil always has 26 = 3³−1 directions, but on a periodic torus several can land
on the *same* rank: if the grid is only two ranks wide along an axis, +x and −x wrap to the same
neighbour. So the number of **distinct** peers climbs with the grid and saturates at 26 once every
axis holds at least four ranks.

The peer count is fully determined by the rank grid, the tile shape and the radius — it is not a
property of a particular run. Recomputing it from geometry reproduces the measured `peers_mean`
**exactly** at every strong-scaling point through w = 1024:

| w | nodes | neurons/GPU | rank grid | tile (neurons) | peers | regime |
|---|---|---|---|---|---|---|
| 16 | 2 | 12,800 | 2×2×4 | 20×20×32 | 11 | ramp |
| 32 | 4 | 6,400 | 2×4×4 | 16×20×20 | 17 | ramp |
| 64 | 8 | 3,200 | 4×4×4 | 10×16×20 | 26 | plateau |
| 128 | 16 | 1,600 | 4×4×8 | 10×10×16 | 26 | plateau |
| 256 | 32 | 800 | 4×8×8 | 8×10×10 | 26 | plateau |
| 512 | 64 | 400 | 8×8×8 | 5×8×10 | 44 | breakdown |
| 1024 | 128 | 200 | 8×8×16 | 5×5×8 | 62 | breakdown |
| 2048 | 256 | 100 | 8×16×16 | 4×5×5 | 95.55 | breakdown |

Two consequences worth stating:

- **It is not monotonic in `w`** — it depends on how `w` *factors*, not on how large it is. Only 22
  values of `w` are realizable at all for N = 204,800 (`w` must divide N), and among them w = 25
  factors as 1×5×5, a flat grid whose wraps collapse neighbours, giving **8 peers — fewer than
  w = 16's 11**. The peer count must never be interpolated across worker counts that were not run.
- **Three regimes.** The low-`w` *ramp* is a rank-grid dimension below 4 collapsing the torus
  wraparound. The *plateau* at 26 is the healthy stencil, and is the only regime over which a
  weak-scaling claim is meaningful. The *breakdown* at w ≥ 512 is what strong scaling adds and
  weak scaling cannot show: holding N fixed shrinks the tile against a fixed radius until the tile
  edge falls under it, at which point the bounded-peer premise stops holding.

## 4. Weak-scaling campaign

**12,500 neurons per worker held constant**, K ∈ {1000, 2000, 4000}, w = 1 → 2048 (1 → 512 nodes),
100 ticks. 36/36 points complete. Total problem size at w = 2048 is 25.6 M neurons.

**The baseline is T(64), not T(1).** w = 64 is the first rank grid with ≥ 4 ranks on every axis,
hence the first configuration in which all 26 Moore neighbours are *distinct* peers. Below it a run
measures a cheaper communication pattern rather than a faster code, so normalising there charges
the plateau for the one-time cost of switching communication on.

F1 plots w = 64 → 2048 only. **The pre-saturation ramp is therefore not on the panel, and is
recorded here** — this is the complete measured record of the excluded regime:

| w | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| distinct peers | 0 | 1 | 3 | 7 | 11 | 17 | 26 |
| K=1000 step (ms) | 4.175 | 4.604 | 5.006 | 5.326 | 5.630 | 6.049 | 6.343 |
| K=2000 step (ms) | 7.491 | 7.788 | 8.384 | 8.752 | 9.020 | 9.313 | 9.729 |
| K=4000 step (ms) | 15.338 | 16.460 | 17.072 | 17.400 | 17.926 | 18.135 | 18.522 |

Results across the plateau. Both baselines are quoted because either alone misleads — T(1) invites
the reader to extrapolate the ramp, T(64) hides what communication costs:

| K | sustained step | efficiency vs T(64) | efficiency vs T(1) | step spread, w=64→2048 |
|---|---|---|---|---|
| 1000 | 6.4 ms/tick | 99.1–100.0 % | 65.3–65.8 % | 0.86 % |
| 2000 | 9.7 ms/tick | 100.0–101.3 % | 77.0–78.0 % | 1.30 % |
| 4000 | 18.6 ms/tick | 95.8–100.0 % | 79.4–82.8 % | 4.33 % |

Sustained rate is the plateau median. Efficiency rises with in-degree because added synapses are
volume work while the halo is a surface.

## 5. Strong-scaling campaign

**N = 204,800 neurons fixed**, K = 1000, R = 8, w = 16 → 2048 at a uniform 8 ranks/node, 100
ticks. 8/8 points complete. Per-rank work falls as 1/w; the ideal curve is the linear diagonal.

**The baseline is w = 16, not one GPU.** This N needs ~205 M synapses, which does not fit on a
single GCD, so a 1-GPU run never happened and quoting speedup against it would be fiction. w = 16
(12,800 neurons/GPU) is the smallest per-rank load already proven safe in the weak campaign — and
it is also the cross-check against that campaign's 12,500 neurons/GPU point. Note that w = 16
itself sits in the ramp at 11 peers, so it is not a clean serial reference.

**The plotted metric is wall time** (`total_time`: generation, model creation, GPU setup,
construction and every tick). Amdahl's law is defined on total runtime, and the question a
fixed-size sweep asks is "more processors, how much sooner is the answer?"

| w | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|---|---|---|
| wall time (s) | 257 | 134 | 70.1 | 41.6 | 25.7 | 19.1 | 18.5 | 19.2 |
| speedup | 1.00× | 1.92× | 3.67× | 6.19× | 10.01× | 13.45× | **13.88×** | 13.40× |
| step speedup | 1.00× | 1.16× | 1.22× | 1.30× | **1.44×** | 1.19× | 1.02× | 0.76× |
| ghost exchange, % of step | 27.3 | 42.4 | 51.7 | 52.7 | 56.7 | 65.4 | 71.3 | 80.5 |

**What the 13.4× does not claim.** Construction is ~99.8 % of a 100-tick run, and construction is
what parallelises here. The steady-state *step* peaks at only 1.44× at w = 256 and then reverses
to 0.76× at w = 2048, because per-tick GPU compute is pinned near 0.370 ms at every point across a
128× span of per-rank work — the kernel is launch-bound from the first point, so shrinking the
tile only grows the halo. The turn lands at w = 512, exactly where the peer count leaves 26.
Communication grows even as bytes-per-peer falls 69 kB → 972 B, which makes the exchange
latency-bound rather than bandwidth-bound.

## 6. Timing methodology

**Timing is recorded, not decided.** Every run writes a per-tick CSV to `outputs/ticks/` — one row
per tick with the across-rank mean/max/min of every timer — and applies **no warm-up window at
collection time**. The window is a `--warmup-ticks` flag on the *analysis* scripts, so it can be
revisited without re-running. This matters: an earlier campaign baked a window in at collection,
left warm-up inside every mean, and reported the ghost exchange at 94.5 % of the step where the
steady-state value is ~57 %. That became a re-analysis rather than a re-run.

| column | definition |
|---|---|
| `first_tick_s` | tick 1 — GPU buffer build plus ghost-topology discovery |
| `step_s` | **mean** of ticks 11–100 (`warmup_ticks = 10`, `n_step_ticks = 90`) |
| `step_median_s` | median of the same window, carried as a divergence check |
| `total_time` | whole wall clock; the F2 metric |
| `peers_mean` | mean distinct MPI peers per rank |

`step_s` and `step_median_s` agree to 0.09 % (median across runs) once 10 ticks are dropped —
where a mean and a median of the same samples coincide, the window has cleared the settling tail.
Both are published so any divergence is checkable per point.

**Why weak-scaling efficiency is built on the step and not on an aggregate.** At 99.7–99.9 %
construction, an end-to-end "simulation time" curve is a construction benchmark wearing a
simulation label: it comes out non-monotonic, with no knee at w = 64 and no plateau. The aggregate
*is* reported — F2 plots exactly it and says so — but the weak-scaling claim is per-step.

## 7. Environment

- **Machine:** Frontier (OLCF). AMD MI250X, **8 GCDs per node**; one MPI rank per GCD.
- **Modules:** `cpe/26.03`, `PrgEnv-gnu`, `miniforge3`, `rocm/7.2.0`, `craype-accel-amd-gfx90a`
- **Environment:** `/lustre/orion/proj-shared/lrn088/objective3/envs/superneuroabm_env_cupy14`
- **Launch:** `srun -N<nodes> -n<workers> -c7 --ntasks-per-gpu=1 --gpu-bind=closest`
- **Ranks per node differ by K** (memory ceiling, `../MEMORY_ANALYSIS.md`): K = 1000 runs
  **8 ranks/node**; K = 2000 and K = 4000 need **4 ranks/node**. The strong campaign is 8/node
  throughout.

## 8. Reproducing the campaigns

Runners take `"<K-list>" <ranks_per_node> "<worker-list>" [max_attempts]`; queue, walltime and node
count go on the `sbatch` command line. All rows append to one shared CSV.

```sh
cd scaling_analysis

# --- weak: 36 points -------------------------------------------------------
sbatch -q batch -N 256 -t 12:00:00 weak_3d_chunk.sh "1000"      8 "1 2 4 8 16 32 64 128 256 512 1024 2048"
sbatch -q batch -N 512 -t 12:00:00 weak_3d_chunk.sh "2000 4000" 4 "1 2 4 8 16 32 64 128 256 512 1024 2048"

# --- strong: 8 points, N held at 204800 ------------------------------------
sbatch -q batch -N 256 -t 12:00:00 strong_3d_chunk.sh "204800"  8 "16 32 64 128 256 512 1024 2048"

# --- consolidate, then draw ------------------------------------------------
python analyze_weak.py                    # -> outputs/weak_3d_final.csv
python analyze_strong.py                  # -> outputs/strong_3d_final.csv
python paper_figures/make_all.py          # -> paper_figures/figs/*.{png,pdf}
python paper_figures/make_all.py --print  # plus every plotted value as text
```

The runners invoke the drivers with `--topology torus3d`, `--neurons-per-worker` (weak) or
`--total-neurons` (strong), `--ticks 100 --update-ticks 1 --in-degree K --g 5.0 --J-E 0.02581
--delay 1.5 --firing-rate 10.0 --diagnostics --csv <shared>`. `--connection-radius` is left empty
so the radius is auto-selected and fixed across the sweep.

## 9. Known caveats

Recorded so they are not discovered downstream:

- **`cupy.jit` transpile crash** (`'Assign' object has no attribute 'name'`) is nondeterministic at
  roughly 1-in-6 and is **not** driven by `PYTHONHASHSEED` (job 5069521: 5 pass / 1 fail with both
  random and fixed seed). Handled by the runners' `MAX_ATTEMPTS` retry with a fresh cache; a
  workaround, not a fix.
- **K = 4000, w = 2048 reads +4.0 % over its own median** — construction settles around tick 30
  there, inside the step window, which is why that curve's plateau spread is 4.33 % against 0.86 %
  and 1.30 %. Reported, not trimmed; `step_median_s` makes it checkable.
- **Strong w = 2048 settles at ~tick 51**, later than any other point, and reads +3.4 % over its
  median for the same reason.
- **K = 4000, w = 32 has `generation_time` = 0.0011 s** against ~14 s at every other K = 4000
  point — almost certainly a run that reused a cached network rather than generating one. It
  affects only the generation phase, not `step_s`, so neither figure is touched, but it is a
  genuine defect in the recorded data.
- **Strong w = 2048: `peers_mean` = 95.55 against 104 geometrically reachable.** The 4×5×5 tile is
  small enough that a rank's K = 1000 draws miss ~8 of the tiles it could reach. This is why it is
  the only non-integer peer value in either dataset, and why the measured count — what actually
  exchanges — is published rather than the geometric bound.
- **`n_runs` = 2 only at weak w = 1.** Every other point in both campaigns is a single run, so
  point-to-point scatter is not characterised.
- **E/I is spatially segregated within a tile** (§1.3), so these runs are a communication
  benchmark and not a statement about Brunel dynamics.

## 10. Figure captions (drafts)

> **F1. Weak scaling, 64→2048 GPUs (12,500 neurons/GPU held constant; K is the in-degree, i.e.
> incoming synapses per neuron).** Parallel efficiency of a simulation step, normalised to
> **w=64** — the smallest configuration whose rank grid is ≥4 ranks along every axis. The Moore
> stencil has 26 = 3³−1 directions, but on a periodic torus two of them land on the same rank when
> the grid is narrower than that, so only at w=64 does every direction reach a *different* rank:
> 26 distinct MPI peers, constant from there on. Smaller runs therefore measure a cheaper
> communication pattern rather than a faster code. Legend values are the sustained per-tick rate
> (plateau median) and the efficiency at 2048 GPUs. **Across the constant-peer plateau
> (w=64→2048, a 32× span) efficiency holds at 99.1–100.0 % (K=1000), 100.0–101.3 % (K=2000) and
> 95.8–100.0 % (K=4000)**, with step time varying by 0.9 %, 1.3 % and 4.3 % respectively — the last
> carried entirely by its w=2048 point. Efficiency rises with in-degree because added synapses are
> volume work while the halo is a surface. Against a 1-GPU baseline the same plateau reads
> 65.3–65.8 %, 77.0–78.0 % and 79.4–82.8 %; that gap is the one-time price of turning communication
> on, not a scaling trend. *Note the y-axis begins at 90 %.*

> **F2. Strong scaling, fixed 204,800-neuron problem, 16→2048 GPUs.** Speedup in **wall time** —
> `total_time`, the whole run: generation, model creation, GPU setup, construction and every tick —
> against w=16, the smallest configuration the problem fits on (a 1-GPU baseline needs ~205 M
> synapses on one GCD and never ran; w=16 itself sits in the ramp at 11 peers). It reaches **13.4×
> on 64× the GPUs**, because network construction — 99.8 % of a 100-tick run — still parallelises.
> Peer count on the right axis leaves 26 at w=512, where the shrinking tile falls under the fixed
> stencil radius — context for the flattening, not a demonstrated cause. What the curve does *not*
> claim is that the simulation step scales: the step peaks at 1.44× at w=256 and reverses to 0.76×
> at w=2048, because per-tick GPU compute is 0.370 ms at every point across a 128× span of per-rank
> work, so the kernel is launch-bound from the first point and shrinking the tile only grows the
> halo.

## 11. Software layering

This directory is a **presentation** layer. It reads the two final CSVs and derives nothing that
`analyze_weak.py` / `analyze_strong.py` already derive — those own consolidation, the warm-up
window and every timing. A second derivation here would drift silently from the diagnostic
figures, which is the failure mode the campaign was rebuilt to avoid.

| | owns | writes |
|---|---|---|
| `outputs/ticks/*.csv` | raw per-tick record, no window applied | the runs |
| `analyze_{weak,strong}.py` | consolidation, warm-up window, all timings | `outputs/*_final.csv`, `figures/*.png` (diagnostic) |
| `paper_figures/` | presentation only | `paper_figures/figs/*.{png,pdf}` |

Style is ported from `GGap/SC2026/figures/scripts/_style.py` (IEEE widths, 600 dpi, `fonttype 42`)
so the two projects' figures sit together in a proceedings. The series palette is **not** ported:
it stays the one `analyze_weak.py` uses, so a given `K` is the same colour in a paper figure and in
the diagnostic figure it was checked against.

## 12. References

- Brunel, N. (2000). Dynamics of sparsely connected networks of excitatory and inhibitory spiking
  neurons. *Journal of Computational Neuroscience* 8, 183–208.
- Mendygral, P. J., et al. (2017). WOMBAT: A scalable and high-performance astrophysical
  magnetohydrodynamics code. *ApJS* 228, 23. — the off-node communication saturation convention.
- `docs/BRUNEL_SCALING.md` §5.5a (weak, 3D stencil), §5.5b (bio-realistic small-world variant),
  §5.5c (strong) — design discussion and the full argument for the wiring convention.
- `scaling_analysis/MEMORY_ANALYSIS.md` — the per-rank memory ceiling behind the ranks/node split.
