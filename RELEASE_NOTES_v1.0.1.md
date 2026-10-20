# SuperNeuroABM Release Notes: v1.0.0 → v1.0.1

**Release period:** Feb 3, 2026 – Mar 27, 2026  
**Diff summary:** 59 files changed, +6,452 / −2,262 lines

---

## 1. New Neuron (Soma) Models

- **HG-LIF Soma** (`hg_lif.py`): Added a new half-Gaussian LIF soma model to support bounded STDP experiments.

## 2. New Synapse & Learning Rule Models

- **Weighted Synapse** (`weighted_synapse.py`): New non-plastic weighted synapse step function with full test coverage.
- **Bounded STDP** (`exp_pair_wise_stdp_bounded.py`): Exponential pair-wise STDP with hard weight bounds.
- **Memristive STDP** (`exp_pair_wise_stdp_memristive.py`): Conductance-based memristive STDP learning rule for hardware-aware simulations.
- **Quantized (Three-Bit) STDP** (`exp_pair_wise_stdp_quantized.py`): Renamed and cleaned up from the old three-bit exponential pair-wise implementation.
- Removed legacy files: `Low_resolution_synapse.py`, `Three-bit_exp_pair_wise.py`, and the standalone `learning_rule_selector.py`.

## 3. Model Core / API Improvements

- **Custom component registration API**: Users can now register their own soma, synapse, and learning rule types via `model.register_*()` methods (PR #50), with a full example (`register_custom_network.py`) and unit tests.
- **Auto-generated learning rule selector**: The learning rule selector is now built automatically during `__init__` instead of being maintained as a separate file.
- **Learning rule helper function**: Added a helper to simplify learning rule configuration (`add learning rule helper function`).
- **Spike mask / subset recording**: Users can record output spikes from a subset of somas, reducing memory and I/O for large networks.
- **Reset redesign**: `reset()` reworked to incorporate SAGESim's direct GPU read-back, improving correctness on GPU backends.
- **Global variable refactor**: Split two global variables and adopted SAGESim's new random-number API, eliminating the need to keep `seed` as a global.
- **Learning rule cleanup**: Removed `_tick` from learning rule selector call args; learning rule params no longer write to internal state.
- **Property length detection**: Replaced hardcoded property lengths with automatic max-length detection.
- **Removed `tau_rise`/`tau_fall`** from weighted synapse config for simplicity.

## 4. Performance Optimizations

- **Input spike tensor pointer optimization** (PR #98): Use `input_spike_tensor[agent_index][1]` to avoid O(N) search each tick.
- **Output spike tensor circular buffer**: Changed `output_spike_tensor` to a length-2 circular buffer with early-exit scan, replacing the running-pointer approach.
- **Sort input spikes by tick** before entering SAGESim `simulation()` for better cache behavior.
- **Euler's approximation for exp()**: Reverted to Euler's estimation of `exp()` for speed.
- **MPI broadcast removal in construction**: Eliminated redundant MPI broadcasts; added config caching in `create_soma`/`create_synapse`.
- **Rank-0-only network loading**: Only rank 0 loads the network file, reducing redundant I/O in distributed runs.
- **Optimized model construction** (`model.py`): Major redesign of network construction path (~+1,169 / −400 lines in model.py).

## 5. Scaling & Benchmarking

- Rewrote weak scaling test infrastructure:
  - New scripts: `run_single_node_test.sh`, `submit_weak_scaling_parallel.sh`, `weak_1node_1to8gpus.sh`, `weak_1node_1to8gpus_profile.sh`, `weak_1to20nodes_sequential.sh`.
  - Added 4 additional CSV time-tracking metrics for detailed performance breakdowns.
  - Removed obsolete scripts (`test_simple_weak_scaling.py`, `weak_scaling_const_density.*`, old `.out` files).

## 6. Examples & Tutorials

- **Masquelier 2008 example**: Full reproduction including experiment runner (`run_experiment_hg.py`), figure generation (`generate_figures.py`), potential plotting, and custom config.
- **Digit classification tutorial**: Added as an example with config and parameter tuning.
- **Custom component registration example**: Demonstrates registering user-defined soma/synapse/learning rules with alpha synapse, bounded STDP, exp-LIF, and symmetric STDP step functions.

## 7. Build System

- **Migrated from `setup.py` to `pyproject.toml`**: Modern Python packaging; `setup.py` removed.
- **Updated SAGESim dependency** to `>=0.6.0`.

## 8. Testing

- **New test files:**
  - `test_weighted_synapse.py` — comprehensive weighted synapse tests (550+ lines)
  - `test_quantized_stdp.py` — quantized STDP validation
  - `test_memristive_stdp.py` — memristive STDP validation
  - `test_memristive_synapse.py` — memristive synapse validation
  - `test_registration_api.py` — custom component registration
  - `test_spike_mask.py` — spike subset recording
- Updated existing tests to match API changes (reset, model setup, internal state tracking).

## 9. Code Quality

- Standardized all file names to lowercase.
- Code reorganization across step functions and model modules.
- Removed redundant/legacy files and cleaned up imports.
- Updated `.gitignore` to exclude generated figures and outputs.
