# SuperNeuroABM Functionality Guide

## What is SuperNeuroABM?

**SuperNeuroABM** is a **neuromorphic agent-based modeling (ABM) framework** for simulating spiking neural networks with GPU acceleration and distributed computing support. In this framework, every neuron and synapse is modeled as an independent **agent** that executes computational steps in parallel on GPUs.

---

## Core Concept: Agents as Neurons and Synapses

### Agent-Based Modeling Architecture

In SuperNeuroABM:

```
Network = Collection of Agents
├─ Soma Agents (Neurons)
│  ├─ Each neuron is an independent agent
│  ├─ Has its own state (membrane potential, spike history, etc.)
│  └─ Executes step functions every simulation tick
│
└─ Synapse Agents (Connections)
   ├─ Each synapse is an independent agent
   ├─ Has its own state (synaptic current, STDP traces, etc.)
   └─ Executes step functions every simulation tick
```

### Why Agent-Based Modeling?

Traditional neural network simulators often use:
- **Matrix operations**: Fast but inflexible (all neurons must be identical)
- **Event-driven queues**: Memory-efficient but hard to parallelize

SuperNeuroABM uses **agent-based modeling** because:

1. **Heterogeneous Networks**: Each neuron/synapse can have different parameters and behaviors
2. **Natural Parallelism**: Agents execute independently → perfect for GPU parallelization
3. **Spatial Distribution**: Agents can be distributed across multiple compute nodes via MPI
4. **Extensibility**: Easy to add new neuron/synapse types without changing the core framework

---

## Agent Breeds: Types of Neurons and Synapses

In SuperNeuroABM, agents are organized by **breeds**. A breed defines:
- What **properties** an agent has (parameters, state variables)
- What **step functions** the agent executes (computational behavior)
- What **priority** each step function runs at (execution order)

**Important**: Each neuron type is a separate breed, and each synapse type is a separate breed.

### Soma Breeds (Neuron Types)

**SuperNeuroABM currently has 2 soma breeds implemented:**

#### 1. LIF Soma (Leaky Integrate-and-Fire Neuron)

**Breed Name**: `lif_soma`

**Description**: A simple, computationally efficient neuron model that integrates input current until reaching a threshold, then emits a spike and resets.

**Mathematical Model**:
```
dv/dt = (v_rest - v)/(R*C) + I/C

if v ≥ v_thr:
    emit spike
    v ← v_reset
    enter refractory period
```

**Parameters** (from `component_base_config.yaml`):

| Parameter | Symbol | Default Value | Description |
|-----------|--------|---------------|-------------|
| `C` | C | 100 pF | Membrane capacitance |
| `R` | R | 100 MΩ | Leak resistance |
| `vthr` | v_thr | -50 mV | Spike threshold |
| `vrest` | v_rest | -70 mV | Resting potential |
| `vreset` | v_reset | -70 mV | Reset potential after spike |
| `tref` | t_ref | 2 ms | Refractory period |
| `tref_allows_integration` | - | 0 | Whether to integrate current during refractory period |
| `I_in` | I_in | 0 | Constant input current |
| `scaling_factor` | - | 1.0 | Scaling factor for synaptic input |

**Internal State Variables**:
```python
[v, tcount, tlast]
# v: membrane potential (mV)
# tcount: number of ticks since simulation start
# tlast: time of last spike (ms)
```

**Step Functions**:
1. `lif_soma_step_func` (Priority 0) - Computes membrane dynamics and spike generation

**Use Cases**:
- Fast, large-scale simulations
- Rate coding networks
- Logic circuits (AND, OR gates)

**Example**:
```python
model = NeuromorphicModel()
soma_id = model.create_soma(
    breed="lif_soma",
    config_name="config_0",
    hyperparameters_overrides={
        "vthr": -55.0e-3,  # Custom threshold
        "tref": 3.0e-3     # Custom refractory period
    }
)
```

---

#### 2. Izhikevich Soma (Izhikevich Neuron)

**Breed Name**: `izh_soma`

**Description**: A biologically realistic neuron model that can reproduce various spiking patterns (regular spiking, bursting, fast spiking, etc.) using only 4 parameters.

**Mathematical Model**:
```
dv/dt = (k*(v - v_rest)*(v - v_thr) - u + I) / C
du/dt = a * (b*(v - v_rest) - u)

if v ≥ v_peak:
    emit spike
    v ← c
    u ← u + d
```

**Parameters** (from `component_base_config.yaml`):

| Parameter | Symbol | Default Value | Description |
|-----------|--------|---------------|-------------|
| `k` | k | 0.7 | Scaling factor for voltage dynamics |
| `vth` | v_thr | -40 mV | Instantaneous threshold |
| `C` | C | 100 pF | Membrane capacitance |
| `a` | a | 0.03 | Recovery time constant |
| `b` | b | -2.0 | Sensitivity of recovery to voltage |
| `c` | c | -50 mV | Reset voltage |
| `d` | d | 100 | Recovery variable reset increment |
| `vrest` | v_rest | -60 mV | Resting potential |
| `vpeak` | v_peak | 35 mV | Spike cutoff value |
| `I_in` | I_in | 0 | Constant input current |
| `scaling_factor` | - | 1.0 | Scaling factor for synaptic input |

**Internal State Variables**:
```python
[v, u, tcount, tlast]
# v: membrane potential (mV)
# u: recovery variable
# tcount: number of ticks since simulation start
# tlast: time of last spike (ms)
```

**Step Functions**:
1. `izh_soma_step_func` (Priority 0) - Computes membrane dynamics and spike generation

**Neuron Types** (by parameter tuning):

| Type | Parameters | Behavior |
|------|------------|----------|
| Regular Spiking | a=0.02, b=0.2, c=-65, d=8 | Adapting spike train |
| Intrinsically Bursting | a=0.02, b=0.2, c=-55, d=4 | Burst of spikes then adaptation |
| Fast Spiking | a=0.1, b=0.2, c=-65, d=2 | Non-adapting rapid firing |
| Chattering | a=0.02, b=0.2, c=-50, d=2 | Stereotypical bursting |

**Use Cases**:
- Biologically realistic cortical networks
- Temporal coding and spike-timing research
- Multi-modal spiking behavior (bursting, adaptation, resonance)

**Example**:
```python
# Create a regular spiking neuron
model = NeuromorphicModel()
rs_neuron = model.create_soma(
    breed="izh_soma",
    config_name="config_0",
    hyperparameters_overrides={
        "a": 0.02,
        "b": 0.2,
        "c": -65.0e-3,
        "d": 8.0
    }
)

# Create a fast spiking neuron
fs_neuron = model.create_soma(
    breed="izh_soma",
    config_name="config_0",
    hyperparameters_overrides={
        "a": 0.1,
        "b": 0.2,
        "c": -65.0e-3,
        "d": 2.0
    }
)
```

---

### Synapse Breeds (Connection Types)

**SuperNeuroABM currently has 1 synapse breed implemented:**

Each synapse type is its own breed. Additional synapse types (e.g., double exponential, NMDA, GABA) can be added as new breeds.

#### Single Exponential Synapse

**Breed Name**: `single_exp_synapse`

**Description**: A current-based synapse with exponential decay dynamics. Each synapse connects a pre-synaptic soma to a post-synaptic soma and converts spike events into synaptic currents.

**Mathematical Model**:
```
I_syn(t + dt) = I_syn(t) * (1 - dt/τ_fall) + spike_pre(t) * weight * scale

Post-synaptic soma receives: I_syn
```

**Parameters** (from `component_base_config.yaml`):

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| `weight` | w | Synaptic strength (can be positive or negative) |
| `synaptic_delay` | delay | Transmission delay (ms) |
| `scale` | scale | Additional scaling factor |
| `tau_fall` | τ_fall | Decay time constant (ms) |
| `tau_rise` | τ_rise | Rise time constant (ms, currently unused) |
| `stdp_type` | - | Learning rule selector (-1 = none, 0 = exp pair-wise STDP) |
| `tau_pre_stdp` | τ_pre | Pre-synaptic trace decay (ms, STDP only) |
| `tau_post_stdp` | τ_post | Post-synaptic trace decay (ms, STDP only) |
| `a_exp_pre` | A+ | Pre-before-post learning rate (STDP only) |
| `a_exp_post` | A- | Post-before-pre learning rate (STDP only) |

**Internal State Variables**:
```python
[I_synapse, I_synapse_supp, pre_trace, post_trace]
# I_synapse: current synaptic current (A)
# I_synapse_supp: supplementary current (reserved)
# pre_trace: pre-synaptic STDP trace
# post_trace: post-synaptic STDP trace
```

**Step Functions**:

Yes, **STDP learning is implemented as a separate step function**! This is a key architectural design:

1. **`synapse_single_exp_step_func`** (Priority 100)
   - Reads pre-synaptic spike from previous tick
   - Updates synaptic current with exponential decay
   - Injects current into post-synaptic soma

2. **`learning_rule_selector`** (Priority 101)
   - Reads pre and post-synaptic spikes
   - Updates STDP traces
   - Modifies synaptic weight based on spike timing
   - Dispatches to specific learning rules (exponential pair-wise STDP, etc.)

**Why Two Step Functions?**

This separation provides several advantages:

1. **Modularity**: Synapse dynamics and learning are decoupled
2. **Priority Control**: Learning happens *after* current propagation (Priority 101 > 100)
3. **Optional Learning**: Can disable learning by setting `stdp_type = -1` without modifying synapse dynamics
4. **Extensibility**: Easy to add new learning rules without changing synapse code

---


**Usage**: Learning networks, temporal association, spike-timing dependent plasticity

---

## Step Function Execution Model

### Priority-Based Scheduling

SuperNeuroABM executes agent step functions in a **priority-ordered sequence** every simulation tick:

```
Simulation Tick N:
├─ Priority 0: Soma step functions (all neuron types)
│  ├─ Integrate synaptic currents from previous tick
│  ├─ Update membrane potential
│  ├─ Detect spikes
│  └─ Write spike to output_spikes_tensor[N]
│
├─ Priority 100: Synapse step functions (synapse dynamics)
│  ├─ Read pre-synaptic spike from output_spikes_tensor[N-1]
│  ├─ Update synaptic current (exponential decay)
│  └─ Inject current into post-synaptic soma
│
└─ Priority 101: Learning step functions (STDP)
   ├─ Read pre-synaptic spike from output_spikes_tensor[N-1]
   ├─ Read post-synaptic spike from output_spikes_tensor[N]
   ├─ Update STDP traces
   └─ Modify synaptic weight based on spike timing
```

### Why This Ordering?

**Priority 0 (Somas First)**:
- Neurons generate spikes based on inputs from *previous* tick
- Ensures all spikes are available for synapses to read
- Implements realistic 1-tick synaptic delay (~1 ms at dt=1ms)

**Priority 100 (Synapse Dynamics)**:
- Synapses read spikes from tick N-1 (no race conditions)
- Compute synaptic currents
- Somas will integrate these currents at tick N+1

**Priority 101 (Learning)**:
- Learning rules need both pre (N-1) and post (N) spikes
- Runs after synapse dynamics to avoid interference
- Can be disabled independently by setting `stdp_type = -1`

### Example Timeline

```
Tick 0:
  [Soma A] v = -70 mV, no spike
  [Synapse A→B] I_syn = 0

Tick 1:
  [Soma A] v = -50 mV, SPIKE! → output_spikes_tensor[1] = 1
  [Synapse A→B] Reads output_spikes_tensor[0] = 0, I_syn = 0
  [STDP] No update (no spike pair)

Tick 2:
  [Soma A] v resets to -70 mV
  [Synapse A→B] Reads output_spikes_tensor[1] = 1, I_syn = weight * scale
  [Soma B] Receives I_syn, v increases
  [STDP] Updates pre_trace

Tick 3:
  [Soma B] v = -51 mV, SPIKE! → output_spikes_tensor[3] = 1
  [STDP] Detects pre-spike at t=1, post-spike at t=3
         Δt = 2 ms (causal), weight INCREASES (LTP)
```

---

## STDP Learning Mechanism

### What is STDP?

**Spike-Timing-Dependent Plasticity (STDP)** is a biological learning rule where synaptic strength changes based on the *relative timing* of pre- and post-synaptic spikes:

- **Causal** (pre before post): Weight increases → **LTP** (Long-Term Potentiation)
- **Anti-causal** (post before pre): Weight decreases → **LTD** (Long-Term Depression)

### Implementation Details

STDP in SuperNeuroABM is implemented via the **`learning_rule_selector`** step function (Priority 101).

#### Learning Rule Selector

**File**: `superneuroabm/step_functions/synapse/stdp/learning_rule_selector.py`

**Purpose**: Dispatches to appropriate learning rule based on `stdp_type` parameter

```python
@jit.rawkernel(device="cuda")
def learning_rule_selector(tick, agent_index, ...):
    stdp_type = synapse_params[agent_index][9]

    if stdp_type == 0:
        # Exponential pair-wise STDP
        exp_pair_wise_stdp(...)
    elif stdp_type == 1:
        # Future: Triplet STDP
        pass
    # else: No learning (stdp_type == -1)
```

**Supported Learning Rules**:

| `stdp_type` | Learning Rule | File |
|-------------|---------------|------|
| -1 | No learning (static weights) | N/A |
| 0 | Exponential pair-wise STDP | `exp_pair_wise_stdp.py` |
| 1+ | Reserved for future rules | - |

#### Exponential Pair-Wise STDP

**File**: `superneuroabm/step_functions/synapse/stdp/exp_pair_wise_stdp.py`

**Algorithm**:

```python
# Update pre-synaptic trace (decays exponentially)
pre_trace = pre_trace * exp(-dt / τ_pre) + pre_spike

# Update post-synaptic trace
post_trace = post_trace * exp(-dt / τ_post) + post_spike

# Weight update
if post_spike == 1:
    # Post-synaptic neuron just spiked
    # Strengthen if pre-trace is high (causal pairing)
    Δw_LTP = A_pre * pre_trace
else:
    Δw_LTP = 0

if pre_spike == 1:
    # Pre-synaptic neuron just spiked
    # Weaken if post-trace is high (anti-causal pairing)
    Δw_LTD = A_post * post_trace  # A_post is negative
else:
    Δw_LTD = 0

# Apply weight update
weight = clip(weight + Δw_LTP + Δw_LTD, w_min, w_max)
```

**Parameters**:
- **τ_pre**: Pre-synaptic trace decay time constant (e.g., 20 ms)
- **τ_post**: Post-synaptic trace decay time constant (e.g., 20 ms)
- **A_pre** (`a_exp_pre`): LTP learning rate (positive, e.g., +0.01)
- **A_post** (`a_exp_post`): LTD learning rate (negative, e.g., -0.01)

**STDP Window**:

```
      Weight Change (Δw)
           ↑
    +0.01  |     ╱
           |    ╱
           |   ╱
    ───────┼──╱────────────────── Δt (ms)
           | ╱   ╲
           |╱     ╲
   -0.01   |       ╲

    ← post before pre    pre before post →
       (anti-causal)        (causal)
           LTD                 LTP
```

**Example**:

```python
# Create synapse with STDP learning
model = NeuromorphicModel()

soma_pre = model.create_soma(breed="lif_soma", config_name="config_0")
soma_post = model.create_soma(breed="lif_soma", config_name="config_0")

synapse = model.create_synapse(
    breed="single_exp_synapse",
    pre_soma_id=soma_pre,
    post_soma_id=soma_post,
    config_name="exp_pair_wise_stdp_config_0",  # STDP enabled
    hyperparameters_overrides={
        "weight": 10.0,
        "tau_pre_stdp": 20.0e-3,    # 20 ms pre trace
        "tau_post_stdp": 20.0e-3,   # 20 ms post trace
        "a_exp_pre": 0.01,          # LTP rate
        "a_exp_post": -0.01         # LTD rate
    }
)

# Simulate with STDP learning active
model.setup()
model.simulate(ticks=1000)

# Check learned weight
final_weight = model.get_agent_property_value(synapse, "hyperparameters")[0]
print(f"Initial weight: 10.0, Final weight: {final_weight}")
```

---

## Agent Properties

Every agent (soma or synapse) has multiple **property arrays** that store its state and parameters:

### Soma Agent Properties

```python
soma_properties = {
    "hyperparameters": [...]        # Fixed parameters (C, R, vthr, etc.)
    "learning_hyperparameters": [...] # Learning-related parameters (STDP)
    "internal_state": [...]         # Dynamic state (v, u, tcount, tlast)
    "internal_learning_state": [...] # Learning state (traces, dW)
    "synapse_delay_reg": []         # Delay buffer for spike transmission
    "input_spikes_tensor": []       # External input spikes
    "output_spikes_tensor": []      # Output spike history
    "internal_states_buffer": []    # History of internal states (optional)
    "internal_learning_states_buffer": []  # History of learning states (optional)
}
```

### Synapse Agent Properties

```python
synapse_properties = {
    "hyperparameters": [weight, delay, scale, tau_fall, tau_rise, ...]
    "learning_hyperparameters": [stdp_type, tau_pre_stdp, tau_post_stdp, a_exp_pre, a_exp_post]
    "internal_state": [I_synapse, I_synapse_supp, pre_trace, post_trace]
    "internal_learning_state": [pre_trace, post_trace, dW]
    "locations": [pre_soma_index, post_soma_index]  # Connectivity (pre-converted indices)
    "input_spikes_tensor": []       # External input spikes (if pre_soma = -1)
    "output_spikes_tensor": []      # Not used for synapses
    "internal_states_buffer": []    # History of synaptic currents (optional)
    "internal_learning_states_buffer": []  # History of STDP traces (optional)
}
```

### The `locations` Property: Network Connectivity

**Important**: The `locations` property is a special property managed by SAGESim that defines the **connectivity structure** of the network—i.e., "who connects to whom."

#### What is `locations`?

In agent-based modeling, agents need to know about their neighbors to interact with them. The `locations` property stores this connectivity information:

**For Soma Agents:**
```python
# Somas use locations to find their input synapses
soma_locations = model.get_agent_property_value(soma_id, "locations")
# Returns: List of synapse indices that connect to this soma
# Example: [synapse_0_index, synapse_1_index, synapse_2_index]
```

**For Synapse Agents:**
```python
# Synapses use locations to find their pre and post-synaptic somas
synapse_locations = model.get_agent_property_value(synapse_id, "locations")
# Returns: [pre_soma_index, post_soma_index]
# Example: [2, 5] means synapse connects soma #2 → soma #5
```

#### Key Features of `locations`

1. **Managed by SAGESim**: You don't directly set `locations`—it's automatically populated when you create synapses using `create_synapse(pre_soma_id, post_soma_id)`

2. **Pre-Converted Indices**: The `locations` array contains **local array indices** (not agent IDs)
   - This is a critical GPU optimization
   - Enables O(1) neighbor access in GPU kernels
   - No need for expensive linear searches

3. **Ordered Connections**: SuperNeuroABM uses `NetworkSpace(ordered=True)` to ensure consistent ordering of connections
   - Important for reproducible simulations
   - GPU kernels rely on stable neighbor ordering

#### How Connectivity is Established

When you create a synapse, SAGESim automatically updates the `locations` property:

```python
model = NeuromorphicModel()

# Create two neurons
soma_A = model.create_soma(breed="lif_soma", config_name="config_0")  # Agent ID: 0
soma_B = model.create_soma(breed="lif_soma", config_name="config_0")  # Agent ID: 1

# Create synapse A → B
synapse = model.create_synapse(
    breed="single_exp_synapse",
    pre_soma_id=soma_A,   # Pre-synaptic neuron
    post_soma_id=soma_B,  # Post-synaptic neuron
    config_name="no_learning_config_0"
)

# After model.setup(), SAGESim populates locations:
# synapse.locations = [soma_A_local_index, soma_B_local_index]
# soma_B.locations = [synapse_local_index, ...]  (all input synapses)
```

#### How Step Functions Use `locations`

**In Synapse Step Function:**
```python
@jit.rawkernel(device="cuda")
def synapse_single_exp_step_func(..., locations, ...):
    # Get pre and post soma indices (already converted by SAGESim)
    pre_soma_index = locations[agent_index][0]   # O(1) access
    post_soma_index = locations[agent_index][1]

    # Read pre-synaptic spike (no search needed!)
    if pre_soma_index >= 0:
        pre_spike = output_spikes_tensor[pre_soma_index][tick - 1]

    # Compute synaptic current...
```

**In Soma Step Function:**
```python
@jit.rawkernel(device="cuda")
def lif_soma_step_func(..., locations, internal_state, ...):
    # Get all input synapse indices
    synapse_indices = locations[agent_index]  # List of synapse indices

    # Sum synaptic currents from all input synapses
    I_synapse = 0.0
    for i in range(len(synapse_indices)):
        synapse_index = int(synapse_indices[i])
        if synapse_index >= 0:
            I_synapse += internal_state[synapse_index][0]  # O(1) access

    # Integrate membrane potential with synaptic input...
```

#### External Input Connections

For external input (no pre-synaptic neuron), use `pre_soma_id = -1`:

```python
input_synapse = model.create_synapse(
    breed="single_exp_synapse",
    pre_soma_id=-1,      # Special value: external input
    post_soma_id=soma_B,
    config_name="no_learning_config_0"
)

# After setup:
# input_synapse.locations = [-1, soma_B_local_index]
# The synapse checks if pre_soma_index == -1 and reads from input_spikes_tensor instead
```

#### Why `locations` Matters

**Without `locations` optimization:**
```python
# Naive approach: Linear search for neighbors (O(N))
for all_agents in network:
    if is_connected_to(current_agent, all_agents):
        interact(current_agent, all_agents)
# Time complexity: O(N²) for dense networks
```

**With `locations` optimization:**
```python
# Direct access to neighbors (O(1))
neighbor_indices = locations[agent_index]
for neighbor_index in neighbor_indices:
    interact(agent_index, neighbor_index)
# Time complexity: O(k) where k = number of neighbors
```

**Performance Impact**: 10-100× speedup in GPU kernels for large networks.

### Property Access

```python
# Get hyperparameters
params = model.get_agent_property_value(agent_id, "hyperparameters")
# For LIF: [C, R, vthr, tref, vrest, vreset, ...]
# For synapse: [weight, delay, scale, tau_fall, ...]

# Get internal state
state = model.get_agent_property_value(agent_id, "internal_state")
# For LIF: [v, tcount, tlast, ...]
# For synapse: [I_synapse, I_synapse_supp, pre_trace, post_trace]

# Get connectivity (locations)
connectivity = model.get_agent_property_value(agent_id, "locations")
# For soma: [synapse_idx_1, synapse_idx_2, ...]  (input synapses)
# For synapse: [pre_soma_idx, post_soma_idx]

# Set properties (before model.setup())
model.set_agent_property_value(agent_id, "hyperparameters", new_params)
```
## Key Features

SuperNeuroABM provides several unique capabilities that make it powerful for neuromorphic computing research:

---

### 1. Heterogeneous Network Support (Mixed Neuron Types)

You can combine different neuron models in the same network, enabling biologically realistic cortical circuits with diverse cell types:

```python
model = NeuromorphicModel()

# Fast-spiking LIF neuron (inhibitory interneuron)
fs_neuron = model.create_soma(
    breed="lif_soma",
    config_name="config_0",
    hyperparameters_overrides={
        "tref": 1.0e-3,  # Short refractory period (1 ms)
        "vthr": -55.0e-3
    },
    tags={"inhibitory"}
)

# Regular-spiking Izhikevich neuron (excitatory pyramidal cell)
rs_neuron = model.create_soma(
    breed="izh_soma",
    config_name="config_0",
    hyperparameters_overrides={
        "a": 0.02,
        "b": 0.2,
        "c": -65.0e-3,
        "d": 8.0
    },
    tags={"excitatory"}
)

# Excitatory connection (positive weight)
exc_synapse = model.create_synapse(
    breed="single_exp_synapse",
    pre_soma_id=rs_neuron,
    post_soma_id=fs_neuron,
    config_name="no_learning_config_0",
    hyperparameters_overrides={"weight": 20.0}
)

# Inhibitory connection (negative weight)
inh_synapse = model.create_synapse(
    breed="single_exp_synapse",
    pre_soma_id=fs_neuron,
    post_soma_id=rs_neuron,
    config_name="no_learning_config_0",
    hyperparameters_overrides={"weight": -30.0}  # Negative = inhibitory
)
```

**Benefits:**
- Model realistic cortical microcircuits (E/I balance)
- Mix simple (LIF) and complex (Izhikevich) neurons for efficiency
- Support for excitatory and inhibitory connections in the same network

---

### 2. Modular STDP Learning

Learning is implemented as a separate step function (Priority 101), allowing:

- **Enable/disable learning** without changing synapse dynamics (`stdp_type = -1`)
- **Mix learning and non-learning synapses** in the same network
- **Easy addition of new learning rules** (triplet STDP, voltage-dependent STDP, etc.)

```python
# Plastic synapse with STDP
learning_syn = model.create_synapse(
    breed="single_exp_synapse",
    pre_soma_id=neuron_A,
    post_soma_id=neuron_B,
    config_name="exp_pair_wise_stdp_config_0"  # STDP enabled
)

# Fixed synapse without learning
fixed_syn = model.create_synapse(
    breed="single_exp_synapse",
    pre_soma_id=neuron_B,
    post_soma_id=neuron_C,
    config_name="no_learning_config_0"  # STDP disabled
)
```

---

### 3. Customizable Step Functions for Hardware Constraints

SuperNeuroABM allows users to **easily modify step functions** to accommodate specific hardware designs or computational constraints. This is critical for neuromorphic hardware research where:

- **Fixed-point arithmetic** is required instead of floating-point
- **Quantized weights and states** must be used
- **Specific energy constraints** limit operations
- **Custom spike encoding schemes** are needed

**Example: Adding Fixed-Point Arithmetic**

```python
# Original LIF step function (superneuroabm/step_functions/soma/lif.py)
@jit.rawkernel(device="cuda")
def lif_soma_step_func(...):
    dv = (vrest - v) / (R * C) + (I_synapse * scaling_factor) / C
    v += dv * dt
    # ... rest of function

# Modified for 16-bit fixed-point (custom_lif.py)
@jit.rawkernel(device="cuda")
def lif_soma_step_func_fixedpoint(...):
    # Scale to 16-bit integer range
    SCALE = 2**15
    v_int = int(v * SCALE)
    dv_int = int(((vrest - v) / (R * C) + I_synapse / C) * SCALE * dt)
    v_int = v_int + dv_int

    # Check overflow and clip
    v_int = max(min(v_int, SCALE-1), -SCALE)

    v = float(v_int) / SCALE
    # ... rest of function
```

**How to Use Custom Step Functions:**

```python
from custom_lif import lif_soma_step_func_fixedpoint

model = NeuromorphicModel(
    soma_breed_info={
        "lif_soma_fixedpoint": [
            (lif_soma_step_func_fixedpoint, Path("custom_lif.py"))
        ]
    }
)

# Create neurons using custom breed
neuron = model.create_soma(breed="lif_soma_fixedpoint", config_name="config_0")
```

**Use Cases:**
- **Neuromorphic chip emulation**: Match hardware precision limits
- **Energy-aware simulations**: Model power consumption constraints
- **Algorithm development**: Test quantization effects before hardware deployment
- **Custom spike encodings**: Implement address-event representation (AER), rate coding variants, etc.

---

### 4. Tag-Based Network Organization

Use tags to organize and query agents for network analysis and selective operations:

```python
# Create neurons with descriptive tags
for i in range(100):
    layer = "layer1" if i < 50 else "layer2"
    neuron_type = "excitatory" if i % 4 != 0 else "inhibitory"

    model.create_soma(
        breed="lif_soma",
        config_name="config_0",
        tags={layer, neuron_type, f"neuron_{i}"}
    )

# Query by tags
layer1_neurons = model.get_agents_with_tag("layer1")
inhibitory_neurons = model.get_agents_with_tag("inhibitory")
specific_neuron = list(model.get_agents_with_tag("neuron_42"))[0]

print(f"Layer 1 has {len(layer1_neurons)} neurons")
print(f"Network has {len(inhibitory_neurons)} inhibitory neurons")
```

---

### 5. Memory Optimization via Optional State Tracking

For large-scale simulations, you can disable internal state tracking to dramatically reduce memory usage:

```python
# Full tracking mode (for analysis and debugging)
model_full = NeuromorphicModel(enable_internal_state_tracking=True)
# Memory: ~1 GB per 10K neurons for 1M timesteps
# Can retrieve: Full voltage traces, STDP traces, weight changes over time

# Lean mode (for large-scale production runs)
model_lean = NeuromorphicModel(enable_internal_state_tracking=False)
# Memory: ~10 MB per 10K neurons (100× reduction)
# Can retrieve: Final states, spike times only

# Simulate
model_lean.setup()
model_lean.simulate(ticks=1_000_000)  # 1 million timesteps, minimal memory

# Access results (still available in lean mode)
spike_times = model_lean.get_spike_times(soma_id=neuron_id)
final_weight = model_lean.get_agent_property_value(synapse_id, "hyperparameters")[0]
```

**Trade-offs:**
- **Full tracking**: High memory, complete history for analysis
- **Lean mode**: Low memory, spike times and final states only

---

### 6. Configuration-Driven Design

All neuron and synapse parameters are defined in `component_base_config.yaml`, making it easy to:

- **Share network configurations** across experiments
- **Version control** network parameters
- **Rapid prototyping** with parameter presets
- **Reproducible research** with documented configurations

```python
# Load default configurations
model = NeuromorphicModel()

# Create neuron using preset
soma = model.create_soma(
    breed="lif_soma",
    config_name="config_0",  # References component_base_config.yaml
    hyperparameters_overrides={"vthr": -55.0e-3}  # Override specific params
)
```

---

### 7. GPU-Accelerated Parallel Execution

All step functions execute in parallel on GPUs via CuPy, providing:

- **100-1000× speedup** over CPU implementations
- **Automatic kernel compilation** from Python code
- **Cross-platform support**: NVIDIA CUDA and AMD ROCm
- **Efficient memory access** via pre-converted indices (`locations` property)
