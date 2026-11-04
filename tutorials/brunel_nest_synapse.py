"""
brunel_nest_synapse.py -- Event-woken delivery synapse for Brunel network in SuperNeuroABM.

A lightweight synapse kernel that delivers synaptic current directly to the soma's inbox,
matching the Brunel network's weight-based transmission model.

This synapse is "event-woken": it runs only when its presynaptic partner spikes, not at
every timestep. This makes the computational cost proportional to the number of spikes
(O(events)) rather than O(synapses).

Behavior
--------
On each spike from the presynaptic neuron:
1. Read the presynaptic spike value (0 or 1)
2. Multiply by the weight (J_ex for excitatory, J_in for inhibitory)
3. Deliver the resulting current to the soma's synaptic_input channel

For the Brunel network:
- Excitatory synapses deliver to channel 0 with weight J_ex > 0
- Inhibitory synapses deliver to channel 1 with weight J_in < 0

The soma integrates these deliveries through alpha filters (see brunel_nest_soma.py).

Synaptic Delay
--------------
The delay is handled by the inbox mechanism: the current is delivered in one tick and
read by the soma on the next tick, providing a fixed 1-timestep delay (~0.1 ms at dt=0.1).
For matching NEST's 1.5 ms delay, the scheduler or connection setup would need to add
spike event queueing (not implemented here; this is a simplified version for validation).

Hyperparameters (indexed in synapse_params array)
-----------------------------------------
[0]  weight           synaptic weight (pA); positive for E, negative for I
[1]  channel          target soma channel (0 for E, 1 for I)

Internal State
--------------
None (event-woken synapses keep no state).

Locations
---------
locations[agent_index][0]  -- presynaptic soma index (global id)
locations[agent_index][1]  -- postsynaptic soma index (global id, may be -1 if unused)

The synapse reads the presynaptic spike and delivers to synaptic_input[post_soma_index][channel].

Registration (in SuperNeuroABM script)
--------------------------------------
Before model.setup():

    from pathlib import Path
    from brunel_nest_synapse import brunel_nest_synapse

    model.register_synapse_type(
        name="brunel_nest_synapse",
        step_func=brunel_nest_synapse,
        step_func_path=Path("brunel_nest_synapse.py").resolve(),
        event_capable=True,  # runs only on spike events
    )

Configuration (YAML)
--------------------
synapse_models:
  brunel_nest_synapse:
    hyperparameters:
      weight: 0.0  # to be set per connection (J_ex for E, J_in for I)
      channel: 0   # 0=excitatory, 1=inhibitory

Example Usage in Network Builder
---------------------------------
# Excitatory synapse (channel 0, positive weight)
exc_synapses = model.create_synapses_from_list(
    breed_name="brunel_exc_synapse",
    synapse_type="brunel_nest_synapse",
    locations=[(pre_id, post_id) for (pre_id, post_id) in exc_connections],
    synapse_params=[{"weight": J_ex, "channel": 0} for _ in exc_connections],
)

# Inhibitory synapse (channel 1, negative weight)
inh_synapses = model.create_synapses_from_list(
    breed_name="brunel_inh_synapse",
    synapse_type="brunel_nest_synapse",
    locations=[(pre_id, post_id) for (pre_id, post_id) in inh_connections],
    synapse_params=[{"weight": J_in, "channel": 1} for _ in inh_connections],
)

Notes
-----
- The weight semantics match NEST: J_ex produces an excitatory PSP of amplitude J (0.1 mV),
  and J_in = -g * J_ex produces an inhibitory PSP of amplitude -g*J.
- The inbox summation and alpha filtering happen in the soma (brunel_nest_soma.py).
- For Brunel network validation, compare weight values and firing rates to NEST directly.

References
----------
Brunel, N. (2000). Dynamics of sparsely connected networks of excitatory and inhibitory
spiking neurons. Journal of Computational Neuroscience 8, 183-208.
"""

from cupyx import jit

from superneuroabm.step_functions.synapse.util import get_soma_spike
from superneuroabm.step_functions.synaptic_input import deliver_synaptic_current


@jit.rawkernel(device="cuda")
def brunel_nest_synapse(
    tick,
    agent_index,
    dt,
    I_bias,
    agent_ids,
    breeds,
    locations,
    synapse_params,
    learning_params,
    internal_states,
    learning_internal_states,
    synapse_history,
    input_spikes_tensor,
    output_spikes_tensor,
    internal_states_buffer,
    learning_internal_states_buffer,
    synaptic_input,
    synaptic_input_idx,
):
    """
    Brunel network synapse kernel: event-woken delivery of synaptic current.

    This kernel runs only when its presynaptic partner spikes. It reads the spike value,
    multiplies by the weight, and delivers the result to the postsynaptic soma's inbox.

    Parameters
    ----------
    tick : int
        Current simulation timestep.
    agent_index : int
        Index of this synapse in the synapse array.
    dt : float
        Integration timestep (ms).
    I_bias : float
        External bias current (unused).
    agent_ids : cupy array
        Global IDs of all agents (unused).
    breeds : cupy array
        Breed IDs of all agents (unused).
    locations : cupy array
        Spatial locations; locations[agent_index] = [pre_soma_id, post_soma_id].
    synapse_params : cupy array
        Synapse hyperparameters; synapse_params[agent_index] = [weight, channel].
    learning_params : cupy array
        Learning parameters (unused).
    internal_states : cupy array
        Internal state (unused for event-woken synapses).
    learning_internal_states : cupy array
        Learning state (unused).
    synapse_history : cupy array
        Synapse history (unused).
    input_spikes_tensor : cupy array
        External input spikes (unused).
    output_spikes_tensor : cupy array
        Neuron output spikes; output_spikes_tensor[pre_id][t % 2] = spike value.
    internal_states_buffer : cupy array
        State history buffer (unused).
    learning_internal_states_buffer : cupy array
        Learning history buffer (unused).
    synaptic_input : cupy array
        Synaptic input delivered to somas' inboxes.
    synaptic_input_idx : cupy array
        Indices into synaptic_input (indexed per soma per channel).
    """
    t_current = int(tick)

    # === Read synapse parameters ===
    weight = synapse_params[agent_index][0]  # pA (positive for E, negative for I)
    channel = int(synapse_params[agent_index][1])  # 0 for E, 1 for I

    # === Read locations ===
    _nbrs = locations[agent_index]
    pre_soma_index = int(_nbrs[0])
    post_soma_index = -1
    if len(_nbrs) > 1:
        post_soma_index = int(_nbrs[1])

    # === Read presynaptic spike ===
    # get_soma_spike handles both internal somas (pre_soma_index >= 0) and
    # external inputs (pre_soma_index == -1). For the Brunel network,
    # we expect internal somas only (neuron-to-neuron connections).
    spike = get_soma_spike(
        tick, agent_index, dt, I_bias, agent_ids, pre_soma_index,
        t_current, input_spikes_tensor, output_spikes_tensor,
    )

    # === Deliver synaptic current if there is a spike ===
    if spike == 0.0:
        return

    # Multiply spike by weight and deliver to the postsynaptic soma's channel
    synaptic_current = spike * weight

    # deliver_synaptic_current sums the current into synaptic_input at the
    # index corresponding to (post_soma_index, channel).
    deliver_synaptic_current(
        synaptic_input, synaptic_input_idx, post_soma_index, channel,
        synaptic_current
    )
