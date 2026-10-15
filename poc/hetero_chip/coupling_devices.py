"""Kernels for run_coupling_probe.py: a device (synapse agent) that reads ANOTHER device's state.

heater  is [0] I [1] heat at even ticks [2] heat at odd ticks   (published, double-buffered)
sensor  is [0] I [1] g                                           (reads its slot-0 neighbour)
Double-buffering by tick parity (like output_spikes_tensor) makes the read race-free on one
GPU and equal to the ghost copy (synced after every tick) on several.
"""
import cupy as cp
from cupyx import jit

from sagesim.math_utils import rand_normal
from superneuroabm.step_functions.synapse.util import get_soma_spike


@jit.rawkernel(device="cuda")
def r9_relay(
    tick, agent_index, dt, I_bias, agent_ids, breeds, locations,
    neuron_params, learning_params, internal_states, learning_internal_states,
    synapse_history, input_spikes_tensor, output_spikes_tensor,
    internal_states_buffer, learning_internal_states_buffer,
):
    synapse_indices = locations[agent_index]
    I = 0.0
    for i in range(len(synapse_indices)):
        synapse_index = int(synapse_indices[i])
        if synapse_index >= 0 and not cp.isnan(synapse_indices[i]):
            I += internal_states[synapse_index][0]
    s = 0.0
    if I >= neuron_params[agent_index][0]:
        s = 1.0
    output_spikes_tensor[agent_index][int(tick) % 2] = s


@jit.rawkernel(device="cuda")
def r9_lif(
    tick, agent_index, dt, I_bias, agent_ids, breeds, locations,
    neuron_params, learning_params, internal_states, learning_internal_states,
    synapse_history, input_spikes_tensor, output_spikes_tensor,
    internal_states_buffer, learning_internal_states_buffer,
):
    synapse_indices = locations[agent_index]
    I = 0.0
    for i in range(len(synapse_indices)):
        synapse_index = int(synapse_indices[i])
        if synapse_index >= 0 and not cp.isnan(synapse_indices[i]):
            I += internal_states[synapse_index][0]
    v = internal_states[agent_index][0]
    v = v - v * dt / neuron_params[agent_index][1] + I
    s = 0.0
    if v >= neuron_params[agent_index][0]:
        s = 1.0
        v = 0.0
    internal_states[agent_index][0] = v
    output_spikes_tensor[agent_index][int(tick) % 2] = s


@jit.rawkernel(device="cuda")
def r9_wire(
    tick, agent_index, dt, I_bias, agent_ids, breeds, locations,
    synapse_params, learning_params, internal_states, learning_internal_states,
    synapse_history, input_spikes_tensor, output_spikes_tensor,
    internal_states_buffer, learning_internal_states_buffer,
):
    spike = get_soma_spike(
        tick, agent_index, dt, I_bias, agent_ids, locations[agent_index][0], int(tick),
        input_spikes_tensor, output_spikes_tensor,
    )
    internal_states[agent_index][0] = spike * synapse_params[agent_index][0]


@jit.rawkernel(device="cuda")
def r9_heater(
    tick, agent_index, dt, I_bias, agent_ids, breeds, locations,
    synapse_params, learning_params, internal_states, learning_internal_states,
    synapse_history, input_spikes_tensor, output_spikes_tensor,
    internal_states_buffer, learning_internal_states_buffer,
):
    """Programmed on every presynaptic spike; each pulse dumps a random amount of heat."""
    t = int(tick)
    spike = get_soma_spike(
        tick, agent_index, dt, I_bias, agent_ids, locations[agent_index][0], t,
        input_spikes_tensor, output_spikes_tensor,
    )
    h = 0.0
    if spike != 0.0:
        h = 1.0 + 0.5 * rand_normal(tick, agent_index, 1)
    internal_states[agent_index][1 + t % 2] = h
    internal_states[agent_index][0] = spike * synapse_params[agent_index][0]


@jit.rawkernel(device="cuda")
def r9_sensor(
    tick, agent_index, dt, I_bias, agent_ids, breeds, locations,
    synapse_params, learning_params, internal_states, learning_internal_states,
    synapse_history, input_spikes_tensor, output_spikes_tensor,
    internal_states_buffer, learning_internal_states_buffer,
):
    """Slot 0 is a neighbouring DEVICE (another synapse), not a soma: its last published heat
    disturbs this device's conductance, which drives the post-neuron every tick."""
    t = int(tick)
    nb = int(locations[agent_index][0])
    heat = internal_states[nb][1 + (t + 1) % 2]          # what nb published at tick t-1
    g = internal_states[agent_index][1]
    g = g + synapse_params[agent_index][2] * heat * (1.0 + 0.1 * rand_normal(tick, agent_index, 2))
    internal_states[agent_index][1] = g
    internal_states[agent_index][0] = g
