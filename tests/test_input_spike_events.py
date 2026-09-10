#!/usr/bin/env python
"""Delivery semantics of the tick-major input-spike event list.

External input spikes live in a host-side event store (``add_spike*``), are compiled
into ``(ev_offsets, ev_syn, ev_val)`` when the kernel launches, and are scattered at
the top of every tick into ``input_spikes_tensor[row] = [tick, value]``, which
``get_soma_spike`` reads back with one comparison. These tests pin what that must
mean for a user:

- a spike injected for tick t is seen by the synapse step exactly at tick t;
- several spikes on one synapse and tick sum;
- injection order is irrelevant;
- the STDP rule (priority 101) sees the same spike as the synapse step (100);
- injecting between two ``simulate()`` calls is identical to injecting everything
  up front (nothing the kernels learned is lost, the tick counter continues);
- ``reset()`` discards injected spikes; re-injecting reproduces the first run;
- spikes scheduled past the end of a run wait for the next ``simulate()``.

Usage:
    python -m pytest tests/test_input_spike_events.py -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR.parent))

from superneuroabm.model import NeuromorphicModel


def _build(tracking=False, learning_rule=None):
    """external -> soma; returns (model, soma, syn)."""
    model = NeuromorphicModel(enable_internal_states_tracking=tracking)
    model.set_seed(7)
    soma = model.create_soma(breed="lif_soma", config_name="config_0")
    kwargs = {"learning_rule": learning_rule} if learning_rule else {}
    syn = model.create_synapse(breed="single_exp_synapse", pre_soma_id=-1,
                               post_soma_id=soma, config_name="config_0", **kwargs)
    model.setup()
    return model, soma, syn


def _spike_column(model, syn):
    """single_exp writes the spike it received into history column 1 every tick."""
    return np.asarray(model.get_internal_states_history(syn), dtype=np.float32)[:, 1]


def _soma_bits(model, soma):
    return {p: np.array(model.get_agent_property_value(soma, p), dtype=np.float32).view(np.uint32)
            for p in ("internal_states", "output_spikes_tensor", "hyperparameters")}


class TestDelivery(unittest.TestCase):

    def test_exact_ticks_and_same_tick_sum(self):
        model, _soma, syn = _build(tracking=True)
        model.add_spike_list(syn, [[3, 1.0], [7, 1.0], [7, 1.0], [12, 0.25]])
        model.simulate(ticks=20, update_data_ticks=1)
        received = _spike_column(model, syn)
        expected = np.zeros(20, dtype=np.float32)
        expected[3], expected[7], expected[12] = 1.0, 2.0, 0.25
        np.testing.assert_array_equal(received, expected)

    def test_row_readback_is_last_delivery(self):
        """input_spikes_tensor row is [last_delivered_tick, value] after a run."""
        model, _soma, syn = _build()
        model.add_spike_list(syn, [[3, 1.0], [11, 0.5]])
        model.simulate(ticks=20, update_data_ticks=1)
        row = [float(x) for x in model.get_agent_property_value(syn, "input_spikes_tensor")]
        self.assertEqual(row, [11.0, 0.5])

    def test_order_independent(self):
        model_a, soma_a, syn_a = _build(tracking=True)
        model_a.add_spike_list(syn_a, [[12, 1.0], [3, 1.0], [7, 1.0]])
        model_b, soma_b, syn_b = _build(tracking=True)
        model_b.add_spike_list(syn_b, [[3, 1.0], [7, 1.0], [12, 1.0]])
        model_a.simulate(ticks=20, update_data_ticks=1)
        model_b.simulate(ticks=20, update_data_ticks=1)
        np.testing.assert_array_equal(_spike_column(model_a, syn_a), _spike_column(model_b, syn_b))
        for p, bits in _soma_bits(model_a, soma_a).items():
            np.testing.assert_array_equal(bits, _soma_bits(model_b, soma_b)[p], err_msg=p)

    def test_stdp_rule_sees_the_same_spike(self):
        """The learning rule at priority 101 reads the external spike on the same tick."""
        model, _soma, syn = _build(tracking=True, learning_rule="exp_pair_wise_stdp")
        model.add_spike(syn, tick=5, value=1.0)
        model.simulate(ticks=10, update_data_ticks=1)
        pre_trace = np.asarray(model.get_learning_internal_states_history(syn), dtype=np.float32)[:, 0]
        self.assertEqual(float(pre_trace[4]), 0.0)
        self.assertGreater(float(pre_trace[5]), 0.0, "pre-trace did not see the tick-5 spike")

    def test_inject_between_runs_equals_single_shot(self):
        """Adding spikes between simulate() calls neither loses GPU state nor shifts time."""
        model_a, soma_a, syn_a = _build(tracking=True)
        model_a.add_spike_list(syn_a, [[t, 1.0] for t in (2, 5, 8, 11, 105, 108, 111)])
        model_a.simulate(ticks=200, update_data_ticks=1)
        spikes_a = model_a.get_spike_times(soma_a)
        received_a = _spike_column(model_a, syn_a)

        model_b, soma_b, syn_b = _build(tracking=True)
        model_b.add_spike_list(syn_b, [[t, 1.0] for t in (2, 5, 8, 11)])
        model_b.simulate(ticks=100, update_data_ticks=1)
        spikes_b = list(model_b.get_spike_times(soma_b))
        received_b = [_spike_column(model_b, syn_b)]
        model_b.add_spike_list(syn_b, [[t, 1.0] for t in (105, 108, 111)])
        model_b.simulate(ticks=100, update_data_ticks=1)
        spikes_b += model_b.get_spike_times(soma_b)
        received_b.append(_spike_column(model_b, syn_b))

        self.assertGreater(len(spikes_a), 1)
        self.assertEqual(spikes_a, spikes_b)
        np.testing.assert_array_equal(received_a, np.concatenate(received_b))
        for p, bits in _soma_bits(model_a, soma_a).items():
            np.testing.assert_array_equal(bits, _soma_bits(model_b, soma_b)[p], err_msg=p)

    def test_reset_discards_and_reinjection_reproduces(self):
        model, soma, syn = _build()
        train = [[t, 1.0] for t in (2, 5, 8, 11)]
        model.add_spike_list(syn, train)
        model.simulate(ticks=50, update_data_ticks=1)
        first = model.get_spike_times(soma)
        self.assertGreater(len(first), 0)

        model.reset()
        self.assertEqual(model.get_input_spikes(syn)[0].size, 0)
        model.simulate(ticks=50, update_data_ticks=1)
        self.assertEqual(model.get_spike_times(soma), [], "reset must discard injected spikes")

        model.reset()
        model.add_spike_list(syn, train)
        model.simulate(ticks=50, update_data_ticks=1)
        self.assertEqual(model.get_spike_times(soma), first)

    def test_future_spikes_wait_for_the_next_run(self):
        model, soma, syn = _build(tracking=True)
        model.add_spike_list(syn, [[t, 1.0] for t in (52, 55, 58, 61)])
        model.simulate(ticks=50, update_data_ticks=1)
        self.assertEqual(model.get_spike_times(soma), [])
        self.assertEqual(float(_spike_column(model, syn).sum()), 0.0)
        model.simulate(ticks=50, update_data_ticks=1)
        self.assertEqual(float(_spike_column(model, syn).sum()), 4.0)
        self.assertGreater(len(model.get_spike_times(soma)), 0)

    def test_tick_rewind_clears_old_stamps(self):
        """Drivers that set model.tick = 0 per presentation must not see last run's spikes.

        A row keeps [last_delivered_tick, value] after a run; rewinding the clock would
        make that stamp match the same tick number again unless the launch clears it."""
        model, _soma, syn = _build(tracking=True)
        model.add_spike(syn, tick=8, value=1.0)
        model.simulate(ticks=20, update_data_ticks=1)
        self.assertEqual(float(_spike_column(model, syn)[8]), 1.0)

        model.tick = 0
        model.clear_input_spikes()
        model.add_spike(syn, tick=5, value=1.0)
        model.simulate(ticks=20, update_data_ticks=1)
        received = _spike_column(model, syn)
        self.assertEqual(float(received[5]), 1.0)
        self.assertEqual(float(received[8]), 0.0, "stale stamp from the previous presentation was re-delivered")

    def test_clear_input_spikes(self):
        model, soma, syn = _build()
        model.add_spike_list(syn, [[t, 1.0] for t in (2, 5, 8, 11)])
        model.clear_input_spikes()
        model.simulate(ticks=50, update_data_ticks=1)
        self.assertEqual(model.get_spike_times(soma), [])

    def test_recurrent_synapse_unaffected(self):
        """A synapse with a real pre soma never reads the input row."""
        model = NeuromorphicModel()
        s0 = model.create_soma(breed="lif_soma", config_name="config_0")
        s1 = model.create_soma(breed="lif_soma", config_name="config_0")
        ext = model.create_synapse(breed="single_exp_synapse", pre_soma_id=-1,
                                   post_soma_id=s0, config_name="config_0")
        rec = model.create_synapse(breed="single_exp_synapse", pre_soma_id=s0,
                                   post_soma_id=s1, config_name="config_0")
        model.setup()
        model.add_spike_list(ext, [[t, 1.0] for t in (2, 5, 8, 11)])
        model.simulate(ticks=100, update_data_ticks=1)
        self.assertGreater(len(model.get_spike_times(s0)), 0)
        self.assertGreater(len(model.get_spike_times(s1)), 0, "relay through the recurrent synapse")
        row = [float(x) for x in model.get_agent_property_value(rec, "input_spikes_tensor")]
        self.assertEqual(row, [-1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
