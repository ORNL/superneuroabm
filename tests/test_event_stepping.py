"""synapse_stepping="event": synapses stepped only after their pre fired.

delivery_synapse + lif_psc_soma in push mode, clocked vs event: the woken synapses
compute exactly what they would have in the clocked run and the idle ones would have
deposited nothing, so spike times and soma state must be bit-identical. Runs under
mpirun too (tests/run_mpi_tests.sh): a pre-soma on another rank raises its event
through the ghost-row exchange.
"""

import unittest

import numpy as np

from superneuroabm.model import NeuromorphicModel

STATE_PROPS = ("internal_states", "output_spikes_tensor")


def _model(stepping):
    m = NeuromorphicModel(synaptic_input="push", synapse_stepping=stepping,
                          enable_internal_states_tracking=False)
    m.set_seed(7)
    return m


def _recurrent(stepping, n=8):
    """n lif_psc somas, one external delivery synapse each, ring + skip-3 recurrence."""
    m = _model(stepping)
    somas = [m.create_soma(breed="lif_psc_soma", config_name="config_0") for _ in range(n)]
    ext = [m.create_synapse(breed="delivery_synapse", pre_soma_id=-1, post_soma_id=s,
                            config_name="config_0") for s in somas]
    for i in range(n):
        for j in (1, 3):
            m.create_synapse(breed="delivery_synapse", pre_soma_id=somas[i],
                             post_soma_id=somas[(i + j) % n], config_name="config_0")
    m.setup()
    rng = np.random.default_rng(0)
    for s in ext:
        ticks = np.sort(rng.choice(np.arange(2, 70), size=12, replace=False))
        m.add_spike_list(s, [[int(t), 1.0] for t in ticks])
    return m, somas


def _recurrent_stdp(stepping, n=8, every_other=False):
    """Same network with pair-wise STDP on every synapse (or on every other recurrent
    one, to exercise the wake-eligibility mask: rule-less synapses are never queued
    for the learning-rule phase)."""
    m = _model(stepping)
    somas = [m.create_soma(breed="lif_psc_soma", config_name="config_0") for _ in range(n)]
    kw = dict(config_name="config_0", learning_rule="exp_pair_wise_stdp",
              learning_rule_config="default")
    plain = dict(config_name="config_0")
    ext = [m.create_synapse(breed="delivery_synapse", pre_soma_id=-1, post_soma_id=s, **kw)
           for s in somas]
    syns = []
    for i in range(n):
        for j in (1, 3):
            k = plain if (every_other and (i + j) % 2) else kw
            syns.append(m.create_synapse(breed="delivery_synapse", pre_soma_id=somas[i],
                                         post_soma_id=somas[(i + j) % n], **k))
    m.setup()
    rng = np.random.default_rng(0)
    for s in ext:
        ticks = np.sort(rng.choice(np.arange(2, 70), size=12, replace=False))
        m.add_spike_list(s, [[int(t), 1.0] for t in ticks])
    return m, somas, ext + syns


def _state(model, somas):
    return {p: np.array([model.get_agent_property_value(s, p) for s in somas],
                        dtype=np.float32) for p in STATE_PROPS}


class TestEventEqualsClocked(unittest.TestCase):

    def test_recurrent_network_bit_exact(self):
        clocked, sc = _recurrent("clocked")
        event, se = _recurrent("event")
        clocked.simulate(ticks=100)
        event.simulate(ticks=100)
        spikes_c = [clocked.get_spike_times(soma_id=s) for s in sc]
        spikes_e = [event.get_spike_times(soma_id=s) for s in se]
        self.assertEqual(spikes_c, spikes_e)
        self.assertGreater(sum(len(x) for x in spikes_c), 20, "network must actually fire")
        a, b = _state(clocked, sc), _state(event, se)
        for p in STATE_PROPS:
            np.testing.assert_array_equal(a[p].view(np.uint32), b[p].view(np.uint32),
                                          err_msg=f"{p} differs between clocked and event")

    def test_resume_simulate(self):
        once, so = _recurrent("event")
        twice, st = _recurrent("event")
        once.simulate(ticks=100)
        chunked = [[] for _ in st]
        for _ in range(10):
            twice.simulate(ticks=10)
            for i, s in enumerate(st):
                chunked[i] += twice.get_spike_times(soma_id=s)
        self.assertEqual([once.get_spike_times(soma_id=s) for s in so], chunked)

    def test_reset_then_rerun(self):
        """reset() between runs (the per-sample inference pattern): the second run must
        equal a fresh model's, so no wake stamp, queue or inbox content survives a reset."""
        fresh, sf = _recurrent("event")
        fresh.simulate(ticks=100)
        reused, sr = _recurrent("event")
        reused.simulate(ticks=60)                       # a different, shorter first run
        reused.reset(retain_parameters=False)
        rng = np.random.default_rng(0)
        ext = [s for s in reused._synapse_ids if reused.get_synapse_connectivity(s)[0] == -1]
        for s in ext:
            ticks = np.sort(rng.choice(np.arange(2, 70), size=12, replace=False))
            reused.add_spike_list(s, [[int(t), 1.0] for t in ticks])
        reused.simulate(ticks=100)
        self.assertEqual([fresh.get_spike_times(soma_id=s) for s in sf],
                         [reused.get_spike_times(soma_id=s) for s in sr])


class TestEventLearning(unittest.TestCase):

    def test_stdp_event_matches_clocked(self):
        """Event-woken rules catch traces up in closed form: same spikes, weights to
        float32 round-off (a (1-dt/tau)^n power versus n multiplies)."""
        clocked, sc, syn_c = _recurrent_stdp("clocked")
        event, se, syn_e = _recurrent_stdp("event")
        clocked.simulate(ticks=100)
        event.simulate(ticks=100)
        self.assertEqual([clocked.get_spike_times(soma_id=s) for s in sc],
                         [event.get_spike_times(soma_id=s) for s in se])
        wc = np.array([clocked.get_agent_property_value(s, "hyperparameters")[0] for s in syn_c])
        we = np.array([event.get_agent_property_value(s, "hyperparameters")[0] for s in syn_e])
        self.assertGreater(float(np.abs(wc - 14.0).max()), 0.0, "weights must have changed")
        np.testing.assert_allclose(we, wc, rtol=1e-5, atol=1e-5)

    def test_mixed_rule_and_plain_synapses(self):
        """With the eligibility mask only rule-carrying synapses are woken for the rule
        phase; results still match the clocked run, and plain synapses keep their weight."""
        clocked, sc, syn_c = _recurrent_stdp("clocked", every_other=True)
        event, se, syn_e = _recurrent_stdp("event", every_other=True)
        clocked.simulate(ticks=100)
        event.simulate(ticks=100)
        self.assertEqual([clocked.get_spike_times(soma_id=s) for s in sc],
                         [event.get_spike_times(soma_id=s) for s in se])
        wc = np.array([clocked.get_agent_property_value(s, "hyperparameters")[0] for s in syn_c])
        we = np.array([event.get_agent_property_value(s, "hyperparameters")[0] for s in syn_e])
        np.testing.assert_allclose(we, wc, rtol=1e-5, atol=1e-5)
        plain = [s for s in syn_e if event._combo_of(s).learning_rule is None]
        self.assertTrue(plain and all(event.get_agent_property_value(s, "hyperparameters")[0] == 14.0
                                      for s in plain))
        self.assertGreater(float(np.abs(wc - 14.0).max()), 0.0)

    def test_rules_must_be_event_capable(self):
        """A rule not known to catch its traces up is refused in event mode."""
        m = _model("event")
        m._learning_rules[99] = {"func_name": "my_rule", "import_line": "",
                                 "sys_path_entry": None, "event_capable": False}
        m.create_soma(breed="lif_psc_soma", config_name="config_0")
        with self.assertRaises(ValueError):
            m.setup()


class TestEventValidation(unittest.TestCase):

    def test_event_needs_push(self):
        with self.assertRaises(ValueError):
            NeuromorphicModel(synaptic_input="pull", synapse_stepping="event")

    def test_stateful_kinds_are_excluded(self):
        m = _model("event")
        self.assertNotIn("single_exp_synapse", m._synapse_breeds)
        self.assertIn("delivery_synapse", m._synapse_breeds)

    def test_custom_type_must_be_event_capable(self):
        from pathlib import Path
        from superneuroabm.step_functions.synapse import delivery as d
        m = _model("event")
        with self.assertRaises(ValueError):
            m.register_synapse_type("mine", d.delivery_synapse_step_func, Path(d.__file__))
        m.register_synapse_type("mine", d.delivery_synapse_step_func, Path(d.__file__),
                                event_capable=True)


if __name__ == "__main__":
    unittest.main()
