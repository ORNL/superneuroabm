"""The step functions merged from origin/new-step-functions on the single input path:
the canonical Izhikevich soma, the AdEx soma, the Pfister-Gerstner triplet STDP rule and
the Froemke-Dan suppression STDP rule.

Somas read the inbox through synaptic_current; both rules decay their traces on wake, so
they run event-woken (the default). Pinned: each soma fires under its documented drive
and under delivered synaptic input; each rule changes weights; stepped every tick or
woken, the same rule gives the same spikes and weights to float32 round-off; the
built-in rule ids stay stable (0-3 unchanged, 4 and 5 new).
"""

import unittest

import numpy as np

from superneuroabm.model import NeuromorphicModel, _default_learning_rules


def _drive(model, ext, seed=0):
    rng = np.random.default_rng(seed)
    for s in ext:
        ticks = np.sort(rng.choice(np.arange(2, 70), size=12, replace=False))
        model.add_spike_list(s, [[int(t), 1.0] for t in ticks])


def _recurrent(stepping, soma_breed, soma_config, rule, weight, n=8, soma_overrides=None):
    """n somas, one external delivery synapse each, ring + skip-3 recurrence with `rule`."""
    m = NeuromorphicModel(synapse_stepping=stepping, enable_internal_states_tracking=False)
    m.set_seed(7)
    somas = [m.create_soma(breed=soma_breed, config_name=soma_config, overrides=soma_overrides)
             for _ in range(n)]
    hp = {"hyperparameters": {"weight": weight}}
    ext = [m.create_synapse(breed="delivery_synapse", pre_soma_id=-1, post_soma_id=s,
                            config_name="config_0", overrides=hp) for s in somas]
    syns = []
    for i in range(n):
        for j in (1, 3):
            syns.append(m.create_synapse(breed="delivery_synapse", pre_soma_id=somas[i],
                                         post_soma_id=somas[(i + j) % n], config_name="config_0",
                                         learning_rule=rule, learning_rule_config="default",
                                         overrides=hp))
    m.setup()
    _drive(m, ext)
    return m, somas, syns


class TestMergedSomas(unittest.TestCase):

    def test_izh_tonic_spiking_under_its_panel_current(self):
        """Figure 1A: a = 0.02, b = 0.2, c = -65, d = 6 with I = 14 spikes tonically."""
        m = NeuromorphicModel(enable_internal_states_tracking=False)
        s = m.create_soma(breed="izh_soma", config_name="config_0",
                          overrides={"hyperparameters": {"I_in": 14.0}})
        m.setup()
        m.simulate(ticks=300)
        spikes = m.get_spike_times(soma_id=s)
        self.assertGreater(len(spikes), 3, f"tonic spiking expected, got {spikes}")
        isi = np.diff(spikes)[2:]            # after the start-up transient from (v, u) = (-70, -14)
        self.assertLess(float(isi.std()), 3.0, f"tonic means regular ISIs, got {np.diff(spikes)}")

    def test_izh_silent_without_drive(self):
        m = NeuromorphicModel(enable_internal_states_tracking=False)
        s = m.create_soma(breed="izh_soma", config_name="config_0")
        m.setup()
        m.simulate(ticks=200)
        self.assertEqual(m.get_spike_times(soma_id=s), [])

    def test_izh_fires_on_delivered_input(self):
        m = NeuromorphicModel(enable_internal_states_tracking=False)
        s = m.create_soma(breed="izh_soma", config_name="config_0")
        ext = m.create_synapse(breed="delivery_synapse", pre_soma_id=-1, post_soma_id=s,
                               config_name="config_0", overrides={"hyperparameters": {"weight": 30.0}})
        m.setup()
        m.add_spike_list(ext, [[t, 1.0] for t in range(5, 60, 2)])
        m.simulate(ticks=100)
        self.assertGreater(len(m.get_spike_times(soma_id=s)), 0)

    def test_adex_tonic_spiking_under_step_current(self):
        """Table 6.1 tonic spiking: 65 pA step drives config_0 to fire regularly."""
        m = NeuromorphicModel(enable_internal_states_tracking=False)
        s = m.create_soma(breed="adex_soma", config_name="config_0",
                          overrides={"hyperparameters": {"I_in": 65.0}})
        m.set_global_property_value("dt", 1e-4)
        m.setup()
        m.simulate(ticks=3000)                      # 300 ms
        spikes = m.get_spike_times(soma_id=s)
        self.assertGreater(len(spikes), 3, f"tonic spiking expected, got {spikes}")

    def test_adex_silent_without_drive(self):
        m = NeuromorphicModel(enable_internal_states_tracking=False)
        s = m.create_soma(breed="adex_soma", config_name="config_0")
        m.setup()
        m.simulate(ticks=200)
        self.assertEqual(m.get_spike_times(soma_id=s), [])

    def test_adex_fires_on_delivered_input(self):
        m = NeuromorphicModel(enable_internal_states_tracking=False)
        s = m.create_soma(breed="adex_soma", config_name="config_0")
        ext = m.create_synapse(breed="delivery_synapse", pre_soma_id=-1, post_soma_id=s,
                               config_name="config_0", overrides={"hyperparameters": {"weight": 200.0}})
        m.setup()
        m.add_spike_list(ext, [[t, 1.0] for t in range(5, 60, 2)])
        m.simulate(ticks=100)
        self.assertGreater(len(m.get_spike_times(soma_id=s)), 0)


class TestMergedRules(unittest.TestCase):

    def test_builtin_ids(self):
        rules = _default_learning_rules()
        self.assertEqual(rules[0]["func_name"], "exp_pair_wise_stdp")
        self.assertEqual(rules[4]["func_name"], "triplet_stdp")
        self.assertEqual(rules[5]["func_name"], "suppression_stdp")

    def _rule_learns_and_is_event_capable(self, rule):
        clocked, sc, syn_c = _recurrent("clocked", "lif_soma", "config_0", rule, 14.0)
        event, se, syn_e = _recurrent("event", "lif_soma", "config_0", rule, 14.0)
        clocked.simulate(ticks=100)
        event.simulate(ticks=100)
        spikes = [clocked.get_spike_times(soma_id=s) for s in sc]
        self.assertGreater(sum(len(x) for x in spikes), 20, "network must actually fire")
        self.assertEqual(spikes, [event.get_spike_times(soma_id=s) for s in se])
        wc = np.array([clocked.get_agent_property_value(s, "hyperparameters")[0] for s in syn_c])
        we = np.array([event.get_agent_property_value(s, "hyperparameters")[0] for s in syn_e])
        self.assertGreater(float(np.abs(wc - 14.0).max()), 0.0, f"{rule} must change weights")
        np.testing.assert_allclose(we, wc, rtol=1e-4, atol=1e-5)

    def test_triplet_stdp(self):
        self._rule_learns_and_is_event_capable("triplet_stdp")

    def test_suppression_stdp(self):
        self._rule_learns_and_is_event_capable("suppression_stdp")

    def test_rules_on_izh_and_adex(self):
        """The merged somas carry the merged rules on the event path."""
        for breed, cfg, weight, ov in (("izh_soma", "config_0", 30.0, {"hyperparameters": {"I_in": 5.0}}),
                                       ("adex_soma", "config_0", 200.0, {"hyperparameters": {"I_in": 40.0}})):
            m, somas, syns = _recurrent("event", breed, cfg, "triplet_stdp", weight, soma_overrides=ov)
            m.simulate(ticks=150)
            self.assertGreater(sum(len(m.get_spike_times(soma_id=s)) for s in somas), 0, breed)
            w = np.array([m.get_agent_property_value(s, "hyperparameters")[0] for s in syns])
            self.assertGreater(float(np.abs(w - weight).max()), 0.0, f"{breed}: weights unchanged")


if __name__ == "__main__":
    unittest.main()
