"""N-channel synaptic input: one soma-side exponential filter per inbox channel.

A soma whose afferents use K distinct time constants needs K channels in push mode: the
delivery synapse's ``channel`` picks the inbox slot and the psc soma decays slot c with
``tau_syn_c``. Checked against the per-synapse model (lif_soma + single_exp_synapse with
a tau_fall per synapse, pull/clocked), which keeps one filter per synapse and therefore
needs no channels: same spike trains, soma state to float32 round-off. Runs under mpirun
too (tests/run_mpi_tests.sh).
"""

import unittest

import numpy as np

from superneuroabm.model import NeuromorphicModel

# dt is 1e-3: channel 0 is a one-tick pulse (factor 1 - dt/tau = 0), channel 1 lingers.
TAUS = (1e-3, 5e-3)
N = 8


def _drive(model, ext):
    rng = np.random.default_rng(0)
    for s in ext:
        ticks = np.sort(rng.choice(np.arange(2, 70), size=12, replace=False))
        model.add_spike_list(s, [[int(t), 1.0] for t in ticks])


def _per_synapse():
    """pull/clocked reference: the time constant lives on each synapse."""
    m = NeuromorphicModel(synaptic_input="pull", synapse_stepping="clocked",
                          enable_internal_states_tracking=False)
    m.set_seed(7)
    somas = [m.create_soma(breed="lif_soma", config_name="config_0") for _ in range(N)]
    tau = lambda c: {"hyperparameters": {"tau_fall": TAUS[c]}}
    ext = [m.create_synapse(breed="single_exp_synapse", pre_soma_id=-1, post_soma_id=s,
                            config_name="config_0", overrides=tau(1)) for s in somas]
    for i in range(N):
        for j, c in ((1, 0), (3, 1)):
            m.create_synapse(breed="single_exp_synapse", pre_soma_id=somas[i],
                             post_soma_id=somas[(i + j) % N], config_name="config_0",
                             overrides=tau(c))
    m.setup()
    _drive(m, ext)
    return m, somas


def _soma_side(stepping, channels=2):
    """push: delivery synapses pick a channel, the psc soma filters per channel."""
    m = NeuromorphicModel(synaptic_input="push", synapse_stepping=stepping,
                          synaptic_channels=channels, enable_internal_states_tracking=False)
    m.set_seed(7)
    soma_ov = {"hyperparameters": {"tau_syn_0": TAUS[0], "tau_syn_1": TAUS[1]}}
    somas = [m.create_soma(breed="lif_psc_soma", config_name="config_0", overrides=soma_ov)
             for _ in range(N)]
    ch = lambda c: {"hyperparameters": {"channel": c}}
    ext = [m.create_synapse(breed="delivery_synapse", pre_soma_id=-1, post_soma_id=s,
                            config_name="config_0", overrides=ch(1)) for s in somas]
    for i in range(N):
        for j, c in ((1, 0), (3, 1)):
            m.create_synapse(breed="delivery_synapse", pre_soma_id=somas[i],
                             post_soma_id=somas[(i + j) % N], config_name="config_0",
                             overrides=ch(c))
    m.setup()
    _drive(m, ext)
    return m, somas


def _spikes(m, somas):
    return [m.get_spike_times(soma_id=s) for s in somas]


def _v_tlast(m, somas):
    st = np.array([m.get_agent_property_value(s, "internal_states") for s in somas],
                  dtype=np.float32)
    return st[:, [0, 2]]


class TestChannelsMatchPerSynapseFilters(unittest.TestCase):

    def test_event_two_channels_equal_two_taus(self):
        ref, sr = _per_synapse()
        ev, se = _soma_side("event")
        ref.simulate(ticks=100)
        ev.simulate(ticks=100)
        spikes_ref = _spikes(ref, sr)
        self.assertGreater(sum(len(x) for x in spikes_ref), 20, "network must actually fire")
        self.assertEqual(spikes_ref, _spikes(ev, se))
        np.testing.assert_allclose(_v_tlast(ev, se), _v_tlast(ref, sr), rtol=1e-5, atol=1e-5)

    def test_clocked_equals_event_bit_exact(self):
        """Same kernels woken or stepped every tick: identical to the bit."""
        clocked, sc = _soma_side("clocked")
        event, se = _soma_side("event")
        clocked.simulate(ticks=100)
        event.simulate(ticks=100)
        self.assertEqual(_spikes(clocked, sc), _spikes(event, se))
        a = np.array([clocked.get_agent_property_value(s, "internal_states") for s in sc], np.float32)
        b = np.array([event.get_agent_property_value(s, "internal_states") for s in se], np.float32)
        np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32))

    def test_channels_are_distinct(self):
        """Swapping the channels of the recurrent synapses changes the dynamics, so the
        match above is not a coincidence of both channels doing the same thing."""
        ref, sr = _per_synapse()
        m = NeuromorphicModel(synaptic_input="push", synapse_stepping="event",
                              enable_internal_states_tracking=False)
        m.set_seed(7)
        soma_ov = {"hyperparameters": {"tau_syn_0": TAUS[0], "tau_syn_1": TAUS[1]}}
        somas = [m.create_soma(breed="lif_psc_soma", config_name="config_0", overrides=soma_ov)
                 for _ in range(N)]
        ch = lambda c: {"hyperparameters": {"channel": c}}
        ext = [m.create_synapse(breed="delivery_synapse", pre_soma_id=-1, post_soma_id=s,
                                config_name="config_0", overrides=ch(1)) for s in somas]
        for i in range(N):
            for j, c in ((1, 1), (3, 0)):      # swapped
                m.create_synapse(breed="delivery_synapse", pre_soma_id=somas[i],
                                 post_soma_id=somas[(i + j) % N], config_name="config_0",
                                 overrides=ch(c))
        m.setup()
        _drive(m, ext)
        ref.simulate(ticks=100)
        m.simulate(ticks=100)
        self.assertNotEqual(_spikes(ref, sr), _spikes(m, somas))


class TestChannelValidation(unittest.TestCase):

    def test_config_must_declare_every_channel(self):
        m = NeuromorphicModel(synaptic_input="push", synapse_stepping="event",
                              synaptic_channels=3, enable_internal_states_tracking=False)
        m.create_soma(breed="lif_psc_soma", config_name="config_0")   # has tau_syn_0, tau_syn_1
        with self.assertRaisesRegex(ValueError, "tau_syn_2"):
            m.setup()

    def test_channel_override_out_of_range(self):
        m = NeuromorphicModel(synaptic_input="push", synapse_stepping="event",
                              synaptic_channels=1, enable_internal_states_tracking=False)
        s = m.create_soma(breed="lif_psc_soma", config_name="config_0")
        with self.assertRaisesRegex(ValueError, "channel"):
            m.create_synapse(breed="delivery_synapse", pre_soma_id=-1, post_soma_id=s,
                             config_name="config_0", overrides={"hyperparameters": {"channel": 1}})

    def test_channel_config_out_of_range(self):
        import copy
        m = NeuromorphicModel(synaptic_input="push", synapse_stepping="event",
                              synaptic_channels=1, enable_internal_states_tracking=False)
        cfgs = m._component_configurations["synapse"]["delivery_synapse"]
        cfgs["wide"] = copy.deepcopy(cfgs["config_0"])
        cfgs["wide"]["hyperparameters"]["channel"] = 1
        m.create_soma(breed="lif_psc_soma", config_name="config_0")
        with self.assertRaisesRegex(ValueError, "channel"):
            m.setup()

    def test_delivery_config_needs_channel(self):
        import copy
        m = NeuromorphicModel(synaptic_input="push", synapse_stepping="event",
                              enable_internal_states_tracking=False)
        cfgs = m._component_configurations["synapse"]["delivery_synapse"]
        cfgs["old"] = copy.deepcopy(cfgs["config_0"])
        del cfgs["old"]["hyperparameters"]["channel"]
        m.create_soma(breed="lif_psc_soma", config_name="config_0")
        with self.assertRaisesRegex(ValueError, "channel"):
            m.setup()

    def test_one_channel_model_runs(self):
        m = NeuromorphicModel(synaptic_input="push", synapse_stepping="event",
                              synaptic_channels=1, enable_internal_states_tracking=False)
        s = m.create_soma(breed="lif_psc_soma", config_name="config_0")
        ext = m.create_synapse(breed="delivery_synapse", pre_soma_id=-1, post_soma_id=s,
                               config_name="config_0")
        m.setup()
        m.add_spike_list(ext, [[2, 1.0], [3, 1.0], [4, 1.0]])
        m.simulate(ticks=20)
        self.assertGreater(len(m.get_spike_times(soma_id=s)), 0)

    def test_bad_channel_count(self):
        with self.assertRaises(ValueError):
            NeuromorphicModel(synaptic_input="push", synaptic_channels=0)


if __name__ == "__main__":
    unittest.main()
