"""N-channel synaptic input: one soma-side exponential filter per inbox channel.

A soma whose afferents use K distinct time constants needs K channels: the delivery
synapse's ``channel`` picks the inbox slot and the soma decays slot c with ``tau_syn_c``.
Pinned here: each channel decays with its own constant (against the closed form), the
channels superpose, the same kernels stepped every tick or event-woken agree to the bit,
swapping channels changes the dynamics, and the refusals. Runs under mpirun too
(tests/run_mpi_tests.sh).
"""

import unittest

import numpy as np

from superneuroabm.model import NeuromorphicModel

DT = 1e-3
# dt is 1e-3: channel 0 lingers 10 ticks, channel 1 is a one-tick pulse (1 - dt/tau = 0)
TAUS = (1e-2, 1e-3)
N = 8
WEIGHT = 14.0
QUIET = {"hyperparameters": {"vthr": 1e9, "tau_syn_0": TAUS[0], "tau_syn_1": TAUS[1]}}  # never fires


def _drive(model, ext):
    rng = np.random.default_rng(0)
    for s in ext:
        ticks = np.sort(rng.choice(np.arange(2, 70), size=12, replace=False))
        model.add_spike_list(s, [[int(t), 1.0] for t in ticks])


def _soma_side(stepping, channels=2):
    """Recurrent net: external synapses on channel 1, ring on channel 0, skip-3 on 1."""
    m = NeuromorphicModel(synapse_stepping=stepping, synaptic_channels=channels,
                          enable_internal_states_tracking=False)
    m.set_seed(7)
    soma_ov = {"hyperparameters": {"tau_syn_0": TAUS[0], "tau_syn_1": TAUS[1]}}
    somas = [m.create_soma(breed="lif_soma", config_name="config_0", overrides=soma_ov)
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


def _current_history(channels_spikes, ticks=30):
    """One quiet soma driven by external synapses; channels_spikes maps a channel to
    the ticks its synapse fires on. Returns the soma's recorded summed current."""
    m = NeuromorphicModel(enable_internal_states_tracking=True)
    soma = m.create_soma(breed="lif_soma", config_name="config_0", overrides=QUIET)
    syns = {c: m.create_synapse(breed="delivery_synapse", pre_soma_id=-1, post_soma_id=soma,
                                config_name="config_0",
                                overrides={"hyperparameters": {"channel": c}})
            for c in channels_spikes}
    m.setup()
    for c, ticks_c in channels_spikes.items():
        m.add_spike_list(syns[c], [[int(t), 1.0] for t in ticks_c])
    m.simulate(ticks=ticks, update_data_ticks=1)
    return np.asarray(m.get_internal_states_history(soma), dtype=np.float64)[:, 3]


def _spikes(m, somas):
    return [m.get_spike_times(soma_id=s) for s in somas]


class TestChannelFilters(unittest.TestCase):

    def test_each_channel_decays_with_its_own_tau(self):
        """A spike at tick 3 is deposited at tick 3 and enters the soma's current at
        tick 4: I(4 + n) = w * (1 - dt/tau_c)^n for the channel it arrived on."""
        for c, tau in enumerate(TAUS):
            I = _current_history({c: [3]})
            n = np.arange(0, 20)
            expected = WEIGHT * (1.0 - DT / tau) ** n
            np.testing.assert_allclose(I[4:24], expected, rtol=1e-5, atol=1e-6,
                                       err_msg=f"channel {c}")
            self.assertTrue(np.all(I[:4] == 0.0))

    def test_channels_superpose(self):
        both = _current_history({0: [3, 9], 1: [5, 9, 15]})
        only0 = _current_history({0: [3, 9]})
        only1 = _current_history({1: [5, 9, 15]})
        np.testing.assert_allclose(both, only0 + only1, rtol=1e-5, atol=1e-6)
        self.assertGreater(float(np.abs(only1).max()), 0.0)

    def test_clocked_equals_event_bit_exact(self):
        """Same kernels woken or stepped every tick: identical to the bit."""
        clocked, sc = _soma_side("clocked")
        event, se = _soma_side("event")
        clocked.simulate(ticks=100)
        event.simulate(ticks=100)
        spikes = _spikes(clocked, sc)
        self.assertGreater(sum(len(x) for x in spikes), 20, "network must actually fire")
        self.assertEqual(spikes, _spikes(event, se))
        a = np.array([clocked.get_agent_property_value(s, "internal_states") for s in sc], np.float32)
        b = np.array([event.get_agent_property_value(s, "internal_states") for s in se], np.float32)
        np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32))

    def test_channels_are_distinct(self):
        """Swapping the recurrent synapses' channels changes the dynamics."""
        ref, sr = _soma_side("event")
        m = NeuromorphicModel(enable_internal_states_tracking=False)
        m.set_seed(7)
        soma_ov = {"hyperparameters": {"tau_syn_0": TAUS[0], "tau_syn_1": TAUS[1]}}
        somas = [m.create_soma(breed="lif_soma", config_name="config_0", overrides=soma_ov)
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
        m = NeuromorphicModel(synaptic_channels=3, enable_internal_states_tracking=False)
        m.create_soma(breed="lif_soma", config_name="config_0")   # has tau_syn_0, tau_syn_1
        with self.assertRaisesRegex(ValueError, "tau_syn_2"):
            m.setup()

    def test_channel_override_out_of_range(self):
        m = NeuromorphicModel(synaptic_channels=1, enable_internal_states_tracking=False)
        s = m.create_soma(breed="lif_soma", config_name="config_0")
        with self.assertRaisesRegex(ValueError, "channel"):
            m.create_synapse(breed="delivery_synapse", pre_soma_id=-1, post_soma_id=s,
                             config_name="config_0", overrides={"hyperparameters": {"channel": 1}})

    def test_channel_config_out_of_range(self):
        import copy
        m = NeuromorphicModel(synaptic_channels=1, enable_internal_states_tracking=False)
        cfgs = m._component_configurations["synapse"]["delivery_synapse"]
        cfgs["wide"] = copy.deepcopy(cfgs["config_0"])
        cfgs["wide"]["hyperparameters"]["channel"] = 1
        m.create_soma(breed="lif_soma", config_name="config_0")
        with self.assertRaisesRegex(ValueError, "channel"):
            m.setup()

    def test_delivery_config_needs_channel(self):
        import copy
        m = NeuromorphicModel(enable_internal_states_tracking=False)
        cfgs = m._component_configurations["synapse"]["delivery_synapse"]
        cfgs["old"] = copy.deepcopy(cfgs["config_0"])
        del cfgs["old"]["hyperparameters"]["channel"]
        m.create_soma(breed="lif_soma", config_name="config_0")
        with self.assertRaisesRegex(ValueError, "channel"):
            m.setup()

    def test_one_channel_model_runs(self):
        m = NeuromorphicModel(synaptic_channels=1, enable_internal_states_tracking=False)
        s = m.create_soma(breed="lif_soma", config_name="config_0")
        ext = m.create_synapse(breed="delivery_synapse", pre_soma_id=-1, post_soma_id=s,
                               config_name="config_0")
        m.setup()
        m.add_spike_list(ext, [[2, 1.0], [3, 1.0], [4, 1.0]])
        m.simulate(ticks=20)
        self.assertGreater(len(m.get_spike_times(soma_id=s)), 0)

    def test_bad_channel_count(self):
        with self.assertRaises(ValueError):
            NeuromorphicModel(synaptic_channels=0)


if __name__ == "__main__":
    unittest.main()
