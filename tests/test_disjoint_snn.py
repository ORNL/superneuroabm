"""
Bit-exact test: two disjoint SNNs must produce identical results
whether run alone or together in the same model.

This tests that SAGESim's multi-breed handling doesn't cause
cross-network interference through shared property tensors.
"""

import unittest
import numpy as np
from superneuroabm.model import NeuromorphicModel


def build_network_alone(seed=42):
    """Build a single 3-soma network: input -> soma0 -> soma1 -> soma2."""
    m = NeuromorphicModel()
    m.set_seed(seed)

    soma_0 = m.create_soma(breed="lif_soma", config_name="config_0")
    soma_1 = m.create_soma(breed="lif_soma", config_name="config_0")
    soma_2 = m.create_soma(breed="lif_soma", config_name="config_0")

    syn_in = m.create_synapse(
        breed="single_exp_synapse", pre_soma_id=-1,
        post_soma_id=soma_0, config_name="config_0")
    syn_01 = m.create_synapse(
        breed="single_exp_synapse", pre_soma_id=soma_0,
        post_soma_id=soma_1, config_name="config_0")
    syn_12 = m.create_synapse(
        breed="single_exp_synapse", pre_soma_id=soma_1,
        post_soma_id=soma_2, config_name="config_0")

    m.setup()
    m.add_spike(synapse_id=syn_in, tick=2, value=1)
    m.add_spike(synapse_id=syn_in, tick=50, value=1)
    return m, [soma_0, soma_1, soma_2]


def build_two_networks(seed=42):
    """Build two disjoint 3-soma networks in the same model."""
    m = NeuromorphicModel()
    m.set_seed(seed)

    # Network A
    a0 = m.create_soma(breed="lif_soma", config_name="config_0")
    a1 = m.create_soma(breed="lif_soma", config_name="config_0")
    a2 = m.create_soma(breed="lif_soma", config_name="config_0")
    syn_a_in = m.create_synapse(
        breed="single_exp_synapse", pre_soma_id=-1,
        post_soma_id=a0, config_name="config_0")
    syn_a01 = m.create_synapse(
        breed="single_exp_synapse", pre_soma_id=a0,
        post_soma_id=a1, config_name="config_0")
    syn_a12 = m.create_synapse(
        breed="single_exp_synapse", pre_soma_id=a1,
        post_soma_id=a2, config_name="config_0")

    # Network B (completely disconnected from A)
    b0 = m.create_soma(breed="lif_soma", config_name="config_0")
    b1 = m.create_soma(breed="lif_soma", config_name="config_0")
    b2 = m.create_soma(breed="lif_soma", config_name="config_0")
    syn_b_in = m.create_synapse(
        breed="single_exp_synapse", pre_soma_id=-1,
        post_soma_id=b0, config_name="config_0")
    syn_b01 = m.create_synapse(
        breed="single_exp_synapse", pre_soma_id=b0,
        post_soma_id=b1, config_name="config_0")
    syn_b12 = m.create_synapse(
        breed="single_exp_synapse", pre_soma_id=b1,
        post_soma_id=b2, config_name="config_0")

    m.setup()
    # Same spikes to network A
    m.add_spike(synapse_id=syn_a_in, tick=2, value=1)
    m.add_spike(synapse_id=syn_a_in, tick=50, value=1)
    # Different spikes to network B
    m.add_spike(synapse_id=syn_b_in, tick=10, value=1)
    m.add_spike(synapse_id=syn_b_in, tick=80, value=1)

    return m, [a0, a1, a2], [b0, b1, b2]


class TestDisjointSNN(unittest.TestCase):
    """Two disjoint SNNs in one model must produce bit-identical results
    to running each network alone."""

    TICKS = 200

    def test_network_a_bit_exact(self):
        """Network A alone must bit-match network A in combined model."""
        m_alone, somas_alone = build_network_alone(seed=42)
        m_combined, somas_a, somas_b = build_two_networks(seed=42)

        m_alone.simulate(ticks=self.TICKS, update_data_ticks=1)
        m_combined.simulate(ticks=self.TICKS, update_data_ticks=1)

        # Compare each soma's internal_states and output_spikes bit-exact
        for i in range(3):
            for prop in ["internal_states", "output_spikes_tensor", "hyperparameters"]:
                val_alone = np.array(
                    m_alone.get_agent_property_value(somas_alone[i], prop),
                    dtype=np.float32)
                val_combined = np.array(
                    m_combined.get_agent_property_value(somas_a[i], prop),
                    dtype=np.float32)
                np.testing.assert_array_equal(
                    val_alone.view(np.uint32), val_combined.view(np.uint32),
                    err_msg=f"soma[{i}] {prop}: bit-differs alone vs combined")

        # Compare spike times
        for i in range(3):
            spikes_alone = m_alone.get_spike_times(soma_id=somas_alone[i])
            spikes_combined = m_combined.get_spike_times(soma_id=somas_a[i])
            self.assertEqual(spikes_alone, spikes_combined,
                msg=f"soma[{i}] spike times differ: alone={spikes_alone} combined={spikes_combined}")


class TestResumeSimulateSNN(unittest.TestCase):
    """simulate(N) once must equal simulate(k) repeated N/k times for SNN."""

    TICKS = 100

    def _build_model(self):
        m = NeuromorphicModel()
        m.set_seed(42)
        s0 = m.create_soma(breed="lif_soma", config_name="config_0")
        s1 = m.create_soma(breed="lif_soma", config_name="config_0")
        syn_in = m.create_synapse(
            breed="single_exp_synapse", pre_soma_id=-1,
            post_soma_id=s0, config_name="config_0")
        syn_01 = m.create_synapse(
            breed="single_exp_synapse", pre_soma_id=s0,
            post_soma_id=s1, config_name="config_0")
        m.setup()
        m.add_spike(synapse_id=syn_in, tick=2, value=1)
        m.add_spike(synapse_id=syn_in, tick=50, value=1)
        return m, [s0, s1]

    def _get_state(self, model, somas):
        vals = {}
        for i, sid in enumerate(somas):
            for prop in ["internal_states", "output_spikes_tensor"]:
                v = np.array(model.get_agent_property_value(sid, prop), dtype=np.float32)
                vals[f"soma{i}_{prop}"] = v
        return vals

    def test_100_vs_10x10(self):
        """simulate(100) once vs simulate(10) x 10."""
        m1, somas1 = self._build_model()
        m2, somas2 = self._build_model()

        m1.simulate(ticks=self.TICKS, update_data_ticks=1)
        for _ in range(10):
            m2.simulate(ticks=10, update_data_ticks=1)

        s1 = self._get_state(m1, somas1)
        s2 = self._get_state(m2, somas2)
        for key in s1:
            np.testing.assert_array_equal(
                s1[key].view(np.uint32), s2[key].view(np.uint32),
                err_msg=f"{key}: simulate(100) vs simulate(10)x10")

    def test_100_vs_1x100(self):
        """simulate(100) once vs simulate(1) x 100."""
        m1, somas1 = self._build_model()
        m2, somas2 = self._build_model()

        m1.simulate(ticks=self.TICKS, update_data_ticks=1)
        for _ in range(self.TICKS):
            m2.simulate(ticks=1, update_data_ticks=1)

        s1 = self._get_state(m1, somas1)
        s2 = self._get_state(m2, somas2)
        for key in s1:
            np.testing.assert_array_equal(
                s1[key].view(np.uint32), s2[key].view(np.uint32),
                err_msg=f"{key}: simulate(100) vs simulate(1)x100")


if __name__ == "__main__":
    unittest.main()
