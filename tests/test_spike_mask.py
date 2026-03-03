"""
Test GPU-side spike recording filter by soma ID subset (spike_mask).
"""

import unittest

from superneuroabm.model import NeuromorphicModel


class TestSpikeMask(unittest.TestCase):
    """Tests that set_recorded_somas filters spike recording on GPU."""

    TAUS = 2.5e-3

    def test_only_target_soma_recorded(self):
        """
        Chain: external -> soma_0 -> soma_1 (via internal synapse).
        Only soma_1 is in the recorded set.

        Both somas fire, but only soma_1 spikes should appear in the record.
        """
        model = NeuromorphicModel(enable_internal_state_tracking=False)

        soma_0 = model.create_soma(breed="hg_lif_soma", config_name="config_0")
        soma_1 = model.create_soma(breed="hg_lif_soma", config_name="config_0")

        # External -> soma_0
        syn_ext = model.create_synapse(
            breed="single_exp_synapse",
            pre_soma_id=-1,
            post_soma_id=soma_0,
            config_name="no_learning_config_0",
            hyperparameters_overrides={"tau_fall": self.TAUS},
        )
        # soma_0 -> soma_1
        syn_int = model.create_synapse(
            breed="single_exp_synapse",
            pre_soma_id=soma_0,
            post_soma_id=soma_1,
            config_name="no_learning_config_0",
            hyperparameters_overrides={"tau_fall": self.TAUS},
        )

        model.setup(use_gpu=True)

        # Only record soma_1
        model.set_recorded_somas([soma_1])

        model.add_spike(synapse_id=syn_ext, tick=2, value=1)
        model.simulate(ticks=200, update_data_ticks=1)

        spikes_0 = model.get_spike_times(soma_id=soma_0)
        spikes_1 = model.get_spike_times(soma_id=soma_1)

        print(f"\nsoma_0 recorded spikes (should be empty): {spikes_0}")
        print(f"soma_1 recorded spikes (should have entries): {spikes_1}")

        self.assertEqual(
            len(spikes_0), 0,
            "soma_0 fired but should NOT be in the spike record (not in recorded set)",
        )
        self.assertGreaterEqual(
            len(spikes_1), 1,
            "soma_1 should fire from chain propagation and be recorded",
        )

    def test_record_all_by_default(self):
        """
        Without calling set_recorded_somas, all somas are recorded (backward compat).
        """
        model = NeuromorphicModel(enable_internal_state_tracking=False)

        soma_0 = model.create_soma(breed="hg_lif_soma", config_name="config_0")
        soma_1 = model.create_soma(breed="hg_lif_soma", config_name="config_0")

        syn_ext = model.create_synapse(
            breed="single_exp_synapse",
            pre_soma_id=-1,
            post_soma_id=soma_0,
            config_name="no_learning_config_0",
            hyperparameters_overrides={"tau_fall": self.TAUS},
        )
        syn_int = model.create_synapse(
            breed="single_exp_synapse",
            pre_soma_id=soma_0,
            post_soma_id=soma_1,
            config_name="no_learning_config_0",
            hyperparameters_overrides={"tau_fall": self.TAUS},
        )

        model.setup(use_gpu=True)
        # No set_recorded_somas call — should record all
        model.add_spike(synapse_id=syn_ext, tick=2, value=1)
        model.simulate(ticks=200, update_data_ticks=1)

        spikes_0 = model.get_spike_times(soma_id=soma_0)
        spikes_1 = model.get_spike_times(soma_id=soma_1)

        print(f"\nDefault recording - soma_0 spikes: {spikes_0}")
        print(f"Default recording - soma_1 spikes: {spikes_1}")

        self.assertGreaterEqual(len(spikes_0), 1, "soma_0 should be recorded by default")
        self.assertGreaterEqual(len(spikes_1), 1, "soma_1 should be recorded by default")

    def test_get_all_spike_times(self):
        """
        Verify get_all_spike_times returns dict keyed by soma ID.
        """
        model = NeuromorphicModel(enable_internal_state_tracking=False)

        soma_0 = model.create_soma(breed="hg_lif_soma", config_name="config_0")
        soma_1 = model.create_soma(breed="hg_lif_soma", config_name="config_0")

        syn_ext = model.create_synapse(
            breed="single_exp_synapse",
            pre_soma_id=-1,
            post_soma_id=soma_0,
            config_name="no_learning_config_0",
            hyperparameters_overrides={"tau_fall": self.TAUS},
        )
        syn_int = model.create_synapse(
            breed="single_exp_synapse",
            pre_soma_id=soma_0,
            post_soma_id=soma_1,
            config_name="no_learning_config_0",
            hyperparameters_overrides={"tau_fall": self.TAUS},
        )

        model.setup(use_gpu=True)
        model.add_spike(synapse_id=syn_ext, tick=2, value=1)
        model.simulate(ticks=200, update_data_ticks=1)

        all_spikes = model.get_all_spike_times()
        print(f"\nget_all_spike_times result: {all_spikes}")

        self.assertIsInstance(all_spikes, dict)
        self.assertIn(soma_0, all_spikes, "soma_0 should appear in all_spike_times")
        self.assertIn(soma_1, all_spikes, "soma_1 should appear in all_spike_times")
        # Verify consistency with get_spike_times
        self.assertEqual(all_spikes[soma_0], model.get_spike_times(soma_0))
        self.assertEqual(all_spikes[soma_1], model.get_spike_times(soma_1))


if __name__ == "__main__":
    unittest.main()
