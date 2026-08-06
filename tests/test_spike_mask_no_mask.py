"""
Test the same 2-soma chain topology as test_spike_mask but WITHOUT
model.set_recorded_somas(), to isolate whether the MPI hang comes from
the spike-mask feature or from the agent count / distribution.
"""

import unittest

import matplotlib
matplotlib.use('Agg')

from superneuroabm.model import NeuromorphicModel

try:
    from util import vizualize_responses
except ImportError:
    from tests.util import vizualize_responses


class TestSpikeMaskNoMask(unittest.TestCase):
    """Same topology as TestSpikeMask but records all somas (default)."""

    def _build_chain(self):
        """
        Chain: external -> soma_0 -> soma_1 (via internal synapse).
        Returns (model, soma_0, soma_1, syn_ext).
        """
        model = NeuromorphicModel(enable_internal_states_tracking=True)

        soma_0 = model.create_soma(breed="lif_soma", config_name="config_0")
        soma_1 = model.create_soma(breed="lif_soma", config_name="config_0")

        syn_ext = model.create_synapse(
            breed="single_exp_synapse",
            pre_soma_id=-1,
            post_soma_id=soma_0,
            config_name="config_0",
        )
        syn_int = model.create_synapse(
            breed="single_exp_synapse",
            pre_soma_id=soma_0,
            post_soma_id=soma_1,
            config_name="config_0",
        )

        model.setup()
        return model, soma_0, soma_1, syn_ext

    def test_chain_both_somas_recorded(self):
        """Both somas fire and both should appear in the spike record."""
        model, soma_0, soma_1, syn_ext = self._build_chain()

        model.add_spike(synapse_id=syn_ext, tick=2, value=1)
        model.simulate(ticks=200, update_data_ticks=1)

        vizualize_responses(model, vthr=-45, fig_name="test_chain_both_somas_recorded.png")

        spikes_0 = model.get_spike_times(soma_id=soma_0)
        spikes_1 = model.get_spike_times(soma_id=soma_1)

        print(f"\nsoma_0 spikes: {spikes_0}")
        print(f"soma_1 spikes: {spikes_1}")

        self.assertGreaterEqual(len(spikes_0), 1, "soma_0 should fire")
        self.assertGreaterEqual(len(spikes_1), 1, "soma_1 should fire from chain propagation")


if __name__ == "__main__":
    unittest.main()
