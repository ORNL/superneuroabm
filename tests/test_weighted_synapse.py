import unittest
import inspect
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

from superneuroabm.model import NeuromorphicModel

try:
    from util import vizualize_responses
except ImportError:
    from tests.util import vizualize_responses

class TestWeightedSynapse(unittest.TestCase):
    """Tests for the simple weighted synapse and lif soma."""


    def _make_model(self):
        return NeuromorphicModel(enable_internal_state_tracking=True)

    def test_single_spike_response_with_wtd_synaspe(self):

        model = self._make_model()

        soma0 = model.create_soma(breed="lif_soma", config_name="config_0")
        soma1 = model.create_soma(breed="lif_soma", config_name="config_0")

        synapse0 = model.create_synapse(
            breed="weighted_synapse",
            pre_soma_id=-1,
            post_soma_id=soma0,
            config_name="no_learning_config_0",
        )
        synapse1 = model.create_synapse(
            breed="weighted_synapse",
            pre_soma_id=soma0,
            post_soma_id=soma1,
            config_name="no_learning_config_0",
        )

        model.setup(use_gpu=True)
        model.add_spike(synapse_id=synapse0, tick=2, value=1)
        model.simulate(ticks=10, update_data_ticks=1)

        spike_times = model.get_spike_times(soma_id=soma0)
        print(f"\nSingle spike response - soma 0 spike times: {spike_times}")
        spike_times = model.get_spike_times(soma_id=soma1)
        print(f"\nSingle spike response - soma spike times: {spike_times}")

        caller_name = inspect.stack()[0].function
        vizualize_responses(model, vthr=10, fig_name=f"{caller_name}_weighted.png")

        # self.assertGreaterEqual(
        #     len(spike_times), 1,
        #     "Soma should fire at least once from a single input spike",
        # )



if __name__ == "__main__":
    unittest.main()


