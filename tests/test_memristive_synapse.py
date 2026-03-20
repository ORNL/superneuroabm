
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


class TestMemristiveSynapse(unittest.TestCase):
    """Tests for the Hathway-Goodman LIF soma with single_exp synapse."""


    def _make_model(self):
        return NeuromorphicModel(enable_internal_state_tracking=True)

    def test_single_spike_response(self):

        model = self._make_model()

        soma = model.create_soma(breed="lif_soma", config_name="config_0")
        synapse = model.create_synapse(
            breed="single_exp_synapse",
            pre_soma_id=-1,
            post_soma_id=soma,
            config_name="config_0",
            learning_rule="memristive_exp_pair_wise_stdp",
        )

        model.setup(use_gpu=True)
        model.add_spike(synapse_id=synapse, tick=2, value=1)
        model.simulate(ticks=200, update_data_ticks=1)

        spike_times = model.get_spike_times(soma_id=soma)
        print(f"\nSingle spike response - soma spike times: {spike_times}")

        caller_name = inspect.stack()[0].function
        vizualize_responses(model, vthr=10, fig_name=f"{caller_name}_memristive.png")

        self.assertGreaterEqual(
            len(spike_times), 1,
            "Soma should fire at least once from a single input spike",
        )



if __name__ == "__main__":
    unittest.main()
