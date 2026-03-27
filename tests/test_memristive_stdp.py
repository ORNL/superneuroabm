
import unittest
import inspect

import matplotlib
matplotlib.use('Agg')

from superneuroabm.model import NeuromorphicModel

try:
    from util import vizualize_responses
except ImportError:
    from tests.util import vizualize_responses


class TestMemristiveSTDP(unittest.TestCase):
    """Tests for memristive exponential pair-wise STDP learning rule."""

    def _make_model(self):
        return NeuromorphicModel(enable_internal_state_tracking=True)

    def test_single_spike_response(self):
        """A single input spike should propagate through the memristive STDP synapse and cause the soma to fire."""
        model = self._make_model()

        soma = model.create_soma(breed="lif_soma", config_name="config_0")
        synapse = model.create_synapse(
            breed="single_exp_synapse",
            pre_soma_id=-1,
            post_soma_id=soma,
            config_name="config_0",
            learning_rule="memristive_exp_pair_wise_stdp",
        )

        model.setup()
        model.add_spike(synapse_id=synapse, tick=2, value=1)
        model.simulate(ticks=200, update_data_ticks=1)

        spike_times = model.get_spike_times(soma_id=soma)
        print(f"\nSingle spike response - soma spike times: {spike_times}")

        caller_name = inspect.stack()[0].function
        vizualize_responses(model, vthr=10, fig_name=f"{caller_name}_memristive_stdp.png")

        self.assertGreaterEqual(
            len(spike_times), 1,
            "Soma should fire at least once from a single input spike",
        )

    def test_stdp_learning_changes_weight(self):
        """STDP learning should modify the synapse weight from its initial value."""
        model = self._make_model()

        soma_pre = model.create_soma(breed="lif_soma", config_name="config_0")
        soma_post = model.create_soma(breed="lif_soma", config_name="config_0")

        synapse_input = model.create_synapse(
            breed="single_exp_synapse",
            pre_soma_id=-1,
            post_soma_id=soma_pre,
            config_name="config_0",
        )

        synapse_stdp = model.create_synapse(
            breed="single_exp_synapse",
            pre_soma_id=soma_pre,
            post_soma_id=soma_post,
            config_name="config_0",
            learning_rule="memristive_exp_pair_wise_stdp",
        )

        model.setup(use_gpu=True)

        initial_weight = model.get_agent_property_value(
            id=synapse_stdp, property_name="hyperparameters"
        )[0]
        self.assertEqual(initial_weight, 14.0, "Initial weight should be 14.0")

        for tick in [10, 30, 50, 70, 90]:
            model.add_spike(synapse_id=synapse_input, tick=tick, value=1.0)

        model.simulate(ticks=200)

        learned_weight = model.get_agent_property_value(
            id=synapse_stdp, property_name="hyperparameters"
        )[0]

        self.assertNotEqual(
            learned_weight, initial_weight,
            f"Memristive STDP should have changed weight from {initial_weight}, but it's still {learned_weight}",
        )
        print(f"Initial weight: {initial_weight}, Learned weight: {learned_weight}")

    def test_weight_bounded(self):
        """Weight should stay within [wmin, wmax] even under heavy stimulation."""
        model = self._make_model()

        soma_pre = model.create_soma(breed="lif_soma", config_name="config_0")
        soma_post = model.create_soma(breed="lif_soma", config_name="config_0")

        synapse_input = model.create_synapse(
            breed="single_exp_synapse",
            pre_soma_id=-1,
            post_soma_id=soma_pre,
            config_name="config_0",
        )

        synapse_stdp = model.create_synapse(
            breed="single_exp_synapse",
            pre_soma_id=soma_pre,
            post_soma_id=soma_post,
            config_name="config_0",
            learning_rule="memristive_exp_pair_wise_stdp",
        )

        model.setup(use_gpu=True)

        for tick in range(5, 500, 5):
            model.add_spike(synapse_id=synapse_input, tick=tick, value=1.0)

        model.simulate(ticks=500, update_data_ticks=500)

        learned_weight = model.get_agent_property_value(
            id=synapse_stdp, property_name="hyperparameters"
        )[0]

        wmin = 0.0
        wmax = 24.0
        self.assertGreaterEqual(learned_weight, wmin,
                                f"Weight {learned_weight} should be >= wmin ({wmin})")
        self.assertLessEqual(learned_weight, wmax,
                             f"Weight {learned_weight} should be <= wmax ({wmax})")
        print(f"Learned weight under heavy stimulation: {learned_weight}")


if __name__ == "__main__":
    unittest.main()
