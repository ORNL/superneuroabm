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
            config_name="config_0",
        )
        synapse1 = model.create_synapse(
            breed="weighted_synapse",
            pre_soma_id=soma0,
            post_soma_id=soma1,
            config_name="config_0",
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


class TestWeightedSynapseExpPairWiseSTDP(unittest.TestCase):
    """Tests for weighted synapse with unbounded exponential pair-wise STDP."""

    def _make_model(self):
        return NeuromorphicModel(enable_internal_state_tracking=True)

    def test_single_spike_response(self):
        """A single input spike should propagate through the STDP synapse and cause the soma to fire."""
        model = self._make_model()

        soma = model.create_soma(breed="lif_soma", config_name="config_0")
        synapse = model.create_synapse(
            breed="weighted_synapse",
            pre_soma_id=-1,
            post_soma_id=soma,
            config_name="config_0",
            learning_rule="exp_pair_wise_stdp",
        )

        model.setup()
        model.add_spike(synapse_id=synapse, tick=2, value=1)
        model.simulate(ticks=200, update_data_ticks=1)

        spike_times = model.get_spike_times(soma_id=soma)
        print(f"\nSingle spike response - soma spike times: {spike_times}")

        caller_name = inspect.stack()[0].function
        vizualize_responses(model, vthr=10, fig_name=f"{caller_name}_weighted_stdp.png")

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
            breed="weighted_synapse",
            pre_soma_id=-1,
            post_soma_id=soma_pre,
            config_name="config_0",
        )

        synapse_stdp = model.create_synapse(
            breed="weighted_synapse",
            pre_soma_id=soma_pre,
            post_soma_id=soma_post,
            config_name="config_0",
            learning_rule="exp_pair_wise_stdp",
        )

        model.setup(use_gpu=True)

        initial_weight = model.get_agent_property_value(
            id=synapse_stdp, property_name="hyperparameters"
        )[0]
        self.assertEqual(initial_weight, 100.0, "Initial weight should be 100.0")

        for tick in [10, 30, 50, 70, 90]:
            model.add_spike(synapse_id=synapse_input, tick=tick, value=1.0)

        model.simulate(ticks=200)

        learned_weight = model.get_agent_property_value(
            id=synapse_stdp, property_name="hyperparameters"
        )[0]

        self.assertNotEqual(
            learned_weight, initial_weight,
            f"STDP should have changed weight from {initial_weight}, but it's still {learned_weight}",
        )
        print(f"Initial weight: {initial_weight}, Learned weight: {learned_weight}")


class TestWeightedSynapseBoundedSTDP(unittest.TestCase):
    """Tests for weighted synapse with bounded exponential pair-wise STDP."""

    def _make_model(self):
        return NeuromorphicModel(enable_internal_state_tracking=True)

    def test_single_spike_response(self):
        """A single input spike should propagate through the bounded STDP synapse and cause the soma to fire."""
        model = self._make_model()

        soma = model.create_soma(breed="lif_soma", config_name="config_0")
        synapse = model.create_synapse(
            breed="weighted_synapse",
            pre_soma_id=-1,
            post_soma_id=soma,
            config_name="config_0",
            learning_rule="exp_pair_wise_stdp_bounded",
        )

        model.setup()
        model.add_spike(synapse_id=synapse, tick=2, value=1)
        model.simulate(ticks=200, update_data_ticks=1)

        spike_times = model.get_spike_times(soma_id=soma)
        print(f"\nSingle spike response - soma spike times: {spike_times}")

        caller_name = inspect.stack()[0].function
        vizualize_responses(model, vthr=10, fig_name=f"{caller_name}_weighted_bounded_stdp.png")

        self.assertGreaterEqual(
            len(spike_times), 1,
            "Soma should fire at least once from a single input spike",
        )

    def test_stdp_learning_changes_weight(self):
        """Bounded STDP learning should modify the synapse weight from its initial value."""
        model = self._make_model()

        soma_pre = model.create_soma(breed="lif_soma", config_name="config_0")
        soma_post = model.create_soma(breed="lif_soma", config_name="config_0")

        synapse_input = model.create_synapse(
            breed="weighted_synapse",
            pre_soma_id=-1,
            post_soma_id=soma_pre,
            config_name="config_0",
        )

        synapse_stdp = model.create_synapse(
            breed="weighted_synapse",
            pre_soma_id=soma_pre,
            post_soma_id=soma_post,
            config_name="config_0",
            learning_rule="exp_pair_wise_stdp_bounded",
        )

        model.setup(use_gpu=True)

        initial_weight = model.get_agent_property_value(
            id=synapse_stdp, property_name="hyperparameters"
        )[0]
        self.assertEqual(initial_weight, 100.0, "Initial weight should be 100.0")

        for tick in [10, 30, 50, 70, 90]:
            model.add_spike(synapse_id=synapse_input, tick=tick, value=1.0)

        model.simulate(ticks=200)

        learned_weight = model.get_agent_property_value(
            id=synapse_stdp, property_name="hyperparameters"
        )[0]

        self.assertNotEqual(
            learned_weight, initial_weight,
            f"Bounded STDP should have changed weight from {initial_weight}, but it's still {learned_weight}",
        )
        print(f"Initial weight: {initial_weight}, Learned weight: {learned_weight}")

    def test_weight_bounded(self):
        """Weight should stay within [wmin, wmax] even under heavy stimulation."""
        model = self._make_model()

        soma_pre = model.create_soma(breed="lif_soma", config_name="config_0")
        soma_post = model.create_soma(breed="lif_soma", config_name="config_0")

        synapse_input = model.create_synapse(
            breed="weighted_synapse",
            pre_soma_id=-1,
            post_soma_id=soma_pre,
            config_name="config_0",
        )

        synapse_stdp = model.create_synapse(
            breed="weighted_synapse",
            pre_soma_id=soma_pre,
            post_soma_id=soma_post,
            config_name="config_0",
            learning_rule="exp_pair_wise_stdp_bounded",
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


class TestWeightedSynapseThreeBitSTDP(unittest.TestCase):
    """Tests for weighted synapse with 3-bit quantized STDP."""

    def _make_model(self):
        return NeuromorphicModel(enable_internal_state_tracking=True)

    def test_single_spike_response(self):
        """A single input spike should propagate through the quantized STDP synapse and cause the soma to fire."""
        model = self._make_model()

        soma = model.create_soma(breed="lif_soma", config_name="config_0")
        synapse = model.create_synapse(
            breed="weighted_synapse",
            pre_soma_id=-1,
            post_soma_id=soma,
            config_name="config_0",
            learning_rule="three_bit_exp_pair_wise_stdp",
        )

        model.setup()
        model.add_spike(synapse_id=synapse, tick=2, value=1)
        model.simulate(ticks=200, update_data_ticks=1)

        spike_times = model.get_spike_times(soma_id=soma)
        print(f"\nSingle spike response - soma spike times: {spike_times}")

        caller_name = inspect.stack()[0].function
        vizualize_responses(model, vthr=10, fig_name=f"{caller_name}_weighted_3bit_stdp.png")

        self.assertGreaterEqual(
            len(spike_times), 1,
            "Soma should fire at least once from a single input spike",
        )

    def test_stdp_learning_changes_weight(self):
        """Quantized STDP learning should modify the synapse weight from its initial value."""
        model = self._make_model()

        soma_pre = model.create_soma(breed="lif_soma", config_name="config_0")
        soma_post = model.create_soma(breed="lif_soma", config_name="config_0")

        synapse_input = model.create_synapse(
            breed="weighted_synapse",
            pre_soma_id=-1,
            post_soma_id=soma_pre,
            config_name="config_0",
        )

        synapse_stdp = model.create_synapse(
            breed="weighted_synapse",
            pre_soma_id=soma_pre,
            post_soma_id=soma_post,
            config_name="config_0",
            learning_rule="three_bit_exp_pair_wise_stdp",
        )

        model.setup(use_gpu=True)

        initial_weight = model.get_agent_property_value(
            id=synapse_stdp, property_name="hyperparameters"
        )[0]
        self.assertEqual(initial_weight, 100.0, "Initial weight should be 100.0")

        for tick in [10, 30, 50, 70, 90]:
            model.add_spike(synapse_id=synapse_input, tick=tick, value=1.0)

        model.simulate(ticks=200, update_data_ticks=200)

        learned_weight = model.get_agent_property_value(
            id=synapse_stdp, property_name="hyperparameters"
        )[0]

        self.assertNotEqual(
            learned_weight, initial_weight,
            f"STDP should have changed weight from {initial_weight}, but it's still {learned_weight}",
        )
        print(f"Initial weight: {initial_weight}, Learned weight: {learned_weight}")

    def test_weight_is_quantized(self):
        """After STDP learning, the weight should land on a valid quantization level."""
        model = self._make_model()

        soma_pre = model.create_soma(breed="lif_soma", config_name="config_0")
        soma_post = model.create_soma(breed="lif_soma", config_name="config_0")

        synapse_input = model.create_synapse(
            breed="weighted_synapse",
            pre_soma_id=-1,
            post_soma_id=soma_pre,
            config_name="config_0",
        )

        synapse_stdp = model.create_synapse(
            breed="weighted_synapse",
            pre_soma_id=soma_pre,
            post_soma_id=soma_post,
            config_name="config_0",
            learning_rule="three_bit_exp_pair_wise_stdp",
        )

        model.setup(use_gpu=True)

        for tick in [10, 30, 50, 70, 90]:
            model.add_spike(synapse_id=synapse_input, tick=tick, value=1.0)

        model.simulate(ticks=200, update_data_ticks=200)

        learned_weight = model.get_agent_property_value(
            id=synapse_stdp, property_name="hyperparameters"
        )[0]

        # From config: wmin=0.0, wmax=24.0, num_levels=8
        wmin = 0.0
        wmax = 24.0
        num_levels = 8
        delta = (wmax - wmin) / (num_levels - 1)
        valid_levels = [wmin + i * delta for i in range(num_levels)]

        is_valid = any(abs(learned_weight - lv) < 1e-4 for lv in valid_levels)
        self.assertTrue(
            is_valid,
            f"Learned weight {learned_weight} is not a valid quantization level. "
            f"Valid levels: {valid_levels}",
        )
        print(f"Learned weight: {learned_weight}, Valid levels: {valid_levels}")

    def test_weight_bounded(self):
        """Weight should stay within [wmin, wmax] even under heavy stimulation."""
        model = self._make_model()

        soma_pre = model.create_soma(breed="lif_soma", config_name="config_0")
        soma_post = model.create_soma(breed="lif_soma", config_name="config_0")

        synapse_input = model.create_synapse(
            breed="weighted_synapse",
            pre_soma_id=-1,
            post_soma_id=soma_pre,
            config_name="config_0",
        )

        synapse_stdp = model.create_synapse(
            breed="weighted_synapse",
            pre_soma_id=soma_pre,
            post_soma_id=soma_post,
            config_name="config_0",
            learning_rule="three_bit_exp_pair_wise_stdp",
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


class TestWeightedSynapseMemristiveSTDP(unittest.TestCase):
    """Tests for weighted synapse with memristive exponential pair-wise STDP."""

    def _make_model(self):
        return NeuromorphicModel(enable_internal_state_tracking=True)

    def test_single_spike_response(self):
        """A single input spike should propagate through the memristive STDP synapse and cause the soma to fire."""
        model = self._make_model()

        soma = model.create_soma(breed="lif_soma", config_name="config_0")
        synapse = model.create_synapse(
            breed="weighted_synapse",
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
        vizualize_responses(model, vthr=10, fig_name=f"{caller_name}_weighted_memristive_stdp.png")

        self.assertGreaterEqual(
            len(spike_times), 1,
            "Soma should fire at least once from a single input spike",
        )

    def test_stdp_learning_changes_weight(self):
        """Memristive STDP learning should modify the synapse weight from its initial value."""
        model = self._make_model()

        soma_pre = model.create_soma(breed="lif_soma", config_name="config_0")
        soma_post = model.create_soma(breed="lif_soma", config_name="config_0")

        synapse_input = model.create_synapse(
            breed="weighted_synapse",
            pre_soma_id=-1,
            post_soma_id=soma_pre,
            config_name="config_0",
        )

        synapse_stdp = model.create_synapse(
            breed="weighted_synapse",
            pre_soma_id=soma_pre,
            post_soma_id=soma_post,
            config_name="config_0",
            learning_rule="memristive_exp_pair_wise_stdp",
        )

        model.setup(use_gpu=True)

        initial_weight = model.get_agent_property_value(
            id=synapse_stdp, property_name="hyperparameters"
        )[0]
        self.assertEqual(initial_weight, 100.0, "Initial weight should be 100.0")

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
            breed="weighted_synapse",
            pre_soma_id=-1,
            post_soma_id=soma_pre,
            config_name="config_0",
        )

        synapse_stdp = model.create_synapse(
            breed="weighted_synapse",
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


