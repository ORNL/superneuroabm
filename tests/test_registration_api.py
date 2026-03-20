"""
Tests for the custom component registration API.

Verifies:
  - register_soma_type: custom soma with existing step function works
  - register_learning_rule: custom learning rule with auto-assigned ID works
  - Error cases: after-setup and duplicate-name errors
"""

import unittest
import copy
from pathlib import Path

from superneuroabm.model import NeuromorphicModel
from superneuroabm.step_functions.soma.lif import lif_soma_step_func
from superneuroabm.step_functions.synapse.stdp.exp_pair_wise_stdp import (
    exp_pair_wise_stdp,
)
from superneuroabm.step_functions.synapse.stdp.exp_pair_wise_stdp_bounded_nn import (
    exp_pair_wise_stdp_bounded_nn,
)

SUPERNEURO_DIR = Path(__file__).resolve().parent.parent / "superneuroabm"


class TestRegistrationAPI(unittest.TestCase):
    """Tests for the custom component registration API."""

    def test_register_soma_type(self):
        """Register existing LIF soma under a new name and verify it fires."""
        model = NeuromorphicModel(enable_internal_state_tracking=True)

        lif_path = SUPERNEURO_DIR / "step_functions" / "soma" / "lif.py"
        model.register_soma_type(
            name="my_lif_soma",
            step_func=lif_soma_step_func,
            step_func_path=lif_path,
        )

        # Add config for the custom breed (copy from lif_soma)
        model._component_configurations["soma"]["my_lif_soma"] = copy.deepcopy(
            model._component_configurations["soma"]["lif_soma"]
        )

        soma = model.create_soma(breed="my_lif_soma", config_name="config_0")
        synapse = model.create_synapse(
            breed="single_exp_synapse",
            pre_soma_id=-1,
            post_soma_id=soma,
            config_name="config_0",
        )

        model.setup(use_gpu=True)
        model.add_spike(synapse_id=synapse, tick=2, value=1)
        model.simulate(ticks=20, update_data_ticks=1)

        spike_times = model.get_spike_times(soma_id=soma)
        print(f"\nCustom soma spike times: {spike_times}")
        self.assertGreaterEqual(
            len(spike_times),
            1,
            "Custom soma should fire at least once from a single input spike",
        )

    def test_register_learning_rule(self):
        """Register exp_pair_wise_stdp_bounded_nn and verify learning works."""
        model = NeuromorphicModel(enable_internal_state_tracking=True)

        stdp_path = (
            SUPERNEURO_DIR
            / "step_functions"
            / "synapse"
            / "stdp"
            / "exp_pair_wise_stdp_bounded_nn.py"
        )
        rule_id = model.register_learning_rule(
            step_func=exp_pair_wise_stdp_bounded_nn,
            step_func_path=stdp_path,
        )

        self.assertEqual(rule_id, 4, "First user rule should get ID 4")

        # Create network
        soma_pre = model.create_soma(breed="lif_soma", config_name="config_0")
        soma_post = model.create_soma(breed="lif_soma", config_name="config_0")

        synapse_input = model.create_synapse(
            breed="single_exp_synapse",
            pre_soma_id=-1,
            post_soma_id=soma_pre,
            config_name="config_0",
        )

        # Use the custom learning rule (override stdp_type to the new ID)
        synapse_stdp = model.create_synapse(
            breed="single_exp_synapse",
            pre_soma_id=soma_pre,
            post_soma_id=soma_post,
            config_name="config_0",
            learning_rule="exp_pair_wise_stdp",
            overrides={"learning_hyperparameters": {"stdp_type": float(rule_id)}},
        )

        model.setup(use_gpu=True)

        initial_weight = model.get_agent_property_value(
            id=synapse_stdp, property_name="hyperparameters"
        )[0]

        # Inject spikes to trigger STDP
        for tick in [10, 30, 50, 70, 90]:
            model.add_spike(synapse_id=synapse_input, tick=tick, value=1.0)

        model.simulate(ticks=200, update_data_ticks=200)

        learned_weight = model.get_agent_property_value(
            id=synapse_stdp, property_name="hyperparameters"
        )[0]

        print(
            f"\nCustom learning rule: initial={initial_weight}, learned={learned_weight}"
        )
        self.assertNotEqual(
            learned_weight,
            initial_weight,
            f"Custom learning rule should have changed weight from {initial_weight}",
        )

    def test_register_after_setup_raises(self):
        """Calling register methods after setup() should raise RuntimeError."""
        model = NeuromorphicModel()

        soma = model.create_soma(breed="lif_soma", config_name="config_0")
        model.create_synapse(
            breed="single_exp_synapse",
            pre_soma_id=-1,
            post_soma_id=soma,
            config_name="config_0",
        )
        model.setup(use_gpu=True)

        dummy_path = Path(__file__)

        with self.assertRaises(RuntimeError):
            model.register_soma_type("late_soma", lif_soma_step_func, dummy_path)

        with self.assertRaises(RuntimeError):
            model.register_synapse_type("late_synapse", lif_soma_step_func, dummy_path)

        with self.assertRaises(RuntimeError):
            model.register_learning_rule(exp_pair_wise_stdp, dummy_path)

    def test_duplicate_name_raises(self):
        """Registering with an existing name should raise ValueError."""
        model = NeuromorphicModel()

        dummy_path = Path(__file__)

        with self.assertRaises(ValueError):
            model.register_soma_type("lif_soma", lif_soma_step_func, dummy_path)

        with self.assertRaises(ValueError):
            model.register_synapse_type(
                "single_exp_synapse", lif_soma_step_func, dummy_path
            )

        with self.assertRaises(ValueError):
            model.register_learning_rule(exp_pair_wise_stdp, dummy_path)


if __name__ == "__main__":
    unittest.main()
