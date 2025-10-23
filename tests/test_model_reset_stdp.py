import unittest
import numpy as np

from superneuroabm.model import NeuromorphicModel


class TestModelResetWithSTDP(unittest.TestCase):
    """
    Tests the model.reset() method's retain_parameters functionality with STDP learning.

    This test suite verifies that:
    - STDP learning actually changes synapse weights during simulation
    - When retain_parameters=True, learned weights are preserved after reset
    - When retain_parameters=False, weights are reset to their default values
    """

    def setUp(self):
        """Set up a fresh model for each test."""
        self.model = NeuromorphicModel(enable_internal_state_tracking=True)

    def test_stdp_learning_changes_weights(self):
        """
        Test that STDP learning actually modifies synapse weights.

        This test:
        1. Creates a network with STDP-enabled synapse
        2. Injects spike patterns that trigger STDP
        3. Runs simulation
        4. Verifies that weights have changed from default
        """
        # Create a simple network: external input -> soma_pre -> synapse_stdp -> soma_post
        soma_pre = self.model.create_soma(
            breed="lif_soma",
            config_name="config_0",
        )

        soma_post = self.model.create_soma(
            breed="lif_soma",
            config_name="config_0",
        )

        # External input synapse to trigger pre-synaptic soma
        synapse_input = self.model.create_synapse(
            breed="single_exp_synapse",
            pre_soma_id=np.nan,  # External input
            post_soma_id=soma_pre,
            config_name="no_learning_config_0",
        )

        # STDP-enabled synapse between two somas
        synapse_stdp = self.model.create_synapse(
            breed="single_exp_synapse",
            pre_soma_id=soma_pre,
            post_soma_id=soma_post,
            config_name="exp_pair_wise_stdp_config_0",
        )

        # Setup the model
        self.model.setup(use_gpu=True)

        # Get the initial weight (should be 14.0 from config)
        initial_hyperparameters = self.model.get_agent_property_value(
            id=synapse_stdp,
            property_name="hyperparameters"
        )
        initial_weight = initial_hyperparameters[0]
        self.assertEqual(initial_weight, 14.0, "Initial weight should be 14.0")

        # Inject multiple spikes to trigger STDP learning
        # Early spikes to cause repeated pre-post pairings (potentiation)
        spike_times = [10, 30, 50, 70, 90]
        for tick in spike_times:
            self.model.add_spike(synapse_id=synapse_input, tick=tick, value=1.0)

        # Run simulation
        simulation_ticks = 200
        self.model.simulate(ticks=simulation_ticks, update_data_ticks=simulation_ticks)

        # Get weight after learning
        learned_hyperparameters = self.model.get_agent_property_value(
            id=synapse_stdp,
            property_name="hyperparameters"
        )
        learned_weight = learned_hyperparameters[0]

        # Verify that STDP has changed the weight
        self.assertNotEqual(learned_weight, initial_weight,
                           f"STDP should have changed weight from {initial_weight}, but it's still {learned_weight}")
        print(f"Initial weight: {initial_weight}, Learned weight: {learned_weight}")

    def test_reset_retain_parameters_true_with_stdp(self):
        """
        Test that reset(retain_parameters=True) preserves STDP-learned weights and continues learning.

        This test:
        1. Creates a network with STDP-enabled synapse
        2. Runs first simulation to let STDP modify weights
        3. Calls reset(retain_parameters=True)
        4. Verifies the learned weights are preserved
        5. Runs second simulation to verify learning continues from preserved weights
        """
        # Create network with STDP
        soma_pre = self.model.create_soma(
            breed="lif_soma",
            config_name="config_0",
        )

        soma_post = self.model.create_soma(
            breed="lif_soma",
            config_name="config_0",
        )

        synapse_input = self.model.create_synapse(
            breed="single_exp_synapse",
            pre_soma_id=np.nan,
            post_soma_id=soma_pre,
            config_name="no_learning_config_0",
        )

        synapse_stdp = self.model.create_synapse(
            breed="single_exp_synapse",
            pre_soma_id=soma_pre,
            post_soma_id=soma_post,
            config_name="exp_pair_wise_stdp_config_0",
        )

        # Setup and run first simulation with STDP learning
        self.model.setup(use_gpu=True)

        # Inject spikes for STDP
        spike_times = [10, 30, 50, 70, 90]
        for tick in spike_times:
            self.model.add_spike(synapse_id=synapse_input, tick=tick, value=1.0)

        simulation_ticks = 200
        self.model.simulate(ticks=simulation_ticks, update_data_ticks=simulation_ticks)

        # Get learned weight after first simulation
        learned_hyperparameters = self.model.get_agent_property_value(
            id=synapse_stdp,
            property_name="hyperparameters"
        )
        learned_weight_first = learned_hyperparameters[0]

        # Verify learning occurred
        self.assertNotEqual(learned_weight_first, 14.0, "Weight should have changed from default")
        print(f"Weight after first simulation: {learned_weight_first}")

        # Reset with retain_parameters=True
        self.model.reset(retain_parameters=True)

        # Verify learned weight is preserved immediately after reset
        after_reset_hyperparameters = self.model.get_agent_property_value(
            id=synapse_stdp,
            property_name="hyperparameters"
        )
        after_reset_weight = after_reset_hyperparameters[0]

        self.assertEqual(after_reset_weight, learned_weight_first,
                        f"Learned weight {learned_weight_first} should be preserved after reset(retain_parameters=True), "
                        f"but got {after_reset_weight}")
        print(f"Weight after reset(retain_parameters=True): {after_reset_weight}")

        # Run second simulation to verify learning continues from preserved weights
        for tick in spike_times:
            self.model.add_spike(synapse_id=synapse_input, tick=tick, value=1.0)

        self.model.simulate(ticks=simulation_ticks, update_data_ticks=simulation_ticks)

        # Get weight after second simulation
        learned_weight_second = self.model.get_agent_property_value(
            id=synapse_stdp,
            property_name="hyperparameters"
        )[0]

        # Verify that learning continued from the preserved weight
        self.assertGreater(learned_weight_second, learned_weight_first,
                          f"Weight should continue to increase from {learned_weight_first}, "
                          f"but got {learned_weight_second}")
        print(f"Weight after second simulation (continued learning): {learned_weight_second}")

    def test_reset_retain_parameters_false_with_stdp(self):
        """
        Test that reset(retain_parameters=False) restores default weights and learns from scratch.

        This test:
        1. Creates a network with STDP-enabled synapse
        2. Runs first simulation to let STDP modify weights
        3. Calls reset(retain_parameters=False)
        4. Verifies weights are restored to default (14.0)
        5. Runs second simulation to verify learning starts from scratch (default weights)
        """
        # Create network with STDP
        soma_pre = self.model.create_soma(
            breed="lif_soma",
            config_name="config_0",
        )

        soma_post = self.model.create_soma(
            breed="lif_soma",
            config_name="config_0",
        )

        synapse_input = self.model.create_synapse(
            breed="single_exp_synapse",
            pre_soma_id=np.nan,
            post_soma_id=soma_pre,
            config_name="no_learning_config_0",
        )

        synapse_stdp = self.model.create_synapse(
            breed="single_exp_synapse",
            pre_soma_id=soma_pre,
            post_soma_id=soma_post,
            config_name="exp_pair_wise_stdp_config_0",
        )

        # Setup and run first simulation with STDP learning
        self.model.setup(use_gpu=True)

        # Store default weight
        default_weight = 14.0

        # Inject spikes for STDP
        spike_times = [10, 30, 50, 70, 90]
        for tick in spike_times:
            self.model.add_spike(synapse_id=synapse_input, tick=tick, value=1.0)

        simulation_ticks = 200
        self.model.simulate(ticks=simulation_ticks, update_data_ticks=simulation_ticks)

        # Get learned weight after first simulation
        learned_weight_first = self.model.get_agent_property_value(
            id=synapse_stdp,
            property_name="hyperparameters"
        )[0]

        # Verify learning occurred
        self.assertNotEqual(learned_weight_first, default_weight,
                           f"Weight should have changed from default {default_weight}")
        print(f"Weight after first simulation: {learned_weight_first}")

        # Reset with retain_parameters=False
        self.model.reset(retain_parameters=False)

        # Verify weight is restored to default immediately after reset
        after_reset_weight = self.model.get_agent_property_value(
            id=synapse_stdp,
            property_name="hyperparameters"
        )[0]

        self.assertEqual(after_reset_weight, default_weight,
                        f"Weight should be restored to default {default_weight} after reset(retain_parameters=False), "
                        f"but got {after_reset_weight}")
        print(f"Weight after reset(retain_parameters=False): {after_reset_weight}")

        # Run second simulation to verify learning starts from scratch
        for tick in spike_times:
            self.model.add_spike(synapse_id=synapse_input, tick=tick, value=1.0)

        self.model.simulate(ticks=simulation_ticks, update_data_ticks=simulation_ticks)

        # Get weight after second simulation
        learned_weight_second = self.model.get_agent_property_value(
            id=synapse_stdp,
            property_name="hyperparameters"
        )[0]

        # Verify that learning started from scratch (should be close to first simulation result)
        # The second simulation should produce similar weight change as the first simulation
        # since it started from the same default weight
        expected_weight_change = learned_weight_first - default_weight
        actual_weight_change = learned_weight_second - default_weight

        # Allow small tolerance for floating point comparison
        self.assertAlmostEqual(actual_weight_change, expected_weight_change, places=5,
                              msg=f"Weight change in second simulation ({actual_weight_change}) should match "
                                  f"first simulation ({expected_weight_change}) since both started from default")
        print(f"Weight after second simulation (learning from scratch): {learned_weight_second}")
        print(f"First simulation weight change: {expected_weight_change}, "
              f"Second simulation weight change: {actual_weight_change}")

    def test_multiple_reset_cycles_with_stdp(self):
        """
        Test multiple learning and reset cycles to ensure consistency.

        This test:
        1. Learns weights through STDP
        2. Resets with retain_parameters=True (should keep learned weights)
        3. Learns again (weights should change further)
        4. Resets with retain_parameters=False (should restore to default)
        """
        # Create network with STDP
        soma_pre = self.model.create_soma(
            breed="lif_soma",
            config_name="config_0",
        )

        soma_post = self.model.create_soma(
            breed="lif_soma",
            config_name="config_0",
        )

        synapse_input = self.model.create_synapse(
            breed="single_exp_synapse",
            pre_soma_id=np.nan,
            post_soma_id=soma_pre,
            config_name="no_learning_config_0",
        )

        synapse_stdp = self.model.create_synapse(
            breed="single_exp_synapse",
            pre_soma_id=soma_pre,
            post_soma_id=soma_post,
            config_name="exp_pair_wise_stdp_config_0",
        )

        default_weight = 14.0
        simulation_ticks = 200
        spike_times = [10, 30, 50, 70, 90]

        # First learning cycle
        self.model.setup(use_gpu=True)
        for tick in spike_times:
            self.model.add_spike(synapse_id=synapse_input, tick=tick, value=1.0)
        self.model.simulate(ticks=simulation_ticks, update_data_ticks=simulation_ticks)

        weight_after_first_learning = self.model.get_agent_property_value(
            id=synapse_stdp, property_name="hyperparameters"
        )[0]
        print(f"Weight after first learning: {weight_after_first_learning}")

        # Reset with retain_parameters=True
        self.model.reset(retain_parameters=True)
        weight_after_retain_reset = self.model.get_agent_property_value(
            id=synapse_stdp, property_name="hyperparameters"
        )[0]
        self.assertEqual(weight_after_retain_reset, weight_after_first_learning,
                        "Weight should be preserved after reset(retain_parameters=True)")

        # Second learning cycle (starting from learned weights)
        for tick in spike_times:
            self.model.add_spike(synapse_id=synapse_input, tick=tick, value=1.0)
        self.model.simulate(ticks=simulation_ticks, update_data_ticks=simulation_ticks)

        weight_after_second_learning = self.model.get_agent_property_value(
            id=synapse_stdp, property_name="hyperparameters"
        )[0]
        print(f"Weight after second learning: {weight_after_second_learning}")

        # Reset with retain_parameters=False
        self.model.reset(retain_parameters=False)
        weight_after_full_reset = self.model.get_agent_property_value(
            id=synapse_stdp, property_name="hyperparameters"
        )[0]

        self.assertEqual(weight_after_full_reset, default_weight,
                        f"Weight should be restored to default {default_weight} after reset(retain_parameters=False)")
        print(f"Weight after reset(retain_parameters=False): {weight_after_full_reset}")


if __name__ == "__main__":
    unittest.main()
