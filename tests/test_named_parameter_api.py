import unittest

from superneuroabm.model import NeuromorphicModel


class TestNamedParameterAPI(unittest.TestCase):
    """get/set_hyperparameters and their learning_ counterparts.

    These replace positional indexing into the property vectors. A name maps to the
    key's position in that agent's YAML block, so the API is the only access path
    that stays correct if a config is reordered.
    """

    SPIKE_TICKS = [10, 30, 50]
    SIM_TICKS = 100

    def setUp(self):
        self.model = NeuromorphicModel(enable_internal_states_tracking=True)
        self.soma_pre = self.model.create_soma(breed="lif_soma", config_name="config_0")
        self.soma_post = self.model.create_soma(breed="lif_soma", config_name="config_0")
        # External input -> soma_pre, no learning rule
        self.synapse_input = self.model.create_synapse(
            breed="single_exp_synapse",
            pre_soma_id=-1,
            post_soma_id=self.soma_pre,
            config_name="config_0",
        )
        # soma_pre -> soma_post, plastic
        self.synapse_stdp = self.model.create_synapse(
            breed="single_exp_synapse",
            pre_soma_id=self.soma_pre,
            post_soma_id=self.soma_post,
            config_name="config_0",
            learning_rule="exp_pair_wise_stdp",
        )

    # -- reads ---------------------------------------------------------

    def test_get_returns_named_dict_in_config_order(self):
        hp = self.model.get_hyperparameters(self.soma_pre)
        self.assertEqual(
            list(hp),
            ["C", "R", "vthr", "tref", "vrest", "vreset",
             "tref_allows_integration", "I_in", "scaling_factor"],
        )
        raw = self.model.get_agent_property_value(
            id=self.soma_pre, property_name="hyperparameters"
        )
        self.assertEqual(hp["vthr"], raw[2], "named read must agree with the positional one")

    def test_get_accepts_iterable(self):
        both = self.model.get_hyperparameters([self.soma_pre, self.soma_post])
        self.assertIsInstance(both, list)
        self.assertEqual(len(both), 2)
        self.assertEqual(both[0], self.model.get_hyperparameters(self.soma_pre))

    def test_synapse_and_learning_rule_names(self):
        hp = self.model.get_hyperparameters(self.synapse_stdp)
        self.assertEqual(
            list(hp), ["weight", "synaptic_delay", "scale", "tau_fall", "tau_rise"]
        )
        lhp = self.model.get_learning_hyperparameters(self.synapse_stdp)
        self.assertEqual(lhp["stdp_type"], 0.0, "exp_pair_wise_stdp is built-in rule 0")
        self.assertIn("a_exp_pre", lhp)

    def test_synapse_without_learning_rule(self):
        """No rule attached: the synthetic one-element vector, not a config lookup."""
        lhp = self.model.get_learning_hyperparameters(self.synapse_input)
        self.assertEqual(list(lhp), ["stdp_type"])
        self.assertEqual(lhp["stdp_type"], -1.0)

    def test_soma_and_synapse_resolve_differently(self):
        """Index 1 is R on a soma and synaptic_delay on a synapse -- the whole point."""
        soma_names = list(self.model.get_hyperparameters(self.soma_pre))
        syn_names = list(self.model.get_hyperparameters(self.synapse_stdp))
        self.assertEqual(soma_names[1], "R")
        self.assertEqual(syn_names[1], "synaptic_delay")

    # -- writes --------------------------------------------------------

    def test_partial_update_leaves_others_untouched(self):
        before = self.model.get_hyperparameters(self.soma_pre)
        self.model.set_hyperparameters(self.soma_pre, {"vthr": -33.0})
        after = self.model.get_hyperparameters(self.soma_pre)
        self.assertEqual(after["vthr"], -33.0)
        for name, value in before.items():
            if name != "vthr":
                self.assertEqual(after[name], value, f"{name} was disturbed")

    def test_set_accepts_iterable_and_scalar(self):
        self.model.set_hyperparameters([self.soma_pre, self.soma_post], {"tref": 0.0})
        for sid in (self.soma_pre, self.soma_post):
            self.assertEqual(self.model.get_hyperparameters(sid)["tref"], 0.0)

        self.model.set_hyperparameters(self.soma_post, {"tref": 0.007})
        self.assertEqual(self.model.get_hyperparameters(self.soma_post)["tref"], 0.007)
        self.assertEqual(
            self.model.get_hyperparameters(self.soma_pre)["tref"], 0.0,
            "a scalar write must not touch the other agent",
        )

    def test_empty_update_is_a_noop(self):
        before = self.model.get_hyperparameters(self.soma_pre)
        self.model.set_hyperparameters(self.soma_pre, {})
        self.assertEqual(self.model.get_hyperparameters(self.soma_pre), before)

    def test_learning_hyperparameter_write(self):
        self.model.set_learning_hyperparameters(self.synapse_stdp, {"a_exp_pre": 0.5})
        self.assertEqual(
            self.model.get_learning_hyperparameters(self.synapse_stdp)["a_exp_pre"], 0.5
        )

    # -- guards --------------------------------------------------------

    def test_unknown_name_raises_and_lists_valid_ones(self):
        with self.assertRaises(KeyError) as ctx:
            self.model.set_hyperparameters(self.soma_pre, {"vthreshold": -33.0})
        message = str(ctx.exception)
        self.assertIn("vthreshold", message)
        self.assertIn("vthr", message, "the error should list the valid names")

    def test_synaptic_delay_rejected(self):
        """Creation-time only: the delay register is sized from it and rebuilt on reset."""
        with self.assertRaises(ValueError) as ctx:
            self.model.set_hyperparameters(self.synapse_stdp, {"synaptic_delay": 3.0})
        self.assertIn("create_synapse", str(ctx.exception))

    def test_write_blocked_while_gpu_holds_unsynced_state(self):
        """After simulate() the GPU has values the CPU never saw; writing would drop them."""
        self.model.setup()
        self.model.simulate(ticks=self.SIM_TICKS)

        with self.assertRaises(RuntimeError) as ctx:
            self.model.set_hyperparameters(self.soma_pre, {"vthr": -33.0})
        self.assertIn("reset(retain_parameters=True)", str(ctx.exception))

        self.model.reset(retain_parameters=True)
        self.model.set_hyperparameters(self.soma_pre, {"vthr": -33.0})
        self.assertEqual(self.model.get_hyperparameters(self.soma_pre)["vthr"], -33.0)

    def test_writes_allowed_before_first_simulate(self):
        self.model.set_hyperparameters(self.soma_pre, {"vthr": -33.0})    # before setup()
        self.model.setup()
        self.model.set_hyperparameters(self.soma_post, {"vthr": -34.0})   # after setup()
        self.assertEqual(self.model.get_hyperparameters(self.soma_pre)["vthr"], -33.0)
        self.assertEqual(self.model.get_hyperparameters(self.soma_post)["vthr"], -34.0)

    # -- round trip ----------------------------------------------------

    def test_values_survive_simulate_and_reset(self):
        self.model.set_hyperparameters(self.soma_post, {"vthr": -33.0, "tref": 0.0})
        self.model.setup()
        for tick in self.SPIKE_TICKS:
            self.model.add_spike(synapse_id=self.synapse_input, tick=tick, value=1.0)
        self.model.simulate(ticks=self.SIM_TICKS)
        self.model.reset(retain_parameters=True)

        hp = self.model.get_hyperparameters(self.soma_post)
        # GPU tensors are float32, so a round trip quantizes.
        self.assertAlmostEqual(hp["vthr"], -33.0, places=4)
        self.assertAlmostEqual(hp["tref"], 0.0, places=6)


class TestKernelPinnedOrderValidation(unittest.TestCase):
    """weight/synaptic_delay/stdp_type positions are read directly by device code.

    Reordering a config would mis-dispatch silently on the GPU, so it must fail at
    model-construction time instead.
    """

    def test_reordered_synapse_config_is_rejected(self):
        model = NeuromorphicModel(enable_internal_states_tracking=False)
        config = model._component_configurations["synapse"]["single_exp_synapse"]["config_0"]
        reordered = dict(config["hyperparameters"])
        # Move weight out of position 0.
        config["hyperparameters"] = {
            "scale": reordered["scale"],
            "weight": reordered["weight"],
            "synaptic_delay": reordered["synaptic_delay"],
            "tau_fall": reordered["tau_fall"],
            "tau_rise": reordered["tau_rise"],
        }
        soma = model.create_soma(breed="lif_soma", config_name="config_0")
        with self.assertRaises(ValueError) as ctx:
            model.create_synapse(
                breed="single_exp_synapse", pre_soma_id=-1, post_soma_id=soma,
                config_name="config_0",
            )
        self.assertIn("weight", str(ctx.exception))
        self.assertIn("device code", str(ctx.exception))

    def test_reordered_learning_rule_config_is_rejected(self):
        model = NeuromorphicModel(enable_internal_states_tracking=False)
        lr = model._learning_rule_configurations["exp_pair_wise_stdp"]["default"]
        original = dict(lr["learning_hyperparameters"])
        lr["learning_hyperparameters"] = {
            "tau_pre_stdp": original["tau_pre_stdp"],
            "stdp_type": original["stdp_type"],
            **{k: v for k, v in original.items() if k not in ("stdp_type", "tau_pre_stdp")},
        }
        soma_a = model.create_soma(breed="lif_soma", config_name="config_0")
        soma_b = model.create_soma(breed="lif_soma", config_name="config_0")
        with self.assertRaises(ValueError) as ctx:
            model.create_synapse(
                breed="single_exp_synapse", pre_soma_id=soma_a, post_soma_id=soma_b,
                config_name="config_0", learning_rule="exp_pair_wise_stdp",
            )
        self.assertIn("stdp_type", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
