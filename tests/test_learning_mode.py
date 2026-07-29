import unittest

from superneuroabm.model import NeuromorphicModel


class TestLearningMode(unittest.TestCase):
    """
    Tests the eval() / train() / set_learning_enabled() plasticity switch.

    eval() flips stdp_type (learning_hyperparameters[0]) to -1.0, the sentinel the
    generated learning-rule selector treats as "no rule", so STDP stops updating
    weights. train() restores the saved value. Weights themselves are never touched.
    """

    SPIKE_TICKS = [10, 30, 50, 70, 90]
    SIM_TICKS = 200

    def setUp(self):
        self.model = NeuromorphicModel(enable_internal_states_tracking=True)
        self.soma_pre = self.model.create_soma(breed="lif_soma", config_name="config_0")
        self.soma_post = self.model.create_soma(breed="lif_soma", config_name="config_0")
        # External input -> soma_pre (no learning rule attached)
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

    # -- helpers -------------------------------------------------------

    def _weight(self):
        return self.model.get_agent_property_value(
            id=self.synapse_stdp, property_name="hyperparameters"
        )[0]

    def _stdp_type(self, synapse_id):
        return self.model.get_agent_property_value(
            id=synapse_id, property_name="learning_hyperparameters"
        )[0]

    def _cpu_stdp_type(self, synapse_id):
        """Read the AgentFactory copy directly.

        get_agent_property_value reads the GPU while the buffers are live, so it
        would mask a CPU/GPU divergence. This reads the durable copy -- the one a
        buffer rebuild repopulates the GPU from.
        """
        af = self.model._agent_factory
        idx = af._rank2agentid2agentidx[0][synapse_id]
        return af._property_name_2_agent_data_tensor["learning_hyperparameters"][idx][0]

    def _invalidate_gpu_buffers(self):
        """Force a rebuild the way any ordinary property write would."""
        hp = self.model.get_agent_property_value(
            id=self.soma_pre, property_name="hyperparameters"
        )
        self.model.set_agent_property_value(self.soma_pre, "hyperparameters", hp)

    def _run(self):
        for tick in self.SPIKE_TICKS:
            self.model.add_spike(synapse_id=self.synapse_input, tick=tick, value=1.0)
        self.model.simulate(ticks=self.SIM_TICKS, update_data_ticks=self.SIM_TICKS)

    # -- tests ---------------------------------------------------------

    def test_default_is_training_mode(self):
        self.assertTrue(self.model.learning_enabled)
        self.model.setup()
        self._run()
        self.assertNotEqual(
            self._weight(), 14.0, "plasticity should be active by default"
        )

    def test_eval_freezes_weights(self):
        self.model.setup()
        self._run()
        learned = self._weight()
        self.assertNotEqual(learned, 14.0, "sanity: STDP should have run")

        # Ordering matters, and it is not specific to eval(): reset() syncs GPU
        # state back to the AgentFactory only while the buffers are still valid.
        # Any property write invalidates them, so disabling BEFORE reset drops
        # the learned weight. The manual set_agent_property_value loop behaves
        # exactly the same way. reset() first, then disable -- as the tutorial does.
        self.model.reset(retain_parameters=True)
        self.model.eval()
        self.assertFalse(self.model.learning_enabled)
        self.assertEqual(self._stdp_type(self.synapse_stdp), -1.0)

        self._run()
        self.assertEqual(
            self._weight(), learned, "eval() must freeze the weight across a simulate"
        )

    def test_train_restores_plasticity(self):
        self.model.setup()
        self.model.eval()
        frozen_type = self._stdp_type(self.synapse_stdp)
        self.assertEqual(frozen_type, -1.0)

        self.model.train()
        self.assertTrue(self.model.learning_enabled)
        self.assertNotEqual(
            self._stdp_type(self.synapse_stdp), -1.0, "train() must restore stdp_type"
        )

        self._run()
        self.assertNotEqual(self._weight(), 14.0, "learning should resume after train()")

    def test_eval_is_idempotent(self):
        """Calling eval() twice must not overwrite the snapshot with the sentinel."""
        self.model.setup()
        original = self._stdp_type(self.synapse_stdp)
        self.model.eval()
        self.model.eval()
        self.model.train()
        self.assertEqual(
            self._stdp_type(self.synapse_stdp),
            original,
            "repeated eval() then train() must return the original stdp_type",
        )

    def test_eval_survives_reset_retain_true(self):
        self.model.setup()
        self.model.eval()
        self.model.reset(retain_parameters=True)
        self.assertFalse(self.model.learning_enabled)
        self.assertEqual(self._stdp_type(self.synapse_stdp), -1.0)

    def test_eval_survives_reset_retain_false(self):
        """retain_parameters=False restores learning_hyperparameters from config,
        which would silently re-enable plasticity without the sticky-mode hook."""
        self.model.setup()
        self.model.eval()
        self.model.reset(retain_parameters=False)
        self.assertFalse(self.model.learning_enabled)
        self.assertEqual(self._stdp_type(self.synapse_stdp), -1.0)

    def test_non_plastic_synapse_untouched(self):
        """Synapses created without a learning rule already sit at -1 and must not
        be captured into the snapshot, so train() cannot resurrect a rule."""
        self.model.setup()
        before = self._stdp_type(self.synapse_input)
        self.model.eval()
        self.model.train()
        self.assertEqual(self._stdp_type(self.synapse_input), before)
        self.assertNotIn(self.synapse_input, self.model._saved_stdp_type)

    def test_selective_synapse_ids(self):
        """set_learning_enabled with explicit ids affects only those, and does not
        move the global mode flag."""
        other = self.model.create_synapse(
            breed="single_exp_synapse",
            pre_soma_id=self.soma_post,
            post_soma_id=self.soma_pre,
            config_name="config_0",
            learning_rule="exp_pair_wise_stdp",
        )
        self.model.setup()
        untouched = self._stdp_type(other)

        self.model.set_learning_enabled(False, synapse_ids=[self.synapse_stdp])
        self.assertEqual(self._stdp_type(self.synapse_stdp), -1.0)
        self.assertEqual(self._stdp_type(other), untouched)
        self.assertTrue(
            self.model.learning_enabled,
            "a targeted call must not flip the global mode flag",
        )

    def test_eval_before_setup(self):
        """eval() must work on the pre-setup path (AgentFactory tensors)."""
        self.model.eval()
        self.model.setup()
        self.assertEqual(self._stdp_type(self.synapse_stdp), -1.0)
        self._run()
        self.assertEqual(self._weight(), 14.0, "no learning should have occurred")

    def test_eval_returns_self(self):
        self.assertIs(self.model.eval(), self.model)
        self.assertIs(self.model.train(), self.model)

    # -- CPU/GPU coherence ---------------------------------------------
    # A GPU-only write looks correct through get_agent_property_value (which
    # reads the GPU while buffers are live) right up until something triggers a
    # rebuild, which repopulates the GPU from the AgentFactory. These pin that.

    def test_eval_writes_cpu_copy(self):
        self.model.setup()
        self.model.simulate(ticks=self.SIM_TICKS)
        self.model.eval()
        self.assertEqual(
            self._cpu_stdp_type(self.synapse_stdp), -1.0,
            "eval() must write the AgentFactory copy, not just the GPU tensor",
        )

    def test_eval_survives_buffer_rebuild(self):
        self.model.setup()
        self.model.simulate(ticks=self.SIM_TICKS)
        self.model.eval()
        self._invalidate_gpu_buffers()
        self.model.simulate(ticks=self.SIM_TICKS)
        self.assertEqual(
            self._stdp_type(self.synapse_stdp), -1.0,
            "eval() was reverted by the GPU buffer rebuild",
        )

    def test_train_survives_buffer_rebuild(self):
        self.model.setup()
        original = self._stdp_type(self.synapse_stdp)
        self.model.eval()
        self.model.train()
        self._invalidate_gpu_buffers()
        self.model.simulate(ticks=self.SIM_TICKS)
        self.assertEqual(
            self._stdp_type(self.synapse_stdp), original,
            "train() was reverted by the GPU buffer rebuild",
        )

    def test_eval_still_frozen_after_rebuild(self):
        """End to end: the weight must stay put across a mid-run invalidation."""
        self.model.setup()
        self._run()
        learned = self._weight()

        self.model.reset(retain_parameters=True)   # sync first (see above)
        self.model.eval()
        self._run()                                 # buffers live again, weight frozen
        self.assertEqual(self._weight(), learned)

        self._invalidate_gpu_buffers()              # forces a rebuild from AgentFactory
        self._run()
        self.assertEqual(
            self._weight(), learned,
            "plasticity resumed after a buffer rebuild despite eval()",
        )


if __name__ == "__main__":
    unittest.main()
