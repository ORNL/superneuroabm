"""synaptic_input="push": synapses deposit their current into the post soma's inbox.

Runs under mpirun too (tests/run_mpi_tests.sh): post-owns placement keeps every deposit
local, so push must match pull at any rank count.

The soma then reads one row instead of looping over its afferents. The values are the
same as the pull loop's, so spike times and soma state must agree with a pull-mode
model driven by the same input. With one afferent per soma the sums are single terms
and the two modes are bit-identical; with several afferents the atomic adds may order
the float additions differently, so state is compared to float32 round-off.
"""

import sys
import unittest

import numpy as np

from superneuroabm.model import NeuromorphicModel

STATE_PROPS = ("internal_states", "output_spikes_tensor")


def _single_rank_only(test):
    try:
        from mpi4py import MPI
        if MPI.COMM_WORLD.Get_size() > 1:
            test.skipTest("single-rank test")
    except ImportError:
        pass


def _chain(mode, synapse_breed="single_exp_synapse"):
    """external -> soma_0 -> soma_1; returns (model, [soma_0, soma_1], syn_ext)."""
    model = NeuromorphicModel(synaptic_input=mode)
    model.set_seed(42)
    soma_0 = model.create_soma(breed="lif_soma", config_name="config_0")
    soma_1 = model.create_soma(breed="lif_soma", config_name="config_0")
    syn_ext = model.create_synapse(breed=synapse_breed, pre_soma_id=-1,
                                   post_soma_id=soma_0, config_name="config_0")
    model.create_synapse(breed=synapse_breed, pre_soma_id=soma_0,
                         post_soma_id=soma_1, config_name="config_0")
    model.setup()
    model.add_spike_list(syn_ext, [[2, 1.0], [5, 1.0], [9, 1.0], [40, 1.0], [41, 1.0]])
    return model, [soma_0, soma_1], syn_ext


def _fan_in(mode, n_afferents=5, breed="lif_soma", synapse_breed="single_exp_synapse"):
    """n external synapses onto one soma, each driven with a different spike value."""
    model = NeuromorphicModel(synaptic_input=mode)
    model.set_seed(42)
    soma = model.create_soma(breed=breed, config_name="config_0")
    syns = [model.create_synapse(breed=synapse_breed, pre_soma_id=-1,
                                 post_soma_id=soma, config_name="config_0")
            for _ in range(n_afferents)]
    model.setup()
    for i, s in enumerate(syns):
        model.add_spike_list(s, [[2 + i, 0.2 + 0.1 * i], [20, 0.5], [21 + i, 0.3]])
    return model, [soma]


def _state(model, somas):
    return {p: np.array([model.get_agent_property_value(s, p) for s in somas],
                        dtype=np.float32) for p in STATE_PROPS}


class TestPushEqualsPull(unittest.TestCase):

    def test_chain_bit_exact(self):
        pull, somas_pull, _ = _chain("pull")
        push, somas_push, _ = _chain("push")
        pull.simulate(ticks=80)
        push.simulate(ticks=80)
        self.assertEqual(pull.get_spike_times(soma_id=somas_pull[0]),
                         push.get_spike_times(soma_id=somas_push[0]))
        self.assertEqual(pull.get_spike_times(soma_id=somas_pull[1]),
                         push.get_spike_times(soma_id=somas_push[1]))
        self.assertGreater(len(push.get_spike_times(soma_id=somas_push[1])), 0,
                           "the chain must actually fire for the test to mean anything")
        sa, sb = _state(pull, somas_pull), _state(push, somas_push)
        for p in STATE_PROPS:
            np.testing.assert_array_equal(sa[p].view(np.uint32), sb[p].view(np.uint32),
                                          err_msg=f"{p} differs between pull and push")

    def test_fan_in_matches_to_roundoff(self):
        pull, sp = _fan_in("pull")
        push, ss = _fan_in("push")
        pull.simulate(ticks=60)
        push.simulate(ticks=60)
        self.assertGreater(len(pull.get_spike_times(soma_id=sp[0])), 0)
        self.assertEqual(pull.get_spike_times(soma_id=sp[0]), push.get_spike_times(soma_id=ss[0]))
        sa, sb = _state(pull, sp), _state(push, ss)
        np.testing.assert_allclose(sa["internal_states"], sb["internal_states"], rtol=1e-5, atol=1e-6)

    def test_weighted_synapse(self):
        pull, sp = _fan_in("pull", n_afferents=3, synapse_breed="weighted_synapse")
        push, ss = _fan_in("push", n_afferents=3, synapse_breed="weighted_synapse")
        pull.simulate(ticks=40)
        push.simulate(ticks=40)
        self.assertEqual(pull.get_spike_times(soma_id=sp[0]), push.get_spike_times(soma_id=ss[0]))

    def test_resume_simulate(self):
        """simulate(k) repeated equals one simulate: the inbox survives launches.
        get_spike_times reports the spikes of the last simulate() call, so the chunked
        run's spikes are collected per chunk."""
        _single_rank_only(self)
        once, so, _ = _chain("push")
        twice, st, _ = _chain("push")
        once.simulate(ticks=80)
        chunked = []
        for _ in range(8):
            twice.simulate(ticks=10)
            chunked += twice.get_spike_times(soma_id=st[1])
        self.assertEqual(once.get_spike_times(soma_id=so[1]), chunked)
        sa, sb = _state(once, so), _state(twice, st)
        for p in STATE_PROPS:
            np.testing.assert_array_equal(sa[p].view(np.uint32), sb[p].view(np.uint32),
                                          err_msg=f"{p} differs between one launch and eight")


class TestPushValidation(unittest.TestCase):

    def test_bad_mode(self):
        with self.assertRaises(ValueError):
            NeuromorphicModel(synaptic_input="atomic")

    def test_custom_synapse_without_inbox_rejected_in_push(self):
        """A synapse kernel that does not take the inbox cannot deliver; push refuses it."""
        from pathlib import Path
        from superneuroabm.step_functions.synapse import weighted_synapse as ws_mod
        # a kernel with the historical signature (no synaptic_input params)
        src = Path(ws_mod.__file__).read_text().replace(
            "    synaptic_input,\n    synaptic_input_idx,\n", "").replace(
            "    deliver_synaptic_current(synaptic_input, synaptic_input_idx, post_soma_index, 0, I_synapse)\n", "")
        src = src.replace("def weighted_synapse_step_func(", "def legacy_weighted_synapse_step_func(")
        import tempfile, importlib.util
        d = Path(tempfile.mkdtemp())
        f = d / "legacy_weighted.py"
        f.write_text(src)
        spec = importlib.util.spec_from_file_location("legacy_weighted", f)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["legacy_weighted"] = mod
        spec.loader.exec_module(mod)

        import copy

        def _model(mode):
            model = NeuromorphicModel(synaptic_input=mode)
            model.register_synapse_type("legacy", mod.legacy_weighted_synapse_step_func, f)
            model._component_configurations["synapse"]["legacy"] = copy.deepcopy(
                model._component_configurations["synapse"]["weighted_synapse"])
            return model

        model = _model("push")
        soma = model.create_soma(breed="lif_soma", config_name="config_0")
        model.create_synapse(breed="legacy", pre_soma_id=-1, post_soma_id=soma,
                             config_name="config_0")
        with self.assertRaises(ValueError):
            model.setup()

        # the same kernel is fine in pull mode (it keeps its old signature)
        model = _model("pull")
        soma = model.create_soma(breed="lif_soma", config_name="config_0")
        syn = model.create_synapse(breed="legacy", pre_soma_id=-1, post_soma_id=soma,
                                   config_name="config_0")
        model.setup()
        model.add_spike_list(syn, [[2, 1.0], [3, 1.0]])
        model.simulate(ticks=10)


if __name__ == "__main__":
    unittest.main()
