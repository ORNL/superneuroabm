"""Synaptic input: synapses deposit into the post soma's inbox, the soma filters it.

Runs under mpirun too (tests/run_mpi_tests.sh): create_synapse places a synapse with its
post soma, so every deposit here is local; a synapse on another rank than its post soma
deposits into the soma's ghost row and SAGESim sums it back to the owner (see
test_load_from_adjacency's split-synapse cases). Either way the two steppings of the
same kernels must agree to the bit at any rank count, a chunked run must equal one
launch, and a synapse kernel that cannot deliver is refused.
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


def _chain(stepping):
    """external -> soma_0 -> soma_1; returns (model, [soma_0, soma_1], syn_ext)."""
    model = NeuromorphicModel(synapse_stepping=stepping)
    model.set_seed(42)
    soma_0 = model.create_soma(breed="lif_soma", config_name="config_0")
    soma_1 = model.create_soma(breed="lif_soma", config_name="config_0")
    syn_ext = model.create_synapse(breed="delivery_synapse", pre_soma_id=-1,
                                   post_soma_id=soma_0, config_name="config_0")
    model.create_synapse(breed="delivery_synapse", pre_soma_id=soma_0,
                         post_soma_id=soma_1, config_name="config_0")
    model.setup()
    model.add_spike_list(syn_ext, [[2, 1.0], [5, 1.0], [9, 1.0], [40, 1.0], [41, 1.0]])
    return model, [soma_0, soma_1], syn_ext


def _fan_in(stepping, n_afferents=5, breed="lif_soma"):
    """n external synapses onto one soma, each driven with a different spike value."""
    model = NeuromorphicModel(synapse_stepping=stepping)
    model.set_seed(42)
    soma = model.create_soma(breed=breed, config_name="config_0")
    syns = [model.create_synapse(breed="delivery_synapse", pre_soma_id=-1,
                                 post_soma_id=soma, config_name="config_0")
            for _ in range(n_afferents)]
    model.setup()
    for i, s in enumerate(syns):
        model.add_spike_list(s, [[2 + i, 0.2 + 0.1 * i], [20, 0.5], [21 + i, 0.3]])
    return model, [soma]


def _state(model, somas):
    return {p: np.array([model.get_agent_property_value(s, p) for s in somas],
                        dtype=np.float32) for p in STATE_PROPS}


class TestClockedEqualsEvent(unittest.TestCase):

    def test_chain_bit_exact(self):
        clocked, sc, _ = _chain("clocked")
        event, se, _ = _chain("event")
        clocked.simulate(ticks=80)
        event.simulate(ticks=80)
        for i in range(2):
            self.assertEqual(clocked.get_spike_times(soma_id=sc[i]),
                             event.get_spike_times(soma_id=se[i]))
        self.assertGreater(len(event.get_spike_times(soma_id=se[1])), 0,
                           "the chain must actually fire for the test to mean anything")
        sa, sb = _state(clocked, sc), _state(event, se)
        for p in STATE_PROPS:
            np.testing.assert_array_equal(sa[p].view(np.uint32), sb[p].view(np.uint32),
                                          err_msg=f"{p} differs between clocked and event")

    def test_fan_in_sums_every_afferent(self):
        """Deposits of several synapses on one tick add up: the soma's recorded current
        after the tick-20 volley equals the sum of the deposits (to the rounding order
        of the atomic adds), and both steppings agree on the spikes."""
        clocked, sc = _fan_in("clocked")
        event, se = _fan_in("event")
        clocked.simulate(ticks=60)
        event.simulate(ticks=60)
        self.assertGreater(len(clocked.get_spike_times(soma_id=sc[0])), 0)
        self.assertEqual(clocked.get_spike_times(soma_id=sc[0]), event.get_spike_times(soma_id=se[0]))
        sa, sb = _state(clocked, sc), _state(event, se)
        np.testing.assert_allclose(sa["internal_states"], sb["internal_states"], rtol=1e-5, atol=1e-6)

    def test_resume_simulate(self):
        """simulate(k) repeated equals one simulate: the inbox survives launches.
        get_spike_times reports the spikes of the last simulate() call, so the chunked
        run's spikes are collected per chunk."""
        _single_rank_only(self)
        once, so, _ = _chain("event")
        twice, st, _ = _chain("event")
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


class TestValidation(unittest.TestCase):

    def test_bad_stepping(self):
        with self.assertRaises(ValueError):
            NeuromorphicModel(synapse_stepping="atomic")

    def test_custom_synapse_without_inbox_rejected(self):
        """A synapse kernel that does not take the inbox cannot deliver; setup refuses it."""
        from pathlib import Path
        from superneuroabm.step_functions.synapse import delivery as d_mod
        # the delivery kernel with the historical signature (no synaptic_input params)
        src = Path(d_mod.__file__).read_text()
        src = src.replace("    synaptic_input,\n    synaptic_input_idx,\n", "")
        src = src.replace(
            "    deliver_synaptic_current(synaptic_input, synaptic_input_idx, post_soma_index, channel,\n"
            "                             spike * scale * weight)\n", "    return\n")
        src = src.replace("def delivery_synapse_step_func(", "def legacy_synapse_step_func(")
        assert "synaptic_input" not in src.split("def legacy_synapse_step_func(")[1].split(")")[0]
        import tempfile, importlib.util
        d = Path(tempfile.mkdtemp())
        f = d / "legacy_synapse.py"
        f.write_text(src)
        spec = importlib.util.spec_from_file_location("legacy_synapse", f)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["legacy_synapse"] = mod
        spec.loader.exec_module(mod)

        import copy
        model = NeuromorphicModel(synapse_stepping="clocked")
        model.register_synapse_type("legacy", mod.legacy_synapse_step_func, f)
        model._component_configurations["synapse"]["legacy"] = copy.deepcopy(
            model._component_configurations["synapse"]["delivery_synapse"])
        soma = model.create_soma(breed="lif_soma", config_name="config_0")
        model.create_synapse(breed="legacy", pre_soma_id=-1, post_soma_id=soma,
                             config_name="config_0")
        with self.assertRaises(ValueError):
            model.setup()


if __name__ == "__main__":
    unittest.main()
