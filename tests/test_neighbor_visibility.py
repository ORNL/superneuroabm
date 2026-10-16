"""What crosses ranks each tick: only a soma's spike slot. Every other property stays
rank-local unless a model opts a property in before setup() with
set_property_neighbor_visible(name, True), the hook for a custom kernel that reads a
neighbour's state on another rank (SAGESim keeps one flag per property name across
breeds, so opting in a name ships it for somas and synapses alike)."""

import unittest

from superneuroabm.model import NeuromorphicModel


def _visible(model):
    af = model._agent_factory
    af._neighbor_visible_indices = []
    af._build_neighbor_visible_indices()
    idx2name = {v: k for k, v in af._property_name_2_index.items()}
    return {idx2name[i] for i in af._neighbor_visible_indices}


def _two_somas(model):
    a = model.create_soma(breed="lif_soma", config_name="config_0")
    b = model.create_soma(breed="lif_soma", config_name="config_0")
    model.create_synapse(breed="delivery_synapse", pre_soma_id=a, post_soma_id=b,
                         config_name="config_0")
    return a, b


class TestNeighborVisibility(unittest.TestCase):

    def test_only_the_spike_slot_is_exchanged_by_default(self):
        m = NeuromorphicModel(enable_internal_states_tracking=False)
        _two_somas(m)
        m.setup()
        self.assertEqual(_visible(m), {"output_spikes_tensor"})

    def test_a_model_can_opt_a_property_in_before_setup(self):
        m = NeuromorphicModel(enable_internal_states_tracking=False)
        _two_somas(m)
        m.set_property_neighbor_visible("internal_states", True)
        m.setup()
        self.assertEqual(_visible(m), {"output_spikes_tensor", "internal_states"})

    def test_opting_in_after_setup_is_refused(self):
        m = NeuromorphicModel(enable_internal_states_tracking=False)
        _two_somas(m)
        m.setup()
        with self.assertRaises(RuntimeError):
            m.set_property_neighbor_visible("internal_states", True)


if __name__ == "__main__":
    unittest.main()
