"""A synapse from a soma to itself (pre == post) delivers, however the model is built.

create_synapse used to leave such a synapse with the neighbour list ``[pre]``: its two
connect_agents calls named the same soma and the ordered space kept a neighbour only once.
The kernel then found no post slot and the synapse delivered nothing, silently, while the
bulk builders kept ``[pre, post]`` and delivered. Found on the tutorial Brunel network,
whose random wiring draws 991 such synapses (docs/EVENT_DRIVEN_NOTES.md section 21). Every
route now goes through one builder; this keeps the case covered on each of them.
"""

import unittest

from superneuroabm.model import NeuromorphicModel

TICKS = 300
SYNAPSE_CONFIG = "config_1"        # weight 100: one input spike fires the soma


def _per_agent(autapse):
    model = NeuromorphicModel()
    model.set_seed(42)
    soma = model.create_soma(breed="lif_soma", config_name="config_0")
    ext = model.create_synapse(breed="delivery_synapse", pre_soma_id=-1, post_soma_id=soma,
                               config_name=SYNAPSE_CONFIG)
    syn = None
    if autapse:
        syn = model.create_synapse(breed="delivery_synapse", pre_soma_id=soma,
                                   post_soma_id=soma, config_name=SYNAPSE_CONFIG)
    return model, soma, ext, syn


def _bulk(autapse):
    model = NeuromorphicModel()
    model.set_seed(42)
    synapses = [{"id": 1, "pre": -1, "post": 0, "config": SYNAPSE_CONFIG}]
    if autapse:
        synapses.append({"id": 2, "pre": 0, "post": 0, "config": SYNAPSE_CONFIG})
    model.load_network({"somas": [{"id": 0}], "synapses": synapses})
    return model, 0, 1, 2 if autapse else None


def _plural(autapse):
    model = NeuromorphicModel()
    model.set_seed(42)
    soma = int(model.create_somas(1, "lif_soma", "config_0")[0])
    ext = int(model.create_synapses(-1, [soma], "delivery_synapse", SYNAPSE_CONFIG)[0])
    syn = None
    if autapse:
        syn = int(model.create_synapses([soma], [soma], "delivery_synapse", SYNAPSE_CONFIG)[0])
    return model, soma, ext, syn


def _spikes(build, autapse):
    model, soma, ext, syn = build(autapse)
    neighbours = model.get_synapse_connectivity(syn) if autapse else None
    model.setup()
    model.add_spike(ext, 2, 1.0)
    model.simulate(ticks=TICKS)
    return model.get_spike_times(soma), neighbours, soma


class TestAutapse(unittest.TestCase):
    def test_neighbour_list_keeps_both_slots(self):
        for build in (_per_agent, _bulk, _plural):
            _, neighbours, soma = _spikes(build, True)
            self.assertEqual(neighbours, [soma, soma], build.__name__)

    def test_autapse_delivers(self):
        """The soma re-excites itself: more spikes with the autapse than without."""
        for build in (_per_agent, _bulk, _plural):
            without, _, _ = _spikes(build, False)
            with_autapse, _, _ = _spikes(build, True)
            self.assertGreater(len(without), 0, build.__name__)
            self.assertGreater(len(with_autapse), len(without), build.__name__)

    def test_both_builds_give_the_same_spikes(self):
        per_agent, _, _ = _spikes(_per_agent, True)
        for build in (_bulk, _plural):
            self.assertEqual(per_agent, _spikes(build, True)[0], build.__name__)


if __name__ == "__main__":
    unittest.main()
