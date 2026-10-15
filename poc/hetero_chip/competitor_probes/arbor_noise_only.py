import sys, arbor as A
from arbor import units as U
cat = A.load_catalogue(sys.argv[1])
class R(A.recipe):
    def __init__(self):
        super().__init__(); self.props = A.neuron_cable_properties(); self.props.catalogue.extend(cat, "")
    def num_cells(self): return 1
    def cell_kind(self, gid): return A.cell_kind.cable
    def global_properties(self, kind): return self.props
    def cell_description(self, gid):
        t = A.segment_tree(); t.append(A.mnpos, A.mpoint(-3,0,0,3), A.mpoint(3,0,0,3), tag=1)
        d = A.decor(); d.place('(location 0 0.5)', A.synapse("memdev_noise"), "s")
        return A.cable_cell(t, d, A.label_dict(), A.cv_policy_single())
    def event_generators(self, gid): return [A.event_generator("s", 1.0, A.regular_schedule(1*U.ms, 1*U.ms))]
    def probes(self, gid): return [A.cable_probe_point_state_cell("memdev_noise", "g", "g")]
sim = A.simulation(R())
h = sim.sample((0, "g"), A.regular_schedule(1*U.ms))
sim.run(10*U.ms, 0.025*U.ms)
print([d for d, m in sim.samples(h)][0][:, 1])
