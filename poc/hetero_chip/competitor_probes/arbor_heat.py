import sys, arbor as A
from arbor import units as U
cat = A.load_catalogue(sys.argv[1])
NH = int(sys.argv[2])
class R(A.recipe):
    def __init__(self):
        super().__init__(); self.props = A.neuron_cable_properties(); self.props.catalogue.extend(cat, "")
        self.props.set_ion("heat", valence=1, int_con=0*U.mM, ext_con=0*U.mM, rev_pot=0*U.mV, diff=1e-9*U.m2/U.s)
    def num_cells(self): return 1
    def cell_kind(self, gid): return A.cell_kind.cable
    def global_properties(self, kind): return self.props
    def cell_description(self, gid):
        t = A.segment_tree(); t.append(A.mnpos, A.mpoint(-3,0,0,3), A.mpoint(3,0,0,3), tag=1)
        d = A.decor()
        for k in range(NH): d.place('(location 0 0.5)', A.synapse("heater", {"p": 0.001*(k+1)}), f"h{k}")
        d.place('(location 0 0.5)', A.synapse("sensor"), "s")
        return A.cable_cell(t, d, A.label_dict(), A.cv_policy_single())
    def probes(self, gid): return [A.cable_probe_point_state_cell("sensor", "seen", "seen")]
sim = A.simulation(R())
h = sim.sample((0, "seen"), A.regular_schedule(5*U.ms))
sim.run(10*U.ms, 0.025*U.ms)
print(f"heaters={NH}: sensor integral of shared heatd =", [d for d, m in sim.samples(h)][0][:, 1])
