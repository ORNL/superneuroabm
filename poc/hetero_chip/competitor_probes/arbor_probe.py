"""Arbor 0.12.2 probe: device-per-synapse on single-CV cable cells.
Mixed mechanisms per neuron, per-instance params, every-step v-dependent current,
event-time randomness hack, invariance across cell-group decomposition/threads."""
import sys, hashlib
import numpy as np
import arbor as A
from arbor import units as U

CAT = sys.argv[1]
cat = A.load_catalogue(CAT)
N_CELLS, K = 6, 9   # K synapses per cell: 3 x memdev, 3 x memdev_ev, 3 x memdev_sde
MECH_SETS = sys.argv[2] if len(sys.argv) > 2 else "all"   # "all" or "hetero"

def mechs_for(gid):
    if MECH_SETS == "hetero" and gid == 0:
        return ["memdev_sde"] * K              # cell 0 has only one mech type
    return ["memdev", "memdev_ev", "memdev_sde"] * (K // 3)

class R(A.recipe):
    def __init__(self):
        super().__init__()
        self.props = A.neuron_cable_properties()
        self.props.catalogue.extend(cat, "")
    def num_cells(self): return N_CELLS
    def cell_kind(self, gid): return A.cell_kind.cable
    def global_properties(self, kind): return self.props
    def cell_description(self, gid):
        tree = A.segment_tree()
        tree.append(A.mnpos, A.mpoint(-3, 0, 0, 3), A.mpoint(3, 0, 0, 3), tag=1)
        d = A.decor().set_property(Vm=-65*U.mV).paint('(all)', A.density('pas'))
        for k, m in enumerate(mechs_for(gid)):
            # per-instance parameters: each synapse its own step size / threshold
            p = {"dg": 1e-4 * (1 + k + 10 * gid)}
            if m == "memdev_ev": p["thr"] = -0.5 + 0.1 * k
            d.place('(location 0 0.5)', A.synapse(m, p), f"syn{k}")
        d.place('(location 0 0.5)', A.threshold_detector(-10*U.mV), "det")
        return A.cable_cell(tree, d, A.label_dict(), A.cv_policy_single())
    def event_generators(self, gid):
        return [A.event_generator(f"syn{k}", 1.0, A.regular_schedule((0.5 + 0.1 * k)*U.ms, 2.0*U.ms))
                for k in range(K)]
    def probes(self, gid):
        pr = [A.cable_probe_membrane_voltage('(location 0 0.5)', "v")]
        for m in set(mechs_for(gid)):
            pr.append(A.cable_probe_point_state_cell(m, "g", f"g_{m}"))
        return pr

def run(threads, group_size):
    rec = R()
    ctx = A.context(threads=threads)
    hints = {A.cell_kind.cable: A.partition_hint(cpu_group_size=group_size)}
    dd = A.partition_load_balance(rec, ctx, hints)
    sim = A.simulation(rec, ctx, dd, seed=42)
    hs = {}
    for gid in range(N_CELLS):
        hs[(gid, "v")] = sim.sample((gid, "v"), A.regular_schedule(0.5*U.ms))
        for m in set(mechs_for(gid)):
            hs[(gid, m)] = sim.sample((gid, f"g_{m}"), A.regular_schedule(0.5*U.ms))
    sim.run(50*U.ms, 0.025*U.ms)
    h = hashlib.sha256()
    out = {}
    for key in sorted(hs, key=str):
        arrs = [d for d, meta in sim.samples(hs[key])]
        a = np.concatenate([np.asarray(x) for x in arrs], axis=1)
        out[key] = a
        h.update(a.tobytes())
    return dd.num_groups, h.hexdigest()[:16], out

res = {}
for threads, gs in [(1, 6), (1, 1), (4, 1), (2, 3), (4, 2)]:
    ng, dig, out = run(threads, gs)
    res[(threads, gs)] = out
    print(f"mechsets={MECH_SETS} threads={threads} group_size={gs} groups={ng} digest={dig}")
base = res[(1, 6)]
for k, out in res.items():
    diffs = [key for key in base if not np.array_equal(base[key], out[key])]
    print(k, "differs from (1,6) in:", diffs[:6])
o = res[(1, 6)]
print("cell1 memdev g final (per-instance dg):", o[(1, "memdev")][-1, 1:])
print("cell1 memdev_ev g final (stochastic):", o[(1, "memdev_ev")][-1, 1:])
print("cell1 v range:", o[(1, "v")][:,1].min(), o[(1, "v")][:,1].max()); print("shape", o[(1,"memdev")].shape)
for k, out in res.items():
    md = max(float(np.max(np.abs(base[key][:, 1:] - out[key][:, 1:]))) for key in base)
    print(k, "max |diff| over all traces:", md)
print("cell1 memdev g at t=5ms (dg differs per instance):", o[(1, "memdev")][10, 1:])
