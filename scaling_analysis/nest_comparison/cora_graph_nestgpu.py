"""Can NEST GPU build the SGNN Cora network (irregular degrees, per-edge weights, STDP readout)?

Builds every synapse of ~/superneuroabm_sgnn/cora_network.pkl from explicit edge lists
(one_to_one rule), then reads the connections back and checks edge count, per-edge weight,
fan-in of every neuron, and which connections carry the STDP group. Graph only: no simulation.

    source ~/software/nest-gpu/install/bin/nestgpu_vars.sh
    /home/xxz/miniforge3/envs/sna-dev/bin/python cora_graph_nestgpu.py

Result 2026-10-01 (NEST GPU 2.0.0): 31 836 neuron->neuron connections identical to the file
(source, target, weight, plastic flag), identical fan-in for every neuron (1..1520), 1000
external inputs. Exit code 0 when everything matches.
"""
import collections
import ctypes
import pickle
import sys

import numpy as np
import nestgpu as ngpu

net = pickle.load(open("/home/xxz/superneuroabm_sgnn/cora_network.pkl", "rb"))
somas, syns = net["somas"], net["synapses"]
N = len(somas)
assert [s["id"] for s in somas] == list(range(N))


def hp(e, k, default):
    return float(e.get("overrides", {}).get("hyperparameters", {}).get(k, default))


def carr(x):
    return (ctypes.c_float * len(x))(*x)


# Synaptic time constant is a property of the receptor port in NEST GPU, not of the synapse:
# map each distinct tau_fall to a port of a multisynapse neuron.
taus = sorted({hp(e, "tau_fall", 1e-2) for e in syns})
port_of = {t: i for i, t in enumerate(taus)}
print("distinct tau_fall:", taus, "-> receptor ports", port_of)

neurons = ngpu.Create("aeif_psc_exp_multisynapse", N, len(taus))
ngpu.SetStatus(neurons, {"tau_syn": [t * 1000.0 for t in taus]})   # ms
stdp = ngpu.CreateSynGroup("stdp")

internal = [e for e in syns if e["pre"] >= 0]
external = [e for e in syns if e["pre"] < 0]
gens = ngpu.Create("spike_generator", len(external))

# One Connect call per (port, plastic?) bucket, each an explicit edge list with per-edge weights.
buckets = collections.defaultdict(list)
for e in internal:
    buckets[(port_of[hp(e, "tau_fall", 1e-2)], e.get("learning_rule") is not None)].append(e)
for (port, plastic), es in sorted(buckets.items()):
    src = [e["pre"] for e in es]
    tgt = [e["post"] for e in es]
    syn = {"weight": {"array": carr([hp(e, "weight", 14.0) for e in es])},
           "delay": 1.0, "receptor": port}
    if plastic:
        syn["synapse_group"] = stdp
    ngpu.Connect([neurons[i] for i in src], [neurons[i] for i in tgt], {"rule": "one_to_one"}, syn)
    print(f"port {port} plastic={plastic}: {len(es)} edges")

ext_tgt = [neurons[e["post"]] for e in external]
ngpu.Connect([gens[i] for i in range(len(external))], ext_tgt, {"rule": "one_to_one"},
             {"weight": {"array": carr([hp(e, "weight", 14.0) for e in external])},
              "delay": 1.0, "receptor": port_of[hp(external[0], "tau_fall", 1e-2)]})

ngpu.Calibrate()

# ---- read back and compare with the file
conns = ngpu.GetConnections(neurons, neurons)
st = list(zip(*ngpu.GetStatus(conns, ["source", "target", "weight", "syn_group"])))  # columns -> rows
got = collections.Counter((int(s), int(t), round(float(w), 4), int(g) > 0) for s, t, w, g in st)
want = collections.Counter((e["pre"], e["post"], round(hp(e, "weight", 14.0), 4),
                            e.get("learning_rule") is not None) for e in internal)
print(f"neuron->neuron connections: built {sum(got.values())}, file {sum(want.values())}")
print("identical edge multiset (source, target, weight, plastic):", got == want)

n0 = neurons[0]
fin_got = collections.Counter(int(t) - n0 for _, t, _, _ in st)
fin_want = collections.Counter(e["post"] for e in internal)
print("identical fan-in for every neuron:", fin_got == fin_want,
      "| fan-in min/max:", min(fin_want.values()), max(fin_want.values()))
ext = ngpu.GetConnections(gens, neurons)
print("external input connections:", len(ext), "file:", len(external))
sys.exit(0 if (got == want and fin_got == fin_want and len(ext) == len(external)) else 1)
