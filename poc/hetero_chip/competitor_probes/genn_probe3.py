"""GeNN: arbitrary device-to-device read via a writable extra global parameter (EGP) array."""
import sys, os
import numpy as np
from pygenn import (GeNNModel, create_weight_update_model, init_weight_update, init_postsynaptic,
                    create_var_ref, VarAccessMode)
os.chdir(sys.argv[1])
NPRE, NPOST = 8, 4
m = GeNNModel("float", "t_egp", backend="single_threaded_cpu"); m.dt = 0.1
pre = m.add_neuron_population("pre", NPRE, "SpikeSourceArray", {}, {"startSpike": np.arange(NPRE), "endSpike": np.arange(NPRE) + 1})
pre.extra_global_params["spikeTimes"].set_init_values(np.arange(NPRE, dtype=np.float32) + 1.0)
post = m.add_neuron_population("post", NPOST, "LIF", {"C": 1.0, "TauM": 20.0, "Vrest": -65.0, "Vreset": -65.0, "Vthresh": -50.0, "Ioffset": 0.0, "TauRefrac": 2.0}, {"V": -65.0, "RefracTime": 0.0})
wum = create_weight_update_model("xtalk", params=["k"], vars=[("g", "scalar"), ("nb", "scalar")],
    extra_global_params=[("mirror", "scalar*")],
    pre_spike_syn_code="g += 0.001;",
    synapse_dynamics_code="""
    mirror[id_pre * 4 + id_post] = g;                      // publish own state
    const unsigned int jn = (id_pre + 1) % 8;              // diagonal neighbour (other row AND column)
    nb = mirror[jn * 4 + ((id_post + 1) % 4)];
    g += k * nb;
    """)
sg = m.add_synapse_population("dev", "SPARSE", pre, post,
    init_weight_update(wum, {"k": 1e-3}, {"g": 1e-4, "nb": 0.0}), init_postsynaptic("DeltaCurr"))
sg.set_sparse_connections(np.repeat(np.arange(NPRE), NPOST), np.tile(np.arange(NPOST), NPRE))
sg.extra_global_params["mirror"].set_init_values(np.zeros(NPRE * NPOST, dtype=np.float32))
m.build(); m.load()
for _ in range(100): m.step_time()
sg.vars["nb"].pull_from_device()
print("neighbour state read via writable EGP:", sg.vars["nb"].values[:6])
