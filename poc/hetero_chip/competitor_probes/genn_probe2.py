"""GeNN workaround probe: per-synapse random numbers at spike time via a custom
connectivity update (row RNG) that refreshes a per-synapse var every step."""
import sys, os
import numpy as np
from pygenn import (GeNNModel, create_weight_update_model, init_weight_update, init_postsynaptic,
                    create_var_ref, create_wu_var_ref, create_custom_connectivity_update_model,
                    VarAccessMode, VarAccess)
os.chdir(sys.argv[1])
NPRE, NPOST = 8, 4
def build(seed):
    m = GeNNModel("float", f"t_rngwork_{seed}", backend="single_threaded_cpu")
    m.dt = 0.1; m.seed = seed
    pre = m.add_neuron_population("pre", NPRE, "SpikeSourceArray", {}, {"startSpike": np.arange(NPRE)*20, "endSpike": np.arange(NPRE)*20 + 20})
    pre.extra_global_params["spikeTimes"].set_init_values(np.tile(np.arange(20, dtype=np.float32) + 1.0, NPRE))
    post = m.add_neuron_population("post", NPOST, "LIF", {"C": 1.0, "TauM": 20.0, "Vrest": -65.0, "Vreset": -65.0, "Vthresh": -50.0, "Ioffset": 0.0, "TauRefrac": 2.0}, {"V": -65.0, "RefracTime": 0.0})
    wum = create_weight_update_model("stoch_dev", params=["v0", "gmax", "p"],
        vars=[("g", "scalar"), ("u", "scalar"), ("z", "scalar"), ("dg", "scalar", VarAccess.READ_ONLY)],
        post_neuron_var_refs=[("V", "scalar", VarAccessMode.READ_ONLY)],
        pre_spike_syn_code="if (u < p) { g = fmin(g + dg * (1.0 + 0.3 * z), gmax); }",
        synapse_dynamics_code="addToPost(g * sinh(V / v0));")
    n = NPRE * NPOST
    sg = m.add_synapse_population("dev", "SPARSE", pre, post,
        init_weight_update(wum, {"v0": 20.0, "gmax": 0.05, "p": 0.5}, {"g": 1e-4, "u": 1.0, "z": 0.0, "dg": np.linspace(1e-3, 8e-3, n)},
                           post_var_refs={"V": create_var_ref(post, "V")}),
        init_postsynaptic("DeltaCurr"))
    sg.set_sparse_connections(np.repeat(np.arange(NPRE), NPOST), np.tile(np.arange(NPOST), NPRE))
    fill = create_custom_connectivity_update_model("fill_rand", var_refs=[("u", "scalar"), ("z", "scalar")],
        row_update_code="for_each_synapse { u = gennrand_uniform(); z = gennrand_normal(); }")
    m.add_custom_connectivity_update("fill", "FillRand", sg, fill,
        var_refs={"u": create_wu_var_ref(sg, "u"), "z": create_wu_var_ref(sg, "z")})
    m.build(); m.load()
    for _ in range(300):
        m.custom_update("FillRand"); m.step_time()
    sg.vars["g"].pull_from_device()
    return sg.vars["g"].values.copy()
a, b, c = build(1), build(1), build(2)
print("per-synapse g (seed1):", np.round(a[:8], 5))
print("same seed identical:", np.array_equal(a, b), " different seed differs:", not np.array_equal(a, c))
