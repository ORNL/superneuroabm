"""GeNN 5.5.0 probe (single_threaded_cpu backend). Each test builds its own model."""
import sys, os, traceback
import numpy as np
import pygenn as G
from pygenn import (GeNNModel, create_weight_update_model, init_weight_update,
                    init_postsynaptic, init_sparse_connectivity, create_var_ref, create_wu_var_ref,
                    create_custom_connectivity_update_model, VarAccessMode, VarAccess)
WORK = sys.argv[1]
os.chdir(WORK)
NPRE, NPOST = 8, 4
LIF = {"C": 1.0, "TauM": 20.0, "Vrest": -65.0, "Vreset": -65.0, "Vthresh": -50.0, "Ioffset": 0.0, "TauRefrac": 2.0}

def base_model(name):
    m = GeNNModel("float", name, backend="single_threaded_cpu")
    m.dt = 0.1
    pre = m.add_neuron_population("pre", NPRE, "SpikeSourceArray", {}, {"startSpike": np.arange(NPRE), "endSpike": np.arange(NPRE) + 1})
    pre.extra_global_params["spikeTimes"].set_init_values(np.arange(NPRE, dtype=np.float32) * 1.0 + 1.0)
    post = m.add_neuron_population("post", NPOST, "LIF", LIF, {"V": -65.0, "RefracTime": 0.0})
    return m, pre, post

def device_model(name, pre_code, dyn_code):
    return create_weight_update_model(
        name, params=["v0", "gmax"],
        vars=[("g", "scalar"), ("n", "scalar"), ("dg", "scalar", VarAccess.READ_ONLY)],
        post_neuron_var_refs=[("V", "scalar", VarAccessMode.READ_ONLY)],
        pre_spike_syn_code=pre_code, synapse_dynamics_code=dyn_code)

def add_devices(m, pre, post, wum, popname):
    # all-to-all as SPARSE, per-synapse dg
    n = NPRE * NPOST
    pre_ind = np.repeat(np.arange(NPRE), NPOST); post_ind = np.tile(np.arange(NPOST), NPRE)
    sg = m.add_synapse_population(popname, "SPARSE", pre, post,
        init_weight_update(wum, {"v0": 20.0, "gmax": 0.05},
                           {"g": 1e-4, "n": 0.0, "dg": np.linspace(1e-3, 8e-3, n)},
                           post_var_refs={"V": create_var_ref(post, "V")}),
        init_postsynaptic("DeltaCurr"))
    sg.set_sparse_connections(pre_ind, post_ind)
    return sg

def run_test(label, fn):
    try:
        out = fn()
        print(f"[PASS] {label}: {out}")
    except Exception as e:
        msg = str(e).strip().splitlines()
        print(f"[FAIL] {label}: {type(e).__name__}: {' | '.join(msg[-3:])[:400]}")

DYN = "addToPost(g * sinh(V / v0));"     # every-step, reads V_post, injects current
PRE_DET = "n += 1.0; g = fmin(g + dg, gmax);"

def t_every_step():
    m, pre, post = base_model("t_every_step")
    sg = add_devices(m, pre, post, device_model("memdev", PRE_DET, DYN), "dev")
    m.build(); m.load()
    for _ in range(200): m.step_time()
    sg.vars["g"].pull_from_device(); post.vars["V"].pull_from_device()
    return f"g[:6]={sg.vars['g'].values[:6]} V={post.vars['V'].values}"

def t_rng_pre_spike():
    m, pre, post = base_model("t_rng_pre")
    code = "n += 1.0; if (gennrand_uniform() < 0.5) { g = fmin(g + dg * (1.0 + 0.3 * gennrand_normal()), gmax); }"
    sg = add_devices(m, pre, post, device_model("memdev_rng", code, DYN), "dev")
    m.build(); m.load()
    for _ in range(200): m.step_time()
    sg.vars["g"].pull_from_device()
    return f"g[:6]={sg.vars['g'].values[:6]}"

def t_rng_dyn():
    m, pre, post = base_model("t_rng_dyn")
    code = "g += 1e-6 * gennrand_normal(); addToPost(g * sinh(V / v0));"
    sg = add_devices(m, pre, post, device_model("memdev_rngdyn", PRE_DET, code), "dev")
    m.build(); m.load()
    for _ in range(50): m.step_time()
    sg.vars["g"].pull_from_device()
    return f"g[:4]={sg.vars['g'].values[:4]}"

def t_mixed():
    m, pre, post = base_model("t_mixed")
    a = add_devices(m, pre, post, device_model("tech_a", PRE_DET, DYN), "techA")
    b = add_devices(m, pre, post, device_model("tech_b", "g = gmax - (gmax - g) * 0.9;", "addToPost(g * (V + 65.0));"), "techB")
    m.build(); m.load()
    for _ in range(200): m.step_time()
    post.vars["V"].pull_from_device()
    return f"two WU models -> same post Isyn OK, V={post.vars['V'].values}"

def t_row_interaction():
    # custom connectivity update: every synapse in a presynaptic row ("row line") reads the
    # sum of its row-mates' g (IR-drop-like), with RNG in the row code.
    m, pre, post = base_model("t_row")
    sg = add_devices(m, pre, post, device_model("memdev_r", PRE_DET, DYN), "dev")
    ccu_model = create_custom_connectivity_update_model(
        "row_couple", params=["k"],
        var_refs=[("g", "scalar")],
        post_var_refs=[("V", "scalar", VarAccessMode.READ_ONLY)],
        row_update_code="""
        scalar tot = 0.0;
        for_each_synapse { tot += g; }
        for_each_synapse { g = g - k * (tot - g) * (1.0 + 0.01 * gennrand_normal()) * (V + 66.0) * 0.0; g = g * (1.0 - k * (tot - g)); }
        """)
    m.add_custom_connectivity_update("row", "RowCouple", sg, ccu_model, {"k": 0.01},
                                     var_refs={"g": create_wu_var_ref(sg, "g")},
                                     post_var_refs={"V": create_var_ref(post, "V")})
    m.build(); m.load()
    for _ in range(200):
        m.step_time(); m.custom_update("RowCouple")
    sg.vars["g"].pull_from_device()
    return f"g[:6]={sg.vars['g'].values[:6]}"

for label, fn in [("1 every-step addToPost(g*sinh(V_post/v0)) + per-syn dg + spike-time state", t_every_step),
                  ("2 gennrand_* in pre_spike_syn_code", t_rng_pre_spike),
                  ("3 gennrand_normal in synapse_dynamics_code", t_rng_dyn),
                  ("4 two WU models (tech A/B) into same post neuron", t_mixed),
                  ("5 custom connectivity row update: synapse reads row-mates + RNG + V_post", t_row_interaction)]:
    run_test(label, fn)
