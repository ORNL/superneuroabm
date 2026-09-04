"""Masquelier 2008 on SuperNeuroABM: one 10 s chunk, with phase timing.

Same experiment and the same spike-train generator as
``/home/xxz/ns-applications/duplicates/masquelier_2008/run_experiment_hg.py``,
reduced to a single chunk and instrumented so the phases line up with
``masquelier_nestgpu.py``. The upstream script is not modified.
"""

import argparse
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MASQ_DIR = Path("/home/xxz/ns-applications/duplicates/masquelier_2008")
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(MASQ_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent.parent))

import numpy as np

import bench_common as bc


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--dt-s", type=float, default=1e-4)
    p.add_argument("--afferents", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--csv", default=str(SCRIPT_DIR / "results" / "masquelier.csv"))
    p.add_argument("--tag", default="", help="Suffix for the variant label, e.g. _afterfix.")
    p.add_argument("--dump-dir", default=None,
                   help="Save all output spike times and all synapse weights as .npy, "
                        "for a bitwise before/after comparison.")
    p.add_argument("--fused", action="store_true",
                   help="Leave verbose_timing off so SAGESim fuses all ticks into one "
                        "kernel launch (the single-worker fast path). Setup sub-timers "
                        "and per-tick records are then unavailable.")
    return p


def run(args):
    import experiment_utils as eu
    from superneuroabm.model import NeuromorphicModel
    from exp_pair_wise_stdp_bounded_nn import exp_pair_wise_stdp_bounded_nn
    from masq_hg_lif_soma import masq_hg_lif_soma_step_func
    from masq_single_exp_synapse import masq_single_exp_synapse_step_func

    log = bc.PhaseLog("sna", ("masquelier-fused" if args.fused else "masquelier") + args.tag,
                      args.afferents, 1, synapses=args.afferents)
    t_pipeline = time.time()

    # ---- spike trains: identical generator, identical seed ----
    n_ticks_gen = int(args.seconds / eu.DT)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    rates = eu.generate_rate_profiles(args.afferents, n_ticks_gen, rng)
    trains = eu.generate_poisson_spikes(rates, rng)
    pattern = eu.generate_pattern(eu.N_PATTERN, eu.PATTERN_DURATION, rng)
    trains, _ = eu.embed_pattern(trains, pattern, n_ticks_gen, rng)
    log.add("generate_spikes", time.time() - t0)

    upscale = int(round(eu.DT / args.dt_s))          # 1 ms generator -> dt_s ticks
    sim_ticks = int(args.seconds / args.dt_s)
    n_spikes = sum(len(t) for t in trains)

    # ---- build the model ----
    t0 = time.time()
    model = NeuromorphicModel(
        soma_breed_info={"masq_hg_lif_soma":
            (masq_hg_lif_soma_step_func, MASQ_DIR / "masq_hg_lif_soma.py")},
        synapse_breed_info={"masq_single_exp_synapse":
            (masq_single_exp_synapse_step_func, MASQ_DIR / "masq_single_exp_synapse.py")},
        learning_rule_info={},
        user_config=MASQ_DIR / "masquelier_config.yaml",
        enable_internal_states_tracking=False,
    )
    stdp_id = model.register_learning_rule(
        exp_pair_wise_stdp_bounded_nn, MASQ_DIR / "exp_pair_wise_stdp_bounded_nn.py")
    model.set_global_property_value("dt", args.dt_s)
    soma = model.create_soma(breed="masq_hg_lif_soma",
                             config_name="masquelier_hg_config_0")
    synapses = [
        model.create_synapse(
            breed="masq_single_exp_synapse", pre_soma_id=-1, post_soma_id=soma,
            config_name="masquelier_config_0",
            learning_rule="exp_pair_wise_stdp_bounded_nn",
            learning_rule_config="masquelier_default",
            overrides={"learning_hyperparameters": {"stdp_type": float(stdp_id)}},
        )
        for _ in range(args.afferents)
    ]
    log.add("create", time.time() - t0)

    # ---- setup (codegen + JIT) ----
    # verbose_timing is what exposes the setup and first-tick sub-timers, but it also
    # unfuses the single-worker tick loop (SAGESim/sagesim/model.py:1251), turning one
    # kernel launch into one per tick. --fused measures the path a user actually gets.
    model.verbose_timing = not args.fused
    t0 = time.time()
    model.setup()
    setup_wall = time.time() - t0
    st = dict(getattr(model, "_setup_timings", {}))
    log.add_all("setup", {k: v for k, v in st.items() if k != "total"})
    log.add("setup.other", max(0.0, setup_wall - sum(
        v for k, v in st.items() if k != "total")))
    log.add("setup.total", setup_wall)

    # ---- inject the chunk's spikes ----
    t0 = time.time()
    for i, syn_id in enumerate(synapses):
        ticks = np.asarray(trains[i], dtype=np.int64) * upscale
        if ticks.size:
            model.add_local_spike_list(syn_id, [[int(t), 1.0] for t in ticks])
    log.add("load_spike_trains", time.time() - t0)

    # ---- simulate one chunk ----
    t0 = time.time()
    model.simulate(ticks=sim_ticks, update_data_ticks=1)
    sim_wall = time.time() - t0
    tick_rows = getattr(model, "_tick_timings", []) or []
    first = tick_rows[0] if tick_rows else {}
    log.add("first_tick.total", first.get("total", 0.0))
    log.add("first_tick.gpu_build", first.get("gpu_buffer_build", 0.0))
    log.add("simulate", sim_wall)
    log.add("tick_path", 0.0 if args.fused else 1.0)
    log.add("construction_in_simulate", getattr(model, "_construction_time", 0.0))
    log.add("total", time.time() - t_pipeline)

    out_spikes = model.get_spike_times(soma)
    weights = [model.get_agent_property_value(id=s, property_name="hyperparameters")[0]
               for s in synapses[:50]]
    if args.dump_dir:
        import os
        os.makedirs(args.dump_dir, exist_ok=True)
        all_w = np.array([model.get_agent_property_value(
            id=s, property_name="hyperparameters")[0] for s in synapses], dtype=np.float64)
        np.save(os.path.join(args.dump_dir, "weights_all.npy"), all_w)
        np.save(os.path.join(args.dump_dir, "spike_times.npy"),
                np.asarray(out_spikes, dtype=np.int64))
        print(f"  dumped {all_w.size} weights and {len(out_spikes)} spike times "
              f"to {args.dump_dir}")

    bc.print_table(log.rows, f"SuperNeuroABM Masquelier  {args.afferents} afferents, "
                             f"{args.seconds:g}s @ dt={args.dt_s*1000:g}ms "
                             f"({sim_ticks:,} ticks)")
    print(f"\n  input spikes injected: {n_spikes:,}")
    print(f"  output spikes: {len(out_spikes)}")
    print(f"  first 5 weights: {[round(w, 4) for w in weights[:5]]}")
    log.write(args.csv)
    print(f"  wrote {args.csv}")


if __name__ == "__main__":
    run(build_parser().parse_args())
