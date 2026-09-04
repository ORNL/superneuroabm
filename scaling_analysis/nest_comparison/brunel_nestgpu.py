"""NEST GPU Brunel setup-time benchmark, single GPU.

Matches the SuperNeuroABM w=1 Brunel point measured by ``brunel_sna_local.py``:
same neuron count, same exact in-degree, same LIF time constants, same
sub-threshold 10 Hz Poisson drive, same 100 ms of biological time.

Differences that cannot be removed, and are disclosed in the write-up:
  * wiring: NEST GPU draws sources globally (fixed_indegree); SuperNeuroABM
    draws them inside a radius-8 ball on a periodic 3D lattice. NEST GPU's
    cost is wiring-agnostic, so this favours neither side on setup time.
  * refractory period: SuperNeuroABM integrates during refractoriness,
    iaf_psc_exp does not. Irrelevant here because the network is silent.
  * delay: SuperNeuroABM truncates its 1.5 ms delay to one 1 ms tick, so this
    script uses 1.0 ms.
"""

import argparse
import math
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import bench_common as bc

# SuperNeuroABM lif_soma config_0 (superneuroabm/component_base_config.yaml:2-17)
C_M_PF = 10_000.0    # C = 10 nF
TAU_M_MS = 10.0      # R * C = 1e6 ohm * 10e-9 F
E_L_MV = -60.0       # vrest
V_TH_MV = -45.0      # vthr
V_RESET_MV = -60.0   # vreset
T_REF_MS = 5.0       # tref
TAU_SYN_MS = 10.0    # single_exp_synapse tau_fall = 1e-2 s


def psp_peak_weight(target_psp_mv):
    """Current amplitude (pA) whose PSP peaks at target_psp_mv.

    For iaf_psc_exp with tau_m == tau_syn == tau the response to one spike is
    V(t) = (J/C) t exp(-t/tau), peaking at t = tau with V = J tau / (C e).
    """
    if abs(TAU_M_MS - TAU_SYN_MS) < 1e-9:
        peak_per_pa = TAU_M_MS / (C_M_PF * math.e)
    else:
        tm, ts = TAU_M_MS, TAU_SYN_MS
        t_peak = math.log(tm / ts) / (1.0 / ts - 1.0 / tm)
        peak_per_pa = (tm * ts / (C_M_PF * (tm - ts))) * (
            math.exp(-t_peak / tm) - math.exp(-t_peak / ts))
    return target_psp_mv / peak_per_pa


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--neurons", type=int, default=12500)
    p.add_argument("--in-degree", type=int, default=1000)
    p.add_argument("--dt-ms", type=float, default=1.0)
    p.add_argument("--ticks", type=int, default=100)
    p.add_argument("--warmup-ticks", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--g", type=float, default=5.0)
    p.add_argument("--psp-mv", type=float, default=0.1,
                   help="Excitatory PSP peak the weight is calibrated to.")
    p.add_argument("--firing-rate", type=float, default=10.0)
    p.add_argument("--csv", default=str(SCRIPT_DIR / "results" / "brunel_nestgpu.csv"))
    return p


def run_once(args, repeat):
    import nestgpu as ngpu

    log = bc.PhaseLog("nestgpu", "fixed_indegree", args.neurons,
                      args.in_degree, repeat=repeat)
    t_pipeline = time.time()

    NE = int(0.8 * args.neurons)
    NI = args.neurons - NE
    C_E = round(0.8 * args.in_degree)
    C_I = args.in_degree - C_E
    J_E = psp_peak_weight(args.psp_mv)
    J_I = -args.g * J_E
    delay_ms = max(args.dt_ms, 1.0)

    # Order matters: the kernel refuses a time resolution larger than the
    # current minimum allowed delay, so raise the delay bound first.
    ngpu.SetKernelStatus("min_allowed_delay", max(args.dt_ms, 1.0))
    ngpu.SetKernelStatus("time_resolution", args.dt_ms)
    ngpu.SetKernelStatus("rnd_seed", args.seed)
    ngpu.SetKernelStatus("verbosity_level", 1)

    # ---- create ----
    t0 = time.time()
    neurons = ngpu.Create("iaf_psc_exp", args.neurons)
    ngpu.SetStatus(neurons, {
        "tau_m": TAU_M_MS, "C_m": C_M_PF, "E_L": E_L_MV, "I_e": 0.0,
        "Theta_rel": V_TH_MV - E_L_MV, "V_reset_rel": V_RESET_MV - E_L_MV,
        "tau_ex": TAU_SYN_MS, "tau_in": TAU_SYN_MS, "t_ref": T_REF_MS,
    })
    pg = ngpu.Create("poisson_generator")
    ngpu.SetStatus(pg, "rate", args.firing_rate)
    log.add("create", time.time() - t0)

    exc = neurons[0:NE]
    inh = neurons[NE:args.neurons]

    # ---- connect ----
    t0 = time.time()
    ngpu.Connect(exc, neurons,
                 {"rule": "fixed_indegree", "indegree": C_E},
                 {"weight": J_E, "delay": delay_ms})
    ngpu.Connect(inh, neurons,
                 {"rule": "fixed_indegree", "indegree": C_I},
                 {"weight": J_I, "delay": delay_ms})
    # One external input per neuron, weight J_E, matching SuperNeuroABM's single
    # pre == -1 input synapse per soma.
    ngpu.Connect(pg, neurons, {"rule": "all_to_all"},
                 {"weight": J_E, "delay": delay_ms})
    log.add("connect", time.time() - t0)

    n_syn = args.neurons * (args.in_degree + 1)
    log.set_synapses(n_syn)

    try:
        ngpu.ActivateSpikeCount(neurons)
    except Exception as exc_err:            # counting is a nicety, not the measurement
        print(f"  (spike count unavailable: {exc_err})")

    # ---- calibrate (NEST GPU's build-the-device-structures phase) ----
    t0 = time.time()
    ngpu.Calibrate()
    log.add("calibrate", time.time() - t0)

    # ---- simulate ----
    warm_ms = args.warmup_ticks * args.dt_ms
    rest_ms = (args.ticks - args.warmup_ticks) * args.dt_ms
    t0 = time.time()
    ngpu.Simulate(warm_ms)
    log.add(f"simulate_warmup_{args.warmup_ticks}", time.time() - t0)
    t0 = time.time()
    ngpu.Simulate(rest_ms)
    log.add(f"ticks_{args.warmup_ticks + 1}_{args.ticks}", time.time() - t0)

    rate = float("nan")
    try:
        counts = ngpu.GetStatus(neurons, "spike_count")
        total = sum(float(c[0]) if isinstance(c, (list, tuple)) else float(c)
                    for c in counts)
        rate = total / args.neurons / (args.ticks * args.dt_ms / 1000.0)
    except Exception:
        pass

    log.add("total", time.time() - t_pipeline)

    bc.print_table(log.rows, f"NEST GPU Brunel  N={args.neurons} "
                             f"K={args.in_degree} synapses={n_syn:,}")
    print(f"\n  J_E = {J_E:.2f} pA  (PSP peak {args.psp_mv} mV), "
          f"J_I = {J_I:.2f} pA, delay = {delay_ms} ms")
    print(f"  mean firing rate: {rate:.3f} Hz")
    return log


def main():
    args = build_parser().parse_args()
    path = None
    for r in range(args.repeat):
        print(f"\n########## repeat {r + 1}/{args.repeat} ##########")
        # NEST GPU keeps one global kernel; a fresh process per repeat is the
        # only clean reset, so repeats beyond the first are run by the driver.
        log = run_once(args, r)
        path = log.write(args.csv)
        break
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
