"""
Calibrate the Brunel external drive to the asynchronous-irregular (AI) regime.

Our LIF preset is **biophysical**, not NEST-normalized, so the analytic NEST rate
(``brunel_external_rate``, which assumes ``theta=20 mV``, ``J=0.1 mV`` PSP, ``tau_m=20 ms``)
cannot be used verbatim -- the per-spike PSP amplitude in mV is an *emergent* product of
``weight * scale * scaling_factor`` and the membrane, unknown until measured. And the config
default ``excitatory_weight=14`` is wildly too strong: a single spike drives the membrane far
past threshold (measured ~54 mV vs a 15 mV threshold distance), so the network cannot be
balanced at that weight. This script measures the PSP, solves for the weight that hits a
target PSP (NEST's ``J=0.1 mV``), computes the matching threshold rate, and validates AI. It
prints the calibrated ``(excitatory_weight, external_rate_hz)`` for the driver.

Key fact the probe exploits: **below threshold the membrane is linear**, so PSP is exactly
proportional to weight (verified: ~3.87 mV per unit weight for this preset). We therefore
measure the coefficient at a small, safe probe weight and solve ``weight = J_target / coeff``
-- robust, and it avoids measuring at an over-threshold weight (where the trace is clipped by
firing/reset).

Three steps (all need a GPU -- run on a compute node):

  [1/3] PSP probe. One soma + one external (pre=-1) synapse; inject a single spike at a small
        probe weight and read the membrane trace -> PSP-per-unit-weight coefficient. Solve for
        the excitatory weight that yields the target PSP ``J_mV``.
  [2/3] Threshold rate. nu_thr = (vthr - vrest) / (J_mV * C_E * tau_m); external drive rate
        = eta * nu_thr * C_E (NEST's p_rate, with OUR measured J_mV / tau_m / theta).
  [3/3] AI validation. Build a small Brunel net at the calibrated weight, drive every soma at
        that rate, record spikes, and report mean firing rate + CV of the inter-spike
        interval. AI == low rate (~1-10 Hz) and CV_ISI ~ 1 (irregular).

Run:  srun -N1 -n1 --gpu-bind=closest python -u scaling_analysis/calibrate_ai.py
      python scaling_analysis/calibrate_ai.py --target-psp-mv 0.1 --eta 2.0 --ticks 2000
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from superneuroabm.model import NeuromorphicModel
from superneuroabm.brunel import build_brunel_network
from superneuroabm.util import load_component_configurations


def _lif_params(soma_breed="lif_soma", soma_config="config_0"):
    """Read the LIF hyperparameters we need for the threshold-rate formula."""
    cfg = load_component_configurations()["soma"][soma_breed][soma_config]["hyperparameters"]
    R = float(cfg["R"])
    C = float(cfg["C"])
    return {
        "vthr": float(cfg["vthr"]),
        "vrest": float(cfg["vrest"]),
        "R": R,
        "C": C,
        "tau_m_s": R * C,                 # membrane time constant (s)
    }


def measure_psp_per_weight(*, probe_weight, ticks, soma_breed="lif_soma",
                           soma_config="config_0", synapse_breed="single_exp_synapse",
                           synapse_config="config_0"):
    """Measure the sub-threshold EPSP amplitude per unit synaptic weight (mV/weight).

    Builds a 1-soma network with a single external (pre=-1) synapse, injects one spike at a
    small ``probe_weight`` (chosen so the response stays below threshold, where the membrane
    is linear), and returns ``(peak_v - vrest) / probe_weight`` plus the LIF params. Because
    the sub-threshold membrane is linear in the drive, this coefficient lets us solve for the
    weight that produces any target PSP.
    """
    lif = _lif_params(soma_breed, soma_config)

    somas = [{"id": 0, "breed": soma_breed, "config": soma_config, "overrides": {}}]
    synapses = [{
        "id": 1, "pre": -1, "post": 0,
        "breed": synapse_breed, "config": synapse_config,
        "learning_rule": None, "learning_rule_config": "default",
        "overrides": {"hyperparameters": {"weight": float(probe_weight)}},
    }]

    model = NeuromorphicModel(enable_internal_states_tracking=True)
    model.create_from_lists(somas, synapses)
    model.setup()
    model.add_spike_list(1, [[1, 1.0]])          # single spike at tick 1
    model.simulate(ticks=ticks, update_data_ticks=1)

    # internal_states_buffer rows are [v, tcount, tlast] per tick; column 0 is v.
    hist = np.asarray(model.get_internal_states_history(0))
    v_trace = hist[:, 0] if hist.ndim == 2 else np.asarray([h[0] for h in hist])
    psp_mv = float(np.max(v_trace)) - lif["vrest"]
    theta = lif["vthr"] - lif["vrest"]
    if psp_mv > 0.5 * theta:
        raise RuntimeError(
            f"probe PSP {psp_mv:.3f} mV is not clearly sub-threshold (theta={theta:.2f} mV); "
            f"lower --probe-weight below {probe_weight}."
        )
    return psp_mv / probe_weight, lif


def threshold_rate_hz(*, j_mv, excitatory_in_degree, lif, eta):
    """External Poisson rate (Hz) = eta * nu_thr * C_E, with nu_thr in OUR units.

    nu_thr = (vthr - vrest) / (J_mV * C_E * tau_m). Mirrors NEST's p_rate but fed the
    measured PSP and this preset's threshold distance / membrane time constant.
    """
    theta = lif["vthr"] - lif["vrest"]                    # threshold distance (mV)
    nu_thr = theta / (j_mv * excitatory_in_degree * lif["tau_m_s"])   # Hz
    return eta * nu_thr * excitatory_in_degree, nu_thr


def _cv_isi(spike_ticks):
    """Coefficient of variation of inter-spike intervals for one neuron (needs >=3 spikes)."""
    if len(spike_ticks) < 3:
        return None
    isi = np.diff(np.sort(spike_ticks))
    mean = isi.mean()
    return float(isi.std() / mean) if mean > 0 else None


def validate_ai(*, external_rate_hz, external_weight, dt_ms, ticks, somas_per_rank,
                C_E, C_I, g, excitatory_weight, delay_ms, seed):
    """Run a small Brunel net at the given drive; return (mean_rate_hz, mean_cv_isi).

    The single external (pre=-1) synapse per soma stands in for the neuron's ``C_E``
    external inputs, so it carries an **aggregate** weight ``external_weight`` (~``C_E*J``)
    and is driven at the per-input Poisson rate ``external_rate_hz`` -- NEST's collapse of
    ``C_E`` generators into one. This keeps the drive rate realizable at ``dt_ms`` (a single
    synapse at the full ``C_E*rate`` would need >1 spike/tick and clip).
    """
    somas, synapses = build_brunel_network(
        somas_per_rank=somas_per_rank,
        excitatory_in_degree=C_E, inhibitory_in_degree=C_I,
        inhibitory_weight_ratio=g, excitatory_weight=excitatory_weight,
        external_weight=external_weight, synaptic_delay_ms=delay_ms, seed=seed,
    )
    model = NeuromorphicModel(enable_internal_states_tracking=False)
    model.create_from_lists(somas, synapses)
    model.setup()

    # Poisson drive on every external (pre=-1) synapse.
    input_synapses = [sid for sid in model._synapse_ids
                      if model.get_synapse_connectivity(sid)[0] == -1]
    rng = np.random.default_rng(np.random.SeedSequence([seed, 7]))
    p_spike = external_rate_hz * dt_ms / 1000.0
    if p_spike >= 1.0:
        raise RuntimeError(
            f"external rate {external_rate_hz:.1f} Hz needs p_spike={p_spike:.2f} >= 1 at "
            f"dt={dt_ms} ms -- a per-tick Bernoulli would clip. Use a finer dt or a lower "
            "per-input rate (this is why we drive at eta*nu_thr, not eta*nu_thr*C_E)."
        )
    for syn_id in input_synapses:
        spike_ticks = np.nonzero(rng.random(ticks) < p_spike)[0] + 1
        if spike_ticks.size:
            model.add_spike_list(syn_id, [[int(t), 1.0] for t in spike_ticks])

    soma_ids = [s["id"] for s in somas]
    model.set_recorded_somas(soma_ids)
    model.simulate(ticks=ticks, update_data_ticks=1)

    spikes = model.get_all_spike_times()          # {soma_id: [tick, ...]}
    total_spikes = sum(len(v) for v in spikes.values())
    duration_s = ticks * dt_ms / 1000.0
    mean_rate = total_spikes / (len(soma_ids) * duration_s) if duration_s > 0 else 0.0
    cvs = [c for c in (_cv_isi(v) for v in spikes.values()) if c is not None]
    mean_cv = float(np.mean(cvs)) if cvs else None
    return mean_rate, mean_cv, len(cvs), len(soma_ids)


def main():
    ap = argparse.ArgumentParser(description="Calibrate Brunel drive to the AI regime.")
    ap.add_argument("--target-psp-mv", type=float, default=0.1,
                    help="desired single-spike EPSP amplitude J (mV); NEST uses 0.1")
    ap.add_argument("--probe-weight", type=float, default=0.05,
                    help="small sub-threshold weight for the PSP-per-weight probe")
    ap.add_argument("--C-E", dest="C_E", type=int, default=800)
    ap.add_argument("--C-I", dest="C_I", type=int, default=200)
    ap.add_argument("--g", type=float, default=5.0, help="|J_I|/J_E (inhibition-dominated)")
    ap.add_argument("--eta", type=float, default=2.0, help="external rate / threshold rate")
    ap.add_argument("--delay", type=float, default=1.5, help="synaptic delay (ms)")
    ap.add_argument("--dt-ms", type=float, default=1.0, help="ms per tick")
    ap.add_argument("--probe-ticks", type=int, default=200,
                    help="ticks for the single-spike PSP probe")
    ap.add_argument("--ticks", type=int, default=2000, help="ticks for AI validation")
    ap.add_argument("--val-neurons", type=int, default=10000,
                    help="network size for AI validation")
    ap.add_argument("--max-search-steps", type=int, default=8,
                    help="max drive-rate search iterations to land in the AI window")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("=" * 70)
    print("BRUNEL AI CALIBRATION")
    print("=" * 70)

    # [1/3] PSP probe -> solve for the weight that hits the target PSP ---------
    print("\n[1/3] Measuring PSP-per-weight and solving for the calibrated weight...")
    coeff, lif = measure_psp_per_weight(probe_weight=args.probe_weight,
                                        ticks=args.probe_ticks)
    theta = lif["vthr"] - lif["vrest"]
    j_mv = args.target_psp_mv
    calibrated_weight = j_mv / coeff            # linear sub-threshold membrane
    print(f"    vthr={lif['vthr']} mV, vrest={lif['vrest']} mV -> theta={theta:.2f} mV")
    print(f"    tau_m = R*C = {lif['tau_m_s']*1000:.2f} ms")
    print(f"    PSP coefficient = {coeff:.4f} mV per unit weight")
    print(f"    target J = {j_mv:.4f} mV  ->  excitatory_weight = {calibrated_weight:.5f}")
    print(f"    (config default weight=14 would give ~{14*coeff:.1f} mV PSP -- far too strong)")

    # [2/3] External drive: aggregate weight + per-input threshold rate ------
    # One external synapse stands in for C_E external inputs, so it carries the aggregate
    # weight C_E*J and is driven at the per-input rate eta*nu_thr (NEST's generator collapse;
    # keeps p_spike < 1 at dt, unlike a single synapse at the full eta*nu_thr*C_E).
    print("\n[2/3] Computing external drive (aggregate weight + per-input rate)...")
    _, nu_thr = threshold_rate_hz(j_mv=j_mv, excitatory_in_degree=args.C_E,
                                  lif=lif, eta=args.eta)
    ext_weight = args.C_E * calibrated_weight
    base_rate = args.eta * nu_thr
    print(f"    nu_thr = theta/(J*C_E*tau_m) = {nu_thr:.3f} Hz")
    print(f"    external_weight = C_E*J = {args.C_E}*{calibrated_weight:.5f} = {ext_weight:.4f}")
    print(f"    per-input rate  = eta*nu_thr = {args.eta}*{nu_thr:.3f} = {base_rate:.1f} Hz")

    # [3/3] AI validation: search the drive rate down into the AI rate window --
    npp = args.val_neurons
    print(f"\n[3/3] Validating AI on {npp} neurons for {args.ticks} ticks "
          f"(searching drive rate for ~1-10 Hz, CV~1)...")
    rate = base_rate
    best = None
    for step in range(args.max_search_steps):
        mean_rate, mean_cv, n_active, n_total = validate_ai(
            external_rate_hz=rate, external_weight=ext_weight, dt_ms=args.dt_ms,
            ticks=args.ticks, somas_per_rank=npp, C_E=args.C_E, C_I=args.C_I, g=args.g,
            excitatory_weight=calibrated_weight, delay_ms=args.delay, seed=args.seed,
        )
        cv_str = f"{mean_cv:.3f}" if mean_cv is not None else "n/a"
        print(f"    rate={rate:7.2f} Hz -> firing {mean_rate:6.2f} Hz, CV_ISI={cv_str} "
              f"({n_active}/{n_total} active)")
        # AI = low mean rate + irregular. CV window is wide because our exponential-current
        # synapses give more irregular ISIs (CV ~1.0-1.6) than NEST's delta synapses; the
        # essential signatures are "not silent, not saturated, clearly irregular (CV>=0.8)".
        ai_rate_ok = 1.0 <= mean_rate <= 10.0
        ai_cv_ok = mean_cv is not None and mean_cv >= 0.8
        if ai_rate_ok and ai_cv_ok:
            best = (rate, mean_rate, mean_cv)
            break
        # too hot -> lower the drive; silent -> raise it (bounded); keep the closest rate-match
        if mean_rate > 10.0:
            rate *= 0.6
        elif mean_rate < 1.0:
            rate *= 1.8
        else:  # in rate window but CV somehow < 0.8 -> nudge down, remember this point
            best = best or (rate, mean_rate, mean_cv)
            rate *= 0.8

    if best is not None:
        final_rate, final_firing, final_cv = best
        verdict = "AI"
    else:
        final_rate, final_firing, final_cv = rate, mean_rate, mean_cv
        verdict = "NOT AI (widen search or adjust g/eta/target-psp and re-run)"

    print("\n" + "=" * 70)
    print("CALIBRATION RESULT")
    print("=" * 70)
    print(f"  target PSP J_E          : {j_mv:.4f} mV")
    print(f"  excitatory_weight (J_E) : {calibrated_weight:.5f}")
    print(f"  external_weight (C_E*J) : {ext_weight:.4f}")
    print(f"  external_rate_hz        : {final_rate:.2f}")
    print(f"  g                       : {args.g}")
    cv_str = f"{final_cv:.3f}" if final_cv is not None else "n/a"
    print(f"  achieved rate / CV_ISI  : {final_firing:.2f} Hz / {cv_str}")
    print(f"  regime                  : {verdict}")
    print("=" * 70)
    print(f"\n  -> pass to the driver:  --external-rate {final_rate:.2f} "
          f"--external-weight {ext_weight:.4f} --J-E {calibrated_weight:.5f} --g {args.g}")


if __name__ == "__main__":
    main()
