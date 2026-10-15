"""
Heterogeneous neuromorphic chip PoC: one network, nine hardware design points, one run.

A linear digit classifier (logistic regression on sklearn's 8x8 digits) is deployed on a
simulated chip as 64 on-chip Poisson encoders -> 640 synaptic devices -> 10 LIF neurons
per core. Each test digit is presented to one core of every design point at once, so a
single simulate() call compares all designs on identical input spike trains:

    ideal              float SRAM synapses, float digital neurons (the reference)
    sram_4bit          4-bit SRAM synapses, 8-fractional-bit digital neurons
    rram_1t1r          RRAM 1T1R cells, analog neurons
    rram_passive       RRAM in a selector-less passive crossbar (leaks every tick)
    rram_passive_sel   the same crossbar with a sinh selector
    pcm_1month         differential PCM, one month after programming
    pcm_1month_gdc     the same with global drift compensation
    hetero_aware       the 20 % most important synapses on 4-bit SRAM, the rest RRAM 1T1R,
                       mixed inside every column (same post-neuron)
    hetero_random      the same 20 % budget spent on randomly chosen synapses

Every device's mismatch, faults and noise are drawn from a counter-based RNG keyed on its
global id, and every synapse lives on its post-neuron's rank. So the run is bitwise the
same on any number of GPUs; the digest printed at the end is how to check that:

    python run_chip_poc.py
    mpirun --oversubscribe -n 2 python run_chip_poc.py
    mpirun --oversubscribe -n 4 python run_chip_poc.py
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import yaml
from mpi4py import MPI
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from superneuroabm.model import NeuromorphicModel

import chip_devices as cd

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "chip_config.yaml"
DEVICES = HERE / "chip_devices.py"

N_IN, N_OUT = 64, 10

# name -> (neuron breed, neuron config), synapse (breed, config) for the bulk of the column,
# and optionally (breed, config, policy) for the important-synapse budget.
SRAM4 = ("chip_sram_synapse", "int4")
DESIGNS = [
    ("ideal",            ("chip_digital_lif", "ideal"),   ("chip_sram_synapse", "float"),       None),
    ("sram_4bit",        ("chip_digital_lif", "fixed8"),  SRAM4,                                None),
    ("rram_1t1r",        ("chip_analog_lif", "default"),  ("chip_rram_synapse", "1t1r"),        None),
    ("rram_passive",     ("chip_analog_lif", "default"),  ("chip_rram_synapse", "passive"),     None),
    ("rram_passive_sel", ("chip_analog_lif", "default"),  ("chip_rram_synapse", "passive_sel"), None),
    ("pcm_1month",       ("chip_analog_lif", "default"),  ("chip_pcm_synapse", "month1"),       None),
    ("pcm_1month_gdc",   ("chip_analog_lif", "default"),  ("chip_pcm_synapse", "month1_gdc"),   None),
    ("hetero_aware",     ("chip_analog_lif", "default"),  ("chip_rram_synapse", "1t1r"),        SRAM4 + ("importance",)),
    ("hetero_random",    ("chip_analog_lif", "default"),  ("chip_rram_synapse", "1t1r"),        SRAM4 + ("random",)),
]


def train_classifier(n_test, seed):
    digits = load_digits()
    X = digits.data / 16.0
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, digits.target, test_size=0.2, random_state=seed, stratify=digits.target)
    clf = LogisticRegression(C=0.5, max_iter=5000).fit(X_tr, y_tr)
    return clf, X_tr, X_te[:n_test], y_te[:n_test]


def sram_budget_masks(W, X_tr, frac, seed):
    """Which of the 640 (k, i) cells go to SRAM. Same mapping on every core."""
    importance = np.abs(W) * X_tr.mean(axis=0)[None, :]     # mean current the cell carries
    n = int(round(frac * W.size))
    aware = np.zeros(W.size, bool)
    aware[np.argsort(importance.ravel())[::-1][:n]] = True
    rand = np.zeros(W.size, bool)
    rand[np.random.default_rng(seed).choice(W.size, n, replace=False)] = True
    return {"importance": aware.reshape(W.shape), "random": rand.reshape(W.shape)}


def build_partition(rank, n_ranks, X_te, W, bias, masks, rate_max, v_swing=None,
                    static_only=False):
    """This rank's somas and synapses, post-owned, listed in global-id order.

    Ids: encoders (n_test x 64), then output neurons (n_cores x 10), then synapses.
    Encoders of digit d live on rank d % R and the neurons of core c on (c + 1) % R,
    so most spikes cross a GPU boundary.
    """
    n_test = len(X_te)
    n_cores = len(DESIGNS) * n_test
    out_base = n_test * N_IN
    syn_base = out_base + n_cores * N_OUT

    somas, synapses, remote = [], [], {}
    for d in range(n_test):
        if d % n_ranks == rank:
            for i in range(N_IN):
                somas.append({
                    "id": d * N_IN + i, "breed": "chip_poisson_encoder", "config": "default",
                    "overrides": {"hyperparameters": {"rate": rate_max * X_te[d, i]}}})

    for c in range(n_cores):
        if (c + 1) % n_ranks != rank:
            continue
        design, d = divmod(c, n_test)
        _, (n_breed, n_cfg), (s_breed, s_cfg), budget = DESIGNS[design]
        for k in range(N_OUT):
            somas.append({
                "id": out_base + c * N_OUT + k, "breed": n_breed, "config": n_cfg,
                "overrides": {"hyperparameters": {"bias": float(bias[k])}}})
        for k in range(N_OUT):
            for i in range(N_IN):
                breed, cfg = s_breed, s_cfg
                if budget is not None and masks[budget[2]][k, i]:
                    breed, cfg = budget[0], budget[1]
                hp = {"weight": float(W[k, i])}
                if v_swing is not None and breed == "chip_rram_synapse":
                    hp["v_swing"] = v_swing
                if static_only and breed in ("chip_rram_synapse", "chip_pcm_synapse"):
                    hp["dynamic"] = 0.0
                pre = d * N_IN + i
                synapses.append({
                    "id": syn_base + (c * N_OUT + k) * N_IN + i,
                    "pre": pre, "post": out_base + c * N_OUT + k,
                    "breed": breed, "config": cfg,
                    "overrides": {"hyperparameters": hp}})
                if d % n_ranks != rank:
                    remote[pre] = d % n_ranks
    somas.sort(key=lambda s: s["id"])
    return {"somas": somas, "synapses": synapses, "remote_ranks": remote}, out_base


def make_model(seed):
    model = NeuromorphicModel(
        soma_breed_info={
            "chip_poisson_encoder": (cd.chip_poisson_encoder, DEVICES),
            "chip_digital_lif": (cd.chip_digital_lif, DEVICES),
            "chip_analog_lif": (cd.chip_analog_lif, DEVICES),
        },
        synapse_breed_info={
            "chip_sram_synapse": (cd.chip_sram_synapse, DEVICES),
            "chip_rram_synapse": (cd.chip_rram_synapse, DEVICES),
            "chip_pcm_synapse": (cd.chip_pcm_synapse, DEVICES),
        },
        user_config=CONFIG,
        enable_internal_states_tracking=False,
    )
    model.set_seed(seed)   # the chip instance: same devices on every rank
    return model


def mcnemar_p(a_ok, b_ok):
    """Exact two-sided McNemar test on paired per-digit outcomes."""
    from math import comb
    b, c = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return mid - half, mid + half


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--n-test", type=int, default=360, help="test digits (cores per design)")
    ap.add_argument("--ticks", type=int, default=150, help="presentation time (dt = 1 ms)")
    ap.add_argument("--rate-max", type=float, default=200.0, help="encoder rate at full intensity, Hz")
    ap.add_argument("--sram-frac", type=float, default=0.2, help="SRAM budget of the hetero designs")
    ap.add_argument("--chip-seed", type=int, default=1, help="chip instance (device draws)")
    ap.add_argument("--static-only", action="store_true",
                    help="ablation: keep only per-device constants (what a fixed-weight "
                         "simulator can represent), drop in-run device effects")
    ap.add_argument("--v-swing", type=float, default=None,
                    help="override the RRAM membrane/read-voltage ratio from the config")
    ap.add_argument("--out", type=Path, default=HERE / "results")
    args = ap.parse_args()

    comm = MPI.COMM_WORLD
    rank, n_ranks = comm.Get_rank(), comm.Get_size()

    clf, X_tr, X_te, y_te = train_classifier(args.n_test, seed=0)
    w_max = yaml.safe_load(CONFIG.read_text())["synapse"]["chip_sram_synapse"]["float"][
        "hyperparameters"]["w_max"]
    alpha = w_max / np.abs(clf.coef_).max()
    W = alpha * clf.coef_                                    # membrane step per input spike
    bias = alpha * clf.intercept_ * args.rate_max * 1e-3     # same scale, per tick
    masks = sram_budget_masks(clf.coef_, X_tr, args.sram_frac, seed=0)

    t0 = time.time()
    part, out_base = build_partition(rank, n_ranks, X_te, W, bias, masks, args.rate_max,
                                     args.v_swing, args.static_only)
    model = make_model(args.chip_seed)
    model.load_post_owned(part)
    n_local = (len(part["somas"]), len(part["synapses"]))
    del part
    model.set_recorded_somas(list(range(out_base, out_base + len(DESIGNS) * args.n_test * N_OUT)))
    model.setup()
    t_build = time.time() - t0

    t0 = time.time()
    model.simulate(ticks=args.ticks, update_data_ticks=1)   # MPI spike exchange every tick
    t_sim = time.time() - t0
    import cupy as cp   # buffers are allocated at the first simulate(), not at setup()
    gpu_bytes = comm.reduce(cp.get_default_memory_pool().used_bytes(), op=MPI.SUM, root=0)

    spikes = model.get_all_spike_times()   # collective
    totals = comm.reduce(np.array(n_local), op=MPI.SUM, root=0)
    if rank != 0:
        return

    n_test = args.n_test
    counts = np.zeros((len(DESIGNS), n_test, N_OUT))
    first = np.full((len(DESIGNS), n_test, N_OUT), np.inf)
    for sid, ticks in spikes.items():
        c, k = divmod(sid - out_base, N_OUT)
        design, d = divmod(c, n_test)
        counts[design, d, k] = len(ticks)
        first[design, d, k] = min(ticks)
    # Most spikes wins; ties go to the earliest first spike.
    score = counts - 1e-6 * np.where(np.isinf(first), args.ticks, first)
    pred = score.argmax(axis=2)

    digest = hashlib.sha256(
        json.dumps(sorted((int(s), sorted(int(t) for t in ts)) for s, ts in spikes.items())
                   ).encode()).hexdigest()

    print(f"\n{'STATIC-ONLY ablation | ' if args.static_only else ''}"
          f"chip seed {args.chip_seed} | {n_ranks} rank(s) | {n_test} digits x "
          f"{len(DESIGNS)} designs | {totals[0]} somas, {totals[1]} synapses")
    print(f"build+setup {t_build:.1f} s, simulate {args.ticks} ticks {t_sim:.1f} s, "
          f"GPU memory in use {gpu_bytes / 2**20:.0f} MiB "
          f"({gpu_bytes / (totals[0] + totals[1]):.0f} B per agent)")
    print(f"float logistic regression on the same digits: "
          f"{(clf.predict(X_te) == y_te).mean() * 100:.1f} %\n")
    print(f"{'design':<18}{'accuracy':>10}{'95% CI':>16}{'out rate (Hz)':>15}")
    rows = []
    for j, (name, *_rest) in enumerate(DESIGNS):
        correct = int((pred[j] == y_te).sum())
        lo, hi = wilson(correct, n_test)
        winner_rate = counts[j][np.arange(n_test), pred[j]].mean() / (args.ticks * 1e-3)
        rows.append({"design": name, "accuracy": correct / n_test, "ci95": [lo, hi],
                     "winner_rate_hz": winner_rate, "pred": pred[j].tolist()})
        print(f"{name:<18}{correct / n_test * 100:>9.1f}%   [{lo * 100:4.1f}, {hi * 100:4.1f}]"
              f"{winner_rate:>14.1f}")
    ok = {name: pred[j] == y_te for j, (name, *_r) in enumerate(DESIGNS)}
    for a, b in [("hetero_aware", "hetero_random"), ("hetero_aware", "rram_1t1r"),
                 ("rram_passive_sel", "rram_passive"), ("pcm_1month_gdc", "pcm_1month")]:
        print(f"paired (McNemar) {a} vs {b}: p = {mcnemar_p(ok[a], ok[b]):.3g}")
    print(f"\nspike digest {digest}")

    args.out.mkdir(exist_ok=True)
    tag = "_static" if args.static_only else ""
    out = args.out / f"chip_seed{args.chip_seed}_ranks{n_ranks}{tag}.json"
    out.write_text(json.dumps({
        "chip_seed": args.chip_seed, "ranks": n_ranks, "n_test": n_test, "ticks": args.ticks,
        "v_swing": args.v_swing, "static_only": args.static_only,
        "somas": int(totals[0]), "synapses": int(totals[1]),
        "build_s": t_build, "simulate_s": t_sim, "gpu_bytes_in_use": int(gpu_bytes),
        "digest": digest, "labels": y_te.tolist(),
        "designs": rows,
    }, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
