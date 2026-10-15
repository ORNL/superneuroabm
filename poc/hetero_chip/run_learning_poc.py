"""
On-chip learning PoC: the chip learns the digits itself, and the learning rule is the
device physics (learning_devices.learn_device).

64 input relays and 10 teacher relays are shared by every core. Each core is 10 output
neurons and 640 memristive devices that start blank (g = g_ref). During training the
teacher drives the correct output and inhibits the others; every presynaptic spike reads
its device and programs it with one SET or RESET pulse. Then programming stops, the
teacher goes silent, and the same cores classify the test digits.

Seven device designs x N chip instances run in one simulation on identical input spike
trains; instances differ only in their device draws (keyed on device id, so the same on
any number of GPUs; the digest at the end checks it):

    python run_learning_poc.py                                   # 1 GPU, full run
    mpirun --oversubscribe -n 2 python run_learning_poc.py --n-train 200 --n-test 60 --instances 4
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from mpi4py import MPI
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from superneuroabm.model import NeuromorphicModel

import learning_devices as ld

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "learning_config.yaml"
KERNELS = HERE / "learning_devices.py"

N_IN, N_OUT = 64, 10
N_RELAY = N_IN + N_OUT          # 64 input encoders + 10 teachers

# (design, learn_device config, what NEST GPU would need)
DESIGNS = [
    ("ideal",     "ideal",     "a new C++ rule (fits its plasticity hook)"),
    ("linear",    "linear",    "a new C++ rule (fits its plasticity hook)"),
    ("d2d",       "d2d",       "per-synapse parameters: not passed to the hook"),
    ("c2c",       "c2c",       "a random number per pulse: not passed to the hook"),
    ("levels16",  "levels16",  "a random number per pulse: not passed to the hook"),
    ("levels4",   "levels4",   "a random number per pulse: not passed to the hook"),
    ("rram_full", "rram_full", "both of the above"),
]


def build_partition(rank, n_ranks, n_cores, t_learn_end, w_ex, w_inh):
    """Post-owned partition, listed in global-id order. Relay r lives on rank r % R and
    core c on rank (c + 1) % R, so device and teacher inputs cross GPU boundaries."""
    ext = N_RELAY
    out = ext + N_RELAY
    dev = out + n_cores * N_OUT
    tch = dev + n_cores * N_OUT * N_IN
    ids = {"ext": ext, "out": out, "dev": dev, "tch": tch}

    somas, synapses, remote = [], [], {}
    for r in range(N_RELAY):
        if r % n_ranks == rank:
            somas.append({"id": r, "breed": "learn_relay", "config": "default"})
            synapses.append({"id": ext + r, "pre": -1, "post": r,
                             "breed": "learn_wire", "config": "default"})

    for c in range(n_cores):
        if (c + 1) % n_ranks != rank:
            continue
        cfg = DESIGNS[c // (n_cores // len(DESIGNS))][1]
        for k in range(N_OUT):
            somas.append({"id": out + c * N_OUT + k, "breed": "learn_lif", "config": "default"})
        for k in range(N_OUT):
            post = out + c * N_OUT + k
            for i in range(N_IN):
                synapses.append({
                    "id": dev + (c * N_OUT + k) * N_IN + i, "pre": i, "post": post,
                    "breed": "learn_device", "config": cfg,
                    "overrides": {"hyperparameters": {"t_learn_end": float(t_learn_end)}}})
                if i % n_ranks != rank:
                    remote[i] = i % n_ranks
        for k in range(N_OUT):
            post = out + c * N_OUT + k
            for j in range(N_OUT):
                pre = N_IN + j
                synapses.append({
                    "id": tch + (c * N_OUT + k) * N_OUT + j, "pre": pre, "post": post,
                    "breed": "learn_wire", "config": "default",
                    "overrides": {"hyperparameters": {"weight": w_ex if j == k else -w_inh}}})
                if pre % n_ranks != rank:
                    remote[pre] = pre % n_ranks
    synapses.sort(key=lambda s: s["id"])
    return {"somas": somas, "synapses": synapses, "remote_ranks": remote}, ids


def input_events(X, y, n_train, t_on, t_off, rate_max, teacher_rate, seed, ext):
    """Poisson input for every presentation, plus the teacher during training only."""
    rng = np.random.default_rng(seed)
    period = t_on + t_off
    p = rate_max * 1e-3 * X                                        # (n_samples, 64)
    fire = rng.random((len(X), t_on, N_IN)) < p[:, None, :]
    s, t, i = np.nonzero(fire)
    ids = [ext + i]
    ticks = [s * period + t]
    tf = rng.random((n_train, t_on)) < teacher_rate * 1e-3
    s, t = np.nonzero(tf)
    ids.append(ext + N_IN + y[s])
    ticks.append(s * period + t)
    return np.concatenate(ids), np.concatenate(ticks)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--instances", type=int, default=16, help="chip instances per design")
    ap.add_argument("--n-train", type=int, default=1437, help="training digits (one pass)")
    ap.add_argument("--n-test", type=int, default=360)
    ap.add_argument("--t-on", type=int, default=50, help="presentation ticks (dt = 1 ms)")
    ap.add_argument("--t-off", type=int, default=20, help="rest ticks between digits")
    ap.add_argument("--rate-max", type=float, default=200.0, help="input rate at full intensity, Hz")
    ap.add_argument("--teacher-rate", type=float, default=1000.0,
                    help="Hz; 1000 = a pulse every tick, i.e. a continuous clamp")
    ap.add_argument("--w-ex", type=float, default=5.0, help="teacher clamp on the correct output")
    ap.add_argument("--w-inh", type=float, default=10.0, help="teacher clamp on wrong outputs")
    ap.add_argument("--chip-seed", type=int, default=1, help="chip instance set (device draws)")
    ap.add_argument("--input-seed", type=int, default=0, help="Poisson input trains")
    ap.add_argument("--chunk", type=int, default=20000, help="ticks per simulate() call")
    ap.add_argument("--plot", action="store_true", help="save learned conductances (1 rank)")
    ap.add_argument("--out", type=Path, default=HERE / "results")
    args = ap.parse_args()

    comm = MPI.COMM_WORLD
    rank, n_ranks = comm.Get_rank(), comm.Get_size()

    digits = load_digits()
    X_tr, X_te, y_tr, y_te = train_test_split(
        digits.data / 16.0, digits.target, test_size=0.2, random_state=0, stratify=digits.target)
    X_tr, y_tr = X_tr[:args.n_train], y_tr[:args.n_train]
    X_te, y_te = X_te[:args.n_test], y_te[:args.n_test]

    period = args.t_on + args.t_off
    t_learn_end = args.n_train * period
    t_total = (args.n_train + args.n_test) * period
    n_cores = len(DESIGNS) * args.instances

    t0 = time.time()
    part, ids = build_partition(rank, n_ranks, n_cores, t_learn_end, args.w_ex, args.w_inh)
    n_local = (len(part["somas"]), len(part["synapses"]))
    model = NeuromorphicModel(
        soma_breed_info={"learn_relay": (ld.learn_relay, KERNELS),
                         "learn_lif": (ld.learn_lif, KERNELS)},
        synapse_breed_info={"learn_wire": (ld.learn_wire, KERNELS),
                            "learn_device": (ld.learn_device, KERNELS)},
        user_config=CONFIG,
        enable_internal_states_tracking=False,
    )
    model.set_seed(args.chip_seed)
    model.load_post_owned(part)
    del part
    model.set_recorded_somas([])                # training-phase spikes are not needed
    ev_ids, ev_ticks = input_events(
        np.concatenate([X_tr, X_te]), np.concatenate([y_tr, y_te]), args.n_train,
        args.t_on, args.t_off, args.rate_max, args.teacher_rate, args.input_seed, ids["ext"])
    model.add_spikes(ev_ids, ev_ticks)          # collective: each rank keeps its own
    model.setup()
    t_build = time.time() - t0

    # Training: programming is on until t_learn_end; output spikes are not needed.
    t0 = time.time()
    while model.tick < t_learn_end:
        model.simulate(ticks=min(args.chunk, t_learn_end - model.tick), update_data_ticks=1)
    t_train = time.time() - t0

    # Test: collect output spikes per presentation window.
    t0 = time.time()
    model.set_recorded_somas(list(range(ids["out"], ids["out"] + n_cores * N_OUT)))
    counts = np.zeros((n_cores, args.n_test, N_OUT))
    first = np.full((n_cores, args.n_test, N_OUT), np.inf)
    hasher = hashlib.sha256()
    while model.tick < t_total:
        model.simulate(ticks=min(args.chunk, t_total - model.tick), update_data_ticks=1)
        spikes = model.get_all_spike_times()    # collective
        if rank == 0:
            for sid in sorted(spikes):
                ticks = np.asarray(sorted(spikes[sid]), dtype=np.int64)
                hasher.update(np.int64(sid).tobytes() + ticks.tobytes())
                c, k = divmod(sid - ids["out"], N_OUT)
                s = (ticks - t_learn_end) // period
                np.add.at(counts[c, :, k], s, 1)
                np.minimum.at(first[c, :, k], s, ticks)
    t_test = time.time() - t0

    if args.plot and n_ranks == 1:
        plot_conductances(model, ids, args)

    totals = comm.reduce(np.array(n_local), op=MPI.SUM, root=0)
    if rank != 0:
        return

    score = counts - 1e-6 * np.where(np.isinf(first), t_total, first)
    pred = score.argmax(axis=2)                              # (n_cores, n_test)
    acc = (pred == y_te[None, :]).mean(axis=1).reshape(len(DESIGNS), args.instances)

    print(f"\nchip seed {args.chip_seed} | {n_ranks} rank(s) | {len(DESIGNS)} designs x "
          f"{args.instances} chip instances | {totals[0]} somas, {totals[1]} synapses")
    print(f"{args.n_train} training digits (one pass), {args.n_test} test digits, "
          f"{t_total} ticks | build+setup {t_build:.1f} s, train {t_train:.1f} s, "
          f"test {t_test:.1f} s\n")
    print(f"{'device':<15}{'mean':>8}{'sd':>7}{'min':>8}{'max':>8}   NEST GPU would need")
    rows = []
    for j, (name, _cfg, nestgpu) in enumerate(DESIGNS):
        a = acc[j] * 100
        rows.append({"design": name, "acc_mean": a.mean(), "acc_sd": a.std(),
                     "acc_min": a.min(), "acc_max": a.max(), "acc": a.tolist(),
                     "nestgpu": nestgpu})
        print(f"{name:<15}{a.mean():>7.1f}%{a.std():>7.1f}{a.min():>7.1f}%{a.max():>7.1f}%   {nestgpu}")
    digest = hasher.hexdigest()
    print(f"\ntest-phase spike digest {digest}")

    args.out.mkdir(exist_ok=True)
    out = args.out / (f"learning_seed{args.chip_seed}_ranks{n_ranks}_inst{args.instances}"
                      f"_train{args.n_train}.json")
    out.write_text(json.dumps({
        "chip_seed": args.chip_seed, "ranks": n_ranks, "instances": args.instances,
        "n_train": args.n_train, "n_test": args.n_test, "t_on": args.t_on, "t_off": args.t_off,
        "somas": int(totals[0]), "synapses": int(totals[1]), "ticks": t_total,
        "build_s": t_build, "train_s": t_train, "test_s": t_test, "digest": digest,
        "designs": rows,
    }, indent=2))
    print(f"wrote {out}")


def plot_conductances(model, ids, args):
    """Learned conductance minus the reference (what the neuron sees), instance 0 of every
    design, as 8x8 maps per class. Stuck cells show their last programmed state."""
    import yaml
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g_ref = yaml.safe_load(CONFIG.read_text())["synapse"]["learn_device"]["ideal"][
        "hyperparameters"]["g_ref"]
    fig, axes = plt.subplots(len(DESIGNS), N_OUT, figsize=(N_OUT * 1.1, len(DESIGNS) * 1.25))
    for j, (name, *_r) in enumerate(DESIGNS):
        c = j * args.instances
        for k in range(N_OUT):
            g = [model.get_agent_property_value(
                    ids["dev"] + (c * N_OUT + k) * N_IN + i, "internal_states")[1]
                 for i in range(N_IN)]
            ax = axes[j, k]
            im = ax.imshow(np.reshape(g, (8, 8)) - g_ref, cmap="RdBu_r", vmin=-0.25, vmax=0.25)
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_title(str(k))
            if k == 0:
                ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=8)
    fig.suptitle("Learned g - g_ref per output class (instance 0 of each design)")
    fig.tight_layout(rect=(0, 0, 0.93, 1))
    fig.colorbar(im, cax=fig.add_axes((0.94, 0.15, 0.015, 0.7)))
    args.out.mkdir(exist_ok=True)
    path = args.out / f"learning_conductances_seed{args.chip_seed}.png"
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
