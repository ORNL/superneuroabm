"""NEST GPU run of a partitioned network: each rank builds only from its own partition file.

The network comes as data, as a partitioned connectome would: <root>/np<N>/partition_<rank>.npz
(make_brunel_partitions.py writes them; any columnar post-owned partition works, any
assignment of somas to ranks). Nothing is drawn by a NEST GPU connection rule, and the input
is replayed from spike lists (partitioned_common.drive), not drawn by a poisson_generator.

How a rank builds:
  * its somas become local iaf_psc_alpha nodes 0..n-1, in ascending soma id;
  * a synapse whose pre soma is local: Connect, one_to_one;
  * a synapse whose pre soma lives on rank i: RemoteConnect(i -> this rank), one_to_one. NEST
    GPU builds the node maps of a host pair from the RemoteConnect calls themselves
    (remote_connect.h: "The connect command is performed on both source and target host"),
    so rank i issues the same calls as this rank: same source and target arrays, same order;
  * no call may name a source node twice: NEST GPU maps only the first entry of a repeated
    source (remote_connect.cu, also in the 2026-10 head), and the others get a wrong
    image node (measured 2026-11-14: 64-70 Hz at 2 ranks, 105-148 Hz at 4, against 28.9 Hz
    at 1). So the edges of a host pair go in as many calls as its busiest source has
    edges, call l holding every source's l-th edge;
  * the partition is post-owned, so a rank first sends each source rank the edges it
    receives from it (timer edge_exchange: two int32 per cross-rank synapse), after one
    alltoall pair has translated the remote soma ids (remote_ids / remote_rank_of) into
    local node indices on their owners (timer boundary_exchange).

usage: srun -n N python run_partitioned_nestgpu.py <root> [--ms 1500] [--out spikes.npz]
"""

import argparse
import time
from pathlib import Path

import numpy as np
from mpi4py import MPI        # first: NEST GPU's MPI_Init is skipped when MPI is already up

import partitioned_common as pc


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root")
    ap.add_argument("--ms", type=float, default=1500.0, help="simulated time (ms)")
    ap.add_argument("--burn-in-ms", type=float, default=500.0)
    ap.add_argument("--eta", type=float, default=2.0, help="drive / threshold rate")
    ap.add_argument("--seed", type=int, default=1234, help="drive seed")
    ap.add_argument("--out", default=None, help="spikes .npz (default <root>/spikes_nestgpu_np<N>.npz)")
    args = ap.parse_args()

    import nestgpu as ngpu
    ngpu.ConnectMpiInit()
    comm = MPI.COMM_WORLD
    rank, P = comm.Get_rank(), comm.Get_size()
    assert ngpu.HostId() == rank and ngpu.HostNum() == P, "NEST GPU and mpi4py disagree on ranks"
    timers = {}
    t_all = time.time()

    # ---- read this rank's part ----------------------------------------------------------
    t0 = time.time()
    par = pc.alpha_parameters()
    path = Path(args.root) / f"np{P}" / f"partition_{rank}.npz"
    part = pc.read_partition(path)
    soma_ids = np.sort(np.asarray(part["soma_ids"], dtype=np.int64))
    n_local = soma_ids.size
    pre = np.asarray(part["pre"], dtype=np.int64)
    post = np.asarray(part["post"], dtype=np.int64)
    weight = pc.synapse_weights(part, par["J_E"] * 1e-12) * 1e12          # A -> pA
    remote_ids = np.asarray(part.get("remote_ids", np.zeros(0)), dtype=np.int64)
    remote_rank_of = np.asarray(part.get("remote_rank_of", np.zeros(0)), dtype=np.int64)
    timers["read"] = time.time() - t0

    def local_index(ids):
        idx = np.searchsorted(soma_ids, ids)
        if ids.size and (idx.max() >= n_local or not np.array_equal(soma_ids[idx], ids)):
            raise ValueError(f"{path}: a soma id is not local to rank {rank}")
        return idx

    external = pre < 0
    local_pre = ~external & np.isin(pre, soma_ids)
    remote_pre = ~external & ~local_pre
    order = np.argsort(remote_ids)
    remote_ids, remote_rank_of = remote_ids[order], remote_rank_of[order]
    at = np.searchsorted(remote_ids, pre[remote_pre])
    if remote_pre.any() and (at.max() >= remote_ids.size
                             or not np.array_equal(remote_ids[at], pre[remote_pre])):
        raise ValueError(f"{path}: a remote pre soma is missing from remote_ids")
    owner = np.full(pre.size, -1, dtype=np.int64)
    owner[remote_pre] = remote_rank_of[at]
    post_l = local_index(post)

    # ---- boundary: who listens to whom, in local node indices -----------------------------
    t0 = time.time()
    listened = [np.unique(pre[remote_pre & (owner == i)]) for i in range(P)]   # global ids on i
    asked = comm.alltoall(listened)                   # asked[j]: my somas rank j listens to
    asked_local = [local_index(np.asarray(a, dtype=np.int64)) for a in asked]
    listened_local = comm.alltoall(asked_local)       # listened_local[i]: listened[i] on rank i
    timers["boundary_exchange"] = time.time() - t0

    # ---- nodes ---------------------------------------------------------------------------
    t0 = time.time()
    # The connection key as NEST GPU's own hpc_benchmark.py sets it: 31 bits of node index
    # (the default 20 overflow silently: a RemoteConnect numbers every entry of its source
    # array, one per synapse here), which leaves one delay bit, so every delay is the
    # minimum allowed one, and no synapse groups or receptor ports.
    ngpu.SetKernelStatus({
        "verbosity_level": 1,
        "time_resolution": pc.DT_MS,
        "max_node_n_bits": 31,
        "max_syn_n_bits": 0,
        "min_allowed_delay": par["delay_ms"],
        "max_n_ports_warning": False,
    })
    neurons = ngpu.Create("iaf_psc_alpha", n_local)
    if neurons.i0 != 0:
        raise RuntimeError("the neurons must be the first nodes: remote lists use 0..n-1")
    ngpu.SetStatus(neurons, {
        "tau_m": par["tau_m"], "C_m": par["C_m"], "E_L": par["E_L"], "I_e": 0.0,
        "Theta_rel": par["V_th"] - par["E_L"], "V_reset_rel": par["V_reset"] - par["E_L"],
        "tau_syn_ex": par["tau_syn_ex"], "tau_syn_in": par["tau_syn_in"], "t_ref": par["t_ref"],
    })
    generators = ngpu.Create("spike_generator", n_local)        # one per soma's external input
    timers["create"] = time.time() - t0

    delay = par["delay_ms"]

    def connect_groups(src, tgt, w, connect):
        for wv in np.unique(w):                     # one call per distinct weight, in order
            m = w == wv
            connect(src[m].tolist(), tgt[m].tolist(), {"rule": "one_to_one"},
                    {"weight": float(wv), "delay": delay})

    # ---- local recurrent synapses ---------------------------------------------------------
    t0 = time.time()
    connect_groups(local_index(pre[local_pre]), post_l[local_pre], weight[local_pre], ngpu.Connect)
    timers["connect_local"] = time.time() - t0

    # ---- remote synapses: the source rank mirrors the target rank's calls -------------------
    t0 = time.time()
    calls_in = [[] for _ in range(P)]     # calls_in[i]: (weight, src on i, tgt here), by weight
    n_remote = 0
    for i in range(P):
        sel = remote_pre & (owner == i)
        if i == rank or not sel.any():
            continue
        src_on_i = listened_local[i][np.searchsorted(listened[i], pre[sel])].astype(np.int32)
        tgt, w = post_l[sel].astype(np.int32), weight[sel]
        for wv in np.unique(w):
            m = w == wv
            calls_in[i].append((float(wv), src_on_i[m], tgt[m]))
        n_remote += int(sel.sum())
    calls_out = comm.alltoall(calls_in)   # calls_out[j]: the calls rank j makes with me as source
    timers["edge_exchange"] = time.time() - t0

    def distinct_source_calls(s, t):
        """One_to_one edges split into calls whose source nodes are all distinct: call l holds
        every source's l-th edge. NEST GPU maps only the first entry of a source node that is
        repeated in one RemoteConnect (remote_connect.cu, searchNodeIndexNotInMapKernel skips
        the others and insertNodesInMapKernel then reads an unset image index for them)."""
        by_source = np.argsort(s, kind="stable")
        ss = s[by_source]
        start = np.flatnonzero(np.r_[True, ss[1:] != ss[:-1]])
        nth = np.arange(ss.size) - np.repeat(start, np.diff(np.r_[start, ss.size]))
        by_call = np.argsort(nth, kind="stable")
        cut = np.searchsorted(nth[by_call], np.arange(int(nth.max()) + 2))
        for l in range(cut.size - 1):
            idx = by_source[by_call[cut[l]:cut[l + 1]]]
            yield s[idx], t[idx]

    t0 = time.time()
    n_calls = 0
    for a in range(P):                    # every rank walks the host pairs in the same order
        for b in range(P):
            if a == b or rank not in (a, b):
                continue
            for wv, s, t in (calls_in[a] if b == rank else calls_out[b]):
                for sl, tl in distinct_source_calls(s, t):
                    ngpu.RemoteConnect(a, sl.tolist(), b, tl.tolist(), {"rule": "one_to_one"},
                                       {"weight": wv, "delay": delay})
                    n_calls += 1
    timers["connect_remote"] = time.time() - t0

    # ---- external drive: replayed spike lists --------------------------------------------
    t0 = time.time()
    steps = int(round(args.ms / pc.DT_MS))
    rate_hz = args.eta * pc.threshold_rate_hz(par)
    idx, tick, count = pc.drive(soma_ids, steps, rate_hz, args.seed)
    timers["drive_draw"] = time.time() - t0
    t0 = time.time()
    bounds = np.searchsorted(idx, np.arange(n_local + 1))
    for k in range(n_local):
        lo, hi = bounds[k], bounds[k + 1]
        if hi > lo:      # the events of one tick are one spike of that multiplicity
            ngpu.SetStatus(generators[k:k + 1], {
                "spike_times": ((tick[lo:hi] + 1) * pc.DT_MS).tolist(),
                "spike_gen_mul": count[lo:hi].astype(float).tolist()})
    ext_post = post_l[external]
    if np.unique(ext_post).size != ext_post.size or ext_post.size != n_local:
        raise ValueError(f"{path}: expected exactly one external input per soma")
    connect_groups(generators.i0 + ext_post, ext_post, weight[external], ngpu.Connect)
    timers["drive_setup"] = time.time() - t0

    # ---- run -----------------------------------------------------------------------------
    ngpu.ActivateRecSpikeTimes(neurons, 1000)
    t0 = time.time()
    ngpu.Calibrate()
    timers["calibrate"] = time.time() - t0
    t0 = time.time()
    ngpu.Simulate(args.ms)
    timers["simulate"] = time.time() - t0
    timers["total"] = time.time() - t_all

    # ---- spikes: (soma id, step), gathered on rank 0 -------------------------------------
    times = ngpu.GetRecSpikeTimes(neurons)
    full = max((len(t) for t in times), default=0) >= 1000
    soma = np.concatenate([np.full(len(t), soma_ids[k], np.int64) for k, t in enumerate(times)]
                          or [np.zeros(0, np.int64)])
    step = np.concatenate([np.rint(np.asarray(t) / pc.DT_MS).astype(np.int64) - 1 for t in times]
                          or [np.zeros(0, np.int64)])
    gathered = comm.gather((soma, step, timers, n_local, int(pre.size), n_remote, full, n_calls),
                           root=0)
    if rank == 0:
        soma = np.concatenate([g[0] for g in gathered])
        step = np.concatenate([g[1] for g in gathered])
        n_somas = sum(g[3] for g in gathered)
        n_syn = sum(g[4] for g in gathered)
        print(f"\nNEST GPU, {P} rank(s): {n_somas:,} somas, {n_syn:,} synapses "
              f"({sum(g[5] for g in gathered):,} cross-rank), {soma.size:,} spikes")
        if any(g[6] for g in gathered):
            print("  WARNING: a neuron filled its 1000-spike record")
        print(f"  RemoteConnect calls per rank: max {max(g[7] for g in gathered):,}   "
              f"min {min(g[7] for g in gathered):,}")
        for key in timers:
            vals = [g[2][key] for g in gathered]
            print(f"  {key:18s} max {max(vals):8.3f} s   min {min(vals):8.3f} s")
        burn = int(round(args.burn_in_ms / pc.DT_MS))
        late = step >= burn
        print(f"  rate after {args.burn_in_ms:.0f} ms: "
              f"{late.sum() / n_somas / ((steps - burn) * pc.DT_MS / 1000):.3f} Hz")
        out = args.out or str(Path(args.root) / f"spikes_nestgpu_np{P}.npz")
        pc.save_spikes(out, soma, step, {"n_somas": n_somas, "steps": steps, "ranks": P,
                                         "simulator": "nestgpu"})
        print(f"  spikes saved to {out}")


if __name__ == "__main__":
    main()
