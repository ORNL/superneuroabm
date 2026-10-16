"""One Brunel network across MPI ranks: the per-tick breakdown of the one-launch-per-tick path.

Each rank loads <split_root>/np<N>/partition_<rank>.npz (N = communicator size; N=1 loads
the single-rank file), applies a partition-INVARIANT Poisson drive (one global draw
indexed by post-soma id, each rank keeping the rows of its own input synapses), runs
<ticks> with SAGESim's per-tick timers on (one launch per tick on every rank), and rank 0
prints the mean per-tick cost of every timer over ticks 50..end, per rank, plus the total
spike count, and saves every (soma, tick) pair to <split_root>/spikes_np<N>.npy for
compare_spikes.py. Numbers from the laptop are in docs/EVENT_DRIVEN_NOTES.md section 16.

usage: mpirun -n N python multirank_tick.py <single_partition.npz> <split_root> [ticks=400]
       [--in-degree K=1000] [--rate-hz 14.53] [--n-somas 12500]
"""
import argparse
import time

import numpy as np
from mpi4py import MPI

from superneuroabm.model import NeuromorphicModel

KEYS = ["total", "data_prep", "kernel_args_build", "kernel_launch_overhead", "gpu_compute",
        "gpu_sync", "process_extras", "write_back", "mpi_gpu_pack", "mpi_gpu_sync_pack",
        "mpi_exchange", "mpi_isend_overhead", "mpi_irecv_overhead", "mpi_wait_time",
        "mpi_gpu_unpack", "mpi_reverse"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("single")
    ap.add_argument("split_root")
    ap.add_argument("ticks", type=int, nargs="?", default=400)
    ap.add_argument("--in-degree", type=int, default=1000)
    ap.add_argument("--rate-hz", type=float, default=14.53)
    ap.add_argument("--n-somas", type=int, default=12500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    comm = MPI.COMM_WORLD
    rank, N = comm.Get_rank(), comm.Get_size()
    part = args.single if N == 1 else f"{args.split_root}/np{N}/partition_{rank}.npz"
    lam = round(0.8 * args.in_degree) * args.rate_hz * 1.0 / 1000.0     # 1 ms ticks

    m = NeuromorphicModel(enable_internal_states_tracking=False, synapse_stepping="event",
                          learning_rule_info={})
    t0 = time.perf_counter()
    m.load_post_owned(part)
    t_load = time.perf_counter() - t0
    t0 = time.perf_counter()
    m.setup()
    t_setup = time.perf_counter() - t0

    with np.load(part, allow_pickle=True) as z:
        pre, post, syn_ids = z["pre"], z["post"], z["synapse_ids"]
    counts = np.random.default_rng(np.random.SeedSequence([args.seed, 7])).poisson(
        lam, size=(args.n_somas, args.ticks))
    ext = pre == -1
    for s, p in zip(syn_ids[ext].tolist(), post[ext].tolist()):
        tk = np.nonzero(counts[p])[0]
        if tk.size:
            m.add_local_spike_list(int(s), np.column_stack((tk, counts[p][tk])))

    m.verbose_timing = True
    comm.barrier()
    t0 = time.perf_counter()
    m.simulate(ticks=args.ticks, update_data_ticks=1)
    t_sim = time.perf_counter() - t0
    tt = list(m._tick_timings)
    n_spk_local = len(m._recorded_spikes) // 2
    n_spk = comm.allreduce(n_spk_local, op=MPI.SUM)
    pairs = np.asarray(m._recorded_spikes, dtype=np.float64).reshape(-1, 2)
    allp = comm.gather(pairs, root=0)
    if rank == 0:
        np.save(f"{args.split_root}/spikes_np{N}.npy", np.concatenate(allp))

    steady = tt[50:]
    means = {k: float(np.mean([t.get(k, 0.0) for t in steady])) * 1e3 for k in KEYS}
    info = {k: float(np.mean([t.get(k, 0) for t in steady]))
            for k in ("mpi_send_bytes", "mpi_recv_bytes", "mpi_num_peers")}
    rows = comm.gather((rank, int(ext.size), n_spk_local, t_load, t_setup,
                        tt[0].get("total", 0.0) if tt else float("nan"), t_sim, means, info), root=0)
    if rank == 0:
        print(f"\nRESULT N={N} ticks={args.ticks} total_spikes={n_spk}", flush=True)
        print(f"{'rank':>4} {'synapses':>10} {'spikes':>8} {'load_s':>7} {'setup_s':>7} {'tick1_s':>7} {'sim_s':>7} "
              + " ".join(f"{k[:14]:>14}" for k in KEYS) + f" {'send_B':>9} {'recv_B':>9} {'peers':>5}")
        for (r, ns, sp, tl, ts, t1, tsim, mn, inf) in sorted(rows):
            print(f"{r:>4} {ns:>10,} {sp:>8,} {tl:>7.1f} {ts:>7.1f} {t1:>7.2f} {tsim:>7.2f} "
                  + " ".join(f"{mn[k]:>14.3f}" for k in KEYS)
                  + f" {inf['mpi_send_bytes']:>9.0f} {inf['mpi_recv_bytes']:>9.0f} {inf['mpi_num_peers']:>5.0f}",
                  flush=True)
        print("(per-tick columns in ms, mean over ticks 50..end; total = SAGESim worker_coroutine wall per tick)",
              flush=True)


if __name__ == "__main__":
    main()
