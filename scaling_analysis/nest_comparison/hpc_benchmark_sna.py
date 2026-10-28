"""Brunel balanced network on SuperNeuroABM, run like NEST GPU's hpc_benchmark.

The SuperNeuroABM side of the NEST GPU comparison in the nmbenchmarking repo
(hpc_benchmark_hg.py there). Every MPI rank builds its own slice of the network in memory
and simulates it; nothing is written to disk.

  * Network: `--neurons-per-rank` neurons per rank, 80 % excitatory, each receiving exactly
    C_E excitatory and C_I inhibitory synapses (C_E + C_I = `--in-degree`) drawn from the
    whole population across all ranks, plus one external synapse carrying a Poisson drive.
    The defaults are hpc_benchmark's (scale 0.1: 1,125 neurons per rank, in-degree 11,250).
  * Model parameters (neuron, the three synapse kinds) are in hpc_benchmark_sna.yaml next
    to this file; edit them there. Network size, drive and run length are flags.
  * Drive: hpc_benchmark's, `--eta` times the threshold rate of the C_E excitatory
    afferents, computed from the YAML so that it follows any parameter change.
  * Timers, as in hpc_benchmark: construct (generate + load), calibrate (setup()),
    presimulate (`--presim-ms`; the first simulate() call also builds the device buffers),
    simulate (`--sim-ms`, the number to compare). Each rank times its own steps; on more
    than one rank each prints its line, and the summary and the CSV row carry rank 0's.

usage:
    python hpc_benchmark_sna.py
    mpirun -n 4 python hpc_benchmark_sna.py --neurons-per-rank 1125 --in-degree 11250
    python hpc_benchmark_sna.py --config my_params.yaml --sim-ms 1000 --csv results/run.csv
"""

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
import yaml
from mpi4py import MPI

from superneuroabm.brunel import brunel_partition
from superneuroabm.model import NeuromorphicModel

HERE = Path(__file__).resolve().parent
SOMA_BREED, SYNAPSE_BREED = "lif_soma", "delivery_synapse"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default=str(HERE / "hpc_benchmark_sna.yaml"),
                   help="Model parameters (YAML, component_base_config schema).")
    p.add_argument("--soma-config", default="brunel_neuron",
                   help="Name of the lif_soma config in --config.")
    p.add_argument("--neurons-per-rank", type=int, default=1125,
                   help="Neurons on each rank (hpc_benchmark: scale * 11250).")
    p.add_argument("--in-degree", type=int, default=11250,
                   help="Recurrent synapses per neuron, C_E + C_I (hpc_benchmark: 11250).")
    p.add_argument("--excitatory-fraction", type=float, default=0.8,
                   help="Fraction of excitatory neurons; also splits the in-degree.")
    p.add_argument("--eta", type=float, default=1.685,
                   help="External drive relative to the threshold rate (hpc_benchmark: 1.685).")
    p.add_argument("--external-rate-hz", type=float, default=None,
                   help="Total Poisson rate per neuron in Hz; overrides --eta.")
    p.add_argument("--dt-ms", type=float, default=0.1, help="Tick length (hpc_benchmark: 0.1).")
    p.add_argument("--presim-ms", type=float, default=50.0,
                   help="Untimed warm-up before the timed run (hpc_benchmark: 50).")
    p.add_argument("--sim-ms", type=float, default=250.0,
                   help="Timed simulation (hpc_benchmark: 250).")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--csv", default=None, help="Append one summary row here (rank 0).")
    return p.parse_args()


def read_parameters(config_path, soma_config):
    """The soma hyperparameters and the three synapse weights from the YAML.

    The columnar loader takes one synapse config plus per-synapse weights, so the three
    kinds may differ only in weight; anything else would be silently ignored, so stop.
    """
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)
    soma = {k: float(v) for k, v in
            cfg["soma"][SOMA_BREED][soma_config]["hyperparameters"].items()}
    kinds = cfg["synapse"][SYNAPSE_BREED]
    weights = {}
    for name in ("excitatory", "inhibitory", "external"):
        weights[name] = float(kinds[name]["hyperparameters"]["weight"])
        for section in ("hyperparameters", "internal_states"):
            a = {k: float(v) for k, v in kinds["excitatory"][section].items() if k != "weight"}
            b = {k: float(v) for k, v in kinds[name][section].items() if k != "weight"}
            if a != b:
                raise ValueError(
                    f"{config_path}: synapse kind '{name}' differs from 'excitatory' in "
                    f"{section} ({b} vs {a}); the kinds may differ only in weight.")
    if weights["excitatory"] <= 0:
        raise ValueError("the excitatory weight must be positive")
    return soma, weights, kinds["external"]["hyperparameters"]


def threshold_rate_hz(soma, ext_hp, c_e):
    """Rate per afferent at which C_E afferents hold the mean potential at threshold.

    In lif_soma a spike adds w to the synaptic current, which decays with tau_syn and is
    integrated as I * scaling_factor / C, so its voltage integral is w * tau_syn *
    scaling_factor * R. hpc_benchmark's nu_thresh is the same quantity for its alpha
    current; with the shipped YAML both are 1.375 Hz.
    """
    charge_mv_s = (float(ext_hp["weight"]) * float(ext_hp["scale"]) * soma["tau_syn_0"]
                   * soma["scaling_factor"] * soma["R"])
    return (soma["vthr"] - soma["vrest"]) / (c_e * charge_mv_s)


def inject_poisson_drive(model, synapse_ids, rate_hz, ticks, dt_ms, rng):
    """Independent Poisson trains, one per external synapse: the count of spikes in a
    tick is one event whose value is that count (the synapse multiplies it by its weight)."""
    lam = rate_hz * dt_ms / 1000.0
    ids = np.asarray(synapse_ids, dtype=np.int64)
    rows_per_chunk = max(1, 20_000_000 // max(ticks, 1))
    for start in range(0, ids.size, rows_per_chunk):
        chunk = ids[start:start + rows_per_chunk]
        counts = rng.poisson(lam, size=(chunk.size, ticks))
        row, tick = np.nonzero(counts)
        model.add_spikes(chunk[row], tick, counts[row, tick])


def main():
    args = parse_args()
    # Only to pick this rank's slice of the network; nothing here communicates.
    rank, size = MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size()

    n = args.neurons_per_rank
    c_e = round(args.excitatory_fraction * args.in_degree)
    c_i = args.in_degree - c_e
    exc_per_rank = round(args.excitatory_fraction * n)
    presim_ticks = int(round(args.presim_ms / args.dt_ms))
    sim_ticks = int(round(args.sim_ms / args.dt_ms))

    soma, w, ext_hp = read_parameters(args.config, args.soma_config)
    rate_hz = (args.external_rate_hz if args.external_rate_hz is not None
               else args.eta * threshold_rate_hz(soma, ext_hp, c_e) * c_e)

    if rank == 0:
        print(f"Brunel hpc_benchmark on SuperNeuroABM: {size} rank(s) x {n} neurons, "
              f"in-degree {args.in_degree} (C_E {c_e}, C_I {c_i})")
        print(f"  weights E {w['excitatory']} I {w['inhibitory']} ext {w['external']} "
              f"(from {args.config})")
        print(f"  drive {rate_hz:.1f} Hz per neuron, dt {args.dt_ms} ms, "
              f"presim {args.presim_ms} ms, sim {args.sim_ms} ms")
        if presim_ticks == 0:
            print("  WARNING: --presim-ms 0, so the timed simulate includes the device "
                  "buffer build of the first call.")

    # ---- construct: this rank's slice, generated as arrays and loaded directly ----
    t0 = time.time()
    network = brunel_partition(
        somas_per_rank=n, num_partitions=size, partition_rank=rank,
        excitatory_fraction=args.excitatory_fraction,
        excitatory_in_degree=c_e, inhibitory_in_degree=c_i, topology="global",
        excitatory_weight=w["excitatory"],
        inhibitory_weight_ratio=-w["inhibitory"] / w["excitatory"],
        external_weight=w["external"],
        soma_breed=SOMA_BREED, soma_config=args.soma_config,
        synapse_breed=SYNAPSE_BREED, synapse_config="excitatory",
        seed=args.seed, output_format="columns")
    t_generate = time.time() - t0
    model = NeuromorphicModel(user_config=args.config,
                              enable_internal_states_tracking=False,
                              learning_rule_info={})   # static synapses: no STDP kernels
    model.set_global_property_value("dt", args.dt_ms / 1000.0)
    synapses_per_rank = network["synapse_ids"].size   # the same on every rank
    model.load_post_owned(network)
    del network
    t_construct = time.time() - t0

    # ---- calibrate ----
    t0 = time.time()
    model.setup()
    t_setup = time.time() - t0

    rng = np.random.default_rng(np.random.SeedSequence([args.seed, rank, 7]))
    inject_poisson_drive(model, model._input_synapse_ids, rate_hz,
                         presim_ticks + sim_ticks, args.dt_ms, rng)

    # ---- presimulate (untimed in the comparison), then the timed run ----
    t0 = time.time()
    if presim_ticks:
        model.simulate(ticks=presim_ticks, update_data_ticks=1)
    t_presim = time.time() - t0

    t0 = time.time()
    model.simulate(ticks=sim_ticks, update_data_ticks=1)
    t_sim = time.time() - t0

    # Spikes of the timed run (simulate() clears the record per call). The getter is
    # collective: every rank calls it and receives every rank's spikes.
    spike_times = model.get_all_spike_times()
    if size > 1:
        print(f"  rank {rank}: construct {t_construct:.3f} s, calibrate {t_setup:.3f} s, "
              f"presimulate {t_presim:.3f} s, simulate {t_sim:.3f} s", flush=True)

    if rank == 0:
        spikes = sum(len(ticks) for ticks in spike_times.values())
        spikes_e = sum(len(ticks) for soma, ticks in spike_times.items()
                       if soma % n < exc_per_rank)
        synapses = synapses_per_rank * size
        sim_s = args.sim_ms / 1000.0
        n_e, n_i = exc_per_rank * size, (n - exc_per_rank) * size
        rate = spikes / (n * size) / sim_s
        rate_e = spikes_e / n_e / sim_s
        rate_i = (spikes - spikes_e) / n_i / sim_s
        print(f"\n  synapses        {synapses:,} ({synapses // size:,} per rank)")
        print(f"  construct       {t_construct:9.3f} s   (generate {t_generate:.3f} s)")
        print(f"  calibrate       {t_setup:9.3f} s   (setup)")
        print(f"  presimulate     {t_presim:9.3f} s   ({presim_ticks} ticks, incl. device buffers)")
        print(f"  simulate        {t_sim:9.3f} s   ({sim_ticks} ticks, "
              f"{t_sim / max(sim_ticks, 1) * 1000:.3f} ms per tick)")
        print(f"  firing rate     {rate:.2f} Hz  (E {rate_e:.2f}, I {rate_i:.2f}; "
              f"{spikes:,} spikes in the timed run)")
        if spikes == 0:
            print("  WARNING: silent network -- the simulate time describes an idle network.")

        if args.csv:
            row = {
                "job_id": os.environ.get("SLURM_JOB_ID", ""), "ranks": size,
                "neurons_per_rank": n, "in_degree": args.in_degree, "synapses": synapses,
                "dt_ms": args.dt_ms, "presim_ms": args.presim_ms, "sim_ms": args.sim_ms,
                "drive_hz": f"{rate_hz:.3f}", "construct_s": f"{t_construct:.4f}",
                "generate_s": f"{t_generate:.4f}", "calibrate_s": f"{t_setup:.4f}",
                "presimulate_s": f"{t_presim:.4f}", "simulate_s": f"{t_sim:.4f}",
                "spikes": spikes, "rate_hz": f"{rate:.3f}",
                "rate_e_hz": f"{rate_e:.3f}", "rate_i_hz": f"{rate_i:.3f}",
            }
            path = Path(args.csv)
            path.parent.mkdir(parents=True, exist_ok=True)
            new = not path.exists()
            with open(path, "a", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(row))
                if new:
                    writer.writeheader()
                writer.writerow(row)
            print(f"\n  appended to {path}")


if __name__ == "__main__":
    main()
