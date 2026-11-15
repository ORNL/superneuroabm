"""What run_partitioned_sna.py and run_partitioned_nestgpu.py share: the network's neuron and
synapse parameters, the partition reader and the external drive.

numpy and yaml only, so it imports in both the SuperNeuroABM and the NEST GPU environment.

Both simulators are handed the same data and nothing else: the per-rank partition files that
make_brunel_partitions.py writes (the columnar post-owned schema of model.load_network), and a
drive that is a fixed function of the soma id, so every rank can draw its own somas' input
without communication and get the same input at every rank count.

The neuron is the tutorial's alpha neuron (tutorials/customized_step_functions/
user_customized_brunel_alpha_soma.py, NEST's iaf_psc_alpha with a 1.5 ms delay), whose match
with NEST GPU's iaf_psc_alpha is verified there. Its timing rules, measured in that
verification, apply here unchanged:

  * a spike reaches the post soma in FRAMEWORK_TICKS on its own; the soma's delay_ticks
    queue adds the rest, so delay_ticks + FRAMEWORK_TICKS = round(delay / dt);
  * recorded spike ticks are emission ticks, delay_ticks late;
  * injected input skips the queue, so SuperNeuroABM injects drive tick t at t + delay
    ticks, and NEST GPU's spike_generator fires it at (t + 1) * dt into a connection with
    the full delay.
"""

from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
TUTORIALS = REPO / "tutorials"
STEP_FUNC_DIR = TUTORIALS / "customized_step_functions"
ALPHA_YAML = TUTORIALS / "customized_configs" / "brunel_alpha_config.yaml"

SOMA_BREED = "user_customized_brunel_alpha_soma"
SOMA_CONFIG = "brunel_alpha_lif_neuron"
SYNAPSE_BREED = "user_customized_brunel_synapse"
SYNAPSE_CONFIG = "brunel_alpha_excitatory_synapse"   # every synapse; its weight is per synapse

DT_MS = 0.1
FRAMEWORK_TICKS = 2


def alpha_parameters():
    """The alpha neuron's YAML (SI units) in NEST units: pF, ms, mV, pA."""
    config = yaml.safe_load(ALPHA_YAML.read_text())
    soma = config["soma"][SOMA_BREED][SOMA_CONFIG]["hyperparameters"]
    synapse = config["synapse"][SYNAPSE_BREED][SYNAPSE_CONFIG]["hyperparameters"]
    par = {
        "C_m": float(soma["C_m"]) * 1e12,
        "tau_m": float(soma["tau_m"]) * 1e3,
        "tau_syn_ex": float(soma["tau_syn_0"]) * 1e3,
        "tau_syn_in": float(soma["tau_syn_1"]) * 1e3,
        "V_th": float(soma["V_th"]) * 1e3,
        "V_reset": float(soma["V_reset"]) * 1e3,
        "E_L": float(soma["E_L"]) * 1e3,
        "t_ref": float(soma["t_ref"]) * 1e3,
        "J_E": float(synapse["weight"]) * 1e12,
        "delay_ms": float(synapse["synaptic_delay"]) * 1e3,
        "delay_ticks": int(round(float(soma["delay_ticks"]))),
    }
    steps = int(round(par["delay_ms"] / DT_MS))
    if par["delay_ticks"] + FRAMEWORK_TICKS != steps:
        raise ValueError(f"{ALPHA_YAML}: delay_ticks {par['delay_ticks']} + {FRAMEWORK_TICKS} "
                         f"!= synaptic_delay of {steps} ticks")
    par["delay_steps"] = steps
    return par


def threshold_rate_hz(par):
    """Poisson rate per neuron at which the mean membrane potential reaches threshold, for
    alpha currents (brunel_alpha_nest.py's nu_th * CE): one event of peak J carries e*J*tau_syn."""
    return ((par["V_th"] - par["E_L"]) * par["C_m"]
            / (np.e * par["J_E"] * par["tau_syn_ex"] * par["tau_m"])) * 1000.0


def read_partition(path):
    """One rank's partition (columnar post-owned npz) as a dict of arrays/scalars."""
    with np.load(path, allow_pickle=True) as z:
        return {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files}


def synapse_weights(part, default_weight):
    """Per-synapse weight from syn_hp_vals (NaN keeps the config's), in the file's units."""
    keys = [str(k) for k in np.asarray(part["syn_hp_keys"]).ravel()]
    w = np.full(np.asarray(part["synapse_ids"]).size, float(default_weight))
    if "weight" in keys:
        col = np.asarray(part["syn_hp_vals"], dtype=np.float64)[:, keys.index("weight")]
        w = np.where(np.isnan(col), w, col)
    return w


def drive(soma_ids, ticks, rate_hz, seed):
    """Poisson events per tick for each soma's external input, drawn from a stream seeded by
    (seed, soma id) alone: the same input whatever rank holds the soma.

    A Brunel drive is dense (at eta = 2 a neuron gets ~1.8 events per 0.1 ms tick, so 1.5 s of
    it is ~12,500 events per neuron), hence the compact types: 9 bytes per event.

    :return: (soma_index int32, tick int32, count uint8) over the events, sorted by soma_index;
        soma_index indexes soma_ids
    """
    lam = rate_hz * DT_MS / 1000.0
    index, tick, count = [], [], []
    for k, soma in enumerate(np.asarray(soma_ids, dtype=np.int64).tolist()):
        c = np.random.default_rng(np.random.SeedSequence([seed, soma])).poisson(lam, ticks)
        if c.max(initial=0) > 255:
            raise ValueError("more than 255 events in one tick")
        nz = np.nonzero(c)[0]
        index.append(np.full(nz.size, k, dtype=np.int32))
        tick.append(nz.astype(np.int32))
        count.append(c[nz].astype(np.uint8))
    if not index:
        return np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.uint8)
    return np.concatenate(index), np.concatenate(tick), np.concatenate(count)


def save_spikes(path, soma, tick, meta):
    """(soma, tick) pairs sorted by tick then soma, plus run metadata."""
    order = np.lexsort((soma, tick))
    np.savez(path, soma=np.asarray(soma, np.int64)[order], tick=np.asarray(tick, np.int64)[order],
             **{k: np.asarray(v) for k, v in meta.items()})
