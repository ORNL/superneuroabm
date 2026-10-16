"""Device-to-device coupling probe (row 9 of docs/WHY_SUPERNEUROABM.md): a device reads
another device, of another neuron, on another GPU.

P post neurons, each with a heater device and a sensor device. Sensor j's slot 0 is heater
(j+1) % P -- a device of a DIFFERENT neuron -- whose published heat disturbs sensor j's
conductance, which drives post j every tick. Post j lives on rank j % R, so with R > 1 every
sensor reads a heater owned by another rank. Pass: same spike digest and sensor
conductances on 1, 2 and 4 ranks, and the conductances are non-zero.
"""
import hashlib
import pickle
import tempfile
from pathlib import Path

import numpy as np
from mpi4py import MPI

from superneuroabm.model import NeuromorphicModel
import coupling_devices as k

HERE = Path(__file__).resolve().parent
N_RELAY, P, TICKS = 8, 16, 400
EXT, POST, HEAT, SENS = 1000, 100, 2000, 3000

comm = MPI.COMM_WORLD
rank, R = comm.Get_rank(), comm.Get_size()

somas, synapses, remote = [], [], {}
for r in range(N_RELAY):
    if r % R == rank:
        somas.append({"id": r, "breed": "r9_relay", "config": "default", "neighbors": [EXT + r]})
        synapses.append({"id": EXT + r, "breed": "r9_wire", "config": "default", "neighbors": [-1]})
for j in range(P):
    if j % R != rank:
        continue
    pre, nb = j % N_RELAY, (j + 1) % P
    somas.append({"id": POST + j, "breed": "r9_lif", "config": "default",
                  "neighbors": [HEAT + j, SENS + j]})
    synapses.append({"id": HEAT + j, "breed": "r9_heater", "config": "default",
                     "neighbors": [pre, POST + j]})
    synapses.append({"id": SENS + j, "breed": "r9_sensor", "config": "default",
                     "neighbors": [HEAT + nb, POST + j]})       # slot 0 = another DEVICE
    if pre % R != rank:
        remote[pre] = pre % R
    if nb % R != rank:
        remote[HEAT + nb] = nb % R

model = NeuromorphicModel(
    soma_breed_info={"r9_relay": (k.r9_relay, HERE / "coupling_devices.py"),
                     "r9_lif": (k.r9_lif, HERE / "coupling_devices.py")},
    synapse_breed_info={"r9_wire": (k.r9_wire, HERE / "coupling_devices.py"),
                        "r9_heater": (k.r9_heater, HERE / "coupling_devices.py"),
                        "r9_sensor": (k.r9_sensor, HERE / "coupling_devices.py")},
    user_config=HERE / "coupling_config.yaml",
    enable_internal_states_tracking=False,
)
model.set_seed(3)
with tempfile.TemporaryDirectory() as td:
    f = Path(td) / f"r{rank}.pkl"
    f.write_bytes(pickle.dumps({"somas": somas, "synapses": synapses, "remote_ranks": remote}))
    model.load_from_adjacency(str(f))
model.set_recorded_somas([POST + j for j in range(P)])
rng = np.random.default_rng(0)
fire = rng.random((TICKS, N_RELAY)) < 0.05
t, r = np.nonzero(fire)
model.add_spikes(EXT + r, t)
# The sensor reads its neighbour heater's internal_states on another rank. Only a
# soma's spike slot crosses ranks by default (2026-10-15); opt the property in.
model.set_property_neighbor_visible("internal_states", True)
model.setup()
model.simulate(ticks=TICKS, update_data_ticks=1)

spikes = model.get_all_spike_times()
g = [model.get_agent_property_value(SENS + j, "internal_states")[1] for j in range(P)]
if rank == 0:
    h = hashlib.sha256()
    for sid in sorted(spikes):
        h.update(np.int64(sid).tobytes() + np.asarray(sorted(spikes[sid]), np.int64).tobytes())
    h.update(np.asarray(g, np.float64).tobytes())
    n_sp = sum(len(v) for v in spikes.values())
    print(f"ranks={R} post spikes={n_sp} sensor g min/max={min(g):.6f}/{max(g):.6f} "
          f"digest={h.hexdigest()[:16]}")
