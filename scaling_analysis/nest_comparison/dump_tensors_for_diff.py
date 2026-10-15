"""Dump every GPU property tensor + simulation output, for a bitwise before/after diff."""
import sys, os
sys.path.insert(0, '/home/xxz/superneuroabm')
sys.path.insert(0, '/home/xxz/superneuroabm/scaling_analysis')
import numpy as np
from superneuroabm.model import NeuromorphicModel
from superneuroabm.brunel import brunel_partition
import scaling_diagnostics as diagnostics

outdir = sys.argv[1]; os.makedirs(outdir, exist_ok=True)
N, K, TICKS, SEED = 2000, 100, 30, 42

arrays = brunel_partition(somas_per_rank=N, num_partitions=1, partition_rank=0,
                          excitatory_in_degree=round(0.8*K), inhibitory_in_degree=K-round(0.8*K),
                          topology="torus3d", seed=SEED, output_format="columns")
# SNA_SYNAPTIC_INPUT=push dumps the push path (docs/EVENT_DRIVEN_NOTES.md)
m = NeuromorphicModel(enable_internal_states_tracking=False,
                      synaptic_input=os.environ.get('SNA_SYNAPTIC_INPUT', 'pull'))
m.load_post_owned(arrays)
m.setup()
inp = m._input_synapse_ids.tolist()
diagnostics.inject_poisson_drive(m, inp, 500.0, TICKS, 1.0, SEED, 0)
m.simulate(ticks=TICKS, update_data_ticks=1)

buf = m._gpu_buffers
idx2name = {v: k for k, v in m._agent_factory._property_name_2_index.items()}
for i, t in enumerate(buf.property_tensors):
    if t is None: continue
    # interned properties (table + codes) are expanded to the dense rows they represent
    dense = buf.rows(i, slice(None)) if hasattr(buf, "is_interned") and buf.is_interned(i) else t
    np.save(os.path.join(outdir, f"prop_{i}_{idx2name.get(i,i)}.npy"), dense.get())
np.save(os.path.join(outdir, "csr_offsets.npy"), buf.neighbor_offsets.get())
np.save(os.path.join(outdir, "csr_values.npy"), buf.neighbor_values.get())
spikes = {int(s): m.get_spike_times(int(s)) for s in list(m._soma_ids)[:200]}
np.save(os.path.join(outdir, "spikes.npy"),
        np.array([len(spikes[k]) for k in sorted(spikes)], dtype=np.int64))
print(f"dumped {len(os.listdir(outdir))} arrays to {outdir}")
