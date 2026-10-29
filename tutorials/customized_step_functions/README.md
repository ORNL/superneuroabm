# Tutorial-owned step functions

Every CUDA kernel the tutorials bring with them instead of using a built-in one. Each file
is one soma, synapse or learning rule, registered on the model at runtime with
`register_soma_type`, `register_synapse_type` or `register_learning_rule`; none of them is
part of the installed `superneuroabm` package. Each file's docstring is its contract: the
model, the parameter layout (the YAML key order is the index order the kernel reads) and the
registration call.

| file | kind | used by |
|---|---|---|
| `user_customized_lif.py` | soma: LIF with an explicit refractory reset | 01 |
| `user_customized_stdp.py` | learning rule: SuperNeuroMAT-style bounded STDP, event-capable | 01 |
| `user_customized_masquelier_soma.py` | soma: three-variable LIF of Masquelier et al. (2008) | 03 |
| `user_customized_masquelier_synapse.py` | synapse: delivery only, event-capable | 03 |
| `user_customized_masquelier_stdp.py` | learning rule: additive nearest-neighbour STDP, event-capable | 03 |
| `user_customized_diehl_cook_soma.py` | soma: conductance-based LIF with adaptive threshold, two inbox channels | 04 |
| `user_customized_diehl_cook_synapse.py` | synapse: same-tick pre spike into a conductance channel (`synapse_stepping="clocked"`) | 04 |
| `user_customized_diehl_cook_stdp.py` | learning rule: triplet STDP of Diehl & Cook (2015) | 04 |

The notebooks put this folder on `sys.path` and import each kernel by its module name. The
file basename and the function name must both be globally unique: `setup()` generates
`from <module stem> import *` for every registered component, so a name that matches a
built-in (`lif.py`, `lif_soma_step_func`) would shadow it. See `docs/CUSTOM_COMPONENTS.md`.
