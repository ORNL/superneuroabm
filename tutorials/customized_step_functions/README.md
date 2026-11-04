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
| `user_customized_double_exp_synapse.py` | synapse: double exponential as `+w` / `-w` into two soma channels, event-capable | 02 |
| `user_customized_stdp.py` | learning rule: SuperNeuroMAT-style bounded STDP, event-capable | 01 |
| `user_customized_masquelier_soma.py` | soma: three-variable LIF of Masquelier et al. (2008) | 03 |
| `user_customized_masquelier_synapse.py` | synapse: delivery only, event-capable | 03 |
| `user_customized_masquelier_stdp.py` | learning rule: additive nearest-neighbour STDP, event-capable | 03 |
| `user_customized_diehl_cook_soma.py` | soma: conductance-based LIF with adaptive threshold, two inbox channels | 04 |
| `user_customized_diehl_cook_synapse.py` | synapse: same-tick pre spike into a conductance channel (`synapse_stepping="clocked"`) | 04, 05 |
| `user_customized_diehl_cook_stdp.py` | learning rule: triplet STDP of Diehl & Cook (2015) | 04 |
| `user_customized_naive_stdp.py` | learning rule: pair-wise STDP without the catch-up, right only when clocked (a counter-example) | 05 |
| `user_customized_gap_junction.py` | synapse: electrical coupling `g (v_pre - v_post)` every tick (`synapse_stepping="clocked"`) | 05 |
| `user_customized_depressing_synapse.py` | synapse: short-term depression, the delivered value shrinks with use and recovers, event-capable | 06 |
| `user_customized_conductance_lif.py` | soma: LIF whose two inbox channels are conductances with reversal potentials | 06 |
| `user_customized_brunel_soma.py` | soma: LIF with one exponential current per inbox channel, explained line by line | 07 |
| `user_customized_brunel_synapse.py` | synapse: delivery only, event-capable, explained line by line | 07 |

The notebooks put this folder on `sys.path` and import each kernel by its module name. The
file basename and the function name must both be globally unique: `setup()` generates
`from <module stem> import *` for every registered component, so a name that matches a
built-in (`lif.py`, `lif_soma_step_func`) would shadow it. See `docs/CUSTOM_COMPONENTS.md`.
