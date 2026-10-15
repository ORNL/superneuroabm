"""Probe what Brian2 can express of the hetero_chip PoCs (run in an env with Brian2 2.9).

    /home/xxz/miniforge3/envs/diehlcook/bin/python probe_brian2.py

Result on Brian2 2.9.0 (2026-10-01):
  [OK]   clock-driven synapse ODE reading v_post every step
  [FAIL] two Synapses objects with summed vars into the same post variable  (NotImplementedError)
  [FAIL] if/else statement in on_pre code
  [FAIL] for-loop in on_pre code
  [OK]   random numbers and per-synapse parameters in on_pre code
  [FAIL] synapse reads another synapse's state (linked_var into a Synapses variable)  (2026-10-13)
"""
import numpy as np
from brian2 import (NeuronGroup, Synapses, linked_var, ms, prefs, run, start_scope)

prefs.codegen.target = "numpy"


def probe(name, fn):
    try:
        fn()
        print(f"[OK]   {name}")
    except Exception as e:  # noqa: BLE001 - we want the error class and first line
        print(f"[FAIL] {name}: {type(e).__name__}: {str(e).splitlines()[0][:160]}")


def clock_driven_reads_vpost():
    """A device whose state evolves every step as a function of the post membrane."""
    start_scope()
    G = NeuronGroup(2, "dv/dt = -v/(10*ms) + I/(10*ms) : 1\nI : 1",
                    threshold="v>1", reset="v=0", method="euler")
    S = Synapses(G, G, "dg/dt = -g*sinh(v_post)/(100*ms) : 1 (clock-driven)",
                 on_pre="v_post += g")
    S.connect(i=0, j=1)
    S.g = 0.5
    run(1 * ms)


def two_summed_same_target():
    """Two device technologies (two Synapses objects) feeding one neuron's input current."""
    start_scope()
    G = NeuronGroup(4, "dv/dt = (-v + Isyn)/(10*ms) : 1\nIsyn : 1",
                    threshold="v>1", reset="v=0", method="euler")
    rram = Synapses(G, G, "g : 1\nIsyn_post = g : 1 (summed)")
    pcm = Synapses(G, G, "g : 1\nIsyn_post = g*0.5 : 1 (summed)")
    rram.connect(i=0, j=2)
    pcm.connect(i=1, j=2)
    run(1 * ms)


def if_in_code():
    """Device code with a branch (e.g. a stuck-at fault)."""
    start_scope()
    G = NeuronGroup(2, "v : 1", threshold="v>1", reset="v=0")
    S = Synapses(G, G, "g : 1\nkind : integer", on_pre="""
if kind == 0:
    v_post += g
else:
    v_post += 0.5*g
""")
    S.connect(i=0, j=1)
    run(1 * ms)


def loop_in_code():
    """Device code with a loop (e.g. an iterative operating-point solve)."""
    start_scope()
    G = NeuronGroup(2, "v : 1", threshold="v>1", reset="v=0")
    S = Synapses(G, G, "g : 1", on_pre="""
for k in range(5):
    g = g - 0.1*g
v_post += g
""")
    S.connect(i=0, j=1)
    run(1 * ms)


def random_and_per_synapse_params():
    """Write noise (randn per update) and device-to-device step size (per-synapse variable)."""
    start_scope()
    G = NeuronGroup(2, "v : 1", threshold="v>1", reset="v=0")
    S = Synapses(G, G, "g : 1\nalpha : 1", on_pre="g = clip(g + alpha*(1 + randn()), 0, 1)")
    S.connect(i=0, j=1)
    S.alpha = 0.01
    run(1 * ms)


def synapse_reads_synapse():
    """A device reading its physical neighbour's conductance (a cell of another neuron)."""
    start_scope()
    G = NeuronGroup(2, "v : 1", threshold="v>1", reset="v=0")
    S = Synapses(G, G, "g : 1\ng_nb : 1 (linked)", on_pre="v_post += g")
    S.connect(i=[0, 0, 1, 1], j=[0, 1, 0, 1])
    S.g_nb = linked_var(S, "g", index=np.array([1, 0, 3, 2]))
    run(1 * ms)


if __name__ == "__main__":
    probe("clock-driven synapse ODE reading v_post every step", clock_driven_reads_vpost)
    probe("two Synapses objects with summed vars into the same post variable", two_summed_same_target)
    probe("if/else statement in on_pre code", if_in_code)
    probe("for-loop in on_pre code", loop_in_code)
    probe("random numbers and per-synapse parameters in on_pre code", random_and_per_synapse_params)
    probe("synapse reads another synapse's state (linked_var into a Synapses variable)",
          synapse_reads_synapse)
