"""
How synaptic current reaches a soma: pulled or pushed.

pull (default): each soma loops over its afferent synapses and sums their current
slots. One thread per soma, so a soma with K afferents does K serial reads per tick;
a single neuron with 2,000 afferents spends 83 % of the tick in that loop
(docs/EVENT_DRIVEN_NOTES.md).

push: each synapse deposits its final current for the tick into the post soma's inbox
(a SAGESim reduce breed-local array, ``synaptic_input``, one row per soma of
``NeuromorphicModel(synaptic_channels=K)`` slots, default 2). A slot is a channel: the
delivery synapse's ``channel`` hyperparameter picks it and the psc soma decays each
channel with its own ``tau_syn_<c>``, so one soma can receive afferents with K distinct
time constants (NEST's receptor ports). The clocked kinetic synapses use slot 0 (and the
double-exponential one slot 1 for its slow component). The soma reads one row. Deposits
are atomic adds into the inbox's write buffer; SAGESim copies and clears it at the end
of the tick, so the soma sees the previous tick's total, exactly what the pull loop
summed. The add order is not fixed, so push runs are not bitwise reproducible
(docs/EVENT_DRIVEN_NOTES.md section 8).

The mode is carried by the inbox's index map, not by a constant: in push mode every
soma has an inbox row, in pull mode the inbox is registered with no member breeds and
every index is -1. ``has_inbox`` is therefore the mode test a soma kernel makes, and a
synapse's deposit to a row-less target is a no-op. (A compile-time constant was tried
first and is unsafe: CuPy compiles a kernel at its first launch and SAGESim reuses an
identical generated kernel across models, so a pull model's compilation could serve a
push model in the same process.)

A kernel takes part by ending its parameter list with ``synaptic_input,
synaptic_input_idx``; ``NeuromorphicModel`` detects that and declares the array for
the breed. Kernels without the pair keep their signature and pull.
"""

from cupyx import jit

from sagesim.utils import deposit_to_agent


@jit.rawkernel(device="cuda")
def has_inbox(agent_index, synaptic_input_idx):
    """True when this soma has an inbox row, i.e. the model runs synaptic_input="push"."""
    return synaptic_input_idx[agent_index] >= 0


@jit.rawkernel(device="cuda")
def read_synaptic_input(agent_index, synaptic_input, synaptic_input_idx, slot):
    """This soma's inbox slot: the total its afferent synapses deposited last tick
    (0.0 in pull mode, where the soma has no row)."""
    row = synaptic_input_idx[agent_index]
    if row >= 0:
        return synaptic_input[row][slot]
    return 0.0


@jit.rawkernel(device="cuda")
def deliver_synaptic_current(synaptic_input, synaptic_input_idx, post_soma_index, slot, current):
    """Deposit this synapse's final current for the tick into the post soma's inbox.
    In pull mode the post soma has no row and nothing happens (the soma reads
    internal_states[synapse][0] instead).

    Call it after the last write to the current: a later priority that still changes
    the synapse's state is not reflected in what the soma receives.
    """
    deposit_to_agent(synaptic_input, synaptic_input_idx, post_soma_index, slot, current)
