"""
How synaptic current reaches a soma: synapses push, somas filter.

Every synapse deposits ``spike * scale * weight`` for the tick into one channel of its
post soma's inbox (a SAGESim reduce breed-local array, ``synaptic_input``, one row per
soma of ``NeuromorphicModel(synaptic_channels=K)`` slots, default 2). Deposits are atomic
adds into the inbox's write buffer; SAGESim copies and clears it at the end of the tick,
so the soma sees the previous tick's total. A synapse whose post soma lives on another
rank deposits into the soma's ghost row and SAGESim sums it back into the owner's row.

The synaptic kinetics live on the soma: ``synaptic_current`` below decays one state
variable per channel with that channel's time constant (``tau_syn_<c>`` in the soma's
hyperparameters, ``I_syn_<c>`` in its internal states), adds last tick's deposits and
returns the sum over channels. One soma can therefore receive afferents with K distinct
time constants (NEST's ``iaf_psc_exp_multisynapse`` receptor ports); the delivery
synapse's ``channel`` hyperparameter picks the slot. A channel whose time constant
equals ``dt`` has a decay factor of zero: an instantaneous, one-tick delivery (what the
former ``weighted_synapse`` did).

A kernel takes part by ending its parameter list with ``synaptic_input,
synaptic_input_idx``; ``NeuromorphicModel`` detects that and declares the array for the
breed. A soma kernel without the pair receives nothing from the inbox (a custom kernel
may still read its afferents' state directly, with synapse_stepping="clocked").

The add order inside a tick is not fixed, so results are reproducible only up to float
rounding order (docs/EVENT_DRIVEN_NOTES.md section 8).
"""

from cupyx import jit

from sagesim.utils import deposit_to_agent


@jit.rawkernel(device="cuda")
def has_inbox(agent_index, synaptic_input_idx):
    """True when this soma has an inbox row (every built-in soma has one)."""
    return synaptic_input_idx[agent_index] >= 0


@jit.rawkernel(device="cuda")
def read_synaptic_input(agent_index, synaptic_input, synaptic_input_idx, slot):
    """This soma's inbox slot: the total its afferent synapses deposited last tick
    (0.0 for an agent without an inbox row)."""
    row = synaptic_input_idx[agent_index]
    if row >= 0:
        return synaptic_input[row][slot]
    return 0.0


@jit.rawkernel(device="cuda")
def synaptic_current(agent_index, dt, neuron_params, tau_at, internal_states, state_at,
                     synaptic_input, synaptic_input_idx):
    """Synaptic current on the soma, one exponential filter per inbox channel.

    Channel c decays ``internal_states[agent_index][state_at + c]`` with
    ``neuron_params[agent_index][tau_at + c]`` (forward Euler, ``1 - dt/tau``), adds
    what the afferents deposited into slot c last tick, stores the new value and
    returns the sum over channels. The channel count is the inbox row width.
    """
    I_total = 0.0
    row = int(synaptic_input_idx[agent_index])
    if row >= 0:
        for c in range(len(synaptic_input[row])):
            I_c = (internal_states[agent_index][state_at + c]
                   * (1 - dt / neuron_params[agent_index][tau_at + c])
                   + synaptic_input[row][c])
            internal_states[agent_index][state_at + c] = I_c
            I_total += I_c
    return I_total


@jit.rawkernel(device="cuda")
def deliver_synaptic_current(synaptic_input, synaptic_input_idx, post_soma_index, slot, current):
    """Deposit this synapse's contribution for the tick into one channel of the post
    soma's inbox. A post soma without an inbox row (or the -1 sentinel) gets nothing.

    Call it after the last write to the value: a later priority that still changes
    the synapse's state is not reflected in what the soma receives.
    """
    deposit_to_agent(synaptic_input, synaptic_input_idx, post_soma_index, slot, current)
