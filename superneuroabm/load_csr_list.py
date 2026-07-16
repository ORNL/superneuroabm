import cupy as cp
import numpy as np


def build_incoming_csr(
    number_of_neurons,
    pre_indices,
    post_indices,
    synapse_types,
    weights,
    delays,
    scales,
    tau_fall,
    tau_rise=None,
):
    """
    Construct GPU arrays for the fused neuron-synapse kernel.

    All delays are specified in integer timesteps.
    """

    pre_indices = np.asarray(pre_indices, dtype=np.int32)
    post_indices = np.asarray(post_indices, dtype=np.int32)
    synapse_types = np.asarray(synapse_types, dtype=np.int32)

    weights = np.asarray(weights, dtype=np.float32)
    delays = np.asarray(delays, dtype=np.int32)
    scales = np.asarray(scales, dtype=np.float32)
    tau_fall = np.asarray(tau_fall, dtype=np.float32)

    number_of_synapses = len(pre_indices)

    if tau_rise is None:
        tau_rise = np.ones(number_of_synapses, dtype=np.float32)
    else:
        tau_rise = np.asarray(tau_rise, dtype=np.float32)

    if not (
        len(post_indices)
        == len(synapse_types)
        == len(weights)
        == len(delays)
        == len(scales)
        == len(tau_fall)
        == len(tau_rise)
        == number_of_synapses
    ):
        raise ValueError("All synapse arrays must have the same length.")

    if np.any(pre_indices < 0) or np.any(
        pre_indices >= number_of_neurons
    ):
        raise ValueError("Invalid presynaptic neuron index.")

    if np.any(post_indices < 0) or np.any(
        post_indices >= number_of_neurons
    ):
        raise ValueError("Invalid postsynaptic neuron index.")

    if np.any(delays < 1):
        raise ValueError(
            "The fused kernel requires delay_steps >= 1."
        )

    valid_types = np.isin(
        synapse_types,
        [
            SYNAPSE_WEIGHTED,
            SYNAPSE_SINGLE_EXP,
            SYNAPSE_BI_EXP,
        ],
    )

    if not np.all(valid_types):
        raise ValueError("Unsupported synapse type.")

    # Sort all synapses by postsynaptic neuron.
    order = np.argsort(post_indices, kind="stable")

    pre_indices = pre_indices[order]
    post_indices = post_indices[order]
    synapse_types = synapse_types[order]
    weights = weights[order]
    delays = delays[order]
    scales = scales[order]
    tau_fall = tau_fall[order]
    tau_rise = tau_rise[order]

    # CSR row pointer.
    counts = np.bincount(
        post_indices,
        minlength=number_of_neurons,
    )

    row_ptr = np.zeros(number_of_neurons + 1, dtype=np.int32)
    row_ptr[1:] = np.cumsum(counts, dtype=np.int32)

    # Parameters are initialized later because decay factors depend on dt.
    raw_params = {
        "weight": weights,
        "delay": delays,
        "scale": scales,
        "tau_fall": tau_fall,
        "tau_rise": tau_rise,
    }

    return {
        "incoming_row_ptr": cp.asarray(row_ptr),
        "pre_neuron_indices": cp.asarray(pre_indices),
        "synapse_types": cp.asarray(synapse_types),
        "raw_params": raw_params,
        "number_of_synapses": number_of_synapses,
    }

def initialize_synapse_arrays(csr_data, dt):
    """
    Create synapse parameter and state arrays.

    synapse_params columns:
        0 weight
        1 delay_steps
        2 scale
        3 decay_fall
        4 decay_rise
        5 bi-exponential normalization
    """

    raw = csr_data["raw_params"]

    weights = raw["weight"]
    delays = raw["delay"]
    scales = raw["scale"]
    tau_fall = raw["tau_fall"]
    tau_rise = raw["tau_rise"]

    number_of_synapses = csr_data["number_of_synapses"]

    if np.any(tau_fall <= 0.0):
        raise ValueError("tau_fall must be greater than zero.")

    decay_fall = np.exp(-dt / tau_fall).astype(np.float32)

    decay_rise = np.zeros(
        number_of_synapses,
        dtype=np.float32,
    )

    normalization = np.ones(
        number_of_synapses,
        dtype=np.float32,
    )

    synapse_types_cpu = cp.asnumpy(
        csr_data["synapse_types"]
    )

    bi_mask = synapse_types_cpu == SYNAPSE_BI_EXP

    if np.any(bi_mask):
        if np.any(tau_rise[bi_mask] <= 0.0):
            raise ValueError(
                "tau_rise must be greater than zero "
                "for bi-exponential synapses."
            )

        if np.any(tau_fall[bi_mask] <= tau_rise[bi_mask]):
            raise ValueError(
                "Bi-exponential synapses normally require "
                "tau_fall > tau_rise."
            )

        decay_rise[bi_mask] = np.exp(
            -dt / tau_rise[bi_mask]
        ).astype(np.float32)

        normalization[bi_mask] = biexponential_normalization(
            tau_rise[bi_mask],
            tau_fall[bi_mask],
        )

    synapse_params = np.zeros(
        (number_of_synapses, 6),
        dtype=np.float32,
    )

    synapse_params[:, 0] = weights
    synapse_params[:, 1] = delays.astype(np.float32)
    synapse_params[:, 2] = scales
    synapse_params[:, 3] = decay_fall
    synapse_params[:, 4] = decay_rise
    synapse_params[:, 5] = normalization

    # Two states per synapse:
    # state 0 = fall/current
    # state 1 = rise
    synapse_states = np.zeros(
        (number_of_synapses, 2),
        dtype=np.float32,
    )

    return (
        cp.asarray(synapse_params),
        cp.asarray(synapse_states),
    )

if __name__ == "__main__":
    # Example usage
    number_of_neurons = 5
    dt = 1.0e-3

    csr = build_incoming_csr(
        number_of_neurons=number_of_neurons,

        pre_indices=[
            0,
            1,
            1,
            3,
            3,
        ],

        post_indices=[
            2,
            2,
            4,
            2,
            4,
        ],

        synapse_types=[
            SYNAPSE_WEIGHTED,
            SYNAPSE_SINGLE_EXP,
            SYNAPSE_BI_EXP,
            SYNAPSE_SINGLE_EXP,
            SYNAPSE_WEIGHTED,
        ],

        weights=[
            0.5,
            0.8,
            1.0,
            -0.4,
            0.3,
        ],

        delays=[
            1,
            2,
            1,
            3,
            1,
        ],

        scales=[
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ],

        tau_fall=[
            1.0,
            5.0e-3,
            10.0e-3,
            7.0e-3,
            1.0,
        ],

        tau_rise=[
            1.0,
            1.0,
            2.0e-3,
            1.0,
            1.0,
        ],
    )

    synapse_params, synapse_states = initialize_synapse_arrays(
        csr,
        dt,
    )

    incoming_row_ptr = csr["incoming_row_ptr"]
    pre_neuron_indices = csr["pre_neuron_indices"]
    synapse_types = csr["synapse_types"]
    
    number_of_neuron_states = 10

    internal_states = cp.zeros(
        (
            number_of_neurons,
            number_of_neuron_states,
        ),
        dtype=cp.float32,
    )
    #Initialize internal states
    internal_states[:, 0] = -65.0 #Vmem
    internal_states[:, 1] = 0.0 #dt


    #Allocate output spike history tensor. The length of the ring buffer is determined by the maximum delay in the network.
    maximum_delay = int(delays.max())
    spike_history_length = maximum_delay + 1

    output_spikes_tensor = cp.zeros(
        (
            number_of_neurons,
            spike_history_length,
        ),
        dtype=cp.float32,
    )
    #launch the fused step function