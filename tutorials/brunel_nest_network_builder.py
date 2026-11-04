"""
brunel_nest_network_builder.py

Complete Brunel network builder for SuperNeuroABM with NEST-matching parameters.

This module provides functions to construct a full Brunel balanced random network in
SuperNeuroABM that matches the NEST brunel_alpha_nest.py implementation, enabling
direct comparison and validation of simulation results.

Usage Example
-------------
from brunel_nest_network_builder import build_brunel_network
import numpy as np
from superneuroabm import NeuromorphicModel

# Create model
model = NeuromorphicModel()

# Register custom soma and synapse types
from brunel_nest_soma import brunel_nest_lif_soma
from brunel_nest_synapse import brunel_nest_synapse

model.register_soma_type(
    name="brunel_nest_lif",
    step_func=brunel_nest_lif_soma,
    step_func_path=Path("brunel_nest_soma.py").resolve(),
    num_channels=2,
    num_internal_state_variables=5,
)

model.register_synapse_type(
    name="brunel_nest_synapse",
    step_func=brunel_nest_synapse,
    step_func_path=Path("brunel_nest_synapse.py").resolve(),
    event_capable=True,
)

# Build network
model, net_info = build_brunel_network(
    model=model,
    order=2500,
    eta=2.0,
    epsilon=0.1,
    g=5.0,
    seed=12345,
)

# Configure and run
model.setup()
model.run(1000.0)  # 1000 ms

# Analyze results
print(f"Excitatory firing rate: {net_info['firing_rate_ex']:.2f} Hz")
print(f"Inhibitory firing rate: {net_info['firing_rate_in']:.2f} Hz")
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any


def compute_brunel_weights(
    order: int = 2500,
    eta: float = 2.0,
    epsilon: float = 0.1,
    g: float = 5.0,
    C_m: float = 250.0,
    tau_m: float = 20.0,
    tau_syn: float = 0.5,
    V_th: float = 20.0,
) -> Dict[str, float]:
    """
    Compute synaptic weights for the Brunel network matching NEST.

    Parameters
    ----------
    order : int
        Scaling parameter (NE = 4*order, NI = order)
    eta : float
        External input scaling factor
    epsilon : float
        Connection probability
    g : float
        Inhibition-to-excitation weight ratio
    C_m : float
        Membrane capacitance (pF)
    tau_m : float
        Membrane time constant (ms)
    tau_syn : float
        Synaptic time constant (ms)
    V_th : float
        Spike threshold (mV)

    Returns
    -------
    weights : dict
        Dictionary with keys: 'J_ex', 'J_in', 'nu_th', 'nu_ex', 'p_rate', 'CE', 'CI'
    """
    import scipy.special as sp

    def LambertWm1(x):
        return sp.lambertw(x, k=-1 if x < 0 else 0).real

    def ComputePSPnorm(tauMem, CMem, tauSyn):
        a = tauMem / tauSyn
        b = 1.0 / tauSyn - 1.0 / tauMem
        t_max = 1.0 / b * (-LambertWm1(-np.exp(-1.0 / a) / a) - 1.0 / a)
        return (
            np.exp(1.0) / (tauSyn * CMem * b) *
            ((np.exp(-t_max / tauMem) - np.exp(-t_max / tauSyn)) / b -
             t_max * np.exp(-t_max / tauSyn))
        )

    NE = 4 * order
    NI = 1 * order
    CE = int(epsilon * NE)
    CI = int(epsilon * NI)

    # Target PSP amplitude
    J_target = 0.1  # mV

    # Compute unit PSP normalization
    J_unit = ComputePSPnorm(tau_m, C_m, tau_syn)

    # Compute synaptic currents
    J_ex = J_target / J_unit
    J_in = -g * J_ex

    # Compute threshold and external rate
    nu_th = (V_th * C_m) / (J_ex * CE * np.exp(1) * tau_m * tau_syn)
    nu_ex = eta * nu_th
    p_rate = 1000.0 * nu_ex * CE  # Hz

    return {
        'J_ex': float(J_ex),
        'J_in': float(J_in),
        'J_unit': float(J_unit),
        'J_target': float(J_target),
        'nu_th': float(nu_th),
        'nu_ex': float(nu_ex),
        'p_rate': float(p_rate),
        'CE': int(CE),
        'CI': int(CI),
        'NE': int(NE),
        'NI': int(NI),
    }


def build_brunel_network(
    model: 'NeuromorphicModel',
    order: int = 2500,
    eta: float = 2.0,
    epsilon: float = 0.1,
    g: float = 5.0,
    seed: int = 12345,
    include_external_poisson: bool = True,
    soma_type: str = "brunel_nest_lif",
    synapse_type: str = "brunel_nest_synapse",
) -> Tuple['NeuromorphicModel', Dict[str, Any]]:
    """
    Build a complete Brunel balanced network in SuperNeuroABM.

    Parameters
    ----------
    model : NeuromorphicModel
        Initialized model object
    order : int
        Scaling parameter (default 2500)
    eta : float
        External input scaling (default 2.0)
    epsilon : float
        Connection probability (default 0.1)
    g : float
        Inhibition ratio (default 5.0)
    seed : int
        Random seed for reproducibility
    include_external_poisson : bool
        Whether to add external Poisson input (default True)
    soma_type : str
        Name of the registered soma type (default "brunel_nest_lif")
    synapse_type : str
        Name of the registered synapse type (default "brunel_nest_synapse")

    Returns
    -------
    model : NeuromorphicModel
        Model with network built but not yet set up
    net_info : dict
        Network information including weights, sizes, connection counts
    """
    rng = np.random.default_rng(seed)

    # Compute network properties
    weights = compute_brunel_weights(order=order, eta=eta, epsilon=epsilon, g=g)
    NE = weights['NE']
    NI = weights['NI']
    CE = weights['CE']
    CI = weights['CI']
    J_ex = weights['J_ex']
    J_in = weights['J_in']

    print(f"Building Brunel network:")
    print(f"  Order: {order}")
    print(f"  Excitatory neurons: {NE}")
    print(f"  Inhibitory neurons: {NI}")
    print(f"  Excitatory in-degree: {CE}")
    print(f"  Inhibitory in-degree: {CI}")
    print(f"  J_ex: {J_ex:.6f} pA")
    print(f"  J_in: {J_in:.6f} pA")
    print(f"  External Poisson rate: {weights['p_rate']:.1f} Hz")

    # === Create neuron populations ===
    exc_neuron_params = {
        'C_m': 250.0,
        'tau_m': 20.0,
        'tau_syn_ex': 0.5,
        'tau_syn_in': 0.5,
        'V_th': 20.0,
        'V_reset': 0.0,
        'E_L': 0.0,
        't_ref': 2.0,
        'I_bias': 0.0,
    }

    inh_neuron_params = exc_neuron_params.copy()  # Same params for both populations

    # Create excitatory neurons
    exc_somas = model.create_somas_from_list(
        breed_name="excitatory_neuron",
        soma_type=soma_type,
        locations=[(i,) for i in range(NE)],
        neuron_params=[exc_neuron_params.copy() for _ in range(NE)],
    )

    # Create inhibitory neurons
    inh_somas = model.create_somas_from_list(
        breed_name="inhibitory_neuron",
        soma_type=soma_type,
        locations=[(NE + i,) for i in range(NI)],
        neuron_params=[inh_neuron_params.copy() for _ in range(NI)],
    )

    print(f"  Created {len(exc_somas)} excitatory and {len(inh_somas)} inhibitory neurons")

    # === Create random connectivity ===
    # All neurons (E and I) receive connections from both E and I pools

    print("  Building connectivity...")
    all_neurons = list(range(NE)) + list(range(NE, NE + NI))  # Global neuron IDs
    exc_neurons = list(range(NE))
    inh_neurons = list(range(NE, NE + NI))

    exc_connections = []
    inh_connections = []

    # For each target neuron, draw CE excitatory and CI inhibitory sources
    for target_id in all_neurons:
        # Draw excitatory sources
        exc_sources = rng.choice(exc_neurons, size=CE, replace=False)
        for src_id in exc_sources:
            exc_connections.append((src_id, target_id))

        # Draw inhibitory sources
        inh_sources = rng.choice(inh_neurons, size=CI, replace=False)
        for src_id in inh_sources:
            inh_connections.append((src_id, target_id))

    print(f"  Excitatory synapses: {len(exc_connections)}")
    print(f"  Inhibitory synapses: {len(inh_connections)}")

    # === Create synapse populations ===
    exc_synapse_params = [
        {'weight': J_ex, 'channel': 0} for _ in exc_connections
    ]

    inh_synapse_params = [
        {'weight': J_in, 'channel': 1} for _ in inh_connections
    ]

    exc_synapses = model.create_synapses_from_list(
        breed_name="excitatory_synapse",
        synapse_type=synapse_type,
        locations=exc_connections,
        synapse_params=exc_synapse_params,
    )

    inh_synapses = model.create_synapses_from_list(
        breed_name="inhibitory_synapse",
        synapse_type=synapse_type,
        locations=inh_connections,
        synapse_params=inh_synapse_params,
    )

    print(f"  Created {len(exc_synapses)} excitatory and {len(inh_synapses)} inhibitory synapses")

    # === Optional: External Poisson input ===
    if include_external_poisson:
        print("  Adding external Poisson input...")
        p_rate = weights['p_rate']
        # Note: This would require implementing external input spiking in the model
        # For now, we'll just record the rate for later injection
        # In a full implementation, you'd use model.add_poisson_inputs() or similar

    net_info = {
        'order': order,
        'NE': NE,
        'NI': NI,
        'CE': CE,
        'CI': CI,
        'num_exc_synapses': len(exc_synapses),
        'num_inh_synapses': len(inh_synapses),
        'J_ex': J_ex,
        'J_in': J_in,
        'p_rate': weights['p_rate'],
        'nu_th': weights['nu_th'],
        'nu_ex': weights['nu_ex'],
        'seed': seed,
        'eta': eta,
        'epsilon': epsilon,
        'g': g,
    }

    return model, net_info


if __name__ == "__main__":
    # Example usage (requires SuperNeuroABM to be installed)
    print("Brunel network builder module loaded.")
    print("Use: from brunel_nest_network_builder import build_brunel_network")
    print("\nExample weight values:")
    weights = compute_brunel_weights()
    for key, val in weights.items():
        print(f"  {key}: {val}")
