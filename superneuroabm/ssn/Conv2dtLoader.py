"""
Load a Conv2dtNet from a saved JSON file.

This module reconstructs a Conv2dtNet object with the exact structure 
and parameters from a JSON file exported by extract_full_network_with_topology().
"""

import json
import math
import numpy as np

# Import your existing Conv2dtNet class
from SpikingConv2dt import Conv2dtNet


def load_network_from_json(json_path, use_gpu=True):
    """
    Load a Conv2dtNet from a saved JSON file.
    
    Args:
        json_path: Path to the JSON file
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        tuple: (Conv2dtNet, soma_id_map, synapse_id_map)
            - Conv2dtNet: Reconstructed network with exact weights and topology
            - soma_id_map: dict mapping old soma IDs to new soma IDs
            - synapse_id_map: dict mapping old synapse IDs to new synapse IDs
    """
    with open(json_path, "r") as f:
        structure = json.load(f)
    
    # Create new instance
    net = Conv2dtNet()
    net.model.setup(use_gpu=use_gpu)
    
    # Build ID mappings: old_id -> new_id
    soma_id_map = {}
    synapse_id_map = {}
    
    # --- Recreate Somas ---
    print(f"Recreating {len(structure['somas'])} somas...")
    for soma_info in structure["somas"]:
        old_id = soma_info["soma_id"]
        hyper = soma_info["hyperparameters"]
        internal = soma_info["internal_state"]
        
        # Handle NaN values in hyperparameters
        hyper = [0.0 if (isinstance(v, float) and math.isnan(v)) else v for v in hyper]
        internal = [0.0 if (isinstance(v, float) and math.isnan(v)) else v for v in internal]
        
        new_soma = net.model.create_soma(
            breed="lif_soma",
            config_name="config_0",
            hyperparameters_overrides={
                'C':               hyper[0],
                'R':               hyper[1],
                'vthr':            hyper[2],
                'tref':            hyper[3],
                'vrest':           hyper[4],
                'vreset':          hyper[5],
                'tref_integration': hyper[6],
                'I_in':            hyper[7],
                'scaling_factor':  hyper[8],
            },
            default_internal_state_overrides={
                'v':      internal[0],
                'tcount': internal[1],
                'tlast':  internal[2],
            }
        )
        soma_id_map[old_id] = new_soma
    
    # --- Recreate Synapses ---
    print(f"Recreating {len(structure['synapses'])} synapses...")
    for syn_info in structure["synapses"]:
        old_id = syn_info["synapse_id"]
        old_pre = syn_info["pre"]
        old_post = syn_info["post"]
        hyper = syn_info["hyperparameters"]
        lhyper = syn_info["learning_hyperparameters"]
        internal = syn_info["internal_state"]
        lint = syn_info["internal_learning_state"]
        
        # Handle NaN values
        hyper = [0.0 if (isinstance(v, float) and math.isnan(v)) else v for v in hyper]
        lhyper = [0.0 if (isinstance(v, float) and math.isnan(v)) else v for v in lhyper]
        internal = [0.0 if (isinstance(v, float) and math.isnan(v)) else v for v in internal]
        lint = [0.0 if (isinstance(v, float) and math.isnan(v)) else v for v in lint]
        
        # Map old IDs to new IDs (-1 stays as -1 for input synapses)
        new_pre = soma_id_map.get(old_pre, -1) if old_pre != -1 else -1
        new_post = soma_id_map[old_post]
        
        # Check if STDP is enabled (stdp_type == -1 means no STDP)
        stdp_type = lhyper[0] if lhyper else -1
        
        if stdp_type == -1:
            # No STDP synapse
            new_syn = net.model.create_synapse(
                breed="single_exp_synapse",
                pre_soma_id=new_pre,
                post_soma_id=new_post,
                config_name="no_learning_config_0",
                hyperparameters_overrides={
                    "weight":         hyper[0],
                    "synaptic_delay": hyper[1],
                    "scale":          hyper[2],
                    "tau_fall":       hyper[3],
                    "tau_rise":       hyper[4],
                },
                default_internal_state_overrides={
                    "I_synapse": internal[0],
                },
                learning_hyperparameters_overrides={
                    "stdp_type": -1
                },
            )
        else:
            # STDP synapse
            new_syn = net.model.create_synapse(
                breed="single_exp_synapse",
                pre_soma_id=new_pre,
                post_soma_id=new_post,
                config_name="exp_pair_wise_stdp_config_0",
                hyperparameters_overrides={
                    "weight":         hyper[0],
                    "synaptic_delay": hyper[1],
                    "scale":          hyper[2],
                    "tau_fall":       hyper[3],
                    "tau_rise":       hyper[4],
                },
                default_internal_state_overrides={
                    "I_synapse": internal[0],
                },
                learning_hyperparameters_overrides={
                    "stdp_type":          lhyper[0],
                    "tau_pre_stdp":       lhyper[1],
                    "tau_post_stdp":      lhyper[2],
                    "a_exp_pre":          lhyper[3],
                    "a_exp_post":         lhyper[4],
                    "stdp_history_length": int(lhyper[5]),
                },
                default_internal_learning_state_overrides={
                    "pre_trace":  lint[0],
                    "post_trace": lint[1],
                    "dW":         lint[2],
                },
            )
        
        synapse_id_map[old_id] = new_syn
    
    # --- Reconstruct Topology ---
    print("Reconstructing topology...")
    topology = structure["topology"]
    
    # Initialize attributes that may not exist in __init__
    net.Downsample = {}
    net.Downsample_Dim = {}
    
    # Output_Channel_Dim
    net.Output_Channel_Dim = {
        int(k): v for k, v in topology["Output_Channel_Dim"].items()
    }
    
    # Downsample_Dim
    net.Downsample_Dim = {
        int(k): v for k, v in topology["Downsample_Dim"].items()
    }
    
    # ConvLayers
    for layer_idx_str, kernels in topology["ConvLayers"].items():
        layer_idx = int(layer_idx_str)
        net.ConvLayers[layer_idx] = []
        
        for kernel_data in kernels:
            # Map old synapse/neuron IDs to new IDs
            old_synapses = np.array(kernel_data["synapses"], dtype=object)
            old_neurons = np.array(kernel_data["neurons"], dtype=object)
            
            new_synapses = np.empty(old_synapses.shape, dtype=object)
            new_neurons = np.empty(old_neurons.shape, dtype=object)
            
            # Iterate through all dimensions
            it = np.nditer(old_synapses, flags=['multi_index', 'refs_ok'])
            while not it.finished:
                idx = it.multi_index
                old_syn = int(old_synapses[idx])
                old_neu = int(old_neurons[idx])
                new_synapses[idx] = synapse_id_map[old_syn]
                new_neurons[idx] = soma_id_map[old_neu]
                it.iternext()
            
            net.ConvLayers[layer_idx].append([
                new_synapses,
                new_neurons,
                kernel_data["stride"],
                kernel_data["num_input_channels"]
            ])
    
    # Output_Channel
    for layer_idx_str, channel_data in topology["Output_Channel"].items():
        layer_idx = int(layer_idx_str)
        
        old_somas = np.array(channel_data["somas"], dtype=object)
        old_synapses = np.array(channel_data["synapses"], dtype=object)
        
        new_somas = np.empty(old_somas.shape, dtype=object)
        new_synapses = np.empty(old_synapses.shape, dtype=object)
        
        it = np.nditer(old_somas, flags=['multi_index', 'refs_ok'])
        while not it.finished:
            idx = it.multi_index
            old_soma = int(old_somas[idx])
            old_syn = int(old_synapses[idx])
            new_somas[idx] = soma_id_map[old_soma]
            new_synapses[idx] = synapse_id_map[old_syn]
            it.iternext()
        
        net.Output_Channel[layer_idx] = [new_somas, new_synapses]
    
    # Downsample
    for layer_idx_str, ds_data in topology["Downsample"].items():
        layer_idx = int(layer_idx_str)
        
        old_somas = np.array(ds_data["somas"], dtype=object)
        old_synapses = np.array(ds_data["input_synapses"], dtype=object)
        
        new_somas = np.empty(old_somas.shape, dtype=object)
        new_synapses = np.empty(old_synapses.shape, dtype=object)
        
        it = np.nditer(old_somas, flags=['multi_index', 'refs_ok'])
        while not it.finished:
            idx = it.multi_index
            old_soma = int(old_somas[idx])
            old_syn = int(old_synapses[idx])
            new_somas[idx] = soma_id_map[old_soma]
            new_synapses[idx] = synapse_id_map[old_syn]
            it.iternext()
        
        # Connections: {(f, px, py): [syn_ids]}
        new_connections = {}
        for key_str, syn_list in ds_data["connections"].items():
            # Parse key "(f, px, py)" from string
            key_tuple = eval(key_str)
            new_connections[key_tuple] = [synapse_id_map[s] for s in syn_list]
        
        net.Downsample[layer_idx] = {
            'somas': new_somas,
            'input_synapses': new_synapses,
            'connections': new_connections,
            'dims': tuple(ds_data["dims"])
        }
    
    # FF (Feedforward layer)
    ff_data = topology["FF"]
    
    input_synapses = [synapse_id_map[s] for s in ff_data["input_synapses"]]
    output_somas = [soma_id_map[s] for s in ff_data["output_somas"]]
    
    hidden_to_output_dict = {}
    for soma_str, syn_list in ff_data["hidden_to_output_dict"].items():
        old_soma = int(soma_str)
        new_soma = soma_id_map[old_soma]
        hidden_to_output_dict[new_soma] = [synapse_id_map[s] for s in syn_list]
    
    net.FF = [input_synapses, output_somas, hidden_to_output_dict]
    
    print(f"Network loaded successfully!")
    print(f"  Somas: {len(net.model._soma_ids)}")
    print(f"  Synapses: {len(net.model._synapse_ids)}")
    print(f"  Conv layers: {len(net.ConvLayers)}")
    print(f"  Output classes: {len(output_somas)}")
    
    return net, soma_id_map, synapse_id_map


# =============================================================================
# Main: Example usage
# =============================================================================

if __name__ == "__main__":
    import sys
    import os
    
    # Default path or command line argument
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    else:
        json_path = "./network_run_2.json"
    
    print(f"Loading network from: {json_path}")
    
    # Load the network
    Model, soma_map, syn_map = load_network_from_json(json_path, use_gpu=True)
    
    print("\n" + "="*50)
    print("Network loaded and ready for inference!")
    print("="*50)
    

    