import os
import json
import numpy as np
from superneuroabm.model import NeuromorphicModel
import sagesim
import time as clocktime
import tonic 
import math
import pandas as pd
import itertools
from itertools import combinations
import tonic.transforms as transforms
from tonic.datasets import NMNIST
import matplotlib.pyplot as plt
import tempfile
from collections import defaultdict

class Conv2dtNet:
    def __init__(self, SpikeData=None):

        self.model = NeuromorphicModel(enable_internal_state_tracking=False)
        self.SpikeData = SpikeData
        self.ConvLayers = {}           
        self.Output_Channel = {}
        self.Output_Channel_Dim={}
        self.FF = []

    def load_bin_as_spike_dict(self, bin_path):
        # Read entire binary file as unsigned bytes
        with open(bin_path, "rb") as f:
            raw = np.frombuffer(f.read(), dtype=np.uint8)

        # Each event = 5 bytes
        if len(raw) % 5 != 0:
            raise ValueError(f"Unexpected file length ({len(raw)} bytes, not multiple of 5)")

        events = raw.reshape(-1, 5)

        # Combine 5 bytes into a 40-bit integer (big-endian)
        data = (
            (events[:, 0].astype(np.uint64) << 32)
            | (events[:, 1].astype(np.uint64) << 24)
            | (events[:, 2].astype(np.uint64) << 16)
            | (events[:, 3].astype(np.uint64) << 8)
            | (events[:, 4].astype(np.uint64))
        )

        # Bit-mask and shift to extract fields
        x = (data >> 32) & 0xFF
        y = (data >> 24) & 0xFF
        p = (data >> 23) & 0x1
        t = data & 0x7FFFFF  # 23 bits

        # Convert timestamp from µs to ms (integer)
        t_ms = (t / 1000.0).astype(np.int32)

        # Filter ON events only
        time_mask = t_ms <= 100
        x, y, t_ms = x[time_mask], y[time_mask], t_ms[time_mask]


        # Group by timestamp into a dict-of-dicts
        spike_dict = {}
        for xi, yi, ti in zip(x, y, t_ms):
            spike_dict.setdefault(int(ti), {})[(int(xi), int(yi))] = 10

        return spike_dict


    '''
lif_soma_adaptive_thr:
    config_0:
      hyperparameters:
        C: 10e-9  # Membrane capacitance in Farads (10 nF)
        R: 1e6  # Membrane resistance in Ohms (1 TΩ)
        vthr_initial: -45  # Spike threshold voltage (mV)
        tref: 5e-3  # Refractory period (5 ms)
        vrest: -60  # Resting potential (mV)
        vreset: -60  # Reset potential after spike (mV)
        tref_allows_integration: 1  # Whether to allow integration during refractory period
        I_in: 0  # Input current (40 nA)
        scaling_factor: 1e-5  # Scaling factor for synaptic current
        delta_thr: 1.0  # Threshold increase after spike (mV)
        tau_decay_thr: 30e-3  # Time constant for threshold decay (100 ms)
      internal_state:
        v: -60.0  # Initial membrane voltage
        tcount: 0.0  # Time counter
        tlast: 0.0  # Last spike time
        vthr: -45.0  # Initial spike threshold voltage

    '''

    def CreateSoma(self, delta_thr=0):
            Soma= self.model.create_soma(
                breed="lif_soma_adaptive_thr",
                config_name="config_0",
                hyperparameters_overrides = {
                    'C':      np.float64(np.random.uniform(5e-9, 15e-9)),
                    'R':      np.float64(np.random.uniform(0.5e6, 2e6)),
                    'vthr_initial': -45,
                    'tref':   np.float64(5e-3),
                    'vrest':  -60,
                    'vreset': -65,
                    'tref_allows_integration':1,
                    'I_in': 0,
                    'scaling_factor':1e-6,
                    'delta_thr': delta_thr,
                    'tau_decay_thr': 30e-3

                },
                default_internal_state_overrides={
                    'v':-60,
                    'tcount':0.0,
                    'tlast':0.0,
                    'vthr':-45.0
                }
            )
            return Soma
        
    def CreateSynapseNoSTDP(self,pre_soma, post_soma, weight):
        Synapse = self.model.create_synapse(
            breed="single_exp_synapse",
            pre_soma_id=pre_soma,
            post_soma_id=post_soma,
            config_name="no_learning_config_0",
            hyperparameters_overrides={
                "weight": weight,
                "synaptic_delay": 1.0,
                "scale": 1.0,
                "tau_fall": 1e-2,
                "tau_rise": 0,
            },
            default_internal_state_overrides={
                "I_synapse": 0.0,
            },
            learning_hyperparameters_overrides={
                "stdp_type": -1
            },
            tags=None,
        )
        return Synapse

    def CreateSynapseSTDP(self,pre_soma, post_soma):
        Synapse = self.model.create_synapse(
            breed="single_exp_synapse",
            pre_soma_id=pre_soma,
            post_soma_id=post_soma,
            config_name="exp_pair_wise_stdp_config_0",
            hyperparameters_overrides={
                "weight": np.random.uniform(0, 10),
                "synaptic_delay": 1.0,
                "scale": 1.0,
                "tau_fall": 1e-2,
                "tau_rise": 0,
            },
            default_internal_state_overrides={
                "I_synapse": 0.0,
            },
            learning_hyperparameters_overrides={
                "stdp_type": 0.0,
                "tau_pre_stdp": 20e-3,
                "tau_post_stdp": 20e-3,
                "a_exp_pre": 0.005,
                "a_exp_post": 0.005,
                "stdp_history_length": 100,
            },
            default_internal_learning_state_overrides={
                "pre_trace": 0,
                "post_trace": 0,
                "dW": 0,
            },
            tags=None,
        )
        return Synapse

    def Conv_Kernel_Construction(self, W, H, Stride, layer_idx=0, num_input_channels=1):

        Kernel = np.empty((num_input_channels, W, H), dtype=object)
        KernelSynapses = np.empty((num_input_channels, W, H), dtype=object)
        
        for c in range(num_input_channels):
            for i in range(W):
                for j in range(H):
                    Kernel_Neuron = self.CreateSoma(delta_thr=1)
                    Kernel[c, i, j] = Kernel_Neuron
                    
        for c in range(num_input_channels):
            for i in range(W):
                for j in range(H):
                    KernelSynapse = self.CreateSynapseSTDP(-1, Kernel[c, i, j])
                    KernelSynapses[c, i, j] = KernelSynapse
    
        if layer_idx not in self.ConvLayers:
            self.ConvLayers[layer_idx] = []
        
        # Store: [Synapses, Neurons, Stride, num_input_channels]
        self.ConvLayers[layer_idx].append([KernelSynapses, Kernel, Stride, num_input_channels])
    
    def Lateral_Inhibition(self, Layer_Tensor, p=0.8, inhibition_strength=-15):
        # Flatten arbitrary tensor of somas to 1D list
        flat = np.array(Layer_Tensor).flatten()

        N = len(flat)

        for i in range(N):
            # Choose random set of targets
            possible_targets = [j for j in range(N) if j != i]
            num_targets = max(1, int(p * len(possible_targets)))

            chosen = np.random.choice(possible_targets, size=num_targets, replace=False)

            # Connect inhibitory synapses
            for j in chosen:
                self.CreateSynapseNoSTDP(
                    flat[i],
                    flat[j],
                    inhibition_strength
                )

        
    def Output_Channel_Construction(self, Layer_idx, N, Input_W, Input_H):
        kernels = self.ConvLayers[Layer_idx]
        K = len(kernels)   # number of kernels / output channels
        
        Output_Channel_Somas = np.empty((K, N, N), dtype=object)
        Output_Channel_Synapses = np.empty((K, N, N), dtype=object)

        for k in range(K):
            for x in range(N):
                for y in range(N):
                    soma = self.CreateSoma()
                    Output_Channel_Somas[k, x, y] = soma
                    
                    # Initialize input synapse for this output neuron
                    synapse = self.CreateSynapseNoSTDP(-1, soma, 10)
                    Output_Channel_Synapses[k, x, y] = synapse

        # Store both somas and synapses for this layer
        self.Output_Channel[Layer_idx] = [Output_Channel_Somas, Output_Channel_Synapses]
        #self.Lateral_Inhibition(Output_Channel_Somas, p=0.2, inhibition_strength=-1)


    def Downsample_Construction(self, Layer_idx, pool_size=2):

        Output_Channel_Somas = self.Output_Channel[Layer_idx][0]
        K, N, _ = Output_Channel_Somas.shape

        # Downsampled spatial size
        M = N // pool_size

        Downsample_Somas = np.empty((K, M, M), dtype=object)
        Downsample_Synapses = np.empty((K, M, M), dtype=object)
        Downsample_Connections = {}

        for f in range(K):               # feature channels
            for px in range(M):          # pooled x
                for py in range(M):      # pooled y

                    # Create pooled neuron
                    pooled = self.CreateSoma()
                    Downsample_Somas[f, px, py] = pooled

                    # Optional external input
                    inp = self.CreateSynapseNoSTDP(-1, pooled, 10)
                    Downsample_Synapses[f, px, py] = inp

                    connections = []

                    # Determine pooling window coordinates
                    sx = px * pool_size
                    sy = py * pool_size
                    ex = sx + pool_size
                    ey = sy + pool_size

                    # FULLY connect all neurons in this block
                    for ox in range(sx, ex):
                        for oy in range(sy, ey):
                            pre = Output_Channel_Somas[f, ox, oy]
                            syn = self.CreateSynapseNoSTDP(pre, pooled, 10)
                            connections.append(syn)

                    # store all synapses for this pooled neuron
                    Downsample_Connections[(f, px, py)] = connections

        # Optional lateral inhibition within the downsampled layer
        self.Lateral_Inhibition(
            Downsample_Somas,
            p=0.5,
            inhibition_strength=-10
        )

        # Store result
        if not hasattr(self, 'Downsample'):
            self.Downsample = {}

        self.Downsample[Layer_idx] = {
            'somas': Downsample_Somas,
            'input_synapses': Downsample_Synapses,
            'connections': Downsample_Connections,
            'dims': (K, M, M)
        }

        self.Downsample_Dim = getattr(self, 'Downsample_Dim', {})
        self.Downsample_Dim[Layer_idx] = [M, M]

        return Downsample_Somas, Downsample_Synapses



    def FeedForwardLayerNN(self, InputSize, HiddenSize, OutputSize):
        InputLayer=np.array([self.CreateSoma(delta_thr=1) for _ in range(InputSize)])
        HiddenLayer=np.array([self.CreateSoma(delta_thr=1) for _ in range(HiddenSize)])
        OutputLayer=np.array([self.CreateSoma(delta_thr=1) for _ in range(OutputSize)])
        #initialize in input synapses
        input_synapses=[]
        for post_soma in InputLayer:
            synapse=self.CreateSynapseSTDP(-1, post_soma)
            input_synapses.append(synapse)
        InputToHidden,InputToHidden_Dict=self.FullyConnected(InputLayer,HiddenLayer)
        HiddenToOutput, HiddenToOutput_Dict=self.FullyConnected(HiddenLayer,OutputLayer)
        self.Lateral_Inhibition(HiddenLayer, p=.8, inhibition_strength=-10)
        self.Lateral_Inhibition(OutputLayer, p=.8, inhibition_strength=-10)

        return [input_synapses,OutputLayer, HiddenToOutput_Dict]

    def FullyConnected(self, InputLayer, OutputLayer):
        synapses = []
        synapse_dict = {}  # {post_soma: [synapses_from_all_inputs]}

        for post_soma in OutputLayer:
            synapse_dict[post_soma] = []  # initialize list for this post neuron

        for pre_soma in InputLayer:
            for post_soma in OutputLayer:
                synapse = self.CreateSynapseSTDP(pre_soma, post_soma)
                #synapse=self.CreateSynapseNoSTDP(pre_soma,post_soma,10)
                synapses.append(synapse)
                synapse_dict[post_soma].append(synapse)

        return np.array(synapses, dtype=object), synapse_dict

    def NetworkConstruction(self, Conv_Kernel_List, output_classes, Input_W, Input_H):
        num_input_channels = 1  # First layer: single channel input
        
        for layer_id in range(len(Conv_Kernel_List)):
            Max_Output_H = 0
            Max_Output_W = 0

            for kernel_specs in Conv_Kernel_List[layer_id]:
                K_H = kernel_specs[0]
                K_W = kernel_specs[1]
                stride = kernel_specs[2]
                
                # Create 3D kernel with correct depth
                self.Conv_Kernel_Construction(K_H, K_W, stride, 
                                            layer_idx=layer_id,
                                            num_input_channels=num_input_channels)

                Output_H = (Input_H - K_H) // stride + 1
                Output_W = (Input_W - K_W) // stride + 1

                Max_Output_H = max(Max_Output_H, Output_H)
                Max_Output_W = max(Max_Output_W, Output_W)

            self.Output_Channel_Construction(layer_id, Max_Output_H, Max_Output_W, Max_Output_H)
            self.Downsample_Construction(layer_id, pool_size=2)
            self.Output_Channel_Dim[layer_id] = [Max_Output_W, Max_Output_H]

            # Update for next layer
            Input_H = self.Downsample_Dim[layer_id][1]
            Input_W = self.Downsample_Dim[layer_id][0]
            
            # CRITICAL: Next layer's input channels = this layer's number of kernels
            num_input_channels = len(Conv_Kernel_List[layer_id])
        
        # Build FF readout
        last_layer = max(self.Downsample_Dim.keys())
        W_last, H_last = self.Downsample_Dim[last_layer]
        K_last = self.Downsample[last_layer]['dims'][0]

        FF_input_size = W_last * H_last * K_last

        self.FF = self.FeedForwardLayerNN(
            FF_input_size,
            2 * FF_input_size,
            output_classes
        )



    def invert_dict(self, input_dict, W, H) :
        inverted = defaultdict(dict)
        
        for key, values in input_dict.items():
            for v in values:
                inverted[v][key] = 10
                
        return dict(inverted)

    
    def compute_kernel_offsets_2d(self, H, W):
        offsets = []
        for i in range(H):
            for j in range(W):
                offsets.append((i, j))
        return offsets

    def Convolve_Spike_3D(self, Conv_Kernel_Dict, SynapseDict, spike, Kernel, offsets, CurrentSpikeSet,
                      time_step, Input_W, Input_H, Stride=1, kernel_idx=0):

        sx, sy, input_channel = spike  # Unpack 3D spike coordinates
        
        num_channels, H, W = Kernel.shape

        for ki0, kj0 in offsets:
            # Compute top-left of kernel window
            row0 = sy - ki0
            col0 = sx - kj0

            # Skip illegal kernel placements
            if row0 < 0 or col0 < 0:
                continue
            if (row0 + H) > Input_H or (col0 + W) > Input_W:
                continue
            if (row0 % Stride != 0) or (col0 % Stride != 0):
                continue

            # Output spatial position (no channel dimension - that's determined by kernel_idx)
            out_x = col0 // Stride
            out_y = row0 // Stride

            # Sweep over spatial kernel window for THIS input channel
            for ki in range(H):
                for kj in range(W):
                    row = row0 + ki
                    col = col0 + kj

                    # Check for spike at (col, row, input_channel)
                    spike_key = (col, row, input_channel)
                    if spike_key in CurrentSpikeSet:
                        # Use the kernel neuron for THIS input channel
                        syn = Kernel[input_channel, ki, kj]
                        val = CurrentSpikeSet[spike_key]

                        SynapseDict.setdefault(syn, []).append([time_step, val])

                        if syn not in Conv_Kernel_Dict:
                            Conv_Kernel_Dict[syn] = {}
                        if time_step not in Conv_Kernel_Dict[syn]:
                            Conv_Kernel_Dict[syn][time_step] = []
                        
                        # Output: (spatial_x, spatial_y, which_kernel_produced_this)
                        entry = (out_x, out_y, kernel_idx)
                        if entry not in Conv_Kernel_Dict[syn][time_step]:
                            Conv_Kernel_Dict[syn][time_step].append(entry)

    def _convert_to_3d_spike_set(self, spike_dict):

        converted = {}
        for key, val in spike_dict.items():
            if len(key) == 2:
                converted[(key[0], key[1], 0)] = val
            else:
                converted[key] = val
        return converted


    def Full_Convolution_And_Extraction(self, Layer_idx, SpikeData, Total_Sim_Time):
        print(f"Layer {Layer_idx} Output Dims: {self.Output_Channel_Dim[Layer_idx]}")
        start = clocktime.time()
        Synpase_Dict = {}
        Conv_Kernel_Dict = {}
        syn_to_soma = {}
        
        is_first_layer = (Layer_idx == 0)
        
        print('Adding Spikes per kernel')
        for kernel_array_index in range(len(self.ConvLayers[Layer_idx])):
            Kernel_Synapses = self.ConvLayers[Layer_idx][kernel_array_index][0]
            Kernel_Neurons = self.ConvLayers[Layer_idx][kernel_array_index][1]
            Stride = self.ConvLayers[Layer_idx][kernel_array_index][2]
            num_input_channels = self.ConvLayers[Layer_idx][kernel_array_index][3]
            
            # Build synapse -> soma mapping (now 3D)
            C, H, W = Kernel_Synapses.shape
            for c in range(C):
                for i in range(H):
                    for j in range(W):
                        syn_to_soma[Kernel_Synapses[c, i, j]] = Kernel_Neurons[c, i, j]
            
            offsets = self.compute_kernel_offsets_2d(H, W)
            
            for time_step in SpikeData:
                current_spikes = SpikeData[time_step]
                
                # Prepare spike set for lookup
                if is_first_layer:
                    # Convert 2D spike set to 3D for consistent lookup
                    spike_set_3d = self._convert_to_3d_spike_set(current_spikes)
                else:
                    spike_set_3d = current_spikes  # Already 3D
                
                for spike in current_spikes:
                    if is_first_layer:
                        input_w = 28
                        input_h = 28
                        # Convert 2D spike to 3D
                        if len(spike) == 2:
                            spike_3d = (spike[0], spike[1], 0)
                        else:
                            spike_3d = spike
                    else:
                        input_w = self.Downsample_Dim[Layer_idx - 1][0]
                        input_h = self.Downsample_Dim[Layer_idx - 1][1]
                        spike_3d = spike
                    
                    self.Convolve_Spike_3D(
                        Conv_Kernel_Dict,
                        Synpase_Dict,
                        spike_3d,
                        Kernel_Synapses,
                        offsets,
                        spike_set_3d,
                        time_step,
                        input_w,
                        input_h,
                        Stride=Stride,
                        kernel_idx=kernel_array_index
                    )

        print(f'done adding spikes: {len(Synpase_Dict)} synapses received input')
        for key in Synpase_Dict:
            self.model.add_spike_list(key, Synpase_Dict[key])
        
        print('begin kernel sim')
        self.model.simulate(ticks=Total_Sim_Time, update_data_ticks=Total_Sim_Time)
        print('end kernel sim')
        
        # Build output spike dict
        output_spike_dict = defaultdict(dict)
        
        for syn_id, time_to_outputs in Conv_Kernel_Dict.items():
            soma_id = syn_to_soma[syn_id]
            spike_times = list(self.model.get_spike_times(soma_id=soma_id))
            spike_times.sort()

            input_times = sorted(time_to_outputs.keys())
            prev_spike_t = -1
            
            for spike_t in spike_times:
                valid_times = [t for t in input_times if prev_spike_t < t < spike_t]
                
                if valid_times:
                    for t in valid_times:
                        output_entries = time_to_outputs[t]
                        for (out_x, out_y, kernel_idx) in output_entries:
                            output_spike_dict[int(spike_t)][(out_x, out_y, kernel_idx)] = 10
                
                prev_spike_t = spike_t
        
        print(f'output_spike_dict has {len(output_spike_dict)} timesteps')
        
        self.model.reset()
        
        # Add spikes to output channel synapses
        Output_Channel_Synapses = self.Output_Channel[Layer_idx][1]
        
        spikes_added = 0
        for spike_time, spike_positions in output_spike_dict.items():
            for (out_x, out_y, kernel_idx), value in spike_positions.items():
                output_syn = Output_Channel_Synapses[kernel_idx, out_x, out_y]
                self.model.add_spike(synapse_id=output_syn, tick=spike_time, value=value)
                spikes_added += 1
        
        print(f'Added {spikes_added} spikes to output channel synapses')
        print('Running output channel + downsample simulation')
        self.model.simulate(ticks=Total_Sim_Time, update_data_ticks=Total_Sim_Time)
        
        # Collect spikes from downsampled layer WITH channel info
        Downsample_Somas = self.Downsample[Layer_idx]['somas']
        K, M, _ = Downsample_Somas.shape
        
        downsample_spike_dict = defaultdict(dict)
        total_spikes = 0
        
        for k in range(K):
            for x in range(M):
                for y in range(M):
                    soma_id = Downsample_Somas[k, x, y]
                    spike_times = self.model.get_spike_times(soma_id=soma_id)
                    for t in spike_times:
                        # KEY: Include channel k for next layer
                        downsample_spike_dict[int(t)][(x, y, k)] = 10
                        total_spikes += 1
        
        end = clocktime.time()
        print(f"Layer {Layer_idx} time: {end-start:.2f}s")
        print(f'downsample_spike_dict: {total_spikes} total spikes')
        return dict(downsample_spike_dict)


    def ForwardPass(self, SpikeData,Total_Sim_Time):
        print('Start Forward Pass')
        start=clocktime.time()
        Dataset = self.load_bin_as_spike_dict(SpikeData)
        #print(Dataset)
        # Fix conv kernel list so that is uses self.layers instead as not passed in anymore
        for layer_id in range(len(self.ConvLayers)):
            print(layer_id)
            Dataset = self.Full_Convolution_And_Extraction(layer_id, Dataset, Total_Sim_Time)
            #print(Dataset)
            # Reset after each layer EXCEPT the last
            self.model.reset()



        print("Feed Forward Part")

        input_synapses = self.FF[0]

        # determine expected dims of final conv output
        last_layer = max(self.Downsample_Dim.keys())
        M, H = self.Downsample_Dim[last_layer]  # always square: M x M
        K_last = self.Downsample[last_layer]['dims'][0]

        expected_inputs = K_last * M * M

        if len(input_synapses) != expected_inputs:
            print(
                f"[WARN] FF input size mismatch: "
                f"expected {expected_inputs}, got {len(input_synapses)}"
            )

        # inject spikes
        for t, pos_dict in Dataset.items():
            for (x, y, k), value in pos_dict.items():

                flat_idx = k * (M * M) + y * M + x

                if flat_idx >= len(input_synapses):
                    continue

                syn_id = input_synapses[flat_idx]

                self.model.add_spike(
                    synapse_id=syn_id,
                    tick=int(t),
                    value=value
                )

        # run the feedforward network
        Spike_Times = []
        self.model.simulate(ticks=Total_Sim_Time, update_data_ticks=Total_Sim_Time)

        for output_neuron in self.FF[1]:
            values = self.model.get_spike_times(soma_id=output_neuron)
            Spike_Times.append((len(values), values))

        # pick winner
        max_index = max(range(len(Spike_Times)), key=lambda i: Spike_Times[i][0])
        end=clocktime.time()
        print("\n================",end-start,"============= TIME")
        print("\n=== Output Layer Spike Times ===")
        for idx, (count, times) in enumerate(Spike_Times):
            print(f"Output neuron {idx}: {count} spikes → {list(times)}")
        print("================================\n")

        return max_index


    def plot_all_output_neurons_single(self, save_path):
        matrices = self.get_all_ff_weight_matrices()

        num_neurons = len(matrices)
        cols = 6  # increased for 30 neurons
        rows = math.ceil(num_neurons / cols)

        # Create figure
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

        # In case rows*cols > num_neurons, flatten indices
        axes = axes.flatten()

        # Fixed color scale for all plots
        vmin, vmax = -10, 30

        for idx, (neuron_idx, W) in enumerate(matrices.items()):
            ax = axes[idx]
            im = ax.imshow(W, cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_title(f"Neuron {neuron_idx}")
            ax.set_xticks([])
            ax.set_yticks([])

        # Hide any unused subplots
        for k in range(len(matrices), len(axes)):
            axes[k].axis('off')

        # Add one shared colorbar
        cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.6)
        cbar.set_label("Synaptic Weight", rotation=270, labelpad=15)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"Saved combined FF output neuron plot: {save_path}")

    def get_all_ff_weight_matrices(self):

        syn_dict = self.FF[2]  # {output_soma: [syn_ids]}
        out_somas = list(syn_dict.keys())

        # Use final *downsampled* dims, because FF connects to that
        last_layer = max(self.Downsample_Dim.keys())
        W, H = self.Downsample_Dim[last_layer]              # e.g., 7 x 7
        K_last = self.Downsample[last_layer]['dims'][0]     # number of channels (kernels)

        # We have HiddenSize = 2 * (K_last * H * W)
        # Visualize as (2 * K_last * H, W) so total elements match:
        # (2 * K_last * H) * W = 2 * K_last * H * W
        mat_rows = 2 * K_last * H
        expected_len = mat_rows * W

        matrices = {}

        for idx, soma_id in enumerate(out_somas):
            syn_list = syn_dict[soma_id]

            if len(syn_list) != expected_len:
                print(
                    f"[WARN] For output neuron {idx}, expected {expected_len} FF "
                    f"weights (2 * {K_last} * {H} * {W}), got {len(syn_list)}."
                )

            weights = []
            for syn in syn_list:
                hyper = self.model.get_agent_property_value(
                    id=syn, property_name="hyperparameters"
                )
                weights.append(float(hyper[0]))  # weight is hyperparameters[0]

            # Truncate or pad if needed, to be safe
            if len(weights) > expected_len:
                weights = weights[:expected_len]
            elif len(weights) < expected_len:
                weights = weights + [0.0] * (expected_len - len(weights))

            matrices[idx] = np.array(weights).reshape(mat_rows, W)

        return matrices

    def extract_ff_weights_df(self):

        syn_dict = self.FF[2]  # {post_soma: [syn_ids]}
        rows = []

        # Use final downsample dims
        last_layer = max(self.Downsample_Dim.keys())
        W, H = self.Downsample_Dim[last_layer]
        K_last = self.Downsample[last_layer]['dims'][0]

        # Matrix shape used in get_all_ff_weight_matrices
        mat_rows = 2 * K_last * H
        expected_len = mat_rows * W

        for post_idx, (post_soma, syn_list) in enumerate(syn_dict.items()):

            # Optional consistency check
            if len(syn_list) != expected_len:
                print(
                    f"[WARN] extract_ff_weights_df: for output neuron {post_idx}, "
                    f"expected {expected_len} synapses, got {len(syn_list)}."
                )

            # Iterate over synapses in the same flat order as reshape
            for flat_i, syn in enumerate(syn_list):
                # Map flat index → (row, col) in the (mat_rows x W) matrix
                row = flat_i // W
                col = flat_i % W

                hyper = self.model.get_agent_property_value(
                    id=syn, property_name="hyperparameters"
                )
                weight = float(hyper[0])

                rows.append({
                    "syn_id": int(syn),
                    "post_idx": post_idx,
                    "weight": weight,
                    "row": row,
                    "col": col,
                })

        return pd.DataFrame(rows)

    
    def compare_stdp(self, baseline_df, current_df):
        merged = baseline_df.merge(current_df, on="syn_id", suffixes=("_base", "_curr"))
        merged["delta"] = merged["weight_curr"] - merged["weight_base"]

        changed = merged[merged["delta"] != 0]
        positive_changes = changed[changed["delta"] > 0]["delta"]
        negative_changes = changed[changed["delta"] < 0]["delta"]
        
        total = len(changed)
        positive = len(positive_changes)
        negative = len(negative_changes)

        print("\n=== STDP Change Summary ===")
        print(f"Total synapses that changed: {total}")
        print(f"  Positive ΔW: {positive}")
        print(f"  Negative ΔW: {negative}")
        
        if positive > 0:
            print(f"    Avg positive ΔW:    {positive_changes.mean():.6f}")
            print(f"    Median positive ΔW: {positive_changes.median():.6f}")
        else:
            print(f"    Avg positive ΔW:    N/A")
            print(f"    Median positive ΔW: N/A")
            
        if negative > 0:
            print(f"    Avg negative ΔW:    {negative_changes.mean():.6f}")
            print(f"    Median negative ΔW: {negative_changes.median():.6f}")
        else:
            print(f"    Avg negative ΔW:    N/A")
            print(f"    Median negative ΔW: N/A")

        return {
            "total_changed": total,
            "positive": positive,
            "negative": negative,
            "avg_positive": positive_changes.mean() if positive > 0 else None,
            "median_positive": positive_changes.median() if positive > 0 else None,
            "avg_negative": negative_changes.mean() if negative > 0 else None,
            "median_negative": negative_changes.median() if negative > 0 else None,
        }




    def extract_full_network_with_topology(self, save_path):
        """Save network with full topology for reconstruction."""
        
        # Helper to convert numpy values → JSON-safe Python types
        def to_py(obj):
            if isinstance(obj, np.ndarray):
                return [to_py(x) for x in obj]
            if isinstance(obj, (np.float32, np.float64, float)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64, int)):
                return int(obj)
            if isinstance(obj, set):
                return [to_py(x) for x in obj]
            if isinstance(obj, (list, tuple)):
                return [to_py(x) for x in obj]
            return obj

        # Handle empty directory path
        dir_path = os.path.dirname(save_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        structure = {}

        # ---- Somas ----
        soma_data = []
        for soma in self.model._soma_ids:
            hyper = self.model.get_agent_property_value(id=soma, property_name="hyperparameters")
            internal = self.model.get_agent_property_value(id=soma, property_name="internal_state")

            soma_data.append({
                "soma_id": int(soma),
                "hyperparameters": to_py(hyper),
                "internal_state": to_py(internal),
                "incoming_synapses": to_py(self.model.soma2synapse_map[soma]["pre"]),
                "outgoing_synapses": to_py(self.model.soma2synapse_map[soma]["post"])
            })
        structure["somas"] = soma_data

        # ---- Synapses ----
        syn_data = []
        for syn in self.model._synapse_ids:
            conn = self.model.get_synapse_connectivity(syn)
            pre, post = int(conn[0]), int(conn[1])

            hyper = self.model.get_agent_property_value(id=syn, property_name="hyperparameters")
            lhyper = self.model.get_agent_property_value(id=syn, property_name="learning_hyperparameters")
            internal = self.model.get_agent_property_value(id=syn, property_name="internal_state")
            lint = self.model.get_agent_property_value(id=syn, property_name="internal_learning_state")

            syn_data.append({
                "synapse_id": int(syn),
                "pre": pre,
                "post": post,
                "hyperparameters": to_py(hyper),
                "learning_hyperparameters": to_py(lhyper),
                "internal_state": to_py(internal),
                "internal_learning_state": to_py(lint)
            })
        structure["synapses"] = syn_data

        # ---- Topology ----
        topology = {
            "ConvLayers": {},
            "Output_Channel": {},
            "Output_Channel_Dim": {str(k): v for k, v in self.Output_Channel_Dim.items()},
            "Downsample": {},
            "Downsample_Dim": {str(k): v for k, v in self.Downsample_Dim.items()},
            "FF": {
                "input_synapses": [int(s) for s in self.FF[0]],
                "output_somas": [int(s) for s in self.FF[1]],
                "hidden_to_output_dict": {
                    str(k): [int(s) for s in v] 
                    for k, v in self.FF[2].items()
                }
            }
        }

        # ConvLayers
        for layer_idx, kernels in self.ConvLayers.items():
            topology["ConvLayers"][str(layer_idx)] = []
            for kernel in kernels:
                synapses, neurons, stride, num_channels = kernel
                topology["ConvLayers"][str(layer_idx)].append({
                    "synapses": to_py(synapses),
                    "neurons": to_py(neurons),
                    "stride": stride,
                    "num_input_channels": num_channels
                })

        # Output_Channel
        for layer_idx, (somas, synapses) in self.Output_Channel.items():
            topology["Output_Channel"][str(layer_idx)] = {
                "somas": to_py(somas),
                "synapses": to_py(synapses)
            }

        # Downsample
        for layer_idx, ds in self.Downsample.items():
            topology["Downsample"][str(layer_idx)] = {
                "somas": to_py(ds["somas"]),
                "input_synapses": to_py(ds["input_synapses"]),
                "dims": list(ds["dims"]),
                "connections": {
                    str(k): [int(s) for s in v] 
                    for k, v in ds["connections"].items()
                }
            }

        structure["topology"] = topology

        with open(save_path, "w") as f:
            json.dump(structure, f, indent=2)

        return structure

if __name__ == "__main__":
    # -------------------------
    #  Initialize model
    # -------------------------
    Model = Conv2dtNet()
    Model.model.setup(use_gpu=True)
    print("GPU setup complete")

    # -------------------------
    #  Build convolutional network
    # -------------------------
    Conv_Kernel_List = [
    [(5, 5, 1)] * 1,     # Layer 1
    [(5, 5, 1)] * 1     # Layer 2
    ]

    Model.NetworkConstruction(
        Conv_Kernel_List=Conv_Kernel_List,
        output_classes=30,
        Input_W=28,
        Input_H=28,
    )

    print("Network constructed")
    print("Number of somas:", len(Model.model._soma_ids))
    print("Number of synapses:", len(Model.model._synapse_ids))
    print("Total agents:", len(Model.model._soma_ids) + len(Model.model._synapse_ids))

    root = "./data/NMNIST/Test"
    assert os.path.isdir(root), f"NMNIST Test directory not found: {root}"

    digit_dirs = sorted(os.listdir(root))

    # -------------------------
    #  Run only 2 examples
    # -------------------------
    results = {}
    NUM_RUNS = 7
    baseline_df = None

    for run_idx in range(NUM_RUNS):
        digit = digit_dirs[run_idx]
        digit_path = os.path.join(root, digit)

        bin_files = [f for f in os.listdir(digit_path) if f.endswith(".bin")]
        assert bin_files, f"No .bin files found in {digit_path}"

        bin_file = bin_files[0]
        dataset_path = os.path.join(digit_path, bin_file)

        print(f"\n=== RUN {run_idx} | Digit {digit} | File {bin_file} ===")

        pred = Model.ForwardPass(dataset_path, Total_Sim_Time=100)
        # Extract Hidden→Output weight matrix DF for this run
        current_df = Model.extract_ff_weights_df()

        if run_idx == 0:
            print("\nSaved baseline synaptic weights for future comparison.\n")
            baseline_df = current_df.copy()

        else:
            # Compare to baseline
            Model.compare_stdp(baseline_df, current_df)

        # -------------------------
        #  Save FF weight plot (single image)
        # -------------------------
        plot_path = f"./ff_plots_run_{run_idx}.png"
        Model.plot_all_output_neurons_single(plot_path)
        print(f"Saved FF weight plot (single image) for run {run_idx} → {plot_path}")

        # -------------------------
        #  Save full network structure
        # -------------------------
        json_path = f"./network_run_{run_idx}.json"
        Model.extract_full_network_with_topology(json_path)
        print(f"Saved full network JSON for run {run_idx} to {json_path}")

        # Reset SNN before next run
        Model.model.reset()

    # -------------------------
    #  Print summary
    # -------------------------
    print("\n=== FINAL SUMMARY ===")
    for run_id, info in results.items():
        print(f"{run_id}: Digit {info['digit']} → Predicted {info['predicted']}")
