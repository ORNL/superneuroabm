import os
import numpy as np
from superneuroabm.model import NeuromorphicModel
import sagesim
import time
import tonic 
import math
import itertools
from itertools import combinations
import tonic.transforms as transforms
from tonic.datasets import NMNIST
import matplotlib.pyplot as plt
import tempfile
from collections import defaultdict

class Conv2dtNet:
    def __init__(self, SpikeData=None):
        # Spike Data should be organized by spike time
        # SpikeData[t] = [(xi, yi, spike val), (xj, yj, spike val)....] all incoming spikes
        self.model = NeuromorphicModel()
        self.SpikeData = SpikeData
        self.layers = {}           # {layer_idx: [(Kernel, Kernel_Neuron), ...]}
        self.kernel_vals = {}
        self.pooling_matrix = {}
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
            value = (self.szudzik_pair(int(xi), int(yi)) // 10) + 1
            spike_dict.setdefault(int(ti), {})[(int(xi), int(yi))] = value

        return spike_dict


    def HashSpikes(self, TimeStep):
        CurrentSpikeSet = {}
        for i in self.SpikeData[TimeStep]:
            CurrentSpikeSet[(i[0], i[1])] = i[2]
        return CurrentSpikeSet
    
    def szudzik_pair(self, x, y):
        if x >= y:
            return x * x + x + y + 1
        else:
            return y * y + x + 1
    
    def Conv_Kernel_Construction(self, W, H, layer_idx=0, input=np.nan):
        InputSomas=np.empty((W, H), dtype=object)
        InputLayer = np.empty((W, H), dtype=object)
        for i in range(W):
            for j in range(H):
                InputSoma=self.CreateSoma()
                InputSomas[i][j] = InputSoma
        for i in range(W):
            for j in range(H):
                InputSynapse=self.CreateSynapseNoSTDP(InputSomas[i][j])
                InputLayer[i][j] = InputSynapse
        Kernel=np.empty((W, H), dtype=object)
        Kernel_Neuron = self.CreateSoma()
        for i in range(W):
            for j in range(H):
                Synapse=self.CreateSynapseSTDP(InputSomas[i][j], Kernel_Neuron)
                Kernel[i][j] = Synapse

        # store in proper layer
        if layer_idx not in self.layers:
            self.layers[layer_idx] = []
        if layer_idx not in self.kernel_vals:
            self.kernel_vals[layer_idx] = []
        self.kernel_vals[layer_idx].append((InputSomas, Kernel))
        self.layers[layer_idx].append((InputLayer, Kernel_Neuron))
    def compute_kernel_offsets(self, Kernel_List_Entry):
        H = len(Kernel_List_Entry)
        W = len(Kernel_List_Entry[0])

        # Generate kernel origin offsets
        # for each position spike could occupy in kernel
        offsets = []
        for i in range(H):
            for j in range(W):
                offsets.append((i, j))
        return offsets


    def Convolve_Spike(self, spike, Kernel, offsets, CurrentSpikeSet, time_step, Stride=1):
        sx, sy = spike
        H = len(Kernel)
        W = len(Kernel[0])

        for i, j in offsets:
            # Top-left of kernel window when spike is at kernel[i][j]
            x0 = sx - i
            y0 = sy - j

            # Apply stride to kernel window origin
            if x0 % Stride != 0 or y0 % Stride != 0:
                continue

            # Sweep kernel window
            for ki in range(H):
                for kj in range(W):
                    x = x0 + ki
                    y = y0 + kj

                    if (x, y) in CurrentSpikeSet:
                        self.model.add_spike(
                            synapse_id = Kernel[ki][kj],
                            tick = time_step,
                            value = CurrentSpikeSet[(x, y)]
                        )


    def AttentionPooling(self, layer_idx, W, H):
        output_channels=len(self.layers[layer_idx])
        output_channels_list=[kernel[1] for kernel in self.layers[layer_idx]]
        combos=[]
        for r in range(1, min(2, output_channels) + 1):   # only r=1 and r=2
            combos.extend(list(c) for c in itertools.combinations(output_channels_list, r))

        pooling_spikes = np.array([self.CreateSoma() for _ in range(len(combos))], dtype=object)
        total = W * H
        if len(pooling_spikes) < total:
            padding = [None] * (total - len(pooling_spikes))
            pooling_spikes = np.concatenate((pooling_spikes, np.array(padding, dtype=object)))

        pooling_spikes = pooling_spikes.reshape(W, H)
        current_combo_connection=0
        for index_i in range(len(pooling_spikes)):
            for index_j in range(len(pooling_spikes[0])):
                if pooling_spikes[index_i][index_j]:
                    for soma in combos[current_combo_connection]:
                        self.CreateSynapseSTDP(soma, pooling_spikes[index_i][index_j])
                    current_combo_connection+=1
                else:
                    continue
        self.pooling_matrix[layer_idx]=pooling_spikes
            
    def SurjectivePooling(self, layer_idx, W, H):
        #MUST BE A SURJECTIVE RELATIONSHP
        output_channels=len(self.layers[layer_idx])
        output_channels_list=[kernel[1] for kernel in self.layers[layer_idx]]
        input_neurons_to_output_num=np.ceil(W*H/len(output_channels))
        pooling_neurons=np.array([self.CreateSoma() for _ in range(len(W*H))], dtype=object)
        pool_index=-1
        for input_neuron_index in range(len(output_channels_list)):
            if input_neuron_index % input_neurons_to_output_num:
                pool_index+=1
            self.CreateSynapseSTDP(output_channels_list[input_neuron_index],pooling_neurons[pool_index])
        pooling_spikes=pooling_neurons.reshape(W,H)
        self.pooling_matrix[layer_idx]=pooling_neurons

    def CreateSoma(self):
        Soma= self.model.create_soma(
            breed="lif_soma",
            config_name="config_0",
        )
        return Soma
    def CreateSynapseNoSTDP(self, post_soma):
        Synapse =  self.model.create_synapse(
            breed="single_exp_synapse",
            pre_soma_id=np.nan,  # External input
            post_soma_id=post_soma,
            config_name="no_learning_config_0",
        )
        return Synapse
    def CreateSynapseSTDP(self,pre_soma, post_soma):
        Synapse = self.model.create_synapse(
            breed="single_exp_synapse",
            pre_soma_id=pre_soma,
            post_soma_id=post_soma,
            config_name="exp_pair_wise_stdp_config_0",
            hyperparameters_overrides={
                "weight": np.random.uniform(10.0, 20.0),
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
                "tau_pre_stdp": 10e-3,
                "tau_post_stdp": 10e-3,
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

    def invert_dict(self, input_dict, spike_value=10):
        inverted = defaultdict(dict)
        
        for key, values in input_dict.items():
            for v in values:
                inverted[v][key] = spike_value
                
        return dict(inverted)
    def Full_Convolution_And_Extraction(self, Layer_idx, SpikeData,Total_Sim_Time):
        start = time.time()

        for kernel_array, kernel_neuron in self.layers[Layer_idx]:
            # Precompute kernel offsets ONCE
            offsets = self.compute_kernel_offsets(kernel_array)

            for time_step in SpikeData:
                current_spikes = SpikeData[time_step]

                for spike in current_spikes:
                    self.Convolve_Spike(
                         spike,
                         kernel_array,
                         offsets,
                         current_spikes,
                         time_step,
                    )

        end = time.time()
        print(end-start)
        Spike_Times={}
        print('Full Convolution', Layer_idx)
        self.model.simulate(ticks=Total_Sim_Time, update_data_ticks=Total_Sim_Time)
        print('simulation done')
        for index_i in range(len(self.pooling_matrix[Layer_idx])):
            for index_j in range(len(self.pooling_matrix[Layer_idx][0])):
                if self.pooling_matrix[Layer_idx][index_i][index_j]:
                    Coor=(index_i,index_j)
                    Spike_Times[Coor]=self.model.get_spike_times(soma_id=self.pooling_matrix[Layer_idx][index_i][index_j])
        Spike_Times=self.invert_dict(Spike_Times)
        print(Spike_Times)   
        self.model.reset()
        return Spike_Times
    
    def FeedForwardLayerNN(self, InputSize,HiddenSize,OutputSize):
        InputLayer=np.array([self.CreateSoma() for _ in range(InputSize)])
        HiddenLayer=np.array([self.CreateSoma() for _ in range(HiddenSize)])
        OutputLayer=np.array([self.CreateSoma() for _ in range(OutputSize)])
        #initialize in input synapses
        input_synapses=[]
        for post_soma in InputLayer:
            synapse=self.CreateSynapseNoSTDP(post_soma)
            input_synapses.append(synapse)
        InputToHidden,InputToHidden_Dict=self.FullyConnected(InputLayer,HiddenLayer)
        HiddenToOutput, HiddenToOutput_Dict=self.FullyConnected(HiddenLayer,OutputLayer)
        print('Hiddne Dict ',HiddenToOutput_Dict)
        return [input_synapses,OutputLayer, HiddenToOutput_Dict]
    def FullyConnected(self, InputLayer, OutputLayer):
        synapses = []
        synapse_dict = {}  # {post_soma: [synapses_from_all_inputs]}

        for post_soma in OutputLayer:
            synapse_dict[post_soma] = []  # initialize list for this post neuron

        for pre_soma in InputLayer:
            for post_soma in OutputLayer:
                synapse = self.CreateSynapseSTDP(pre_soma, post_soma)
                synapses.append(synapse)
                synapse_dict[post_soma].append(synapse)

        return np.array(synapses, dtype=object), synapse_dict

    def ConstructionOfConvKernel(self, Conv_Kernel_List, output_classes):
        for layer_id in range(len(Conv_Kernel_List)):
            for kernel in range(len(Conv_Kernel_List[layer_id])):
                self.Conv_Kernel_Construction(Conv_Kernel_List[layer_id][kernel][0],Conv_Kernel_List[layer_id][kernel][0],layer_idx=layer_id)
            kernel_count = len(Conv_Kernel_List[layer_id])
            num_combos = kernel_count + (kernel_count * (kernel_count - 1)) // 2  # size-1 + size-2
            Dimension = math.ceil(math.sqrt(num_combos))  
            self.AttentionPooling(layer_id, Dimension, Dimension)
        Last_Layer_idx=len(self.pooling_matrix)-1
        Input_Dimension = self.pooling_matrix[Last_Layer_idx].size

        #need to grab the spikes from this layer 
        self.FF = self.FeedForwardLayerNN(Input_Dimension,2*Input_Dimension,output_classes)
    def ForwardPass(self, SpikeData,Total_Sim_Time):
        Dataset = self.load_bin_as_spike_dict(SpikeData)
        # Fix conv kernel list so that is uses self.layers instead as not passed in anymore
        for layer_id in range(len(self.layers)):
            Dataset=self.Full_Convolution_And_Extraction(layer_id, Dataset, Total_Sim_Time)

        #need to grab the spikes from this layer 
        for time in Dataset:
            for Coor in Dataset[time]:
                self.model.add_spike(
                            synapse_id=self.FF[0][Coor[0]+Coor[1]],
                            tick=time,
                            value=10
                        )
        Spike_Times=[]
        self.model.simulate(ticks=Total_Sim_Time, update_data_ticks=Total_Sim_Time)

        for output_neuron in self.FF[1]:
                
            Values=self.model.get_spike_times(soma_id=output_neuron)
            Spike_Times.append((len(Values),Values))
        
        max_index = max(range(len(Spike_Times)), key=lambda i: Spike_Times[i][0])
        return max_index
    def plot_output_layer_weights(self, forward_idx=None, save_dir="./ff_weight_maps"):
        os.makedirs(save_dir, exist_ok=True)

        syn_dict = self.FF[2]  # {output_soma: [synapse_ids]}
        model = self.model

        num_classes = len(syn_dict)
        syn_per_class = 10  # you took first 10
        grid_rows = num_classes
        grid_cols = 1  # one tile per class, each tile is its own heatmap

        # Build a figure with one heatmap per class (vertically stacked)
        fig, axs = plt.subplots(grid_rows, 1, figsize=(4, 2 * num_classes))

        if grid_rows == 1:  # handle single class case
            axs = [axs]

        for class_idx, (out_soma, syn_list) in enumerate(syn_dict.items()):
            
            synapses = syn_list[:syn_per_class]

            weights = []
            for syn in synapses:
                hyper = model.get_agent_property_value(id=syn, property_name="hyperparameters")
                weights.append(hyper[0])

            # Reshape to 5x2 instead of 2x5
            weight_grid = np.array(weights).reshape(4, 2)

            ax = axs[class_idx]
            im = ax.imshow(weight_grid)  # no cmap, per your plotting rules
            ax.set_title(f"Output Neuron {class_idx}")
            ax.set_xticks([]); ax.set_yticks([])

            # Add colorbar next to each heatmap
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Title for entire figure
        if forward_idx is not None:
            fig.suptitle(f"Output Layer Weights | Pass {forward_idx}", fontsize=14)

        save_path = os.path.join(
            save_dir,
            f"output_layer_pass_{forward_idx if forward_idx is not None else 'final'}.png"
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=250, bbox_inches="tight")
        plt.close()
        print(f"Saved combined figure: {save_path}")

if __name__ == '__main__':    #Create Convolutionary NN
    Model = Conv2dtNet()
    Model.model.setup(use_gpu=True)
    print('gpu set up')
    #make sure I am not loading the entire thing use tonic. 
    Conv_Kernel_List = [
        [(3,3)],  
        [(3,3),(2,3),(3,2)],        
        [(3,3),(2,3),(3,2),(4,4)],              
        [(3,3),(2,3),(3,2)],               
        [(3,3),(2,3),(3,2),(4,4)],                        
        [(3,3),(2,3),(3,2)],
        [(3,3),(2,3)] # this gets fed into a 3 layer spiking FF neural network with majority voting               
    ]
    Model.ConstructionOfConvKernel(Conv_Kernel_List, 10)
    before_reset_hyperparameters = Model.model.get_agent_property_value(
            id=Model.kernel_vals[2][1][1][1][1],
            property_name="hyperparameters"
        )
    print('constructed')
    Dataset = './superneuroabm/ssn/data/NMNIST/Test/0/00011.bin'
    Ans = Model.ForwardPass(Dataset, 100)
    Model.plot_output_layer_weights(forward_idx=0)
    Model.model.reset()
    Dataset = './superneuroabm/ssn/data/NMNIST/Test/0/00004.bin'
    Ans = Model.ForwardPass(Dataset, 100)
    Model.plot_output_layer_weights(forward_idx=1)
    Model.model.reset()
    Dataset = './superneuroabm/ssn/data/NMNIST/Test/0/00014.bin'
    Ans = Model.ForwardPass(Dataset, 100)
    Model.model.reset()
    Model.plot_output_layer_weights(forward_idx=2)
    Dataset = './superneuroabm/ssn/data/NMNIST/Test/0/00026.bin'
    Ans = Model.ForwardPass(Dataset, 100)
    Model.plot_output_layer_weights(forward_idx=3)