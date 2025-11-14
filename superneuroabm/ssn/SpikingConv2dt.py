import os
import numpy as np
from superneuroabm.model import NeuromorphicModel
import sagesim
import time as clocktime
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
        time_mask = t_ms <= 5
        x, y, t_ms = x[time_mask], y[time_mask], t_ms[time_mask]


        # Group by timestamp into a dict-of-dicts
        spike_dict = {}
        for xi, yi, ti in zip(x, y, t_ms):
            value = (self.spiral_value_for_coord(28,28,int(xi), int(yi))) + 1
            spike_dict.setdefault(int(ti), {})[(int(xi), int(yi))] = value

        return spike_dict


    def spiral_value_for_coord(self, width, height, x, y):
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0

        dx = x - cx
        dy = y - cy

        layer = int(max(abs(dx), abs(dy)))

        side_len = 2 * layer + 1

        prev_count = (side_len - 2) ** 2 if layer > 0 else 0

        top_right_x = cx + layer
        top_right_y = cy - layer

        if y == cy - layer: 
            offset = int(top_right_x - x)
        elif x == cx - layer:  
            offset = int(2 * layer + (y - (cy - layer)))
        elif y == cy + layer:  
            offset = int(4 * layer + (x - (cx - layer)))
        else:  
            offset = int(6 * layer + ((cy + layer) - y))

        return prev_count + offset + 5

    def CreateSoma(self):
            Soma= self.model.create_soma(
                breed="lif_soma",
                config_name="config_0",
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
                "weight": np.random.uniform(-5.0, 20.0),
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
                "tau_pre_stdp": 20-3,
                "tau_post_stdp": 33e-3,
                "a_exp_pre": 0.01,
                "a_exp_post": 0.0065,
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

    def Conv_Kernel_Construction(self, W, H, layer_idx=0, input=-1, Stride=1):
        Kernel=np.empty((W, H), dtype=object)
        KernelSynapses = np.empty((W, H), dtype=object)
        for i in range(W):
            for j in range(H):
                Kernel_Neuron=self.CreateSoma()
                Kernel[i][j] = Kernel_Neuron
        for i in range(W):
            for j in range(H):
                KernelSynapse=self.CreateSynapseSTDP(-1,Kernel[i][j])
                KernelSynapses[i][j] = KernelSynapse
       
        if layer_idx not in self.ConvLayers:
            self.ConvLayers[layer_idx] = []
        self.ConvLayers[layer_idx].append([KernelSynapses, Kernel, Stride])
    
    def Lateral_Inhibition(self, Layer_Tensor):
            for neuron_index_i in range(0,len(Layer_Tensor)):
                for neuron_index_j in range(0,len(Layer_Tensor)):
                    if neuron_index_i == neuron_index_j:
                        continue
                    else:
                        self.CreateSynapseNoSTDP(Layer_Tensor[neuron_index_i], Layer_Tensor[neuron_index_j], -20)

    
    def Output_Channel_Construction(self, Layer_idx, Output_Layer_Size, Input_W, Input_H):
        #Ensure Output Channel is equal to or greater then 
        Output_Channel=np.empty(int(Output_Layer_Size), dtype=object)
        for i in range(int(Output_Layer_Size)):
            Output_Channel[i]=self.CreateSoma()
        self.Output_Channel[Layer_idx]=Output_Channel
        for kernel in self.ConvLayers[Layer_idx]:
            Output_H = (Input_H-len(kernel[0]))/1
            Output_W = (Input_W - len(kernel[0][0]))// kernel[2]
            Output = Output_H * Output_W
            for kernel_i in range(len(kernel[1])):
                for kernel_j in range(len(kernel[1][0])):
                    Counter=0
                    for output_neuron in self.Output_Channel[Layer_idx]:
                        if Counter>= Output:
                            break
                        self.CreateSynapseSTDP(kernel[1][kernel_i][kernel_j], output_neuron)
                        Counter+=1
        self.Lateral_Inhibition(self.Output_Channel[Layer_idx])

    def FeedForwardLayerNN(self, InputSize, HiddenSize, OutputSize):
        InputLayer=np.array([self.CreateSoma() for _ in range(InputSize)])
        HiddenLayer=np.array([self.CreateSoma() for _ in range(HiddenSize)])
        OutputLayer=np.array([self.CreateSoma() for _ in range(OutputSize)])
        #initialize in input synapses
        input_synapses=[]
        for post_soma in InputLayer:
            synapse=self.CreateSynapseSTDP(-1, post_soma)
            input_synapses.append(synapse)
        InputToHidden,InputToHidden_Dict=self.FullyConnected(InputLayer,HiddenLayer)
        HiddenToOutput, HiddenToOutput_Dict=self.FullyConnected(HiddenLayer,OutputLayer)
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

    def NetworkConstruction(self, Conv_Kernel_List, output_classes, Input_W, Input_H):
        for layer_id in range(len(Conv_Kernel_List)):
            Max_Output_Channel=0
            Max_Output_H = 0
            Max_Output_W = 0 
            for kernel in range(len(Conv_Kernel_List[layer_id])):
                self.Conv_Kernel_Construction(Conv_Kernel_List[layer_id][kernel][0],Conv_Kernel_List[layer_id][kernel][1],layer_idx=layer_id)
                Output_H = (Input_H-Conv_Kernel_List[layer_id][kernel][1])/1
                Output_W = (Input_W - Conv_Kernel_List[layer_id][kernel][0])/1
                Max_Output_Channel=max((Output_W*Output_W), Max_Output_Channel)
                Max_Output_H = max(Output_W, Output_H)
                Max_Output_W = max(Output_W, Output_W)
            self.Output_Channel_Construction(layer_id, Max_Output_Channel, Input_W, Input_H)
            self.Output_Channel_Dim[layer_id]=[Max_Output_W, Max_Output_H]
            Input_H = Max_Output_H
            Input_W = Max_Output_W
        self.FF = self.FeedForwardLayerNN(int(Max_Output_Channel),2*int(Max_Output_Channel),output_classes)           

    
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


    def Convolve_Spike(self, SynapseDict, spike, Kernel, offsets, CurrentSpikeSet, time_step, Stride=1):
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
                        if Kernel[ki][kj] in SynapseDict:
                            SynapseDict[Kernel[ki][kj]].append([time_step, CurrentSpikeSet[(x, y)]])
                        else:
                            SynapseDict[Kernel[ki][kj]]=[[time_step, CurrentSpikeSet[(x, y)]]]
                            #G
                    
        return SynapseDict     

    def invert_dict(self, input_dict, W, H) :
        inverted = defaultdict(dict)
        
        for key, values in input_dict.items():
            for v in values:
                inverted[v][key] = self.spiral_value_for_coord(W, H, key[0], key[1])
                
        return dict(inverted)


    def process_spikes_spiral(self, Layer_idx, Total_Sim_Time):
        self.model.simulate(ticks=Total_Sim_Time, update_data_ticks=Total_Sim_Time)
        print('Finished Simulate')
        Spike_Times = {}
        for i in range(len(self.Output_Channel[Layer_idx])):
            Spike_Times[i] = self.model.get_spike_times(
                soma_id=self.Output_Channel[Layer_idx][i]
            )

        sorted_neurons = sorted(Spike_Times.items(), key=lambda x: len(x[1]), reverse=True)

        Spike_Matrix = np.empty((int(self.Output_Channel_Dim[Layer_idx][0]), int(self.Output_Channel_Dim[Layer_idx][1])), dtype=object)

        spiral_coords = self.generate_spiral_order(int(self.Output_Channel_Dim[Layer_idx][0]), int(self.Output_Channel_Dim[Layer_idx][1]))

        for (neuron_idx, _), (x, y) in zip(sorted_neurons, spiral_coords):
            Spike_Matrix[x, y] = neuron_idx

        Spike_Times = {}
        for index_i in range(len(Spike_Matrix)):
            for index_j in range(len(Spike_Matrix[0])):
                if Spike_Matrix[index_i][index_j]:
                    Coor=(index_i,index_j)
                    Spike_Times[Coor]=self.model.get_spike_times(soma_id=Spike_Matrix[index_i][index_j])
        Spike_Times=self.invert_dict(Spike_Times, len(Spike_Matrix), len(Spike_Matrix[0]))

        self.model.reset()

        return Spike_Times


    def generate_spiral_order(self, W, H):
        coords = []

        cx = (W - 1) / 2.0
        cy = (H - 1) / 2.0
        max_layer = int(max(cx, cy))

        for layer in range(max_layer + 1):
            top_right_x = int(cx + layer)
            top_right_y = int(cy - layer)

            # --- Top edge (right → left)
            for x in range(top_right_x, int(cx - layer) - 1, -1):
                y = top_right_y
                if 0 <= x < W and 0 <= y < H:
                    coords.append((x, y))

            # --- Left edge (top → bottom)
            for y in range(int(cy - layer) + 1, int(cy + layer) + 1):
                x = int(cx - layer)
                if 0 <= x < W and 0 <= y < H:
                    coords.append((x, y))

            # --- Bottom edge (left → right)
            for x in range(int(cx - layer) + 1, int(cx + layer) + 1):
                y = int(cy + layer)
                if 0 <= x < W and 0 <= y < H:
                    coords.append((x, y))

            # --- Right edge (bottom → top)
            for y in range(int(cy + layer) - 1, int(cy - layer), -1):
                x = int(cx + layer)
                if 0 <= x < W and 0 <= y < H:
                    coords.append((x, y))

        return coords



    def Full_Convolution_And_Extraction(self, Layer_idx, SpikeData,Total_Sim_Time):
        Synpase_Dict={}
        print('Adding Spikes per kernel')
        for kernel_array_index in range(len(self.ConvLayers[Layer_idx])):
            # Precompute kernel offsets ONCE
            offsets = self.compute_kernel_offsets(self.ConvLayers[Layer_idx][kernel_array_index][0])
            for time_step in SpikeData:
                current_spikes = SpikeData[time_step]
                for spike in current_spikes:
                    Synpase_Dict=self.Convolve_Spike(
                         Synpase_Dict,
                         spike,
                         self.ConvLayers[Layer_idx][kernel_array_index][0],
                         offsets,
                         current_spikes,
                         time_step,
                    )
        print('Convolved')
        for key in Synpase_Dict:
            self.model.add_spike_list(key,Synpase_Dict[key])
        print('start sim')
        Spike_Times=self.process_spikes_spiral(Layer_idx, Total_Sim_Time)
        print('end sim')
        return Spike_Times
    


    def ForwardPass(self, SpikeData,Total_Sim_Time):
        print('Start Forward Pass')
        start=clocktime.time()
        Dataset = self.load_bin_as_spike_dict(SpikeData)
        # Fix conv kernel list so that is uses self.layers instead as not passed in anymore
        for layer_id in range(len(self.ConvLayers)):
            print(Dataset)
            print(layer_id)
            Dataset=self.Full_Convolution_And_Extraction(layer_id, Dataset, Total_Sim_Time)

        #need to grab the spikes from this layer 
        print('Feed Forward Part')
        for time in Dataset:
            for Coor in Dataset[time]:
                self.model.add_spike(
                            synapse_id=self.FF[0][Coor[0]+Coor[1]],
                            tick=time,
                            value=self.spiral_value_for_coord(self.Output_Channel_Dim[-1][0], self.Output_Channel_Dim[-1][1], Coor[0], Coor[1])
                        )
        Spike_Times=[]
        self.model.simulate(ticks=Total_Sim_Time, update_data_ticks=Total_Sim_Time)

        for output_neuron in self.FF[1]:
                
            Values=self.model.get_spike_times(soma_id=output_neuron)
            Spike_Times.append((len(Values),Values))
        
        max_index = max(range(len(Spike_Times)), key=lambda i: Spike_Times[i][0])
        end=clocktime.time()
        print('Forward Pass Wall clock time',end-start)
        return max_index
    def plot_output_layer_weights(self, forward_idx=None, save_dir="./ff_weight_maps"):
        os.makedirs(save_dir, exist_ok=True)

        syn_dict = self.FF[2]  # {output_soma: [synapse_ids]}
        model = self.model

        # --- Gather all weights first to get global min/max for consistent color scale ---
        all_weights = []
        for syn_list in syn_dict.values():
            for syn in syn_list:
                hyper = model.get_agent_property_value(id=syn, property_name="hyperparameters")
                all_weights.append(hyper[0])
        vmin, vmax = np.min(all_weights), np.max(all_weights)

        # --- Plot setup ---
        num_classes = len(syn_dict)
        grid_rows = num_classes
        fig, axs = plt.subplots(grid_rows, 1, figsize=(4, 2 * num_classes))
        if grid_rows == 1:
            axs = [axs]

        # --- Plot each class ---
        for class_idx, (out_soma, syn_list) in enumerate(syn_dict.items()):
            weights = []
            for syn in syn_list:
                hyper = model.get_agent_property_value(id=syn, property_name="hyperparameters")
                weights.append(hyper[0])

            weights = np.array(weights)
            n_weights = len(weights)
            n_cols = math.ceil(math.sqrt(n_weights))
            n_rows = math.ceil(n_weights / n_cols)

            # pad with NaN so we can reshape safely
            padded = np.full(n_rows * n_cols, np.nan)
            padded[:n_weights] = weights
            weight_grid = padded.reshape(n_rows, n_cols)

            ax = axs[class_idx]
            im = ax.imshow(weight_grid, vmin=vmin, vmax=vmax, cmap="viridis")
            ax.set_title(f"Output Neuron {class_idx}")
            ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # --- Save figure ---
        if forward_idx is not None:
            fig.suptitle(f"Output Layer Weights | Pass {forward_idx}", fontsize=14)

        plt.tight_layout()
        save_path = os.path.join(
            save_dir,
            f"output_layer_pass_{forward_idx if forward_idx is not None else 'final'}.png"
        )
        plt.savefig(save_path, dpi=250, bbox_inches="tight")
        plt.close()
        print(f"Saved figure with shared color scale: {save_path}")
if __name__ == "__main__":
    # --- Initialize model ---
    Model = Conv2dtNet()
    Model.model.setup(use_gpu=True)
    print("GPU setup complete")

    # --- Define convolutional architecture ---
    Conv_Kernel_List = [
        [(22, 22)],
        [(3,3)]
    ]

    # --- Build network ---
    Model.NetworkConstruction(
        Conv_Kernel_List=Conv_Kernel_List,
        output_classes=10,
        Input_W=28,
        Input_H=28,
    )
    print("Network constructed")
    print("Number of somas:", len(Model.model._soma_ids))
    print("Number of synapses:", len(Model.model._synapse_ids))
    print("Total agents:", len(Model.model._soma_ids) + len(Model.model._synapse_ids))
    # --- Run one NMNIST example ---
    # get NMNIST test data
    # test_dataset = tonic.datasets.NMNIST(save_to="./superneuroabm/ssn/data/", train=False)
    root = "./data/NMNIST/Test"
    assert os.path.isdir(root), f"NMNIST Test directory not found: {root}"

    # Pick first available digit and file
    first_digit = sorted(os.listdir(root))[0]
    digit_path = os.path.join(root, first_digit)
    bin_files = [f for f in os.listdir(digit_path) if f.endswith(".bin")]
    assert bin_files, f"No .bin files found in {digit_path}"
    bin_file = bin_files[0]
    dataset_path = os.path.join(digit_path, bin_file)

    print(f"Running example → Digit {first_digit} | File: {bin_file}")
    predicted_class = Model.ForwardPass(dataset_path, Total_Sim_Time=5)
    print(f"Predicted class: {predicted_class} (True: {first_digit})")
