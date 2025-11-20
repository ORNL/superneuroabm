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
        time_mask = t_ms <= 100
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

        layer = max(abs(dx), abs(dy))
        max_layer = max((width - 1) / 2, (height - 1) / 2)
        r_norm = layer / max_layer

        # Compute offset same way as before
        offset = self._spiral_offset(dx, dy, layer)

        # Normalize within layer
        if layer == 0:
            o_norm = 0
        else:
            o_norm = offset / (8 * layer)

        return int(20*(r_norm + o_norm) / 2.0)
    def _spiral_offset(self, dx, dy, layer):
        # layer == max(|dx|, |dy|)
        L = int(layer)

        # Top edge (y = -L, x decreasing)
        if dy == -L:
            return int((L - dx))

        # Left edge (x = -L, y increasing)
        if dx == -L:
            return int(2*L + (dy + L))

        # Bottom edge (y = L, x increasing)
        if dy == L:
            return int(4*L + (dx + L))

        # Right edge (x = L, y decreasing)
        # If no other case matched, we are here
        return int(6*L + (L - dy))


    def CreateSoma(self):
            Soma= self.model.create_soma(
                breed="lif_soma",
                config_name="config_0",
                hyperparameters_overrides = {
                    'C':      np.float64(np.random.uniform(5e-9, 15e-9)),
                    'R':      np.float64(np.random.uniform(0.5e6, 2e6)),
                    'vthr':   np.float64(np.random.uniform(-55, -35)),
                    'tref':   np.float64(5e-3),
                    'vrest':  np.float64(np.random.uniform(-65, -55)),
                    'vreset': np.float64(np.random.uniform(-65, -55)),
                    'tref_integration':1,
                    'I_in': 0,
                    'scaling_factor':1e-6,
                },
                default_internal_state_overrides={
                    'v':-60,
                    'tcount':0.0,
                    'tlast':0.0
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
                "weight": np.random.uniform(-5,20),
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
                "a_exp_pre": 0.1,
                "a_exp_post": 0.065,
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

    def Conv_Kernel_Construction(self, W, H, Stride,layer_idx=0, input=-1, ):
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
    
    def Lateral_Inhibition(self, Layer_Tensor, p=0.4):
        N = len(Layer_Tensor)

        for i in range(N):
            # Choose a random 20% subset of OTHER neurons
            possible_targets = [j for j in range(N) if j != i]
            num_targets = max(1, int(p * len(possible_targets)))

            chosen_indices = np.random.choice(
                possible_targets,
                size=num_targets,
                replace=False
            )

            # Create inhibitory connections
            for j in chosen_indices:
                self.CreateSynapseNoSTDP(
                    Layer_Tensor[i],
                    Layer_Tensor[j],
                    -15
                )
        
    def Output_Channel_Construction(self, Layer_idx, Output_Layer_Size, Input_W, Input_H, p=0.2):
        # 1. Create output channel neurons
        Output_Channel = np.empty(int(Output_Layer_Size), dtype=object)
        for i in range(int(Output_Layer_Size)):
            Output_Channel[i] = self.CreateSoma()

        self.Output_Channel[Layer_idx] = Output_Channel

        # 2. For each kernel in this layer
        for kernel in self.ConvLayers[Layer_idx]:

            kernel_height = len(kernel[0])
            kernel_width = len(kernel[0][0])
            stride = kernel[2]

            # Compute how many rastered positions exist (standard conv output size)
            Output_H = (Input_H - kernel_height) // stride + 1
            Output_W = (Input_W - kernel_width) // stride + 1
            Output = Output_H * Output_W

            kernel_somas = kernel[1]
            flat_kernel_somas = [
                kernel_somas[i][j]
                for i in range(kernel_height)
                for j in range(kernel_width)
            ]

            for out_idx in range(Output):
                for k_soma in flat_kernel_somas:
                    self.CreateSynapseSTDP(k_soma, Output_Channel[out_idx])
        self.Lateral_Inhibition(self.Output_Channel[Layer_idx])


    def FeedForwardLayerNN(self, InputSize, HiddenSize, OutputSize):
        InputLayer=np.array([self.CreateSoma() for _ in range(InputSize)])
        HiddenLayer=np.array([self.CreateSoma() for _ in range(HiddenSize)])
        OutputLayer=np.array([self.CreateSoma() for _ in range(OutputSize)])
        #initialize in input synapses
        input_synapses=[]
        for post_soma in InputLayer:
            synapse=self.CreateSynapseSTDP(-1, post_soma)
            #synapse=self.CreateSynapseNoSTDP(-1,post_soma,10)
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
                #synapse=self.CreateSynapseNoSTDP(pre_soma,post_soma,10)
                synapses.append(synapse)
                synapse_dict[post_soma].append(synapse)

        return np.array(synapses, dtype=object), synapse_dict

    def NetworkConstruction(self, Conv_Kernel_List, output_classes, Input_W, Input_H):

        for layer_id in range(len(Conv_Kernel_List)):

            Max_Output_Channel = 0
            Max_Output_H = 0
            Max_Output_W = 0

            for kernel_specs in Conv_Kernel_List[layer_id]:
                K_H = kernel_specs[0]
                K_W = kernel_specs[1]
                stride=kernel_specs[2]
                self.Conv_Kernel_Construction(kernel_specs[0], kernel_specs[1], stride,layer_idx=layer_id, )

                # Correct output size formula
                Output_H = (Input_H - K_H) // stride + 1
                Output_W = (Input_W - K_W) // stride + 1

                Output_Channel = Output_H * Output_W

                Max_Output_Channel = max(Max_Output_Channel, Output_Channel)
                Max_Output_H = max(Max_Output_H, Output_H)
                Max_Output_W = max(Max_Output_W, Output_W)

            # Build output neurons for this layer
            self.Output_Channel_Construction(layer_id, Max_Output_Channel, Max_Output_W, Max_Output_H)

            # Save dims
            self.Output_Channel_Dim[layer_id] = [Max_Output_W, Max_Output_H]

            # Update input for the next layer
            Input_H = Max_Output_H
            Input_W = Max_Output_W

        # Build final fully connected readout
        last_layer = max(self.Output_Channel_Dim.keys())
        W_last, H_last = self.Output_Channel_Dim[last_layer]

        FF_input_size = W_last * H_last

        self.FF = self.FeedForwardLayerNN(
            FF_input_size,
            2 * FF_input_size,
            output_classes
        )
       

    
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


    def Convolve_Spike(self, SynapseDict, spike, Kernel, offsets, CurrentSpikeSet,
                   time_step, Input_W, Input_H, Stride=1):

        sx, sy = spike  # sx=x(col), sy=y(row)

        H = len(Kernel)        # kernel height  (# rows)
        W = len(Kernel[0])     # kernel width   (# cols)

        for ki0, kj0 in offsets:

            # Compute top-left of kernel window in (row, col) space
            row0 = sy - ki0
            col0 = sx - kj0

            # Skip illegal kernel placements
            if row0 < 0 or col0 < 0:
                continue
            if (row0 + H) > Input_H or (col0 + W) > Input_W:
                continue

            # Stride constraint
            if (row0 % Stride != 0) or (col0 % Stride != 0):
                continue

            # Sweep over the kernel window
            for ki in range(H):
                for kj in range(W):

                    row = row0 + ki
                    col = col0 + kj

                    # Check for spike at (col,row) since spike dict uses (x,y)
                    if (col, row) in CurrentSpikeSet:

                        syn = Kernel[ki][kj]          # correct kernel neuron
                        val = CurrentSpikeSet[(col, row)]

                        SynapseDict.setdefault(syn, []).append([time_step, val])


            
    def invert_dict(self, input_dict, W, H) :
        inverted = defaultdict(dict)
        
        for key, values in input_dict.items():
            for v in values:
                inverted[v][key] = self.spiral_value_for_coord(W, H, key[0], key[1])
                
        return dict(inverted)


    def process_spikes_spiral(self, Layer_idx, Total_Sim_Time):

        # --- 1. Run simulation ---
        self.model.simulate(ticks=Total_Sim_Time, update_data_ticks=Total_Sim_Time)
        print("Finished Simulate")

        # --- 2. Get spike times for each soma in this layer's output channel ---
        spike_counts = []   # list of (soma_id, spike_times, count)

        for soma_id in self.Output_Channel[Layer_idx]:
            times = self.model.get_spike_times(soma_id=soma_id)
            spike_counts.append((soma_id, times, len(times)))

        # --- 3. Sort by spike count DESC ---
        spike_counts.sort(key=lambda x: x[2], reverse=True)

        # --- 4. Build spiral mapping (x,y) -> soma_id ---
        W = self.Output_Channel_Dim[Layer_idx][0]
        H = self.Output_Channel_Dim[Layer_idx][1]

        spiral_coords = self.generate_spiral_order(W, H)

        coord_map = {}   # (x,y) -> soma_id

        for (soma_id, times, _), (x, y) in zip(spike_counts, spiral_coords):
            coord_map[(x, y)] = (soma_id, times)

        # --- 5. Build time-indexed spike output: { t : { (x,y): value } } ---
        time_dict = defaultdict(dict)

        for (x, y), (soma_id, times) in coord_map.items():
            for t in times:
                value = self.spiral_value_for_coord(W, H, x, y)
                time_dict[t][(x, y)] = value

        return dict(time_dict)



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
        print('SPIKE DATA')
        print(SpikeData)
        print(self.Output_Channel_Dim[Layer_idx])
        start=clocktime.time()
        Synpase_Dict={}
        print('Adding Spikes per kernel')
        for kernel_array_index in range(len(self.ConvLayers[Layer_idx])):
            # Precompute kernel offsets ONCE
            offsets = self.compute_kernel_offsets(self.ConvLayers[Layer_idx][kernel_array_index][0])
            for time_step in SpikeData:
                current_spikes = SpikeData[time_step]
                for spike in current_spikes:
                    if Layer_idx==0:
                        input_w=28
                        input_h=28
                    else:
                        input_w=self.Output_Channel_Dim[Layer_idx-1][0]
                        input_h=self.Output_Channel_Dim[Layer_idx-1][1]
                    self.Convolve_Spike(
                        Synpase_Dict,
                        spike,
                        self.ConvLayers[Layer_idx][kernel_array_index][0],
                        offsets,
                        current_spikes,
                        time_step,
                        input_w,
                        input_h,
                        Stride=self.ConvLayers[Layer_idx][kernel_array_index][2]
                    )
        for key in Synpase_Dict:
            self.model.add_spike_list(key,Synpase_Dict[key])
        print('start sim')
        Spike_Times=self.process_spikes_spiral(Layer_idx, Total_Sim_Time)
        print('end sim')
        end=clocktime.time()
        print(end-start)
        return Spike_Times
    


    def ForwardPass(self, SpikeData,Total_Sim_Time):
        print('Start Forward Pass')
        start=clocktime.time()
        Dataset = self.load_bin_as_spike_dict(SpikeData)
        # Fix conv kernel list so that is uses self.layers instead as not passed in anymore
        for layer_id in range(len(self.ConvLayers)):
            print(layer_id)
            Dataset = self.Full_Convolution_And_Extraction(layer_id, Dataset, Total_Sim_Time)
            
            # Reset after each layer EXCEPT the last
            if layer_id != len(self.ConvLayers) - 1:
                self.model.reset()


        print("Feed Forward Part")

        last_layer = max(self.Output_Channel.keys())
        final_neurons = self.Output_Channel[last_layer]

        # Gather all spike times from the model BEFORE reset
        final_spike_dict = defaultdict(list)

        for idx, soma_id in enumerate(final_neurons):
            spike_times = self.model.get_spike_times(soma_id=soma_id)
            for t in spike_times:
                final_spike_dict[idx].append(int(t))

        # Reset before building the FF layer activity
        self.model.reset()

        # ----------------------------------------------
        #  Inject the final-layer spikes into FF synapses
        # ----------------------------------------------
        # FF[0] = list of input synapses
        input_synapses = self.FF[0]

        for idx, times in final_spike_dict.items():
            syn_id = input_synapses[idx]
            for t in times:
                self.model.add_spike(
                    synapse_id=syn_id,
                    tick=t,
                    value=1   # constant input value (important!)
                )

        # ----------------------------------------------
        #  Run FF simulation
        # ----------------------------------------------
        Spike_Times = []
        self.model.simulate(ticks=Total_Sim_Time, update_data_ticks=Total_Sim_Time)

        for output_neuron in self.FF[1]:
            Values = self.model.get_spike_times(soma_id=output_neuron)
            Spike_Times.append((len(Values), Values))

        # Choose class with most spikes
        max_index = max(range(len(Spike_Times)), key=lambda i: Spike_Times[i][0])

        print("\n=== Output Layer Spike Times ===")
        for idx, (count, times) in enumerate(Spike_Times):
            print(f"Output neuron {idx}: {count} spikes → {list(times)}")
        print("================================\n")

        return max_index

    def plot_all_passes_weight_maps(self, weight_matrices, save_path="all_forward_passes.png"):
        num_passes = len(weight_matrices)

        # global color scale
        vmin = min(np.nanmin(W) for W in weight_matrices)
        vmax = max(np.nanmax(W) for W in weight_matrices)

        fig, axs = plt.subplots(
            nrows=1,
            ncols=num_passes,
            figsize=(4 * num_passes, 4)
        )

        if num_passes == 1:
            axs = [axs]

        for i, (ax, W) in enumerate(zip(axs, weight_matrices)):
            im = ax.imshow(W, vmin=vmin, vmax=vmax, cmap="viridis")
            ax.set_title(f"Pass {i}")
            ax.set_xticks([])
            ax.set_yticks([])

        # shared colorbar
        fig.colorbar(im, ax=axs, fraction=0.025, pad=0.04)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved combined plot: {save_path}")

    def get_ff_weight_matrix(self, neuron_index=0):
        syn_dict = self.FF[2]  # {output_soma: [syn_id]}
        out_somas = list(syn_dict.keys())

        if neuron_index >= len(out_somas):
            raise ValueError(f"Invalid neuron index {neuron_index}")

        soma_id = out_somas[neuron_index]
        syn_list = syn_dict[soma_id]

        # Final-layer dims
        last_layer = max(self.Output_Channel_Dim.keys())
        W, H = self.Output_Channel_Dim[last_layer]

        # Fetch weights
        weights = []
        for syn in syn_list:
            hyper = self.model.get_agent_property_value(id=syn, property_name="hyperparameters")
            weights.append(hyper[0])

        # Reshape into matrix
        return np.array(weights).reshape(2 * H, W)


if __name__ == "__main__":
    # --- Initialize model ---
    Model = Conv2dtNet()
    Model.model.setup(use_gpu=True)
    print("GPU setup complete")

    # --- Define convolutional architecture ---
    Conv_Kernel_List = [
        [(3,3,2)],
        [(5,5,1)],
        [(4,4,1)],
        [(3,3,1)]
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

    # --- Run one example per class ---
    root = "./data/NMNIST/Test"
    assert os.path.isdir(root), f"NMNIST Test directory not found: {root}"

    digit_dirs = sorted(os.listdir(root))
    print("\n=== Running one example per class ===")

    results = {}
    all_weight_matrices = []   # <-- store matrices for each forward pass
    counter=0
    for idx, digit in enumerate(digit_dirs):
        digit_path = os.path.join(root, digit)
        bin_files = [f for f in os.listdir(digit_path) if f.endswith(".bin")]
        assert bin_files, f"No .bin files found in {digit_path}"

        # pick the first file for this digit
        bin_file = bin_files[0]
        dataset_path = os.path.join(digit_path, bin_file)

        print(f"\n--- Forward pass {idx} | Digit {digit} | File: {bin_file} ---")

        predicted_class = Model.ForwardPass(dataset_path, Total_Sim_Time=100)
        results[digit] = predicted_class

        # grab weight matrix instead of printing
        W = Model.get_ff_weight_matrix(neuron_index=0)
        all_weight_matrices.append(W)

        Model.model.reset()
        counter+=1
        if counter==5:
            break

    print("\n=== Summary of Predictions ===")
    for digit, pred in results.items():
        print(f"Digit {digit} → Predicted {pred}")

    # --- Plot weight matrices side-by-side ---
    Model.plot_all_passes_weight_maps(
        all_weight_matrices,
        save_path="all_forward_passes.png"
    )
