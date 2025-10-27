import os
import numpy as np
from superneuroabm.model import NeuromorphicModel
import sagesim
import tonic 
import math
import itertools
from itertools import combinations
import tonic.transforms as transforms
import tempfile
from collections import defaultdict

class Conv2dtSingleLayer:
    def __init__(self, SpikeData=None):
        # Spike Data should be organized by spike time
        # SpikeData[t] = [(xi, yi, spike val), (xj, yj, spike val)....] all incoming spikes
        self.model = NeuromorphicModel()
        self.SpikeData = SpikeData
        self.layers = {}           # {layer_idx: [(Kernel, Kernel_Neuron), ...]}
        self.kernel_vals = {}
        self.pooling_matrix = {}

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
        on_mask = p == 1
        x, y, t_ms = x[on_mask], y[on_mask], t_ms[on_mask]

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


    def Convolve_Spike(self, SpikeCoordinate, Kernel_List_Entry, CurrentSpikeSet, time_step, Stride):
        """
        Apply convolution for a given spike coordinate and kernel.
        """
        Ranges = []
        for i in range(len(Kernel_List_Entry)):
            for j in range(len(Kernel_List_Entry[0])):
                Ranges_X = (SpikeCoordinate[0] - i, SpikeCoordinate[0] - i + len(Kernel_List_Entry))
                Ranges_Y = (SpikeCoordinate[1] - j, SpikeCoordinate[1] - j + len(Kernel_List_Entry[0]))
                Range = (Ranges_X, Ranges_Y)
                Ranges.append(Range)

        for x_range, y_range in Ranges:
            x_start, x_end = x_range
            y_start, y_end = y_range

            for x in range(x_start, x_end, Stride):
                for y in range(y_start, y_end):
                    coor = (x, y)
                    if coor in CurrentSpikeSet:
                        kernel_location_x = x - x_start
                        kernel_location_y = y - y_start
                        self.model.add_spike(
                            synapse_id=Kernel_List_Entry[kernel_location_x][kernel_location_y],
                            tick=time_step,
                            value=CurrentSpikeSet[coor]
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
        for time in SpikeData:
            for kernel_array, kernel_neuron in self.layers[Layer_idx]:
                    for spike in SpikeData[time]:
                        self.Convolve_Spike(spike, kernel_array, SpikeData[time], time, 1)
        
        Spike_Times={}
        print('Full Convolution', Layer_idx)
        self.model.simulate(ticks=Total_Sim_Time, update_data_ticks=Total_Sim_Time)
        #print('simulation done')
        for index_i in range(len(self.pooling_matrix[Layer_idx])):
            for index_j in range(len(self.pooling_matrix[Layer_idx][0])):
                if self.pooling_matrix[Layer_idx][index_i][index_j]:
                    Coor=(index_i,index_j)
                    Spike_Times[Coor]=self.model.get_spike_times(soma_id=Model.pooling_matrix[Layer_idx][index_i][index_j])
                    #{(x,y):[spike time list],....}
        Spike_Times=self.invert_dict(Spike_Times)
        #print(Spike_Times)   
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
        InputToHidden=self.FullyConnected(InputLayer,HiddenLayer)
        HiddenToOutputSize=self.FullyConnected(HiddenLayer,OutputLayer)
        return [input_synapses,OutputLayer]
    def FullyConnected(self, InputLayer, OutputLayer):
        synapses = []   # collect here

        for pre_soma in InputLayer:
            for post_soma in OutputLayer:
                synapse = self.CreateSynapseSTDP(pre_soma, post_soma)
                synapses.append(synapse)

        return np.array(synapses, dtype=object)
    def ForwardPass(self, SpikeData, Conv_Kernel_List,Total_Sim_Time):
        #Conv_Kernel_List is a list of lists of tuples with matrix sizes [[(w,h), (w_1,h_1),...][(w,h),...],...]
        #Each list is a layer respectively
        #Spike Data is a single bin file
        for layer_id in range(len(Conv_Kernel_List)):
            for kernel in range(len(Conv_Kernel_List[layer_id])):
                self.Conv_Kernel_Construction(Conv_Kernel_List[layer_id][kernel][0],Conv_Kernel_List[layer_id][kernel][0],layer_idx=layer_id)
            kernel_count = len(Conv_Kernel_List[layer_id])
            num_combos = kernel_count + (kernel_count * (kernel_count - 1)) // 2  # size-1 + size-2
            Dimension = math.ceil(math.sqrt(num_combos))  
            self.AttentionPooling(layer_id, Dimension, Dimension)
        Dataset = self.load_bin_as_spike_dict(SpikeData)

        for layer_id in range(len(Conv_Kernel_List)):
            Dataset=self.Full_Convolution_And_Extraction(layer_id, Dataset, Total_Sim_Time)
        Last_Layer_idx=len(self.pooling_matrix)-1
        Input_Dimension = self.pooling_matrix[Last_Layer_idx].size

        #need to grab the spikes from this layer 
        FF=self.FeedForwardLayerNN(Input_Dimension,2*Input_Dimension,10) # 10 represents the classes in MNIST.
        for time in Dataset:
            for Coor in Dataset[time]:
                self.model.add_spike(
                            synapse_id=FF[0][Coor[0]+Coor[1]],
                            tick=time,
                            value=10
                        )
        Spike_Times=[]
        self.model.simulate(ticks=Total_Sim_Time, update_data_ticks=Total_Sim_Time)

        for output_neuron in FF[1]:
                
            Values=self.model.get_spike_times(soma_id=output_neuron)
            Spike_Times.append((len(Values),Values))
        
        max_index = max(range(len(Spike_Times)), key=lambda i: Spike_Times[i][0])
        return max_index

if __name__ == '__main__':
    #Create Convolutionary NN
    Model = Conv2dtSingleLayer()
    Model.model.setup(use_gpu=True)

    Dataset = './data/NMNIST/Test/1/00003.bin'
    Conv_Kernel_List = [
        [(3,3),(3,2),(2,3),(4,4)],  
        [(3,3),(2,3),(3,2)],        
        [(3,3),(2,3),(3,2),(4,4)],              
        [(3,3),(2,3),(3,2)],               
        [(3,3),(2,3),(3,2),(4,4)],                        
        [(3,3),(2,3),(3,2)],
        [(3,3),(2,3)]                  
    ]
    Ans = Model.ForwardPass(Dataset, Conv_Kernel_List, 100)
    print('Class returned is', Ans)