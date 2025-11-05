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
        self._queued_syn_ids = []
        self._queued_ticks = []
        self._queued_vals = []

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

    def all_relative_coordinate_grids(self, W, H):
        # base coordinate grid (W,H,2)
        base = np.stack(np.meshgrid(np.arange(W), np.arange(H), indexing="ij"), axis=-1)  # (W,H,2)

        # flatten all centers (W*H,2)
        centers = base.reshape(-1, 2)  # (W*H, 2)

        # subtract center from base to get relative coords
        # broadcasting: (W*H,1,1,2) - (1,W,H,2)
        rel = base[None,:,:,:] - centers[:,None,None,:]

        return rel  # shape (W*H, W, H, 2)
    def get_neighborhood_spikes(self, Neighborhood, CurrentSpikeDict):
        # Create structured dtype matching Neighborhood
        dtype = [('x', Neighborhood.dtype), ('y', Neighborhood.dtype)]

        # Convert dict keys to structured np array
        dict_keys = np.array(list(CurrentSpikeDict.keys()), dtype=dtype)

        # Convert dict values to array
        dict_vals = np.array(list(CurrentSpikeDict.values()))

        # View neighborhood as structured for comparison
        nb = Neighborhood.view(dtype).reshape(-1)

        # Mask where neighborhood coords exist in spike dict
        mask = np.isin(nb, dict_keys)

        if not mask.any():
            return None, None

        # --- NEW PART: kernel index positions instead of coordinate values ---
        # mask is flat — convert to 2D kernel indices
        kernel_indices = np.argwhere(mask.reshape(Neighborhood.shape[0], Neighborhood.shape[1]))

        # Map mask positions to spike values
        idx = np.argmax(nb[:,None] == dict_keys, axis=1)[mask]
        vals = dict_vals[idx]

        return kernel_indices, vals

    def Convolve_Spike(self, SpikeCoordinate, Kernel_List_Entry, Offsets, CurrentSpikeSet, time_step, Stride):
        """
        Apply convolution for a given spike coordinate and kernel.
        """
        coord=np.array([SpikeCoordinate[0],SpikeCoordinate[1]])
        Neighborhoods=coord+Offsets
        for Neighborhood in Neighborhoods:
            kernel_indices, vals= self.get_neighborhood_spikes(Neighborhood, CurrentSpikeSet)
            for (i,j), v in zip(kernel_indices, vals):
                self._queued_syn_ids.append(Kernel_List_Entry[i][j])
                self._queued_ticks.append(time_step)
                self._queued_vals.append(float(v))

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

    def invert_dict(self, input_dict, spike_value=100):
        inverted = defaultdict(dict)
        
        for key, values in input_dict.items():
            for v in values:
                inverted[v][key] = spike_value
                
        return dict(inverted)
    def Full_Convolution_And_Extraction(self, Layer_idx, SpikeData,Total_Sim_Time):
        self._queued_syn_ids = []
        self._queued_ticks = []
        self._queued_vals = []
        print(SpikeData)
        start=time.time()        
        for time_step in SpikeData:
            for kernel_array, kernel_neuron in self.layers[Layer_idx]:
                offsets=self.all_relative_coordinate_grids(len(kernel_array),len(kernel_array[0]))
                for spike in SpikeData[time_step]:
                    self.Convolve_Spike(spike, kernel_array, offsets, SpikeData[time_step], time_step, 1)
        end=time.time()
        print(end-start)
        Spike_Times={}
        print('Full Convolution', Layer_idx)
        # Flush all convolution spikes to GPU queue
        if self._queued_syn_ids:
            print('entered if')
            self.model.queue_spikes(self._queued_syn_ids, self._queued_ticks, self._queued_vals)
            print('exited if')
        start=time.time()
        self.model.simulate(ticks=Total_Sim_Time, update_data_ticks=Total_Sim_Time)
        end=time.time()
        print(end-start)
        print('simulation done')
        
        for index_i in range(len(self.pooling_matrix[Layer_idx])):
            for index_j in range(len(self.pooling_matrix[Layer_idx][0])):
                if self.pooling_matrix[Layer_idx][index_i][index_j]:
                    Coor=(index_i,index_j)
                    Spike_Times[Coor]=self.model.get_spike_times(soma_id=self.pooling_matrix[Layer_idx][index_i][index_j])
        Spike_Times=self.invert_dict(Spike_Times)
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
            self.model.reset()
        #need to grab the spikes from this layer 
        # reset local spike queues
        self._queued_syn_ids = []
        self._queued_ticks = []
        self._queued_vals = []

        for time in Dataset:
            for Coor in Dataset[time]:
                queued_id = self.FF[0][Coor[0]+Coor[1]]
                self._queued_syn_ids.append(queued_id)
                self._queued_ticks.append(time)
                self._queued_vals.append(10.0)                 
        Spike_Times=[]
        # Flush FF network spikes to GPU queue
        if self._queued_syn_ids:
            self.model.queue_spikes(self._queued_syn_ids, self._queued_ticks, self._queued_vals)

        self.model.simulate(ticks=Total_Sim_Time, update_data_ticks=Total_Sim_Time)

        for output_neuron in self.FF[1]:
                
            Values=self.model.get_spike_times(soma_id=output_neuron)
            Spike_Times.append((len(Values),Values))
        
        max_index = max(range(len(Spike_Times)), key=lambda i: Spike_Times[i][0])
        return max_index

if __name__ == '__main__':    #Create Convolutionary NN
    Model = Conv2dtNet()
    Model.model.setup(use_gpu=True)
    print('gpu set up')
    #make sure I am not loading the entire thing use tonic. 
    Conv_Kernel_List = [
         [(3,3),(2,3),(3,2),(4,4)],              
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
    before_reset_weight = before_reset_hyperparameters[0]
    print('constructed')
    Dataset = './superneuroabm/ssn/data/NMNIST/Test/0/00011.bin'
    Ans = Model.ForwardPass(Dataset, 100)
    after_reset_hyperparameters = Model.model.get_agent_property_value(
                id=Model.kernel_vals[2][1][1][1][1],
                property_name="hyperparameters"
            )
    after_reset_weight = after_reset_hyperparameters[0]
    print(before_reset_weight,'before reset')
    print(after_reset_weight,'after reset')