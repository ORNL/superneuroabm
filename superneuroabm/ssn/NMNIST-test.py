import os
import numpy as np
from superneuroabm.model import NeuromorphicModel
import sagesim
import tonic 
import math

class Conv2dtSingleLayer:
    def __init__(self, input_x, input_y, SpikeData=None):
        # Spike Data should be organized by spike time
        # SpikeData[t] = [(xi, yi, spike val), (xj, yj, spike val)....] all incoming spikes
        self.model = NeuromorphicModel()
        self.input_x = input_x
        self.input_y = input_y
        self.SpikeData = SpikeData

        # filters and pooling organized by layer
        self.layers = {}           # {layer_idx: [(Kernel, Kernel_Neuron), ...]}
        self.pooling_layers = {}   # {layer_idx: {kernel_neuron: [outputs]}}
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
        """
        Construct a convolutional kernel and assign it to a specific layer index.
        """
        Kernel = np.empty((W, H), dtype=object)
        Kernel_Neuron = self.model.create_soma(
            breed='lif_soma',
            config_name='config_0',
            hyperparameters_overrides={'C': 1e-8, 'R': 1e6, 'vthr': -45, 'tref': 5e-3,
                                       'vrest': -60, 'vreset': -60, 'tref_allows_integration': 1,
                                       'I_in': 0, 'scaling_factor': 1e-5},
            default_internal_state_overrides={'v': -60, 'tcount': 0.0, 'tlast': 0.0}
        )
        for i in range(W):
            for j in range(H):
                Synapse = self.model.create_synapse(
                    breed='single_exp_synapse',
                    pre_soma_id=input,
                    post_soma_id=Kernel_Neuron,
                    config_name='exp_pair_wise_stdp_config_0',
                    hyperparameters_overrides={'weight': np.random.uniform(50.0, 100.0),
                                               'synpatic_delay': 1.0, 'scale': 1.0,
                                               'tau_fall': 1e-2, 'tau_rise': 0},
                    default_internal_state_overrides={'internal_state': 0.0},
                    learning_hyperparameters_overrides={'stdp_type': 10e-3, 'tau_pre_stdp': 10e-3,
                                                        'tau_post_stdp': 10e-3, 'a_exp_pre': 0.005,
                                                        'a_exp_post': 0.005, 'stdp_history_length': 100},
                    default_internal_learning_state_overrides={'pre_trace': 0, 'post_trace': 0, 'dw': 0}
                )
                Kernel[i][j] = Synapse

        # store in proper layer
        if layer_idx not in self.layers:
            self.layers[layer_idx] = []
        self.layers[layer_idx].append((Kernel, Kernel_Neuron))


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
    def MaxPooling(self, layer_idx, W, H):
        
        pooling_neurons = []

        for kernel_array, kernel_neuron in self.layers[layer_idx]:
            # one pooling neuron per kernel output
            pooling_neuron = self.model.create_soma(
                breed='lif_soma',
                config_name='config_0',
                hyperparameters_overrides={'C': 1e-8, 'R': 1e6, 'vthr': -45, 'tref': 5e-3,
                                        'vrest': -60, 'vreset': -60, 'tref_allows_integration': 1,
                                        'I_in': 0, 'scaling_factor': 1e-5},
                default_internal_state_overrides={'v': -60, 'tcount': 0.0, 'tlast': 0.0}
            )

            # connect the *single* kernel neuron to the pooling neuron
            self.model.create_synapse(
                breed='single_exp_synapse',
                pre_soma_id=kernel_neuron,
                post_soma_id=pooling_neuron,
                config_name='exp_pair_wise_stdp_config_0',
                hyperparameters_overrides={'weight': np.random.uniform(50.0, 100.0),
                                        'synpatic_delay': 1.0, 'scale': 1.0,
                                        'tau_fall': 1e-2, 'tau_rise': 0},
                default_internal_state_overrides={'internal_state': 0.0},
                learning_hyperparameters_overrides={'stdp_type': 10e-3, 'tau_pre_stdp': 10e-3,
                                                    'tau_post_stdp': 10e-3, 'a_exp_pre': 0.005,
                                                    'a_exp_post': 0.005, 'stdp_history_length': 100},
                default_internal_learning_state_overrides={'pre_trace': 0, 'post_trace': 0, 'dw': 0}
            )

            pooling_neurons.append(pooling_neuron)

        # arrange pooling neurons in a matrix
        pooling_spikes = np.empty((W, H), dtype=object)
        idx = 0
        for x in range(W):
            for y in range(H):
                if idx < len(pooling_neurons):
                    pooling_spikes[x][y] = pooling_neurons[idx]
                    idx += 1

        self.pooling_matrix[layer_idx] = pooling_spikes

    

        
    def ForwardPass(self, SpikeData):
        #TODO 
        #Collect Spikes at each layer and pass through
        '''
        We are going to process all the spikes through and they will be collected an reprocessed at each max pooling layer
        '''
        return None

import tonic
import tonic.transforms as transforms
import tempfile

if __name__ == '__main__':

    Model = Conv2dtSingleLayer(34, 34)
    Model.model.setup(use_gpu=True)
    Model.Conv_Kernel_Construction(3, 3, layer_idx=0)
    Model.Conv_Kernel_Construction(4, 4, layer_idx=0)
    Model.Conv_Kernel_Construction(4, 4, layer_idx=0)
    Model.Conv_Kernel_Construction(5, 5, layer_idx=0)
    Model.MaxPooling(0,2,2)
    Dataset = Model.load_bin_as_spike_dict('/lustre/orion/proj-shared/lrn088/objective3/wfishell/superneuroabm/superneuroabm/ssn/data/NMNIST/Test/1/00003.bin')
    print(Dataset[5])

    for time in Dataset:
        for kernel_array, kernel_neuron in Model.layers[0]:
                for spike in Dataset[time]:
                    Model.Convolve_Spike(spike, kernel_array, Dataset[time], time, 1)
    
    Model.model.simulate(ticks=1300, update_data_ticks=1300)

    soma=Model.pooling_matrix[0][0][1]
    print(Model.model.get_spike_times(soma_id=soma))
    internal_states_history_soma = np.array(
            Model.model.get_internal_states_history(agent_id=soma)
        )
    print(internal_states_history_soma)

    
    