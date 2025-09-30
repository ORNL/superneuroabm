import os
import numpy as np
from superneuroabm.model import NeuromorphicModel
import sagesim
import tonic 


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

    def HashSpikes(self, TimeStep):
        CurrentSpikeSet = {}
        for i in self.SpikeData[TimeStep]:
            CurrentSpikeSet[(i[0], i[1])] = i[2]
        return CurrentSpikeSet

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

    def Add_Output_Channels(self, layer_idx, num_channels):
        """
        Add output channels for all filters in a given layer.
        """
        if layer_idx not in self.layers:
            raise ValueError(f"No convolutional filters found in layer {layer_idx}")

        output_layer = {}
        for kernel_array, kernel_neuron in self.layers[layer_idx]:
            output_list = []
            for _ in range(num_channels):
                Output = self.model.create_soma(
                    breed='lif_soma',
                    config_name='config_0',
                    hyperparameters_overrides={'C': 1e-8, 'R': 1e6, 'vthr': -45, 'tref': 5e-3,
                                               'vrest': -60, 'vreset': -60, 'tref_allows_integration': 1,
                                               'I_in': 0, 'scaling_factor': 1e-5},
                    default_internal_state_overrides={'v': -60, 'tcount': 0.0, 'tlast': 0.0}
                )
                Synapse = self.model.create_synapse(
                    breed='single_exp_synapse',
                    pre_soma_id=kernel_neuron,
                    post_soma_id=Output,
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
                output_list.append(Output)
            output_layer[kernel_neuron] = output_list

        self.pooling_layers[layer_idx] = output_layer

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

        # iterate over the output channel lists, not just the keys
        for output_channel in self.pooling_layers[layer_idx].values():
            pooling_neuron = self.model.create_soma(
                breed='lif_soma',
                config_name='config_0',
                hyperparameters_overrides={'C': 1e-8, 'R': 1e6, 'vthr': -45, 'tref': 5e-3,
                                        'vrest': -60, 'vreset': -60, 'tref_allows_integration': 1,
                                        'I_in': 0, 'scaling_factor': 1e-5},
                default_internal_state_overrides={'v': -60, 'tcount': 0.0, 'tlast': 0.0}
            )

            # connect all neurons in this channel to the pooling neuron
            for Output_Neuron in output_channel:
                self.model.create_synapse(
                    breed='single_exp_synapse',
                    pre_soma_id=Output_Neuron,
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

        # arrange them in a matrix
        pooling_spikes = np.empty((W, H), dtype=object)
        idx = 0
        for x in range(W):
            for y in range(H):
                if idx < len(pooling_neurons):
                    pooling_spikes[x][y] = pooling_neurons[idx]
                    idx += 1

        # store by layer index (not +1, unless you really want offset)
        self.pooling_matrix[layer_idx] = pooling_spikes

    def Collect_Spikes(self, Layer_Idx):
        Pooling_Spike_Data = {}
        Spiking_Neurons = self.pooling_matrix[Layer_Idx]

        for x in range(len(Spiking_Neurons)):
            for y in range(len(Spiking_Neurons[0])):
                if Spiking_Neurons[x][y]:
                    Neuron_Spike_List = self.model.get_spike_times(soma_id=Spiking_Neurons[x][y])
                    for Spike_Time in Neuron_Spike_List:
                        if Spike_Time not in Pooling_Spike_Data:
                            Pooling_Spike_Data[Spike_Time] = {(x, y): 1}
                        else:
                            Pooling_Spike_Data[Spike_Time][(x, y)] = 1

        return Pooling_Spike_Data

        
    def ForwardPass(self, SpikeData):
        #TODO 
        #Collect Spikes at each layer and pass through
        '''
        We are going to process all the spikes through and they will be collected an reprocessed at each max pooling layer
        '''
        return None


if __name__ == '__main__':
    Model = Conv2dtSingleLayer(5, 5)

    # add conv filters to two different layers
    #
    Model.Conv_Kernel_Construction(3, 3, layer_idx=0)
    Model.Conv_Kernel_Construction(4, 4, layer_idx=0)
    Model.Add_Output_Channels(layer_idx=0, num_channels=5)
    Model.MaxPooling(0,2,1)

    Model.Conv_Kernel_Construction(1, 1, layer_idx=1)
    Model.Add_Output_Channels(layer_idx=1, num_channels=2)

    Model.MaxPooling(1,1,1)
    
    #Initial Spiking Pattern
    CurrentSpikes = {1:{(1, 5): 10, (3, 4): 10, (5, 5): 10, (2, 2): 10, (3, 1): 10},2:{(1, 4): 10, (3, 3): 10, (5, 4): 10, (2, 1): 10, (1, 1): 10},
                    3:{(2, 4): 10, (1, 3): 10, (5, 5): 10, (1, 1): 10, (1, 3): 10}}

    print(Model.pooling_matrix[0])
    Model.model.setup(use_gpu=True)

    
    for kernel_array, kernel_neuron in Model.layers[0]:
        for spike_time in CurrentSpikes:
            for spike in CurrentSpikes[spike_time]:
                Model.Convolve_Spike(spike, kernel_array, CurrentSpikes[spike_time], spike_time, 1)
    
    Model.model.simulate(ticks=10, update_data_ticks=10)

    Spike_Data=Model.Collect_Spikes(0)
    print(Spike_Data)
'''
    # read from first layer output neuron 0
    first_layer_outputs = Model.pooling_layers[0]
    first_kernel_neuron = list(first_layer_outputs.keys())[0]
    soma0 = first_layer_outputs[first_kernel_neuron][0]
    internal_spike_times=Model.model.get_spike_times(soma_id=soma0)
    print(f"Soma 0 internal spike times:' {internal_spike_times}")
    internal_states_history_soma0 = np.array(
        Model.model.get_internal_states_history(agent_id=soma0)
    )
    print(f"Soma 0 internal states: {internal_states_history_soma0}")
    '''
