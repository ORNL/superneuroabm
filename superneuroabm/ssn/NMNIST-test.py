import os 
import numpy as np
from superneuroabm.model import NeuromorphicModel
import sagesim
class Conv2dtSingleLayer:
    def __init__(self,input_x,input_y,SpikeData= None):
        #Spike Data should be organized by spike time
        #SpikeData[t]=[(xi,yi,spike val),(xj,yj,spike val)....] all incoming spikes
        self.model=NeuromorphicModel()
        self.input_x=input_x
        self.input_y=input_y
        self.SpikeData=SpikeData
        self.ConvKernelList=[]
    def HashSpikes(TimeStep):
        CurrentSpikeSet={}
        for i in self.SpikeData[TimeStep]:
            CurrentSpikeSet[(i[0],i[1])]=i[2]
        return CurrentSpikeSet
    def Conv_Kernel_Construction(self,W,H):
        #W and H represent the width and height of the conv kernel
        #currently one kernel neuron. This can be expanded and should
        #expand so that there is an input argument which taks
        Kernel=np.empty((W, H),dtype=object)
        Kernel_Neuron=self.model.create_soma(
            breed='lif_soma',
            config_name='config_0',
            hyperparameters_overrides={'C':10e-9,'R':1e6,'vthr':-45,'tref':5e-3,'vrest':-60,'vreset':-60,'tref_allows_integration':1,'I_in':0,'scaling_factor':1e-5},
            default_internal_state_overrides={'v':-60,'tcount':0.0,'tlast':0.0}
            )
        for i in range(W):
            for j in range(H):
                Synapse= self.model.create_synapse(
                    breed='single_exp_synapse',
                    pre_soma_id=np.nan,
                    post_soma_id=Kernel_Neuron,
                    config_name='exp_pair_wise_stdp_config_0',
                    hyperparameters_overrides={'weight':np.random.uniform(low=50.0, high=100.0),'synpatic_delay':1.0,'scale':1.0,'tau_fall':1e-2,'tau_rise':0},
                    default_internal_state_overrides={'internal_state':0.0},
                    learning_hyperparameters_overrides={'stdp_type':10e-3,'tau_pre_stdp':10e-3,'tau_post_stdp':10e-3,'a_exp_pre':0.005,'a_exp_post':0.005,'stdp_history_length':100},
                    default_internal_learning_state_overrides={'pre_trace':0,'post_trace':0,'dw':0}
                )
                Kernel[i][j]=Synapse
        self.ConvKernelList.append((Kernel,Kernel_Neuron))

    def Load_File(self,path):
        #Takes in NMNIST data set and converts to usable form
        #Specifically converts data into (X,Y,Spike,Time) set
        with open(path, 'rb') as f:
            binary_data = f.read()
        events_list = []  # start empty

        for i in range(0, len(binary_data), 5):
            if i+4 >= len(binary_data):  # safety check
                break

            x = binary_data[i]
            y = binary_data[i+1]
            b3 = binary_data[i+2]
            b4 = binary_data[i+3]
            b5 = binary_data[i+4]

            # polarity bit = MSB of b3
            polarity = b3 >> 7  # 0=OFF,1=ON

            # timestamp bits: lower 7 bits of b3 + all of b4 + all of b5
            timestamp = ((b3 & 0x7F) << 16) | (b4 << 8) | b5

            # append as a tuple (or dict) to the list
            events_list.append((x, y, polarity, timestamp))

        return events_list
    
    def Add_Output_Channels(Number_Of_Output_Channels): 
        self.output_layer={}
        for i in self.ConvKernelList:
            output_list = []
            for output in Number_Of_Output_Channels:
                Output_=self.model.create_soma(
                    breed='lif_soma',
                    config_name='config_0',
                    hyperparameters_overrides={'C':10e-9,'R':1e6,'vthr':-45,'tref':5e-3,'vrest':-60,'vreset':-60,'tref_allows_integration':1,'I_in':0,'scaling_factor':1e-5},
                    default_internal_state_overrides={'v':-60,'tcount':0.0,'tlast':0.0}
                    )
                Synapse= self.model.create_synapse(
                    breed='single_exp_synapse',
                    pre_soma_id=i[1],
                    post_soma_id=Output,
                    config_name='exp_pair_wise_stdp_config_0',
                    hyperparameters_overrides={'weight':np.random.uniform(low=50.0, high=100.0),'synpatic_delay':1.0,'scale':1.0,'tau_fall':1e-2,'tau_rise':0},
                    default_internal_state_overrides={'internal_state':0.0},
                    learning_hyperparameters_overrides={'stdp_type':10e-3,'tau_pre_stdp':10e-3,'tau_post_stdp':10e-3,'a_exp_pre':0.005,'a_exp_post':0.005,'stdp_history_length':100},
                    default_internal_learning_state_overrides={'pre_trace':0,'post_trace':0,'dw':0}
                )
                output_list.append(Output)
            self.output_layer[i]=output_list
        
    
    def Convolve_Spike(SpikeCoordinate, Kernel_List_Entry,CurrentSpikeSet, time_step, Stride):
        #Kernel_List_Entry is an Entry for the kernel we want to convolve on
        Ranges=[]
        for i in range(len(Kernel_List_Entry)):
            for j in range(len(Kernel_List_Entry[0])):
                Ranges_X=(SpikeCoordinate[0]-i, SpikeCoordinate[0]-i + len(Kernel_List_Entry))
                Ranges_Y=(SpikeCoordinate[1]-j,SpikeCoordinate[1]-j + len(Kernel_List_Entry[0]))
                Range = (Ranges_X, Ranges_Y)
                Ranges.append(Range)
        for x in range(Ranges[0][0],Ranges[0][1],Stride):
            for y in range(Ranges[1][0],Ranges[1][1]):
                coor=(x,y)
                if coor in CurrentSpikeSet:
                    kernel_location_x= x - Ranges[0][0]
                    kernel_location_y = y- Ranges[1][0]
                    self.model.add_spike(synapse_id=Kernel_List_Entry[kernel_location_x][kernel_location_y],tick=time_step,value=CurrentSpikeSet[coor])
    
if __name__=='__main__':
    Model=Conv2dtSingleLayer(2,3)
  
    Test_Neuron=Model.model.create_soma(
            breed='lif_soma',
            config_name='config_0',
            hyperparameters_overrides={'C':10e-9,'R':1e6,'vthr':-45,'tref':5e-3,'vrest':-60,'vreset':-60,'tref_allows_integration':1,'I_in':0,'scaling_factor':1e-5},
            default_internal_state_overrides={'v':-60,'tcount':0.0,'tlast':0.0}
            )


    Model.Conv_Kernel_Construction(3,3)
    Model.Conv_Kernel_Construction(4,4)
    Model.model.setup(use_gpu=True)    
    print(Model.ConvKernelList[0][0],'Conv Kernel')
    print(Model.ConvKernelList[1][0],'Conv Kernel')

    for i in range(len(Model.ConvKernelList[0][0])):
        for j in range(len(Model.ConvKernelList[0][0][0])):
            print('adding Spike')
            Model.model.add_spike(synapse_id=Model.ConvKernelList[0][0][i][j],tick=np.random.randint(10, 20),value=1)
            
    Model.model.simulate(ticks=100,update_data_ticks=100)
    internal_states_history_soma0 = np.array(
            Model.model.get_internal_states_history(agent_id=Model.ConvKernelList[1][1])
        )
    np.set_printoptions(suppress=True, precision=6)
    print(f"Internal states history from Soma 1: {internal_states_history_soma0}")
    internal_states_history_soma0 = np.array(
            Model.model.get_internal_states_history(agent_id=Model.ConvKernelList[0][1])
        )
    np.set_printoptions(suppress=True, precision=6)
    print(f"Internal states history from synapse 0: {internal_states_history_soma0}")
