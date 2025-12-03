
import numpy as np
import os
import sys
# # Add paths after ensuring correct working directory
# sagesim_path = '/home/xxz/SAGESim'
# if sagesim_path not in sys.path:
#     sys.path.insert(0, sagesim_path)

# # Set up path for SuperNeuroABM
# superneuroabm_path = '/home/xxz/superneuroabm'

# working_dir = '/home/xxz/superneuroabm/superneuroabm/ssn'
# if os.getcwd() != working_dir:
#     os.chdir(working_dir)


# # Add SuperNeuroABM path to sys.path
# if superneuroabm_path not in sys.path:
#     sys.path.insert(0, superneuroabm_path)

from superneuroabm.model import NeuromorphicModel


class Conv2dT():
    """
    Neuromorphic Convolution Layer supporting multiple output channels.
    """

    def __init__(
        self,
        ticks,
        in_channels,
        out_channels,
        kernel_size,
        input_size,
        stride=1,
        padding=0,
        output_padding=0,
        bias=True,
    ):

        if not isinstance(kernel_size, tuple):
            kernel_size = (kernel_size, kernel_size)
        
        if not isinstance(input_size, tuple):
            input_size = (input_size, input_size)

        self.model = NeuromorphicModel()
        self.ticks = ticks

        self.kernel_size = kernel_size
        self.input_size = input_size
        self.out_channels = out_channels
        self.stride = stride
        self.num_synapses_per_soma = kernel_size[0] * kernel_size[1]
        
        # Calculate output size based on convolution formula
        self.output_size = (
            (input_size[0] - kernel_size[0]) // stride + 1,
            (input_size[1] - kernel_size[1]) // stride + 1
        )

        # Configuration is loaded automatically by the model using config names

        # Create kernel somas (one per channel) with input synapses
        self.kernel_somas = []
        self.input_synapses = []

        for _ in range(out_channels):
            soma = self.model.create_soma(
                breed="izh_soma",
                config_name="config_0",
            )
            self.kernel_somas.append(soma)

            channel_synapses = []
            for _ in range(self.num_synapses_per_soma):
                synapse = self.model.create_synapse(
                    breed="single_exp_synapse",
                    pre_soma_id=np.nan,  # External input
                    post_soma_id=soma,  # Connected to the kernel soma
                    config_name="no_learning_config_0",
                )
                channel_synapses.append(synapse)
            self.input_synapses.append(channel_synapses)

        # Create output somas (one per spatial location per channel)
        self.output_somas = []
        self.output_synapses = []
        
        for out_ch in range(out_channels):
            channel_output_somas = []
            channel_output_synapses = []
            
            for y in range(self.output_size[0]):
                for x in range(self.output_size[1]):
                    # Create output soma for this spatial location
                    output_soma = self.model.create_soma(
                        breed="izh_soma",
                        config_name="config_0",
                    )
                    channel_output_somas.append(output_soma)
                    
                    # Create synapse connecting kernel soma to output soma
                    output_synapse = self.model.create_synapse(
                        breed="single_exp_synapse",
                        pre_soma_id=self.kernel_somas[out_ch],
                        post_soma_id=output_soma,
                        config_name="no_learning_config_0",
                    )
                    channel_output_synapses.append(output_synapse)
            
            self.output_somas.append(channel_output_somas)
            self.output_synapses.append(channel_output_synapses)

        # Initialize the simulation environment
        self.model.setup(use_gpu=True)

    def forward(self, input_spikes, stride=1):
        """
        Perform proper spatial convolution on DVS camera spike events.
        
        Args:
            input_spikes: List of spike events [(tick, (x, y))]
            stride: Convolution stride (default=1)
            
        Returns:
            List of output spikes for each output channel
        """
        # Group spikes by spatial position for efficient lookup
        spike_map = {}
        for spike in input_spikes:
            tick, (x, y) = spike
            if (x, y) not in spike_map:
                spike_map[(x, y)] = []
            spike_map[(x, y)].append(tick)
        
        # Calculate maximum valid output positions based on input size
        max_out_x = self.input_size[1] - self.kernel_size[1]
        max_out_y = self.input_size[0] - self.kernel_size[0]
        
        # Process each spike position (sparse optimization)
        for (input_x, input_y), spike_times in spike_map.items():
            # For each kernel position, calculate which output position this spike contributes to
            for ky in range(self.kernel_size[0]):
                for kx in range(self.kernel_size[1]):
                    # Calculate output position for this kernel position
                    out_x = input_x - kx
                    out_y = input_y - ky
                    
                    # Only process if output position is within all boundaries
                    if (out_x >= 0 and out_y >= 0 and 
                        out_x <= max_out_x and out_y <= max_out_y and 
                        out_x % stride == 0 and out_y % stride == 0):
                        synapse_idx = ky * self.kernel_size[1] + kx
                        
                        # Add spikes only to input synapses of kernel somas
                        for out_ch in range(self.out_channels):
                            for tick in spike_times:
                                self.model.add_spike(
                                    synapse_id=self.input_synapses[out_ch][synapse_idx],
                                    tick=tick,
                                    value=1.0
                                )

        # Run simulation
        self.model.simulate(ticks=self.ticks, update_data_ticks=1)

        # Collect output spikes with spatial information
        output_spikes = []
        for out_ch in range(self.out_channels):
            channel_spikes = []
            for y in range(self.output_size[0]):
                for x in range(self.output_size[1]):
                    soma_idx = y * self.output_size[1] + x
                    spikes = self.model.get_spike_times(soma_id=self.output_somas[out_ch][soma_idx])
                    # Convert to spatial format: [(tick, (x, y))]
                    spatial_spikes = [(tick, (x, y)) for tick in spikes]
                    channel_spikes.extend(spatial_spikes)
            output_spikes.append(channel_spikes)

        return output_spikes



if __name__ == "__main__":
    import os
    import sys
    import numpy as np
    import matplotlib.pyplot as plt

    from superneuroabm.model import NeuromorphicModel
    from superneuroabm.ssn.conv2dt import Conv2dT
    ticks = 100
    kernel_size = (3, 3)
    out_channels = 1
    in_channels = 1

    conv_layer = Conv2dT(
        ticks=ticks,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        input_size=(10, 10),

    )

    # Create input spike at position that activates all 9 synapses
    input_spikes = [(1, (5, 5))]
    print(f"Input spikes: {input_spikes}")

    # Process through convolution
    output_spikes = conv_layer.forward(input_spikes)
    print(f"Output spikes: {output_spikes}")