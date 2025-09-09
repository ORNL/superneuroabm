
import numpy as np
import torch.nn as nn

class Conv2dT(nn.Module):
    """
    Neuromorphic Convolution Layer supporting multiple output channels.
    """

    def __init__(
        self,
        model,
        ticks,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        bias=True,
    ):
        super().__init__()

        if not isinstance(kernel_size, tuple):
            kernel_size = (kernel_size, kernel_size)

        self.model = model
        self.ticks = ticks
        # Set global simulation parameters like in the test
        self.model.register_global_property("dt", 1e-3)  # Time step
        self.model.register_global_property("I_bias", 0)  # No bias current

        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.num_synapses_per_soma = kernel_size[0] * kernel_size[1]

        # Configuration is loaded automatically by the model using config names

        # Create multiple somas and synapses
        self.somas = []
        self.synapses = []

        for _ in range(out_channels):
            soma = self.model.create_soma(
                breed="izh_soma",
                config_name="config_0",
            )
            self.somas.append(soma)

            channel_synapses = []
            for _ in range(self.num_synapses_per_soma):
                synapse = self.model.create_synapse(
                    breed="single_exp_synapse",
                    pre_soma_id=np.nan,  # External input
                    post_soma_id=soma,  # Connected to the created soma
                    config_name="no_learning_config_0",
                )
                channel_synapses.append(synapse)
            self.synapses.append(channel_synapses)

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
        
        # Process each spike position (sparse optimization)
        for (input_x, input_y), spike_times in spike_map.items():
            # For each kernel position, calculate which output position this spike contributes to
            for ky in range(self.kernel_size[0]):
                for kx in range(self.kernel_size[1]):
                    # Calculate output position for this kernel position
                    out_x = input_x - kx
                    out_y = input_y - ky
                    
                    # Only process if output position is valid (non-negative)
                    if out_x >= 0 and out_y >= 0 and out_x % stride == 0 and out_y % stride == 0:
                        synapse_idx = ky * self.kernel_size[1] + kx
                        
                        # Add spikes to all output channels
                        for out_ch in range(self.out_channels):
                            for tick in spike_times:
                                self.model.add_spike(
                                    synapse_id=self.synapses[out_ch][synapse_idx],
                                    tick=tick,
                                    value=1.0
                                )

        # Run simulation
        self.model.simulate(ticks=self.ticks, update_data_ticks=1)

        # Collect output spikes
        output_spikes = []
        for out_ch in range(self.out_channels):
            spikes = self.model.get_spike_times(soma_id=self.somas[out_ch])
            output_spikes.append(spikes)

        return output_spikes


if __name__ == "__main__":
    import sys
    import os

    import numpy as np
    import matplotlib.pyplot as plt

    sagesim_path = '/home/xxz/SAGESim'
    if sagesim_path not in sys.path:
        sys.path.insert(0, sagesim_path)

    # test if sagesim is correctly imported
    try:
        import sagesim
        print("✓ SAGESim imported successfully")
    except ImportError as e:
        print("⚠️  Warning: SAGESim not found. Please ensure SAGESim is installed")

    # Set up path for SuperNeuroABM
    superneuroabm_path = '/home/xxz/superneuroabm'

    # Add SuperNeuroABM path to sys.path
    if superneuroabm_path not in sys.path:
        sys.path.insert(0, superneuroabm_path)



    # Ensure we're running from the tutorials folder
    current_dir = os.getcwd()
    if not current_dir.endswith('tutorials'):
        # Check if we're in the superneuroabm directory
        if 'superneuroabm' in current_dir:
            tutorials_dir = os.path.join(current_dir, 'tutorials') if current_dir.endswith('superneuroabm') else os.path.join(os.path.dirname(current_dir), 'superneuroabm', 'tutorials')
            if os.path.exists(tutorials_dir):
                os.chdir(tutorials_dir)
                print(f"✓ Changed working directory to: {tutorials_dir}")
            else:
                print(f"⚠️  Warning: tutorials directory not found. Current dir: {current_dir}")
        else:
            print(f"⚠️  Warning: Please run this notebook from the tutorials folder")
            print(f"   Current directory: {current_dir}")
            print(f"   Expected: .../superneuroabm/tutorials/")
    else:
        print(f"✓ Running from tutorials directory: {current_dir}")



    from superneuroabm.model import NeuromorphicModel
    from superneuroabm.ssn.conv2dt import Conv2dT
    model = NeuromorphicModel()
    model.register_global_property("dt", 1e-3)  # Time step (100 μs)
    model.register_global_property("I_bias", 0)
    
    ticks = 200
    out_channels = 1
    kernel_size = (3, 3)
    conv_layer = Conv2dT(
        model=model,
        ticks=ticks,
        in_channels=1,
        out_channels=out_channels,
        kernel_size=kernel_size
    )

    print(f"Created Conv2dT layer:")
    print(f"  Kernel size: {conv_layer.kernel_size}")
    print(f"  Output channels: {conv_layer.out_channels}")
    print(f"  Synapses per output neuron: {conv_layer.num_synapses_per_soma}")
    print(f"  Simulation ticks: {conv_layer.ticks}")


    input_spikes = [(10, (10, 10))]

    output_spikes = conv_layer.forward(input_spikes, stride=1)
    print(f"Output spikes: {output_spikes[0]}")