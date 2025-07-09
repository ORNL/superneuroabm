
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
        soma_parameters=None,
        synapse_parameters=None
    ):
        super().__init__()

        if not isinstance(kernel_size, tuple):
            kernel_size = (kernel_size, kernel_size)

        self.model = model
        self.ticks = ticks
        self.model.register_global_property("dt", 1e-1)
        self.model.register_global_property("I_bias", 0)

        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.num_synapses_per_soma = kernel_size[0] * kernel_size[1]

        # Default soma parameters
        if soma_parameters is None:
            soma_parameters = [
                1.2, -45, 150, 0.01, 5, 50, -75, 130, -56, 450
            ]
        default_internal_state = [-75, 0]

        # Default synapse parameters
        if synapse_parameters is None:
            synapse_parameters = [
                1.0, 1.0, 1.0, 1e-3, 0
            ]
        synapse_internal_state = [0.0]

        # Create multiple somas and synapses
        self.somas = []
        self.synapses = []

        for out_ch in range(out_channels):
            soma = self.model.create_soma(
                breed="IZH_Soma",
                parameters=soma_parameters,
                default_internal_state=default_internal_state,
            )
            self.somas.append(soma)

            channel_synapses = []
            for _ in range(self.num_synapses_per_soma):
                synapse = self.model.create_synapse(
                    breed="Single_Exp_Synapse_STDP1",
                    pre_soma_id=np.nan,
                    post_soma_id=soma,
                    parameters=synapse_parameters,
                    default_internal_state=synapse_internal_state,
                )
                channel_synapses.append(synapse)
            self.synapses.append(channel_synapses)

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
    # Example usage with DVS camera data
    from superneuroabm.model import NeuromorphicModel
    
    # Create a NeuromorphicModel with proper setup
    model = NeuromorphicModel()
    model.setup(use_gpu=False)
    
    # Create a 3x3 convolution layer with 2 output channels
    conv_layer = Conv2dT(
        model=model,
        ticks=100,
        in_channels=1,
        out_channels=2,
        kernel_size=(3, 3)
    )
    
    # Example DVS spike data: [(tick, (x, y))]
    dvs_spikes = [
        (10, (0, 0)), (12, (1, 0)), (15, (2, 0)),
        (20, (0, 1)), (22, (1, 1)), (25, (2, 1)),
        (30, (0, 2)), (32, (1, 2)), (35, (2, 2))
    ]
    
    # Process spikes through convolution (no input shape needed!)
    print("Using sparse spatial convolution:")
    output_spikes_spatial = conv_layer.forward(dvs_spikes, stride=1)
    for i, channel_spikes in enumerate(output_spikes_spatial):
        print(f"Channel {i}: {len(channel_spikes)} spikes")
        print(f"  Spike times: {channel_spikes}")
