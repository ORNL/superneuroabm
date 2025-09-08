
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
        # Global properties from tests/logic_gates_test_lif.py:test_dual_external_synapses_dual_somas
        self.model.register_global_property("dt", 1e-3)  # Updated from test file (was 1e-1)
        self.model.register_global_property("I_bias", 0)

        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.num_synapses_per_soma = kernel_size[0] * kernel_size[1]

        # Default Izhikevich soma parameters from tests/logic_gates_test_lif.py:test_dual_external_synapses_dual_somas
        if soma_parameters is None:
            # IZH soma parameters: [k, vthr, C, a, b, vpeak, vrest, d, vreset, I_in]
            k = 1.2
            vthr = -45
            C = 150
            a = 0.01
            b = 5
            vpeak = 50
            vrest = -75
            d = 130
            vreset = -56
            I_in = 350  # Updated from test file
            soma_parameters = [k, vthr, C, a, b, vpeak, vrest, d, vreset, I_in]
        
        # Initial state: [v, u] for Izhikevich model
        v = -75  # vrest
        u = 0
        default_internal_state = [v, u]

        # Default synapse parameters from tests/logic_gates_test_lif.py:test_dual_external_synapses_dual_somas
        if synapse_parameters is None:
            # Single_Exp_Synapse parameters: [weight, synaptic_delay, scale, tau_fall, tau_rise]
            weight = 1.0
            synaptic_delay = 1.0
            scale = 1.0
            tau_fall = 1e-2  # Updated from test file (was 1e-3)
            tau_rise = 0     # Updated from test file (was 1e-4)
            synapse_parameters = [weight, synaptic_delay, scale, tau_fall, tau_rise]
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
                    breed="Single_Exp_Synapse",
                    pre_soma_id=np.nan,
                    post_soma_id=soma,
                    parameters=synapse_parameters,
                    default_internal_state=synapse_internal_state,
                )
                channel_synapses.append(synapse)
            self.synapses.append(channel_synapses)

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
    print("Conv2dT Layer - Neuromorphic 2D Convolution")
    print("=" * 50)
    print("This module implements a neuromorphic convolution layer.")
    print("For examples and tutorials, see:")
    print("  - tutorials/conv2dt/basic_examples.py")
    print("  - tutorials/conv2dt/educational_examples.py")
    print("  - tutorials/conv2dt/README.md")
    print("=" * 50)
