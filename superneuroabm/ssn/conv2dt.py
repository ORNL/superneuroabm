
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

    def forward(self, x):
        """
        x: list of list of spike events:
        [
          [(tick, (x,y))],  # for output channel 0
          [(tick, (x,y)), (tick, (x,y)), ...],  # for output channel 1
          ...
        ]
        """
        if len(x) != self.out_channels:
            raise ValueError(f"Expected input for {self.out_channels} output channels")

        for out_ch, spikes_per_channel in enumerate(x):
            x_coord = [spike[1][0] for spike in spikes_per_channel]
            y_coord = [spike[1][1] for spike in spikes_per_channel]

            spikes_grouped_by_what = [[], []]
            for spike in spikes_grouped_by_what:
                tick, _ = spike
        



            for synapse in self.synapses[out_ch]:

                # Add spike to the synapse
                self.model.add_spike(
                    synapse_id=synapse,
                    tick=tick,
                    value=1.0
                )

        self.model.simulate(ticks=self.ticks, update_data_ticks=1)

        output_spikes = []
        for out_ch in range(self.out_channels):
            spikes = self.model.get_spike_times(soma_id=self.somas[out_ch])
            output_spikes.append(spikes)

        return output_spikes
