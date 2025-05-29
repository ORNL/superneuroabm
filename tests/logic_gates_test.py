import unittest

from superneuroabm.model import NeuromorphicModel


class LogicGatesTest(unittest.TestCase):
    """
    Tests SNN semantics by ensuring that basic logic
    gate functionality can be replicated.

    """

    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        # Create NeuromorphicModel
        self._model = NeuromorphicModel()
        self._use_gpu = False

    def test_two_somas(self):
        """Tests working of two somas"""

        # Create soma
        k = 1.2
        vthr = -45
        C = 150
        a = 0.01
        b = 5
        vpeak = 50
        vrest = -75
        d = 130
        vreset = -56
        soma_parameters = [k, vthr, C, a, b, vpeak, vrest, d, vreset]
        v = vrest
        u = 0
        default_internal_state = [v, u]
        soma_0 = self._model.create_soma(
            breed="IZH_Soma",
            parameters=soma_parameters,
            default_internal_state=default_internal_state,
        )
        soma_1 = self._model.create_soma(
            breed="IZH_Soma",
            parameters=soma_parameters,
            default_internal_state=default_internal_state,
        )

        # Create synapse
        weight = 1.0
        synaptic_delay = 1.0
        scale = 1.0
        tau_fall = 1e-3
        tau_rise = 0
        synapse_parameters = [
            weight,
            synaptic_delay,
            scale,
            tau_fall,
            tau_rise,
        ]
        I_synapse = 0.0
        synapse_internal_state = [I_synapse]
        syn_ext = self._model.create_synapse(
            pre_soma_id=Nan,
            post_soma_id=soma_0,
            parameters=synapse_parameters,
            default_internal_state=synapse_internal_state,
        )
        syn_int = self._model.create_synapse(
            pre_soma_id=soma_0,
            post_soma_id=soma_1,
            parameters=synapse_parameters,
            default_internal_state=synapse_internal_state,
        )

        # Setup and simulate
        self._model.setup(use_gpu=self._use_gpu)

        # Add spikes
        spikes = [(1, 1), (2, 1)]
        for spike in spikes:
            self._model.add_spike(synapse_id=syn_ext, tick=spike[0], value=spike[1])

        #spikes = [(9, 1), (5, 1), (4, 1)]
        #for spike in spikes:
        #    self._model.add_spike(soma_id=soma_1, tick=spike[0], value=spike[1])     #TODO: fix this

        self._model.simulate(ticks=10)

        expected_times = [1, 2]
        print(self._model.get_spike_times(soma_id=soma_0))
        print(self._model.get_spike_times(soma_id=soma_1))
        assert (
            self._model.get_spike_times(soma_id=soma_0) == expected_times
        ), f"Spike times are {self._model.get_spike_times(soma_id=soma_0)} but should be {expected_times}"
        #expected_times = [2, 3, 4, 5, 9]
        #assert (
        #    self._model.get_spike_times(soma_id=soma_1) == expected_times
        #), f"Spike times are {self._model.get_spike_times(soma_id=soma_1)} but should be {expected_times}"

    def test_or_gate(self):
        """Builds a neuromorphic circuit for OR gate and simulates it"""

        # Create somas
        input_0 = self._model.create_soma(threshold=0.0)
        input_1 = self._model.create_soma(threshold=0.0)
        output_2 = self._model.create_soma(threshold=0.0)

        # Create synapses
        self._model.create_synapse(
            pre_soma_id=input_0, post_soma_id=output_2, weight=1.0
        )
        self._model.create_synapse(
            pre_soma_id=input_1, post_soma_id=output_2, weight=1.0
        )

        # Setup and simulate
        self._model.setup(output_buffer_len=10, use_gpu=self._use_gpu)

        # test_cases in format time -> ([(soma, value), (soma, value)]
        test_cases = {
            1: [
                (input_0, 0),
                (input_1, 0),
            ],  # Input: (0, 0); Expected output: No spike
            3: [
                (input_0, 0),
                (input_1, 1),
            ],  # Input: (0, 1); Expected output: Spike at time 4
            5: [
                (input_0, 1),
                (input_1, 0),
            ],  # Input: (1, 0); Expected output: Spike at time 6
            7: [
                (input_0, 1),
                (input_1, 1),
            ],  # Input: (1, 1); Expected output: Spike at time 8
        }
        # Overall expected spike times for output_2 soma: [4, 6, 8]
        expected_times = [4, 6, 8]
        for time in test_cases:
            for soma, value in test_cases[time]:
                self._model.add_spike(soma_id=soma, tick=time, value=value)

        self._model.simulate(ticks=10)

        print(self._model.get_spike_times(soma_id=output_2))
        assert (
            self._model.get_spike_times(soma_id=output_2) == expected_times
        ), f"Spike times are {self._model.get_spike_times(soma_id=output_2)} but should be {expected_times}"

    def test_and_gate(self):
        """Builds a neuromorphic circuit for AND gate and simulates it"""

        # Create somas
        input_0 = self._model.create_soma(threshold=0.0)
        input_1 = self._model.create_soma(threshold=0.0)
        output_2 = self._model.create_soma(threshold=2.0)

        # Create synapses
        self._model.create_synapse(
            pre_soma_id=input_0, post_soma_id=output_2, weight=1.0
        )
        self._model.create_synapse(
            pre_soma_id=input_1, post_soma_id=output_2, weight=1.0
        )

        # Setup and simulate
        self._model.setup(output_buffer_len=10, use_gpu=self._use_gpu)

        # test_cases in format time -> ([(soma, value), (soma, value)]
        test_cases = {
            1: [
                (input_0, 0),
                (input_1, 0),
            ],  # Input: (0, 0); Expected output: No spike
            3: [
                (input_0, 0),
                (input_1, 1),
            ],  # Input: (0, 1); Expected output: Spike at time 4
            5: [
                (input_0, 1),
                (input_1, 0),
            ],  # Input: (1, 0); Expected output: Spike at time 6
            7: [
                (input_0, 1),
                (input_1, 1),
            ],  # Input: (1, 1); Expected output: Spike at time 8
        }
        # Overall expected spike times for output_2 soma: [8]
        expected_times = [8]
        for time in test_cases:
            for soma, value in test_cases[time]:
                self._model.add_spike(soma_id=soma, tick=time, value=value)

        self._model.simulate(ticks=10)
        print(self._model.get_spike_times(soma_id=output_2))
        assert (
            self._model.get_spike_times(soma_id=output_2) == expected_times
        ), f"Spike times are {self._model.get_spike_times(soma_id=output_2)} but should be {expected_times}"

        print(self._model.summary())


class LogicGatesTestGPU(LogicGatesTest):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self._model = NeuromorphicModel()
        self._use_gpu = True


if __name__ == "__main__":
    unittest.main()
