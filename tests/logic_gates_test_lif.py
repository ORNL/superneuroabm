import unittest

import numpy as np
import csv

from superneuroabm.model import NeuromorphicModel
from matplotlib import pyplot as plt

class LogicGatesTestLIF(unittest.TestCase):
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
        self._model.register_global_property("dt", 1e-1)
        self._model.register_global_property("I_bias", 0)
        # Create soma
        C = 10e-9 # Capacitance in Farads
        R = 1e12 # Resistance in Ohms
        vthr = -45
        tref = 5e-3 # refractory period
        vrest = -60 # resting potential   
        vreset = -60 # reset potential
        tref_allows_integration = 1 # whether to allow integration during refractory period
        I_in = 4e-8
        soma_parameters = [C, R,vthr, tref, vrest, vreset, tref_allows_integration, I_in]
        v = vrest
        tcount = 0
        tlast = 0
        default_internal_state = [v, tcount, tlast]
        soma_0 = self._model.create_soma(
            breed="LIF_Soma",
            parameters=soma_parameters,
            default_internal_state=default_internal_state,
        )
        soma_1 = self._model.create_soma(
            breed="LIF_Soma",
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
            breed="Single_Exp_Synapse_STDP1",
            pre_soma_id=np.nan,
            post_soma_id=soma_0,
            parameters=synapse_parameters,
            default_internal_state=synapse_internal_state,
        )
        syn_int = self._model.create_synapse(
            breed="Single_Exp_Synapse_STDP1",
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

        # spikes = [(9, 1), (5, 1), (4, 1)]
        # for spike in spikes:
        #    self._model.add_spike(soma_id=soma_1, tick=spike[0], value=spike[1])     #TODO: fix this

        self._model.simulate(ticks=100, update_data_ticks=100)

        minimum_expected_spikes = 2
        # print(self._model.get_spike_times(soma_id=soma_0))
        # print(self._model.get_spike_times(soma_id=soma_1))
        # print(self._model.get_internal_states_history(agent_id=soma_0))
        print(*self._model.get_internal_states_history(agent_id=soma_0))
        plt.figure(figsize=(5, 5))
        plt.plot(*self._model.get_internal_states_history(agent_id=soma_0), label='Membrane Potential of Soma 0')
        plt.ylabel('Mem. Pot. (mV)') 
        plt.xlabel('Time (ms)')
        plt.savefig("logic_gates_test_lif_test_two_soma_soma0.png")
        with open('output_LIF.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(self._model.get_internal_states_history(agent_id=soma_0)) 
        assert (
            len(self._model.get_spike_times(soma_id=soma_0)) >= minimum_expected_spikes
        ), f"Total number of spikes are {len(self._model.get_spike_times(soma_id=soma_0))} but should be at least {minimum_expected_spikes}"
        # expected_times = [2, 3, 4, 5, 9]
        # assert (
        #    self._model.get_spike_times(soma_id=soma_1) == expected_times
        # ), f"Spike times are {self._model.get_spike_times(soma_id=soma_1)} but should be {expected_times}"



class LogicGatesTestGPU(LogicGatesTestLIF):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self._model = NeuromorphicModel()
        self._use_gpu = True


if __name__ == "__main__":
    unittest.main()
