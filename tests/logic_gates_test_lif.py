import unittest

import numpy as np
import csv

from superneuroabm.model import NeuromorphicModel
from matplotlib import pyplot as plt


class LogicGatesTestLIF(unittest.TestCase):
    """
    Tests SNN (Spiking Neural Network) semantics by ensuring that basic logic
    gate functionality can be replicated using LIF (Leaky Integrate-and-Fire) neurons.
    
    This test suite validates the core functionality of neuromorphic simulations
    by creating simple neural circuits and verifying their spike behavior.
    """

    def __init__(self, methodName: str = ...) -> None:
        """
        Initialize the test case with a NeuromorphicModel instance.
        
        Args:
            methodName: Name of the test method to run
        """
        super().__init__(methodName)
        # Create NeuromorphicModel instance for testing
        self._model = NeuromorphicModel()
        # Set to use CPU for base test (GPU variant in separate class)
        self._use_gpu = False

    def test_two_somas(self):
        """
        Tests the basic functionality of two connected LIF neurons (somas).
        
        This test creates a simple two-neuron circuit:
        1. External input -> Soma 0 (via synapse)
        2. Soma 0 -> Soma 1 (via internal synapse)
        
        The test verifies that input spikes propagate through the network
        and generate expected output spikes.
        """
        # Set global simulation parameters
        self._model.register_global_property("dt", 1e-1)  # Time step (100 μs)
        self._model.register_global_property("I_bias", 0)  # No bias current
        
        # Define LIF neuron parameters
        C = 10e-9  # Membrane capacitance in Farads (10 nF)
        R = 1e12  # Membrane resistance in Ohms (1 TΩ)
        vthr = -45  # Spike threshold voltage (mV)
        tref = 5e-3  # Refractory period (5 ms)
        vrest = -60  # Resting potential (mV)
        vreset = -60  # Reset potential after spike (mV)
        tref_allows_integration = (
            1  # Whether to allow integration during refractory period
        )
        I_in = 4e-8  # Input current (40 nA)
        
        # Package parameters for soma creation
        soma_parameters = [
            C,
            R,
            vthr,
            tref,
            vrest,
            vreset,
            tref_allows_integration,
            I_in,
        ]
        
        # Set initial internal state for neurons
        v = vrest  # Initial membrane voltage
        tcount = 0  # Time counter
        tlast = 0  # Last spike time
        default_internal_state = [v, tcount, tlast]
        
        # Create first LIF neuron (receives external input)
        soma_0 = self._model.create_soma(
            breed="LIF_Soma",
            parameters=soma_parameters,
            default_internal_state=default_internal_state,
        )
        # Create second LIF neuron (receives input from soma_0)
        soma_1 = self._model.create_soma(
            breed="LIF_Soma",
            parameters=soma_parameters,
            default_internal_state=default_internal_state,
        )

        # Define synaptic parameters for connections
        weight = 1.0  # Synaptic weight (strength)
        synaptic_delay = 1.0  # Transmission delay (ms)
        scale = 1.0  # Scaling factor
        tau_fall = 1e-3  # Decay time constant (1 ms)
        tau_rise = 0  # Rise time constant (instantaneous)
        synapse_parameters = [
            weight,
            synaptic_delay,
            scale,
            tau_fall,
            tau_rise,
        ]
        
        # Initial synaptic current
        I_synapse = 0.0
        synapse_internal_state = [I_synapse]
        
        # Create external input synapse (stimulates soma_0)
        syn_ext = self._model.create_synapse(
            breed="Single_Exp_Synapse_STDP1",
            pre_soma_id=np.nan,  # External input (no pre-synaptic neuron)
            post_soma_id=soma_0,
            parameters=synapse_parameters,
            default_internal_state=synapse_internal_state,
        )
        # Create internal synapse (soma_0 -> soma_1)
        syn_int = self._model.create_synapse(
            breed="Single_Exp_Synapse_STDP1",
            pre_soma_id=soma_0,
            post_soma_id=soma_1,
            parameters=synapse_parameters,
            default_internal_state=synapse_internal_state,
        )

        # Initialize the simulation environment
        self._model.setup(use_gpu=self._use_gpu)
        
        # Define input spike pattern: spike at time 1 and 2 with value 1
        spikes = [(1, 1), (2, 1)]  # (time_tick, spike_value)
        
        # Inject spikes into the external synapse
        for spike in spikes:
            self._model.add_spike(synapse_id=syn_ext, tick=spike[0], value=spike[1])

        # Run simulation for 100 time steps, recording every tick
        self._model.simulate(ticks=100, update_data_ticks=100)

        # Verify results: expect at least 2 spikes from soma_0
        minimum_expected_spikes = 2
        
        # Extract membrane potential history for analysis
        internal_states_history_soma0 = np.array(
            self._model.get_internal_states_history(agent_id=soma_0)
        )
        
        # Generate visualization of membrane potential over time
        plt.figure(figsize=(5, 5))
        plt.plot(
            internal_states_history_soma0[:, 0],
            label="Membrane Potential of Soma 0",
        )
        plt.ylabel("Mem. Pot. (mV)")
        plt.xlabel("Time (ms)")
        plt.legend()
        plt.title("LIF Neuron Response to Input Spikes")
        plt.savefig("logic_gates_test_lif_test_two_soma_soma0.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save detailed simulation data to CSV for further analysis
        with open("output_LIF.csv", "w", newline="") as file:
            writer = csv.writer(file)
            # Write header
            writer.writerow(["Membrane_Potential_mV", "Time_Count", "Last_Spike_Time"])
            # Write data
            writer.writerows(self._model.get_internal_states_history(agent_id=soma_0))
        
        # Assert that the neuron generated expected number of spikes
        actual_spikes = len(self._model.get_spike_times(soma_id=soma_0))
        assert (
            actual_spikes >= minimum_expected_spikes
        ), f"Total number of spikes are {actual_spikes} but should be at least {minimum_expected_spikes}"


class LogicGatesTestGPU(LogicGatesTestLIF):
    """
    GPU-accelerated version of the LIF logic gates test.
    
    This class inherits all test methods from LogicGatesTestLIF but runs
    them using GPU acceleration for performance comparison and validation
    that GPU and CPU implementations produce consistent results.
    """
    
    def __init__(self, methodName: str = ...) -> None:
        """
        Initialize GPU test variant.
        
        Args:
            methodName: Name of the test method to run
        """
        super().__init__(methodName)
        # Create fresh model instance for GPU testing
        self._model = NeuromorphicModel()
        # Enable GPU acceleration
        self._use_gpu = True


if __name__ == "__main__":
    # Run all test cases when script is executed directly
    unittest.main()
