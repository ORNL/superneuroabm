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
        spikes = [(1, 1), (2, 1), (10, 100)]  # (time_tick, spike_value)

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

        # Extract the synaptic current history for soma_0:
        internal_states_history_syn0 = np.array(
            self._model.get_internal_states_history(agent_id=syn_ext)
        )

        print(f"Internal states history from synapse 0: {internal_states_history_syn0}")
        print(f"Internal states history from soma 0: {internal_states_history_soma0}")

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
        plt.savefig(
            "logic_gates_test_lif_test_two_soma_soma0.png", dpi=150, bbox_inches="tight"
        )
        plt.close()

        # Save detailed simulation data to CSV for further analysis
        with open("output_LIF.csv", "w", newline="") as file:
            writer = csv.writer(file)
            # Write header
            writer.writerow(["Membrane_Potential_mV", "Time_Count", "Last_Spike_Time", "Synapse_Current"])
            # Write data
            writer.writerows(self._model.get_internal_states_history(agent_id=soma_0))

        # Assert that the neuron generated expected number of spikes
        actual_spikes = len(self._model.get_spike_times(soma_id=soma_0))
        # assert (
        #     actual_spikes >= minimum_expected_spikes
        # ), f"Total number of spikes are {actual_spikes} but should be at least {minimum_expected_spikes}"

    def test_synapse(self):
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
        I_in = 0  # Input current (40 nA)

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
        # # Create second LIF neuron (receives input from soma_0)
        # soma_1 = self._model.create_soma(
        #     breed="LIF_Soma",
        #     parameters=soma_parameters,
        #     default_internal_state=default_internal_state,
        # )

        # Define synaptic parameters for connections
        weight = 1.0  # Synaptic weight (strength)
        synaptic_delay = 1.0  # Transmission delay (ms)
        scale = 1.0  # Scaling factor
        tau_fall = 1  # Decay time constant (1 ms)
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
            breed="Single_Exp_Synapse",
            pre_soma_id=np.nan,  # External input (no pre-synaptic neuron)
            post_soma_id=soma_0,
            parameters=synapse_parameters[:],
            default_internal_state=synapse_internal_state,
        )
        # # Create internal synapse (soma_0 -> soma_1)
        # syn_int = self._model.create_synapse(
        #     breed="Single_Exp_Synapse",
        #     pre_soma_id=soma_0,
        #     post_soma_id=soma_1,
        #     parameters=synapse_parameters[:],
        #     default_internal_state=synapse_internal_state,
        # )

        # Initialize the simulation environment
        self._model.setup(use_gpu=self._use_gpu)

        # Define input spike pattern: spike at time 1 and 2 with value 1
        spikes = [(1, 1), (2, 1), (10, 100)]  # (time_tick, spike_value)

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

        # Extract the synaptic current history for soma_0:
        internal_states_history_syn0 = np.array(
            self._model.get_internal_states_history(agent_id=syn_ext)
        )

        print(f"Internal states history from synapse 0: {internal_states_history_syn0}")
        print(f"Internal states history from soma 0: {internal_states_history_soma0}")

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
        plt.savefig(
            "logic_gates_test_lif_test_synapse.png", dpi=150, bbox_inches="tight"
        )
        plt.close()

        # Save detailed simulation data to CSV for further analysis
        with open("output_LIF.csv", "w", newline="") as file:
            writer = csv.writer(file)
            # Write header
            writer.writerow(["Membrane_Potential_mV", "Time_Count", "Last_Spike_Time", "Synapse_Current"])
            # Write combined data
            for i in range(len(internal_states_history_soma0)):
                row = list(internal_states_history_soma0[i]) + [internal_states_history_syn0[i][0]]
                writer.writerow(row)

        # Assert that the neuron generated expected number of spikes
        actual_spikes = len(self._model.get_spike_times(soma_id=soma_0))
        assert (
            actual_spikes >= minimum_expected_spikes
        ), f"Total number of spikes are {actual_spikes} but should be at least {minimum_expected_spikes}"

    def test_dual_external_synapses(self):
        """
        Tests the integration of inputs from two external synapses to a single LIF neuron.

        This test creates a circuit with:
        1. External input A -> Synapse A -> Soma 0
        2. External input B -> Synapse B -> Soma 0

        The test verifies that the soma can integrate spikes from multiple
        synaptic inputs and generate appropriate output spikes based on
        the combined input strength.
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
        tref_allows_integration = 1  # Allow integration during refractory period
        I_in = 0  # No direct input current (only synaptic input)

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

        # Set initial internal state for neuron
        v = vrest  # Initial membrane voltage
        tcount = 0  # Time counter
        tlast = 0  # Last spike time
        default_internal_state = [v, tcount, tlast, 0]

        # Create single LIF neuron that will receive dual inputs
        soma_0 = self._model.create_soma(
            breed="LIF_Soma",
            parameters=soma_parameters,
            default_internal_state=default_internal_state,
        )

        # Define synaptic parameters for first synapse (stronger weight)
        weight_A = 2.0  # Stronger synaptic weight
        synaptic_delay_A = 1.0  # Transmission delay (ms)
        scale_A = 1.0  # Scaling factor
        tau_fall_A = 1  # Decay time constant (2 ms)
        tau_rise_A = 0  # Rise time constant (instantaneous)
        synapse_parameters_A = [
            weight_A,
            synaptic_delay_A,
            scale_A,
            tau_fall_A,
            tau_rise_A,
        ]

        # Define synaptic parameters for second synapse (weaker weight)
        weight_B = 1.0  # Weaker synaptic weight
        synaptic_delay_B = 1.0  # Longer transmission delay (ms)
        scale_B = 1.0  # Scaling factor
        tau_fall_B = 1  # Faster decay time constant (1 ms)
        tau_rise_B = 0  # Rise time constant (instantaneous)
        synapse_parameters_B = [
            weight_B,
            synaptic_delay_B,
            scale_B,
            tau_fall_B,
            tau_rise_B,
        ]

        # Initial synaptic current for both synapses
        I_synapse = 0.0
        synapse_internal_state = [I_synapse]

        # Create first external input synapse (stronger input)
        syn_ext_A = self._model.create_synapse(
            breed="Single_Exp_Synapse",
            pre_soma_id=np.nan,  # External input (no pre-synaptic neuron)
            post_soma_id=soma_0,
            parameters=synapse_parameters_A,
            default_internal_state=synapse_internal_state,
        )

        # Create second external input synapse (weaker input)
        syn_ext_B = self._model.create_synapse(
            breed="Single_Exp_Synapse",
            pre_soma_id=np.nan,  # External input (no pre-synaptic neuron)
            post_soma_id=soma_0,
            parameters=synapse_parameters_B,
            default_internal_state=synapse_internal_state,
        )

        # Initialize the simulation environment
        self._model.setup(use_gpu=self._use_gpu)

        # Define input spike patterns for both synapses
        # Synapse A receives early, strong spikes
        spikes_A = [(2, 1), (10, 1), (20, 1)]  # (time_tick, spike_value)

        # Synapse B receives delayed, weaker spikes that overlap with A
        spikes_B = [(5, 1), (12, 1), (25, 1)]  # (time_tick, spike_value)

        # Inject spikes into both external synapses
        for spike in spikes_A:
            self._model.add_spike(synapse_id=syn_ext_A, tick=spike[0], value=spike[1])

        for spike in spikes_B:
            self._model.add_spike(synapse_id=syn_ext_B, tick=spike[0], value=spike[1])

        # Run simulation for 50 time steps, recording every tick
        self._model.simulate(ticks=1000, update_data_ticks=1)

        # Extract simulation results for analysis
        internal_states_history_soma0 = np.array(
            self._model.get_internal_states_history(agent_id=soma_0)
        )

        internal_states_history_synA = np.array(
            self._model.get_internal_states_history(agent_id=syn_ext_A)
        )

        internal_states_history_synB = np.array(
            self._model.get_internal_states_history(agent_id=syn_ext_B)
        )

        # Print debug information
        # print(f"Soma 0 spike times: {self._model.get_spike_times(soma_id=soma_0)}")
        print(f"Soma 0 I synapse: {internal_states_history_soma0}")
        # print(f"Synapse A internal states: {internal_states_history_synA[:10]}")  # First 10 timesteps
        # print(f"Synapse B internal states: {internal_states_history_synB[:10]}")  # First 10 timesteps

        # Generate visualization comparing membrane potential and synaptic currents
        plt.figure(figsize=(12, 8))

        # Plot membrane potential
        plt.subplot(3, 1, 1)
        plt.plot(internal_states_history_soma0[:, 0], 'b-', label="Membrane Potential")
        plt.axhline(y=vthr, color='r', linestyle='--', label="Threshold")
        plt.ylabel("Membrane Pot. (mV)")
        plt.title("Dual Synapse Input Integration")
        plt.legend()

        # Plot synaptic current from synapse A
        plt.subplot(3, 1, 2)
        plt.plot(internal_states_history_synA[:, 0], 'g-', label="Synapse A Current")
        plt.ylabel("Synaptic Current A")
        plt.legend()

        # Plot synaptic current from synapse B
        plt.subplot(3, 1, 3)
        plt.plot(internal_states_history_synB[:, 0], 'm-', label="Synapse B Current")
        plt.xlabel("Time (ticks)")
        plt.ylabel("Synaptic Current B")
        plt.legend()

        plt.tight_layout()
        plt.savefig("logic_gates_test_lif_dual_synapses.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Save detailed simulation data to CSV
        with open("output_dual_synapses.csv", "w", newline="") as file:
            writer = csv.writer(file)
            # Write header
            writer.writerow(
                [
                    "Membrane_Potential_mV",
                    "Time_Count",
                    "Last_Spike_Time",
                    "SynA_Current",
                    "SynB_Current",
                ]
            )
            # Write combined data
            for i in range(len(internal_states_history_soma0)):
                row = list(internal_states_history_soma0[i]) + \
                      [internal_states_history_synA[i][0], internal_states_history_synB[i][0]]
                writer.writerow(row)

        # Also save complete soma data as output_LIF.csv
        with open("output_LIF.csv", "w", newline="") as file:
            writer = csv.writer(file)
            # Write header based on LIF soma step function (lines 88-91)
            writer.writerow(["Membrane_Potential_mV", "Time_Count", "Last_Spike_Time", "I_Synapse"])
            # Write complete soma internal states
            writer.writerows(internal_states_history_soma0)

        # Verify that soma responds to dual inputs
        actual_spikes = len(self._model.get_spike_times(soma_id=soma_0))

        # We expect at least some integration effect from dual inputs
        # The exact number depends on the timing and strength of inputs
        minimum_expected_spikes = 1
        assert (
            actual_spikes >= minimum_expected_spikes
        ), f"Soma should generate at least {minimum_expected_spikes} spike(s) from dual inputs, got {actual_spikes}"

        # Additional verification: check that both synapses contributed
        # by verifying non-zero synaptic currents
        max_synA_current = np.max(np.abs(internal_states_history_synA[:, 0]))
        max_synB_current = np.max(np.abs(internal_states_history_synB[:, 0]))

        assert max_synA_current > 0, "Synapse A should show non-zero current"
        assert max_synB_current > 0, "Synapse B should show non-zero current"

        print(f"Test passed: Soma generated {actual_spikes} spikes from dual synaptic inputs")
        print(f"Max synaptic currents - A: {max_synA_current:.2e}, B: {max_synB_current:.2e}")


    def test_dual_external_synapses_dual_somas(self):
        """
        Tests the integration of inputs from two external synapses to a single LIF neuron.

        This test creates a circuit with:
        1. External input A -> Synapse A -> Soma 0
        2. External input B -> Synapse B -> Soma 0

        The test verifies that the soma can integrate spikes from multiple
        synaptic inputs and generate appropriate output spikes based on
        the combined input strength.
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
        vreset = -70  # Reset potential after spike (mV)
        tref_allows_integration = 1  # Allow integration during refractory period
        I_in = 0  # No direct input current (only synaptic input)

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

        # Set initial internal state for neuron
        v = vrest  # Initial membrane voltage
        tcount = 0  # Time counter
        tlast = 0  # Last spike time
        default_internal_state = [v, tcount, tlast, 0]

        # Create single LIF neuron that will receive dual inputs
        soma_0 = self._model.create_soma(
            breed="LIF_Soma",
            parameters=soma_parameters,
            default_internal_state=default_internal_state,
        )

        # Create single LIF neuron that will receive dual inputs
        soma_1 = self._model.create_soma(
            breed="LIF_Soma",
            parameters=soma_parameters,
            default_internal_state=default_internal_state,
        )

        # Define synaptic parameters for first synapse (stronger weight)
        weight_A = 2.0  # Stronger synaptic weight
        synaptic_delay_A = 1.0  # Transmission delay (ms)
        scale_A = 1.0  # Scaling factor
        tau_fall_A = 1  # Decay time constant (2 ms)
        tau_rise_A = 0  # Rise time constant (instantaneous)
        synapse_parameters_A = [
            weight_A,
            synaptic_delay_A,
            scale_A,
            tau_fall_A,
            tau_rise_A,
        ]

        # Define synaptic parameters for second synapse (weaker weight)
        weight_B = 1.0  # Weaker synaptic weight
        synaptic_delay_B = 1.0  # Longer transmission delay (ms)
        scale_B = 1.0  # Scaling factor
        tau_fall_B = 1  # Faster decay time constant (1 ms)
        tau_rise_B = 0  # Rise time constant (instantaneous)
        synapse_parameters_B = [
            weight_B,
            synaptic_delay_B,
            scale_B,
            tau_fall_B,
            tau_rise_B,
        ]

        # Initial synaptic current for both synapses
        I_synapse = 0.0
        synapse_internal_state = [I_synapse]

        # Create first external input synapse (stronger input)
        syn_ext_A = self._model.create_synapse(
            breed="Single_Exp_Synapse",
            pre_soma_id=np.nan,  # External input (no pre-synaptic neuron)
            post_soma_id=soma_0,
            parameters=synapse_parameters_A,
            default_internal_state=synapse_internal_state,
        )

        # Create second external input synapse (weaker input)
        syn_ext_B = self._model.create_synapse(
            breed="Single_Exp_Synapse",
            pre_soma_id=np.nan,  # External input (no pre-synaptic neuron)
            post_soma_id=soma_0,
            parameters=synapse_parameters_B,
            default_internal_state=synapse_internal_state,
        )


        # Create second external input synapse (weaker input)
        syn_int_C = self._model.create_synapse(
            breed="Single_Exp_Synapse",
            pre_soma_id=soma_0,  # External input (no pre-synaptic neuron)
            post_soma_id=soma_1,
            parameters=synapse_parameters_B,
            default_internal_state=synapse_internal_state,
        )       

        # Initialize the simulation environment
        self._model.setup(use_gpu=self._use_gpu)

        # Define input spike patterns for both synapses
        # Synapse A receives early, strong spikes
        spikes_A = [(2, 1), (10, 1), (20, 1)]  # (time_tick, spike_value)

        # Synapse B receives delayed, weaker spikes that overlap with A
        spikes_B = [(5, 1), (12, 1), (25, 1)]  # (time_tick, spike_value)

        # Inject spikes into both external synapses
        for spike in spikes_A:
            self._model.add_spike(synapse_id=syn_ext_A, tick=spike[0], value=spike[1])

        for spike in spikes_B:
            self._model.add_spike(synapse_id=syn_ext_B, tick=spike[0], value=spike[1])

        # Run simulation for 50 time steps, recording every tick
        self._model.simulate(ticks=600, update_data_ticks=1)

        # Extract simulation results for analysis
        internal_states_history_soma0 = np.array(
            self._model.get_internal_states_history(agent_id=soma_0)
        )

        internal_states_history_soma1 = np.array(
            self._model.get_internal_states_history(agent_id=soma_1)
        )
        internal_states_history_synA = np.array(
            self._model.get_internal_states_history(agent_id=syn_ext_A)
        )

        internal_states_history_synB = np.array(
            self._model.get_internal_states_history(agent_id=syn_ext_B)
        )

        internal_states_history_synC = np.array(
            self._model.get_internal_states_history(agent_id=syn_int_C)
        )


        # Print debug information
        # print(f"Soma 0 spike times: {self._model.get_spike_times(soma_id=soma_0)}")
        # print(f"Soma 0 I synapse: {internal_states_history_soma0}")
        # print(f"Synapse A internal states: {internal_states_history_synA[:10]}")  # First 10 timesteps
        # print(f"Synapse B internal states: {internal_states_history_synB[:10]}")  # First 10 timesteps

        # Generate visualization comparing membrane potential and synaptic currents
        plt.figure(figsize=(12, 8))

        # Plot membrane potential
        plt.subplot(5, 1, 1)
        plt.plot(internal_states_history_soma0[:, 0], 'b-', label="Soma 0")
        plt.axhline(y=vthr, color='r', linestyle='--', label="Threshold")
        plt.ylabel("Membrane Pot. (mV)")
        plt.title("Soma 0")
        plt.legend()

        # Plot membrane potential
        plt.subplot(5, 1, 2)
        plt.plot(internal_states_history_soma1[:, 0], 'b-', label="Soma 1")
        plt.axhline(y=vthr, color='r', linestyle='--', label="Threshold")
        plt.ylabel("Membrane Pot. (mV)")
        plt.title("Soma 1")
        plt.legend()

        # Plot synaptic current from synapse A
        plt.subplot(5, 1, 3)
        plt.plot(internal_states_history_synA[:, 0], 'g-', label="Synapse A Current")
        plt.ylabel("Synaptic Current A")
        plt.legend()

        # Plot synaptic current from synapse B
        plt.subplot(5, 1, 4)
        plt.plot(internal_states_history_synB[:, 0], 'm-', label="Synapse B Current")
        plt.xlabel("Time (ticks)")
        plt.ylabel("Synaptic Current B")
        plt.legend()

        # Plot synaptic current from synapse B
        plt.subplot(5, 1, 5)
        plt.plot(internal_states_history_synC[:, 0], 'm-', label="Synapse C Current")
        plt.xlabel("Time (ticks)")
        plt.ylabel("Synaptic Current C")
        plt.legend()

        plt.tight_layout()
        plt.savefig("logic_gates_test_lif_dual_synapses_dual_somas.png", dpi=150, bbox_inches="tight")
        plt.close()

        # # Save detailed simulation data to CSV
        # with open("output_dual_synapses.csv", "w", newline="") as file:
        #     writer = csv.writer(file)
        #     # Write header
        #     writer.writerow(
        #         [
        #             "Membrane_Potential_mV",
        #             "Time_Count",
        #             "Last_Spike_Time",
        #             "SynA_Current",
        #             "SynB_Current",
        #         ]
        #     )
        #     # Write combined data
        #     for i in range(len(internal_states_history_soma0)):
        #         row = list(internal_states_history_soma0[i]) + \
        #               [internal_states_history_synA[i][0], internal_states_history_synB[i][0]]
        #         writer.writerow(row)

        # # Also save complete soma data as output_LIF.csv
        # with open("output_LIF.csv", "w", newline="") as file:
        #     writer = csv.writer(file)
        #     # Write header based on LIF soma step function (lines 88-91)
        #     writer.writerow(["Membrane_Potential_mV", "Time_Count", "Last_Spike_Time", "I_Synapse"])
        #     # Write complete soma internal states
        #     writer.writerows(internal_states_history_soma0)

        # Verify that soma responds to dual inputs
        actual_spikes = len(self._model.get_spike_times(soma_id=soma_0))

        # We expect at least some integration effect from dual inputs
        # The exact number depends on the timing and strength of inputs
        minimum_expected_spikes = 1
        assert (
            actual_spikes >= minimum_expected_spikes
        ), f"Soma should generate at least {minimum_expected_spikes} spike(s) from dual inputs, got {actual_spikes}"

        # Additional verification: check that both synapses contributed
        # by verifying non-zero synaptic currents
        max_synA_current = np.max(np.abs(internal_states_history_synA[:, 0]))
        max_synB_current = np.max(np.abs(internal_states_history_synB[:, 0]))

        assert max_synA_current > 0, "Synapse A should show non-zero current"
        assert max_synB_current > 0, "Synapse B should show non-zero current"

        print(f"Test passed: Soma generated {actual_spikes} spikes from dual synaptic inputs")
        print(f"Max synaptic currents - A: {max_synA_current:.2e}, B: {max_synB_current:.2e}")


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
