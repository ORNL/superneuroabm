#!/usr/bin/env python
"""
Run tests with internal state tracking disabled (no buffer).
Only prints spike times without generating plots.
"""

from test_synapse_and_soma_models import TestSynapseAndSomaModels

if __name__ == "__main__":
    # print("=" * 70)
    # print("Running tests with internal state tracking DISABLED")
    # print("=" * 70)

    # # Test 1: LIF soma with single exponential synapse
    # print("\n[Test 1/5] LIF soma + single exp synapse")
    # print("-" * 70)
    # test1 = TestSynapseAndSomaModels('test_lif_soma_single_exp_synapse', enable_internal_state_tracking=False)
    # test1.test_lif_soma_single_exp_synapse()

    # # Test 2: Izhikevich soma with single exponential synapse
    # print("\n[Test 2/5] Izhikevich soma + single exp synapse")
    # print("-" * 70)
    # test2 = TestSynapseAndSomaModels('test_izh_soma_single_exp_synapse', enable_internal_state_tracking=False)
    # test2.test_izh_soma_single_exp_synapse()

    # # Test 3: LIF soma with two external synapses
    # print("\n[Test 3/5] LIF soma + two external synapses")
    # print("-" * 70)
    # test3 = TestSynapseAndSomaModels('test_lif_soma_two_external_synapses', enable_internal_state_tracking=False)
    # test3.test_lif_soma_two_external_synapses()

    # # Test 4: LIF soma with two internal synapses
    # print("\n[Test 4/5] LIF soma + two internal synapses")
    # print("-" * 70)
    # test4 = TestSynapseAndSomaModels('test_lif_soma_two_internal_synapses', enable_internal_state_tracking=False)
    # test4.test_lif_soma_two_internal_synapses()

    # Test 5: LIF soma with mixed synapses
    print("\n[Test 5/5] LIF soma + mixed synapses")
    print("-" * 70)
    test5 = TestSynapseAndSomaModels('test_lif_soma_mixed_synapses', enable_internal_state_tracking=False)
    test5.test_lif_soma_mixed_synapses()

    # print("\n" + "=" * 70)
    # print("All tests completed successfully!")
    # print("=" * 70)
