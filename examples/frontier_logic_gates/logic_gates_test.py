from superneuroabm.model import NeuromorphicModel


def create_and_simulate_or_gate(use_gpu=True):
    """Builds a neuromorphic circuit for OR gate and simulates it"""
    model = NeuromorphicModel()

    # Create neurons
    input_0 = model.create_neuron(threshold=0.0)
    input_1 = model.create_neuron(threshold=0.0)
    output_2 = model.create_neuron(threshold=0.0)

    # Create synapses
    model.create_synapse(pre_neuron_id=input_0, post_neuron_id=output_2, weight=1.0)
    model.create_synapse(pre_neuron_id=input_1, post_neuron_id=output_2, weight=1.0)

    # Setup and simulate
    model.setup(output_buffer_len=10, use_gpu=use_gpu)

    # test_cases in format time -> ([(Neuron, value), (Neuron, value)]
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
    # Overall expected spike times for output_2 neuron: [4, 6, 8]
    expected_times = [4, 6, 8]
    for time in test_cases:
        for neuron, value in test_cases[time]:
            model.add_spike(neuron_id=neuron, tick=time, value=value)

    model.simulate(ticks=10)

    print(model.get_spike_times(neuron_id=output_2))
    assert (
        model.get_spike_times(neuron_id=output_2) == expected_times
    ), f"Spike times are {model.get_spike_times(neuron_id=output_2)} but should be {expected_times}"


def create_and_simulate_and_gate(use_gpu=True):
    """Builds a neuromorphic circuit for AND gate and simulates it"""

    model = NeuromorphicModel()
    # Create neurons
    input_0 = model.create_neuron(threshold=0.0)
    input_1 = model.create_neuron(threshold=0.0)
    output_2 = model.create_neuron(threshold=2.0)

    # Create synapses
    model.create_synapse(pre_neuron_id=input_0, post_neuron_id=output_2, weight=1.0)
    model.create_synapse(pre_neuron_id=input_1, post_neuron_id=output_2, weight=1.0)

    # Setup and simulate
    model.setup(output_buffer_len=10, use_gpu=use_gpu)

    # test_cases in format time -> ([(Neuron, value), (Neuron, value)]
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
    # Overall expected spike times for output_2 neuron: [8]
    expected_times = [8]
    for time in test_cases:
        for neuron, value in test_cases[time]:
            model.add_spike(neuron_id=neuron, tick=time, value=value)

    model.simulate(ticks=10)
    print(model.get_spike_times(neuron_id=output_2))
    assert (
        model.get_spike_times(neuron_id=output_2) == expected_times
    ), f"Spike times are {model.get_spike_times(neuron_id=output_2)} but should be {expected_times}"

    print(model.summary())


if __name__ == "__main__":
    create_and_simulate_or_gate(use_gpu=True)
    create_and_simulate_and_gate(use_gpu=True)
