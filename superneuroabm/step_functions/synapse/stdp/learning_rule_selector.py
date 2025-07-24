from cupyx import jit

from superneuroabm.step_functions.synapse.util import get_pre_soma_spike
from superneuroabm.step_functions.synapse.stdp.exp_pair_wise_stdp import exp_pair_wise_stdp

@jit.rawkernel(device="cuda")
def learning_rule_selector(
    tick,
    agent_index,
    globals,
    agent_ids,
    breeds,
    locations,
    synapse_params,
    learning_params,  # STDP_function name, 
    internal_state,  #
    internal_learning_state,
    synapse_history,  # delay
    input_spikes_tensor,  # input spikes
    output_spikes_tensor,
    internal_states_buffer,
    internal_learning_states_buffer,
):
    t_current = int(tick)

    dt = globals[1]  # time step size

    stdpType = learning_params[agent_index][0] #0 for None, 1 for exp_pair_wise_stdp
    #Wmax, Wmin
    if stdpType == -1:
        pass
    elif stdpType == 0:
        
        exp_pair_wise_stdp(
            tick,
            agent_index,
            globals,
            agent_ids,
            breeds,
            locations,
            synapse_params,  # scale, time constant (tau_rise and tau_fall)
            learning_params,
            internal_state,  #
            synapse_history,  # delay
            input_spikes_tensor,  # input spikes
            output_spikes_tensor,
            internal_states_buffer,
        )
