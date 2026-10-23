from pathlib import Path
import numpy as np
import sys

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR.parent))

from superneuroabm.model import NeuromorphicModel

from superneuroabm.step_functions.soma.rf import (
    resonate_and_fire_soma_step_func,
)

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent

CONFIG_PATH = SCRIPT_DIR / "component_base_config.yaml"
#CONFIG_PATH = SCRIPT_DIR / "resonate_and_fire_config.yaml"

#SOMA_STEP_PATH = (
#    SCRIPT_DIR
#    / "user_step_functions"
#    / "resonate_and_fire_soma.py"
#)


def main():
    model = NeuromorphicModel(
        #user_config=CONFIG_PATH,
        enable_internal_states_tracking=True,
    )

    ## Registration must occur before model.setup().
    #model.register_soma_type(
    #    name="resonate_and_fire_soma",
    #    step_func=resonate_and_fire_soma_step_func,
    #    step_func_path=SOMA_STEP_PATH,
    #)

    soma = model.create_soma(
        breed="rf_soma",
        config_name="config_0",
    )

    # External input through the built-in exponential synapse.
    input_synapse = model.create_synapse(
        breed="single_exp_synapse",
        pre_soma_id=-1,
        post_soma_id=soma,
        config_name="config_0",
        overrides={
            "hyperparameters": {
                "weight": 14.0,
                "scale": 1.0,
                "tau_fall": 10e-3,
            }
        },
    )

    model.setup()

    # A periodic input near the 10-Hz preferred frequency.
    # With dt=1 ms, a 100-tick interval is 100 ms or 10 Hz.
    for tick in range(10, 1000, 100):
        model.add_spike(
            synapse_id=input_synapse,
            tick=tick,
            value=1.1,
        )

    model.simulate(
        ticks=1000,
        update_data_ticks=1000,
    )

    print("Output spike times:")
    print(model.get_spike_times(soma_id=soma))

    states = model.get_internal_states_history(agent_id=soma)

    print("First ten [v, u, tcount, refractory] states:")
    for state in states[:10]:
        print(state)

    spike_ticks = model.get_spike_times(soma_id=soma)

    print("Number of output spikes:", len(spike_ticks))
    print("Output spike ticks:", spike_ticks)

    # Default SuperNeuroABM timestep is 1 ms
    dt = 1e-3
    
    spike_times_seconds = [tick * dt for tick in spike_ticks]
    spike_times_ms = [tick * dt * 1000.0 for tick in spike_ticks]

    print("Spike times (seconds):", spike_times_seconds)
    print("Spike times (ms):", spike_times_ms)

    plt.eventplot(
        spike_times_ms,
        lineoffsets=1,
        linelengths=0.8,
    )

    plt.xlabel("Time (ms)")
    plt.ylabel("Resonate-and-fire neuron")
    plt.yticks([1], [f"Soma {soma}"])
    plt.title("Resonate-and-Fire Output Spikes")
    plt.tight_layout()
    #plt.savefig("resonate_and_fire_spikes.png", dpi=300)

    ############################################################
    ### PLOTS:
    ############################################################
    voltage = np.asarray(states)[:, 0]
    vthr = 1.0

    fig, ax_voltage = plt.subplots(figsize=(11, 5))
    
    # ------------------------------------------------------------
    # Membrane-potential trace
    # ------------------------------------------------------------
    time_ms = np.arange(len(voltage)) * dt * 1000.0
    ax_voltage.plot(
        time_ms,
        voltage,
        color="tab:blue",
        linewidth=1.5,
        label="Membrane potential, v",
    )
    
    ax_voltage.axhline(
        vthr,
        color="tab:red",
        linestyle="--",
        linewidth=1.2,
        label="Spike threshold",
    )
    
    ax_voltage.set_xlabel("Time (ms)")
    ax_voltage.set_ylabel("Membrane potential, v", color="tab:blue")
    ax_voltage.tick_params(axis="y", labelcolor="tab:blue")
    ax_voltage.grid(alpha=0.25)
    
    # ------------------------------------------------------------
    # Mark output spikes at threshold on the membrane-potential axis.
    # ------------------------------------------------------------
    if len(spike_times_ms) > 0:
        ax_voltage.scatter(
            spike_times_ms,
            np.full(len(spike_times_ms), vthr),
            marker="o",
            s=36,
            color="black",
            label="Output spikes",
            zorder=3,
        )
    
    ax_voltage.legend(loc="upper right")
    
    ax_voltage.set_title(
        "Resonate-and-Fire Membrane Potential and Output Spikes"
    )
    
    fig.tight_layout()
    fig.savefig(
        "resonate_and_fire_voltage_spikes.png",
        dpi=300,
        bbox_inches="tight",
    )



if __name__ == "__main__":
    main()
