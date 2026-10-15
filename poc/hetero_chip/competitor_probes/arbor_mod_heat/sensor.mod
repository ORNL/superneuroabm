: device whose conductance drifts with the shared per-CV "heat" written by other devices
NEURON {
    POINT_PROCESS sensor
    USEION heat READ heatd
    NONSPECIFIC_CURRENT i
}
STATE { g seen }
INITIAL {
    g = 0.0001
    seen = 0
}
BREAKPOINT {
    SOLVE s METHOD cnexp
    i = g*v
}
DERIVATIVE s {
    g' = 1e-6*heatd
    seen' = heatd
}
