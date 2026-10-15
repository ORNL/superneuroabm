: device that deposits "heat" into a shared per-CV pseudo-ion concentration every step
NEURON {
    POINT_PROCESS heater
    USEION heat WRITE heatd
    RANGE p
}
PARAMETER { p = 0.001 }
BREAKPOINT {
    heatd = heatd + p
}
