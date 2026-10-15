: Memristive device: per-instance state g (conductance) and n (write count),
: per-instance parameter dg (step), every-step selector current I = g*sinh(v/v0),
: NET_RECEIVE programming pulse.
NEURON {
    POINT_PROCESS memdev
    RANGE dg, v0, gmax
    NONSPECIFIC_CURRENT i
}
PARAMETER {
    dg = 0.001
    v0 = 20
    gmax = 0.01
}
STATE { g n }
INITIAL { g = 0.0001  n = 0 }
BREAKPOINT {
    i = g * (exp(v/v0) - exp(-v/v0)) * 0.5
}
NET_RECEIVE(w) {
    n = n + 1
    if (g + dg*w < gmax) { g = g + dg*w } else { g = gmax }
}
