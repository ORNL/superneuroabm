: Per-step stochastic drift of g via SDE (white noise in DERIVATIVE).
NEURON {
    POINT_PROCESS memdev_sde
    RANGE dg, sigma
    NONSPECIFIC_CURRENT i
}
PARAMETER { dg = 0.001 sigma = 0.0001 }
STATE { g }
WHITE_NOISE { W }
INITIAL { g = 0.0001 }
BREAKPOINT {
    SOLVE dyn METHOD stochastic
    i = g*v
}
DERIVATIVE dyn { g' = sigma*W }
NET_RECEIVE(w) { g = g + dg*w }
