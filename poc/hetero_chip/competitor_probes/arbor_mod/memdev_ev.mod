: Hack: event-time randomness. W is "used" in a dummy SDE so Arbor allocates
: and refreshes per-instance N(0,1) draws each step; NET_RECEIVE then reads W.
: Stochastic switching: step only if W > thr (thr per instance).
NEURON {
    POINT_PROCESS memdev_ev
    RANGE dg, thr, v0, sig
    NONSPECIFIC_CURRENT i
}
PARAMETER { dg = 0.001  thr = 0  v0 = 20  sig = 0.5 }
STATE { g dummy nsw }
WHITE_NOISE { W }
INITIAL { g = 0.0001  dummy = 0  nsw = 0 }
BREAKPOINT {
    SOLVE dyn METHOD stochastic
    i = g * (exp(v/v0) - exp(-v/v0)) * 0.5
}
DERIVATIVE dyn { dummy' = W }
NET_RECEIVE(w) {
    if (W > thr) {
        g = g + dg*w*(1 + sig*W)
        nsw = nsw + 1
    }
}
