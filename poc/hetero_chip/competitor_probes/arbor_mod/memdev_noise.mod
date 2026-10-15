: Try to use a WHITE_NOISE variable inside NET_RECEIVE (event-time randomness).
NEURON {
    POINT_PROCESS memdev_noise
    RANGE dg
    NONSPECIFIC_CURRENT i
}
PARAMETER { dg = 0.001 }
STATE { g }
WHITE_NOISE { W }
INITIAL { g = 0.0001 }
BREAKPOINT { i = g*v }
NET_RECEIVE(w) {
    g = g + dg*w*W
}
