import numpy as np
from superneuroabm.model import NeuromorphicModel

class PoissonInput:
    def __init__(self, rate_hz, dt_ms, seed=None):
        self.rate_hz = rate_hz
        self.dt_ms = dt_ms
        self.rng = np.random.default_rng(seed)

    def add_to_model(self, model, synapse_id, n_ticks, start_tick=0):
        # Mean Poisson events per simulation tick
        mean_per_tick = self.rate_hz * self.dt_ms / 1000.0
        counts = self.rng.poisson(mean_per_tick, size=n_ticks)

        for offset, count in enumerate(counts):
            if count:
                # `value=count` represents that many coincident unit inputs
                model.add_spike(
                    synapse_id=synapse_id,
                    tick=start_tick + offset,
                    value=float(count),
                )

        return counts