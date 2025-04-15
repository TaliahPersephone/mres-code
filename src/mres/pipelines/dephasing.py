"""Pipeline for calculating weak values with decoherence"""

import numpy as np

from mres.bathevo import ohmic_decoherence_function
from mres.paramprep import prepare_params_dephasing
from mres.pipelines import WvPipeline
from mres.wvformulas import wv_pure_dephasing_x, wv_pure_dephasing_z


class DephasingPipeline(WvPipeline):
    def __init__(self, params):
        super().__init__(params)

    def run(self):
        """Calculate wvs for the pure dephasing model"""
        params = self.params
        times = params["times"]
        u = params["u"]
        v = params["v"]
        m = params["m"]
        spectral_params = params["spectral_params"]

        times, spectral_params["s"] = prepare_params_dephasing(
            times, spectral_params["s"]
        )

        gamma_ts, phi_ts = ohmic_decoherence_function(times, **spectral_params)

        if np.all(m == np.array([0, 0, 1])):
            wvs = wv_pure_dephasing_z(v, u, gamma_ts)
        elif np.all(m == np.array([1, 0, 0])):
            wvs = wv_pure_dephasing_x(v, u, gamma_ts, phi_ts)

        self.results = {
            "params": params,
            "gamma_ts": gamma_ts,
            "phi_ts": phi_ts,
            "wvs": wvs,
        }

        return self.results
