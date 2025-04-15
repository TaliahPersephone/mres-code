"""Pipeline for calculating weak values with decoherence"""

from mres.pipelines import WvPipeline
from mres.wvformulas import wv_decoherence_simple
from mres.bathevo import solvable_ohmic, solvable_ohmic_shifted, solvable_jaynes_cummings


class DecoherencePipeline(WvPipeline):
    def __init__(self, params):
        super().__init__(params)

    def run(self):
        """Calculate wvs with decoherence using simple method"""
        params = self.params
        times = params["times"]
        v = params["v"]
        m = params["m"]
        spectral_params = params["spectral_params"]
        spectral = params["spectral"]

        if spectral == "ohmic":
            c1_ts, c_vecs = solvable_ohmic(times, **spectral_params)
        elif spectral == "ohmic_shifted":
            c1_ts, c_vecs = solvable_ohmic_shifted(times, **spectral_params)
        elif spectral == "jc":
            c1_ts, c_vecs = solvable_jaynes_cummings(times, **spectral_params)
        else:
            return

        wvs = wv_decoherence_simple(v, m, c_vecs)

        self.results = {
                'params': params,
                'c1_ts': c1_ts,
                'c_vecs': c_vecs,
                'wvs': wvs,
        }

        return self.results
