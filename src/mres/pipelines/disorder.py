"""Pipeline for calculating disordered weak values"""

from mres.pipelines import WvPipeline
from mres.wvformulas import calculate_wvs_ferraz
from mres.fieldquads import field_quadrature_expectations, recover_wvs_vacuum


class DisorderPipeline(WvPipeline):
    def __init__(self, params):
        super(params)

    def run(self, field_quads=True):
        """Calculate disordered wvs and quads"""
        params = self.params

        wvs = calculate_wvs_ferraz(
            params["u"],
            params["v"],
            params["m"],
            params["omega_as"],
            params["t"],
            params["tau"],
            params["gamma"],
            params["m_to_int"],
            params["a"],
            params["b"],
        )

        wv_0_0 = calculate_wvs_ferraz(
            params["u"],
            params["v"],
            params["m"],
            params["omega_a_0"],
            params["t"],
            0,
            0,
            params["m_to_int"],
            params["a"],
            params["b"],
        )

        wv_0 = calculate_wvs_ferraz(
            params["u"],
            params["v"],
            params["m"],
            params["omega_a_0"],
            params["t"],
            params["tau"],
            params["gamma"],
            params["m_to_int"],
            params["a"],
            params["b"],
        )

        if field_quads:
            qs, ps = field_quadrature_expectations(
                wvs,
                params["omega_f"],
                params["g"],
                params["t"],
                params["tau"],
                params["omega_as"],
                params["rwa"],
            )

            q = qs.mean(axis=-1)
            p = ps.mean(axis=-1)

            wvs_recovered = recover_wvs_vacuum(
                qs,
                ps,
                params["omega_f"],
                params["g"],
                params["t"],
                params["tau"],
                params["omega_as"],
                params["rwa"],
            )

            q_0, p_0 = field_quadrature_expectations(
                wv_0,
                params["omega_f"],
                params["g"],
                params["t"],
                0,
                params["omega_a_0"],
                params["rwa"],
            )

            wv_0_recovered = recover_wvs_vacuum(
                q_0,
                p_0,
                params["omega_f"],
                params["g"],
                params["t"],
                0,
                params["omega_a_0"],
                params["rwa"],
            )
        else:
            qs = None
            q = None
            ps = None
            p = None
            wvs_recovered = None
            q_0 = None
            p_0 = None
            wv_0_recovered = None

        self.results = {
            "params": params,
            "wvs": wvs,
            "wv": wvs.mean(axis=-1),
            "qs": qs,
            "q": q,
            "ps": ps,
            "p": p,
            "wvs_recovered": wvs_recovered,
            "wv_0": wv_0,
            "wv_0_0": wv_0_0,
            "p_0": p_0,
            "q_0": q_0,
            "wv_0_recovered": wv_0_recovered,
        }

        return self.results
