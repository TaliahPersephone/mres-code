"Calculate field quadrature expecations and recover weak values from those expectaions."

import numpy as np
from mres.paramprep import prepare_params_disorder


def field_quadrature_expectations(wvs, omega_f, g, t, taus, omega_as=None, rwa=False):
    """
    Return post-selected expectations of Q and P, according to (39) and (40) of Ferraz et al., assuming n=0.
    If rwa is False, then (57) and (58) will be used instead.
    """
    omega_as, taus = prepare_params_disorder(omega_as, taus)

    taus_shaped = taus.reshape((-1, 1, 1))
    times = omega_f * (t * 0.5 + taus_shaped)
    omega_times = np.angle(wvs) - times

    if not rwa:
        omega_times = omega_times - 0.5 * t * omega_as

    prefactor = 2 * g * t * np.sqrt(1 / 2)

    Q = prefactor * np.sqrt(1 / omega_f) * np.abs(wvs) * np.sin(omega_times)
    P = -1 * prefactor * np.sqrt(omega_f) * np.abs(wvs) * np.cos(omega_times)

    return Q, P


def recover_wvs_vacuum(qs, ps, omega_f, g, t, taus, omega_as=None, rwa=False):
    """Recover the weak value from the measured field quadrature expectations, assuming initial vacuum state."""
    omega_as, taus = prepare_params_disorder(omega_as, taus)

    taus_shaped = taus.reshape(-1, 1, 1)
    omega_times = omega_f * (t * 0.5 + taus_shaped)

    if not rwa:
        omega_times = omega_times - t * omega_as / 2

    qs_clean = qs * np.sqrt(omega_f) / (np.sqrt(2) * g * t)
    ps_clean = -1 * ps / (np.sqrt(2 * omega_f) * g * t)

    wv_mags = np.sqrt(qs_clean**2 + ps_clean**2)
    wv_phis = omega_times + np.arctan(qs_clean / ps_clean)

    wv_phis += np.pi * (np.sign(np.sin(wv_phis - omega_times)) != np.sign(qs_clean))

    return wv_mags * np.exp(1j * wv_phis)
