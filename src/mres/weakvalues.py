"""Weak value calculations"""

import numpy as np


def prepare_params(omega_as, taus):
    """Wrap arguments in arrays, and mesh times."""
    omega_as_array = np.array([omega_as]) if np.isscalar(omega_as) else omega_as

    omega_as_array = (
        omega_as_array.reshape((1, -1)) if omega_as_array.ndim == 1 else omega_as_array
    )

    taus_array = np.array([taus]) if np.isscalar(taus) else taus

    return omega_as_array, taus_array


def bvs_interaction_picture(v, omega_as, t, taus):
    """Return the interaction picture representation of a bloch vector, according to (42) of Ferraz et al. (2024)"""
    times = np.einsum("a,bc", t + taus, omega_as)
    sin_times = np.sin(times)
    cos_times = np.cos(times)

    v_int_x = v[0] * cos_times + v[1] * sin_times
    v_int_y = v[1] * cos_times - v[0] * sin_times
    v_int_z = np.broadcast_to(v[2], v_int_x.shape)

    return np.stack([v_int_x, v_int_y, v_int_z], axis=-1)


def calculate_vgammas(v_ints, taus, gamma=0):
    """Return vgamma_int, according to (43) of Ferraz et al. (2024)"""
    e_gtaus = np.exp(-0.5 * gamma * taus)
    gamma_terms = np.stack([e_gtaus, e_gtaus, e_gtaus**2], axis=-1).reshape(
        (-1, 1, 1, 3)
    )

    return v_ints * gamma_terms


def wvs_ferraz(u, v_ints, vgamma_ints, m_ints, a=0, b=1):
    """Return weak value according to (47) of Ferraz et al. (2024)"""
    v_z_terms = 1 + vgamma_ints[..., 2] - v_ints[..., 2]

    numerator = np.einsum("...j,...j", vgamma_ints, m_ints)
    numerator += np.einsum("j,...j", u, m_ints) * v_z_terms
    numerator += 1j * np.einsum("...j,...j", vgamma_ints, np.cross(m_ints, u))

    denominator = v_z_terms + np.einsum("...j,j", vgamma_ints, u)

    wv = a + b * numerator / denominator

    return wv


def calculate_wvs_ferraz(u, v, m, omega_as, t, taus, gamma=0, m_to_int=True, a=0, b=1):
    """Calculate weak values across omega_as"""
    omega_as, taus = prepare_params(omega_as, taus)

    v_ints = bvs_interaction_picture(v, omega_as, t, taus)
    m_ints = bvs_interaction_picture(m, omega_as, 0.5 * t, 0) if m_to_int else m

    vgamma_ints = calculate_vgammas(v_ints, taus, gamma)

    return wvs_ferraz(u, v_ints, vgamma_ints, m_ints, a, b)


def field_quadrature_expectations(wvs, omega_f, g, t, taus, omega_as=None, rwa=False):
    """
    Return post-selected expectations of Q and P, according to (39) and (40) of Ferraz et al., assuming n=0.
    If rwa is False, then (57) and (58) will be used instead.
    """
    omega_as, taus = prepare_params(omega_as, taus)

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
    omega_as, taus = prepare_params(omega_as, taus)

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


def wv_pipeline(params, field_quads=True):
    """Calculate disordered wvs and quads"""
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

    results = {
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

    return results
