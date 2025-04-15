"""Formulas for calculating weak values"""

import mpmath as mpm
import numpy as np

from mres.bvevo import bv_int_sigmaz_disorder, calculate_vgammas
from mres.paramprep import prepare_params_disorder


def wv_ferraz(u, v_ints, vgamma_ints, m_ints, a=0, b=1):
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
    omega_as, taus = prepare_params_disorder(omega_as, taus)

    v_ints = bv_int_sigmaz_disorder(v, omega_as, t, taus)
    m_ints = bv_int_sigmaz_disorder(m, omega_as, 0.5 * t, 0) if m_to_int else m

    vgamma_ints = calculate_vgammas(v_ints, taus, gamma)

    return wv_ferraz(u, v_ints, vgamma_ints, m_ints, a, b)


def wv_decoherence_simple(v_ints, m_ints, c_vecs, a=0, b=1):
    """Calculate the weak value according to my current formula"""
    v_dot_m = np.einsum("...j,...j", v_ints, m_ints)
    m_dot_c = np.einsum("...j,...j", m_ints, c_vecs)
    v_cross_m_dot_c = 1j * np.einsum("...j,...j", np.cross(v_ints, m_ints), c_vecs)

    denominator = 1 + np.einsum("...j,...j", v_ints, c_vecs)

    return a + b * (v_dot_m + m_dot_c + v_cross_m_dot_c) / denominator


def wv_pure_dephasing_z(v, u, gamma_ts):
    """Calculate the weak value of sigma z in the pure dephasing model"""
    vexp = np.vectorize(mpm.exp)
    e_gammas = vexp(-gamma_ts)

    numerator = v[2] + u[2] + 1j * (v[1] * u[0] - v[0] * u[1]) * e_gammas
    denominator = 1 + v[2] * u[2] + (v[0] * u[0] + v[1] * u[1]) * e_gammas

    return numerator / denominator


def wv_pure_dephasing_x(v, u, gamma_ts, phi_ts):
    """Calculate the weak value of sigma x in the pure dephasing model"""
    vexp = np.vectorize(mpm.exp)
    e_gammas = vexp(-gamma_ts).reshape(gamma_ts.shape)

    psi_ts = phi_ts[:, 1] - phi_ts[:, 0] + phi_ts[:, 2]
    e_phis = vexp(-1j * psi_ts)

    numerator = (u[0] + 1j * v[2] * u[1]) * e_gammas[:, 1] + (
        v[0] + 1j * v[1] * u[2]
    ) * e_gammas[:, 2] * e_phis
    denominator = 1 + v[2] * u[2] + (v[0] * u[0] + v[1] * u[1]) * e_gammas[:, 0]

    return numerator / denominator
