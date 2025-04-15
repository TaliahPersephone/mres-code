"""Function for simulating system bath evolution."""

import mpmath as mpm
import numpy as np

from mres.bvevo import calculate_c_vecs


def solvable_ohmic(times, c1_0, c0, ohm, omega_0, gamma):
    """Solve for c1_t given an Ohmic spectral density of the form"""
    """(gamma/(pi omega_0)) * omega * (ohm^2 / (ohm^2 + omega^2))"""
    d = np.emath.sqrt(
        np.power(ohm - 1j * omega_0, 2) + 1j * 4 * gamma * np.power(ohm, 2) / omega_0
    )

    exp_times = np.exp(-(ohm - 1j * omega_0) * 0.5 * times)

    hyp_times = 0.5 * d * times
    cosh_times = np.cosh(hyp_times)
    sinh_times = np.sinh(hyp_times)

    c1_ts = c1_0 * exp_times * (cosh_times + sinh_times * (ohm + 1j * omega_0) / d)

    return c1_ts, calculate_c_vecs(c1_ts, c0)


def solvable_ohmic_shifted(times, c1_0, c0, ohm, omega_0, gamma):
    """Solve for c1_t given an Ohmic spectral density of the form"""
    """(gamma/(pi omega_0)) * (omega-omega_0) * (ohm^2 / (ohm^2 + (omega-omega_0)^2))"""
    d = ohm * np.emath.sqrt(1 - 1 * 4 * gamma / omega_0)

    exp_times = np.exp(-ohm * 0.5 * times)

    hyp_times = 0.5 * d * times
    cosh_times = np.cosh(hyp_times)
    sinh_times = np.sinh(hyp_times)

    c1_ts = c1_0 * exp_times * (cosh_times + sinh_times * ohm / d)

    return c1_ts, calculate_c_vecs(c1_ts, c0)


def solvable_jaynes_cummings(times, c1_0, c0, lam, gamma_0):
    """Solve for c1_t given a frequency shifted JC spectral density"""
    """(1/(2 pi)) * gamma * lambda^2 / ((omega_0 - omega)^2 + lambda^2)"""
    d = np.emath.sqrt(np.power(lam, 2) - 2 * gamma_0 * lam)

    exp_times = np.exp(-lam * 0.5 * times)

    hyp_times = 0.5 * d * times
    cosh_times = np.cosh(hyp_times)
    sinh_times = np.sinh(hyp_times)

    c1_ts = c1_0 * exp_times * (cosh_times + sinh_times * lam / d)

    return c1_ts, calculate_c_vecs(c1_ts, c0)


def s3_ohmic_decoherance_function(times, alpha, ohm, beta):
    """Solve for the decoherence function of a super ohmic (s=3) spectral density in a purely dephasing model"""
    z_re = 1 / (ohm * beta)
    z = z_re + 1j * times / beta

    vzeta = np.vectorize(mpm.zeta)

    h_zeta_0 = vzeta(2, 1 + z_re)
    h_zeta_1 = vzeta(2, 1 + z)
    h_zeta_2 = vzeta(2, 1 + np.conj(z))
    h_zeta_term = 2 * h_zeta_0 - h_zeta_1 - h_zeta_2

    thermal_term = np.power(z_re, 2) * h_zeta_term

    ohm_t_squared = np.power(ohm * times, 2)
    vacuum_term = 1 - (1 - ohm_t_squared) / np.power(ohm_t_squared + 1, 2)

    return alpha * (thermal_term + vacuum_term)


def ohmic_decoherence_function(times, s, alpha, ohm, beta, tm_ratio=None):
    """Solve for the decoherence function of an ohmic spectral density in a purely dephasing model"""
    if tm_ratio is not None:
        times = np.stack([times, times * tm_ratio, times * (1 - tm_ratio)], axis=0)

    z_re = 1 / (ohm * beta)
    z = z_re + 1j * times / beta
    ohm_t_sq = 1 + np.power(ohm * times, 2)
    trig_term = np.arctan(ohm * times)

    vgamma = np.vectorize(mpm.gamma)

    if np.isin(1, s):
        vlog = np.vectorize(mpm.log)
        vloggamma = np.vectorize(mpm.loggamma)
        vabs = np.vectorize(mpm.fabs)
        vacuum_term = 0.5 * vlog(ohm_t_sq)
        thermal_term = 2 * (vloggamma(1 + z_re) - vlog(vabs(vgamma(1 + z))))

        gamma_ts_s1 = alpha * (vacuum_term + thermal_term)

        if gamma_ts_s1.ndim > 2:
            gamma_ts_s1 = np.swapaxes(gamma_ts_s1, 0, 1)

        if tm_ratio is not None:
            phi_ts_s1 = np.swapaxes(alpha * trig_term, 0, 1)

    if not np.isin(1, s) or s.size > 1:
        ss_slice = np.where(s != 1)
        ss = s[ss_slice].reshape(-1, 1)

        gamma_ss_m1 = vgamma(ss - 1)
        gamma_func_term = ss * (ss - 1) * gamma_ss_m1 * gamma_ss_m1 / vgamma(ss + 1)

        if z_re == 0:
            thermal_term = 0
        else:
            vzeta = np.vectorize(mpm.zeta)

            h_zeta_0 = vzeta(ss - 1, 1 + z_re)
            h_zeta_1 = vzeta(ss - 1, 1 + z)
            h_zeta_2 = vzeta(ss - 1, 1 + np.conj(z))
            h_zeta_term = 2 * h_zeta_0 - h_zeta_1 - h_zeta_2

            thermal_term = np.power(z_re, ss - 1) * gamma_func_term * h_zeta_term

        trig_term = np.arctan(ohm * times)

        vacuum_term = vgamma(ss - 1) * (
            1 - np.cos((ss - 1) * trig_term) / np.power(ohm_t_sq, (ss - 1) / 2)
        )

        gamma_ts = alpha * (thermal_term + vacuum_term)

        if tm_ratio is not None:
            gamma_ts = np.swapaxes(gamma_ts, 0, 1)
            phi_ts = (
                gamma_func_term * alpha * np.sin(trig_term) / np.power(ohm_t_sq, (ss - 1) / 2)
            )
            phi_ts = np.swapaxes(phi_ts, 0, 1)

            if np.isin(1, s):
                phi_ts = np.insert(phi_ts, np.where(s == 1)[0], phi_ts_s1, axis=0)
        else:
            phi_ts = None

        if np.isin(1, s):
            gamma_ts = np.insert(gamma_ts, np.where(s == 1)[0], gamma_ts_s1, axis=0)
    else:
        gamma_ts = gamma_ts_s1
        phi_ts = phi_ts_s1 if tm_ratio is not None else None

    return gamma_ts, phi_ts
