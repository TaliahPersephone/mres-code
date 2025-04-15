"""Class representing the interaction picture representation of a bloch vector under a given Hamiltonian."""

import mpmath as mpm
import numpy as np


def bv_int_sigmaz_disorder(v, **kwargs):
    """Return the interaction picture representations of a bloch vector, given H_0 = 0.5 omega_a sigmaz."""
    """At all omega_as and times. According to (42) of Ferraz et al. (2024)."""
    times = np.einsum("a,bc", kwargs["t"] + kwargs["taus"], kwargs["omega_as"])
    sin_times = np.sin(times)
    cos_times = np.cos(times)

    v_int_x = v[0] * cos_times + v[1] * sin_times
    v_int_y = v[1] * cos_times - v[0] * sin_times
    v_int_z = np.broadcast_to(v[2], v_int_x.shape)

    return np.stack([v_int_x, v_int_y, v_int_z], axis=-1)


def bv_int_fixed(v, **kwargs):
    """Return the interaction picture representations of a bloch vector, assuming they are fixed - i.e. do nothing."""
    return np.broadcast_to(v, (*kwargs["times"].shape, 3))


def calculate_vgammas(v_ints, taus, gamma=0):
    """Return vgamma_int, according to (43) of Ferraz et al. (2024)"""
    e_gtaus = np.exp(-0.5 * gamma * taus)
    gamma_terms = np.stack([e_gtaus, e_gtaus, e_gtaus**2], axis=-1).reshape(
        (-1, 1, 1, 3)
    )

    return v_ints * gamma_terms


def calculate_c_vecs(c1s, c0):
    """Calculate the expectations of sigma{x,y,z} from the solution to the two level decay B&P model."""
    c0c1_stars = c0 * np.conjugate(c1s)

    c0c1_x = 2 * np.real(c0c1_stars)
    c0c1_y = 2 * np.imag(c0c1_stars)
    c0c1_z = 2 * np.power(np.abs(c1s), 2) - 1

    return np.stack([c0c1_x, c0c1_y, c0c1_z], axis=-1)


def calculate_pure_dephasing_vecs(gamma_ts, u0):
    """Calculate the expectations of sigma{x,y,z} from the solution to the pure dephasing model, given decay rates."""
    vexp = np.vectorize(mpm.exp)
    e_gammas = vexp(-gamma_ts)

    u_xs = u0[0] * e_gammas
    u_ys = u0[1] * e_gammas
    u_zs = np.full(e_gammas.shape, u0[2])

    return np.stack([u_xs, u_ys, u_zs], axis=-1)
