"Prepare parameters for pipelines."

import numpy as np


def prepare_params_disorder(omega_as, taus):
    """Wrap arguments in arrays, and mesh times."""
    omega_as_array = np.array([omega_as]) if np.isscalar(omega_as) else omega_as

    omega_as_array = (
        omega_as_array.reshape((1, -1)) if omega_as_array.ndim == 1 else omega_as_array
    )

    taus_array = np.array([taus]) if np.isscalar(taus) else taus

    return omega_as_array, taus_array


def prepare_params_decoherence(times, ohms, omega_0s, gammas):
    """Wrap arguments in arrays, and mesh times."""
    omega_0s_array = np.array([omega_0s]) if np.isscalar(omega_0s) else omega_0s

    omega_0s_array = (
        omega_0s_array.reshape((1, -1)) if omega_0s_array.ndim == 1 else omega_0s_array
    )

    times_array = np.array([times]) if np.isscalar(times) else times

    return omega_0s_array, times_array


def prepare_params_dephasing(times, ss):
    """Wrap arguments in arrays."""

    times = times.reshape((1, -1)) if times.ndim == 1 else times

    ss = np.array([ss]).reshape((-1, 1))

    return times, ss
