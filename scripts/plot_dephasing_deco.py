"""Script for plotting Gamma in the dehpasing model"""

import matplotlib.pyplot as plt
import mpmath as mpm
import numpy as np
from matplotlib.colors import Normalize


def calculate_gamma_th(beta):
    z_re = 1 / beta
    z = z_re + 1j * times / beta

    if np.isin(1, ss):
        th_s1 = 2 * alpha * (vloggamma(1 + z_re) - vlog(vabs(vgamma(1 + z))))

    if np.isin(2, ss):
        th_s2 = - 2 * alpha * (vharm(z_re) - vre(vharm(z))) * z_re

    if ss_slice.any():
        h_zeta_re = vzeta(ss_slice - 1, 1 + z_re)
        h_zeta = vzeta(ss_slice - 1, 1 + z)
        h_zeta_term = 2 * (h_zeta_re - vre(h_zeta))

        # gamma_func_term = gamma_func_slices * gamma_func_slices * ss_slice * (ss_slice - 1) / vgamma(ss_slice + 1)
        th_ss = alpha * np.power(z_re, ss_slice - 1) * gamma_func_slices * h_zeta_term

        if np.isin(1, ss):
            th_ss = np.insert(th_ss, np.where(ss == 1)[0], th_s1, axis=0)
        if np.isin(2, ss):
            th_ss = np.insert(th_ss, np.where(ss == 2)[0], th_s2, axis=0)
    else:
        th_ss = th_s1

    return th_ss


plot_exp = 0
plot_breakdown = 0

times = np.linspace(1, 3, int(1e2))
times = times.reshape((1, -1))

num_ss = 11
ss = np.linspace(2, 3, num_ss)
ss = np.array([ss]).reshape((-1, 1))

if not np.isin(1, ss) or ss.size > 1:
    ss_slice = np.where(ss != 1)
    ss_slice = ss[ss_slice].reshape(-1, 1)
else:
    ss_slice = np.array([0])

alpha = 0.3
beta = 5

z_re = 1 / beta
z = z_re + 1j * times / beta
ohm_t_sq = 1 + np.power(times, 2)

# Vectorize mpmath functions
vlog = np.vectorize(mpm.log)
vloggamma = np.vectorize(mpm.loggamma)
vabs = np.vectorize(mpm.fabs)
vre = np.vectorize(mpm.re)
vim = np.vectorize(mpm.im)
vzeta = np.vectorize(mpm.zeta)
vgamma = np.vectorize(mpm.gamma)
vexp = np.vectorize(mpm.exp)
vharm = np.vectorize(mpm.harmonic)

gamma_func_slices = vgamma(ss_slice - 1)

# Vacuum terms
if np.isin(1, ss):
    vac_s1 = alpha * 0.5 * vlog(ohm_t_sq)
else:
    vac_s1 = 0

if ss_slice.any():
    vac_ss = alpha * gamma_func_slices * (1 - vre(np.power(1 + 1j * times, 1 - ss_slice)))

    if np.isin(1, ss):
        vac_ss = np.insert(vac_ss, np.where(ss == 1)[0], vac_s1, axis=0)
else:
    vac_ss = vac_s1

if np.isin(2, ss_slice):
    ss_no2 = np.where(ss_slice != 2)
    ss_slice = ss_slice[ss_no2].reshape(-1, 1)
    gamma_func_slices = gamma_func_slices[ss_no2].reshape(-1, 1)

gamma_th = calculate_gamma_th(beta)

# gamma_th = (
#     2 * vzeta(ss_slice, z) - np.power(z, -ss_slice)
# )
# gamma_th = (
#     2 * (vzeta(ss_slice, 1 + z) - vzeta(ss_slice, 1 + np.conj(z)))
#     + np.power(z, -ss_slice)
#     - np.power(np.conj(z), -ss_slice)
# )

gamma_vac = vac_ss

# gamma_vac = (
#     2 * ss_slice * (ss_slice - 1) * vgamma(ss_slice - 1) / vgamma(ss_slice + 1)
# ) * vzeta(ss_slice, 1 + np.conj(z)) + np.power(np.conj(z), -ss_slice)

gammas = gamma_th + gamma_vac

plot_gamma = vexp(-gammas) if plot_exp else gammas
plot_gamma_vac = vexp(-gamma_vac) if plot_exp else gamma_vac
plot_gamma_th = vexp(-gamma_th) if plot_exp else gamma_th

fig, ax = plt.subplots(figsize=(20, 10))

ax.set_xlabel(r"$\Omega t$")
ax.set_ylabel(r"$e^{-\gamma(t)}$" if plot_exp else r"$\gamma(t)$")

cmap = plt.get_cmap("turbo")
norm = Normalize(min(ss), max(ss))

colors = cmap(norm(ss))

for idx, s in enumerate(ss):
    ax.plot(
        times.flatten(),
        vabs(plot_gamma[idx]).astype(np.float64),
        color=colors[idx],
        markevery=None,
        lw=1,
        mec=None,
        label=r"$s={}$".format(s),
    )
    if plot_breakdown:
        ax.plot(
            times.flatten(),
            vabs(plot_gamma_vac[idx]).astype(np.float64),
            color=colors[idx],
            markevery=None,
            ls="--",
            lw=1,
            mec=None,
        )
        ax.plot(
            times.flatten(),
            vabs(plot_gamma_th[idx]).astype(np.float64),
            color=colors[idx],
            markevery=None,
            ls=":",
            lw=1,
            mec=None,
        )

# ax.set_yscale("log")
# ax.set_xscale("log")

ax.legend()
plt.show()
