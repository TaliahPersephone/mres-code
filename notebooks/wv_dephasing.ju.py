"""
# %%

%load_ext autoreload
%autoreload 2

# %% [md]
"""

# %%


import matplotlib.pyplot as plt
import mpmath as mpm
import numpy as np

from mres import plot_dephasing
from mres.pipelines import DephasingPipeline
from mres.wvformulas import wv_pure_dephasing_x, wv_pure_dephasing_z

# %%

delta = 1e-2
cos_d = np.cos(delta)
sin_d = np.sin(delta)
sq2 = np.sqrt(2)

# %% [md]

## $\hat{\sigma}_z$

# %%

theta_u = np.pi * 35 / 180
theta_v = np.pi / 4


params = {
    "u": np.array([-np.sin(theta_u * 2), 0, np.cos(theta_u * 2)]),
    "v": np.array([np.sin(theta_v * 2), 0, np.cos(theta_v * 2)]),
    "m": np.array([0, 0, 1]),
    "times": np.linspace(0, 20, int(1e4)),
    "spectral_params": {
        "s": [0.5, 1, 3],
        "gamma": 0.3,
        "ohm": 1,
        "beta": 5,
    },
}

dephasing_pipeline_z = DephasingPipeline(params)
results_z = dephasing_pipeline_z.run()
plot_dephasing(results_z, save_fig="figures/dephasing_sigmaz_reproduction.pdf")

# %%


delta = 1e-3

params = {
    "u": np.array([1, 0, 0]),
    "v": np.array([-1 / sq2, 0, 1 / sq2]),
    "m": np.array([0, 0, 1]),
    "times": np.linspace(0, 20, int(1e4)),
    "spectral_params": {"s": [0.5, 1, 3], "gamma": 0.3, "ohm": 1, "beta": 5},
}

dephasing_pipeline_z = DephasingPipeline(params)
results_z = dephasing_pipeline_z.run()
plot_dephasing(results_z, save_fig="figures/dephasing_sigmaz_uplus.pdf")


# %% [md]

## $\hat{\sigma}_x$

# %%

theta_u = np.pi * (90 - 33) / 180
theta_v = np.pi * (90 - 77) / 180

params = {
    "u": np.array([np.sin(theta_u * 2), 0, np.cos(theta_u * 2)]),
    "v": np.array([-np.sin(theta_v * 2), 0, np.cos(theta_v * 2)]),
    "m": np.array([1, 0, 0]),
    "times": np.linspace(0, 20, int(1e4)),
    "spectral_params": {
        "s": [0.5, 1, 3],
        "gamma": 0.3,
        "ohm": 1,
        "beta": 5,
        "tm_ratio": 0.5,
    },
}

dephasing_pipeline_x = DephasingPipeline(params)
results_x = dephasing_pipeline_x.run()
plot_dephasing(results_x, save_fig="figures/dephasing_sigmax_reproduction.pdf")

# %%

delta = 1e-1

params = {
    "u": np.array([0, 1, 0]),
    "v": np.array([0, -np.cos(delta), np.sin(delta)]),
    "m": np.array([1, 0, 0]),
    "times": np.linspace(0, 20, int(1e2)),
    "spectral_params": {
        "s": [0.5, 1, 3, 4],
        "gamma": 0.3,
        "ohm": 1,
        "beta": 5,
        "tm_ratio": 0.1,
    },
}

dephasing_pipeline_x = DephasingPipeline(params)
results_x = dephasing_pipeline_x.run()
plot_dephasing(results_x)

# %%

params = {
    "u": np.array([0, np.sin(1e-2), -np.cos(1e-2)]),
    "v": np.array([0, 0, 1]),
    "m": np.array([1, 0, 0]),
    "times": np.linspace(2.5, 20, int(1e2)),
    "spectral_params": {
        "s": [1, 3, 4, 5],
        "gamma": 0.3,
        "ohm": 1,
        "beta": 1,
        "tm_ratio": 0.5,
    },
}

dephasing_pipeline_x = DephasingPipeline(params)
results_x = dephasing_pipeline_x.run()
plot_dephasing(results_x, save_fig="figures/dephasing_sigmax_deltaminus3.pdf")

# %%

mpm.gamma(5)
# %%

delta = 1e-2

params = {
    "u": np.array([0, 1 / sq2, 1 / sq2]),
    "v": np.array([0, -1 / sq2, -1 / sq2]),
    "m": np.array([1, 0, 0]),
    "times": np.linspace(1, 20, int(1e4)),
    "spectral_params": {
        "s": [0.5, 1, 3],
        "gamma": 0.3,
        "ohm": 1,
        "beta": 5,
        "tm_ratio": 0.9,
    },
}

dephasing_pipeline_x = DephasingPipeline(params)
results_x = dephasing_pipeline_x.run()
plot_dephasing(results_x, save_fig="figures/dephasing_sigmax_plusminusi.pdf")

# %%
np.sin(np.pi / 4)

# %%

deltas = np.linspace(1e-2, 2 * np.pi - 1e-2, int(1e2) - 1)
phi_t = results_x["phi_ts"][-1, :, -1].reshape(1, 3, 1)
gamma_t = results_x["gamma_ts"][-1, :, -1].reshape(1, 3, 1)
v = np.array([0, np.sin(np.pi*0.9), np.cos(np.pi*0.9)])
thetas = np.stack(
    [
        np.sin(deltas),
        np.sin(deltas),
        np.cos(deltas),
    ]
)
phis = np.stack(
    [
        np.cos(deltas),
        np.sin(deltas),
        np.ones_like(deltas),
    ]
)
us = np.einsum("...i,...j->...ij", thetas, phis)
wvs = wv_pure_dephasing_x(v, us, gamma_t, phi_t)
wvs_abs = (np.vectorize(mpm.fabs)(wvs)).astype(np.float64)

print(us[:, *np.unravel_index(np.argmax(wvs_abs), wvs_abs.shape)])

fig, ax = plt.subplots()
c = ax.pcolormesh(deltas, deltas, wvs_abs, cmap="plasma")
# c = ax.imshow(wvs_abs<6, cmap="plasma")
fig.colorbar(c, ax=ax)

# %%

us[:,88,23], wvs_abs[88,23]

# %%

np.unravel_index(np.argmax(wvs_abs), wvs_abs.shape)

# %%

wvs_abs.shape

# %%

thetas = np.stack(
    [
        np.sin(deltas),
        np.sin(deltas),
        np.cos(deltas),
    ]
)
phis = np.stack(
    [
        np.cos(deltas),
        np.sin(deltas),
        np.ones_like(deltas),
    ]
)
thetas.shape

# %%

us[:, 0, 0]

# %%

wvs_abs.astype(np.float64)

# %%

# vs = np.array(
#     [
#         [
#             0,
#             -np.sin(delta),
#             np.cos(delta),
#         ]
#         for delta in deltas
#     ]
# )
# us = np.array(
#     [
#         [
#             0,
#             np.sin(delta),
#             np.cos(delta),
#         ]
#         for delta in deltas[5:-5]
#     ]
# ).T
