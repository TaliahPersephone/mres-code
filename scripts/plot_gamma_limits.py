"""Script for plottininit_beta Gamma in the dehpasing model"""

import matplotlib.pyplot as plt
import mpmath as mpm
import numpy as np
from matplotlib.widgets import Button, Slider

primary = "#2ea3a3"
secondary = "#2d59bc"
tertiary = "#b347ab"
dark = "#303a44"


def super_ohmic_th_limit(s, beta):

    z_re = 1 / (beta)
    if s > 2:
        return 2 * np.power(z_re, s - 1) * mpm.zeta(s - 1, 1 + z_re)
    elif s == 2:
        return -2 * z_re * mpm.harmonic(z_re)
    else:
        return np.inf


num_ss = 1e3
ss = np.linspace(1.1, 10, int(num_ss))
# ss = [1.8, 1.9, 1.95, 2.05, 2.1, 2.2]
alpha = 0.3
init_beta = 5
plot_exp = 1


gamma_vac = alpha * np.vectorize(mpm.gamma)(ss - 1)

vlimit = np.vectorize(super_ohmic_th_limit)
gamma_th = vlimit(ss, init_beta)
gamma = gamma_vac * (1 + gamma_th)

vexp = np.vectorize(mpm.exp)
gamma_exp = vexp(-gamma)
gamma_th_exp = vexp(-gamma_th)
gamma_vac_exp = vexp(-gamma_vac)

fig, ax = plt.subplots(figsize=(20, 10))

# line_collection.set_label([[r"$s={}$".format(np.round(s, 4))] for s in ss])
ax.set_xlabel(r"$s$")
ax.set_ylabel(r"$\lim_{t \to \infty}e^{-\gamma(t)}$" if plot_exp else r"$\lim_{t \to \infty}\gamma(t)$")
vabs = np.vectorize(mpm.fabs)

if plot_exp:
    gamma_plot = gamma_exp
    gamma_th_plot = gamma_th_exp
    gamma_vac_plot = gamma_vac_exp
else:
    gamma_plot = gamma
    gamma_th_plot = gamma_th
    gamma_vac_plot = gamma_vac

(th_line,) = ax.plot(
    ss,
    vabs(gamma_th_plot).astype(np.float64),
    color=secondary,
    # marker=".",
    linewidth=0.8,
    # edgecolors=None,
    label=r"$\gamma_{th}$",
)
ax.plot(
    ss,
    vabs(gamma_vac_plot).astype(np.float64),
    color=tertiary,
    # marker=".",
    linewidth=0.8,
    # edgecolors=None,
    label=r"$\gamma_{vac}$",
)
(gamma_line,) = ax.plot(
    ss,
    vabs(gamma_plot).astype(np.float64),
    color=primary,
    # marker=".",
    linewidth=1.5,
    # edgecolors=None,
    label=r"$\gamma$",
)

# Max of gamma_vac
vac_max_s = np.float64(mpm.exp(mpm.lambertw(0.5)))
ax.axvline(vac_max_s + 1, color=dark)

ax.set_xticks(range(1, 11))
ax.set_xticks(np.linspace(1, 10, 37), minor=True)
# adjust the main plot to make room for the sliders
fig.subplots_adjust(bottom=0.15)

# Make a horizontal slider to control the frequency.
axfreq = fig.add_axes([0.2, 0.07, 0.65, 0.03])
beta_ohm_slider = Slider(
    ax=axfreq,
    label=r"$\beta \Omega$",
    valmin=0.01,
    valmax=10,
    valinit=init_beta,
)


# The function to be called anytime a slider's value changes
def update(val):
    gamma_th = vlimit(ss, beta_ohm_slider.val)
    gamma = gamma_vac * (1 + gamma_th)

    if plot_exp:
        gamma_plot = vexp(-gamma)
        gamma_th_plot = vexp(-gamma_th)
    else:
        gamma_plot = gamma
        gamma_th_plot = gamma_th

    th_line.set_ydata(vabs(gamma_th_plot).astype(np.float64))
    gamma_line.set_ydata(vabs(gamma_plot).astype(np.float64))

    fig.canvas.draw_idle()


# register the update function with each slider
beta_ohm_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = fig.add_axes([0.8, 0.01, 0.1, 0.04])
button = Button(resetax, "Reset", hovercolor="0.975")


def reset(event):
    beta_ohm_slider.reset()


button.on_clicked(reset)
ax.legend()
ax.grid(True, "both")
plt.show()
