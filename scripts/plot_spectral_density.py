"""Script for plotting Ohmic spectral densities"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.widgets import Button, Slider


# The parametrized function to be plotted
def f(omegas, omega_c, alpha):
    J_ws = (
        2
        * alpha
        * np.power(omega_c, 1 - ss).reshape((-1, 1))
        * omegas_ss
        * np.exp(-omegas / omega_c)
    )
    return np.dstack((omegas_rep, J_ws))


# Define initial parameters
init_omega_c = 100
init_alpha = 0.3
init_xscale = 2e3

num_ss = 28
ss = np.linspace(0.2, 5.6, num_ss)
omegas = np.linspace(0, init_xscale, int(1e4))
omegas_rep = np.repeat(omegas.reshape(1, -1), len(ss), 0)
omegas_ss = np.power(omegas, ss.reshape(-1, 1))

cmap = plt.get_cmap("turbo")
norm = Normalize(min(ss), max(ss))

colors = cmap(norm(ss))

# Create the figure and the line that we will manipulate
fig, ax = plt.subplots(figsize=(20, 10))
line_collection = LineCollection(
    f(omegas, init_omega_c, init_alpha),
    colors=colors,
    linewidths=0.5,
)

# line_collection.set_label([[r"$s={}$".format(np.round(s, 4))] for s in ss])
ax.set_xlabel(r"$\omega$")
ax.set_ylabel(r"$J(\omega)$")
ax.add_collection(line_collection)
ax.set_xlim(0, init_xscale)
ax.set_ylim(0, 2000)

cbar = fig.colorbar(None, ax=ax, cmap=cmap, norm=norm, ticks=ss, label="s")

# adjust the main plot to make room for the sliders
fig.subplots_adjust(left=0.1, bottom=0.15)

# Make a horizontal slider to control the frequency.
axfreq = fig.add_axes([0.2, 0.07, 0.65, 0.03])
alpha_slider = Slider(
    ax=axfreq,
    label=r"$\alpha$",
    valmin=0.01,
    valmax=1,
    valinit=init_alpha,
)

# Make a vertically oriented slider to control the amplitude
axamp = fig.add_axes([0.03, 0.2, 0.0225, 0.63])
omega_c_slider = Slider(
    ax=axamp,
    label=r"$\omega_c$",
    valmin=1,
    valmax=2000,
    valinit=init_omega_c,
    orientation="vertical",
)


# The function to be called anytime a slider's value changes
def update(val):
    line_collection.set_segments(f(omegas, omega_c_slider.val, alpha_slider.val))
    fig.canvas.draw_idle()


# register the update function with each slider
omega_c_slider.on_changed(update)
alpha_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = fig.add_axes([0.8, 0.01, 0.1, 0.04])
button = Button(resetax, "Reset", hovercolor="0.975")


def reset(event):
    omega_c_slider.reset()
    alpha_slider.reset()


button.on_clicked(reset)

plt.show()
