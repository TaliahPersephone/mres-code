"""Script for plotting the hurwitz zeta function"""

import matplotlib.pyplot as plt
import mpmath as mpm
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.widgets import Button, Slider


def calculate_zetas(beta):
    z_re = 1 / (beta)
    z = z_re + 1j * times / beta

    zetas = (2 * vzeta(ss, 1 + z) + np.power(z, -ss)) * (ss - 1) / beta
    im_zetas = np.dstack((times_rep, vim(zetas).astype(np.float32)))

    zetas = (2 * vzeta(ss - 1, 1 + z) + np.power(z, 1 - ss))
    re_zetas = np.dstack((times_rep, vre(zetas).astype(np.float32)))

    return re_zetas, im_zetas


num_ss = 11
ss = np.linspace(2.5, 3, int(num_ss))
ss = np.array([ss]).reshape((-1, 1))
init_beta = 5

min_time = 1.5
max_time = 5
times = np.linspace(min_time, max_time, int(1e2))
times = times.reshape((1, -1))
times_rep = np.repeat(times.reshape(1, -1), len(ss), 0)

vzeta = np.vectorize(mpm.zeta)
vre = np.vectorize(mpm.re)
vim = np.vectorize(mpm.im)

values = calculate_zetas(init_beta)

fig, axs = plt.subplots(1, 2, figsize=(20, 10))

cmap = plt.get_cmap("turbo")
norm = Normalize(min(ss), max(ss))

colors = cmap(norm(ss))

axs[0].set_xlabel(r"$t$")
axs[1].set_xlabel(r"$t$")
axs[0].set_ylabel(r"$Re[2\zeta(s - 1, \alpha) - \alpha^{1 - s}]$")
axs[1].set_ylabel(r"$(s-1)\beta^{-1}Im[2\zeta(s, \alpha) - \alpha^{- s}]$")


re_collection = LineCollection(
    values[0],
    colors=colors,
    linewidths=0.5,
)
axs[0].add_collection(re_collection)
axs[0].add_collection(re_collection)
axs[0].set_xlim(min_time, max_time)
axs[0].set_ylim(1.1 * np.min(values[0][:, :, 1]), 1.1 * np.max(values[0][:, :, 1]))
axs[0].grid(True, "both")

im_collection = LineCollection(
    values[1],
    colors=colors,
    linewidths=0.5,
)
axs[1].add_collection(im_collection)
axs[1].set_xlim(min_time, max_time)
axs[1].set_ylim(1.1 * np.min(values[1][:, :, 1]), 1.1 * np.max(values[1][:, :, 1]))
axs[1].grid(True, "both")

# adjust the main plot to make room for the sliders
fig.subplots_adjust(bottom=0.15)

cbar = fig.colorbar(
    None, ax=axs.ravel().tolist(), cmap=cmap, norm=norm, ticks=ss.flatten(), label="s"
)

# Make a horizontal slider to control the frequency.
axfreq = fig.add_axes([0.2, 0.07, 0.65, 0.03])
beta_slider = Slider(
    ax=axfreq,
    label=r"$\beta$",
    valmin=0.01,
    valmax=10,
    valinit=init_beta,
)


# The function to be called anytime a slider's value changes
def update(val):
    values = calculate_zetas(beta_slider.val)

    re_collection.set_segments(values[0])
    im_collection.set_segments(values[1])

    fig.canvas.draw_idle()


# register the update function with each slider
beta_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = fig.add_axes([0.8, 0.01, 0.1, 0.04])
button = Button(resetax, "Reset", hovercolor="0.975")


def reset(event):
    beta_slider.reset()


button.on_clicked(reset)

plt.show()
