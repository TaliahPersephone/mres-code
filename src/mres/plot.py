import matplotlib.pyplot as plt
import mpmath as mpm
import numpy as np
from matplotlib.gridspec import GridSpec

primary = "#2ea3a3"
secondary = "#2d59bc"
tertiary = "#b347ab"
quartery = "#9389ec"
extra_1 = "#79ac62"
extra_2 = "#91b2eb"
dark = "#303a44"
colours = [primary, secondary, tertiary, quartery, extra_1, extra_2]


def setup_complex_plot(xs, yss, line_label, x_label, y_label, x_logscale, y_logscale):
    fig = plt.figure(figsize=(16, 10), layout="constrained")
    gs = GridSpec(2, 2, figure=fig)

    ax00 = fig.add_subplot(gs[0, :])
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])

    ax00.set_xlabel(r"${}$".format(x_label), fontsize="xx-large")
    ax00.set_ylabel(
        r"$|{}|$".format(y_label),
        fontsize="xx-large",
    )
    ax10.set_xlabel(r"${}$".format(x_label), fontsize="xx-large")
    ax10.set_ylabel(
        r"$Re[{}]$".format(y_label),
        fontsize="xx-large",
    )
    ax11.set_xlabel(r"${}$".format(x_label), fontsize="xx-large")
    ax11.set_ylabel(
        r"$Im[{}]$".format(y_label),
        fontsize="xx-large",
    )

    scatter_options = {"marker": ".", "linewidths": 0, "edgecolors": None}

    for idx, label_param in enumerate(yss["labels"]):
        scatter_options["color"] = colours[idx]

        ax00.scatter(
            xs,
            yss["abs"][idx],
            label=r"${}$".format(line_label.format(label_param)),
            **scatter_options
        )
        ax10.scatter(xs, yss["real"][idx], **scatter_options)
        ax11.scatter(xs, yss["imag"][idx], **scatter_options)

    if x_logscale:
        ax00.set_xscale("log")
        ax10.set_xscale("log")
        ax11.set_xscale("log")
    if y_logscale:
        ax00.set_yscale("log")
        ax10.set_yscale("log")
        ax11.set_yscale("log")

    if yss["labels"].size > 1:
        ax00.legend(markerscale=3, fontsize="xx-large")

    return fig, (ax00, ax10, ax11)


def plot_fixed_results(results, plot_field_quads=None, log_yscale=False):
    params = results["params"]
    omega_as = params["omega_as"]
    omega_as = omega_as[0] if omega_as.ndim == 2 else omega_as
    omega_a_0 = params["omega_a_0"]
    # omega_f = params['omega_f']
    # g = params['g']
    # t = params['t']

    wvs = results["wvs"][0, 0]
    wvs_mean = results["wv"][0, 0]

    wv_0 = results["wv_0"][0, 0]
    # wv_0_tau = results['wv_0_tau']
    wv_0_0 = results["wv_0_0"][0, 0]

    plot_field_quads = plot_field_quads or (results["qs"] is not None)

    if plot_field_quads:
        qs = results["qs"][0, 0]
        qs_mean = results["q"][0, 0]
        ps = results["ps"][0, 0]
        ps_mean = results["p"][0, 0]
        q_0 = results["q_0"][0, 0]
        p_0 = results["p_0"][0, 0]

    fig = plt.figure(figsize=(12, 12), layout="constrained")
    gs = GridSpec(3 if plot_field_quads else 2, 3, figure=fig)

    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax02 = fig.add_subplot(gs[0, 2])
    if plot_field_quads:
        ax10 = fig.add_subplot(gs[1, 0])
        ax11 = fig.add_subplot(gs[1, 1])
        ax12 = fig.add_subplot(gs[1, 2])
    ax21 = fig.add_subplot(gs[-1, :])

    # wvs against omega_as
    ax00.scatter(omega_as, np.real(wvs), marker=".", linewidths=0, edgecolors=None)
    ax00.axhline(
        y=np.real(wvs_mean), color="tab:blue", linestyle="dashed", label="<A_w>"
    )
    ax00.axhline(y=np.real(wv_0), color="tab:orange", linestyle="dashed", label="A_w_0")
    # ax00.axhline(y=np.real(wv_0_tau), color='tab:red', linestyle='dashed', label='A_w_0 gamma=0')
    ax00.axhline(
        y=np.real(wv_0_0), color="tab:red", linestyle="dashed", label="A_w_0 gamma=0"
    )
    ax00.set_title("Re[A_w] against omega_a")
    # ax00.set_ylim(-1.01, -0.98)
    if log_yscale:
        ax00.set_yscale("log")
    ax00.legend()

    ax01.scatter(
        omega_as,
        np.real(wvs),
        marker=".",
        linewidths=0,
        edgecolors=None,
        label="Re[A_w]",
    )
    ax01.scatter(
        omega_as,
        np.imag(wvs),
        marker=".",
        linewidths=0,
        edgecolors=None,
        label="Im[A_w]",
    )
    ax01.set_title("A_w against omega_a")
    # ax01.set_ylim(-1*1.05,1*1.05)
    if log_yscale:
        ax01.set_yscale("log")
    ax01.legend()

    ax02.scatter(omega_as, np.imag(wvs), marker=".", linewidths=0, edgecolors=None)
    ax02.axhline(
        y=np.imag(wvs_mean), color="tab:blue", linestyle="dashed", label="<A_w>"
    )
    ax02.axhline(y=np.imag(wv_0), color="tab:orange", linestyle="dashed", label="A_w_0")
    # ax02.axhline(y=np.imag(wv_0_tau), color='tab:red', linestyle='dashed', label='A_w_0 gamma=0')
    ax02.axhline(
        y=np.imag(wv_0_0), color="tab:red", linestyle="dashed", label="A_w_0 gamma=0"
    )
    ax02.set_title("Im[A_w] against omega_a")
    # ax02.set_ylim(-1*1.05, 1*1.05)
    if log_yscale:
        ax02.set_yscale("log")
    ax02.legend()

    # field quads
    if plot_field_quads:
        ax10.scatter(omega_as, qs, marker=".", linewidths=0, edgecolors=None)
        ax10.axhline(y=qs_mean, color="tab:blue", linestyle="dashed", label="<<Q>_f>")
        ax10.axhline(y=q_0, color="tab:orange", linestyle="dashed", label="<Q>_f_0")
        ax10.set_title("<Q> against omega_a")
        # ax10.set_ylim(-1 * np.sqrt(2 / omega_f) * g * t * 1.05, 1 * np.sqrt(2 / omega_f) * g * t * 1.05)
        ax10.legend()

        ax11.scatter(
            omega_as, qs, marker=".", linewidths=0, edgecolors=None, label="<Q>_f"
        )
        ax11.scatter(
            omega_as, ps, marker=".", linewidths=0, edgecolors=None, label="<P>_f"
        )
        ax11.set_title("Field quadratures against omega_a")
        # ax11.set_ylim(-1 * np.sqrt(2 * omega_f) * 1.05, 1 * np.sqrt(2 * omega_f) * 1.05)
        ax11.legend()

        ax12.scatter(
            omega_as, ps, marker=".", linewidths=0, edgecolors=None, label="<P>_f"
        )
        ax12.axhline(y=ps_mean, color="tab:blue", linestyle="dashed", label="<<P>>")
        ax12.axhline(y=p_0, color="tab:orange", linestyle="dashed", label="<P>_0")
        ax12.set_title("<P> against omega_a")
        # ax12.set_ylim(-1 * np.sqrt(2 * omega_f) * g * t * 1.05, 1 * np.sqrt(2 * omega_f) * g * t * 1.05)
        ax12.legend()

    ax21.hist(omega_as, bins=np.sqrt(len(omega_as)).astype(int))
    ax21.axvline(x=omega_a_0, color="tab:orange", linestyle="dashed", label="omega_a_0")
    ax21.set_title("omega_a freqs")

    plt.show()


def plot_against_sigma(
    results,
    plot_wv_0=True,
    log_xscale=False,
    log_yscale=False,
    plot_real=True,
    plot_solution=False,
):
    params = results["params"]
    sigmas = params["sigmas"]

    wvs = results["wv"]

    wv_0 = results["wv_0"][0, 0]
    # wv_0_0 = results['wv_0_0'][0,0]

    fig = plt.figure(figsize=(20, 10), layout="constrained")
    gs = GridSpec(2, 2, figure=fig)

    ax00 = fig.add_subplot(gs[0, :])
    if plot_real:
        ax10 = fig.add_subplot(gs[1, 0])
        ax11 = fig.add_subplot(gs[1, 1])
    else:
        ax11 = fig.add_subplot(gs[1, :])

    # |wvs_mean| / |wv_0| against sigmas
    ax00.scatter(
        sigmas,
        np.abs(wvs) / np.abs(wv_0),
        label="Numerical sol.",
        color=secondary,
        marker=".",
        linewidths=0,
        edgecolors=None,
    )
    ax00.set_ylabel(
        r"$|\langle\sigma'_{-,w}(\omega'_a)\rangle|/|\sigma_{-,w}|$",
        fontsize="xx-large",
    )
    ax00.set_xlabel(r"$\sigma t$", fontsize="xx-large")

    # real agains sigmas
    if plot_real:
        ax10.scatter(
            sigmas,
            np.real(wvs),
            color=secondary,
            label=r"Numerical sol.",
            marker=".",
            linewidths=0,
            edgecolors=None,
        )

        ax10.set_ylabel(
            r"$Re[\langle\sigma'_{-,w}(\omega'_a)\rangle]$", fontsize="xx-large"
        )
        ax10.set_xlabel(r"$\sigma t$", fontsize="xx-large")

    ax11.scatter(
        sigmas,
        np.imag(wvs),
        color=secondary,
        marker=".",
        label=r"Numerical sol.",
        linewidths=0,
        edgecolors=None,
    )

    ax11.set_ylabel(
        r"$Im[\langle\sigma'_{-,w}(\omega'_a)\rangle]$", fontsize="xx-large"
    )
    ax11.set_xlabel(r"$\sigma t$", fontsize="xx-large")

    if plot_wv_0:
        if plot_real:
            ax10.axhline(
                y=np.real(wv_0),
                color=primary,
                linestyle="dashed",
                label=r"$\sigma_{-,w}$",
            )
        ax11.axhline(
            y=np.imag(wv_0), color=primary, linestyle="dashed", label=r"$\sigma_{-,w}$"
        )

    if plot_solution:
        solution = (
            (params["v"][0] - 1j * params["v"][1]) / (1 + params["v"][2])
        ) * np.exp(-0.5 * np.pow(sigmas, 2))
        ax00.scatter(
            sigmas,
            np.abs(solution) / np.abs(wv_0),
            color=tertiary,
            marker=".",
            linewidths=0,
            edgecolors=None,
            label="Analytical sol.",
        )
        if plot_real:
            ax10.scatter(
                sigmas,
                np.real(solution),
                color=tertiary,
                marker=".",
                linewidths=0,
                edgecolors=None,
                label="Analytical sol.",
            )
        ax11.scatter(
            sigmas,
            np.imag(solution),
            color=tertiary,
            marker=".",
            linewidths=0,
            edgecolors=None,
            label="Analytical sol.",
        )

    if log_xscale:
        ax00.set_xscale("log")
        if plot_real:
            ax10.set_xscale("log")
        ax11.set_xscale("log")
    if log_yscale:
        ax00.set_yscale("log")
        if plot_real:
            ax10.set_yscale("log")
        ax11.set_yscale("log")

    ax00.legend(markerscale=3)
    if plot_real:
        ax10.legend(markerscale=3)
    ax11.legend(markerscale=3)

    plt.show()


def plot_im_deltas(delta_results, sigmas):
    fig = plt.figure(figsize=(20, 10), layout="constrained")

    ax11 = fig.add_subplot()

    # Ims for each delta against sigmas
    idx = 0
    for delta_pow, results in delta_results.items():

        wvs = results["wv"]

        ax11.scatter(
            sigmas,
            np.imag(wvs),
            label=r"$\delta = 1e{}$".format(delta_pow),
            color=colours[idx],
            marker=".",
            linewidths=0,
            edgecolors=None,
        )
        ax11.set_ylabel(
            r"$Im[\langle\sigma'_{-,w}(\omega'_a)\rangle]$", fontsize="xx-large"
        )
        ax11.set_xlabel(r"$\sigma t$", fontsize="xx-large")

        idx += 1

    n = 2 * np.exp((-2 * np.pow(np.pi, 2)) / (np.pow(sigmas, 2)))
    approx_first_f = (1 + n) * np.sqrt(np.pi) / np.sqrt(2 * np.pow(sigmas, 2)) - 0.5
    approx_first_f[approx_first_f > (1.1 * np.imag(results["wv_0_0"][0, 0, 0]))] = (
        np.nan
    )

    ax11.scatter(
        sigmas,
        approx_first_f,
        color=dark,
        marker=".",
        linewidths=0,
        edgecolors=None,
        label=r"Upper bound, $f(\sigma t)$ to first order",
    )

    ax11.set_xscale("log")
    ax11.set_yscale("log")

    ax11.legend(markerscale=3, loc="lower left")

    plt.show()


def plot_im_against_sigma(
    results, plot_solutions=False, log_xscale=False, log_yscale=False
):
    params = results["params"]
    sigmas = params["sigmas"]

    wvs = results["wv"]

    fig = plt.figure(figsize=(20, 10), layout="constrained")

    ax11 = fig.add_subplot()

    # |wvs_mean| / |wv_0| against sigmas
    ax11.scatter(
        sigmas,
        np.imag(wvs),
        label="Numerical sol.",
        color=secondary,
        marker=".",
        linewidths=0,
        edgecolors=None,
    )
    ax11.set_ylabel(
        r"$Im[\langle\sigma'_{-,w}(\omega'_a)\rangle]$", fontsize="xx-large"
    )
    ax11.set_xlabel(r"$\sigma t$", fontsize="xx-large")

    if plot_solutions:
        delta = params["delta"]
        delta_trigs = (np.sin(delta) + 1) / np.cos(delta)
        sigma_coss = np.cos(sigmas)
        sigma_sins = np.sin(sigmas)
        exact_insides = sigma_sins / (sigma_coss - delta_trigs)
        uniform_points = -np.atan(exact_insides) / sigmas

        n = 2 * np.exp((-2 * np.pow(np.pi, 2)) / (np.pow(sigmas, 2)))
        approx_first_f = (1 + n) * np.sqrt(np.pi) / np.sqrt(2 * np.pow(sigmas, 2)) - 0.5
        approx_first_f[
            approx_first_f > (1.02 * np.imag(results["wv_0_0"][0, 0, 0]))
        ] = np.nan

        ax11.scatter(
            sigmas,
            uniform_points,
            color=dark,
            marker=".",
            linewidths=0,
            edgecolors=None,
            label="Analytical sol. - uniform disorder",
        )
        ax11.scatter(
            sigmas,
            approx_first_f,
            color=tertiary,
            marker=".",
            linewidths=0,
            edgecolors=None,
            label=r"Analytical sol. - upper bound, $f(\sigma t)$ to first order",
        )

        ax11.legend()

    if log_xscale:
        ax11.set_xscale("log")
    if log_yscale:
        ax11.set_yscale("log")

    ax11.legend(markerscale=3)

    plt.show()


def plot_against_tau(
    results,
    show_wv_0_0=True,
    plot_wv_0s=False,
    plot_real=True,
    log_xscale=False,
    log_yscale=False,
):
    """Plot results for fixed t and sigma and varying tau"""
    params = results["params"]
    taus = params["tau"]
    sigmas = params["sigmas"]

    wv = results["wv"]
    wv_0_0 = results["wv_0_0"]
    wv_0s = results["wv_0"]

    fig = plt.figure(figsize=(20, 10), layout="constrained")
    gs = GridSpec(2, 2, figure=fig)

    ax00 = fig.add_subplot(gs[0, :])

    if plot_real:
        ax10 = fig.add_subplot(gs[1, 0])
        ax11 = fig.add_subplot(gs[1, 1])
    else:
        ax11 = fig.add_subplot(gs[1, :])

    for idx, sigma in enumerate(sigmas):
        wv_sigma = wv[:, idx]
        ax00.scatter(
            taus,
            np.abs(wv_sigma) / np.abs(wv_0_0),
            color=colours[idx],
            marker=".",
            linewidths=0,
            edgecolors=None,
            label=r"$\sigma / \gamma = {}$".format(sigma),
        )
        if plot_real:
            ax10.scatter(
                taus,
                np.real(wv_sigma),
                color=colours[idx],
                marker=".",
                linewidths=0,
                edgecolors=None,
                label=r"$\sigma / \gamma ={}$".format(sigma),
            )
        ax11.scatter(
            taus,
            np.imag(wv_sigma),
            color=colours[idx],
            marker=".",
            linewidths=0,
            edgecolors=None,
            label=r"$\sigma / \gamma ={}$".format(sigma),
        )

    if plot_wv_0s:
        ax00.scatter(
            taus,
            np.abs(wv_0s) / np.abs(wv_0_0),
            color=dark,
            label=r"$\sigma_{-,w}(\tau)$",
            marker=".",
            linewidths=0,
            edgecolors=None,
        )
        if plot_real:
            ax10.scatter(
                taus,
                np.real(wv_0s),
                color=dark,
                label=r"$\sigma_{-,w}(\tau)$",
                marker=".",
                linewidths=0,
                edgecolors=None,
            )
        ax11.scatter(
            taus,
            np.imag(wv_0s),
            color=dark,
            label=r"$\sigma_{-,w}(\tau)$",
            marker=".",
            linewidths=0,
            edgecolors=None,
        )
    if show_wv_0_0:
        if plot_real:
            ax10.axhline(
                y=np.real(wv_0_0),
                color=dark,
                linestyle="dashed",
                label=r"$\sigma_{-,w}$",
            )
        ax11.axhline(
            y=np.imag(wv_0_0), color=dark, linestyle="dashed", label=r"$\sigma_{-,w}$"
        )

    if log_xscale:
        ax00.set_xscale("log")
        if plot_real:
            ax10.set_xscale("log")
        ax11.set_xscale("log")
    if log_yscale:
        ax00.set_yscale("log")
        if plot_real:
            ax10.set_yscale("log")
        ax11.set_yscale("log")

    ax00.set_xlabel(r"$\gamma\tau$", fontsize="xx-large")
    ax00.set_ylabel(
        r"$|\langle\sigma'_{-,w}(\omega'_a,\tau)\rangle|/|\sigma_{-,w}|$",
        fontsize="x-large",
    )
    if plot_real:
        ax10.set_xlabel(r"$\gamma\tau$", fontsize="xx-large")
        ax10.set_ylabel(
            r"$Re[\langle\sigma'_{-,w}(\omega'_a,\tau)\rangle]$", fontsize="xx-large"
        )
    ax11.set_xlabel(r"$\gamma\tau$", fontsize="xx-large")
    ax11.set_ylabel(
        r"$Im[\langle\sigma'_{-,w}(\omega'_a,\tau)\rangle]$", fontsize="xx-large"
    )

    ax00.legend(markerscale=3)
    if plot_real:
        ax10.legend(markerscale=3)
    ax11.legend(markerscale=3)

    plt.show()


def plot_im_against_tau(results, plot_wv_0s=True, log_xscale=False, log_yscale=False):
    """Plot results for fixed t and sigma and varying tau"""
    params = results["params"]
    taus = params["tau"]
    sigmas = params["sigmas"]

    wv = results["wv"]
    wv_0s = results["wv_0"]

    fig = plt.figure(figsize=(20, 10), layout="constrained")
    ax11 = fig.add_subplot()

    for idx, sigma in enumerate(sigmas):
        wv_sigma = wv[:, idx]
        ax11.scatter(
            taus,
            np.imag(wv_sigma),
            color=colours[idx],
            marker=".",
            linewidths=0,
            edgecolors=None,
            label=r"$\sigma / \gamma ={}$".format(sigma),
        )

    if plot_wv_0s:
        ax11.scatter(
            taus,
            np.imag(wv_0s),
            color=dark,
            label=r"$\sigma_{-,w}(\tau)$",
            marker=".",
            linewidths=0,
            edgecolors=None,
        )

    if log_xscale:
        ax11.set_xscale("log")
    if log_yscale:
        ax11.set_yscale("log")

    ax11.set_xlabel(r"$\gamma\tau$", fontsize="xx-large")
    ax11.set_ylabel(r"$Im[\langle\sigma'_{-,w}(\tau)\rangle]$", fontsize="xx-large")

    ax11.legend(markerscale=3)

    plt.show()


def plot_decoherence(results, log_xscale=False, log_yscale=False):
    fig = plt.figure(figsize=(20, 10), layout="constrained")
    gs = GridSpec(3, 2, figure=fig)

    times = results["params"]["times"]
    wvs = results["wvs"]
    c1_ts = results["c1_ts"]

    ax00 = fig.add_subplot(gs[0, :])
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])
    ax20 = fig.add_subplot(gs[2, :])

    ax00.scatter(
        times, np.abs(wvs), color=colours[0], marker=".", linewidths=0, edgecolors=None
    )
    ax10.scatter(
        times, np.real(wvs), color=colours[1], marker=".", linewidths=0, edgecolors=None
    )
    ax11.scatter(
        times, np.imag(wvs), color=colours[2], marker=".", linewidths=0, edgecolors=None
    )
    ax20.scatter(
        times,
        np.power(np.abs(c1_ts), 2),
        color=colours[3],
        marker=".",
        linewidths=0,
        edgecolors=None,
    )

    ax00.set_xlabel(r"$\gamma t_B$", fontsize="xx-large")
    ax00.set_ylabel(
        r"$|A_w(t_B)|$",
        fontsize="xx-large",
    )
    ax10.set_xlabel(r"$\gamma t_B$", fontsize="xx-large")
    ax10.set_ylabel(
        r"$Re[A_w(t_B)]$",
        fontsize="xx-large",
    )
    ax11.set_xlabel(r"$\gamma t_B$", fontsize="xx-large")
    ax11.set_ylabel(
        r"$Im[A_w(t_B)]$",
        fontsize="xx-large",
    )
    ax20.set_xlabel(r"$\gamma t_B$", fontsize="xx-large")
    ax20.set_ylabel(
        r"$|c_1(t_B)|^2$",
        fontsize="xx-large",
    )

    if log_xscale:
        ax00.set_xscale("log")
        ax10.set_xscale("log")
        ax11.set_xscale("log")
        ax20.set_xscale("log")
    if log_yscale:
        ax00.set_yscale("log")
        ax10.set_yscale("log")
        ax11.set_yscale("log")
        ax20.set_yscale("log")

    ax20.set_ylim([-0.05, 1.05])

    plt.show()


def plot_dephasing(results, x_logscale=False, y_logscale=False, save_fig=""):
    params = results["params"]
    times = params["times"]
    ss = params["spectral_params"]["s"]
    wvs = results["wvs"]

    wvs_abs = np.vectorize(mpm.fabs)(wvs)
    wvs_re = np.vectorize(mpm.re)(wvs)
    wvs_im = np.vectorize(mpm.im)(wvs)

    yss = {"abs": wvs_abs, "real": wvs_re, "imag": wvs_im, "labels": ss.flatten()}

    fig, (ax00, _, _) = setup_complex_plot(
        times, yss, "s={}", "\\Omega t", "A_w(t)", x_logscale, y_logscale
    )
    ax00.axhline(y=1, color=dark, linestyle="dashed")

    if save_fig:
        plt.savefig(save_fig)

    plt.show()
