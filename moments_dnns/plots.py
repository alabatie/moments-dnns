"""Plotting functions."""
from functools import partial

import numpy as np

from moments_dnns.plot_utils import (
    draw_line,
    plot_histo,
    plot_moments,
    save_figure,
    set_plot,
)


def plot_vanilla_histo(
    moments: dict[str, np.ndarray], use_tex: bool = True, name_fig: str | None = None
):
    """Plot histograms at 4 depths of (a) log nu2(x^l) and (b) log mu2(dx^l) = log nu2(dx^l).

    # Args
        moments: moments from the experiment
        use_tex: whether latex is used in legends and annotations
            (requires LaTeX distribution)
        name_fig: name of the figure to be saved
            (if left to None, no figure is saved)
    """
    fig, grid_spec = set_plot(fig_size=(16.5, 4.0), grid_spec=(1, 2), use_tex=use_tex)
    if use_tex:
        annotation_a = r"$\log \nu_2(\mathbf{x}^l) - " + r"\log \nu_2(\mathbf{x}^0)$"
        annotation_b = (
            r"$\log \mu_2(\mathrm{d}\mathbf{x}^l)"
            + r"-\log \mu_2(\mathrm{d}\mathbf{x}^0)$"
        )
        letter_a = r"\textit{(a)}"
        letter_b = r"\textit{(b)}"
        labels = [r"$l=50$", r"$l=100$", r"$l=150$", r"$l=200$"]
    else:
        annotation_a = r"log nu2(x^l) - log nu_2(x^0)"
        annotation_b = r"log mu2(dx^l) - log mu_2(dx^0)"
        letter_a = r"(a)"
        letter_b = r"(b)"
        labels = [r"l = 50", r"l = 100", r" l = 150", r"l = 200"]

    # second-order moments of signal
    axis = fig.add_subplot(grid_spec[0, 0])
    plot_histo(
        axis=axis,
        moment=moments["nu2_signal_loc3"],
        xfac=[1.1, 1.6],
        yfac=1.22,
        labels=labels,
        annotation=annotation_a,
        xannotation=0.53,
    )
    axis.text(0.02, 0.87, letter_a, fontsize=30, transform=axis.transAxes)

    # second-order moments of noise
    axis = fig.add_subplot(grid_spec[0, 1])
    plot_histo(
        axis=axis,
        moment=moments["mu2_noise_loc3"],
        xfac=[1.1, 1.25],
        yfac=1.255,
        labels=labels,
        annotation=annotation_b,
        xannotation=0.46,
    )
    axis.text(0.02, 0.87, letter_b, fontsize=30, transform=axis.transAxes)

    save_figure(name_fig=name_fig)


def plot_vanilla(
    moments: dict[str, np.ndarray], use_tex: bool = True, name_fig: str | None = None
):
    """Plot containing 2 subplots for vanilla nets.

    # Args
        moments: moments from the experiment
        use_tex: whether latex is used in legends and annotations
            (requires LaTeX distribution)
        name_fig: name of the figure to be saved
            (if left to None, no figure is saved)
    """
    plot_fn = partial(plot_moments, depth=moments["depth"])
    fig, grid_spec = set_plot(fig_size=(16.5, 4.0), grid_spec=(1, 2), use_tex=use_tex)

    yrange_list = [[0.96, 1.26], [0.6, 5.5]]
    log_list = [False, False]
    if use_tex:
        label_a = [r"$\delta\chi^l$"]
        label_b = [r"$r_\textrm{\LARGE eff}(\mathbf{x}^l)$"]
        letter_list = [r"\textit{(a)}", r"\textit{(b)}"]
    else:
        label_a = [r"delta chi^l"]
        label_b = [r"reff(x^l)"]
        letter_list = [r"(a)", r"(b)"]

    # plot normalized sensitivity increments
    axis = fig.add_subplot(grid_spec[0, 0])
    plot_fn(
        axis=axis,
        moments=[moments["chi_loc3"] / moments["chi_loc1"]],
        colors=["blue"],
        labels=label_a,
        loc="upper right",
        ncol=2,
        bbox_to_anchor=(0.95, 1.02),
        log_scale=log_list[0],
        yrange=yrange_list[0],
    )
    draw_line(axis=axis, depth=moments["depth"], value=1)
    axis.text(0.02, 0.87, letter_list[0], fontsize=30, transform=axis.transAxes)

    # plot effective rank of signal
    axis = fig.add_subplot(grid_spec[0, 1])
    plot_fn(
        axis=axis,
        moments=[moments["reff_signal_loc3"]],
        colors=["purple"],
        labels=label_b,
        loc="upper right",
        bbox_to_anchor=(0.95, 1.02),
        log_scale=log_list[1],
        yrange=yrange_list[1],
    )
    draw_line(axis=axis, depth=moments["depth"], value=1)
    axis.text(0.02, 0.87, letter_list[1], fontsize=30, transform=axis.transAxes)

    save_figure(name_fig=name_fig)


def plot_bn_ff(
    moments: dict[str, np.ndarray], use_tex: bool = True, name_fig: str | None = None
):
    """Plot containing 4 subplots for batch-normalized feedforward nets.

    # Args
        moments: moments from the experiment
        use_tex: whether latex is used in legends and annotations
            (requires LaTeX distribution)
        name_fig: name of the figure to be saved
            (if left to None, no figure is saved)
    """
    plot_fn = partial(plot_moments, depth=moments["depth"])
    fig, grid_spec = set_plot(fig_size=(16.5, 8.5), grid_spec=(2, 2), use_tex=use_tex)

    yrange_list = [[0.99, 1.4], [1.0, 10**17], [0.8, 10000], [0.1, 30000]]
    log_list = [False, True, True, True]
    if use_tex:
        label_a = [
            r"$\delta^{}_\textrm{\LARGE BN} \hspace{.03em} \chi^l$",
            r"$\delta_{\phi} \hspace{.03em} \chi^l$",
            r"$\delta\chi^l$",
        ]
        label_b = [r"$\chi^l$"]
        label_c = [
            r"$r_\textrm{\LARGE eff}(\mathrm{d}\mathbf{x}^l)$",
            r"$r_\textrm{\LARGE eff}(\mathbf{x}^l)$",
        ]
        label_d = [r"$\mu_4(\mathbf{z}^l)$", r"$\nu_1(|\mathbf{z}^l|)$"]
        letter_list = [
            r"\textit{(a)}",
            r"\textit{(b)}",
            r"\textit{(c)}",
            r"\textit{(d)}",
        ]
    else:
        label_a = [r"deltaBN chi^l", r"deltaphi chi^l", r"delta chi^l"]
        label_b = [r"chi^l"]
        label_c = [r"reff(dx^l)", r"reff(x^l)"]
        label_d = [r"mu_4(z^l)", r"nu_1(|z^l|)"]
        letter_list = [r"(a)", r"(b)", r"(c)", r"(d)"]

    # plot normalized sensitivity increments
    axis = fig.add_subplot(grid_spec[0, 0])
    plot_fn(
        axis=axis,
        moments=[
            moments["chi_loc3"] / moments["chi_loc1"],
            moments["chi_loc4"] / moments["chi_loc3"],
            moments["chi_loc4"] / moments["chi_loc1"],
        ],
        colors=["purple", "red", "blue"],
        labels=label_a,
        loc="upper right",
        ncol=2,
        bbox_to_anchor=(0.95, 1.02),
        log_scale=log_list[0],
        yrange=yrange_list[0],
    )
    draw_line(axis=axis, depth=moments["depth"], value=1)
    axis.text(0.02, 0.86, letter_list[0], fontsize=30, transform=axis.transAxes)

    # plot normalized sensitivity
    axis = fig.add_subplot(grid_spec[0, 1])
    plot_fn(
        axis=axis,
        moments=[moments["chi_loc4"]],
        colors=["blue"],
        labels=label_b,
        loc="upper center",
        bbox_to_anchor=(0.72, 1.02),
        log_scale=log_list[1],
        yrange=yrange_list[1],
    )
    axis.text(0.02, 0.86, letter_list[1], fontsize=30, transform=axis.transAxes)

    # plot effective ranks of noise and signal
    axis = fig.add_subplot(grid_spec[1, 0])
    plot_fn(
        axis=axis,
        moments=[moments["reff_noise_loc4"], moments["reff_signal_loc4"]],
        colors=["purple", "purple"],
        line_styles=[":", "-"],
        labels=label_c,
        loc="upper right",
        bbox_to_anchor=(0.95, 1.02),
        log_scale=log_list[2],
        yrange=yrange_list[2],
    )
    draw_line(axis=axis, depth=moments["depth"], value=1)
    axis.text(0.02, 0.86, letter_list[2], fontsize=30, transform=axis.transAxes)

    # plot signal moments
    axis = fig.add_subplot(grid_spec[1, 1])
    plot_fn(
        axis=axis,
        moments=[moments["mu4_signal_loc3"], moments["nu1_abs_signal_loc3"]],
        colors=["red", "red"],
        line_styles=[":", "-"],
        labels=label_d,
        loc="upper right",
        bbox_to_anchor=(0.95, 1.02),
        log_scale=log_list[3],
        yrange=yrange_list[3],
    )
    axis.text(0.02, 0.86, letter_list[3], fontsize=30, transform=axis.transAxes)

    save_figure(name_fig=name_fig)


def plot_bn_res(
    moments: dict[str, np.ndarray], use_tex: bool = True, name_fig: str | None = None
):
    """Plot containing 4 subplots for batch-normalized feedforward ResNets.

    # Args
        moments: moments from the experiment
        use_tex: whether latex is used in legends and annotations
            (requires LaTeX distribution)
        name_fig: name of the figure to be saved
            (if left to None, no figure is saved)
    """
    plot_fn = partial(plot_moments, depth=moments["depth"])
    fig, grid_spec = set_plot(fig_size=(16.5, 8.5), grid_spec=(2, 2), use_tex=use_tex)

    yrange_list = [[0.99, 1.4], [1.0, 50], [0.8, 20000], [0.0, 7.2]]
    log_list = [False, False, True, False]
    if use_tex:
        label_a = [
            r"$\delta^{}_\textrm{\LARGE BN} \hspace{.03em} \chi^{l,1}$",
            r"$\delta_{\phi} \hspace{.03em} \chi^{l,1}$",
            r"$\delta\chi^{l,1}$",
        ]
        label_b = [r"$\chi^l$", r"$l^{\tau}$"]
        label_c = [
            r"$r_\textrm{\LARGE eff}(\mathrm{d}\mathbf{x}^{l,1})$",
            r"$r_\textrm{\LARGE eff}(\mathbf{x}^{l,1})$",
        ]
        label_d = [r"$\mu_4(\mathbf{z}^{l,1})$", r"$\nu_1(|\mathbf{z}^{l,1}|)$"]
        letter_list = [
            r"\textit{(a)}",
            r"\textit{(b)}",
            r"\textit{(c)}",
            r"\textit{(d)}",
        ]
    else:
        label_a = [r"deltaBN chi^{l,1}", r"deltaphi chi^{l,1}", r"delta chi^{l,1}"]
        label_b = [r"chi^l", r"l^tau"]
        label_c = [r"reff(dx^{l,1})", r"reff(x^{l,1})"]
        label_d = [r"mu_4(z^{l,1})", r"nu_1(|z^{l,1}|)"]
        letter_list = [r"(a)", r"(b)", r"(c)", r"(d)"]

    # compute power law fit
    delta_chi = moments["chi_loc4"].mean(0) / moments["chi_loc1"].mean(0)
    avg_delta_chi = delta_chi.mean()  # average over the whole evolution
    tau = (avg_delta_chi ** (2 * moments["res_depth"]) - 1) / 2.0

    chi = moments["chi_loc5"].mean(0)
    pow_law = moments["depth"] ** tau
    const = (chi * pow_law).mean() / (pow_law * pow_law).mean()
    pow_law_fit = const * pow_law

    # plot normalized sensitivity increments
    axis = fig.add_subplot(grid_spec[0, 0])
    plot_fn(
        axis=axis,
        moments=[
            moments["chi_loc2"] / moments["chi_loc1"],
            moments["chi_loc4"] / moments["chi_loc2"],
            moments["chi_loc4"] / moments["chi_loc1"],
        ],
        colors=["purple", "red", "blue"],
        labels=label_a,
        loc="upper right",
        ncol=2,
        bbox_to_anchor=(0.95, 1.02),
        log_scale=log_list[0],
        yrange=yrange_list[0],
    )
    draw_line(axis=axis, depth=moments["depth"], value=1)
    axis.text(0.02, 0.86, letter_list[0], fontsize=30, transform=axis.transAxes)

    # plot normalized sensitivity
    axis = fig.add_subplot(grid_spec[0, 1])
    plot_fn(
        axis=axis,
        moments=[moments["chi_loc5"], pow_law_fit],
        colors=["blue", "red"],
        line_styles=["-", "--"],
        line_widths=[3, 4],
        labels=label_b,
        loc="upper center",
        bbox_to_anchor=(0.70, 1.02),
        log_scale=log_list[1],
        yrange=yrange_list[1],
    )
    axis.text(0.02, 0.86, letter_list[1], fontsize=30, transform=axis.transAxes)

    # plot effective ranks of noise and signal
    axis = fig.add_subplot(grid_spec[1, 0])
    plot_fn(
        axis=axis,
        moments=[moments["reff_noise_loc3"], moments["reff_signal_loc3"]],
        colors=["purple", "purple"],
        line_styles=[":", "-"],
        labels=label_c,
        loc="upper right",
        bbox_to_anchor=(0.95, 1.02),
        log_scale=log_list[2],
        yrange=yrange_list[2],
    )
    draw_line(axis=axis, depth=moments["depth"], value=1)
    axis.text(0.02, 0.86, letter_list[2], fontsize=30, transform=axis.transAxes)

    # plot signal moments
    axis = fig.add_subplot(grid_spec[1, 1])
    plot_fn(
        axis=axis,
        moments=[moments["mu4_signal_loc2"], moments["nu1_abs_signal_loc2"]],
        colors=["red", "red"],
        line_styles=[":", "-"],
        labels=label_d,
        loc="upper right",
        bbox_to_anchor=(0.95, 1.02),
        log_scale=log_list[3],
        yrange=yrange_list[3],
    )
    axis.text(0.02, 0.86, letter_list[3], fontsize=30, transform=axis.transAxes)

    save_figure(name_fig=name_fig)
