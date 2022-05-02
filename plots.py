from moments_dnns.plot_utils import set_plot, draw_line
from moments_dnns.plot_utils import plot_moments, plot_histo
from moments_dnns.plot_utils import save_figure


def plot_vanilla_histo(moments, use_tex=True, name_fig=None):
    """plot_vanilla_histo
    Plot containing two subplots showing the histograms at 4 depths of
            (a) log nu2(x^l)
            (b) log mu2(dx^l) = log nu2(dx^l)

    # Arguments
        moments (dict): moments from the experiment
        use_tex (bool): whether latex is used in legends and annotations
            (If use_tex = True and no LaTeX distribution is found,
             python will crash)
        name_fig (str): the figure is saved as figures/name_fig.pdf
            (if name_fig is left to None, no figure is saved)
    """
    fig, gs = set_plot(fig_size=(16.5, 4.0), grid_spec=(1, 2), use_tex=use_tex)
    if use_tex:
        annotation_a = (
            "$\\log \\nu_2(\\mathbf{x}^l) - " + "\\log \\nu_2(\\mathbf{x}^0)$"
        )
        annotation_b = (
            "$\\log \\mu_2(\\mathrm{d}\\mathbf{x}^l)"
            + "-\\log \\mu_2(\\mathrm{d}\\mathbf{x}^0)$"
        )
        letter_a = "\\textit{(a)}"
        letter_b = "\\textit{(b)}"
        labels = ["$l=50$", "$l=100$", "$l=150$", "$l=200$"]
    else:
        annotation_a = "log nu2(x^l) - log nu_2(x^0)"
        annotation_b = "log mu2(dx^l) - log mu_2(dx^0)"
        letter_a = "(a)"
        letter_b = "(b)"
        labels = ["l = 50", "l = 100", " l = 150", "l = 200"]

    # second-order moments of signal
    ax = fig.add_subplot(gs[0, 0])
    plot_histo(
        ax=ax,
        moment=moments["nu2_signal_loc3"],
        xfac=[1.1, 1.6],
        yfac=1.22,
        labels=labels,
        annotation=annotation_a,
        xannotation=0.53,
    )
    ax.text(0.02, 0.87, letter_a, fontsize=30, transform=ax.transAxes)

    # second-order moments of noise
    ax = fig.add_subplot(gs[0, 1])
    plot_histo(
        ax=ax,
        moment=moments["mu2_noise_loc3"],
        xfac=[1.1, 1.25],
        yfac=1.255,
        labels=labels,
        annotation=annotation_b,
        xannotation=0.46,
    )
    ax.text(0.02, 0.87, letter_b, fontsize=30, transform=ax.transAxes)

    # save figure
    save_figure(name_fig=name_fig)


def plot_vanilla(moments, use_tex=True, name_fig=None):
    """plot_vanilla
    Plot containing 2 subplots for vanilla nets with the depth evolution of
        (a) delta chi^l
        (b) reff(x^l)

    # Arguments
        moments (dict): moments from the experiment
        use_tex (bool): whether latex is used in legends and annotations
            (If use_tex = True and no LaTeX distribution is found,
             python will crash)
        name_fig (str): the figure is saved as figures/name_fig.pdf
            (if name_fig is left to None, no figure is saved)
    """
    fig, gs = set_plot(fig_size=(16.5, 4.0), grid_spec=(1, 2), use_tex=use_tex)

    yrange_list = [[0.96, 1.26], [0.6, 5.5]]
    log_list = [False, False]
    if use_tex:
        label_a = ["$\\delta\\chi^l$"]
        label_b = ["$r_\\textrm{\LARGE eff}(\\mathbf{x}^l)$"]
        letter_list = ["\\textit{(a)}", "\\textit{(b)}"]
    else:
        label_a = ["delta chi^l"]
        label_b = ["reff(x^l)"]
        letter_list = ["(a)", "(b)"]

    # plot normalized sensitivity increments
    ax = fig.add_subplot(gs[0, 0])
    plot_moments(
        ax=ax,
        depth=moments["depth"],
        moments=[moments["chi_loc3"] / moments["chi_loc1"]],
        colors=["blue"],
        labels=label_a,
        loc="upper right",
        ncol=2,
        bbox_to_anchor=(0.95, 1.02),
        log_scale=log_list[0],
        yrange=yrange_list[0],
    )
    draw_line(ax=ax, depth=moments["depth"], value=1)  # draw reference at 1
    ax.text(0.02, 0.87, letter_list[0], fontsize=30, transform=ax.transAxes)

    # plot effective rank of signal
    ax = fig.add_subplot(gs[0, 1])
    plot_moments(
        ax=ax,
        depth=moments["depth"],
        moments=[moments["reff_signal_loc3"]],
        colors=["purple"],
        linestyles=["-", ":"],
        labels=label_b,
        loc="upper right",
        bbox_to_anchor=(0.95, 1.02),
        log_scale=log_list[1],
        yrange=yrange_list[1],
    )
    draw_line(ax=ax, depth=moments["depth"], value=1)  # draw reference at 1
    ax.text(0.02, 0.87, letter_list[1], fontsize=30, transform=ax.transAxes)

    # save figure
    save_figure(name_fig=name_fig)


def plot_bn_ff(moments, use_tex=True, name_fig=None):
    """plot_bn_ff
    Plot containing 4 subplots for batch-normalized feedforward nets:
        (a) delta chi^l decomposed as delta_BN chi^l * delta_phi chi^l
        (b) chi^l
        (c) The effective ranks: reff(dx^l), reff(x^l)
        (c) The moments of the pre-activations: mu4(z^l), nu1(|z^l|)
            (since z^l is standardized after batch norm, this enables to
             see whether z^l is Gaussian with e.g. deviation of mu4(z^l)
             from the Gaussian kurtosis of 3)

    # Arguments
        moments (dict): moments from the experiment
        use_tex (bool): whether latex is used in legends and annotations
            (If use_tex = True and no LaTeX distribution is found,
             python will crash)
        name_fig (str): the figure is saved as figures/name_fig.pdf
            (if name_fig is left to None, no figure is saved)
    """
    fig, gs = set_plot(fig_size=(16.5, 8.5), grid_spec=(2, 2), use_tex=use_tex)

    yrange_list = [[0.99, 1.4], [1.0, 10**17], [0.8, 10000], [0.1, 30000]]
    log_list = [False, True, True, True]
    if use_tex:
        label_a = [
            "$\delta^{}_\\textrm{\LARGE BN} \\hspace{.03em} \chi^l$",
            "$\delta_{\phi} \\hspace{.03em} \chi^l$",
            "$\delta\chi^l$",
        ]
        label_b = ["$\chi^l$"]
        label_c = [
            "$r_\\textrm{\LARGE eff}(\mathrm{d}\mathbf{x}^l)$",
            "$r_\\textrm{\LARGE eff}(\mathbf{x}^l)$",
        ]
        label_d = ["$\\mu_4(\mathbf{z}^l)$", "$\\nu_1(|\mathbf{z}^l|)$"]
        letter_list = [
            "\\textit{(a)}",
            "\\textit{(b)}",
            "\\textit{(c)}",
            "\\textit{(d)}",
        ]
    else:
        label_a = ["deltaBN chi^l", "deltaphi chi^l", "delta chi^l"]
        label_b = ["chi^l"]
        label_c = ["reff(dx^l)", "reff(x^l)"]
        label_d = ["mu_4(z^l)", "nu_1(|z^l|)"]
        letter_list = ["(a)", "(b)", "(c)", "(d)"]

    # plot normalized sensitivity increments
    ax = fig.add_subplot(gs[0, 0])
    plot_moments(
        ax=ax,
        depth=moments["depth"],
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
    draw_line(ax=ax, depth=moments["depth"], value=1)  # draw reference at 1
    ax.text(0.02, 0.86, letter_list[0], fontsize=30, transform=ax.transAxes)

    # plot normalized sensitivity
    ax = fig.add_subplot(gs[0, 1])
    plot_moments(
        ax=ax,
        depth=moments["depth"],
        moments=[moments["chi_loc4"]],
        colors=["blue"],
        labels=label_b,
        loc="upper center",
        bbox_to_anchor=(0.72, 1.02),
        log_scale=log_list[1],
        yrange=yrange_list[1],
    )
    ax.text(0.02, 0.86, letter_list[1], fontsize=30, transform=ax.transAxes)

    # plot effective ranks of noise and signal
    ax = fig.add_subplot(gs[1, 0])
    plot_moments(
        ax=ax,
        depth=moments["depth"],
        moments=[moments["reff_noise_loc4"], moments["reff_signal_loc4"]],
        colors=["purple", "purple"],
        linestyles=[":", "-"],
        labels=label_c,
        loc="upper right",
        bbox_to_anchor=(0.95, 1.02),
        log_scale=log_list[2],
        yrange=yrange_list[2],
    )
    draw_line(ax=ax, depth=moments["depth"], value=1)
    ax.text(0.02, 0.86, letter_list[2], fontsize=30, transform=ax.transAxes)

    # plot moments of the signal
    ax = fig.add_subplot(gs[1, 1])
    plot_moments(
        ax=ax,
        depth=moments["depth"],
        moments=[moments["mu4_signal_loc3"], moments["nu1_abs_signal_loc3"]],
        colors=["red", "red"],
        linestyles=[":", "-"],
        labels=label_d,
        loc="upper right",
        bbox_to_anchor=(0.95, 1.02),
        log_scale=log_list[3],
        yrange=yrange_list[3],
    )
    ax.text(0.02, 0.86, letter_list[3], fontsize=30, transform=ax.transAxes)

    # save figure
    save_figure(name_fig=name_fig)


def plot_bn_res(moments, use_tex=True, name_fig=None):
    """plot_bn_res
    Plot containing 4 subplots for batch-normalized resets
        with the depth evolution of:
        (a) delta chi^{l,1} decomposed as
            delta_BN chi^{l,1} * delta_phi chi^{l,1}
        (b) chi^l and the power-law fit l^tau
                - the power tau is obtained by averaging delta chi^{l,1} over
                all realizations and all depth l
                - alternatively, we could have computed a pow-law fit
                per-realization and taken the average over all realizations.
                This would have led to an even better fit.
        (c) The effective ranks reff(dx^{l,1}), reff(x^{l,1})
        (c) The moments of the pre-activations: mu4(z^{l,1}), nu1(|z^{l,1}|)
            (again since z^{l,1} is standardized after batch norm,
            this enables to see whether z^{l,1} is Gaussian with e.g.
            deviation of mu4(z^{l,1}) from the Gaussian kurtosis of 3)

    # Arguments
        moments (dict): moments from the experiment
        use_tex (bool): whether latex is used in legends and annotations
            (If use_tex = True and no LaTeX distribution is found,
             python will crash)
        name_fig (str): the figure is saved as figures/name_fig.pdf
            (if name_fig is left to None, no figure is saved)
    """
    fig, gs = set_plot(fig_size=(16.5, 8.5), grid_spec=(2, 2), use_tex=use_tex)

    yrange_list = [[0.99, 1.4], [1.0, 50], [0.8, 20000], [0.0, 7.2]]
    log_list = [False, False, True, False]
    if use_tex:
        label_a = [
            "$\\delta^{}_\\textrm{\LARGE BN} \\hspace{.03em} \\chi^{l,1}$",
            "$\\delta_{\\phi} \\hspace{.03em} \\chi^{l,1}$",
            "$\\delta\\chi^{l,1}$",
        ]
        label_b = ["$\\chi^l$", "$l^{\\tau}$"]
        label_c = [
            "$r_\\textrm{\LARGE eff}(\\mathrm{d}\\mathbf{x}^{l,1})$",
            "$r_\\textrm{\LARGE eff}(\\mathbf{x}^{l,1})$",
        ]
        label_d = ["$\\mu_4(\\mathbf{z}^{l,1})$", "$\\nu_1(|\\mathbf{z}^{l,1}|)$"]
        letter_list = [
            "\\textit{(a)}",
            "\\textit{(b)}",
            "\\textit{(c)}",
            "\\textit{(d)}",
        ]
    else:
        label_a = ["deltaBN chi^{l,1}", "deltaphi chi^{l,1}", "delta chi^{l,1}"]
        label_b = ["chi^l", "l^tau"]
        label_c = ["reff(dx^{l,1})", "reff(x^{l,1})"]
        label_d = ["mu_4(z^{l,1})", "nu_1(|z^{l,1}|)"]
        letter_list = ["(a)", "(b)", "(c)", "(d)"]

    # make power law fit
    delta_chi = moments["chi_loc4"].mean(0) / moments["chi_loc1"].mean(0)
    av_delta_chi = delta_chi.mean()  # average over the whole evolution
    tau = (av_delta_chi ** (2 * moments["res_depth"]) - 1) / 2.0

    chi = moments["chi_loc5"].mean(0)
    pow_law = moments["depth"] ** tau
    Const = (chi * pow_law).mean() / (pow_law * pow_law).mean()
    pow_law_fit = Const * pow_law

    # plot normalized sensitivity increments
    ax = fig.add_subplot(gs[0, 0])
    plot_moments(
        ax=ax,
        depth=moments["depth"],
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
    draw_line(ax=ax, depth=moments["depth"], value=1)
    ax.text(0.02, 0.86, letter_list[0], fontsize=30, transform=ax.transAxes)

    # plot normalized sensitivity
    ax = fig.add_subplot(gs[0, 1])
    plot_moments(
        ax=ax,
        depth=moments["depth"],
        moments=[moments["chi_loc5"], pow_law_fit],
        colors=["blue", "red"],
        linestyles=["-", "--"],
        linewidths=[3, 4],
        labels=label_b,
        loc="upper center",
        bbox_to_anchor=(0.70, 1.02),
        log_scale=log_list[1],
        yrange=yrange_list[1],
    )
    ax.text(0.02, 0.86, letter_list[1], fontsize=30, transform=ax.transAxes)

    # plot effective ranks of noise and signal
    ax = fig.add_subplot(gs[1, 0])
    plot_moments(
        ax=ax,
        depth=moments["depth"],
        moments=[moments["reff_noise_loc3"], moments["reff_signal_loc3"]],
        colors=["purple", "purple"],
        linestyles=[":", "-"],
        labels=label_c,
        loc="upper right",
        bbox_to_anchor=(0.95, 1.02),
        log_scale=log_list[2],
        yrange=yrange_list[2],
    )
    draw_line(ax=ax, depth=moments["depth"], value=1)
    ax.text(0.02, 0.86, letter_list[2], fontsize=30, transform=ax.transAxes)

    # plot moments of the signal
    ax = fig.add_subplot(gs[1, 1])
    plot_moments(
        ax=ax,
        depth=moments["depth"],
        moments=[moments["mu4_signal_loc2"], moments["nu1_abs_signal_loc2"]],
        colors=["red", "red"],
        linestyles=[":", "-"],
        labels=label_d,
        loc="upper right",
        bbox_to_anchor=(0.95, 1.02),
        log_scale=log_list[3],
        yrange=yrange_list[3],
    )
    ax.text(0.02, 0.86, letter_list[3], fontsize=30, transform=ax.transAxes)

    # save figure
    save_figure(name_fig=name_fig)
