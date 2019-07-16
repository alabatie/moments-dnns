import seaborn as sns
from matplotlib import rc, rcParams
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import os
import numpy as np

import warnings
warnings.filterwarnings("ignore")  # remove matplotlib warnings


def save_figure(name_fig=None):
    """ save_figure
    Save figure in pdf in figures/pdf/name_fig.pdf
    Save figure in png in figures/png/name_fig.png
    """
    if name_fig is not None:
        # save pdf
        path = os.path.join('figures', 'pdf', name_fig + '.pdf')
        plt.savefig(path, bbox_inches='tight')

        # save png
        path = os.path.join('figures', 'png', name_fig + '.png')
        plt.savefig(path, bbox_inches='tight')


def set_plot(fig_size, grid_spec, use_tex=False):
    """ set_style
    Set seaborn style for plots
    Create a figure of size fig_size
    Create a grid specified by grid_spec
        -> (2, 2) for a grid 2 x 2 (i.e. 2 lines, 2 columns)
        -> (1, 2) for a grid 1 x 2 (i.e. 1 line, 2 columns)

    # Arguments
        use_latex (bool): whether latex is enabled for legends
    """
    sns.set_style('whitegrid', {'grid.linewidth': 2,
                                'grid.color': '0.93',
                                'axes.facecolor': '0.97',
                                'axes.edgecolor': '1'})

    rc('font', **{'family': 'serif', 'serif': ['Palatino']})
    if use_tex:
        rc('text', usetex=True)
        rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

    fig = plt.figure(figsize=fig_size)
    gs = gridspec.GridSpec(grid_spec[0], grid_spec[1])
    return fig, gs


def draw_line(ax, depth, value):
    """ draw_line
    Draw thin black dashed line to compare moments curves with a reference

    # Arguments
        ax (axis): axis used for plot
        depth (numpy array): numpy array with depth values
        value (float): constant value drawn as reference
    """
    ax.plot(depth, np.full_like(depth, value), ls='--', color='k', lw=1)


def plot_moments(ax, depth, moments, colors, labels,
                 linestyles=None, linewidths=None,
                 ncol=1, loc='best', bbox_to_anchor=None,
                 yrange=None, log_scale=False):
    """ plot_moments
    Plot a list of moments on a given axis

    # Arguments
        ax (axis): axis used for the plot
        depth (numpy array): numpy array with depth values
        moments (dict): moments
        colors (list): colors for each moment (try to get sns color)
        labels (list): labels corresponding to each moment
        linestyles (list): linestyles corresponding to each moment
        linewidths (list): linewidths corresponding to each moment
        ncol (int): number of columns in the legend
        loc (str): location of the legend (default is 'best')
        bbox_to_anchor (tuple): set the part of the bounding box defined by loc
            at position (x, y) (if None, loc simply defines the legend's loc)
        yrange (list of 2 values): range of y (if None, use default range)
        log_scale (bool): whether to use log scale or normal scale
    """
    num_moments = len(moments)
    linestyles = ['-'] * num_moments if linestyles is None else linestyles
    linewidths = [4] * num_moments if linewidths is None else linewidths
    colors = [sns.xkcd_rgb[color] if color in sns.xkcd_rgb else color
              for color in colors]

    for moment, color, ls, lw, label in zip(moments, colors, linestyles,
                                            linewidths, labels):
        if moment.ndim > 1:
            # if more than 1 dimensions, also plot 1 sigma intervals
            ax.plot(depth, np.mean(moment, axis=0),
                    ls=ls, lw=lw, color=color, label=label)
            ax.fill_between(depth,
                            np.percentile(moment, 16, axis=0),
                            np.percentile(moment, 84, axis=0),
                            color=color, alpha=0.1)
            ax.plot(depth, np.percentile(moment, 16, axis=0),
                    ls=':', color='grey', lw=0.7)
            ax.plot(depth, np.percentile(moment, 84, axis=0),
                    ls=':', color='grey', lw=0.7)
        else:
            ax.plot(depth, moment, ls=ls, lw=lw, color=color, label=label)

    ax.tick_params(labelsize=18)
    plt.legend(fontsize=22, loc=loc, ncol=ncol, bbox_to_anchor=bbox_to_anchor,
               framealpha=1.0, edgecolor=plt.rcParams["axes.facecolor"],
               borderpad=0)
    if log_scale:
        plt.yscale('log')

    # set limits
    plt.xlim([0, np.max(depth)])
    if yrange is not None:
        plt.ylim(yrange)


def plot_histo(ax, moment, xfac, yfac, labels, annotation, xannotation):
    """ plot_histo
    Plot histogram of moments at four different depths
    Customized to the experiments on histograms of vanilla nets

    # Arguments
        ax (axis): axis used for plot
        moment (numpy array): moment realizations at various depths
        xfac (list of 2 values): factor of expansion of the x-axis
        yfac (float): factor of expansion of the y-axis
        labels (list): labels for legend
        annotation (str): complementary annotation with the name of the moment
        xannotation (float): position on the x-axis of complementary annotation
    """
    bins0, histo0 = make_histo(moment[:, 0])
    bins1, histo1 = make_histo(moment[:, 1])
    bins2, histo2 = make_histo(moment[:, 2])
    bins3, histo3 = make_histo(moment[:, 3])

    ax.plot(bins0, histo0, lw=2., color=sns.xkcd_rgb['blue'])
    ax.plot(bins1, histo1, lw=2., color=sns.xkcd_rgb['purple'])
    ax.plot(bins2, histo2, lw=2., color=sns.xkcd_rgb['magenta'])
    ax.plot(bins3, histo3, lw=2., color=sns.xkcd_rgb['red'])

    # expand xlim and ylim
    ax.set_ylim(0, yfac * ax.get_ylim()[1])
    ax.set_xlim(xfac[0] * ax.get_xlim()[0], xfac[1] * ax.get_xlim()[1])
    ax.set_yticklabels([])  # remove y ticks

    # set legend
    ax.tick_params(labelsize=18)
    plt.legend(labels,
               fontsize=20, loc='center right', bbox_to_anchor=[1., 0.54],
               framealpha=1.0, edgecolor=plt.rcParams["axes.facecolor"],
               borderpad=0.4)

    # set annotation
    ax.text(xannotation, 0.86, annotation,
            transform=ax.transAxes, fontsize=20, zorder=100,
            bbox=dict(alpha=1.0, boxstyle='round',
                      edgecolor=plt.rcParams["axes.facecolor"],
                      facecolor=plt.rcParams["axes.facecolor"],
                      pad=0.3))


def make_histo(moment):
    """ make_histo
    Compute histogram of the logarithm of moments at given depth
    First we get the results from np.histogram
    Second we convert bin edges to centered bins
    Finally we convert the histogram to a density (this last step
        does not matter for our plots since we do not show the y-ticks)

    # Arguments
        moment (numpy array): all moment realizations at given depth
    """
    moment = np.log(moment)
    histo, bins = np.histogram(moment, bins=50)
    delta_bins = (bins[1] - bins[0])
    bins = np.array(bins)[:-1] + delta_bins / 2.

    histo = np.array(histo) / (len(moment) * delta_bins)  # convert to density
    return bins, histo
