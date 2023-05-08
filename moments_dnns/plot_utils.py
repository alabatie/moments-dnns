"""Utils for plots."""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from moments_dnns import ROOT_DIR

warnings.filterwarnings("ignore")  # remove matplotlib warnings


def save_figure(name_fig: bool = None):
    """Save_figure.

    Save in pdf in figures/pdf/name_fig.pdf, and in png in figures/png/name_fig.png.
    """
    if name_fig is not None:
        path = ROOT_DIR / "figures" / "pdf" / f"{name_fig}.pdf"
        plt.savefig(path, bbox_inches="tight")

        path = ROOT_DIR / "figures" / "png" / f"{name_fig}.png"
        plt.savefig(path, bbox_inches="tight")


def delete_figure(name_fig: bool = None):
    """Delete_figure."""
    path = ROOT_DIR / "figures" / "pdf" / f"{name_fig}.pdf"
    path.unlink()

    path = ROOT_DIR / "figures" / "png" / f"{name_fig}.png"
    path.unlink()


def set_plot(
    fig_size: tuple[int, int], grid_spec: tuple[int, int], use_tex: bool = False
) -> tuple[Figure, GridSpec]:
    """Set seaborn style for plots.

    # Args
        fig_size: size of figure
        grid_spec: grid specification
            - (2, 2) for a grid 2 x 2 (i.e. 2 lines, 2 columns)
            - (1, 2) for a grid 1 x 2 (i.e. 1 line, 2 columns)
        use_tex: whether latex is enabled for legends
    """
    sns.set_style(
        "whitegrid",
        {
            "grid.linewidth": 2,
            "grid.color": "0.93",
            "axes.facecolor": "0.97",
            "axes.edgecolor": "1",
        },
    )
    if use_tex:
        plt.rc("font", **{"family": "serif", "serif": ["Palatino"]})
        plt.rc("text", usetex=True)
        plt.rc("text.latex", preamble=r"\usepackage{amsmath}")

    fig = plt.figure(figsize=fig_size)
    grid_spec = GridSpec(grid_spec[0], grid_spec[1])
    return fig, grid_spec


def draw_line(axis: Axes, depth: np.ndarray, value: float | int):
    """Draw thin black dashed line to compare moments curves with a reference.

    # Args
        axis: axis used for the plot
        depth: numpy array with depth values
        value: constant value used as reference
    """
    axis.plot(depth, np.full_like(depth, value), ls="--", color="k", lw=1)


def plot_moments(
    axis: Axes,
    depth: np.ndarray,
    moments: dict[str, np.ndarray],
    colors: list[str],
    labels: list[str],
    line_styles: list[str] | None = None,
    line_widths: list[int] | None = None,
    ncol: int = 1,
    loc: str = "best",
    bbox_to_anchor: tuple[float, float] | None = None,
    yrange: list[float] | None = None,
    log_scale: bool = False,
):
    """Plot list of moments on a given axis.

    # Args
        axis: axis used for the plot
        depth: numpy array with depth values
        moments: moments
        colors: colors corresponding to each moment
        labels: labels corresponding to each moment
        line_styles: line styles corresponding to each moment
        line_widths: line widths corresponding to each moment
        ncol: number of columns in the legend
        loc: location of the legend (default is 'best')
        bbox_to_anchor: set the part of the bounding box defined by loc
            at position (x, y) (if None, only loc defines the legend's loc)
        yrange: range of y (if None, use default range)
        log_scale: whether to use log scale or normal scale
    """
    # pylint: disable=too-many-arguments
    num_moments = len(moments)
    line_styles = ["-"] * num_moments if line_styles is None else line_styles
    line_widths = [4] * num_moments if line_widths is None else line_widths
    colors = [
        sns.xkcd_rgb[color] if color in sns.xkcd_rgb else color for color in colors
    ]

    for moment, color, line_style, line_width, label in zip(
        moments, colors, line_styles, line_widths, labels
    ):
        if moment.ndim > 1:
            # if more than 1 dimensions, also plot 1 sigma intervals
            axis.plot(
                depth,
                np.mean(moment, axis=0),
                ls=line_style,
                lw=line_width,
                color=color,
                label=label,
            )
            axis.fill_between(
                depth,
                np.percentile(moment, 16, axis=0),
                np.percentile(moment, 84, axis=0),
                color=color,
                alpha=0.1,
            )
            axis.plot(
                depth, np.percentile(moment, 16, axis=0), ls=":", color="grey", lw=0.7
            )
            axis.plot(
                depth, np.percentile(moment, 84, axis=0), ls=":", color="grey", lw=0.7
            )
        else:
            axis.plot(
                depth, moment, ls=line_style, lw=line_width, color=color, label=label
            )

    axis.tick_params(labelsize=18)
    plt.legend(
        fontsize=22,
        loc=loc,
        ncol=ncol,
        bbox_to_anchor=bbox_to_anchor,
        framealpha=1.0,
        edgecolor=plt.rcParams["axes.facecolor"],
        borderpad=0,
    )
    plt.xlim([0, np.max(depth)])
    if yrange is not None:
        plt.ylim(yrange)
    if log_scale:
        plt.yscale("log")


def plot_histo(
    axis: Axes,
    moment: np.ndarray,
    xfac: list[float],
    yfac: float,
    labels: list[str],
    annotation: str,
    xannotation: float,
):
    """Plot the histogram of moments at four different depths.

    # Args
        axis: axis used for plot
        moment: moments from all simulations at various depths
        xfac: factors of expansion of the x-axis
        yfac: factor of expansion of the y-axis
        labels: labels for legend
        annotation: complementary annotation with the name of the moment
        xannotation: position on the x-axis of complementary annotation
    """
    bins0, histo0 = make_histo(moment[:, 0])
    bins1, histo1 = make_histo(moment[:, 1])
    bins2, histo2 = make_histo(moment[:, 2])
    bins3, histo3 = make_histo(moment[:, 3])

    axis.plot(bins0, histo0, lw=2.0, color=sns.xkcd_rgb["blue"])
    axis.plot(bins1, histo1, lw=2.0, color=sns.xkcd_rgb["purple"])
    axis.plot(bins2, histo2, lw=2.0, color=sns.xkcd_rgb["magenta"])
    axis.plot(bins3, histo3, lw=2.0, color=sns.xkcd_rgb["red"])

    # set xlim and ylim
    axis.set_ylim(0, yfac * axis.get_ylim()[1])
    axis.set_xlim(xfac[0] * axis.get_xlim()[0], xfac[1] * axis.get_xlim()[1])
    axis.set_yticklabels([])  # remove y ticks

    # set legend
    axis.tick_params(labelsize=18)
    plt.legend(
        labels,
        fontsize=20,
        loc="center right",
        bbox_to_anchor=[1.0, 0.54],
        framealpha=1.0,
        edgecolor=plt.rcParams["axes.facecolor"],
        borderpad=0.4,
    )

    # set annotation
    axis.text(
        xannotation,
        0.86,
        annotation,
        transform=axis.transAxes,
        fontsize=20,
        zorder=100,
        bbox={
            "alpha": 1.0,
            "boxstyle": "round",
            "edgecolor": plt.rcParams["axes.facecolor"],
            "facecolor": plt.rcParams["axes.facecolor"],
            "pad": 0.3,
        },
    )


def make_histo(moment: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute histogram of the logarithm of moments at given depth.

    Steps:
        - Get the results from np.histogram
        - Convert bin edges to centered bins
        - Convert the histogram to a density
        (this step actually doesn't matter for our plots since we do not show the y-ticks)

    # Args
        moment: moments from all simulations at a given depth
    """
    moment = np.log(moment)
    histo, bins = np.histogram(moment, bins=50)
    delta_bins = bins[1] - bins[0]
    bins = np.array(bins)[:-1] + delta_bins / 2.0

    histo = np.array(histo) / (len(moment) * delta_bins)  # convert to density
    return bins, histo
