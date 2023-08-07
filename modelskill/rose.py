from typing import List, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.legend import Legend
from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import Polygon, Rectangle


def wind_rose(
    data,
    mag_step: Optional[float] = None,
    n_sectors: int = 16,
    calm_threshold=None,  # TODO rename to vmin?
    resize_calm=0.05,
    calm_text="Calm",
    r_step: float = 0.1,
    r_max: Optional[float] = None,
    legend=True,
    cmap1: str = "viridis",
    cmap2: str = "Greys",
    mag_bins: Optional[List[float]] = None,
    max_bin=None,  # TODO rename to vmax?
    n_labels: Optional[int] = None,
    ax=None,
    **kwargs,
):

    """Function for plotting Wave, Wind or Current roses with Calms.
    Parameters:
        data: array-like
            array with 4 columns
        mag_step: float, (optional) Default= None
            discretization for magnitude (delta_r, in radial direction )
        n_sectors: int (optional) Default= 16
        calm_threshold: float (optional) Default= None (auto calculated)
            minimum value for data being counted as valid (i.e. below this is calm)
        resize_calm: bool or float (optional) Default: 0.05
            resize the size of calm in plot. Useful when the calms are very large or small.
        calm_text: str (optional) Default: 'Calm'
            text to display in calm. Set to None or '' for blank
        r_step: float (optional) Default= 0.1
            radial axis discretization. By default this is every 10%.
        r_max: float (optional) Default= None
            maximum radius (%) of plot, eg if 50% wanted then r_max=0.5 By default this is automatically calculated.
        max_bin:  float (optional) Default= None
            max value to truncate the data, eg,  max_bin=1.0 if hm0=1m is the desired final bin.
        mag_bins : array of floats (optional) Default = None
            force bins to array of values, e.g. when specifying non-equidistant bins.
        legend: boolean. Default= True
            if None the legend is not ploted
        cmap1 : string. Default= 'viridis'
            colormap for main axis
        cmap2 : string. Default= 'Greys'
            colormap for secondary axis
        n_labels : int. Default= 4
            number of labels in the polar plot, choose between 4, 8 or 16, default is to use the same as n_sectors
        ax: Matplotlib axis Default= None
            Matplotlib axis to plot on defined as polar, it can be done using "subplot_kw = dict(projection = 'polar')". Default = None, new axis created.

    ------------------------------------------------------------------------------------------------
    Returns
        ax: Matplotlib axes
    """
    data_1 = data[:, 0:2]
    data_2 = data[:, 2:4]

    # magnitude bins
    ui, vmin, vmax = pretty_intervals(
        data_1[:, 0],
        data_2[:, 0],
        mag_bins,
        mag_step,
        calm_threshold,
        max_bin,
    )

    dir_step = 360 // n_sectors

    n_labels = n_sectors if n_labels is None else n_labels

    ### create vectors to evaluate the histogram
    thetai = np.linspace(
        start=dir_step / 2,
        stop=360 + dir_step / 2,
        num=int(((360 + dir_step / 2) - dir_step / 2) / dir_step + 1),
    )
    thetac = thetai[:-1] + dir_step / 2

    mask_1 = data_1[:, 0] >= vmin
    mask_2 = data_2[:, 0] >= vmin

    ### compute total calms
    N = len(data_1)
    calm = len(data_1[~mask_1]) / N
    calm2 = len(data_2[~mask_2]) / N

    counts, counts_2 = _calc_histograms(
        data_1=data_1, mask_1=mask_1, data_2=data_2, mask_2=mask_2, ui=ui, thetai=thetai
    )

    ### compute radial ticks
    ri, rmax = _calc_radial_ticks(counts=counts, step=r_step, stop=r_max)

    # Resize calm
    calm = resize_calm

    cmap = _get_cmap(cmap1)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    colors_ = [cmap(norm(x)) for x in ui]

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    labels = directional_labels(n_labels)
    grid = np.linspace(0, 360, n_labels + 1)[:-1]
    ax.set_thetagrids(grid, labels)

    # ax.tick_params(pad=-24)

    ax.set_ylim(0, calm + rmax)
    ax.set_yticks(ri + calm)
    tick_labels = [f"{tick * 100 :.0f}%" for tick in ri]
    ax.set_yticklabels(tick_labels)
    ax.set_rlabel_position(5)
    ### add calms
    if vmin > 0:
        _add_calms_to_ax(ax, threshold=calm, text=calm_text)

    ### plot each bar of the histogram
    patches = []
    colors = []
    cumcount = counts.cumsum(axis=0)  # Integrate histogram
    arc_res = dir_step  # numnber of points along arc = 1 pt per degree.

    arc_x = np.deg2rad(
        np.linspace(thetac - dir_step / 2, thetac + dir_step / 2, arc_res)
    )
    # Loop through velocities
    for i, vel in enumerate(ui[1:]):
        # Loop through directions
        for j, c in enumerate(counts[i]):
            arc_xj = np.concatenate([arc_x[:, j], np.flip(arc_x[:, j])])
            arc_yj = np.concatenate(
                [np.full(arc_res, calm), np.full(arc_res, calm + cumcount[i, j])]
            )

            polygon = Polygon(
                np.array((arc_xj, arc_yj)).T, True
            )  # Conflict with shapely
            patches.append(polygon)
            colors.append(cmap(norm(vel)))

    p = PatchCollection(
        np.flip(patches),
        facecolors=np.flip(colors, axis=0),
        edgecolor="k",
        linewidth=0.5,
    )
    ax.add_collection(p)

    if legend:
        _add_legend_to_ax(
            ax,
            colors=colors_,
            vmin=vmin,
            ui=ui,
            calm=calm,
            counts=counts,
            label="Model",
            primary=True,
        )

    cmap = _get_cmap(cmap2)
    norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
    colors_ = [cmap(norm(x)) for x in ui]

    ### plot each bar of the histogram
    patches = []
    colors = []
    cumcount = counts_2.cumsum(axis=0)

    dir_step2 = dir_step / 2

    arc_x = np.deg2rad(
        np.linspace(thetac - dir_step2 / 2, thetac + dir_step2 / 2, arc_res)
    )
    # Loop through velocities
    for i, vel in enumerate(ui[1:]):
        # Loop through directions
        for j, c in enumerate(counts[i]):
            arc_xj = np.concatenate([arc_x[:, j], np.flip(arc_x[:, j])])
            arc_yj = np.concatenate(
                [np.full(arc_res, calm), np.full(arc_res, calm + cumcount[i, j])]
            )

            polygon = Polygon(
                np.array((arc_xj, arc_yj)).T, True
            )  # Conflict with shapely
            patches.append(polygon)
            colors.append(cmap(norm(vel)))

    p = PatchCollection(
        np.flip(patches),
        facecolors=np.flip(colors, axis=0),
        edgecolor="k",
        linewidth=0.5,
    )
    ax.add_collection(p)

    if legend:
        _add_legend_to_ax(
            ax,
            colors=colors_,
            vmin=vmin,
            ui=ui,
            calm=calm2,
            counts=counts_2,
            label="Observation",
            primary=False,
        )

    if "watermark" in kwargs:
        _add_watermark_to_ax(ax, kwargs["watermark"])
    return ax


def directional_labels(n: int) -> Tuple[str, ...]:
    """Return labels for n directions.

    Parameters
    ----------
    n : int
        Number of directions. Must be 4, 8 or 16.

    Returns
    -------
    Tuple[str, ...]
        labels

    Examples
    --------
    >>> directional_labels(4)
    ('N', 'E', 'S', 'W')
    """
    if n == 4:
        return ("N", "E", "S", "W")
    elif n == 8:
        return ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
    elif n == 16:
        return (
            "N",
            "NNE",
            "NE",
            "ENE",
            "E",
            "ESE",
            "SE",
            "SSE",
            "S",
            "SSW",
            "SW",
            "WSW",
            "W",
            "WNW",
            "NW",
            "NNW",
        )
    else:
        raise ValueError("n must be 4, 8 or 16")


def pretty_intervals(
    data_1: np.typing.ArrayLike,
    data_2: np.typing.ArrayLike,
    mag_bins: Optional[List[float]] = None,
    mag_step: Optional[float] = None,
    vmin: Optional[float] = None,
    max_bin: Optional[float] = None,
    n_decimals: int = 3,
) -> Tuple[np.ndarray, float, float]:
    """Pretty intervals for the magnitude bins"""

    data_1_max = data_1.max()
    data_2_max = data_2.max()

    # Magnitude bins
    ## Check if there's double counting in inputs
    if mag_bins is not None:

        # Set values

        assert len(mag_bins) >= 3, "Must have at least 3 bins"
        mag_bins_ = np.array(mag_bins)
        ui = np.concatenate((mag_bins_, mag_bins_[[-1]] * 999))
        vmin = ui[0]
        max_bin = ui[-2]
        dbin = np.diff(ui)[-2]
        vmax = max_bin + dbin * 2  # TODO what is happening here?

    else:
        if mag_step is None:
            mag_step = _calc_mag_step(data_1_max, data_2_max)

        if vmin is None:
            vmin = mag_step

        # Auto find max
        magmax = data_1_max

        magmax2 = data_2_max
        magmax = max(magmax, magmax2)
        # Bins
        ui = np.arange(vmin, magmax, mag_step)
        ui = np.append(ui, data_1_max)

        if max_bin is None:
            max_bin = magmax / 2
        dbin = ui[1] - ui[0]
        vmax = max_bin + dbin * 2
        ui = np.arange(ui[0], vmax, dbin)
        ui[-1] = (
            ui[-1] * 2
        )  # safety factor * 2 as sometimes max is not in the iterations
        # Round bins to make them pretty
        ui = ui.round(n_decimals)

    # TODO return a better object?
    return ui, vmin, vmax


def _calc_mag_step(xmax: float, ymax: float, factor: float = 16.0):
    """
    Calculate the magnitude step size for a rose plot.

    Parameters
    ----------
    x : float
        The maximum value of the histogram.
    y : float
        The maximum value of the histogram.
    factor : float, optional
        The factor to use to calculate the magnitude step size, by default 16.0

    Returns
    -------
    float
    """
    mag_step = np.round(xmax / factor, 1)
    if mag_step == 0:
        mag_step = np.round(xmax / factor, 2)

    mag_step2 = np.round(ymax / factor, 1)
    if mag_step2 == 0:
        mag_step2 = np.round(ymax / factor, 2)
    mag_step = max(mag_step, mag_step2)
    return mag_step


def _calc_histograms(
    *, data_1, mask_1, data_2, mask_2, ui, thetai
) -> Tuple[np.ndarray, np.ndarray]:

    N = len(data_1)
    counts, _, _ = np.histogram2d(
        data_1[mask_1][:, 0],
        data_1[mask_1][:, 1],
        bins=[ui, thetai],
    )
    counts = counts / N

    counts_2, _, _ = np.histogram2d(
        data_2[mask_2][:, 0],
        data_2[mask_2][:, 1],
        bins=[ui, thetai],
    )
    counts_2 = counts_2 / N

    return counts, counts_2


def _calc_radial_ticks(
    *, counts: np.ndarray, step: float, stop: Optional[float]
) -> np.ndarray:
    cmax = counts.sum(axis=0).max()
    if stop is None:
        rmax = np.ceil((cmax + step) / step) * step
    else:
        rmax = stop

    ri = np.linspace(0, rmax, int(rmax / step) + 1)
    ri = ri[1:-1]

    return ri, rmax


def _add_calms_to_ax(ax, *, threshold: np.ndarray, text: str) -> None:
    ax.bar(np.pi, threshold, color="white", ec="k", zorder=0)
    ax.bar(
        np.pi, threshold, width=2 * np.pi, label="_nolegend_", color="white", zorder=3
    )
    ax.text(
        0.5,
        0.5,
        text,
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
    )


def _add_legend_to_ax(
    ax, *, colors, vmin, ui, calm, counts, label, primary: bool
) -> None:

    percentages = np.sum(counts, axis=1) * 100

    legend_items = []

    for j in range(len(ui[1:-1])):
        legend_items.append(f"{np.round(ui[j],2)} - {np.round(ui[j+1],2)}")
    items = [f"<{vmin} ({np.round(calm*100,2)}%)"]
    items.extend(legend_items)
    items.append(f">= {ui[-2]} ({np.round(percentages[-1], 2)}%)")

    handles = [Rectangle((0, 0), 1, 1, color=c, ec="k") for c in colors]
    handles[0].set_color("white")
    handles[0].set_ec("k")

    if primary:
        bbox_to_anchor = (1.05, -0.06, 0.1, 0.8)
        loc = "lower left"
    else:
        bbox_to_anchor = (-0.13, -0.06, 0.1, 0.8)
        loc = "lower right"

        # TODO why are these two lines here?
        items[0] = f"<{vmin} ({np.round(calm*100,2)}%)"
        items[-1] = f">= {ui[-2]} ({np.round(percentages[-1], 2)}%)"

    leg = Legend(
        ax,
        handles[::-1],
        items[::-1],
        frameon=True,
        title=label,
        bbox_to_anchor=bbox_to_anchor,
        loc=loc,
    )
    box_width = 0.32

    if primary:
        ax_left = ax.inset_axes([-box_width * 1.15, -0.05, box_width * 1.15, 0.5])
        ax_left.axis("off")
    else:
        ax_right = ax.inset_axes([1.15, -0.05, box_width * 1.15, 0.5])
        ax_right.axis("off")
    ax.add_artist(leg)


def _add_watermark_to_ax(ax, watermark: str) -> None:
    text = AnchoredText(
        watermark,
        "center right",
        frameon=False,
        borderpad=-27.5,
        prop=dict(fontsize="xx-small", alpha=0.15, rotation=90),
    )
    ax.add_artist(text)


def _get_cmap(cmap: Union[str, mpl.colors.ListedColormap]) -> mpl.colors.ListedColormap:
    if isinstance(cmap, str):
        cmap = mpl.colormaps[cmap]
    elif isinstance(cmap, mpl.colors.ListedColormap):
        cmap = cmap
    else:
        raise ValueError(f"Invalid cmap {cmap}")
    return cmap
