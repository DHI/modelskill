import numpy as np
import matplotlib as mpl
from matplotlib.patches import Rectangle, Polygon
from matplotlib.legend import Legend
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText


def wind_rose(
    data,
    mag_step=None,
    ax=None,
    dir_step=30,
    calm_threshold=None,
    resize_calm=0.05,
    calm_text="Calm",
    r_step=0.1,
    r_max=None,
    cbar_label=None,
    legend=True,
    cmap1="viridis",
    cmap2="Greys",
    mag_bins=None,
    invert_dir=False,
    max_bin=None,
    **kwargs,
):

    """Function for plotting Wave, Wind or Current roses with Calms.
    Parameters:
        data: array-like
            array with 2 or 4 columns
        mag_step: float, (optional) Default= None
            discretization for magnitude (delta_r, in radial direction )
        ax: Matplotlib axis Default= None
            Matplotlib axis to plot on defined as polar, it can be done using "subplot_kw = dict(projection = 'polar')". Default = None, new axis created.
        dir_step: float (optional) Default= None
            discretization for dir1s. Default= 30
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
        cbar_label: string. Default= None
            Label of the legend, if None, mag it is used. Examples cbar_label='Hm0', cbar_label='WS' or cbar_label='CS'
        legend: boolean. Default= True
            if None the legend is not ploted
        cmap1 : string. Default= 'viridis'
            colormap for main axis
        cmap2 : string. Default= 'Greys'
            colormap for secondary axis
        invert_dir: string (optional)
            invert directions (coming-from, going to). Default= False

    ------------------------------------------------------------------------------------------------
    Returns
        ax: Matplotlib axes
    """
    cols = data.columns.values
    data_1 = data.iloc[:, 0:2].copy()
    if data.shape[1] == 4:
        second_rose = True
        data_2 = data.iloc[:, 2:4].copy()
        mag1, dir1, mag2, dir2 = cols
    else:
        second_rose = False
        mag1, dir1 = cols

    # Magnitude bins
    ## Check if there's double counting in inputs
    if mag_bins != None:
        if np.any([max_bin != None, mag_step != None, calm_threshold != None]):
            flagged = [
                j
                for i, j in zip(
                    [max_bin, mag_step, calm_threshold],
                    ["max_bin", "mag_step", "calm_threshold"],
                )
                if i != None
            ]
            flagged = ", ".join(flagged)
            print(
                "Warning, both mag_bins and {} specified. Defaulting to mag_bins".format(
                    flagged
                )
            )

        # Set values
        mag_bins_ = np.array(mag_bins)
        ui = np.concatenate((mag_bins_, mag_bins_[[-1]] * 999))
        calm_threshold = thresh = ui[0]
        max_bin = ui[-2]
        dbin = np.diff(ui)[-2]

    else:
        if mag_step == None:
            mag_step = np.round(data_1[mag1].max() / 16, 1)
            if mag_step == 0:
                mag_step = np.round(data_1[mag1].max() / 16, 2)
            if second_rose:
                mag_step2 = np.round(data_2[mag2].max() / 16, 1)
                if mag_step2 == 0:
                    mag_step2 = np.round(data_2[mag2].max() / 16, 2)
                mag_step = max(mag_step, mag_step2)

        if calm_threshold is None:
            calm_threshold = mag_step

        thresh = calm_threshold

        # Auto find max
        magmax = data_1[mag1].max()
        if second_rose:
            magmax2 = data_2[mag2].max()
            magmax = max(magmax, magmax2)
        # Bins
        ui = np.arange(thresh, magmax, mag_step)
        ui = np.append(ui, np.max(data_1[mag1]))

        if max_bin is None:
            max_bin = magmax / 2
        dbin = ui[1] - ui[0]
        ui = np.arange(ui[0], max_bin + dbin * 2, dbin)
        ui[-1] = (
            ui[-1] * 2
        )  # safety factor * 2 as sometimes max is not in the iterations
        # Round bins to 3 decimal, this could be an input
        ui = [np.round(x, 3) for x in ui]

    ### create vectors to evaluate the histogram
    thetai = np.linspace(
        start=dir_step / 2,
        stop=360 + dir_step / 2,
        num=int(((360 + dir_step / 2) - dir_step / 2) / dir_step + 1),
    )
    thetac = thetai[:-1] + dir_step / 2

    ### compute total calms
    N = len(data_1)
    calm = len(data_1[data_1[mag1] < calm_threshold]) / N
    calm_value = calm

    calm_2 = len(data_2[data_2[mag2] < calm_threshold]) / N
    calm_value_2 = calm_2

    ### add 360 to all dir1s from 0 to dir_step/2
    if invert_dir == True:
        d = [np.mod(x - 180, 360) for x in data_1[dir1].values]
        d2 = [np.mod(x - 180, 360) for x in data_2[dir2].values]
    elif invert_dir == False:
        d = data_1[dir1]  # dir1 as coming from
        d2 = data_2[dir2]
    else:
        return None

    data_1["dir_proxy"] = d
    data_1.loc[data_1[dir1] < dir_step / 2, "dir_proxy"] = (
        data_1.loc[data_1[dir1] < dir_step / 2, dir1] + 360
    )

    data_2["dir_proxy"] = d2
    data_2.loc[data_2[dir2] < dir_step / 2, "dir_proxy"] = (
        data_2.loc[data_2[dir2] < dir_step / 2, dir2] + 360
    )

    ### compute histograms
    counts, _, _ = np.histogram2d(
        data_1[data_1[mag1] >= calm_threshold][mag1],
        data_1[data_1[mag1] >= calm_threshold]["dir_proxy"],
        bins=[ui, thetai],
    )
    counts = counts / N

    counts_2, _, _ = np.histogram2d(
        data_2[data_2[mag2] >= calm_threshold][mag2],
        data_2[data_2[mag2] >= calm_threshold]["dir_proxy"],
        bins=[ui, thetai],
    )
    counts_2 = counts_2 / N

    ### compute radial ticks
    if r_max == None:
        rmax = np.ceil((counts.sum(axis=0).max() + r_step) / r_step) * r_step
    else:
        rmax = r_max

    ri = np.linspace(0, rmax, int(rmax / r_step) + 1)
    ri = ri[1:-1]

    # Resize calm
    calm = resize_calm
    if isinstance(cmap1, str):
        cmap = mpl.cm.get_cmap(cmap1)
    elif isinstance(cmap1, mpl.colors.ListedColormap):
        cmap = cmap1
    else:
        raise Exception("Invalid cmap {}".format(cmap1))
    norm = mpl.colors.Normalize(vmin=thresh, vmax=max_bin + dbin * 2)
    # norm = mpl.colors.Normalize(vmin=thresh, vmax=np.mean(ui[-2:]))
    colors_ = [cmap(norm(x)) for x in ui]
    ### setup figure
    if ax == None:
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.linspace(0, 360, 5)[:-1], ["NORTH", "EAST", "SOUTH", "WEST"])
    ax.tick_params(pad=-24)

    ax.set_ylim(0, calm + rmax)
    ax.set_yticks(ri + calm)
    ax.set_yticklabels(["{:.0f}%".format(tick * 100) for tick in ri])
    ax.set_rlabel_position(5)
    ### add calms
    if calm_threshold > 0:
        ax.bar(np.pi, calm, color="white", ec="k", zorder=0)
        ax.bar(
            np.pi, calm, width=2 * np.pi, label="_nolegend_", color="white", zorder=3
        )
        ax.text(
            0.5,
            0.5,
            calm_text,
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )

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

            shape_x = [
                np.deg2rad(thetac[j] - dir_step / 2),
                np.deg2rad(thetac[j] + dir_step / 2),
                np.deg2rad(thetac[j] + dir_step / 2),
                np.deg2rad(thetac[j] - dir_step / 2),
            ]
            shape_y = [calm, calm, calm + cumcount[i, j], calm + cumcount[i, j]]

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

    percentages = np.sum(counts, axis=1) * 100

    # Labels
    if cbar_label != None:
        if cbar_label == "Hm0":
            label = f"N= {N}\nHm0 > {thresh} [m] \nHm0 [m] \n "
        elif cbar_label == "WS":
            label = f"N= {N}\nWS > {thresh} [m/s] \nWS [m/s] \n "
        elif cbar_label == "CS":
            label = f"N= {N}\nCS > {thresh} [m/s] \nCS [m/s] \n "
        else:
            label = cbar_label
    else:
        label = mag1

    legen_items = []

    for j in range(len(ui[1:-1])):
        legen_items.append(f"{np.round(ui[j],2)} - {np.round(ui[j+1],2)}")
    _items = [f"<{calm_threshold} ({np.round(calm_value*100,2)}%)"]
    _items.extend(legen_items)
    _items.append(f">= {ui[-2]} ({np.round( percentages[-1],2)}%)")

    if legend == True:
        handles = [Rectangle((0, 0), 1, 1, color=c, ec="k") for c in colors_[:]]
        handles[0].set_color("white")
        handles[0].set_ec("k")

        if "right_label" in kwargs:
            right_label = kwargs["right_label"]
            rlabel = f"{right_label}\n" + label
        else:
            rlabel = label

        leg = Legend(
            ax,
            handles[::-1],
            _items[::-1],
            frameon=True,
            title=rlabel,
            bbox_to_anchor=(1.05, -0.06, 0.1, 0.8),
            loc="lower left",
        )
        box_width = 0.32
        ax_right = ax.inset_axes([1.15, -0.05, box_width * 1.15, 0.5])
        ax_right.axis("off")
        ax.add_artist(leg)

    if second_rose:
        if isinstance(cmap2, str):
            cmap = mpl.cm.get_cmap(cmap2)
        elif isinstance(cmap2, mpl.colors.ListedColormap):
            cmap = cmap2
        else:
            raise Exception("Invalid cmap {}".format(cmap2))
        norm = mpl.colors.Normalize(vmin=0, vmax=max_bin + dbin * 2)
        percentages = np.sum(counts_2, axis=1) * 100
        colors_ = [cmap(norm(x)) for x in ui]
        if "left_label" in kwargs:
            left_label = kwargs["left_label"]
            llabel = f"{left_label}\n" + label
        else:
            llabel = label

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

                shape_x = [
                    np.deg2rad(thetac[j] - dir_step2 / 2),
                    np.deg2rad(thetac[j] + dir_step2 / 2),
                    np.deg2rad(thetac[j] + dir_step2 / 2),
                    np.deg2rad(thetac[j] - dir_step2 / 2),
                ]
                shape_y = [calm, calm, calm + cumcount[i, j], calm + cumcount[i, j]]

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

        _items[0] = f"<{calm_threshold} ({np.round(calm_value_2*100,2)}%)"
        _items[-1] = f">= {ui[-2]} ({np.round(percentages[-1],2)}%)"

        handles = [Rectangle((0, 0), 1, 1, color=c, ec="k") for c in colors_[:]]
        handles[0].set_color("white")
        handles[0].set_ec("k")

        leg = Legend(
            ax,
            handles[::-1],
            _items[::-1],
            frameon=True,
            title=llabel,
            bbox_to_anchor=(-0.13, -0.06, 0.1, 0.8),
            loc="lower right",
        )

        box_width = 0.32
        ax_left = ax.inset_axes([-box_width * 1.15, -0.05, box_width * 1.15, 0.5])
        ax_left.axis("off")
        ax.add_artist(leg)

    if "watermark" in kwargs:
        watermark = kwargs["watermark"]
        text = AnchoredText(
            watermark,
            "center right",
            frameon=False,
            borderpad=-27.5,
            prop=dict(fontsize="xx-small", alpha=0.15, rotation=90),
        )
        ax.add_artist(text)
    return ax
