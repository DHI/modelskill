from __future__ import annotations
from typing import Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.axes

import matplotlib.pyplot as plt
import numpy as np

from ._misc import _get_fig_ax


def temporal_coverage(
    obs=None,
    mod=None,
    *,
    limit_to_model_period=True,
    marker="_",
    ax=None,
    figsize=None,
    title=None,
) -> matplotlib.axes.Axes:
    """Plot graph showing temporal coverage for all observations and models

    Parameters
    ----------
    obs : List[Observation], optional
        Show observation(s) as separate lines on plot
    mod : List[ModelResult], optional
        Show model(s) as separate lines on plot, by default None
    limit_to_model_period : bool, optional
        Show temporal coverage only for period covered
        by the model, by default True
    marker : str, optional
        plot marker for observations, by default "_"
    ax: matplotlib.axes, optional
        Adding to existing axis, instead of creating new fig
    figsize : Tuple(float, float), optional
        size of figure, by default (7, 0.45*n_lines)
    title: str, optional
        plot title, default empty

    See Also
    --------
    spatial_overview

    Returns
    -------
    matplotlib.axes.Axes
        The matplotlib axes object

    Examples
    --------
    >>> import modelskill as ms
    >>> o1 = ms.PointObservation('HKNA_Hm0.dfs0', item=0, x=4.2420, y=52.6887, name="HKNA")
    >>> o2 = ms.TrackObservation("Alti_c2_Dutch.dfs0", item=3, name="c2")
    >>> mr1 = ms.DfsuModelResult('HKZN_local_2017_DutchCoast.dfsu', name='SW_1', item=0)
    >>> mr2 = ms.DfsuModelResult('HKZN_local_2017_DutchCoast_v2.dfsu', name='SW_2', item=0)
    >>> ms.plotting.temporal_coverage([o1, o2], [mr1, mr2])
    >>> ms.plotting.temporal_coverage([o1, o2], mr2, limit_to_model_period=False)
    >>> ms.plotting.temporal_coverage(o2, [mr1, mr2], marker=".")
    >>> ms.plotting.temporal_coverage(mod=[mr1, mr2], figsize=(5,3))
    """
    obs = [] if obs is None else list(obs) if isinstance(obs, Sequence) else [obs]
    mod = [] if mod is None else list(mod) if isinstance(mod, Sequence) else [mod]

    n_lines = len(obs) + len(mod)
    if figsize is None:
        ysize = max(2.0, 0.45 * n_lines)
        figsize = (7, ysize)

    fig, ax = _get_fig_ax(ax=ax, figsize=figsize)
    y = np.repeat(0.0, 2)
    labels = []

    if len(mod) > 0:
        for mr in mod:
            y += 1.0
            plt.plot([mr.time[0], mr.time[-1]], y)
            labels.append(mr.name)

    for o in obs:
        y += 1.0
        plt.plot(o.time, y[0] * np.ones(len(o.time)), marker, markersize=5)
        labels.append(o.name)

    if len(mod) > 0 and limit_to_model_period:
        mr = mod[0]  # take first model
        plt.xlim([mr.time[0], mr.time[-1]])

    plt.yticks(np.arange(n_lines) + 1, labels)
    if len(mod) > 0:
        for j in range(len(mod)):
            ax.get_yticklabels()[j].set_fontstyle("italic")
            ax.get_yticklabels()[j].set_weight("bold")
            # set_color("#004165")
    fig.autofmt_xdate()

    if title:
        ax.set_title(title)
    return ax
