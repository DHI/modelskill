from __future__ import annotations
from typing import Optional, Iterable, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.axes
    from ..model import DfsuModelResult
    from mikeio import GeometryFM2D

from ..model.point import PointModelResult
from ..model.track import TrackModelResult
from ..obs import Observation, PointObservation, TrackObservation
from ._misc import _get_ax


def spatial_overview(
    obs: Observation | Iterable[Observation],
    mod: Optional[
        DfsuModelResult
        | GeometryFM2D
        | Iterable[DfsuModelResult]
        | Iterable[GeometryFM2D]
    ] = None,
    ax=None,
    figsize: Optional[Tuple] = None,
    title: Optional[str] = None,
) -> matplotlib.axes.Axes:
    """Plot observation points on a map showing the model domain

    Parameters
    ----------
    obs: list[Observation]
        List of observations to be shown on map
    mod : DfsuModelResult, optional
        Model domain to be shown as outline
    ax: matplotlib.axes, optional
        Adding to existing axis, instead of creating new fig
    figsize : (float, float), optional
        figure size, by default None
    title: str, optional
        plot title, default empty

    See Also
    --------
    temporal_coverage

    Returns
    -------
    matplotlib.axes.Axes
        The matplotlib axes object

    Examples
    --------
    ```{python}
    import modelskill as ms
    from pathlib import Path
    p = Path("../data/SW")

    o1 = ms.PointObservation(p / "HKNA_Hm0.dfs0", item=0, x=4.2420, y=52.6887, name="HKNA")
    o2 = ms.TrackObservation(p / "Alti_c2_Dutch.dfs0", item=3, name="c2")
    mr = ms.DfsuModelResult(p / "HKZN_local_2017_DutchCoast.dfsu", name='SW_1', item=0)
    ms.plotting.spatial_overview([o1, o2], mr)
    ```
    """
    obs = [] if obs is None else list(obs) if isinstance(obs, Iterable) else [obs]  # type: ignore
    mods = [] if mod is None else list(mod) if isinstance(mod, Iterable) else [mod]  # type: ignore

    ax = _get_ax(ax=ax, figsize=figsize)

    # TODO: support Gridded ModelResults
    for m in mods:
        if isinstance(m, (PointModelResult, TrackModelResult)):
            raise ValueError(
                f"Model type {type(m)} not supported. Only DfsuModelResult and mikeio.GeometryFM supported!"
            )
        if hasattr(m, "data") and hasattr(m.data, "geometry"):
            # mod_name = m.name  # TODO: better support for multiple models
            g = m.data.geometry
        else:
            g = m

            # TODO this is not supported for all model types
        g.plot.outline(ax=ax)  # type: ignore

    for o in obs:
        if isinstance(o, PointObservation):
            ax.scatter(x=o.x, y=o.y, marker="x")
        elif isinstance(o, TrackObservation):
            if o.n_points < 10000:
                ax.scatter(x=o.x, y=o.y, c=o.values, marker=".", cmap="Reds")
            else:
                print(f"{o.name}: Too many points to plot")
                # TODO: group by lonlat bin or sample randomly
        else:
            raise ValueError(
                f"Could not show observation {o}. Only PointObservation and TrackObservation supported."
            )

    xlim = ax.get_xlim()
    offset_x = 0.02 * (xlim[1] - xlim[0])

    for o in obs:
        if isinstance(o, PointObservation):
            # TODO adjust xlim to accomodate text
            ax.annotate(o.name, (o.x + offset_x, o.y))  # type: ignore

    if not title:
        title = "Spatial coverage"
    ax.set_title(title)

    return ax
