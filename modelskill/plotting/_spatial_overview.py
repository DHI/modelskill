from __future__ import annotations
from typing import List, Optional, Sequence, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.axes

from ..model.point import PointModelResult
from ..model.track import TrackModelResult
from ..obs import Observation, PointObservation, TrackObservation
from ._misc import _get_ax


def spatial_overview(
    obs: List[Observation],
    mod=None,
    ax=None,
    figsize: Optional[Tuple] = None,
    title: Optional[str] = None,
) -> matplotlib.axes.Axes:
    """Plot observation points on a map showing the model domain

    Parameters
    ----------
    obs: list[Observation]
        List of observations to be shown on map
    mod : Union[ModelResult, mikeio.GeometryFM], optional
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
    >>> import modelskill as ms
    >>> o1 = ms.PointObservation('HKNA_Hm0.dfs0', item=0, x=4.2420, y=52.6887, name="HKNA")
    >>> o2 = ms.TrackObservation("Alti_c2_Dutch.dfs0", item=3, name="c2")
    >>> mr1 = ms.DfsuModelResult('HKZN_local_2017_DutchCoast.dfsu', name='SW_1', item=0)
    >>> mr2 = ms.DfsuModelResult('HKZN_local_2017_DutchCoast_v2.dfsu', name='SW_2', item=0)
    >>> ms.plotting.spatial_overview([o1, o2], [mr1, mr2])
    """
    obs = [] if obs is None else list(obs) if isinstance(obs, Sequence) else [obs]  # type: ignore
    mod = [] if mod is None else list(mod) if isinstance(mod, Sequence) else [mod]  # type: ignore

    ax = _get_ax(ax=ax, figsize=figsize)
    offset_x = 1  # TODO: better default

    for m in mod:
        # TODO: support Gridded ModelResults
        if isinstance(m, (PointModelResult, TrackModelResult)):
            raise ValueError(
                f"Model type {type(m)} not supported. Only DfsuModelResult and mikeio.GeometryFM supported!"
            )
        if hasattr(m, "data") and hasattr(m.data, "geometry"):
            # mod_name = m.name  # TODO: better support for multiple models
            m = m.data.geometry
        if hasattr(m, "node_coordinates"):
            xn = m.node_coordinates[:, 0]
            offset_x = 0.02 * (max(xn) - min(xn))
        m.plot.outline(ax=ax)

    for o in obs:
        if isinstance(o, PointObservation):
            ax.scatter(x=o.x, y=o.y, marker="x")
            ax.annotate(o.name, (o.x + offset_x, o.y))  # type: ignore
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

    if not title:
        title = "Spatial coverage"
    ax.set_title(title)

    return ax
