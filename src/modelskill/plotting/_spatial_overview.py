from __future__ import annotations
from typing import Iterable, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..model import DfsuModelResult
    from mikeio import GeometryFM2D

from ..model.point import PointModelResult
from ..model.track import TrackModelResult
from ..model.vertical import VerticalModelResult
from ..obs import Observation, PointObservation, TrackObservation, VerticalObservation
from ._backend import (
    Backend,
    PlotResult,
    reject_matplotlib_axes,
    validate_backend,
)
from ._misc import _get_ax
from . import _plotly

# a track with more points than this is dropped rather than drawn; a
# multi-million-point altimetry track is unreadable and slow on either backend
MAX_TRACK_POINTS = 10000


def spatial_overview(
    obs: Observation | Iterable[Observation],
    mod: (
        DfsuModelResult
        | GeometryFM2D
        | Iterable[DfsuModelResult]
        | Iterable[GeometryFM2D]
    )
    | None = None,
    ax=None,
    figsize: Tuple | None = None,
    title: str | None = None,
    backend: Backend = "matplotlib",
) -> PlotResult:
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
    backend : str, optional
        "matplotlib" (static) or "plotly" (interactive), by default "matplotlib"

    See Also
    --------
    temporal_coverage

    Returns
    -------
    matplotlib.axes.Axes or plotly.graph_objects.Figure
        The axes (matplotlib backend) or figure (plotly backend)

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
    validate_backend(backend)
    reject_matplotlib_axes(ax, backend)

    obs = [] if obs is None else list(obs) if isinstance(obs, Iterable) else [obs]  # type: ignore
    mods = [] if mod is None else list(mod) if isinstance(mod, Iterable) else [mod]  # type: ignore

    geometries = [_model_geometry(m) for m in mods]
    points, tracks = _classify_observations(obs)

    if backend == "plotly":
        return _plotly.spatial_overview(
            outlines=_domain_outlines(geometries),
            points=points,
            tracks=tracks,
            title=title,
            figsize=figsize,
        )

    ax = _get_ax(ax=ax, figsize=figsize)

    for g in geometries:
        g.plot.outline(ax=ax)  # type: ignore

    for _, x, y in points:
        ax.scatter(x=x, y=y, marker="x")

    for _, x, y in tracks:
        ax.scatter(x=x, y=y, marker=".")

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


def _classify_observations(obs):
    """Split observations into labelled points and tracks, for either backend

    Raises
    ------
    ValueError
        if an observation is neither a point nor a track observation
    """
    points, tracks = [], []
    for o in obs:
        if isinstance(o, (PointObservation, VerticalObservation)):
            points.append((o.name, o.x, o.y))
        elif isinstance(o, TrackObservation):
            if len(o.x) < MAX_TRACK_POINTS:
                tracks.append((o.name, o.x, o.y))
            else:
                # TODO: group by lonlat bin or sample randomly
                print(f"{o.name}: Too many points to plot")
        else:
            raise ValueError(
                f"Could not show observation {o}. Only PointObservation and TrackObservation supported."
            )
    return points, tracks


def _domain_outlines(geometries):
    """Boundary polygons of the model domains, islands included"""
    return [
        polygon.xy
        for g in geometries
        for polygon in (
            list(g.boundary_polygons.exteriors) + list(g.boundary_polygons.interiors)
        )
    ]


def _model_geometry(m):
    """The 2D flexible mesh geometry of a model result or geometry"""
    # TODO: support Gridded ModelResults
    if isinstance(m, (PointModelResult, TrackModelResult, VerticalModelResult)):
        raise ValueError(
            f"Model type {type(m)} not supported. Only DfsuModelResult and mikeio.GeometryFM supported!"
        )
    if hasattr(m, "data") and hasattr(m.data, "geometry"):
        # TODO: better support for multiple models
        g = m.data.geometry
    else:
        g = m

    # mikeio's 3D geometries (GeometryFM3D) cannot be plotted directly
    if hasattr(g, "to_2d_geometry"):
        g = g.to_2d_geometry()
    return g
