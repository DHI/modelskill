from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from modelskill.timeseries import (
    TimeSeries,
    _parse_network_breakpoint_input,
    _parse_network_node_input,
)
from ._base import SelectedItems
from ..obs import NodeObservation, ReachObservation
from ..timeseries._coords import network_location
from ..quantity import Quantity
from ..types import GeometryType

if TYPE_CHECKING:
    from mikeio1d.network import Network


def _network_class() -> type[Network]:
    # Imported here, not at module scope, so this module stays importable
    # without the optional network dependencies (ADR-010).
    try:
        from mikeio1d.network import Network
    except ImportError as err:
        raise ImportError(
            "NetworkModelResult needs the network topology layer from mikeio1d, "
            "which the 'network' extra installs: pip install modelskill[network]"
        ) from err
    return Network


class NodeModelResult(TimeSeries):
    """Model result at one network location.

    What :meth:`NetworkModelResult.extract` returns: the timeseries of a single
    node or break point, carrying in its coordinates the location it was taken
    from. Extract one from a :class:`NetworkModelResult` rather than building it
    directly, since a location is named by the network it belongs to.

    Parameters
    ----------
    data : xr.Dataset
        Timeseries for one location, carrying a ``node`` coordinate, or a
        ``reach`` coordinate with ``distance`` for a break point.

    Raises
    ------
    TypeError
        If data is not an xarray.Dataset.
    ValueError
        If data carries no network location.

    See Also
    --------
    NetworkModelResult.extract : Extract a model result at a node or a reach.
    """

    def __init__(self, data: xr.Dataset) -> None:
        if not isinstance(data, xr.Dataset):
            raise TypeError(
                "'NodeModelResult' takes an xarray.Dataset carrying its own "
                f"location, got {type(data).__name__}. A model result for a "
                "network location comes from NetworkModelResult.extract()."
            )
        if GeometryType.from_network_coords(data) is None:
            raise ValueError(
                "'NodeModelResult' needs data carrying a 'node' coordinate, or a "
                "'reach' coordinate for a reach or a break point. A model result "
                "for a network location comes from NetworkModelResult.extract()."
            )
        data_var = str(list(data.data_vars)[0])
        data[data_var].attrs["kind"] = "model"
        super().__init__(data=data)

    @classmethod
    def _from_network(
        cls,
        data: xr.Dataset,
        *,
        location: str | tuple[str, float | None],
        node_index: int,
        name: str | None = None,
        item: str | int | None = None,
        quantity: Quantity | None = None,
        aux_items: Sequence[int | str] | None = None,
    ) -> NodeModelResult:
        """Build a result from the data a network keeps at one location.

        Parameters
        ----------
        data : xr.Dataset
            Timeseries for one location, as the network stored it.
        location : str or tuple of (str, float or None)
            A node name, or a break point as ``(reach_id, distance)``.
        node_index : int
            The integer the network used for this location, recorded as
            provenance. Nothing reads it back.
        name : str, optional
            The name of the model result, by default None (taken from the item)
        item : str or int, optional
            Item to take when the data holds more than one, by default None
        quantity : Quantity, optional
            Model quantity, by default None (inferred from the data)
        aux_items : sequence of int or str, optional
            Auxiliary items, by default None

        Returns
        -------
        NodeModelResult
            the result at that location
        """
        if isinstance(location, tuple):
            reach, distance = location
            ds = _parse_network_breakpoint_input(
                data,
                name=name,
                item=item,
                quantity=quantity,
                aux_items=aux_items,
                reach=str(reach),
                distance=distance,
            )
        else:
            ds = _parse_network_node_input(
                data,
                name=name,
                item=item,
                quantity=quantity,
                node=location,
                aux_items=aux_items,
            )
        return cls(ds.assign_coords(node_index=int(node_index)))

    @property
    def node(self) -> Any:
        """Where this result was extracted, as its network named it."""
        return network_location(self.data)

    def _location_repr(self) -> str | None:
        return f"Location: {self.node}"

    @property
    def node_index(self) -> int | None:
        """Graph integer this location had in the network it came from, if recorded.

        Provenance only. Nothing reads it back: the numbering belongs to one
        network built by one version, so a saved result is identified by
        :attr:`node` instead.
        """
        if "node_index" not in self.data.coords:
            return None
        return int(np.atleast_1d(self.data.coords["node_index"].values)[0])


class NetworkModelResult:
    """Model result for network data with time and node dimensions.

    Construct one from a result file, or from a :class:`mikeio1d.network.Network`
    already built. Observations name the location they sit at, and no spatial
    interpolation is performed.

    Parameters
    ----------
    data : Network, str or Path
        Path to a ``.res1d``, ``.res11`` or ``.res`` result file, or a
        :class:`mikeio1d.network.Network`.
    name : str, optional
        The name of the model result,
        by default None (will be set to first data variable name)
    item : str | int | None, optional
        If multiple items/arrays are present in the input an item
        must be given (as either an index or a string), by default None
    quantity : Quantity, optional
        Model quantity
    aux_items : list[int | str], optional
        Auxiliary items, by default None

    Examples
    --------
    >>> import modelskill as ms
    >>> mr = ms.NetworkModelResult("model.res1d", item="WaterLevel")
    >>> obs = ms.NodeObservation(data, at="node_A")
    >>> extracted = mr.extract(obs)

    Open the network yourself to name EPANET companion files, or to keep memory
    down on a large model by reading only the locations you will score:

    >>> from mikeio1d.network import Network
    >>> network = Network.open("model.res1d", nodes=["node_A", "node_B"])
    >>> mr = ms.NetworkModelResult(network, name="MyModel")

    Notes
    -----
    The network is used as given, not copied, so ``mr.network`` is the caller's
    object.

    See Also
    --------
    mikeio1d.network.Network.open : Read a network from a result file.
    """

    def __init__(
        self,
        data: Network | str | Path,
        *,
        name: str | None = None,
        item: str | int | None = None,
        quantity: Quantity | None = None,
        aux_items: Sequence[int | str] | None = None,
    ):
        network_class = _network_class()
        if isinstance(data, (str, Path)):
            self.network = network_class.open(data)
        elif isinstance(data, network_class):
            self.network = data
        else:
            raise TypeError(
                "NetworkModelResult takes a mikeio1d.network.Network or a path to a "
                f"result file, got {type(data).__name__}"
            )

        ds = self.network.to_dataset()
        sel_items = SelectedItems.parse(
            list(ds.data_vars), item=item, aux_items=aux_items
        )
        name = name or sel_items.values

        self.data = ds[sel_items.all]
        self.name = name
        self.sel_items = sel_items

        if quantity is None:
            # Read straight off the attributes rather than through
            # Quantity.from_cf_attrs, which needs a unit as well as a name and
            # reports neither without both: a result file names its quantity and
            # carries no unit for it, so that would report nothing at all. A unit
            # is still used if one ever travels with the data.
            attrs = self.data[sel_items.values].attrs
            quantity = Quantity(
                name=attrs.get("long_name") or str(sel_items.values),
                unit=attrs.get("units", ""),
            )
        self.quantity = quantity

        # Mark data variables as model data
        self.data[sel_items.values].attrs["kind"] = "model"

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>: {self.name}"

    #: Coordinates mikeio1d puts on to_dataset() to say what each column is. They
    #: are re-applied through NodeModelResult, which knows modelskill's names for
    #: them, so they never reach a comparer under these.
    _UPSTREAM_IDENTITY_COORDS = ("name", "reach", "distance")

    @property
    def time(self) -> pd.DatetimeIndex:
        """Return the time coordinate as a pandas.DatetimeIndex."""
        return pd.DatetimeIndex(self.data.time.to_index())

    def extract(
        self,
        observation: NodeObservation | ReachObservation,
    ) -> NodeModelResult:
        """Extract ModelResult at exact node or reach locations

        Parameters
        ----------
        observation : NodeObservation or ReachObservation
            observation naming a node, a breakpoint, or a reach

        Returns
        -------
        NodeModelResult
            extracted model result
        """
        if isinstance(observation, NodeObservation):
            return self._extract_node(observation)
        elif isinstance(observation, ReachObservation):
            return self._extract_reach(observation)
        else:
            raise TypeError(
                f"NetworkModelResult supports NodeObservation and ReachObservation, got {type(observation).__name__}"
            )

    def _extract_node(self, observation: NodeObservation) -> NodeModelResult:
        node_id = self._resolve_alias(observation.at)

        if node_id not in self.data.indexes["node"]:
            raise ValueError(
                f"{observation.at!r} exists in the network topology but its "
                "timeseries was not loaded. Re-create the NetworkModelResult with "
                "the relevant nodes populated, e.g. "
                "NetworkModelResult(Network.open(path, nodes=[...]))."
            )

        # A location that carries no data for this quantity is all-NaN here, just
        # as it is on the reach path: MIKE 1D stores quantities at different grid
        # points, so a breakpoint carrying Discharge may carry no WaterLevel.
        item = self.sel_items.values
        if not bool(self.data[item].sel(node=node_id).notnull().any()):
            raise ValueError(
                f"{observation.at!r} was found in the network but has no data for "
                f"quantity '{item}'. Choose a location that has this quantity, or a "
                "model result for a quantity this location has."
            )

        return self._as_node_result(node_id)

    def _extract_reach(self, observation: ReachObservation) -> NodeModelResult:
        # A reach observation matches any breakpoint along the reach, so long as
        # they agree. Which breakpoints those are is read off the dataset's own
        # coordinates; the network is consulted only to explain a failure.
        item = self.sel_items.values
        reach_id = observation.reach

        if reach_id not in self.network.reaches:
            raise ValueError(f"Reach {reach_id} not found in network.")

        on_reach = np.flatnonzero(self.data["reach"].values == reach_id)
        # A location that carries no data for this quantity is all-NaN here,
        # since quantities with different coverage are aligned on the way in.
        with_data = self.data[item].isel(node=on_reach).notnull().any("time").values
        candidates = self.data.isel(node=on_reach[np.flatnonzero(with_data)])

        if candidates.sizes["node"] == 0:
            raise ValueError(self._explain_no_reach_data(reach_id, item))

        values = candidates[item].transpose("time", "node").values
        if not np.allclose(values, values[:, :1], equal_nan=True):
            raise ValueError(
                "Not all data in breakpoints are equivalent. "
                "Select a specific node instead of the reach."
            )

        # Lowest distance first, unknown distances last, so the breakpoint chosen
        # does not depend on the numbering mikeio1d happened to hand out.
        distance = np.nan_to_num(candidates["distance"].values, nan=np.inf)
        order = np.lexsort((candidates["node"].values, distance))
        return self._as_node_result(int(candidates["node"].values[order[0]]))

    def _explain_no_reach_data(self, reach_id: str, item: str) -> str:
        # Whether the reach names the quantity at all is a distinction only the
        # network can make: its breakpoints say which columns they were read for,
        # where the dataset shows only that the values are missing.
        has_source_data = any(
            breakpoint.data is not None and item in breakpoint.data.columns
            for breakpoint in self.network.reaches[reach_id].breakpoints
        )
        if has_source_data:
            return (
                f"Reach '{reach_id}' has breakpoints that name quantity "
                f"'{item}', but none of them carry values for it in this model "
                "result. Re-create the NetworkModelResult with the relevant "
                "reaches populated."
            )
        return (
            f"Reach '{reach_id}' was found in the network but none of its "
            f"breakpoints have data loaded for quantity '{item}'. "
            f"Re-create the NetworkModelResult with the relevant reaches populated."
        )

    def _as_node_result(self, node_id: int) -> NodeModelResult:
        # The location is taken from the network rather than from the observation,
        # so a distance given as 24.5001 is recorded as the network's own 24.5.
        where = self.network.recall(int(node_id))
        location = (
            where["node"] if "node" in where else (where["reach"], where["distance"])
        )
        data = self.data.sel(node=node_id).drop_vars(
            ("node", *self._UPSTREAM_IDENTITY_COORDS), errors="ignore"
        )
        return NodeModelResult._from_network(
            data,
            location=location,
            node_index=int(node_id),
            name=self.name,
            item=self.sel_items.values,
            quantity=self.quantity,
            aux_items=self.sel_items.aux,
        )

    def _resolve_alias(self, alias: str | tuple[str, float]) -> int:
        # Delegated to Network.find rather than matched against the dataset's own
        # name/reach/distance coords: find() searches the whole topology, so a hit
        # that is missing from the dataset is a location whose timeseries was not
        # loaded, which is a different mistake from one that does not exist.
        try:
            if isinstance(alias, tuple):
                reach_id, distance = alias
                return int(self.network.find(reach=str(reach_id), distance=distance))
            return int(self.network.find(node=str(alias)))
        except KeyError as err:
            raise ValueError(f"Location {alias!r} not found. {err.args[0]}") from err
