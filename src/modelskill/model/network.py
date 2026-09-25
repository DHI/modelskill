from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from modelskill.timeseries import TimeSeries, _parse_point_input
from modelskill.timeseries._coords import NetworkCoords
from ._base import SelectedItems
from ..obs import NodeObservation, ReachObservation, _at_from_coords
from ..quantity import Quantity
from ..types import GeometryType

if TYPE_CHECKING:
    from mikeio1d.network import Address, Location, Network


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
        # Mark the kind on our own copy. The dataset handed in may belong to an
        # observation, and a shallow copy rebuilds each variable's attrs, so
        # reading it here does not turn that observation into a model result.
        data = data.copy()
        data_var = str(list(data.data_vars)[0])
        data[data_var].attrs["kind"] = "model"
        super().__init__(data=data)

    @property
    def at(self) -> str | tuple[str, float]:
        """Where this result was extracted: a node name, or a ``(reach_id, distance)`` breakpoint."""
        return _at_from_coords(self.data)

    @property
    def node(self) -> Any:
        """Name of the node this result was extracted at, or None for a break point."""
        return self._coordinate_values("node")

    def _location_repr(self) -> str | None:
        return f"Location: {self.at}"


class NetworkModelResult:
    """Model result for network data with time and node dimensions.

    Construct one from a result file, or from a :class:`mikeio1d.network.Network`
    already built. Observations name the location they sit at, and no spatial
    interpolation is performed.

    Nothing is read when the model result is built. :meth:`extract` reads the
    series of the one location an observation names, so matching a handful of
    sensors against a large model reads a handful of locations.

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

    Open the network yourself to name EPANET companion files:

    >>> from mikeio1d.network import Network
    >>> network = Network.open("model.res", companions=["model.resx", "model.inp"])
    >>> mr = ms.NetworkModelResult(network, item="Head", name="MyModel")

    Notes
    -----
    The network is used as given, not copied, so ``mr.network`` is the caller's
    object. It reads from its result file on every :meth:`extract`.

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

        # What the network can read somewhere, which it knows without reading.
        units = self.network.quantities
        sel_items = SelectedItems.parse(list(units), item=item, aux_items=aux_items)
        name = name or sel_items.values

        self.name = name
        self.sel_items = sel_items

        if quantity is None:
            # A result file names its quantity and, mostly, its unit. Where it
            # gives no unit the name is still worth keeping, so the unit is left
            # empty rather than the quantity undefined.
            quantity = Quantity(
                name=str(sel_items.values), unit=units[sel_items.values] or ""
            )
        self.quantity = quantity

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>: {self.name}"

    @property
    def period(self) -> tuple[datetime, datetime]:
        """First and last timestep of the result file.

        Read from the file header, so no timeseries is read.

        Returns
        -------
        tuple[datetime, datetime]
            Start and end of the result file's time axis.
        """
        return self.network.period

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
            found = self.network.resolve(observation.at)
            if found is None:
                raise ValueError(
                    f"Location {observation.at!r} not found in the network. "
                    "network.locations() lists the nodes and break points it has, "
                    "and locations(reach=...) the break points along one reach."
                )
            [df] = self._read([found])
            return self._to_result(found.address, df)
        elif isinstance(observation, ReachObservation):
            return self._extract_reach(observation)
        else:
            raise TypeError(
                f"NetworkModelResult supports NodeObservation and ReachObservation, got {type(observation).__name__}"
            )

    def _extract_reach(self, observation: ReachObservation) -> NodeModelResult:
        # A reach observation matches any breakpoint along the reach, so long as
        # they agree. The network says which breakpoints carry the quantity, and
        # only those are read, all in one read.
        item = self.sel_items.values
        reach_id = observation.reach

        # locations() raises a KeyError naming the reach if the network has none
        # such, and every point it gives resolves.
        found = [
            self.network.resolve(point)
            for point in self.network.locations(reach=reach_id, quantity=item)
        ]
        if not found:
            raise ValueError(
                f"Reach '{reach_id}' was found in the network but none of its "
                f"breakpoints carry quantity '{item}'. Choose a reach that has "
                "this quantity, or a model result for a quantity this reach has."
            )

        # A breakpoint can name a quantity and hold nothing for it.
        with_data = [
            (location, df)
            for location, df in zip(found, self._read(found))
            if df[item].notna().any()
        ]
        if not with_data:
            raise ValueError(
                f"Reach '{reach_id}' has breakpoints that name quantity "
                f"'{item}', but none of them carry values for it in this model "
                "result."
            )
        values = np.column_stack([df[item].to_numpy() for _, df in with_data])
        if not np.allclose(values, values[:, :1], equal_nan=True):
            raise ValueError(
                "Not all data in breakpoints are equivalent. "
                "Select a specific node instead of the reach."
            )

        # Lowest distance first, so the breakpoint chosen does not depend on the
        # order the network happened to list them in.
        location, df = min(with_data, key=lambda pair: pair[0].address[1])
        return self._to_result(location.address, df)

    def _read(self, found: Sequence[Location]) -> list[pd.DataFrame]:
        # One read for every location, each giving a frame of the selected items.
        # An auxiliary item a location does not carry is missing there, as it
        # would be anywhere else it is not measured.
        keys = [
            (n, q)
            for n, location in enumerate(found)
            for q in self.sel_items.all
            if q in location.quantities
        ]
        df = self.network.read([(found[n].address, q) for n, q in keys])
        df.columns = pd.MultiIndex.from_tuples(keys, names=[None, None])
        df = df.reindex(
            columns=pd.MultiIndex.from_product([range(len(found)), self.sel_items.all])
        ).rename_axis("time")
        return [df[n] for n in range(len(found))]

    def _to_result(self, address: Address, df: pd.DataFrame) -> NodeModelResult:
        # The address is the network's own spelling of it rather than the
        # observation's, so a distance given as 24.5001 is recorded as 24.5.
        item = self.sel_items.values
        # MIKE 1D stores quantities at different grid points, so a breakpoint
        # carrying Discharge may carry no WaterLevel; and a location can name a
        # quantity and hold nothing for it.
        if not df[item].notna().any():
            raise ValueError(
                f"{address!r} was found in the network but has no data for "
                f"quantity '{item}'. Choose a location that has this quantity, or a "
                "model result for a quantity this location has."
            )

        if isinstance(address, tuple):
            address = (str(address[0]), address[1])
        ds = _parse_point_input(
            xr.Dataset.from_dataframe(df),
            self.name,
            item,
            self.quantity,
            self.sel_items.aux,
            coords=NetworkCoords(address),
        )
        return NodeModelResult(ds)
