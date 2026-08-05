"""Opt-in network module for network model results (e.g. MIKE 1D / res1d).

Requires the ``networks`` dependency group (networkx, mikeio1d).
Install with::

    uv sync --group networks

Import this module explicitly to use network functionality::

    from modelskill.network import Network

"""

from __future__ import annotations

import sys

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Sequence, overload, TYPE_CHECKING
from copy import deepcopy

import networkx as nx
import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    from mikeio1d import Res1D
    from mikeio1d.result_network import ResultReach
    from .model.adapters._res1d import Res1DReach


_MIKE_EXTENSIONS = frozenset({".res1d", ".res11"})
_EPANET_EXTENSIONS = frozenset({".res"})

_NO_FIXTURE = (
    "{product} results are not supported yet: modelskill has no test fixture for "
    "this format, so support cannot be verified. Please open an issue if you need it."
)
# A result file that holds timeseries but no topology of its own. The connectivity
# is in a companion file we do not parse yet.
_TOPOLOGY_IN_COMPANION_FILE = (
    "SWMM '.out' files carry no reach connectivity of their own - it lives in the "
    "companion '.inp' input file, which modelskill does not read yet. Tracked in "
    "https://github.com/DHI/modelskill/issues/689."
)
# A companion result file: readable, but it describes a network defined elsewhere.
_COMPANION_RESULT_FILE = (
    "'.resx' holds extra EPANET results (tank volume, pump energy) for a network "
    "defined in the sibling '.res' file, so it has no topology of its own. Read the "
    "'.res' file and pass this one alongside it: "
    "Network.from_epanet(res, resx=...)."
)

# extension -> why modelskill will not read it, even though mikeio1d can
_UNSUPPORTED_EXTENSIONS: dict[str, str] = {
    ".out": _TOPOLOGY_IN_COMPANION_FILE,
    ".resx": _COMPANION_RESULT_FILE,
    ".prf": _NO_FIXTURE.format(product="MOUSE"),
    ".crf": _NO_FIXTURE.format(product="MOUSE"),
    ".xrf": _NO_FIXTURE.format(product="MOUSE"),
    ".whr": _NO_FIXTURE.format(product="Water Hammer"),
}

# extension -> the constructor that reads it, for "use X instead" errors
_EXTENSION_CONSTRUCTORS: dict[str, str] = {
    **{extension: "from_mike" for extension in _MIKE_EXTENSIONS},
    **{extension: "from_epanet" for extension in _EPANET_EXTENSIONS},
}


def _check_file_path_is_str(res: Res1D) -> None:
    """Reject a Res1D opened with a path object rather than a string.

    mikeio1d resolves reach topology with ``str.endswith`` on
    ``Res1D.file_path``, which raises ``AttributeError`` from deep inside the
    load when that attribute is a ``Path``. Fail here instead, where the cause
    can be named.
    """
    file_path = getattr(res, "file_path", None)
    if file_path is not None and not isinstance(file_path, str):
        raise TypeError(
            f"This Res1D was opened with a {type(file_path).__name__} file_path, "
            "which mikeio1d cannot resolve reach topology from. Re-open it as "
            "Res1D(str(path)), or pass the path to the constructor directly."
        )


class NetworkNode(ABC):
    """Abstract base class for a node in a network.

    A node represents a discrete location in the network (e.g. a junction,
    reservoir, or boundary point) that carries time-series data for one or
    more physical quantities.

    Three properties must be implemented:

    * :attr:`id` - a unique string identifier for the node.
    * :attr:`data` - a time-indexed :class:`pandas.DataFrame` whose columns
      are quantity names.
    * :attr:`boundary` - a dict of boundary-condition metadata (may be empty).

    The concrete helper :class:`BasicNode` is provided for the common case
    where the data is already available as a DataFrame.

    See Also
    --------
    BasicNode : Ready-to-use concrete implementation.
    NetworkReach : Connects two NetworkNode instances.
    Network : Container that assembles nodes and reaches into a graph.
    """

    @property
    @abstractmethod
    def id(self) -> str:
        """Unique string identifier for this node."""
        pass

    @property
    @abstractmethod
    def data(self) -> pd.DataFrame:
        """Time-indexed DataFrame with one column per quantity."""
        pass

    @property
    @abstractmethod
    def boundary(self) -> dict[str, Any]:
        """Boundary-condition metadata dict (may be empty)."""
        pass

    @property
    def quantities(self) -> list[str]:
        """List of quantity names available at this node."""
        return list(self.data.columns)


class ReachBreakPoint(ABC):
    """Abstract base class for an intermediate break point along a network reach.

    Break points represent locations between the start and end nodes of a
    reach (e.g. cross-section chainage points along a river reach) that carry
    their own time-series data.

    Two properties must be implemented:

    * :attr:`id` - a ``(reach_id, distance)`` tuple that uniquely locates the
      break point within the network.
    * :attr:`data` - a time-indexed :class:`pandas.DataFrame` whose columns
      are quantity names.

    The :attr:`distance` convenience property returns ``id[1]`` (the
    along-reach distance in the units used by the parent network).

    Examples
    --------
    Minimal subclass:

    >>> class MyBreakPoint(ReachBreakPoint):
    ...     def __init__(self, reach_id, chainage, df):
    ...         self._id = (reach_id, chainage)
    ...         self._data = df
    ...     @property
    ...     def id(self): return self._id
    ...     @property
    ...     def data(self): return self._data

    See Also
    --------
    NetworkReach : Owns a list of ReachBreakPoint instances.
    NetworkNode : Represents a start/end node of a reach.
    Network : Assembles reaches (and their break points) into a graph.
    """

    @property
    @abstractmethod
    def id(self) -> tuple[str, float]:
        """``(reach_id, distance)`` tuple uniquely identifying this break point."""
        pass

    @property
    @abstractmethod
    def data(self) -> pd.DataFrame:
        """Time-indexed DataFrame with one column per quantity."""
        pass

    @property
    def distance(self) -> float:
        """Along-reach distance of this break point, measured from the start node."""
        return self.id[1]

    @property
    def quantities(self) -> list[str]:
        """List of quantity names available at this break point."""
        return list(self.data.columns)


class NetworkReach(ABC):
    """Abstract base class for a reach in a network.

    A reach represents a directed connection between two :class:`NetworkNode`
    instances (e.g. a river reach between two junctions).  It may also carry
    a list of :class:`ReachBreakPoint` objects for intermediate chainage
    locations.

    Subclass this to integrate your own network topology.  Four properties
    must be implemented:

    * :attr:`id` - a unique string identifier for the reach.
    * :attr:`start` - the upstream/start :class:`NetworkNode`.
    * :attr:`end` - the downstream/end :class:`NetworkNode`.
    * :attr:`breakpoints` - list of :class:`ReachBreakPoint` instances ordered
      by increasing distance from the start node (empty list if none).

    :attr:`length` is optional and defaults to ``None``. Reach length matters
    in some domains (rivers, sewer networks) and not in others (link-node water
    distribution models), so override it only where a length exists.

    The concrete helper :class:`BasicReach` is provided for the common case
    where all data is already available in memory.

    Examples
    --------
    Minimal subclass, without a length:

    >>> class MyReach(NetworkReach):
    ...     def __init__(self, rid, start_node, end_node):
    ...         self._id = rid
    ...         self._start = start_node
    ...         self._end = end_node
    ...     @property
    ...     def id(self): return self._id
    ...     @property
    ...     def start(self): return self._start
    ...     @property
    ...     def end(self): return self._end
    ...     @property
    ...     def breakpoints(self): return []

    Add a :attr:`length` property on top of that when the domain has one:

    >>> class MyMeasuredReach(MyReach):
    ...     def __init__(self, rid, start_node, end_node, length):
    ...         super().__init__(rid, start_node, end_node)
    ...         self._length = length
    ...     @property
    ...     def length(self): return self._length

    See Also
    --------
    BasicReach : Ready-to-use concrete implementation.
    NetworkNode : Represents the start/end of this reach.
    ReachBreakPoint : Intermediate data points along this reach.
    Network : Assembles a list of NetworkReach objects into a graph.
    """

    @property
    @abstractmethod
    def id(self) -> str:
        """Unique string identifier for this reach."""
        pass

    @property
    @abstractmethod
    def start(self) -> NetworkNode:
        """Start (upstream) node of this reach."""
        pass

    @property
    @abstractmethod
    def end(self) -> NetworkNode:
        """End (downstream) node of this reach."""
        pass

    @property
    def length(self) -> float | None:
        """Total length of this reach in network units, or ``None`` if undefined."""
        return None

    @property
    @abstractmethod
    def breakpoints(self) -> list[ReachBreakPoint]:
        """Ordered list of intermediate :class:`ReachBreakPoint` objects (may be empty)."""
        pass

    @property
    def n_breakpoints(self) -> int:
        """Number of break points in the reach."""
        return len(self.breakpoints)


class BasicNode(NetworkNode):
    """Concrete :class:`NetworkNode` for programmatic network construction.

    Parameters
    ----------
    id : str
        Unique node identifier.
    data : pd.DataFrame
        Time-indexed DataFrame with one column per quantity.
    boundary : dict, optional
        Boundary condition metadata, by default empty.

    Examples
    --------
    >>> import pandas as pd
    >>> time = pd.date_range("2020", periods=3, freq="h")
    >>> node = BasicNode("junction_1", pd.DataFrame({"WaterLevel": [1.0, 1.1, 1.2]}, index=time))
    """

    def __init__(
        self,
        id: str,
        data: pd.DataFrame,
        boundary: dict[str, Any] | None = None,
    ) -> None:
        self._id = id
        self._data = data
        self._boundary: dict[str, Any] = boundary or {}

    @property
    def id(self) -> str:
        return self._id

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @property
    def boundary(self) -> dict[str, Any]:
        return self._boundary


class BasicReach(NetworkReach):
    """Concrete :class:`NetworkReach` for programmatic network construction.

    Parameters
    ----------
    id : str
        Unique reach identifier.
    start : NetworkNode
        Start node.
    end : NetworkNode
        End node.
    length : float, optional
        Reach length, by default None (undefined).
    breakpoints : list[ReachBreakPoint], optional
        Intermediate break points, by default empty.

    Examples
    --------
    >>> reach = BasicReach("reach_1", node_a, node_b, length=250.0)

    Where the domain has no reach length, leave it out:

    >>> reach = BasicReach("pipe_1", node_a, node_b)
    """

    def __init__(
        self,
        id: str,
        start: NetworkNode,
        end: NetworkNode,
        length: float | None = None,
        breakpoints: list[ReachBreakPoint] | None = None,
    ) -> None:
        self._id = id
        self._start = start
        self._end = end
        self._length = length
        self._breakpoints: list[ReachBreakPoint] = breakpoints or []

    @property
    def id(self) -> str:
        return self._id

    @property
    def start(self) -> NetworkNode:
        return self._start

    @property
    def end(self) -> NetworkNode:
        return self._end

    @property
    def length(self) -> float | None:
        return self._length

    @property
    def breakpoints(self) -> list[ReachBreakPoint]:
        return self._breakpoints


class Network:
    """Network built from a set of reaches, with coordinate lookup and data access."""

    def __init__(self, reaches: Sequence[NetworkReach]):
        graph = self._generate_graph(reaches)
        self._initialize_network_attributes(graph)
        self._reaches = self._generate_reaches_dict(reaches)

    def _initialize_network_attributes(self, graph: nx.Graph):
        self._alias_map = self._generate_alias_map(graph)
        self._df = self._build_dataframe(graph)
        self._graph = graph.copy()

    def __repr__(self) -> str:
        time = self._df.index
        time_window = "N/A - N/A" if len(time) == 0 else f"{time[0]} - {time[-1]}"
        out = [
            "<Network>",
            f"Reaches: {len(self._reaches)}",
            f"Nodes: {self._graph.number_of_nodes()}",
            f"Quantities: {self.quantities}",
            f"Time: {time_window}",
        ]
        return "\n".join(out)

    @classmethod
    def from_mike(
        cls,
        res: str | Path | Res1D,
        *,
        nodes: str | list[str] | None = None,
        reaches: str | list[str] | None = None,
    ) -> Network:
        """Create a Network from a MIKE 1D or MIKE 11 result file.

        Parameters
        ----------
        res : str, Path or Res1D
            Path to a ``.res1d`` or ``.res11`` file, or an already-opened
            :class:`mikeio1d.Res1D` object.
        nodes : str, list of str, or None, optional
            Controls which nodes have their timeseries data loaded into memory.

            * ``None`` *(default)* — data is loaded for every node.
            * A single node ID or a list of node IDs — only those nodes get
              data; others are topology-only.
            * ``[]`` (empty list) — no node data is loaded at all.

            The full network topology is always constructed regardless of this
            setting, so ``find()`` and ``recall()`` still work on all nodes.
        reaches : str, list of str, or None, optional
            Controls which reaches have their intermediate gridpoint data
            populated.

            * ``None`` *(default)* — gridpoints are populated for every reach.
            * A single reach name or a list of reach names — only those reaches
              get gridpoint data; others are topology-only.
            * ``[]`` (empty list) — no gridpoint data is loaded at all.

        Returns
        -------
        Network

        Raises
        ------
        NotImplementedError
            If the file extension is not one modelskill can read.
        ValueError
            If the extension belongs to another constructor, such as EPANET.

        Examples
        --------
        Load everything (default behaviour):

        >>> from modelskill.network import Network
        >>> network = Network.from_mike("model.res1d")

        Load data only for the two nodes where observations exist, and skip
        all intermediate gridpoint data to keep memory usage low:

        >>> network = Network.from_mike(
        ...     "model.res1d",
        ...     nodes=["node_a", "node_b"],
        ...     reaches=[],
        ... )

        Load data for selected nodes and gridpoints for one specific reach:

        >>> network = Network.from_mike(
        ...     "model.res1d",
        ...     nodes=["node_a", "node_b"],
        ...     reaches=["reach_1"],
        ... )

        Notes
        -----
        MIKE 11 keeps its timeseries on reach gridpoints rather than on nodes,
        so the nodes of a ``.res11`` network carry no data of their own. Pass
        ``reaches`` rather than ``nodes`` to control what gets loaded.

        See Also
        --------
        from_epanet : Read an EPANET result file.
        """
        return cls._from_mikeio1d(
            res,
            nodes=nodes,
            reaches=reaches,
            allowed=_MIKE_EXTENSIONS,
            caller="from_mike",
        )

    @classmethod
    def from_epanet(
        cls,
        res: str | Path | Res1D,
        *,
        resx: str | Path | Res1D | None = None,
        inp: str | Path | None = None,
        nodes: str | list[str] | None = None,
        reaches: str | list[str] | None = None,
    ) -> Network:
        """Create a Network from an EPANET result file and its companions.

        An EPANET run writes up to three files that modelskill can use. The
        ``.res`` holds the network and its main timeseries; the optional
        ``.resx`` holds extra results; and the optional ``.inp`` is the input
        file, which is the only one of the three carrying reach lengths.

        Parameters
        ----------
        res : str, Path or Res1D
            Path to a ``.res`` file, or an already-opened
            :class:`mikeio1d.Res1D` object.
        resx : str, Path, Res1D or None, optional
            Companion ``.resx`` file from the same run. Its extra node
            quantities (tank ``Volume`` and ``Volume Percentage``) are merged
            onto the matching nodes. By default None, and those quantities are
            simply absent.
        inp : str, Path or None, optional
            EPANET ``.inp`` input file for the same model, read for its
            ``[PIPES]`` lengths. By default None, and reach lengths are
            undefined.
        nodes : str, list of str, or None, optional
            Which nodes get their timeseries loaded. See :meth:`from_mike`.
        reaches : str, list of str, or None, optional
            Which reaches get their gridpoint data loaded. See
            :meth:`from_mike`. EPANET results have no intermediate gridpoints,
            so this argument has no effect.

        Returns
        -------
        Network

        Raises
        ------
        NotImplementedError
            If the file extension is not one modelskill can read.
        ValueError
            If the extension belongs to another constructor, such as MIKE, if a
            companion file has the wrong extension, or if ``resx`` does not come
            from the same run as ``res``.

        Examples
        --------
        >>> from modelskill.network import Network
        >>> network = Network.from_epanet("model.res")

        With both companions, for real edge lengths and the extra quantities:

        >>> network = Network.from_epanet(
        ...     "model.res",
        ...     resx="model.resx",
        ...     inp="model.inp",
        ... )

        Notes
        -----
        EPANET is a link-node model, and mikeio1d reports no length and a
        single synthetic gridpoint for each of its reaches. As a result:

        * without ``inp``, every edge of :attr:`graph` has ``length=None``, so a
          length-weighted graph algorithm fails rather than returning a
          meaningless number. Pumps and valves keep ``length=None`` even with
          ``inp``, since ``[PIPES]`` is the only section carrying lengths
        * reaches have no breakpoints, so
          :class:`~modelskill.obs.ReachObservation` cannot be matched against
          an EPANET network — use :class:`~modelskill.obs.NodeObservation`
        * ``find(reach=..., distance=<number>)`` never resolves; only
          ``distance="start"`` and ``distance="end"`` work

        For the same reason, ``resx`` merges node quantities only. Its
        reach-level quantities (pump energy, efficiency and costs) have no
        breakpoint to live on, which is tracked in issue #680.

        Node timeseries, :meth:`to_dataframe`, :meth:`to_dataset`,
        ``find(node=...)`` and :meth:`recall` are unaffected.

        See Also
        --------
        from_mike : Read a MIKE 1D or MIKE 11 result file.
        """
        return cls._from_mikeio1d(
            res,
            nodes=nodes,
            reaches=reaches,
            allowed=_EPANET_EXTENSIONS,
            caller="from_epanet",
            resx=resx,
            inp=inp,
        )

    @classmethod
    def _from_mikeio1d(
        cls,
        res: str | Path | Res1D,
        *,
        nodes: str | list[str] | None,
        reaches: str | list[str] | None,
        allowed: frozenset[str],
        caller: str,
        resx: str | Path | Res1D | None = None,
        inp: str | Path | None = None,
    ) -> Network:
        """Shared implementation behind the public ``from_*`` constructors.

        Parameters
        ----------
        allowed : frozenset of str
            Extensions this constructor accepts.
        caller : str
            Name of the public method, used in error messages.
        resx : str, Path, Res1D or None, optional
            Companion result file whose node quantities are merged in.
        inp : str, Path or None, optional
            Companion input file read for reach lengths.
        """
        if sys.version_info >= (3, 14):
            raise NotImplementedError(
                f"Current version of 'mikeio1d' requires python < 3.14 and {sys.version} is being used."
            )

        from mikeio1d import Res1D as _Res1D

        if isinstance(res, (str, Path)):
            path = Path(res)
            cls._validate_extension(path.suffix, allowed=allowed, caller=caller)
            res = _Res1D(str(path))
        elif isinstance(res, _Res1D):
            _check_file_path_is_str(res)
            suffix = Path(res.file_path).suffix
            cls._validate_extension(suffix, allowed=allowed, caller=caller)
        else:
            raise TypeError(
                f"Expected a str, Path or Res1D object, got {type(res).__name__!r}"
            )

        if nodes is None:
            nodes_list: list[str] = list(res.nodes.keys())
        elif isinstance(nodes, str):
            nodes_list = [nodes]
        else:
            nodes_list = list(nodes)

        if reaches is None:
            reaches_list: list[str] = list(res.reaches.keys())
        elif isinstance(reaches, str):
            reaches_list = [reaches]
        else:
            reaches_list = list(reaches)

        extra = None if resx is None else cls._open_companion_result(res, resx)
        lengths = None if inp is None else cls._read_companion_lengths(inp)

        list_of_reaches = cls._load_res1d_network(
            res, nodes_list, reaches_list, extra=extra, lengths=lengths
        )
        return cls(list_of_reaches)

    @staticmethod
    def _read_companion_lengths(inp: str | Path) -> dict[str, float]:
        """Read reach lengths from a companion ``.inp`` input file."""
        from modelskill.model.adapters._inp import read_pipe_lengths

        path = Path(inp)
        if path.suffix.lower() != ".inp":
            raise ValueError(
                f"Expected an EPANET '.inp' input file, got '{path.suffix}'. "
                "This argument reads reach lengths from the model input, not "
                "from a result file."
            )
        return read_pipe_lengths(path)

    @staticmethod
    def _open_companion_result(res: Res1D, resx: str | Path | Res1D) -> Res1D:
        """Open and validate a companion ``.resx`` result file.

        Raises
        ------
        ValueError
            If the extension is not ``.resx``, or if the file does not come from
            the same run as ``res``.
        """
        from mikeio1d import Res1D as _Res1D

        if isinstance(resx, (str, Path)):
            path = Path(resx)
            if path.suffix.lower() != ".resx":
                raise ValueError(
                    f"Expected an EPANET '.resx' companion file, got '{path.suffix}'."
                )
            extra = _Res1D(str(path))
        elif isinstance(resx, _Res1D):
            _check_file_path_is_str(resx)
            if Path(resx.file_path).suffix.lower() != ".resx":
                raise ValueError(
                    "Expected an EPANET '.resx' companion file, got "
                    f"'{Path(resx.file_path).suffix}'."
                )
            extra = resx
        else:
            raise TypeError(
                f"Expected a str, Path or Res1D object, got {type(resx).__name__!r}"
            )

        # Merging two different runs would line up silently and produce a network
        # that is wrong in a way no later error would reveal.
        if not res.time_index.equals(extra.time_index):
            raise ValueError(
                "The '.resx' companion does not share a time axis with the "
                "'.res' file, so the two are not from the same run. Got "
                f"{len(extra.time_index)} steps ending {extra.end_time} against "
                f"{len(res.time_index)} ending {res.end_time}."
            )

        unknown_nodes = set(extra.nodes) - set(res.nodes)
        if unknown_nodes:
            raise ValueError(
                f"The '.resx' companion holds nodes {sorted(unknown_nodes)} that are "
                "absent from the '.res' network, so the two files do not describe "
                "the same model."
            )

        unknown_reaches = set(extra.reaches) - set(res.reaches)
        if unknown_reaches:
            raise ValueError(
                f"The '.resx' companion holds reaches {sorted(unknown_reaches)} that are "
                "absent from the '.res' network, so the two files do not describe "
                "the same model."
            )

        return extra

    @staticmethod
    def _validate_extension(
        suffix: str, *, allowed: frozenset[str], caller: str
    ) -> None:
        """Check a file extension against mikeio1d and against one constructor.

        Raises
        ------
        NotImplementedError
            If modelskill cannot read the extension, either because mikeio1d
            does not support it or because modelskill does not.
        ValueError
            If another constructor is the one that reads this extension.
        """
        from mikeio1d import Res1D as _Res1D

        extension = suffix.lower()

        # Checked before the supported set below, since these all *are* readable
        # by mikeio1d - it is modelskill that cannot use the result.
        reason = _UNSUPPORTED_EXTENSIONS.get(extension)
        if reason is not None:
            raise NotImplementedError(f"Cannot read '{suffix}' files. {reason}")

        supported = _Res1D.get_supported_file_extensions()
        if extension not in supported:
            readable = sorted(supported - set(_UNSUPPORTED_EXTENSIONS))
            raise NotImplementedError(
                f"Unsupported file extension '{suffix}'. "
                f"Supported extensions are {readable}."
            )

        if extension not in allowed:
            constructor = _EXTENSION_CONSTRUCTORS.get(extension)
            if constructor is None:
                raise NotImplementedError(
                    f"File extension '{suffix}' is supported by mikeio1d but is not mapped "
                    "to a Network constructor in this version of modelskill. "
                    "Please upgrade modelskill or open an issue."
                )
            raise ValueError(
                f"Network.{caller}() reads {sorted(allowed)} files, got '{suffix}'. "
                f"Use Network.{constructor}() instead."
            )

    @staticmethod
    def _load_res1d_network(
        res: Res1D,
        nodes: list[str],
        reaches: list[str],
        *,
        extra: Res1D | None = None,
        lengths: dict[str, float] | None = None,
    ) -> list[Res1DReach]:
        from modelskill.model.adapters._res1d import (
            Res1DReach,
            Res1DNode,
            _merge_extra_quantities,
            _simplify_colnames,
        )

        nodes_set = set(nodes)
        reaches_set = set(reaches)
        lengths = lengths or {}

        # In order to work with bigger files, we might want to select a subset of nodes and avoid
        # potential memory issues. For this reason, we create this intermediate step that populates
        # only the data in the passed nodes

        def _init_node(reach: ResultReach, is_end: bool) -> Res1DNode:
            id = reach.end_node if is_end else reach.start_node
            gpt_idx = -1 if is_end else 0
            if id in nodes_set:
                node = res.nodes[id]
                df = _simplify_colnames(node)
                # Merged here rather than up front so selective loading still
                # decides what is held in memory.
                if extra is not None and id in extra.nodes:
                    df = _merge_extra_quantities(
                        df, _simplify_colnames(extra.nodes[id]), node_id=id
                    )
                overlapping_gridpoint = reach.gridpoints[gpt_idx]
                boundary = _simplify_colnames(overlapping_gridpoint)
                return Res1DNode(id, data=df, boundary={reach.name: boundary})
            else:
                return Res1DNode(id)

        return [
            Res1DReach(
                reach,
                _init_node(reach, False),
                _init_node(reach, True),
                populate_gridpoints=reach.name in reaches_set,
                length=lengths.get(reach.name),
            )
            for reach in res.reaches.values()
        ]

    @staticmethod
    def _generate_alias_map(g: nx.Graph) -> dict[str | tuple[str, float], int]:
        return {g.nodes[id]["alias"]: id for id in g.nodes()}

    @staticmethod
    def _generate_reaches_dict(
        reaches: Sequence[NetworkReach],
    ) -> dict[str, NetworkReach]:
        return {r.id: r for r in reaches}

    @staticmethod
    def _build_dataframe(g: nx.Graph) -> pd.DataFrame:
        data_in_nodes = {
            k: v["data"] for k, v in g.nodes.items() if v["data"] is not None
        }
        if len(data_in_nodes) == 0:
            columns = pd.MultiIndex.from_arrays([[], []], names=["node", "quantity"])
            return pd.DataFrame(index=pd.Index([], name="time"), columns=columns)
        df = pd.concat(data_in_nodes, axis=1)
        df.columns = df.columns.set_names(["node", "quantity"])
        df.index.name = "time"
        return df.copy()

    def to_dataframe(self, sel: str | None = None) -> pd.DataFrame:
        """Dataframe using node ids as column names.

        It will be multiindex unless 'sel' is passed.

        Parameters
        ----------
        sel : Optional[str], optional
            Quantity to select, by default None

        Returns
        -------
        pd.DataFrame
            Timeseries contained in graph nodes
        """
        df = self._df.copy()
        if sel is None:
            return df
        else:
            df.attrs["quantity"] = sel
            return df.reorder_levels(["quantity", "node"], axis=1).loc[:, sel]

    def to_dataset(self) -> xr.Dataset:
        """Dataset using node ids as coords.

        Returns
        -------
        xr.Dataset
            Timeseries contained in graph nodes
        """
        df_raw = self.to_dataframe()
        if len(df_raw.columns) == 0:
            return xr.Dataset()
        df = df_raw.reorder_levels(["quantity", "node"], axis=1)
        quantities = df.columns.get_level_values("quantity").unique()
        return xr.Dataset(
            {q: xr.DataArray(df[q], dims=["time", "node"]) for q in quantities}
        )

    @property
    def graph(self) -> nx.Graph:
        """Graph of the network."""
        return self._graph

    @property
    def quantities(self) -> list[str]:
        """Quantities present in data.

        Returns
        -------
        List[str]
            List of quantities
        """
        return list(self.to_dataframe().columns.get_level_values(1).unique())

    @staticmethod
    def _generate_graph(reaches: Sequence[NetworkReach]) -> nx.Graph:
        g0 = nx.Graph()
        for reach in reaches:
            # 1) Add start and end nodes
            for node in [reach.start, reach.end]:
                node_key = node.id
                if node_key in g0.nodes:
                    g0.nodes[node_key]["boundary"].update(node.boundary)
                else:
                    g0.add_node(node_key, data=node.data, boundary=node.boundary)

            # 2) Add edges connecting start/end nodes to their adjacent breakpoints
            start_key = reach.start.id
            end_key = reach.end.id
            if reach.n_breakpoints == 0:
                g0.add_edge(start_key, end_key, length=reach.length)
            else:
                bp_keys = [bp.id for bp in reach.breakpoints]
                for bp, bp_key in zip(reach.breakpoints, bp_keys):
                    g0.add_node(bp_key, data=bp.data)

                g0.add_edge(start_key, bp_keys[0], length=reach.breakpoints[0].distance)

                # Only the final segment needs the total length. Break point
                # distances are known even when the total is not, so a reach
                # without a length still gets real lengths on every edge but
                # this one.
                tail_length = (
                    None
                    if reach.length is None
                    else reach.length - reach.breakpoints[-1].distance
                )
                g0.add_edge(bp_keys[-1], end_key, length=tail_length)

            # 3) Connect consecutive intermediate breakpoints
            for i in range(reach.n_breakpoints - 1):
                current_ = reach.breakpoints[i]
                next_ = reach.breakpoints[i + 1]
                length = next_.distance - current_.distance
                g0.add_edge(
                    current_.id,
                    next_.id,
                    length=length,
                )

        return nx.convert_node_labels_to_integers(g0, label_attribute="alias")

    @overload
    def find(
        self,
        *,
        node: str,
        reach: None = None,
        distance: None = None,
    ) -> int:
        pass

    @overload
    def find(
        self,
        *,
        node: list[str],
        reach: None = None,
        distance: None = None,
    ) -> list[int]:
        pass

    @overload
    def find(
        self,
        *,
        node: None = None,
        reach: str | list[str],
        distance: str | float,
    ) -> int:
        pass

    @overload
    def find(
        self,
        *,
        node: None = None,
        reach: str | list[str],
        distance: list[str | float],
    ) -> list[int]:
        pass

    def find(
        self,
        node: str | list[str] | None = None,
        reach: str | list[str] | None = None,
        distance: str | float | list[str | float] | None = None,
    ) -> int | list[int]:
        """Find node or breakpoint id in the Network object based on former coordinates.

        Parameters
        ----------
        node : str | List[str], optional
            Node id(s) in the original network, by default None
        reach : str | List[str], optional
            Reach id(s) for breakpoint lookup or reach endpoint lookup, by default None
        distance : str | float | List[str | float], optional
            Distance(s) along reach for breakpoint lookup, or "start"/"end"
            for reach endpoints, by default None

        Returns
        -------
        int | List[int]
            Node or breakpoint id(s) in the generic network

        Raises
        ------
        ValueError
            If invalid combination of parameters is provided
        KeyError
            If requested node/breakpoint is not found in the network
        """
        by_node = node is not None
        by_breakpoint = reach is not None or distance is not None

        if by_node and by_breakpoint:
            raise ValueError(
                "Cannot specify both 'node' and 'reach'/'distance' parameters simultaneously"
            )

        if not by_node and not by_breakpoint:
            raise ValueError(
                "Must specify either 'node' or both 'reach' and 'distance' parameters"
            )

        ids: list[str | tuple[str, float]]

        if by_node:
            assert node is not None
            if not isinstance(node, list):
                node = [node]
            ids = list(node)

        else:
            if reach is None or distance is None:
                raise ValueError(
                    "Both 'reach' and 'distance' parameters are required for breakpoint/endpoint lookup"
                )

            if not isinstance(reach, list):
                reach = [reach]

            if not isinstance(distance, list):
                distance = [distance]

            if len(reach) == 1:
                reach = reach * len(distance)

            if len(reach) != len(distance):
                raise ValueError(
                    "Incompatible lengths of 'reach' and 'distance' arguments. One 'reach' admits multiple distances, otherwise they must be the same length."
                )

            ids = []
            for reach_i, distance_i in zip(reach, distance):
                if distance_i in ["start", "end"]:
                    if reach_i not in self._reaches:
                        raise KeyError(f"Reach '{reach_i}' not found in the network.")

                    network_reach = self._reaches[reach_i]
                    if distance_i == "start":
                        ids.append(network_reach.start.id)
                    else:
                        ids.append(network_reach.end.id)
                else:
                    if not isinstance(distance_i, (int, float)):
                        raise ValueError(
                            "Invalid 'distance' value for breakpoint lookup: "
                            f"{distance_i!r}. Expected a numeric value or 'start'/'end'."
                        )
                    ids.append((reach_i, distance_i))

        _CHAINAGE_TOLERANCE = 1e-3

        def _resolve_id(id):
            if id in self._alias_map:
                return self._alias_map[id]
            if isinstance(id, tuple):
                reach_id, distance = id
                for key, val in self._alias_map.items():
                    if (
                        isinstance(key, tuple)
                        and key[0] == reach_id
                        and abs(key[1] - distance) <= _CHAINAGE_TOLERANCE
                    ):
                        return val
            return None

        resolved = [_resolve_id(id) for id in ids]
        missing_ids = [ids[i] for i, v in enumerate(resolved) if v is None]
        if missing_ids:
            raise KeyError(
                f"Node/breakpoint(s) {missing_ids} not found in the network. Available nodes are {set(self._alias_map.keys())}"
            )
        if len(resolved) == 1:
            return resolved[0]
        return resolved

    @overload
    def recall(self, id: int) -> dict[str, Any]:
        pass

    @overload
    def recall(self, id: list[int]) -> list[dict[str, Any]]:
        pass

    def recall(self, id: int | list[int]) -> dict[str, Any] | list[dict[str, Any]]:
        """Recover the original coordinates of an element given the node id(s) in the Network object.

        Parameters
        ----------
        id : int | List[int]
            Node id(s) in the generic network

        Returns
        -------
        Dict[str, Any] | List[Dict[str, Any]]
            Original coordinates. For single input returns dict, for multiple inputs returns list of dicts.
            Dict contains coordinates:
            - For nodes: 'node' key with node id
            - For breakpoints: 'reach' and 'distance' keys with reach id and distance

        Raises
        ------
        KeyError
            If node id is not found in the network
        ValueError
            If node id string format is invalid
        """
        if not isinstance(id, list):
            id = [id]

        reverse_alias_map = {v: k for k, v in self._alias_map.items()}

        results: list[dict[str, Any]] = []
        for node_id in id:
            if node_id not in reverse_alias_map:
                raise KeyError(f"Node ID {node_id} not found in the network.")

            key = reverse_alias_map[node_id]
            if isinstance(key, str):
                results.append({"node": key})
            else:
                results.append({"reach": key[0], "distance": key[1]})

        if len(results) == 1:
            return results[0]
        else:
            return results

    def copy(self) -> "Network":
        """Create a deep copy of the Network.

        Returns
        -------
        Network
            Deep copy of the Network object
        """
        return deepcopy(self)


def _make_basic_network(node_ids, time, data, quantity="WaterLevel"):
    nodes = [
        BasicNode(nid, pd.DataFrame({quantity: data[:, i]}, index=time))
        for i, nid in enumerate(node_ids)
    ]
    reaches = [
        BasicReach(f"r{i}", nodes[i], nodes[i + 1], length=100.0)
        for i in range(len(nodes) - 1)
    ]
    return Network(reaches)
