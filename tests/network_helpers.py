"""Helpers shared by the tests that score a network model result.

A Network comes from a result file and from nothing else, so these open the test
data rather than building a topology by hand. Importing this module needs
mikeio1d, which is an optional dependency (ADR-010), so guard the import with
``pytest.importorskip("mikeio1d.network")`` first.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from mikeio1d import Res1D
from mikeio1d.network import Network

_TESTDATA = Path(__file__).parent / "testdata"

RES1D = str(_TESTDATA / "network.res1d")
"""A MIKE urban result: WaterLevel on every node, Discharge on one gridpoint per reach."""

EPANET = str(_TESTDATA / "epanet.res")
"""A link-node result, read with the .inp and .resx beside it.

Its nodes each carry several quantities, which no node of RES1D does.
"""

NODE_IDS = ["1", "2", "3"]
"""Three nodes of RES1D, each carrying a WaterLevel series of its own."""

EPANET_NODES = ["11", "12"]
"""Two nodes of EPANET. Not '10', which is also the id of a reach there."""

REACH = "100l1"
REACH_ITEM = "Discharge"
DISTANCE = 23.8413574216414
"""Where REACH keeps its one REACH_ITEM gridpoint; its neighbours carry WaterLevel."""

BREAKPOINT = (REACH, DISTANCE)


def open_network(path: str = RES1D, **kwargs) -> Network:
    """The test data as a Network."""
    return Network.open(path, **kwargs)


def modified_network(
    tmp_path, *, offset: float = 0.0, blank: bool = False, path: str = RES1D, **kwargs
) -> Network:
    """A copy of `path` with every value shifted by `offset`, or blanked to NaN.

    Two model results have to differ before a crossed model column can show, and
    a break point that names a quantity while holding nothing for it is a state
    no fixture file is in. Both are written out rather than assembled in memory,
    since a Network comes from a file and from nothing else.
    """
    out = Path(tmp_path) / "modified.res1d"
    res = Res1D(path)
    values = res.read(column_mode="all")
    res.modify(values * np.nan if blank else values + offset)
    res.save(str(out))
    return Network.open(str(out), **kwargs)


def node_series(network, quantity="WaterLevel", nodes=None) -> pd.DataFrame:
    """Each node's own series for `quantity`, keyed by node id, read off the reaches.

    Restricted to `nodes` when given, since a result file holds more of them than
    a test wants to reason about, and the order is the one asked for.
    """
    found = {}
    for reach in network.reaches.values():
        for node in (reach.start, reach.end):
            if quantity in node.data.columns:
                found[node.id] = node.data[quantity]
    if nodes is None:
        return pd.DataFrame(found)
    return pd.DataFrame({node_id: found[node_id] for node_id in nodes})
