"""Helpers shared by the tests that build networks by hand.

Importing this module needs mikeio1d, which is an optional dependency (ADR-010),
so guard the import with ``pytest.importorskip("mikeio1d.network")`` first.
"""

from __future__ import annotations

import pandas as pd
from mikeio1d.network import Network, BasicNode, BasicReach, ReachBreakPoint


class BreakPoint(ReachBreakPoint):
    """A break point at a known distance along a reach."""

    def __init__(self, reach, distance, data):
        self._id = (reach, distance)
        self._data = data

    @property
    def id(self):
        return self._id

    @property
    def data(self):
        return self._data


def make_network(node_ids, time, data, quantity="WaterLevel"):
    """A chain of nodes, each carrying one quantity, joined by unit reaches."""
    nodes = [
        BasicNode(node_id, pd.DataFrame({quantity: data[:, i]}, index=time))
        for i, node_id in enumerate(node_ids)
    ]
    reaches = [
        BasicReach(f"r{i}", nodes[i], nodes[i + 1], length=100.0)
        for i in range(len(nodes) - 1)
    ]
    return Network(reaches)


def make_breakpoint_network(reach_id, distance, data):
    """A one-reach network whose data sits on a break point, not on its nodes."""
    empty = pd.DataFrame()
    reach = BasicReach(
        reach_id,
        BasicNode("start", empty),
        BasicNode("end", empty),
        length=100.0,
        breakpoints=[BreakPoint(reach_id, distance, data)],
    )
    return Network([reach])


def node_series(network, quantity="WaterLevel"):
    """Each node's own series for `quantity`, keyed by node id, read off the reaches.

    The three nodes of `sample_network` carry three different series, so a
    comparer built from one of them cannot be satisfied by any of the others.
    """
    nodes = {}
    for reach in network.reaches.values():
        for node in (reach.start, reach.end):
            nodes[node.id] = node.data[quantity]
    return pd.DataFrame(nodes)
