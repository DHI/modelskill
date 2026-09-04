"""Test network models and observations"""

# ruff: noqa: E402
import sys
from pathlib import Path
import pytest

pytest.importorskip("networkx")

import pandas as pd
import xarray as xr
import numpy as np
import modelskill as ms
from modelskill.model.network import (
    NetworkModelResult,
    NodeModelResult,
)
from modelskill.model.adapters._inp import read_pipe_lengths, read_sections
from modelskill.model.adapters._res1d import (
    Res1DNode,
    Res1DReach,
    _simplify_colnames,
)
from modelskill.network import (
    Network,
    BasicNode,
    BasicReach,
    NetworkReach,
    ReachBreakPoint,
    _EPANET_EXTENSIONS,
    _MIKE_EXTENSIONS,
    _UNSUPPORTED_EXTENSIONS,
)
from modelskill.obs import NodeObservation
from modelskill.quantity import Quantity


def _make_network(node_ids, time, data, quantity="WaterLevel"):
    nodes = [
        BasicNode(nid, pd.DataFrame({quantity: data[:, i]}, index=time))
        for i, nid in enumerate(node_ids)
    ]
    reaches = [
        BasicReach(f"r{i}", nodes[i], nodes[i + 1], length=100.0)
        for i in range(len(nodes) - 1)
    ]
    return Network(reaches)


@pytest.fixture
def sample_network_data():
    """Sample network data as xr.Dataset"""
    time = pd.date_range("2010-01-01", periods=10, freq="h")
    nodes = [123, 456, 789]

    # Create sample data
    np.random.seed(42)  # For reproducible tests
    data = np.random.randn(len(time), len(nodes))

    ds = xr.Dataset(
        {
            "WaterLevel": (["time", "node"], data),
        },
        coords={
            "time": time,
            "node": nodes,
        },
    )
    ds["WaterLevel"].attrs["units"] = "m"
    ds["WaterLevel"].attrs["long_name"] = "Water Level"

    return ds


@pytest.fixture
def sample_network():
    """Sample Network with 3 nodes (WaterLevel quantity)"""
    time = pd.date_range("2010-01-01", periods=10, freq="h")
    np.random.seed(42)
    data = np.random.randn(10, 3)
    return _make_network(["123", "456", "789"], time, data)


@pytest.fixture
def sample_network_multivars():
    """Sample Network with 2 nodes and 2 quantities (WaterLevel + Discharge)"""
    time = pd.date_range("2010-01-01", periods=10, freq="h")
    np.random.seed(42)
    raw = np.random.randn(10, 2)
    nodes = [
        BasicNode(
            nid,
            pd.DataFrame(
                {"WaterLevel": raw[:, i], "Discharge": raw[:, i] * 10},
                index=time,
            ),
        )
        for i, nid in enumerate(["123", "456"])
    ]
    reaches = [BasicReach("r1", nodes[0], nodes[1], length=100.0)]
    return Network(reaches)


@pytest.fixture
def dataset_without_node():
    time = pd.date_range("2010-01-01", periods=10, freq="h")

    # Create sample data
    np.random.seed(42)  # For reproducible tests
    data = np.random.randn(len(time))

    ds = xr.Dataset(
        {
            "WaterLevel": (["time"], data),
        },
        coords={
            "time": time,
        },
    )
    ds["WaterLevel"].attrs["units"] = "m"
    ds["WaterLevel"].attrs["long_name"] = "Water Level"

    return ds


@pytest.fixture
def sample_node_data():
    """Sample node observation data"""
    time = pd.date_range("2010-01-01", periods=10, freq="h")

    # Create sample data with some variation
    np.random.seed(42)
    data = np.random.randn(len(time)) * 0.1 + 1.5

    df = pd.DataFrame({"WaterLevel": data}, index=time)

    return df


@pytest.fixture
def sample_series(sample_node_data):
    """Sample node observation data as series"""
    return sample_node_data["WaterLevel"]


class TestNetworkModelResult:
    """Test NetworkModelResult class"""

    def test_init_with_network(self, sample_network):
        """Test initialization with a Network object"""
        nmr = NetworkModelResult(sample_network)

        assert len(nmr.time) == 10
        assert isinstance(nmr.time, pd.DatetimeIndex)
        assert len(nmr.nodes) == 3

    def test_init_with_name(self, sample_network):
        """Test initialization with explicit name"""
        nmr = NetworkModelResult(sample_network, name="Test_Network")
        assert nmr.name == "Test_Network"

    def test_init_with_item_selection(self, sample_network_multivars):
        """Test initialization with specific item selection"""
        nmr = NetworkModelResult(
            sample_network_multivars, item="WaterLevel", name="Network_WL"
        )

        assert nmr.name == "Network_WL"
        assert "WaterLevel" in nmr.data.data_vars
        assert "Discharge" not in nmr.data.data_vars

    def test_init_fails_with_unsupported_type(self):
        """Test that passing a non-Network object raises an error"""
        with pytest.raises((TypeError, AttributeError)):
            NetworkModelResult(xr.Dataset())  # type: ignore[arg-type]

    def test_repr(self, sample_network):
        """Test string representation"""
        nmr = NetworkModelResult(sample_network, name="Test_Network")
        repr_str = repr(nmr)

        assert "NetworkModelResult" in repr_str
        assert "Test_Network" in repr_str

    def test_extract_valid_node(self, sample_network, sample_node_data):
        """Test extraction of a valid node"""
        nmr = NetworkModelResult(sample_network)
        node_id = sample_network.find(node="123")
        obs = NodeObservation(sample_node_data, at=node_id, name="Node_123")

        extracted = nmr.extract(obs)

        assert isinstance(extracted, NodeModelResult)
        assert extracted.node == node_id
        assert len(extracted.time) == 10

    def test_extract_invalid_node(self, sample_network, sample_node_data):
        """Test extraction of a node not present in the network"""
        nmr = NetworkModelResult(sample_network)
        obs = NodeObservation(sample_node_data, at=999, name="Node_999")

        with pytest.raises(ValueError, match="Node 999 not found"):
            nmr.extract(obs)

    def test_extract_wrong_observation_type(self, sample_network):
        """Test extraction with wrong observation type"""
        nmr = NetworkModelResult(sample_network)

        df = pd.DataFrame(
            {"WL": [1, 2, 3]}, index=pd.date_range("2010-01-01", periods=3, freq="h")
        )
        obs = ms.PointObservation(df, x=0.0, y=0.0)

        with pytest.raises(
            TypeError,
            match="NetworkModelResult supports NodeObservation and ReachObservation",
        ):
            nmr.extract(obs)


class TestNodeObservation:
    """Test NodeObservation class"""

    @pytest.fixture
    def multi_data(self, sample_node_data):
        """Multi-column DataFrame with 3 stations"""
        return pd.DataFrame(
            {
                "station_0": sample_node_data["WaterLevel"],
                "station_1": sample_node_data["WaterLevel"] + 0.1,
                "station_2": sample_node_data["WaterLevel"] + 0.2,
            }
        )

    def test_init_with_df(self, sample_node_data):
        """Test initialization with pandas DataFrame"""

        obs = NodeObservation(
            sample_node_data, at=123, name="Sensor_1", item="WaterLevel"
        )

        assert obs.at == 123
        assert obs.name == "Sensor_1"
        assert len(obs.time) == 10
        assert isinstance(obs.time, pd.DatetimeIndex)

    def test_init_with_series(self, sample_series):
        """Test initialization with pandas Series"""
        obs = NodeObservation(sample_series, at=456, name="Node_456")

        assert obs.at == 456
        assert obs.name == "Node_456"
        assert len(obs.time) == 10

    def test_node_attrs(self, sample_node_data):
        """Test attrs property"""
        attrs = {"source": "test", "version": "1.0"}
        obs = NodeObservation(sample_node_data, at=123, attrs=attrs, weight=2.5)

        assert obs.attrs["source"] == "test"
        assert obs.attrs["version"] == "1.0"
        assert obs.weight == 2.5
        assert obs.quantity == Quantity.undefined()

    def test_multiple_nodes_returns_list_of_observations(self, multi_data):
        """Test that from_multiple returns a list of NodeObservation objects"""
        obs_list = NodeObservation.from_multiple(
            data=multi_data,
            nodes={123: "station_0", 456: "station_1", 789: "station_2"},
        )

        assert len(obs_list) == 3
        assert all(isinstance(obs, NodeObservation) for obs in obs_list)

    def test_node_ids_are_assigned_correctly(self, multi_data):
        obs_list = NodeObservation.from_multiple(
            data=multi_data,
            nodes={123: "station_0", 456: "station_1", 789: "station_2"},
        )

        assert obs_list[0].node == 123
        assert obs_list[1].node == 456
        assert obs_list[2].node == 789

    def test_names_derived_from_column_names(self, multi_data):
        obs_list = NodeObservation.from_multiple(
            data=multi_data,
            nodes={123: "station_0", 456: "station_1", 789: "station_2"},
        )

        assert obs_list[0].name == "station_0"
        assert obs_list[1].name == "station_1"
        assert obs_list[2].name == "station_2"

    def test_from_xarray_dataset(self, sample_node_data):
        ds = xr.Dataset(
            {
                "station_0": ("time", sample_node_data["WaterLevel"].values),
                "station_1": ("time", sample_node_data["WaterLevel"].values + 0.1),
            },
            coords={"time": sample_node_data.index},
        )
        obs_list = NodeObservation.from_multiple(
            data=ds, nodes={123: "station_0", 456: "station_1"}
        )

        assert len(obs_list) == 2
        assert obs_list[0].node == 123
        assert obs_list[1].node == 456

    def test_nodes_must_be_dict(self, multi_data):
        with pytest.raises(TypeError, match="'nodes' must be a dict"):
            NodeObservation.from_multiple(data=multi_data, nodes=123)

    def test_attrs_propagated_to_all_observations(self, multi_data):
        attrs = {"source": "sensor_array", "version": 2}
        obs_list = NodeObservation.from_multiple(
            data=multi_data,
            nodes={1: "station_0", 2: "station_1", 3: "station_2"},
            attrs=attrs,
        )

        for obs in obs_list:
            assert obs.attrs["source"] == "sensor_array"
            assert obs.attrs["version"] == 2

    def test_init_from_csv(self):
        obs = NodeObservation(
            "tests/testdata/network_sensor_1.csv", at=1, item="water_level@sens1"
        )

        assert obs.at == 1
        assert len(obs.time) == 110
        assert isinstance(obs.time, pd.DatetimeIndex)

    def test_from_multiple_csvs_via_dict(self):
        obs_list = NodeObservation.from_multiple(
            nodes={
                1: "tests/testdata/network_sensor_1.csv",
                2: "tests/testdata/network_sensor_2.csv",
                3: "tests/testdata/network_sensor_3.csv",
            }
        )

        assert len(obs_list) == 3
        assert all(isinstance(obs, NodeObservation) for obs in obs_list)
        assert obs_list[0].node == 1
        assert obs_list[1].node == 2
        assert obs_list[2].node == 3
        for obs in obs_list:
            assert len(obs.time) > 0

    def test_nodes_dict_maps_node_to_item(self, multi_data):
        obs_list = NodeObservation.from_multiple(
            data=multi_data, nodes={123: "station_0", 456: "station_1"}
        )

        assert len(obs_list) == 2
        assert obs_list[0].node == 123
        assert obs_list[1].node == 456
        assert obs_list[0].name == "station_0"
        assert obs_list[1].name == "station_1"

    def test_nodes_none_raises(self, multi_data):
        with pytest.raises(ValueError, match="'nodes' argument is required"):
            NodeObservation.from_multiple(data=multi_data, nodes=None)

    def test_single_node_dict(self, sample_node_data):
        obs_list = NodeObservation.from_multiple(
            data=sample_node_data, nodes={123: "WaterLevel"}
        )

        assert len(obs_list) == 1
        assert isinstance(obs_list[0], NodeObservation)
        assert obs_list[0].node == 123


class TestNodeModelResult:
    """Test NodeModelResult class"""

    @pytest.mark.parametrize("fixture_name", ["sample_node_data", "sample_series"])
    def test_init_(self, request, fixture_name):
        """Test initialization with pandas DataFrame"""
        data = request.getfixturevalue(fixture_name)
        nmr = NodeModelResult(data, node=123, name="Node_123_Model")

        assert nmr.node == 123
        assert nmr.name == "Node_123_Model"
        assert len(nmr.time) == 10


class TestNetworkIntegration:
    """Test integration between network models and observations"""

    def test_network_to_node_extraction(self, sample_network, sample_node_data):
        """Test complete workflow from network model to node extraction"""
        nmr = NetworkModelResult(sample_network, name="Network_Model")
        node_id = sample_network.find(node="123")
        obs = NodeObservation(sample_node_data, at=node_id, name="Node_123_Obs")

        extracted = nmr.extract(obs)

        assert isinstance(extracted, NodeModelResult)
        assert extracted.node == node_id
        assert extracted.name == "Network_Model"
        assert len(extracted.time) == len(obs.time)

    def test_matching_workflow(self, sample_network, sample_node_data):
        """Test matching workflow with network data"""
        nmr = NetworkModelResult(sample_network, name="Network_Model")
        node_id = sample_network.find(node="123")
        obs = NodeObservation(sample_node_data, at=node_id, name="Node_123_Obs")

        comparer = ms.match(obs, nmr)

        assert comparer is not None
        assert "Network_Model" in comparer.mod_names
        assert comparer.n_points > 0

    def test_matching_workflow_multiple_nodes(self, sample_network, sample_node_data):
        """Test matching workflow with multiple node observations"""
        nmr = NetworkModelResult(sample_network, name="Network_Model")

        multi_data = pd.DataFrame(
            {
                "station_0": sample_node_data["WaterLevel"],
                "station_1": sample_node_data["WaterLevel"] + 0.1,
                "station_2": sample_node_data["WaterLevel"] + 0.2,
            }
        )

        node_0 = sample_network.find(node="123")
        node_1 = sample_network.find(node="456")
        node_2 = sample_network.find(node="789")

        # Create multiple NodeObservations using .from_multiple
        obs_list = NodeObservation.from_multiple(
            data=multi_data,
            nodes={node_0: "station_0", node_1: "station_1", node_2: "station_2"},
        )

        # Test that matching works
        comparer_collection = ms.match(obs_list, nmr)

        assert comparer_collection is not None
        assert len(comparer_collection) == 3

        for comparer in comparer_collection:
            assert "Network_Model" in comparer.mod_names
            assert comparer.n_points > 0


@pytest.mark.skipif(
    sys.version_info >= (3, 14), reason="mikeio1d requires Python < 3.14"
)
def test_open_res1d():
    path_to_file = "./tests/testdata/network.res1d"
    network = Network.from_mike(path_to_file)
    assert network.graph.number_of_nodes() == 259


@pytest.mark.skipif(
    sys.version_info >= (3, 14), reason="mikeio1d requires Python < 3.14"
)
def test_extract_reach_observation_happy_path(sample_node_data):
    path_to_file = "./tests/testdata/network.res1d"
    network = Network.from_mike(path_to_file)
    nmr = NetworkModelResult(network, item="Discharge", name="network_model")
    obs_data = sample_node_data.rename(columns={"WaterLevel": "Discharge"})
    obs = ms.ReachObservation(obs_data, reach="100l1", item="Discharge")

    extracted = nmr.extract(obs)

    assert isinstance(extracted, NodeModelResult)
    assert extracted.name == "network_model"
    assert extracted.node in nmr.nodes


@pytest.mark.skipif(
    sys.version_info >= (3, 14), reason="mikeio1d requires Python < 3.14"
)
def test_extract_reach_observation_non_equivalent_breakpoints_raises(sample_node_data):
    path_to_file = "./tests/testdata/network.res1d"
    network = Network.from_mike(path_to_file)
    nmr = NetworkModelResult(network, item="Discharge")
    obs_data = sample_node_data.rename(columns={"WaterLevel": "Discharge"})
    obs = ms.ReachObservation(obs_data, reach="113l1", item="Discharge")

    with pytest.raises(ValueError, match="Not all data in breakpoints are equivalent"):
        nmr.extract(obs)


@pytest.mark.skipif(
    sys.version_info >= (3, 14), reason="mikeio1d requires Python < 3.14"
)
def test_extract_reach_observation_with_reaches_not_populated_raises_valueerror(
    sample_node_data,
):
    path_to_file = "./tests/testdata/network.res1d"
    network = Network.from_mike(path_to_file, reaches=[])
    nmr = NetworkModelResult(network, item="WaterLevel")
    obs = ms.ReachObservation(sample_node_data, reach="100l1", item="WaterLevel")

    with pytest.raises(ValueError, match="none of its breakpoints have data loaded"):
        nmr.extract(obs)


@pytest.mark.skipif(
    sys.version_info >= (3, 14), reason="mikeio1d requires Python < 3.14"
)
def test_extract_reach_observation_breakpoint_node_missing_raises_valueerror(
    sample_node_data,
):
    path_to_file = "./tests/testdata/network.res1d"
    network = Network.from_mike(path_to_file)
    nmr = NetworkModelResult(network, item="Discharge")
    obs_data = sample_node_data.rename(columns={"WaterLevel": "Discharge"})
    baseline_obs = ms.ReachObservation(obs_data, reach="100l1", item="Discharge")
    node_id = nmr.extract(baseline_obs).node
    remaining_nodes = []
    for node in nmr.data.node.values:
        node_int = int(node)
        if node_int != node_id:
            remaining_nodes.append(node_int)
    nmr.data = nmr.data.sel(node=remaining_nodes)

    obs = ms.ReachObservation(obs_data, reach="100l1", item="Discharge")

    with pytest.raises(ValueError, match="matching breakpoint nodes are missing"):
        nmr.extract(obs)


@pytest.mark.skipif(
    sys.version_info >= (3, 14), reason="mikeio1d requires Python < 3.14"
)
def test_from_mike_nodes_filter_creates_full_network():
    """When nodes is specified, the full network topology is created."""
    path_to_file = "./tests/testdata/network.res1d"
    full_network = Network.from_mike(path_to_file)

    selected_nodes = ["1", "108"]
    partial_network = Network.from_mike(path_to_file, nodes=selected_nodes)

    # Full topology is preserved
    assert (
        partial_network.graph.number_of_nodes() == full_network.graph.number_of_nodes()
    )


@pytest.mark.skipif(
    sys.version_info >= (3, 14), reason="mikeio1d requires Python < 3.14"
)
def test_from_mike_nodes_filter_only_selected_have_data():
    """When nodes is specified, only selected nodes contain non-empty data."""
    path_to_file = "./tests/testdata/network.res1d"

    selected_nodes = ["1", "108"]
    network = Network.from_mike(path_to_file, nodes=selected_nodes, reaches=[])
    g = network.graph.copy()

    n_nodes = network.graph.number_of_nodes()
    assert sum([g.nodes[n]["data"].empty for n in g.nodes]) == n_nodes - 2
    for n in selected_nodes:
        assert not g.nodes[network.find(n)]["data"].empty


@pytest.mark.skipif(
    sys.version_info >= (3, 14), reason="mikeio1d requires Python < 3.14"
)
def test_from_mike_nodes_single_string():
    """nodes argument accepts a single string (not just a list)."""
    path_to_file = "./tests/testdata/network.res1d"
    full_network = Network.from_mike(path_to_file)

    network = Network.from_mike(path_to_file, nodes="108", reaches=[])
    g = network.graph.copy()

    assert g.number_of_nodes() == full_network.graph.number_of_nodes()

    nodes_with_data = [n for n in g.nodes if not g.nodes[n]["data"].empty]
    nodes_with_data = [network.recall(n)["node"] for n in nodes_with_data]
    assert nodes_with_data == ["108"]


@pytest.mark.skipif(
    sys.version_info >= (3, 14), reason="mikeio1d requires Python < 3.14"
)
def test_dataframe_from_partial_network():
    """nodes argument accepts a single string (not just a list)."""
    path_to_file = "./tests/testdata/network.res1d"
    selected_nodes = ["108", "101"]
    network = Network.from_mike(path_to_file, nodes=selected_nodes, reaches=[])
    nodes_in_df = network.to_dataframe().droplevel(axis=1, level=1).columns

    assert set(nodes_in_df) == set([network.find(n) for n in selected_nodes])


@pytest.mark.skipif(
    sys.version_info >= (3, 14), reason="mikeio1d requires Python < 3.14"
)
def test_nodes_filtered_network_keeps_datetime_index():
    """Topology-only nodes must not degrade the time index to object dtype.

    A nodes-filtered network keeps the full topology, storing empty data for
    the unselected nodes. Concatenating those empty (RangeIndex) frames must
    not corrupt the DatetimeIndex, otherwise ms.match() later fails with
    "time must be datetime".
    """
    path_to_file = "./tests/testdata/network.res1d"
    network = Network.from_mike(path_to_file, nodes=["108", "101"], reaches=[])

    assert isinstance(network._df.index, pd.DatetimeIndex)
    assert network._df.index.dtype == "datetime64[ns]"
    assert network.to_dataset()["time"].dtype == np.dtype("datetime64[ns]")


@pytest.mark.skipif(
    sys.version_info >= (3, 14), reason="mikeio1d requires Python < 3.14"
)
def test_from_res1d_empty_nodes_and_reaches_keeps_topology_and_empty_outputs():
    path_to_file = "./tests/testdata/network.res1d"
    network = Network.from_mike(path_to_file, nodes=["108", "101"], reaches=[])

    assert isinstance(network._df.index, pd.DatetimeIndex)
    assert network._df.index.dtype == "datetime64[ns]"
    assert network.to_dataset()["time"].dtype == np.dtype("datetime64[ns]")


@pytest.mark.skipif(
    sys.version_info >= (3, 14), reason="mikeio1d requires Python < 3.14"
)
def test_from_mike_empty_nodes_and_reaches_keeps_topology_and_empty_outputs():
    path_to_file = "./tests/testdata/network.res1d"
    full_network = Network.from_mike(path_to_file)
    network = Network.from_mike(path_to_file, nodes=[], reaches=[])

    assert network.graph.number_of_nodes() == full_network.graph.number_of_nodes()

    df = network.to_dataframe()
    assert df.empty
    assert isinstance(df.columns, pd.MultiIndex)
    assert df.columns.names == ["node", "quantity"]
    assert df.index.name == "time"

    ds = network.to_dataset()
    assert isinstance(ds, xr.Dataset)
    assert len(ds.data_vars) == 0


@pytest.mark.skipif(
    sys.version_info >= (3, 14), reason="mikeio1d requires Python < 3.14"
)
def test_from_mike_topology_only_locations_share_one_empty_frame():
    """Topology-only locations must not each allocate their own empty frame.

    That is two DataFrame allocations per reach, which profiling showed to be
    the single largest cost of a filtered load (gh #679).
    """
    path_to_file = "./tests/testdata/network.res1d"

    network = Network.from_mike(path_to_file, nodes=[], reaches=[])
    g = network.graph

    frames = [g.nodes[n]["data"] for n in g.nodes]
    assert all(frame.empty for frame in frames)
    assert len({id(frame) for frame in frames}) == 1


@pytest.mark.skipif(
    sys.version_info >= (3, 14), reason="mikeio1d requires Python < 3.14"
)
def test_from_mike_reads_each_selected_node_once(monkeypatch):
    """A node shared by several reaches is read once, not once per reach.

    _generate_graph keeps only the first copy of a node's data, so the repeat
    reads were discarded work (gh #679). The per-reach boundary reads are
    genuinely distinct and must not be collapsed.
    """
    from modelskill.model.adapters import _res1d

    reads: dict[str, int] = {}
    unpatched = _res1d._simplify_colnames

    def counting_simplify_colnames(location, *args, **kwargs):
        name = type(location).__name__
        reads[name] = reads.get(name, 0) + 1
        return unpatched(location, *args, **kwargs)

    monkeypatch.setattr(_res1d, "_simplify_colnames", counting_simplify_colnames)

    path_to_file = "./tests/testdata/network.res1d"
    network = Network.from_mike(path_to_file, reaches=[])

    n_unique_nodes = len(
        {reach.start.id for reach in network._reaches.values()}
        | {reach.end.id for reach in network._reaches.values()}
    )
    n_reach_endpoints = 2 * len(network._reaches)

    assert reads["ResultNode"] == n_unique_nodes
    assert reads["ResultGridPoint"] == n_reach_endpoints


# ---------------------------------------------------------------------------
# Optional reach length
# ---------------------------------------------------------------------------


class _StubBreakPoint(ReachBreakPoint):
    """Minimal concrete ReachBreakPoint for building reaches by hand."""

    def __init__(self, reach_id, distance, data=None):
        self._id = (reach_id, distance)
        self._data = pd.DataFrame() if data is None else data

    @property
    def id(self):
        return self._id

    @property
    def data(self):
        return self._data


def _two_node_pair():
    time = pd.date_range("2020", periods=3, freq="h")
    df = pd.DataFrame({"WaterLevel": [1.0, 1.1, 1.2]}, index=time)
    return BasicNode("a", df), BasicNode("b", df.copy())


class TestOptionalReachLength:
    """Reach length is undefined in some domains, so it must be omittable."""

    def test_subclass_may_omit_length(self):
        class LengthlessReach(NetworkReach):
            def __init__(self, id, start, end):
                self._id, self._start, self._end = id, start, end

            @property
            def id(self):
                return self._id

            @property
            def start(self):
                return self._start

            @property
            def end(self):
                return self._end

            @property
            def breakpoints(self):
                return []

        a, b = _two_node_pair()
        reach = LengthlessReach("r1", a, b)

        assert reach.length is None
        assert Network([reach]).graph.number_of_nodes() == 2

    def test_basic_reach_length_defaults_to_none(self):
        a, b = _two_node_pair()

        assert BasicReach("r1", a, b).length is None

    def test_edge_length_is_none_when_undefined(self):
        a, b = _two_node_pair()

        network = Network([BasicReach("r1", a, b)])

        assert [d["length"] for *_, d in network.graph.edges(data=True)] == [None]

    def test_breakpoint_distances_survive_an_undefined_length(self):
        """Only the final segment needs the total, so the rest keep real lengths."""
        a, b = _two_node_pair()
        breakpoints = [_StubBreakPoint("r1", d) for d in (30.0, 70.0)]

        network = Network([BasicReach("r1", a, b, breakpoints=breakpoints)])

        lengths = sorted(
            (d["length"] for *_, d in network.graph.edges(data=True)),
            key=lambda v: (v is None, v),
        )
        assert lengths == [30.0, 40.0, None]

    def test_length_weighted_algorithms_fail_loudly(self):
        """Storing None keeps networkx honest.

        Omitting the attribute instead would let networkx default the weight to
        1, so every call below would return a plausible but meaningless number.
        With None, shortest-path treats the edge as hidden and the arithmetic
        consumers raise.
        """
        import networkx as nx

        a, b = _two_node_pair()
        g = Network([BasicReach("r1", a, b)]).graph

        with pytest.raises(nx.NetworkXNoPath):
            nx.shortest_path_length(g, 0, 1, weight="length")

        with pytest.raises(TypeError):
            g.size(weight="length")

    def test_known_length_is_unchanged(self):
        a, b = _two_node_pair()
        breakpoints = [_StubBreakPoint("r1", 40.0)]

        network = Network([BasicReach("r1", a, b, 100.0, breakpoints)])

        assert sorted(d["length"] for *_, d in network.graph.edges(data=True)) == [
            40.0,
            60.0,
        ]


# ---------------------------------------------------------------------------
# Which extensions each constructor accepts, and why the rest are refused
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    sys.version_info >= (3, 14), reason="mikeio1d requires Python < 3.14"
)
class TestExtensionPolicy:
    @pytest.mark.parametrize("suffix", [".res1d", ".res11", ".RES1D"])
    def test_from_mike_accepts_mike_extensions(self, tmp_path, suffix):
        """The file does not exist, so mikeio1d - not the guard - is what complains."""
        with pytest.raises((FileExistsError, FileNotFoundError)):
            Network.from_mike(tmp_path / f"network{suffix}")

    def test_error_lists_only_readable_extensions(self):
        with pytest.raises(NotImplementedError) as excinfo:
            Network.from_mike("network.nc")

        message = str(excinfo.value)
        for extension in _MIKE_EXTENSIONS | _EPANET_EXTENSIONS:
            assert extension in message
        for extension in _UNSUPPORTED_EXTENSIONS:
            assert extension not in message

    def test_swmm_refusal_names_the_companion_inp(self):
        """A real file, so this fails the day SWMM support lands."""
        with pytest.raises(NotImplementedError, match=r"companion '\.inp'"):
            Network.from_mike("./tests/testdata/swmm.out")

    def test_resx_refusal_points_at_the_resx_argument(self):
        """'.resx' is a companion, so the message must name what to do instead."""
        with pytest.raises(
            NotImplementedError, match=r"from_epanet\(res, resx=\.\.\.\)"
        ):
            Network.from_mike("./tests/testdata/epanet.resx")

    @pytest.mark.parametrize("suffix", [".prf", ".crf", ".xrf", ".whr"])
    def test_formats_without_a_fixture_are_refused(self, tmp_path, suffix):
        with pytest.raises(NotImplementedError, match="no test fixture"):
            Network.from_mike(tmp_path / f"network{suffix}")

    def test_every_mikeio1d_extension_is_accounted_for(self):
        """A new mikeio1d format must be read or explicitly refused, never ignored."""
        from mikeio1d import Res1D

        accounted_for = (
            _MIKE_EXTENSIONS | _EPANET_EXTENSIONS | set(_UNSUPPORTED_EXTENSIONS)
        )

        assert accounted_for == Res1D.get_supported_file_extensions()

    def test_res1d_opened_with_a_path_is_refused(self):
        """mikeio1d calls str.endswith on file_path, so a Path breaks it later on."""
        from mikeio1d import Res1D

        res = Res1D(Path("./tests/testdata/network.res1d"))

        with pytest.raises(TypeError, match="file_path"):
            Network.from_mike(res)


# ---------------------------------------------------------------------------
# from_mike — quantities filter
#
# In network.res1d every node carries WaterLevel only, interior gridpoints
# carry either Discharge or WaterLevel.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    sys.version_info >= (3, 14), reason="mikeio1d requires Python < 3.14"
)
def test_from_mike_reads_all_quantities_by_default():
    path_to_file = "./tests/testdata/network.res1d"
    network = Network.from_mike(path_to_file)

    assert set(network.quantities) == {"WaterLevel", "Discharge"}


@pytest.mark.skipif(
    sys.version_info >= (3, 14), reason="mikeio1d requires Python < 3.14"
)
def test_from_mike_quantities_filter_reads_only_requested_quantity():
    path_to_file = "./tests/testdata/network.res1d"
    network = Network.from_mike(path_to_file, quantities="Discharge")

    assert network.quantities == ["Discharge"]


@pytest.mark.skipif(
    sys.version_info >= (3, 14), reason="mikeio1d requires Python < 3.14"
)
def test_from_mike_quantities_filter_leaves_other_locations_topology_only():
    """A location that does not carry the requested quantity is not an error."""
    path_to_file = "./tests/testdata/network.res1d"
    network = Network.from_mike(path_to_file, quantities="Discharge")
    g = network.graph

    assert g.nodes[network.find(node="1")]["data"].empty
    assert (
        g.number_of_nodes() == Network.from_mike(path_to_file).graph.number_of_nodes()
    )


@pytest.mark.skipif(
    sys.version_info >= (3, 14), reason="mikeio1d requires Python < 3.14"
)
def test_from_mike_quantities_filter_populates_matching_breakpoints():
    path_to_file = "./tests/testdata/network.res1d"
    network = Network.from_mike(path_to_file, quantities="Discharge")

    breakpoints = network._reaches["100l1"].breakpoints
    populated = [bp for bp in breakpoints if not bp.data.empty]

    assert [bp.quantities for bp in populated] == [["Discharge"]]


@pytest.mark.skipif(
    sys.version_info >= (3, 14), reason="mikeio1d requires Python < 3.14"
)
def test_from_mike_quantities_filter_keeps_node_data_for_node_quantity():
    path_to_file = "./tests/testdata/network.res1d"
    network = Network.from_mike(path_to_file, quantities="WaterLevel")

    assert network.quantities == ["WaterLevel"]
    assert not network.graph.nodes[network.find(node="1")]["data"].empty


@pytest.mark.skipif(
    sys.version_info >= (3, 14), reason="mikeio1d requires Python < 3.14"
)
def test_from_mike_empty_quantities_keeps_topology_and_loads_nothing():
    path_to_file = "./tests/testdata/network.res1d"
    full_network = Network.from_mike(path_to_file)
    network = Network.from_mike(path_to_file, quantities=[])

    assert network.graph.number_of_nodes() == full_network.graph.number_of_nodes()
    assert network.quantities == []

    df = network.to_dataframe()
    assert df.empty
    assert isinstance(df.columns, pd.MultiIndex)


@pytest.mark.skipif(
    sys.version_info >= (3, 14), reason="mikeio1d requires Python < 3.14"
)
def test_from_mike_quantities_filter_combines_with_nodes_filter():
    path_to_file = "./tests/testdata/network.res1d"
    network = Network.from_mike(
        path_to_file, nodes=["1"], reaches=[], quantities=["WaterLevel"]
    )
    g = network.graph

    nodes_with_data = [n for n in g.nodes if not g.nodes[n]["data"].empty]
    assert [network.recall(n)["node"] for n in nodes_with_data] == ["1"]
    assert list(g.nodes[network.find(node="1")]["data"].columns) == ["WaterLevel"]


# ---------------------------------------------------------------------------
# NodeObservation — alias / breakpoint node forms
# ---------------------------------------------------------------------------


class TestNodeObservationAliases:
    """NodeObservation accepts int, str alias, and (reach, distance) tuple."""

    def test_integer_node_unchanged(self, sample_node_data):
        obs = NodeObservation(sample_node_data, at=42)
        assert obs.at == 42
        assert isinstance(obs.at, int)

    def test_integer_node_coord(self, sample_node_data):
        obs = NodeObservation(sample_node_data, at=42)
        assert "node" in obs.data.coords
        assert int(obs.data.coords["node"].item()) == 42

    def test_string_alias_stored(self, sample_node_data):
        obs = NodeObservation(sample_node_data, at="node_A", name="test")
        assert obs.at == "node_A"
        assert isinstance(obs.at, str)

    def test_string_alias_has_node_coord(self, sample_node_data):
        obs = NodeObservation(sample_node_data, at="node_A")
        assert "node" in obs.data.coords
        assert obs.data.coords["node"].item() == "node_A"

    def test_string_alias_gtype_is_node(self, sample_node_data):
        obs = NodeObservation(sample_node_data, at="node_A")
        assert obs.data.attrs["gtype"] == "node"

    def test_tuple_node_stored(self, sample_node_data):
        obs = NodeObservation(sample_node_data, at=("reach_1", 24.5))
        assert obs.at == ("reach_1", 24.5)
        assert isinstance(obs.at, tuple)

    def test_tuple_node_gtype_is_node(self, sample_node_data):
        obs = NodeObservation(sample_node_data, at=("reach_1", 24.5))
        assert obs.data.attrs["gtype"] == "node"

    def test_tuple_node_has_reach_distance_coords(self, sample_node_data):
        obs = NodeObservation(sample_node_data, at=("reach_1", 24.5))
        assert "reach" in obs.data.coords
        assert "distance" in obs.data.coords
        assert str(obs.data.coords["reach"].item()) == "reach_1"
        assert float(obs.data.coords["distance"].item()) == pytest.approx(24.5)

    def test_tuple_node_has_no_node_coord(self, sample_node_data):
        obs = NodeObservation(sample_node_data, at=("reach_1", 24.5))
        assert "node" not in obs.data.coords

    def test_tuple_node_roundtrip_via_create_new_instance(self, sample_node_data):
        obs = NodeObservation(sample_node_data, at=("reach_1", 24.5))
        obs2 = obs._create_new_instance(obs.data)
        assert obs2.at == ("reach_1", 24.5)

    def test_string_roundtrip_via_create_new_instance(self, sample_node_data):
        obs = NodeObservation(sample_node_data, at="node_A")
        obs2 = obs._create_new_instance(obs.data)
        assert obs2.at == "node_A"


# ---------------------------------------------------------------------------
# NetworkModelResult — alias resolution in extract()
# ---------------------------------------------------------------------------


class TestNetworkModelResultAliasResolution:
    """NetworkModelResult.extract() resolves str and tuple aliases via alias_map."""

    def test_network_stored(self, sample_network):
        nmr = NetworkModelResult(sample_network)
        assert hasattr(nmr, "network")
        assert "123" in nmr.network._alias_map
        assert "456" in nmr.network._alias_map
        assert "789" in nmr.network._alias_map

    def test_extract_with_string_alias(self, sample_network, sample_node_data):
        nmr = NetworkModelResult(sample_network)
        obs = NodeObservation(sample_node_data, at="123", name="Node_123")
        extracted = nmr.extract(obs)
        expected_id = sample_network.find(node="123")
        assert isinstance(extracted, NodeModelResult)
        assert extracted.node == expected_id

    def test_extract_string_alias_wrong_key_raises(
        self, sample_network, sample_node_data
    ):
        nmr = NetworkModelResult(sample_network)
        obs = NodeObservation(sample_node_data, at="nonexistent_node")
        with pytest.raises(ValueError, match="not found"):
            nmr.extract(obs)

    def test_extract_with_tuple_breakpoint(self, sample_network, sample_node_data):
        """Tuple alias is resolved via _alias_map (mapping injected for this test)."""
        nmr = NetworkModelResult(sample_network)
        existing_int = int(sample_network.find(node="123"))
        nmr.network._alias_map[("reach_test", 10.0)] = existing_int
        obs = NodeObservation(sample_node_data, at=("reach_test", 10.0))
        extracted = nmr.extract(obs)
        assert extracted.node == existing_int

    def test_extract_with_tuple_breakpoint_tolerance(
        self, sample_network, sample_node_data
    ):
        nmr = NetworkModelResult(sample_network)
        existing_int = int(sample_network.find(node="123"))
        base_distance = 10.0
        tol = NetworkModelResult._CHAINAGE_TOLERANCE
        nmr.network._alias_map[("reach_test", base_distance)] = existing_int
        obs = NodeObservation(
            sample_node_data, at=("reach_test", base_distance + tol / 2)
        )
        extracted = nmr.extract(obs)
        assert extracted.node == existing_int

    def test_extract_with_tuple_breakpoint_outside_tolerance_raises(
        self, sample_network, sample_node_data
    ):
        nmr = NetworkModelResult(sample_network)
        existing_int = int(sample_network.find(node="123"))
        base_distance = 10.0
        tol = NetworkModelResult._CHAINAGE_TOLERANCE
        nmr.network._alias_map[("reach_test", base_distance)] = existing_int
        obs = NodeObservation(
            sample_node_data, at=("reach_test", base_distance + tol + 1e-4)
        )
        with pytest.raises(ValueError, match="not found"):
            nmr.extract(obs)

    def test_extract_with_tuple_breakpoint_uses_closest_within_tolerance(
        self, sample_network, sample_node_data
    ):
        nmr = NetworkModelResult(sample_network)
        base_distance = 10.0
        tol = NetworkModelResult._CHAINAGE_TOLERANCE
        node_a = int(sample_network.find(node="123"))
        node_b = int(sample_network.find(node="456"))
        nmr.network._alias_map[("reach_test", base_distance + 2e-4)] = node_a
        nmr.network._alias_map[("reach_test", base_distance + 8e-4)] = node_b

        obs = NodeObservation(
            sample_node_data, at=("reach_test", base_distance + tol * 0.6)
        )
        extracted = nmr.extract(obs)
        assert extracted.node == node_b

    def test_extract_with_tuple_breakpoint_tie_uses_smallest_node_id(
        self, sample_network, sample_node_data
    ):
        nmr = NetworkModelResult(sample_network)
        base_distance = 10.0
        tol = NetworkModelResult._CHAINAGE_TOLERANCE
        node_a = int(sample_network.find(node="123"))
        node_b = int(sample_network.find(node="456"))
        nmr.network._alias_map[("reach_test", base_distance + 4e-4)] = node_a
        nmr.network._alias_map[("reach_test", base_distance + 8e-4)] = node_b

        obs = NodeObservation(
            sample_node_data, at=("reach_test", base_distance + tol * 0.6)
        )
        extracted = nmr.extract(obs)
        assert extracted.node == min(node_a, node_b)

    def test_extract_tuple_alias_wrong_key_raises(
        self, sample_network, sample_node_data
    ):
        nmr = NetworkModelResult(sample_network)
        obs = NodeObservation(sample_node_data, at=("nonexistent_reach", 0.0))
        with pytest.raises(ValueError, match="not found"):
            nmr.extract(obs)

    def test_match_with_string_alias(self, sample_network, sample_node_data):
        """Full ms.match() workflow works end-to-end with a string alias."""
        nmr = NetworkModelResult(sample_network, name="Network_Model")
        obs = NodeObservation(sample_node_data, at="123", name="Node_123")
        comparer = ms.match(obs, nmr)
        assert comparer.n_points > 0
        assert "Network_Model" in comparer.mod_names


# ---------------------------------------------------------------------------
# Res1D adapter — no mikeio1d required, the adapter is duck-typed
# ---------------------------------------------------------------------------


class _StubLocation:
    """Stands in for a mikeio1d ResultNode / ResultGridPoint."""

    def __init__(self, quantities, df=None):
        self.quantities = quantities
        self._df = df

    def to_dataframe(self):
        if self._df is None:
            raise AssertionError("to_dataframe() should not be called")
        return self._df


class TestSimplifyColnames:
    def test_location_without_quantities_gives_empty_frame(self):
        """MIKE 11 keeps its data on gridpoints, leaving nodes with no quantities."""
        df = _simplify_colnames(_StubLocation(quantities=[]))

        assert df.empty
        assert list(df.columns) == []

    def test_quantity_columns_are_stripped_of_location_suffix(self):
        time = pd.date_range("2020", periods=2, freq="h")
        raw = pd.DataFrame({"WaterLevel:node_1": [1.0, 2.0]}, index=time)

        df = _simplify_colnames(_StubLocation(quantities=["WaterLevel"], df=raw))

        assert list(df.columns) == ["WaterLevel"]


class _StubReach:
    """Stands in for a mikeio1d ResultReach."""

    def __init__(self, name="r1", start_node="a", end_node="b", length=100.0):
        self.name = name
        self.start_node = start_node
        self.end_node = end_node
        self.length = length
        self.gridpoints = []


class TestRes1DReachConnectivity:
    """Formats that expose no reach connectivity must fail with a clear message."""

    @pytest.mark.parametrize("missing", ["start_node", "end_node"])
    def test_missing_node_raises(self, missing):
        reach = _StubReach(**{missing: None})

        with pytest.raises(ValueError, match="no start/end node for reach 'r1'"):
            Res1DReach(reach, Res1DNode("a"), Res1DNode("b"))

    def test_both_nodes_missing_raises(self):
        """.resx reports None for both, which the identity checks alone would allow."""
        reach = _StubReach(start_node=None, end_node=None)

        with pytest.raises(ValueError, match="no start/end node"):
            Res1DReach(reach, Res1DNode(None), Res1DNode(None))  # type: ignore[arg-type]

    def test_mismatched_start_node_still_raises(self):
        with pytest.raises(ValueError, match="Incorrect starting node"):
            Res1DReach(_StubReach(), Res1DNode("wrong"), Res1DNode("b"))


class TestRes1DReachLength:
    """mikeio1d returns 0 when it cannot read a length; that is not a real zero."""

    @pytest.mark.parametrize("reported", [0, 0.0])
    def test_zero_becomes_undefined(self, reported):
        reach = Res1DReach(_StubReach(length=reported), Res1DNode("a"), Res1DNode("b"))

        assert reach.length is None

    def test_real_length_passes_through(self):
        reach = Res1DReach(_StubReach(length=47.5), Res1DNode("a"), Res1DNode("b"))

        assert reach.length == 47.5


# ---------------------------------------------------------------------------
# from_mike / from_epanet
# ---------------------------------------------------------------------------

requires_mikeio1d = pytest.mark.skipif(
    sys.version_info >= (3, 14), reason="mikeio1d requires Python < 3.14"
)


@requires_mikeio1d
class TestFromMike:
    def test_res1d(self):
        network = Network.from_mike("./tests/testdata/network.res1d")

        assert network.graph.number_of_nodes() == 259

    def test_res11(self):
        """MIKE 11 keeps its data on gridpoints, so its nodes are empty."""
        network = Network.from_mike("./tests/testdata/network_cali.res11")

        assert len(network._reaches) == 3
        assert network.graph.number_of_nodes() == 71
        assert set(network.quantities) == {"Discharge", "Water Level"}
        assert [r.n_breakpoints for r in network._reaches.values()] == [23, 21, 23]

    def test_res11_reaches_have_real_lengths(self):
        network = Network.from_mike("./tests/testdata/network_cali.res11")

        lengths = [d["length"] for *_, d in network.graph.edges(data=True)]
        assert all(length > 0 for length in lengths)

    def test_open_res1d_object(self):
        from mikeio1d import Res1D

        res = Res1D("./tests/testdata/network.res1d")

        network = Network.from_mike(res, nodes=[], reaches=[])

        assert network.graph.number_of_nodes() == 259

    def test_epanet_file_is_redirected(self):
        with pytest.raises(ValueError, match=r"Use Network\.from_epanet\(\)"):
            Network.from_mike("./tests/testdata/epanet.res")

    def test_unknown_extension(self):
        with pytest.raises(NotImplementedError, match="Unsupported file extension"):
            Network.from_mike("./tests/testdata/obs.dfs0")

    def test_unsupported_type(self):
        with pytest.raises(TypeError, match="Expected a str, Path or Res1D object"):
            Network.from_mike(42)  # type: ignore[arg-type]


@requires_mikeio1d
class TestFromEpanet:
    def test_epanet(self):
        network = Network.from_epanet("./tests/testdata/epanet.res")

        assert network.graph.number_of_nodes() == 11
        assert len(network._reaches) == 13
        assert set(network.quantities) == {
            "Demand",
            "Head",
            "Pressure",
            "WaterQuality",
        }
        assert not network.to_dataframe().empty

    def test_link_node_reaches_have_no_length_or_breakpoints(self):
        """Without inp=, mikeio1d reports neither - documented in the docstring."""
        network = Network.from_epanet("./tests/testdata/epanet.res")

        lengths = [d["length"] for *_, d in network.graph.edges(data=True)]
        assert lengths and all(length is None for length in lengths)
        assert all(r.n_breakpoints == 0 for r in network._reaches.values())

    def test_reach_observation_cannot_be_matched(self, sample_node_data):
        """Follows from having no breakpoints; also documented in the docstring."""
        network = Network.from_epanet("./tests/testdata/epanet.res")
        nmr = NetworkModelResult(network, item="Pressure")
        obs = ms.ReachObservation(sample_node_data, reach="10", item="WaterLevel")

        with pytest.raises(ValueError, match="breakpoints"):
            nmr.extract(obs)

    def test_mike_file_is_redirected(self):
        with pytest.raises(ValueError, match=r"Use Network\.from_mike\(\)"):
            Network.from_epanet("./tests/testdata/network.res1d")

    def test_open_res1d_object_is_validated(self):
        from mikeio1d import Res1D

        res = Res1D("./tests/testdata/network.res1d")

        with pytest.raises(ValueError, match=r"Use Network\.from_mike\(\)"):
            Network.from_epanet(res)

    @pytest.mark.parametrize("suffix", [".res", ".RES"])
    def test_extension_is_case_insensitive(self, tmp_path, suffix):
        with pytest.raises((FileExistsError, FileNotFoundError)):
            Network.from_epanet(tmp_path / f"network{suffix}")


# ---------------------------------------------------------------------------
# EPANET companion files: .inp for reach lengths, .resx for extra quantities
# ---------------------------------------------------------------------------

_EPANET_RES = "./tests/testdata/epanet.res"
_EPANET_RESX = "./tests/testdata/epanet.resx"
_EPANET_INP = "./tests/testdata/epanet.inp"

# The 12 [PIPES] entries; reach "9" is the pump, which carries no length.
_PUMP_REACH = "9"


@requires_mikeio1d
class TestEpanetCompanionInp:
    """`.inp` is the only one of the three files carrying reach lengths."""

    def test_pipe_reaches_get_real_lengths(self):
        network = Network.from_epanet(_EPANET_RES, inp=_EPANET_INP)

        lengths = {r.id: r.length for r in network._reaches.values()}
        assert lengths["10"] == pytest.approx(3209.544)
        assert lengths["110"] == pytest.approx(60.96)

    def test_pump_reach_stays_undefined(self):
        """[PIPES] is the only section with lengths, so pumps keep None."""
        network = Network.from_epanet(_EPANET_RES, inp=_EPANET_INP)

        lengths = {r.id: r.length for r in network._reaches.values()}
        assert lengths[_PUMP_REACH] is None
        assert sum(v is None for v in lengths.values()) == 1

    def test_graph_edges_carry_the_lengths(self):
        network = Network.from_epanet(_EPANET_RES, inp=_EPANET_INP)

        lengths = [d["length"] for *_, d in network.graph.edges(data=True)]
        assert sum(v is not None for v in lengths) == 12

    def test_node_ids_overlapping_reach_ids_are_not_confused(self):
        """Most IDs here name both a node and a reach, e.g. '9', '10', '21'."""
        network = Network.from_epanet(_EPANET_RES, inp=_EPANET_INP)

        assert set(network._reaches) & set(network._alias_map)  # they do overlap
        # Reach "10" is 3209.544 long; node "10" is untouched by the length map.
        assert network._reaches["10"].length == pytest.approx(3209.544)
        node_10 = network.find(node="10")
        assert "Head" in network.to_dataframe()[node_10].columns

    def test_wrong_suffix_is_refused(self):
        with pytest.raises(ValueError, match=r"Expected an EPANET '\.inp'"):
            Network.from_epanet(_EPANET_RES, inp=_EPANET_RESX)

    def test_file_without_a_pipes_section_is_refused(self, tmp_path):
        other = tmp_path / "not-epanet.inp"
        other.write_text("[JUNCTIONS]\n;;Name\n9   1000\n")

        with pytest.raises(ValueError, match=r"no \[PIPES\] section"):
            Network.from_epanet(_EPANET_RES, inp=other)


@requires_mikeio1d
class TestEpanetCompanionResx:
    """`.resx` holds extra results for the network defined in the sibling `.res`."""

    def test_extra_node_quantities_are_merged(self):
        network = Network.from_epanet(_EPANET_RES, resx=_EPANET_RESX)

        assert set(network.quantities) == {
            "Demand",
            "Head",
            "Pressure",
            "WaterQuality",
            "Volume",
            "Volume Percentage",
        }

    def test_only_the_nodes_present_in_the_resx_gain_them(self):
        """The .resx covers the tank and the reservoir, not all eleven nodes."""
        network = Network.from_epanet(_EPANET_RES, resx=_EPANET_RESX)
        df = network.to_dataframe()

        with_volume = {
            node
            for node in df.columns.get_level_values("node").unique()
            if "Volume" in df[node].columns
        }
        # Node IDs are re-indexed to integers, so recall the original labels.
        assert {network.recall(node)["node"] for node in with_volume} == {"2", "9"}

    def test_values_come_through(self):
        network = Network.from_epanet(_EPANET_RES, resx=_EPANET_RESX)

        reservoir = network.find(node="9")
        volume = network.to_dataframe()[(reservoir, "Volume Percentage")]
        assert len(volume) == 25
        assert volume.notna().all()

    def test_selective_loading_still_governs_what_is_read(self):
        network = Network.from_epanet(_EPANET_RES, resx=_EPANET_RESX, nodes=["2"])

        df = network.to_dataframe()
        tank = network.find(node="2")
        assert set(df.columns.get_level_values("node").unique()) == {tank}
        assert "Volume" in df[tank].columns

    def test_both_companions_together(self):
        network = Network.from_epanet(_EPANET_RES, resx=_EPANET_RESX, inp=_EPANET_INP)

        assert "Volume" in network.quantities
        assert network._reaches["10"].length == pytest.approx(3209.544)

    def test_an_open_res1d_object_is_accepted(self):
        from mikeio1d import Res1D

        network = Network.from_epanet(_EPANET_RES, resx=Res1D(_EPANET_RESX))

        assert "Volume" in network.quantities

    def test_wrong_suffix_is_refused(self):
        with pytest.raises(ValueError, match=r"Expected an EPANET '\.resx'"):
            Network.from_epanet(_EPANET_RES, resx=_EPANET_RES)

    def test_a_result_file_of_another_format_is_refused(self):
        from mikeio1d import Res1D

        other = Res1D("./tests/testdata/network.res1d")

        with pytest.raises(ValueError, match=r"Expected an EPANET '\.resx'"):
            Network.from_epanet(_EPANET_RES, resx=other)

    def test_a_companion_from_another_run_is_refused(self, monkeypatch):
        """Merging two runs would line up silently and give a wrong network."""
        from mikeio1d import Res1D

        res = Res1D(_EPANET_RES)
        resx = Res1D(_EPANET_RESX)
        shifted = resx.time_index + pd.Timedelta("1D")

        # Both objects share the Res1D class, so shift only this one instance.
        original = type(resx).time_index.fget
        monkeypatch.setattr(
            type(resx),
            "time_index",
            property(lambda self: shifted if self is resx else original(self)),
        )

        with pytest.raises(ValueError, match="does not share a time axis"):
            Network.from_epanet(res, resx=resx)

    def test_a_companion_naming_an_unknown_node_is_refused(self, monkeypatch):
        """A node the .res has never heard of means these are different models."""
        from mikeio1d import Res1D

        res = Res1D(_EPANET_RES)
        resx = Res1D(_EPANET_RESX)
        strangers = dict(resx.nodes) | {"not_in_the_res": None}

        original = type(resx).nodes.fget
        monkeypatch.setattr(
            type(resx),
            "nodes",
            property(lambda self: strangers if self is resx else original(self)),
        )

        with pytest.raises(ValueError, match="not_in_the_res"):
            Network.from_epanet(res, resx=resx)

    def test_unsupported_type_is_refused(self):
        with pytest.raises(TypeError, match="Expected a str, Path or Res1D object"):
            Network.from_epanet(_EPANET_RES, resx=42)  # type: ignore[arg-type]


class TestReadInp:
    """Minimal .inp reader - see modelskill/model/adapters/_inp.py."""

    def _write(self, tmp_path, text):
        path = tmp_path / "model.inp"
        path.write_text(text)
        return path

    def test_sections_are_keyed_without_brackets_and_upper_cased(self, tmp_path):
        path = self._write(tmp_path, "[Pipes]\n1  a  b  10\n[TANKS]\n2  5\n")

        assert set(read_sections(path)) == {"PIPES", "TANKS"}

    def test_comment_and_blank_lines_are_dropped(self, tmp_path):
        path = self._write(
            tmp_path,
            ";a leading banner\n\n[PIPES]\n"
            ";;ID  Node1  Node2  Length\n"
            ";;--  -----  -----  ------\n"
            "1  a  b  10\n\n",
        )

        assert read_sections(path) == {"PIPES": [["1", "a", "b", "10"]]}

    def test_trailing_comment_is_stripped_from_a_data_row(self, tmp_path):
        path = self._write(tmp_path, "[PIPES]\n1  a  b  10  ; the short one\n")

        assert read_sections(path)["PIPES"] == [["1", "a", "b", "10"]]

    def test_rows_before_any_section_are_ignored(self, tmp_path):
        path = self._write(tmp_path, "stray  row\n[PIPES]\n1  a  b  10\n")

        assert read_sections(path) == {"PIPES": [["1", "a", "b", "10"]]}

    def test_lengths_are_read_from_the_fourth_field(self, tmp_path):
        path = self._write(tmp_path, "[PIPES]\n1  a  b  10.5  300  100\n")

        assert read_pipe_lengths(path) == {"1": 10.5}

    def test_a_short_row_raises_rather_than_dropping_a_length(self, tmp_path):
        path = self._write(tmp_path, "[PIPES]\n1  a  b\n")

        with pytest.raises(ValueError, match="Cannot read a pipe length"):
            read_pipe_lengths(path)

    def test_a_repeated_section_header_accumulates(self, tmp_path):
        path = self._write(
            tmp_path, "[PIPES]\n1  a  b  10\n[TANKS]\n2  5\n[PIPES]\n3  c  d  20\n"
        )

        assert read_pipe_lengths(path) == {"1": 10.0, "3": 20.0}
