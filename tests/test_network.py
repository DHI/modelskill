"""Test network models and observations"""

# ruff: noqa: E402
import sys
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
from modelskill.network import (
    Network,
    BasicNode,
    BasicEdge,
)
from modelskill.obs import NodeObservation
from modelskill.quantity import Quantity


def _make_network(node_ids, time, data, quantity="WaterLevel"):
    nodes = [
        BasicNode(nid, pd.DataFrame({quantity: data[:, i]}, index=time))
        for i, nid in enumerate(node_ids)
    ]
    edges = [
        BasicEdge(f"e{i}", nodes[i], nodes[i + 1], length=100.0)
        for i in range(len(nodes) - 1)
    ]
    return Network(edges)


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
    edges = [BasicEdge("e1", nodes[0], nodes[1], length=100.0)]
    return Network(edges)


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
        obs = NodeObservation(sample_node_data, node=node_id, name="Node_123")

        extracted = nmr.extract(obs)

        assert isinstance(extracted, NodeModelResult)
        assert extracted.node == node_id
        assert len(extracted.time) == 10

    def test_extract_invalid_node(self, sample_network, sample_node_data):
        """Test extraction of a node not present in the network"""
        nmr = NetworkModelResult(sample_network)
        obs = NodeObservation(sample_node_data, node=999, name="Node_999")

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
            TypeError, match="NetworkModelResult only supports NodeObservation"
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
            sample_node_data, node=123, name="Sensor_1", item="WaterLevel"
        )

        assert obs.node == 123
        assert obs.name == "Sensor_1"
        assert len(obs.time) == 10
        assert isinstance(obs.time, pd.DatetimeIndex)

    def test_init_with_series(self, sample_series):
        """Test initialization with pandas Series"""
        obs = NodeObservation(sample_series, node=456, name="Node_456")

        assert obs.node == 456
        assert obs.name == "Node_456"
        assert len(obs.time) == 10

    def test_node_attrs(self, sample_node_data):
        """Test attrs property"""
        attrs = {"source": "test", "version": "1.0"}
        obs = NodeObservation(sample_node_data, node=123, attrs=attrs, weight=2.5)

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
            "tests/testdata/network_sensor_1.csv", node=1, item="water_level@sens1"
        )

        assert obs.node == 1
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
        obs = NodeObservation(sample_node_data, node=node_id, name="Node_123_Obs")

        extracted = nmr.extract(obs)

        assert isinstance(extracted, NodeModelResult)
        assert extracted.node == node_id
        assert extracted.name == "Network_Model"
        assert len(extracted.time) == len(obs.time)

    def test_matching_workflow(self, sample_network, sample_node_data):
        """Test matching workflow with network data"""
        nmr = NetworkModelResult(sample_network, name="Network_Model")
        node_id = sample_network.find(node="123")
        obs = NodeObservation(sample_node_data, node=node_id, name="Node_123_Obs")

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
    network = Network.from_res1d(path_to_file)
    assert network.graph.number_of_nodes() == 259


@pytest.mark.skipif(
    sys.version_info >= (3, 14), reason="mikeio1d requires Python < 3.14"
)
def test_network_subset_copy():
    path_to_file = "./tests/testdata/network.res1d"
    network = Network.from_res1d(path_to_file)
    small_network = network.reduce_around(node=52)
    assert small_network.graph.number_of_nodes() == 22
    assert 52 in small_network.graph.nodes()
