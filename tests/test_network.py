"""Test network models and observations"""

# ruff: noqa: E402
import sys
import pytest

pytest.importorskip("mikeio1d.network")

import pandas as pd
import xarray as xr
import numpy as np
import modelskill as ms
from mikeio1d.network import Network, BasicNode, BasicReach, ReachBreakPoint
from modelskill.model.network import (
    NetworkModelResult,
    NodeModelResult,
)
from modelskill.obs import NodeObservation, ReachObservation
from modelskill.quantity import Quantity


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
def breakpoint_network():
    """A one-reach network whose data sits on a break point, not on the nodes."""
    time = pd.date_range("2010-01-01", periods=10, freq="h")
    np.random.seed(42)
    values = pd.DataFrame({"WaterLevel": np.random.randn(10)}, index=time)
    empty = pd.DataFrame()
    reach = BasicReach(
        "r1",
        BasicNode("start", empty),
        BasicNode("end", empty),
        length=100.0,
        breakpoints=[BreakPoint("r1", 50.0, values)],
    )
    return Network([reach])


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

    def test_quantity_name_survives_to_the_model_result(self, sample_network):
        """The network knows its quantity by name even without a unit."""
        nmr = NetworkModelResult(sample_network)

        assert nmr.quantity.name == "WaterLevel"
        assert nmr.quantity != Quantity.undefined()

    def test_quantity_carries_into_extracted_node(self, sample_network):
        nmr = NetworkModelResult(sample_network)
        obs_data = pd.DataFrame({"sensor": np.zeros(len(nmr.time))}, index=nmr.time)
        extracted = nmr.extract(NodeObservation(obs_data, at="123"))

        assert extracted.quantity.name == "WaterLevel"

    def test_explicit_quantity_wins(self, sample_network):
        given = Quantity(name="Water Level", unit="meter")
        nmr = NetworkModelResult(sample_network, quantity=given)

        assert nmr.quantity == given

    def test_unit_is_used_when_the_data_carries_one(self, sample_network):
        network = sample_network.copy()
        ds = network.to_dataset()
        ds["WaterLevel"].attrs["units"] = "meter"
        network.to_dataset = lambda: ds  # type: ignore[method-assign]

        nmr = NetworkModelResult(network)

        assert nmr.quantity == Quantity(name="WaterLevel", unit="meter")

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
        node_id = "123"
        obs = NodeObservation(sample_node_data, at=node_id, name="Node_123")

        extracted = nmr.extract(obs)

        assert isinstance(extracted, NodeModelResult)
        assert extracted.node == node_id
        assert len(extracted.time) == 10

    def test_extract_invalid_node(self, sample_network, sample_node_data):
        """Test extraction of a node not present in the network"""
        nmr = NetworkModelResult(sample_network)
        obs = NodeObservation(sample_node_data, at="999", name="Node_999")

        with pytest.raises(ValueError, match="not found"):
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
            sample_node_data, at="123", name="Sensor_1", item="WaterLevel"
        )

        assert obs.at == "123"
        assert obs.name == "Sensor_1"
        assert len(obs.time) == 10
        assert isinstance(obs.time, pd.DatetimeIndex)

    def test_init_with_series(self, sample_series):
        """Test initialization with pandas Series"""
        obs = NodeObservation(sample_series, at="456", name="Node_456")

        assert obs.at == "456"
        assert obs.name == "Node_456"
        assert len(obs.time) == 10

    def test_node_attrs(self, sample_node_data):
        """Test attrs property"""
        attrs = {"source": "test", "version": "1.0"}
        obs = NodeObservation(sample_node_data, at="123", attrs=attrs, weight=2.5)

        assert obs.attrs["source"] == "test"
        assert obs.attrs["version"] == "1.0"
        assert obs.weight == 2.5
        assert obs.quantity == Quantity.undefined()

    def test_multiple_nodes_returns_list_of_observations(self, multi_data):
        """Test that from_multiple returns a list of NodeObservation objects"""
        obs_list = NodeObservation.from_multiple(
            data=multi_data,
            nodes={"123": "station_0", "456": "station_1", "789": "station_2"},
        )

        assert len(obs_list) == 3
        assert all(isinstance(obs, NodeObservation) for obs in obs_list)

    def test_node_ids_are_assigned_correctly(self, multi_data):
        obs_list = NodeObservation.from_multiple(
            data=multi_data,
            nodes={"123": "station_0", "456": "station_1", "789": "station_2"},
        )

        assert obs_list[0].node == "123"
        assert obs_list[1].node == "456"
        assert obs_list[2].node == "789"

    def test_names_derived_from_column_names(self, multi_data):
        obs_list = NodeObservation.from_multiple(
            data=multi_data,
            nodes={"123": "station_0", "456": "station_1", "789": "station_2"},
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
            data=ds, nodes={"123": "station_0", "456": "station_1"}
        )

        assert len(obs_list) == 2
        assert obs_list[0].node == "123"
        assert obs_list[1].node == "456"

    def test_nodes_must_be_dict(self, multi_data):
        with pytest.raises(TypeError, match="'nodes' must be a dict"):
            NodeObservation.from_multiple(data=multi_data, nodes=123)

    def test_attrs_propagated_to_all_observations(self, multi_data):
        attrs = {"source": "sensor_array", "version": 2}
        obs_list = NodeObservation.from_multiple(
            data=multi_data,
            nodes={"1": "station_0", "2": "station_1", "3": "station_2"},
            attrs=attrs,
        )

        for obs in obs_list:
            assert obs.attrs["source"] == "sensor_array"
            assert obs.attrs["version"] == 2

    def test_init_from_csv(self):
        obs = NodeObservation(
            "tests/testdata/network_sensor_1.csv", at="1", item="water_level@sens1"
        )

        assert obs.at == "1"
        assert len(obs.time) == 110
        assert isinstance(obs.time, pd.DatetimeIndex)

    def test_from_multiple_csvs_via_dict(self):
        obs_list = NodeObservation.from_multiple(
            nodes={
                "1": "tests/testdata/network_sensor_1.csv",
                "2": "tests/testdata/network_sensor_2.csv",
                "3": "tests/testdata/network_sensor_3.csv",
            }
        )

        assert len(obs_list) == 3
        assert all(isinstance(obs, NodeObservation) for obs in obs_list)
        assert obs_list[0].node == "1"
        assert obs_list[1].node == "2"
        assert obs_list[2].node == "3"
        for obs in obs_list:
            assert len(obs.time) > 0

    def test_nodes_dict_maps_node_to_item(self, multi_data):
        obs_list = NodeObservation.from_multiple(
            data=multi_data, nodes={"123": "station_0", "456": "station_1"}
        )

        assert len(obs_list) == 2
        assert obs_list[0].node == "123"
        assert obs_list[1].node == "456"
        assert obs_list[0].name == "station_0"
        assert obs_list[1].name == "station_1"

    def test_nodes_none_raises(self, multi_data):
        with pytest.raises(ValueError, match="'nodes' argument is required"):
            NodeObservation.from_multiple(data=multi_data, nodes=None)

    def test_single_node_dict(self, sample_node_data):
        obs_list = NodeObservation.from_multiple(
            data=sample_node_data, nodes={"123": "WaterLevel"}
        )

        assert len(obs_list) == 1
        assert isinstance(obs_list[0], NodeObservation)
        assert obs_list[0].node == "123"

    def test_nodes_keys_accept_aliases(self, multi_data):
        obs_list = NodeObservation.from_multiple(
            data=multi_data, nodes={"node_A": "station_0", "node_B": "station_1"}
        )

        assert [obs.at for obs in obs_list] == ["node_A", "node_B"]

    def test_nodes_keys_accept_breakpoints(self, multi_data):
        obs_list = NodeObservation.from_multiple(
            data=multi_data,
            nodes={("reach_1", 24.5): "station_0", ("reach_1", 50.0): "station_1"},
        )

        assert [obs.at for obs in obs_list] == [("reach_1", 24.5), ("reach_1", 50.0)]


class TestReachObservationFromMultiple:
    @pytest.fixture
    def multi_data(self, sample_node_data):
        return pd.DataFrame(
            {
                "station_0": sample_node_data["WaterLevel"].values,
                "station_1": sample_node_data["WaterLevel"].values + 0.1,
            },
            index=sample_node_data.index,
        )

    def test_returns_list_of_reach_observations(self, multi_data):
        obs_list = ReachObservation.from_multiple(
            data=multi_data, reaches={"reach_1": "station_0", "reach_2": "station_1"}
        )

        assert len(obs_list) == 2
        assert all(isinstance(obs, ReachObservation) for obs in obs_list)
        assert [obs.reach for obs in obs_list] == ["reach_1", "reach_2"]
        assert [obs.name for obs in obs_list] == ["station_0", "station_1"]

    def test_separate_data_sources(self):
        obs_list = ReachObservation.from_multiple(
            reaches={
                "reach_1": "tests/testdata/network_sensor_1.csv",
                "reach_2": "tests/testdata/network_sensor_2.csv",
            }
        )

        assert [obs.reach for obs in obs_list] == ["reach_1", "reach_2"]
        assert all(len(obs.time) > 0 for obs in obs_list)

    def test_attrs_propagated(self, multi_data):
        obs_list = ReachObservation.from_multiple(
            data=multi_data,
            reaches={"reach_1": "station_0"},
            attrs={"source": "sensor_array"},
        )

        assert obs_list[0].attrs["source"] == "sensor_array"

    def test_reaches_none_raises(self, multi_data):
        with pytest.raises(ValueError, match="'reaches' argument is required"):
            ReachObservation.from_multiple(data=multi_data, reaches=None)

    def test_reaches_must_be_dict(self, multi_data):
        with pytest.raises(TypeError, match="'reaches' must be a dict"):
            ReachObservation.from_multiple(data=multi_data, reaches="reach_1")


class TestNodeModelResult:
    """Test NodeModelResult class"""

    @pytest.mark.parametrize("fixture_name", ["sample_node_data", "sample_series"])
    def test_init_(self, request, fixture_name):
        """Test initialization with pandas DataFrame"""
        data = request.getfixturevalue(fixture_name)
        nmr = NodeModelResult(data, node="123", name="Node_123_Model")

        assert nmr.node == "123"
        assert nmr.name == "Node_123_Model"
        assert len(nmr.time) == 10


class TestNetworkIntegration:
    """Test integration between network models and observations"""

    def test_network_to_node_extraction(self, sample_network, sample_node_data):
        """Test complete workflow from network model to node extraction"""
        nmr = NetworkModelResult(sample_network, name="Network_Model")
        node_id = "123"
        obs = NodeObservation(sample_node_data, at=node_id, name="Node_123_Obs")

        extracted = nmr.extract(obs)

        assert isinstance(extracted, NodeModelResult)
        assert extracted.node == node_id
        assert extracted.name == "Network_Model"
        assert len(extracted.time) == len(obs.time)

    def test_matching_workflow(self, sample_network, sample_node_data):
        """Test matching workflow with network data"""
        nmr = NetworkModelResult(sample_network, name="Network_Model")
        node_id = "123"
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

        node_0 = "123"
        node_1 = "456"
        node_2 = "789"

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
def test_a_model_result_can_be_built_from_a_result_file():
    mr = NetworkModelResult("./tests/testdata/network.res1d", item="WaterLevel")

    assert mr.quantity.name == "WaterLevel"
    assert len(mr.nodes) > 0


@pytest.mark.skipif(
    sys.version_info >= (3, 14), reason="mikeio1d requires Python < 3.14"
)
def test_extract_reach_observation_happy_path(sample_node_data):
    path_to_file = "./tests/testdata/network.res1d"
    network = Network.open(path_to_file)
    nmr = NetworkModelResult(network, item="Discharge", name="network_model")
    obs_data = sample_node_data.rename(columns={"WaterLevel": "Discharge"})
    obs = ms.ReachObservation(obs_data, reach="100l1", item="Discharge")

    extracted = nmr.extract(obs)

    assert isinstance(extracted, NodeModelResult)
    assert extracted.name == "network_model"
    reach, _ = extracted.node
    assert reach == "100l1"


@pytest.mark.skipif(
    sys.version_info >= (3, 14), reason="mikeio1d requires Python < 3.14"
)
def test_extract_reach_observation_non_equivalent_breakpoints_raises(sample_node_data):
    path_to_file = "./tests/testdata/network.res1d"
    network = Network.open(path_to_file)
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
    network = Network.open(path_to_file, reaches=[])
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
    network = Network.open(path_to_file)
    nmr = NetworkModelResult(network, item="Discharge")
    obs_data = sample_node_data.rename(columns={"WaterLevel": "Discharge"})
    on_reach = set(nmr.data.node.values[nmr.data["reach"].values == "100l1"])
    nmr.data = nmr.data.sel(
        node=[int(node) for node in nmr.data.node.values if node not in on_reach]
    )

    obs = ms.ReachObservation(obs_data, reach="100l1", item="Discharge")

    with pytest.raises(ValueError, match="matching breakpoint nodes are missing"):
        nmr.extract(obs)


@pytest.mark.skipif(
    sys.version_info >= (3, 14), reason="mikeio1d requires Python < 3.14"
)
def test_extract_breakpoint_without_data_for_the_quantity_raises_valueerror(
    sample_node_data,
):
    """MIKE 1D stores WaterLevel and Discharge at alternating grid points, so a
    breakpoint that exists can still hold nothing for the selected quantity."""
    network = Network.open("./tests/testdata/network.res1d")
    nmr = NetworkModelResult(network, item="WaterLevel")
    obs = ms.NodeObservation(sample_node_data, at=("94l1", 21.285), item="WaterLevel")

    with pytest.raises(ValueError, match="no data for quantity 'WaterLevel'"):
        nmr.extract(obs)


class TestNodeObservationAliases:
    """NodeObservation accepts a node name or a (reach, distance) tuple."""

    @pytest.mark.parametrize("at", [42, np.int64(42)])
    def test_an_integer_is_refused(self, sample_node_data, at):
        with pytest.raises(TypeError, match="not an integer"):
            NodeObservation(sample_node_data, at=at)

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
    """extract() resolves a node name or a (reach, distance) pair to a location."""

    def test_the_network_is_kept_as_given(self, sample_network):
        nmr = NetworkModelResult(sample_network)

        assert nmr.network is sample_network

    def test_extract_with_string_alias(self, sample_network, sample_node_data):
        nmr = NetworkModelResult(sample_network)
        obs = NodeObservation(sample_node_data, at="123", name="Node_123")

        extracted = nmr.extract(obs)

        assert isinstance(extracted, NodeModelResult)
        assert extracted.node == "123"

    def test_extract_string_alias_wrong_key_raises(
        self, sample_network, sample_node_data
    ):
        nmr = NetworkModelResult(sample_network)
        obs = NodeObservation(sample_node_data, at="nonexistent_node")

        with pytest.raises(ValueError, match="not found"):
            nmr.extract(obs)

    def test_a_failed_lookup_names_the_near_misses(
        self, sample_network, sample_node_data
    ):
        nmr = NetworkModelResult(sample_network)
        obs = NodeObservation(sample_node_data, at="124")

        with pytest.raises(ValueError, match="123"):
            nmr.extract(obs)

    def test_extract_with_tuple_breakpoint(self, breakpoint_network, sample_node_data):
        nmr = NetworkModelResult(breakpoint_network)
        obs = NodeObservation(sample_node_data, at=("r1", 50.0))

        extracted = nmr.extract(obs)

        assert extracted.node == ("r1", 50.0)

    def test_extract_with_tuple_breakpoint_tolerance(
        self, breakpoint_network, sample_node_data
    ):
        nmr = NetworkModelResult(breakpoint_network)
        obs = NodeObservation(sample_node_data, at=("r1", 50.0 + 5e-4))

        extracted = nmr.extract(obs)

        # The distance recorded is the network's own, not the one typed.
        assert extracted.node == ("r1", 50.0)

    def test_extract_with_tuple_breakpoint_outside_tolerance_raises(
        self, breakpoint_network, sample_node_data
    ):
        nmr = NetworkModelResult(breakpoint_network)
        obs = NodeObservation(sample_node_data, at=("r1", 50.0 + 2e-3))

        with pytest.raises(ValueError, match="not found"):
            nmr.extract(obs)

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


# ======================== location identity ========================


class TestLocationIdentity:
    """A network timeseries is identified by the name its network gave it."""

    def test_breakpoint_observation_converts_to_a_dataframe(self, sample_node_data):
        obs = ms.NodeObservation(sample_node_data, at=("r1", 24.5), item="WaterLevel")

        df = obs.to_dataframe()

        assert list(df.columns) == ["WaterLevel"]
        assert len(df) == len(sample_node_data)

    def test_reach_observation_converts_to_a_dataframe(self, sample_node_data):
        obs = ms.ReachObservation(sample_node_data, reach="r1", item="WaterLevel")

        df = obs.to_dataframe()

        assert list(df.columns) == ["WaterLevel"]

    def test_a_named_node_survives_trimming(self, sample_network, sample_node_data):
        nmr = NetworkModelResult(sample_network)
        extracted = nmr.extract(ms.NodeObservation(sample_node_data, at="123"))

        trimmed = extracted.trim(
            start_time=extracted.time[1], end_time=extracted.time[-1]
        )

        assert trimmed.node == extracted.node
        assert len(trimmed) == len(extracted) - 1
