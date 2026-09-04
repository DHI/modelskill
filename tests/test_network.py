"""Test network models and observations"""

# ruff: noqa: E402
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
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
    _merge_extra_quantities,
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
    _Companion,
    _find_epanet_companions,
    _network_from_path,
    _repair_mis_decoded,
    _rekey_by_main_file,
)
from modelskill.obs import NodeObservation, ReachObservation
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
    # 259 topology nodes, plus 2 reach-end breakpoints per reach (118 reaches)
    # now that they are promoted from boundary metadata to real breakpoints.
    assert network.graph.number_of_nodes() == 259 + 2 * 118


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
    reads were discarded work (gh #679). Reach-end gridpoints are now ordinary
    breakpoints, so - like every other breakpoint - reaches=[] skips reading
    them entirely; there is one loading rule, not a special case for the ones
    that used to be boundary metadata.
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

    assert reads["ResultNode"] == n_unique_nodes
    assert "ResultGridPoint" not in reads


@pytest.mark.skipif(
    sys.version_info >= (3, 14), reason="mikeio1d requires Python < 3.14"
)
def test_from_mike_reads_each_reach_endpoint_gridpoint_when_loaded(monkeypatch):
    """The default (reaches=None) loads every reach's endpoint gridpoints too."""
    from modelskill.model.adapters import _res1d

    reads: dict[str, int] = {}
    unpatched = _res1d._simplify_colnames

    def counting_simplify_colnames(location, *args, **kwargs):
        name = type(location).__name__
        reads[name] = reads.get(name, 0) + 1
        return unpatched(location, *args, **kwargs)

    monkeypatch.setattr(_res1d, "_simplify_colnames", counting_simplify_colnames)

    path_to_file = "./tests/testdata/network.res1d"
    network = Network.from_mike(path_to_file, nodes=[])

    n_breakpoints = sum(r.n_breakpoints for r in network._reaches.values())

    assert reads["ResultGridPoint"] == n_breakpoints


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


class TestBoundaryEdges:
    """A breakpoint coincident with a reach's own start/end node is tagged.

    This is the mechanism that replaced the removed `boundary` attribute:
    reach-end data lives on an ordinary breakpoint, connected to its node by
    an edge tagged boundary=True and clamped to an exact 0.0.
    """

    def test_edge_with_no_breakpoints_is_tagged_non_boundary(self):
        a, b = _two_node_pair()

        network = Network([BasicReach("r1", a, b, length=100.0)])

        ((_, _, data),) = network.graph.edges(data=True)
        assert data["boundary"] is False

    def test_breakpoint_at_reach_start_is_tagged_boundary(self):
        a, b = _two_node_pair()
        breakpoints = [_StubBreakPoint("r1", 0.0), _StubBreakPoint("r1", 40.0)]

        network = Network([BasicReach("r1", a, b, 100.0, breakpoints)])

        edges = {frozenset((u, v)): d for u, v, d in network.graph.edges(data=True)}
        start_bp_key = frozenset(
            (network.find(node="a"), network.find(reach="r1", distance=0.0))
        )
        interior_key = frozenset(
            (
                network.find(reach="r1", distance=0.0),
                network.find(reach="r1", distance=40.0),
            )
        )

        assert edges[start_bp_key] == {"length": 0.0, "boundary": True}
        assert edges[interior_key] == {"length": 40.0, "boundary": False}

    def test_breakpoint_at_reach_end_is_tagged_boundary(self):
        a, b = _two_node_pair()
        breakpoints = [_StubBreakPoint("r1", 60.0), _StubBreakPoint("r1", 100.0)]

        network = Network([BasicReach("r1", a, b, 100.0, breakpoints)])

        end_bp_key = frozenset(
            (network.find(node="b"), network.find(reach="r1", distance=100.0))
        )

        assert network.graph.edges[tuple(end_bp_key)]["boundary"] is True
        assert network.graph.edges[tuple(end_bp_key)]["length"] == 0.0

    def test_end_breakpoint_length_is_clamped_despite_floating_point_noise(self):
        """Clamp near-zero noise between two independently-sourced values.

        reach.length and the breakpoint's own distance come from independent
        sources and are not guaranteed to be bit-identical; the near-zero
        difference must be clamped to an exact 0.0, not left as tiny noise
        that could turn negative and break weighted graph algorithms.
        """
        a, b = _two_node_pair()
        noisy_length = 100.0 + 1e-9
        breakpoints = [_StubBreakPoint("r1", 100.0)]

        network = Network([BasicReach("r1", a, b, noisy_length, breakpoints)])

        end_bp_key = frozenset(
            (network.find(node="b"), network.find(reach="r1", distance=100.0))
        )
        data = network.graph.edges[tuple(end_bp_key)]

        assert data["boundary"] is True
        assert data["length"] == 0.0

    def test_interior_breakpoint_is_never_tagged_boundary_even_near_zero_length(self):
        """Only edges to a reach's own endpoints are ever boundary edges.

        A genuinely tiny reach still tags interior segments correctly.
        """
        a, b = _two_node_pair()
        breakpoints = [_StubBreakPoint("r1", 0.0005)]

        network = Network([BasicReach("r1", a, b, 0.001, breakpoints)])

        bp_alias = network.find(reach="r1", distance=0.0005)
        start_key = frozenset((network.find(node="a"), bp_alias))
        end_key = frozenset((network.find(node="b"), bp_alias))

        # Both are boundary edges here since the reach itself is shorter than
        # the chainage tolerance - not a contradiction, just a degenerate case.
        assert network.graph.edges[tuple(start_key)]["boundary"] is True
        assert network.graph.edges[tuple(end_key)]["boundary"] is True


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


class TestMergeExtraQuantities:
    """_merge_extra_quantities serves both node and reach loading.

    Its error message is format-neutral ("Location", not "Node") since it
    no longer knows whether location_id names a node or a reach.
    """

    def test_collision_error_names_the_location(self):
        base = pd.DataFrame({"Flow": [1.0]})
        extra = pd.DataFrame({"Flow": [2.0]})

        with pytest.raises(ValueError, match=r"Location '42'"):
            _merge_extra_quantities(base, extra, location_id="42")

    def test_collision_error_reports_the_overlapping_columns(self):
        base = pd.DataFrame({"Flow": [1.0], "Velocity": [1.0]})
        extra = pd.DataFrame({"Flow": [2.0]})

        with pytest.raises(ValueError, match=r"\['Flow'\]"):
            _merge_extra_quantities(base, extra, location_id="9")


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

        # 259 topology nodes, plus 2 reach-end breakpoints per reach (118
        # reaches), now that they are promoted from boundary metadata to
        # real breakpoints.
        assert network.graph.number_of_nodes() == 259 + 2 * 118

    def test_res11(self):
        """MIKE 11 keeps its data on gridpoints, so its nodes are empty."""
        network = Network.from_mike("./tests/testdata/network_cali.res11")

        assert len(network._reaches) == 3
        # 71 topology nodes, plus 2 reach-end breakpoints per reach (3 reaches).
        assert network.graph.number_of_nodes() == 71 + 2 * 3
        assert set(network.quantities) == {"Discharge", "Water Level"}
        # +2 per reach: the first/last gridpoint is now a breakpoint too.
        assert [r.n_breakpoints for r in network._reaches.values()] == [25, 23, 25]

    def test_reach_end_waterlevel_differs_from_node_waterlevel(self):
        """The reach-end gridpoint is a distinct h-point, not the node's own value.

        This is the data #599/#680 asked to make reachable: it was previously
        discarded (never read) or hidden in the now-removed `boundary` dict.
        """
        network = Network.from_mike("./tests/testdata/network.res1d")
        reach = network._reaches["100l1"]

        node_waterlevel = reach.start.data["WaterLevel"]
        breakpoint_waterlevel = reach.breakpoints[0].data["WaterLevel"]

        assert not np.allclose(node_waterlevel.values, breakpoint_waterlevel.values)

    def test_find_reach_end_breakpoint_distinct_from_node(self):
        """distance="start" resolves to the node; a number resolves elsewhere.

        distance="start"/"end" still resolve to the node; a numeric distance
        resolves to the new, distinct reach-end breakpoint at the same location.
        """
        network = Network.from_mike("./tests/testdata/network.res1d")
        reach = network._reaches["100l1"]

        node_alias = network.find(node=reach.start.id)
        start_by_keyword = network.find(reach=reach.id, distance="start")
        start_by_distance = network.find(reach=reach.id, distance=0.0)

        assert start_by_keyword == node_alias
        assert start_by_distance != node_alias
        assert network.recall(start_by_distance) == {"reach": reach.id, "distance": 0.0}

    def test_res11_boundary_edges_are_exactly_zero(self):
        """Reach-end breakpoints are coincident with their node, tagged and clamped."""
        network = Network.from_mike("./tests/testdata/network_cali.res11")

        edges = list(network.graph.edges(data=True))
        boundary_lengths = {d["length"] for *_, d in edges if d["boundary"]}
        non_boundary_lengths = [d["length"] for *_, d in edges if not d["boundary"]]

        assert boundary_lengths == {0.0}
        assert all(length > 0 for length in non_boundary_lengths)

    def test_open_res1d_object(self):
        from mikeio1d import Res1D

        res = Res1D("./tests/testdata/network.res1d")

        network = Network.from_mike(res, nodes=[], reaches=[])

        # Topology is always fully constructed regardless of nodes=[]/reaches=[].
        assert network.graph.number_of_nodes() == 259 + 2 * 118

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

        # 11 topology nodes, plus 2 duplicated breakpoints per reach (13 reaches).
        assert network.graph.number_of_nodes() == 11 + 2 * 13
        assert len(network._reaches) == 13
        assert set(network.quantities) == {
            "Demand",
            "Head",
            "Pressure",
            "WaterQuality",
            "Flow",
            "Velocity",
            "HeadlossPer1000Unit",
            "AvgWaterQuality",
            "StatusCode",
            "Setting",
            "ReactorRate",
            "FrictionFactor",
        }
        assert not network.to_dataframe().empty

    def test_link_node_reaches_get_two_breakpoints(self):
        """The single synthetic gridpoint is duplicated to both ends of the reach.

        Without inp=, reach.length is unknown, so only the leading edge (at
        the reach's start, distance 0.0) is real; the connecting edge and the
        trailing edge are both undefined - documented in the docstring.
        """
        network = Network.from_epanet("./tests/testdata/epanet.res")

        assert all(r.n_breakpoints == 2 for r in network._reaches.values())

        for reach in network._reaches.values():
            assert reach.breakpoints[0].distance == 0.0
            assert reach.breakpoints[1].distance is None

        lengths = [d["length"] for *_, d in network.graph.edges(data=True)]
        boundary_lengths = [
            d["length"] for *_, d in network.graph.edges(data=True) if d["boundary"]
        ]
        assert len(lengths) == 3 * 13
        assert boundary_lengths == [0.0] * 13
        assert sum(length is None for length in lengths) == 2 * 13

    def test_reach_observation_matches_via_the_duplicated_breakpoint(
        self, sample_node_data
    ):
        """A reach's Flow quantity is now reachable through its breakpoints."""
        network = Network.from_epanet("./tests/testdata/epanet.res")
        nmr = NetworkModelResult(network, item="Flow", name="epanet_model")
        obs_data = sample_node_data.rename(columns={"WaterLevel": "Flow"})
        obs = ms.ReachObservation(obs_data, reach="10", item="Flow")

        extracted = nmr.extract(obs)

        assert isinstance(extracted, NodeModelResult)
        assert extracted.name == "epanet_model"

    def test_find_reach_end_breakpoint_with_unknown_length_is_not_addressable(self):
        """A None-distance breakpoint can't be found by a numeric distance.

        This must raise the normal KeyError, not crash inside find()'s
        tolerance-matching loop just because some breakpoint's distance is
        unknown.
        """
        network = Network.from_epanet("./tests/testdata/epanet.res")
        assert network._reaches["10"].breakpoints  # guard against a vacuous check

        with pytest.raises(KeyError):
            network.find(reach="10", distance=100.0)

    def test_resolve_alias_tolerance_match_skips_unknown_distance_breakpoint(self):
        """NetworkModelResult._resolve_alias must not crash on a None-distance sibling.

        Reach "10" has two breakpoints: one at distance 0.0, one at an
        unknown distance (None). Resolving a nearby-but-not-exact numeric
        distance walks the alias map's tolerance-matching loop, which must
        skip the None-distance key rather than compute abs(None - distance).
        """
        network = Network.from_epanet("./tests/testdata/epanet.res")
        nmr = NetworkModelResult(network, item="Flow", name="epanet_model")

        exact_id = nmr._resolve_alias(("10", 0.0))
        nearby_id = nmr._resolve_alias(("10", 0.0005))

        assert nearby_id == exact_id

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

        # 12 pipes x 3 real edges (leading 0.0, the pipe's full length, trailing
        # 0.0) + the pump's 1 real edge (leading 0.0; its trailing/connecting
        # edges stay None since its own length is unknown).
        lengths = [d["length"] for *_, d in network.graph.edges(data=True)]
        assert len(lengths) == 13 * 3
        assert sum(v is not None for v in lengths) == 12 * 3 + 1

    def test_pipe_breakpoints_sit_at_both_ends(self):
        """A known-length reach's duplicate sits at the far end, not the middle.

        Its two edges (start->end breakpoint, end breakpoint->end node) are
        both tagged boundary=True (0.0), and the connecting edge in between
        carries the pipe's full real length.
        """
        network = Network.from_epanet(_EPANET_RES, inp=_EPANET_INP)
        pipe = network._reaches["10"]

        assert pipe.breakpoints[0].distance == 0.0
        assert pipe.breakpoints[1].distance == pytest.approx(3209.544)

        start_alias = network.find(reach="10", distance="start")
        far_alias = network.find(reach="10", distance=3209.544)
        end_alias = network.find(reach="10", distance="end")
        assert len({start_alias, far_alias, end_alias}) == 3

    def test_pump_breakpoint_stays_at_start_only(self):
        """The pump's unknown length means only its leading breakpoint is real."""
        network = Network.from_epanet(_EPANET_RES, inp=_EPANET_INP)
        pump = network._reaches[_PUMP_REACH]

        assert pump.breakpoints[0].distance == 0.0
        assert pump.breakpoints[1].distance is None

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
            "Flow",
            "Velocity",
            "HeadlossPer1000Unit",
            "AvgWaterQuality",
            "StatusCode",
            "Setting",
            "ReactorRate",
            "FrictionFactor",
            "Pump efficiency",
            "Pump energy costs",
            "Pump energy",
        }

    def test_extra_reach_quantities_are_merged_onto_the_pump(self):
        """resx's reach-level quantities merge onto the reach's breakpoints.

        Pump energy, efficiency, and cost, alongside its own Flow/Velocity/etc.
        """
        network = Network.from_epanet(_EPANET_RES, resx=_EPANET_RESX)
        pump = network._reaches[_PUMP_REACH]

        assert pump.breakpoints
        for breakpoint in pump.breakpoints:
            assert "Pump energy" in breakpoint.data.columns
            assert "Flow" in breakpoint.data.columns

    def test_only_the_pump_reach_gains_resx_reach_quantities(self):
        """The .resx covers only the pump reach, not the other twelve."""
        network = Network.from_epanet(_EPANET_RES, resx=_EPANET_RESX)

        non_pump_reaches = {
            k: v for k, v in network._reaches.items() if k != _PUMP_REACH
        }
        assert non_pump_reaches
        assert all(reach.breakpoints for reach in non_pump_reaches.values())
        for reach in non_pump_reaches.values():
            for breakpoint in reach.breakpoints:
                assert "Pump energy" not in breakpoint.data.columns

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
        # reaches=[] isolates this to node selection: without it, every
        # reach's breakpoints would also get real data now (reaches=None
        # loads all of them), adding unexpected columns.
        network = Network.from_epanet(
            _EPANET_RES, resx=_EPANET_RESX, nodes=["2"], reaches=[]
        )

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


class TestCompanionNameEncoding:
    """mikeio1d reads '.res' names as UTF-8 and '.resx' names as CP1252.

    A node called 'ØST' in one file is 'Ã˜ST' in the other, so the same model
    looks like two - and the four Danish tank names in a real MIKE+ EPANET
    model were the ones that surfaced it.
    """

    def test_a_mis_decoded_name_is_recovered(self):
        assert _repair_mis_decoded("Ã˜ST") == ["ØST"]
        assert _repair_mis_decoded("VandvÃ¦rk_Vest") == ["Vandværk_Vest"]

    def test_an_ascii_name_has_nothing_to_recover(self):
        assert _repair_mis_decoded("Junction_1") == []

    def test_a_name_that_no_encoding_explains_is_left_alone(self):
        """'ØST' is already correct: its bytes are not valid UTF-8 on their own."""
        assert _repair_mis_decoded("ØST") == []

    def test_a_companion_location_is_keyed_by_the_main_files_name(self):
        rekeyed = _rekey_by_main_file({"Ã˜ST": "data"}, {"ØST", "Junction_1"})

        assert rekeyed == {"ØST": "data"}

    def test_a_matching_name_is_untouched(self):
        rekeyed = _rekey_by_main_file({"Junction_1": "data"}, {"Junction_1"})

        assert rekeyed == {"Junction_1": "data"}

    def test_a_name_from_another_model_keeps_its_own_spelling(self):
        """Otherwise a genuinely different companion would slip past validation."""
        rekeyed = _rekey_by_main_file({"Ã˜ST": "data"}, {"Junction_1"})

        assert rekeyed == {"Ã˜ST": "data"}

    def test_a_companion_rekeys_both_nodes_and_reaches(self):
        res = SimpleNamespace(nodes={"ØST": 1}, reaches={"Vandværk_Vest": 2})
        extra = SimpleNamespace(nodes={"Ã˜ST": 3}, reaches={"VandvÃ¦rk_Vest": 4})

        companion = _Companion(res, extra)

        assert companion.nodes == {"ØST": 3}
        assert companion.reaches == {"Vandværk_Vest": 4}


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


# ---------------------------------------------------------------------------
# Building a Network from a bare path
# ---------------------------------------------------------------------------


def _copy_epanet(tmp_path, *suffixes):
    """Copy the EPANET fixture set into tmp_path, keeping only some companions."""
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    for suffix in (".res", *suffixes):
        shutil.copy(f"./tests/testdata/epanet{suffix}", tmp_path / f"model{suffix}")
    return tmp_path / "model.res"


class TestFindEpanetCompanions:
    """A companion is the file sharing the result file's folder and stem."""

    def test_both_companions_are_found(self, tmp_path):
        res = _copy_epanet(tmp_path, ".resx", ".inp")

        assert _find_epanet_companions(res) == (
            tmp_path / "model.resx",
            tmp_path / "model.inp",
        )

    def test_a_missing_companion_is_none(self, tmp_path):
        res = _copy_epanet(tmp_path, ".inp")

        assert _find_epanet_companions(res) == (None, tmp_path / "model.inp")

    def test_a_lone_result_file_has_neither(self, tmp_path):
        res = _copy_epanet(tmp_path)

        assert _find_epanet_companions(res) == (None, None)

    def test_a_differently_named_sibling_is_not_a_companion(self, tmp_path):
        res = _copy_epanet(tmp_path)
        shutil.copy("./tests/testdata/epanet.inp", tmp_path / "other.inp")

        assert _find_epanet_companions(res) == (None, None)


@requires_mikeio1d
class TestNetworkFromPath:
    """The extension names the product, so it picks the constructor."""

    def test_res1d_goes_to_from_mike(self):
        network = _network_from_path("./tests/testdata/network.res1d")

        assert network.graph.number_of_nodes() == 259 + 2 * 118

    def test_res11_goes_to_from_mike(self):
        network = _network_from_path(Path("./tests/testdata/network_cali.res11"))

        assert set(network.quantities) == {"Discharge", "Water Level"}

    def test_res_goes_to_from_epanet_with_both_companions(self, tmp_path):
        res = _copy_epanet(tmp_path, ".resx", ".inp")

        network = _network_from_path(res)

        # Lengths come from the .inp; Volume comes from the .resx.
        assert network._reaches["10"].length == pytest.approx(3209.544)
        assert "Volume" in network.quantities

    def test_res_without_companions_still_loads(self, tmp_path):
        res = _copy_epanet(tmp_path)

        network = _network_from_path(res)

        assert network._reaches["10"].length is None
        assert "Volume" not in network.quantities

    def test_each_companion_is_found_on_its_own(self, tmp_path):
        with_inp = _network_from_path(_copy_epanet(tmp_path / "a", ".inp"))
        with_resx = _network_from_path(_copy_epanet(tmp_path / "b", ".resx"))

        assert with_inp._reaches["10"].length == pytest.approx(3209.544)
        assert "Volume" not in with_inp.quantities
        assert with_resx._reaches["10"].length is None
        assert "Volume" in with_resx.quantities

    def test_an_unreadable_format_keeps_its_reason(self):
        with pytest.raises(NotImplementedError, match="companion '.inp' input file"):
            _network_from_path("./tests/testdata/swmm.out")

    def test_an_unknown_extension_is_refused(self):
        with pytest.raises(NotImplementedError, match="Unsupported file extension"):
            _network_from_path("./tests/testdata/obs.dfs0")

    def test_a_failing_companion_names_the_file_it_picked_up(
        self, tmp_path, monkeypatch
    ):
        """A companion the caller never asked for must be named when it fails."""
        res = _copy_epanet(tmp_path, ".resx")
        monkeypatch.setattr(
            Network,
            "_open_companion_result",
            staticmethod(lambda *a, **kw: (_ for _ in ()).throw(ValueError("boom"))),
        )

        with pytest.raises(ValueError, match="model.resx") as excinfo:
            _network_from_path(res)

        assert "boom" in str(excinfo.value)
        assert "Network.from_epanet" in str(excinfo.value)

    def test_a_failure_without_companions_is_left_alone(self, tmp_path, monkeypatch):
        res = _copy_epanet(tmp_path)
        monkeypatch.setattr(
            Network,
            "_load_res1d_network",
            staticmethod(lambda *a, **kw: (_ for _ in ()).throw(ValueError("boom"))),
        )

        with pytest.raises(ValueError, match="^boom$"):
            _network_from_path(res)


@requires_mikeio1d
class TestNetworkModelResultFromPath:
    """A path spares the caller a separate Network import and load."""

    _RES1D = "./tests/testdata/network.res1d"

    def test_a_path_gives_the_same_result_as_a_loaded_network(self):
        from_path = NetworkModelResult(self._RES1D, item="WaterLevel")
        from_network = NetworkModelResult(
            Network.from_mike(self._RES1D), item="WaterLevel"
        )

        assert from_path.name == from_network.name
        assert from_path.quantity == from_network.quantity
        assert np.array_equal(from_path.nodes, from_network.nodes)
        assert from_path.time.equals(from_network.time)

    def test_a_str_and_a_path_are_interchangeable(self):
        as_str = NetworkModelResult(self._RES1D, item="WaterLevel")
        as_path = NetworkModelResult(Path(self._RES1D), item="WaterLevel")

        assert np.array_equal(as_str.nodes, as_path.nodes)

    def test_the_network_is_reachable_afterwards(self):
        mr = NetworkModelResult(self._RES1D, item="WaterLevel")

        assert isinstance(mr.network, Network)
        assert mr.network.find(node="100") in mr.nodes

    def test_extract_works_from_a_path_loaded_model(self):
        mr = NetworkModelResult(self._RES1D, item="WaterLevel")
        node = mr.network.find(node="100")
        obs_data = pd.DataFrame({"sensor": np.zeros(len(mr.time))}, index=mr.time)

        extracted = mr.extract(NodeObservation(obs_data, at=node))

        assert isinstance(extracted, NodeModelResult)
        assert extracted.node == node

    def test_an_epanet_path_reads_its_companions(self, tmp_path):
        res = _copy_epanet(tmp_path, ".resx", ".inp")

        mr = NetworkModelResult(res, item="Head")

        assert "Volume" in mr.network.quantities
        assert mr.network._reaches["10"].length == pytest.approx(3209.544)

    def test_an_unreadable_format_is_refused(self):
        with pytest.raises(NotImplementedError, match="Unsupported file extension"):
            NetworkModelResult("./tests/testdata/obs.dfs0")
