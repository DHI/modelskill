"""Test network models and observations"""

import pytest
import pandas as pd
import xarray as xr
import numpy as np
import modelskill as ms
from modelskill.model.network import NetworkModelResult, NodeModelResult
from modelskill.obs import NodeObservation, MultiNodeObservation
from modelskill.quantity import Quantity


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

    def test_init_with_xarray_dataset(self, sample_network_data):
        """Test initialization with xr.Dataset"""
        nmr = NetworkModelResult(sample_network_data)

        assert nmr.name == "WaterLevel"
        assert len(nmr.time) == 10
        assert isinstance(nmr.time, pd.DatetimeIndex)
        assert list(sample_network_data.node.values) == [123, 456, 789]

    def test_init_with_dataframe(self):
        """Test successful initialization with properly formatted DataFrame"""
        # Create a valid DataFrame with MultiIndex columns
        arrays = [[123, 456], ["WaterLevel", "WaterLevel"]]
        columns = pd.MultiIndex.from_arrays(arrays, names=["node", "quantity"])

        data = np.random.randn(5, 2)
        df = pd.DataFrame(data, columns=columns)
        df["time"] = pd.date_range("2010-01-01", periods=5, freq="h")
        df.set_index("time", inplace=True)

        nmr = NetworkModelResult(df, name="DataFrame_Network")

        assert nmr.name == "DataFrame_Network"
        assert len(nmr.time) == 5
        assert isinstance(nmr.time, pd.DatetimeIndex)
        # After conversion, the nodes should be accessible
        assert 123 in nmr.nodes
        assert 456 in nmr.nodes

    def test_init_with_item_selection(self, sample_network_data):
        """Test initialization with specific item"""
        # Add another variable
        sample_network_data["Discharge"] = sample_network_data["WaterLevel"] * 10

        nmr = NetworkModelResult(
            sample_network_data, item="WaterLevel", name="Network_WL"
        )

        assert nmr.name == "Network_WL"
        assert "WaterLevel" in nmr.data.data_vars
        assert "Discharge" not in nmr.data.data_vars

    @pytest.mark.parametrize("coord", ["time", "node"])
    def test_init_fails_without_coord(self, coord, sample_network_data):
        """Test that initialization fails without time dimension"""
        data_no_time = sample_network_data.rename_vars({coord: "another_name"})

        with pytest.raises(
            ValueError, match=f"Dataset must have '{coord}' as coordinate"
        ):
            NetworkModelResult(data_no_time)

    def test_init_fails_with_invalid_dataframe_index(self):
        """Test that initialization fails with non-DatetimeIndex DataFrame"""
        df = pd.DataFrame({"a": [1, 2, 3]})  # Regular RangeIndex
        with pytest.raises(
            TypeError,
            match="DataFrame index must be a pd.DatetimeIndex",
        ):
            NetworkModelResult(df)

    def test_init_fails_with_invalid_dataframe_columns(self):
        """Test that initialization fails with non-MultiIndex columns"""
        df = pd.DataFrame(
            {"a": [1, 2, 3]}, index=pd.date_range("2010-01-01", periods=3, freq="h")
        )
        with pytest.raises(
            TypeError,
            match="DataFrame columns must be a pd.MultiIndex",
        ):
            NetworkModelResult(df)

    def test_init_fails_with_wrong_multiindex_levels(self):
        """Test that initialization fails with wrong MultiIndex level names"""
        arrays = [[123, 456], ["WL", "WL"]]
        columns = pd.MultiIndex.from_arrays(arrays, names=["wrong", "level"])
        df = pd.DataFrame(
            np.random.randn(3, 2),
            index=pd.date_range("2010-01-01", periods=3, freq="h"),
            columns=columns,
        )
        with pytest.raises(
            ValueError,
            match="DataFrame column level names must be 'node' and 'quantity'",
        ):
            NetworkModelResult(df)

    def test_init_fails_with_wrong_multiindex_nlevels(self):
        """Test that initialization fails with wrong number of MultiIndex levels"""
        arrays = [[123, 456], ["WL", "WL"], ["extra", "level"]]
        columns = pd.MultiIndex.from_arrays(arrays, names=["node", "quantity", "extra"])
        df = pd.DataFrame(
            np.random.randn(3, 2),
            index=pd.date_range("2010-01-01", periods=3, freq="h"),
            columns=columns,
        )
        with pytest.raises(
            ValueError,
            match="DataFrame columns must have exactly 2 levels",
        ):
            NetworkModelResult(df)

    def test_init_fails_with_unsupported_type(self):
        """Test that initialization fails with unsupported data types"""
        with pytest.raises(
            TypeError,
            match="NetworkModelResult expects a pd.DataFrame or xr.Dataset",
        ):
            NetworkModelResult([1, 2, 3])  # List is not supported

    def test_repr(self, sample_network_data):
        """Test string representation"""
        nmr = NetworkModelResult(sample_network_data, name="Test_Network")
        repr_str = repr(nmr)

        assert "NetworkModelResult" in repr_str
        assert "Test_Network" in repr_str

    def test_extract_valid_node(self, sample_network_data, sample_node_data):
        """Test extraction of valid node"""
        nmr = NetworkModelResult(sample_network_data)
        obs = NodeObservation(sample_node_data, node=123, name="Node_123")

        extracted = nmr.extract(obs)

        assert isinstance(extracted, NodeModelResult)
        assert extracted.node == 123
        assert extracted.name == "WaterLevel"
        assert len(extracted.time) == 10

    def test_extract_invalid_node(self, sample_network_data, sample_node_data):
        """Test extraction of invalid node"""
        nmr = NetworkModelResult(sample_network_data)
        obs = NodeObservation(sample_node_data, node=999, name="Node_999")

        with pytest.raises(ValueError, match="Node 999 not found"):
            nmr.extract(obs)

    def test_extract_wrong_observation_type(self, sample_network_data):
        """Test extraction with wrong observation type"""
        nmr = NetworkModelResult(sample_network_data)

        # Create a proper PointObservation with DataFrame
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

    def test_multiple_nodes_auto_assign_items(self, sample_node_data):
        """Test auto-assignment of items when nodes match column count"""
        # Create a multi-column DataFrame
        multi_data = pd.DataFrame(
            {
                "station_0": sample_node_data["WaterLevel"],
                "station_1": sample_node_data["WaterLevel"] + 0.1,
                "station_2": sample_node_data["WaterLevel"] + 0.2,
            }
        )

        # Only provide nodes - items should be auto-assigned [0, 1, 2]
        nodes = [123, 456, 789]
        obs_list = MultiNodeObservation(multi_data, nodes=nodes)

        # Should return a list of NodeObservation objects
        assert isinstance(obs_list, list)
        assert len(obs_list) == 3
        assert all(isinstance(obs, NodeObservation) for obs in obs_list)

        # Check that nodes are assigned correctly
        assert obs_list[0].node == 123
        assert obs_list[1].node == 456
        assert obs_list[2].node == 789

        # Check default names (str(node_id))
        assert obs_list[0].name == "station_0"
        assert obs_list[1].name == "station_1"
        assert obs_list[2].name == "station_2"

    def test_multiple_nodes_auto_assign_mismatched_count(self, sample_node_data):
        """Test error when nodes don't match column count for auto-assignment"""
        multi_data = pd.DataFrame(
            {
                "station_0": sample_node_data["WaterLevel"],
                "station_1": sample_node_data["WaterLevel"] + 0.1,
            }
        )

        # Provide 3 nodes but only 2 columns - should fail
        nodes = [123, 456, 789]

        with pytest.raises(
            ValueError,
            match="must match the number of columns in data",
        ):
            MultiNodeObservation(multi_data, nodes=nodes)  # No item provided

    def test_multiple_nodes_creation(self, sample_node_data):
        """Test creating multiple NodeObservations with lists"""
        # Create a multi-column DataFrame
        multi_data = pd.DataFrame(
            {
                "station_0": sample_node_data["WaterLevel"],
                "station_1": sample_node_data["WaterLevel"] + 0.1,
                "station_2": sample_node_data["WaterLevel"] + 0.2,
            }
        )

        nodes = [123, 456, 789]
        obs_list = MultiNodeObservation(multi_data, nodes=nodes)

        # Should return a list of NodeObservation objects
        assert isinstance(obs_list, list)
        assert len(obs_list) == 3
        assert all(isinstance(obs, NodeObservation) for obs in obs_list)

        # Check that nodes are assigned correctly
        assert obs_list[0].node == 123
        assert obs_list[1].node == 456
        assert obs_list[2].node == 789

        # Check default names (str(node_id))
        assert obs_list[0].name == "station_0"
        assert obs_list[1].name == "station_1"
        assert obs_list[2].name == "station_2"

    def test_only_one_list_provided(self, sample_node_data):
        """Test error when lists are passed to NodeObservation instead of MultiNodeObservation"""
        with pytest.raises(ValueError, match="Use MultiNodeObservation"):
            NodeObservation(sample_node_data, node=[123, 456], item=0)

        with pytest.raises(ValueError, match="Use MultiNodeObservation"):
            NodeObservation(sample_node_data, node=123, item=[0, 1])


class TestMultiNodeObservation:
    """Test MultiNodeObservation class"""

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

    def test_returns_list_of_node_observations(self, multi_data):
        obs_list = MultiNodeObservation(multi_data, nodes=[123, 456, 789])

        assert isinstance(obs_list, list)
        assert len(obs_list) == 3
        assert all(isinstance(o, NodeObservation) for o in obs_list)

    def test_node_ids_are_assigned_correctly(self, multi_data):
        obs_list = MultiNodeObservation(multi_data, nodes=[123, 456, 789])

        assert obs_list[0].node == 123
        assert obs_list[1].node == 456
        assert obs_list[2].node == 789

    def test_names_derived_from_column_names(self, multi_data):
        obs_list = MultiNodeObservation(multi_data, nodes=[123, 456, 789])

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
        obs_list = MultiNodeObservation(ds, nodes=[123, 456])

        assert len(obs_list) == 2
        assert obs_list[0].node == 123
        assert obs_list[1].node == 456

    def test_nodes_must_be_list(self, multi_data):
        with pytest.raises(ValueError, match="node must be a list"):
            MultiNodeObservation(multi_data, nodes=123)

    def test_nodes_length_must_match_columns(self, multi_data):
        with pytest.raises(ValueError, match="Length of nodes"):
            MultiNodeObservation(multi_data, nodes=[123, 456])  # 2 nodes, 3 columns

    def test_attrs_propagated_to_all_observations(self, multi_data):
        attrs = {"source": "sensor_array", "version": 2}
        obs_list = MultiNodeObservation(multi_data, nodes=[1, 2, 3], attrs=attrs)

        for obs in obs_list:
            assert obs.attrs["source"] == "sensor_array"
            assert obs.attrs["version"] == 2

    def test_repr(self, multi_data):
        obs_list = MultiNodeObservation(multi_data, nodes=[1, 2, 3])
        r = repr(obs_list)

        assert "MultiNodeObservation" in r
        assert "3" in r


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

    def test_network_to_node_extraction(self, sample_network_data, sample_node_data):
        """Test complete workflow from network model to node extraction"""
        # Create network model result
        nmr = NetworkModelResult(sample_network_data, name="Network_Model")

        # Create node observation
        obs = NodeObservation(sample_node_data, node=123, name="Node_123_Obs")

        # Extract node model result
        extracted = nmr.extract(obs)

        # Verify extraction worked
        assert isinstance(extracted, NodeModelResult)
        assert extracted.node == 123
        assert extracted.name == "Network_Model"

        # Verify time alignment possibilities exist
        assert len(extracted.time) == len(obs.time)

    def test_matching_workflow(self, sample_network_data, sample_node_data):
        """Test matching workflow with network data"""
        # Create network model result
        nmr = NetworkModelResult(sample_network_data, name="Network_Model")

        # Create node observation
        obs = NodeObservation(sample_node_data, node=123, name="Node_123_Obs")

        # Test that matching works
        comparer = ms.match(obs, nmr)

        assert comparer is not None
        assert "Network_Model" in comparer.mod_names
        assert comparer.n_points > 0

    def test_matching_workflow_multiple_nodes(
        self, sample_network_data, sample_node_data
    ):
        """Test matching workflow with multiple node observations using enhanced NodeObservation"""
        # Create network model result
        nmr = NetworkModelResult(sample_network_data, name="Network_Model")

        # Create multi-column observation data
        multi_data = pd.DataFrame(
            {
                "station_0": sample_node_data["WaterLevel"],
                "station_1": sample_node_data["WaterLevel"] + 0.1,
                "station_2": sample_node_data["WaterLevel"] + 0.2,
            }
        )

        # Create multiple NodeObservations using MultiNodeObservation (auto-assign items)
        nodes = [123, 456, 789]
        obs_list = MultiNodeObservation(multi_data, nodes=nodes)  # Items auto-assigned

        # Test that matching works
        comparer_collection = ms.match(obs_list, nmr)

        assert comparer_collection is not None
        assert len(comparer_collection) == 3  # Should have 3 comparers

        # Check that all model names are correct
        for comparer in comparer_collection:
            assert "Network_Model" in comparer.mod_names
            assert comparer.n_points > 0
