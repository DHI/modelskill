"""Test network models and observations"""

import pytest
import pandas as pd
import xarray as xr
import numpy as np
import modelskill as ms
from modelskill.model.network import NetworkModelResult, NodeModelResult
from modelskill.obs import NodeObservation


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
def sample_node_data():
    """Sample node observation data"""
    time = pd.date_range("2010-01-01", periods=10, freq="h")

    # Create sample data with some variation
    np.random.seed(42)
    data = np.random.randn(len(time)) * 0.1 + 1.5

    df = pd.DataFrame({"WaterLevel": data}, index=time)

    return df


class TestNetworkModelResult:
    """Test NetworkModelResult class"""

    def test_init_with_xarray_dataset(self, sample_network_data):
        """Test initialization with xr.Dataset"""
        nmr = NetworkModelResult(sample_network_data)

        assert nmr.name == "WaterLevel"
        assert len(nmr.time) == 10
        assert isinstance(nmr.time, pd.DatetimeIndex)
        assert list(sample_network_data.node.values) == [123, 456, 789]

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

    def test_init_fails_without_time_dimension(self, sample_network_data):
        """Test that initialization fails without time dimension"""
        data_no_time = sample_network_data.drop_dims("time")

        with pytest.raises(AssertionError, match="Dataset must have time dimension"):
            NetworkModelResult(data_no_time)

    def test_init_fails_without_node_dimension(self, sample_network_data):
        """Test that initialization fails without node dimension"""
        data_no_node = sample_network_data.drop_dims("node")

        with pytest.raises(AssertionError, match="Dataset must have node dimension"):
            NetworkModelResult(data_no_node)

    def test_init_fails_with_non_dataset(self):
        """Test that initialization fails with non-xarray.Dataset"""
        with pytest.raises(
            AssertionError, match="NetworkModelResult requires xarray.Dataset"
        ):
            NetworkModelResult(pd.DataFrame({"a": [1, 2, 3]}))

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

    def test_init_with_dataframe(self, sample_node_data):
        """Test initialization with pandas DataFrame"""
        obs = NodeObservation(
            sample_node_data, node=123, name="Node_123", item="WaterLevel"
        )

        assert obs.node == 123
        assert obs.name == "Node_123"
        assert len(obs.time) == 10
        assert isinstance(obs.time, pd.DatetimeIndex)

    def test_init_with_series(self, sample_node_data):
        """Test initialization with pandas Series"""
        series = sample_node_data["WaterLevel"]
        obs = NodeObservation(series, node=456, name="Node_456")

        assert obs.node == 456
        assert obs.name == "Node_456"
        assert len(obs.time) == 10

    def test_node_property(self, sample_node_data):
        """Test node property"""
        obs = NodeObservation(sample_node_data, node=789, name="Node_789")

        assert obs.node == 789
        assert isinstance(obs.node, int)

    def test_node_property_missing_coordinate(self, sample_node_data):
        """Test node property when coordinate is missing"""
        obs = NodeObservation(sample_node_data, node=123, name="Node_123")

        # Manually remove the node coordinate to test error handling
        del obs.data.coords["node"]

        with pytest.raises(ValueError, match="Node coordinate not found"):
            _ = obs.node

    def test_weight_property(self, sample_node_data):
        """Test weight property"""
        obs = NodeObservation(sample_node_data, node=123, weight=2.5)

        assert obs.weight == 2.5

    def test_attrs_property(self, sample_node_data):
        """Test attrs property"""
        attrs = {"source": "test", "version": "1.0"}
        obs = NodeObservation(sample_node_data, node=123, attrs=attrs)

        assert obs.attrs["source"] == "test"
        assert obs.attrs["version"] == "1.0"

    def test_multiple_nodes_auto_assign_items(self, sample_node_data):
        """Test auto-assignment of items when nodes match column count"""
        # Create a multi-column DataFrame
        multi_data = pd.DataFrame({
            'station_0': sample_node_data['WaterLevel'],
            'station_1': sample_node_data['WaterLevel'] + 0.1,
            'station_2': sample_node_data['WaterLevel'] + 0.2,
        })
        
        # Only provide nodes - items should be auto-assigned [0, 1, 2]
        nodes = [123, 456, 789]
        obs_list = NodeObservation(multi_data, node=nodes)
        
        # Should return a list of NodeObservation objects
        assert isinstance(obs_list, list)
        assert len(obs_list) == 3
        assert all(isinstance(obs, NodeObservation) for obs in obs_list)
        
        # Check that nodes are assigned correctly
        assert obs_list[0].node == 123
        assert obs_list[1].node == 456
        assert obs_list[2].node == 789
        
        # Check default names
        assert obs_list[0].name == "Node_123"
        assert obs_list[1].name == "Node_456"
        assert obs_list[2].name == "Node_789"
        
    def test_multiple_nodes_auto_assign_mismatched_count(self, sample_node_data):
        """Test error when nodes don't match column count for auto-assignment"""
        multi_data = pd.DataFrame({
            'station_0': sample_node_data['WaterLevel'],
            'station_1': sample_node_data['WaterLevel'] + 0.1,
        })
        
        # Provide 3 nodes but only 2 columns - should fail
        nodes = [123, 456, 789]
        
        with pytest.raises(ValueError, match="Number of nodes.*must match.*data columns.*when item is not specified"):
            NodeObservation(multi_data, node=nodes)  # No item provided
            
    def test_multiple_nodes_creation(self, sample_node_data):
        """Test creating multiple NodeObservations with lists"""
        # Create a multi-column DataFrame
        multi_data = pd.DataFrame({
            'station_0': sample_node_data['WaterLevel'],
            'station_1': sample_node_data['WaterLevel'] + 0.1,
            'station_2': sample_node_data['WaterLevel'] + 0.2,
        })
        
        nodes = [123, 456, 789]
        items = [0, 1, 2]
        obs_list = NodeObservation(multi_data, node=nodes, item=items)
        
        # Should return a list of NodeObservation objects
        assert isinstance(obs_list, list)
        assert len(obs_list) == 3
        assert all(isinstance(obs, NodeObservation) for obs in obs_list)
        
        # Check that nodes are assigned correctly
        assert obs_list[0].node == 123
        assert obs_list[1].node == 456
        assert obs_list[2].node == 789
        
        # Check default names
        assert obs_list[0].name == "Node_123"
        assert obs_list[1].name == "Node_456"
        assert obs_list[2].name == "Node_789"
        
    def test_multiple_nodes_with_custom_names(self, sample_node_data):
        """Test creating multiple NodeObservations with custom names"""
        multi_data = pd.DataFrame({
            'station_0': sample_node_data['WaterLevel'],
            'station_1': sample_node_data['WaterLevel'] + 0.1,
        })
        
        nodes = [123, 456]
        items = [0, 1]
        names = ["Sensor_A", "Sensor_B"]
        obs_list = NodeObservation(multi_data, node=nodes, item=items, name=names)
        
        assert len(obs_list) == 2
        assert obs_list[0].name == "Sensor_A"
        assert obs_list[1].name == "Sensor_B"
        
    def test_multiple_nodes_with_string_name_prefix(self, sample_node_data):
        """Test creating multiple NodeObservations with string name prefix"""
        multi_data = pd.DataFrame({
            'station_0': sample_node_data['WaterLevel'],
            'station_1': sample_node_data['WaterLevel'] + 0.1,
        })
        
        nodes = [123, 456]
        items = [0, 1]
        name = "Station"
        obs_list = NodeObservation(multi_data, node=nodes, item=items, name=name)
        
        assert len(obs_list) == 2
        assert obs_list[0].name == "Station_0"
        assert obs_list[1].name == "Station_1"
        
    def test_mismatched_item_node_lists(self, sample_node_data):
        """Test error when item and node lists have different lengths"""
        multi_data = pd.DataFrame({
            'station_0': sample_node_data['WaterLevel'],
            'station_1': sample_node_data['WaterLevel'] + 0.1,
        })
        
        nodes = [123, 456, 789]  # 3 nodes
        items = [0, 1]  # 2 items
        
        with pytest.raises(ValueError, match="Length of item list.*must match.*node list"):
            NodeObservation(multi_data, node=nodes, item=items)
            
    def test_mismatched_name_list(self, sample_node_data):
        """Test error when name list doesn't match item/node lists"""
        multi_data = pd.DataFrame({
            'station_0': sample_node_data['WaterLevel'],
            'station_1': sample_node_data['WaterLevel'] + 0.1,
        })
        
        nodes = [123, 456]
        items = [0, 1]
        names = ["Sensor_A"]  # Only 1 name for 2 nodes
        
        with pytest.raises(ValueError, match="Length of name list.*must match.*item/node lists"):
            NodeObservation(multi_data, node=nodes, item=items, name=names)
            
    def test_only_one_list_provided(self, sample_node_data):
        """Test error when only one of item or node is a list"""
        with pytest.raises(ValueError, match="If node is a list, item must also be a list or None"):
            NodeObservation(sample_node_data, node=[123, 456], item=0)
            
        with pytest.raises(ValueError, match="If item is a list, node must also be a list"):
            NodeObservation(sample_node_data, node=123, item=[0, 1])


class TestNodeModelResult:
    """Test NodeModelResult class"""

    def test_init_with_dataframe(self, sample_node_data):
        """Test initialization with pandas DataFrame"""
        nmr = NodeModelResult(sample_node_data, node=123, name="Node_123_Model")

        assert nmr.node == 123
        assert nmr.name == "Node_123_Model"
        assert len(nmr.time) == 10

    def test_init_with_series(self, sample_node_data):
        """Test initialization with pandas Series"""
        series = sample_node_data["WaterLevel"]
        nmr = NodeModelResult(series, node=456, name="Node_456_Model")

        assert nmr.node == 456
        assert nmr.name == "Node_456_Model"

    def test_node_property(self, sample_node_data):
        """Test node property"""
        nmr = NodeModelResult(sample_node_data, node=789, name="Node_789_Model")

        assert nmr.node == 789
        assert isinstance(nmr.node, int)

    def test_node_property_missing_coordinate(self, sample_node_data):
        """Test node property when coordinate is missing"""
        nmr = NodeModelResult(sample_node_data, node=123, name="Node_123_Model")

        # Manually remove the node coordinate to test error handling
        del nmr.data.coords["node"]

        with pytest.raises(ValueError, match="Node coordinate not found"):
            _ = nmr.node


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

    def test_matching_workflow_multiple_nodes(self, sample_network_data, sample_node_data):
        """Test matching workflow with multiple node observations using enhanced NodeObservation"""
        # Create network model result
        nmr = NetworkModelResult(sample_network_data, name="Network_Model")
        
        # Create multi-column observation data
        multi_data = pd.DataFrame({
            'station_0': sample_node_data['WaterLevel'],
            'station_1': sample_node_data['WaterLevel'] + 0.1,
            'station_2': sample_node_data['WaterLevel'] + 0.2,
        })
        
        # Create multiple NodeObservations using the enhanced NodeObservation (auto-assign items)
        nodes = [123, 456, 789]
        obs_list = NodeObservation(multi_data, node=nodes)  # Items auto-assigned
        
        # Test that matching works
        comparer_collection = ms.match(obs_list, nmr)
        
        assert comparer_collection is not None
        assert len(comparer_collection) == 3  # Should have 3 comparers
        
        # Check that all model names are correct
        for comparer in comparer_collection:
            assert "Network_Model" in comparer.mod_names
            assert comparer.n_points > 0
