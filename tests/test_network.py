"""Test network models and observations"""

import pytest
import pandas as pd
import xarray as xr
import numpy as np
import modelskill as ms
from modelskill.model.network import NetworkModelResult, NodeModelResult
from modelskill.obs import NodeObservation
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

    def test_init_fails_with_non_dataset(self):
        """Test that initialization fails with non-xarray.Dataset"""
        with pytest.raises(
            ValueError, match="'NetworkModelResult' requires xarray.Dataset"
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

    def test_node_property_missing_coordinate(self, sample_node_data):
        """Test node property when coordinate is missing"""
        obs = NodeObservation(sample_node_data, node=123, name="Node_123")

        # Manually remove the node coordinate to test error handling
        del obs.data.coords["node"]

        with pytest.raises(ValueError, match="Node coordinate not found"):
            _ = obs.node

    def test_node_attrs(self, sample_node_data):
        """Test attrs property"""
        attrs = {"source": "test", "version": "1.0"}
        obs = NodeObservation(sample_node_data, node=123, attrs=attrs, weight=2.5)

        assert obs.attrs["source"] == "test"
        assert obs.attrs["version"] == "1.0"
        assert obs.weight == 2.5
        assert obs.quantity == Quantity.undefined()


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
