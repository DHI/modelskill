import pytest
import pandas as pd
import numpy as np

from modelskill.observation import TrackObservation
from modelskill import Quantity


@pytest.fixture
def c2():
    return "tests/testdata/SW/Alti_c2_Dutch.dfs0"


def test_read(c2):
    o1 = TrackObservation(c2, item=2, name="c2")
    assert o1.n_points == 298
    assert len(o1.x) == o1.n_points
    assert o1.name == "c2"
    assert pytest.approx(o1.values.max()) == 17.67
    # o2 = TrackObservation(c2, item=2, name="c2", units="inches/hour")
    TrackObservation(
        c2, item=2, name="c2", quantity=Quantity(name="Wind speed", unit="inches/hour")
    )
    assert "x" in o1.data
    assert "y" in o1.data


def test_from_df():

    n = 5

    df = pd.DataFrame(
        {
            "t": pd.date_range("2010-01-01", freq="10s", periods=n),
            "x": np.linspace(0, 10, n),
            "y": np.linspace(45000, 45100, n),
            "swh": [0.1, 0.3, 0.4, 0.5, 0.3],
        }
    )

    df = df.set_index("t")

    t1 = TrackObservation(df, name="fake")
    assert t1.n_points == n


def test_non_unique_index():
    fn = "tests/testdata/altimetry_NorthSea_20171027.csv"
    df = pd.read_csv(fn, index_col=0, parse_dates=True)
    assert not df.index.is_unique
    assert df.index[160] == df.index[161]

    with pytest.warns(UserWarning, match="Time axis has duplicate entries"):
        o = TrackObservation(df, item=2)
    assert o.data.index.is_unique
    assert not df.index.is_unique  # did not change input data
    assert o.data.index[160].to_pydatetime().microsecond == 1000
    assert o.data.index[161].to_pydatetime().microsecond == 2000

    with pytest.warns(UserWarning, match="Time axis has duplicate entries"):
        o = TrackObservation(df, item=2, offset_duplicates=0.0001)
    assert o.data.index.is_unique
    assert o.data.index[160].to_pydatetime().microsecond == 100
    assert o.data.index[161].to_pydatetime().microsecond == 200


def test_trackobservation_item_dfs0(c2):
    with pytest.raises(ValueError, match="Input has more than 3 items"):
        TrackObservation(c2)

    o1 = TrackObservation(c2, item=2)
    assert o1.n_points == 298

    o2 = TrackObservation(c2, item="swh")
    assert o2.n_points == 298


def test_trackobservation_item_csv():
    fn = "tests/testdata/altimetry_NorthSea_20171027.csv"
    df = pd.read_csv(fn, index_col=0, parse_dates=True)

    with pytest.raises(ValueError, match="It should have at least 3"):
        TrackObservation(df[["lon", "surface_elevation"]])

    with pytest.raises(ValueError, match="Input has more than 3 items"):
        TrackObservation(df)

    with pytest.warns(UserWarning, match="Time axis has duplicate entries"):
        o1 = TrackObservation(df, item=-1)
    assert o1.n_points == 1115
    assert o1.data.columns[-1] == "wind_speed"

    with pytest.warns(UserWarning, match="Time axis has duplicate entries"):
        o2 = TrackObservation(df, item="significant_wave_height")
    assert o2.n_points == 1115


def test_hist(c2):
    o1 = TrackObservation(c2, item=2)
    o1.plot.hist()

    o1.plot.hist(bins=20, title="new_title", color="red")


def test_trackobservation_x_y_item(c2):
    fn = "tests/testdata/altimetry_NorthSea_20171027.csv"
    df_in = pd.read_csv(fn, index_col=0, parse_dates=True)
    cols = ["lat", "surface_elevation", "lon", "wind_speed"]
    df = df_in[cols]  # re-order columns

    with pytest.raises(ValueError, match="Input has more than 3 items"):
        TrackObservation(df)

    with pytest.warns(UserWarning, match="Time axis has duplicate entries"):
        o1 = TrackObservation(df, item=-1, x_item="lon", y_item="lat")
    assert o1.n_points == 1115
    assert o1.data.columns[-1] == "wind_speed"

    with pytest.warns(UserWarning, match="Time axis has duplicate entries"):
        o2 = TrackObservation(df, item="surface_elevation", x_item=2, y_item=0)
    assert o2.n_points == 1115

    with pytest.raises(ValueError, match="must be different!"):
        TrackObservation(df, item=-1, x_item="lon", y_item="lon")

    cols = ["lat", "surface_elevation", "lon"]
    df = df_in[cols]
    with pytest.raises(ValueError, match="must be different!"):
        TrackObservation(df, x_item="lon", y_item="lat")


def test_force_keyword_args(c2):

    with pytest.raises(TypeError):
        TrackObservation(c2, 2, "c2")
