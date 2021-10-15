import pytest
import pandas as pd
import numpy as np

from fmskill.observation import TrackObservation


@pytest.fixture
def c2():
    return "tests/testdata/SW/Alti_c2_Dutch.dfs0"


def test_read(c2):
    o1 = TrackObservation(c2, item=2, name="c2")
    assert o1.n_points == 298
    assert len(o1.x) == o1.n_points
    assert o1.name == "c2"
    assert pytest.approx(o1.values.max()) == 17.67


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

    o = TrackObservation(df, item=2)
    assert o.df.index.is_unique
    assert o.df.index[160].to_pydatetime().microsecond == 10000
    assert o.df.index[161].to_pydatetime().microsecond == 20000


def test_trackobservation_item_dfs0(c2):
    with pytest.raises(ValueError, match="Input has more than 3 items"):
        TrackObservation(c2)

    o1 = TrackObservation(c2, item=2)
    assert o1.n_points == 298

    o2 = TrackObservation(c2, item="swh")
    assert o2.n_points == 298


def test_trackobservation_item_csv(c2):
    fn = "tests/testdata/altimetry_NorthSea_20171027.csv"
    df = pd.read_csv(fn, index_col=0, parse_dates=True)

    with pytest.raises(ValueError, match="Input has more than 3 items"):
        TrackObservation(df)

    o1 = TrackObservation(df, item=-1)
    assert o1.n_points == 1115
    assert o1.df.columns[-1] == "wind_speed"

    o2 = TrackObservation(df, item="significant_wave_height")
    assert o2.n_points == 1115


def test_hist(c2):
    o1 = TrackObservation(c2, item=2)
    o1.hist()

    o1.hist(bins=20, title="new_title", color="red")
