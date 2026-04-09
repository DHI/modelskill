import pytest
import numpy as np
import pandas as pd
import xarray as xr
import modelskill as ms


@pytest.fixture
def simple_vertical_comparer():
    obs_time = pd.DatetimeIndex(
        ["2020-01-01 13:00:00"] * 2 + ["2020-01-02 11:00:00"] * 2
    )
    obs = ms.VerticalObservation(
        pd.DataFrame(
            {
                "v": [1.0, 2.0, 1.1, 2.1],
                "z": [-1.0, -2.0, -1.0, -2.0],
                "x": [20.0, 20.0, 20.0, 20.0],
                "y": [55.0, 55.0, 55.0, 55.0],
            },
            index=obs_time,
        ),
        z_item="z",
        item="v",
        name="obs",
        x=20.0,
        y=55.0,
    )

    mod_time = pd.DatetimeIndex(
        ["2020-01-01 12:00:00"] * 4 + ["2020-01-02 12:00:00"] * 4
    )
    mod = ms.VerticalModelResult(
        pd.DataFrame(
            {
                "mod": [1.1, 2.1, 3.1, 4.1, 1.2, 2.2, 3.2, 4.2],
                "z": [-1.0, -2.0, -3.0, -4.0, -1.0, -2.0, -3.0, -4.0],
            },
            index=mod_time,
        ),
        z_item=1,
        item=0,
    )

    return ms.match(obs, mod)


def test_vertical_skill_with_int_bins(simple_vertical_comparer):
    sk = simple_vertical_comparer.vertical.skill(bins=1, metrics="rmse")

    assert sk is not None
    df = sk.to_dataframe()
    assert list(df.columns) == ["n", "rmse"]
    assert df.loc["2.0m-1.0m", "n"] == 2
    assert df.loc["2.0m-1.0m", "rmse"] == pytest.approx(0.1)


def test_vertical_skill_with_explicit_bins(simple_vertical_comparer):
    sk = simple_vertical_comparer.vertical.skill(
        bins=[(-2.1, -1.9), (-1.1, -0.9)],
        metrics="rmse",
    )

    assert sk is not None
    df = sk.to_dataframe()
    assert list(df.index) == ["2.1m-1.9m", "1.1m-0.9m"]
    assert np.all(df["n"].to_numpy() == [2, 2])
    assert np.allclose(df["rmse"].to_numpy(), [0.1, 0.1])


def test_vertical_skill_with_binsize(simple_vertical_comparer):
    sk = simple_vertical_comparer.vertical.skill(binsize=1.0, metrics="rmse")

    assert sk is not None
    df = sk.to_dataframe()
    assert df.loc["2.0m-1.0m", "n"] == 2
    assert df.loc["2.0m-1.0m", "rmse"] == pytest.approx(0.1)


def test_vertical_skill_returns_none_for_none_bins(simple_vertical_comparer):
    assert simple_vertical_comparer.vertical.skill(bins=None, metrics="rmse") is None


# @pytest.mark.parametrize(
#     "method, expected_obs, expected_mod",
#     [
#         ("mean", [1.5, 1.6], [1.6, 1.7]),
#         ("min", [1.0, 1.1], [1.1, 1.2]),
#         ("max", [2.0, 2.1], [2.1, 2.2]),
#     ],
# )
# def test_vertical_aggregations_use_observation_depth_range(
#     simple_vertical_comparer, method, expected_obs, expected_mod
# ):
#     agg_cmp = getattr(simple_vertical_comparer.vertical, method)()
#     mod_name = agg_cmp.mod_names[0]

#     assert np.allclose(agg_cmp.data["Observation"].values, expected_obs)
#     assert np.allclose(agg_cmp.data[mod_name].values, expected_mod)
#     assert np.allclose(agg_cmp.raw_mod_data[mod_name].values, expected_mod)
