import pytest
import numpy as np
import pandas as pd
import matplotlib as mpl
import modelskill as ms

mpl.use("Agg")


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
    sk = simple_vertical_comparer.vertical.skill(bins=2, metrics="rmse")

    assert sk is not None
    df = sk.to_dataframe()
    assert "n" in df.columns
    assert "rmse" in df.columns
    assert isinstance(df.index, pd.IntervalIndex)
    assert np.all(df["n"].to_numpy() == [2, 2])

    assert df.iloc[0, df.columns.get_loc("rmse")] == pytest.approx(0.1)
    assert df.iloc[1, df.columns.get_loc("rmse")] == pytest.approx(0.1)


def test_vertical_skill_with_explicit_bins(simple_vertical_comparer):
    sk = simple_vertical_comparer.vertical.skill(
        bins=[-2.1, -1.9, -1.1, -0.9],
        metrics="rmse",
    )

    assert sk is not None
    df = sk.to_dataframe()

    assert isinstance(df.index, pd.IntervalIndex)
    assert np.array_equal(df["n"].to_numpy(), [2, np.nan, 2], equal_nan=True)
    assert np.allclose(df["rmse"].to_numpy(), [0.1, np.nan, 0.1], equal_nan=True)


def test_vertical_skill_multiple_models(simple_vertical_comparer):
    cmp = simple_vertical_comparer
    mod_time = cmp.raw_mod_data["mod"].data.time.values
    z_vals = cmp.raw_mod_data["mod"].data["z"].values
    mod_vals = cmp.raw_mod_data["mod"].values

    # Create a second model with a constant offset
    mod2 = ms.VerticalModelResult(
        pd.DataFrame(
            {"mod2": mod_vals + 0.1, "z": z_vals},
            index=mod_time,
        ),
        z_item="z",
        item="mod2",
    )

    obs_df = (
        cmp.data[["Observation", "z", "x", "y"]]
        .to_dataframe()
        .rename(columns={"Observation": "v"})
    )
    obs = ms.VerticalObservation(
        obs_df, z_item="z", item="v", name=cmp.name, x=cmp.x, y=cmp.y
    )
    mod1 = ms.VerticalModelResult(
        pd.DataFrame({"mod": mod_vals, "z": z_vals}, index=mod_time),
        z_item="z",
        item="mod",
    )

    cmp2 = ms.match(obs, [mod1, mod2])
    sk = cmp2.vertical.skill(bins=2, metrics="rmse")

    assert sk is not None
    assert "model" in sk.data.dims
    assert set(sk.mod_names) == {"mod", "mod2"}
    df = sk.to_dataframe()
    assert "rmse" in df.columns
    assert "n" in df.columns
    assert isinstance(df.index, pd.MultiIndex)
    # assert first index level is pd.IntervalIndex and seconds is model name
    assert isinstance(df.index.levels[0], pd.IntervalIndex)
    assert np.array_equal(df.index.levels[1].values, ["mod", "mod2"])


def _add_second_model(cmp):
    mod_time = cmp.raw_mod_data["mod"].data.time.values
    z_vals = cmp.raw_mod_data["mod"].data["z"].values
    mod_vals = cmp.raw_mod_data["mod"].values

    mod2 = ms.VerticalModelResult(
        pd.DataFrame(
            {"mod2": mod_vals + 0.1, "z": z_vals},
            index=mod_time,
        ),
        z_item="z",
        item="mod2",
    )

    obs_df = (
        cmp.data[["Observation", "z", "x", "y"]]
        .to_dataframe()
        .rename(columns={"Observation": "v"})
    )
    obs = ms.VerticalObservation(
        obs_df, z_item="z", item="v", name=cmp.name, x=cmp.x, y=cmp.y
    )
    mod1 = ms.VerticalModelResult(
        pd.DataFrame({"mod": mod_vals, "z": z_vals}, index=mod_time),
        z_item="z",
        item="mod",
    )

    return ms.match(obs, [mod1, mod2])


def test_vertical_hovmoller_requires_model_for_multi_model_comparer(
    simple_vertical_comparer,
):
    cmp2 = _add_second_model(simple_vertical_comparer)

    with pytest.raises(ValueError, match="Multiple models found"):
        cmp2.vertical.plot.hovmoller()


def test_vertical_hovmoller_accepts_named_model(simple_vertical_comparer):
    cmp2 = _add_second_model(simple_vertical_comparer)

    ax = cmp2.vertical.plot.hovmoller(model="mod2")

    assert ax is not None
    assert "mod2" in ax.get_title()


def test_vertical_skill_n_min(simple_vertical_comparer):
    # Each bin has n=2, so n_min=3 should NaN out metrics but preserve n
    sk = simple_vertical_comparer.vertical.skill(bins=2, metrics="rmse", n_min=3)

    df = sk.to_dataframe()
    assert np.all(df["n"].to_numpy() == [2, 2])
    assert np.all(np.isnan(df["rmse"].to_numpy()))


def test_vertical_skill_raises_for_none_bins(simple_vertical_comparer):
    with pytest.raises(ValueError, match="All bin edges are NaN"):
        simple_vertical_comparer.vertical.skill(bins=None, metrics="rmse")


def test_skill_identical_observation_and_model_gives_perfect_scores():
    time = pd.DatetimeIndex(["2020-01-01 12:00:00"] * 2 + ["2020-01-02 12:00:00"] * 2)
    df = pd.DataFrame(
        {
            "v": [1.0, 2.0, 1.1, 2.1],
            "z": [-1.0, -2.0, -1.0, -2.0],
        },
        index=time,
    )

    obs = ms.VerticalObservation(df, item="v", z_item="z", name="obs")
    mod = ms.VerticalModelResult(
        df.rename(columns={"v": "mod"}), item="mod", z_item="z"
    )
    cmp = ms.match(obs, mod)

    sk = cmp.skill(metrics=["bias", "rmse", "mae", "r2", "cc"])
    row = sk.to_dataframe().iloc[0]

    assert row["bias"] == pytest.approx(0.0)
    assert row["rmse"] == pytest.approx(0.0)
    assert row["mae"] == pytest.approx(0.0)
    assert row["r2"] == pytest.approx(1.0)
    assert row["cc"] == pytest.approx(1.0)


def test_sel_z_scalar_updates_matched_and_raw_vertical_data():
    obs_time = pd.DatetimeIndex(["2020-01-01 12:00:00", "2020-01-02 12:00:00"])
    obs = ms.VerticalObservation(
        pd.DataFrame(
            {
                "obs": [1.0, 2.0],
                "z": [-2.0, -2.0],
            },
            index=obs_time,
        ),
        item="obs",
        z_item="z",
        name="obs",
    )

    mod_time = pd.DatetimeIndex(
        ["2020-01-01 12:00:00"] * 3 + ["2020-01-02 12:00:00"] * 3
    )
    mod = ms.VerticalModelResult(
        pd.DataFrame(
            {
                "mod": [1.1, 2.1, 3.1, 1.2, 2.2, 3.2],
                "z": [-1.2, -2.1, -3.0, -1.0, -1.8, -2.6],
            },
            index=mod_time,
        ),
        item="mod",
        z_item="z",
        name="mod",
    )

    cmp = ms.match(obs, mod)
    cmp_sel = cmp.sel(z=-2.0)

    assert cmp_sel.n_points == cmp.n_points == 2
    assert np.allclose(cmp_sel.data["z"].to_numpy(), [-2.0, -2.0])
    assert np.allclose(cmp_sel.data["Observation"].to_numpy(), [1.0, 2.0])
    assert np.allclose(cmp_sel.data["mod"].to_numpy(), cmp.data["mod"].to_numpy())

    raw_selected_z = cmp_sel.raw_mod_data["mod"].data["z"].to_numpy()
    raw_selected_values = cmp_sel.raw_mod_data["mod"].values
    assert np.allclose(raw_selected_z, [-2.1, -1.8])
    assert np.allclose(raw_selected_values, [2.1, 2.2])


def _obs_mod_frames_from_comparer(cmp):
    mod_name = cmp.mod_names[0]
    obs_df = (
        cmp.data[["Observation", "z", "x", "y"]]
        .to_dataframe()
        .rename(columns={"Observation": "v"})
    )
    mod_df = cmp.raw_mod_data[mod_name].data[[mod_name, "z"]].to_dataframe()
    return obs_df, mod_df, mod_name


def test_sel_z_slice_filters_whole_comparer(simple_vertical_comparer):
    cmp_sel = simple_vertical_comparer.sel(z=slice(-2.1, -1.9))

    assert cmp_sel.n_points == 2
    assert np.allclose(cmp_sel.data["z"].to_numpy(), [-2.0, -2.0])
    assert np.allclose(cmp_sel.data["Observation"].to_numpy(), [2.0, 2.1])
    assert np.allclose(cmp_sel.data["mod"].to_numpy(), [2.1, 2.2])
    assert np.allclose(cmp_sel.raw_mod_data["mod"].data["z"].to_numpy(), [-2.0, -2.0])
    assert np.allclose(cmp_sel.raw_mod_data["mod"].values, [2.1, 2.2])


def test_sel_z_exact_depth_matches_exact_slice(simple_vertical_comparer):
    cmp_scalar = simple_vertical_comparer.sel(z=-2.0)
    cmp_slice = simple_vertical_comparer.sel(z=slice(-2.0, -2.0))

    assert cmp_scalar.n_points == 2
    assert cmp_slice.n_points == 2
    assert np.allclose(cmp_scalar.data["z"].to_numpy(), [-2.0, -2.0])
    assert np.allclose(cmp_slice.data["z"].to_numpy(), [-2.0, -2.0])
    assert np.allclose(cmp_scalar.data["Observation"].to_numpy(), [2.0, 2.1])
    assert np.allclose(cmp_slice.data["Observation"].to_numpy(), [2.0, 2.1])
    assert np.allclose(cmp_scalar.data["mod"].to_numpy(), [2.1, 2.2])
    assert np.allclose(cmp_slice.data["mod"].to_numpy(), [2.1, 2.2])
    assert np.allclose(
        cmp_scalar.raw_mod_data["mod"].data["z"].to_numpy(), [-2.0, -2.0]
    )
    assert np.allclose(cmp_slice.raw_mod_data["mod"].data["z"].to_numpy(), [-2.0, -2.0])
    assert np.allclose(cmp_scalar.raw_mod_data["mod"].values, [2.1, 2.2])
    assert np.allclose(cmp_slice.raw_mod_data["mod"].values, [2.1, 2.2])


def test_vertical_accessor_raises_for_non_vertical_comparer():
    df_point = pd.DataFrame(
        {
            "obs": [1.0, 2.0, 3.0],
            "mod": [1.1, 2.1, 3.1],
        },
        index=pd.date_range("2020-01-01", periods=3, freq="h"),
    )
    cmp_point = ms.from_matched(df_point, obs_item="obs", mod_items=["mod"])

    with pytest.raises(
        AttributeError, match="vertical accessor is only available for vertical data"
    ):
        cmp_point.vertical


def test_sel_z_scalar_outside_matched_range_returns_empty_match(
    simple_vertical_comparer,
):
    cmp_sel = simple_vertical_comparer.sel(z=2.0)

    assert cmp_sel.n_points == 0
    assert cmp_sel.data.time.size == 0
    assert np.allclose(cmp_sel.raw_mod_data["mod"].data["z"].to_numpy(), [-1.0, -1.0])
    assert np.allclose(cmp_sel.raw_mod_data["mod"].values, [1.1, 1.2])


def test_vertical_matching_positive_depth_behaves_like_negative_depth(
    simple_vertical_comparer,
):
    obs_df, mod_df, mod_name = _obs_mod_frames_from_comparer(simple_vertical_comparer)

    obs_df_pos = obs_df.copy()
    obs_df_pos["z"] = -obs_df_pos["z"]
    mod_df_pos = mod_df.copy()
    mod_df_pos["z"] = -mod_df_pos["z"]

    cmp_pos = ms.match(
        ms.VerticalObservation(
            obs_df_pos,
            z_item="z",
            item="v",
            name=simple_vertical_comparer.name,
            x=simple_vertical_comparer.x,
            y=simple_vertical_comparer.y,
        ),
        ms.VerticalModelResult(
            mod_df_pos,
            z_item="z",
            item=mod_name,
            name=mod_name,
        ),
    )

    assert cmp_pos.n_points == simple_vertical_comparer.n_points
    assert np.allclose(
        cmp_pos.data["Observation"].to_numpy(),
        simple_vertical_comparer.data["Observation"].to_numpy(),
    )
    assert np.allclose(
        cmp_pos.data[mod_name].to_numpy(),
        simple_vertical_comparer.data[mod_name].to_numpy(),
    )
    assert np.allclose(
        np.abs(cmp_pos.data["z"].to_numpy()),
        np.abs(simple_vertical_comparer.data["z"].to_numpy()),
    )


def test_vertical_matching_no_overlap_for_negative_obs_and_positive_model_depths(
    simple_vertical_comparer,
):
    obs_df, mod_df, mod_name = _obs_mod_frames_from_comparer(simple_vertical_comparer)
    mod_df["z"] = np.abs(mod_df["z"])

    cmp = ms.match(
        ms.VerticalObservation(
            obs_df,
            z_item="z",
            item="v",
            name=simple_vertical_comparer.name,
            x=simple_vertical_comparer.x,
            y=simple_vertical_comparer.y,
        ),
        ms.VerticalModelResult(
            mod_df,
            z_item="z",
            item=mod_name,
            name=mod_name,
        ),
    )
    assert cmp.n_points == 0


@pytest.mark.parametrize(
    "method, expected_obs, expected_mod",
    [
        ("mean", [1.5, 1.6], [1.6, 1.7]),
        ("min", [1.0, 1.1], [1.1, 1.2]),
        ("max", [2.0, 2.1], [2.1, 2.2]),
    ],
)
def test_vertical_aggregations_use_observation_depth_range(
    simple_vertical_comparer, method, expected_obs, expected_mod
):
    agg_cmp = getattr(simple_vertical_comparer.vertical, method)()
    mod_name = agg_cmp.mod_names[0]

    assert np.allclose(agg_cmp.data["Observation"].values, expected_obs)
    assert np.allclose(agg_cmp.data[mod_name].values, expected_mod)
    assert np.allclose(agg_cmp.raw_mod_data[mod_name].values, expected_mod)


@pytest.mark.parametrize(
    "method, expected_mod, expected_mod2",
    [
        ("mean", [1.6, 1.7], [1.7, 1.8]),
        ("min", [1.1, 1.2], [1.2, 1.3]),
        ("max", [2.1, 2.2], [2.2, 2.3]),
    ],
)
def test_vertical_aggregations_multi_model_keeps_raw_mod_data_consistent(
    simple_vertical_comparer, method, expected_mod, expected_mod2
):
    cmp2 = _add_second_model(simple_vertical_comparer)
    agg_cmp = getattr(cmp2.vertical, method)()

    assert set(agg_cmp.mod_names) == {"mod", "mod2"}
    assert set(agg_cmp.raw_mod_data.keys()) == {"mod", "mod2"}

    assert np.allclose(agg_cmp.data["mod"].values, expected_mod)
    assert np.allclose(agg_cmp.data["mod2"].values, expected_mod2)
    assert np.allclose(agg_cmp.raw_mod_data["mod"].values, expected_mod)
    assert np.allclose(agg_cmp.raw_mod_data["mod2"].values, expected_mod2)
