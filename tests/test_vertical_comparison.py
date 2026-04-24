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
    assert list(df.index) == ["2.0-1.5", "1.5-1.0"]
    assert np.all(df["n"].to_numpy() == [2, 2])
    assert df.loc["2.0-1.5", "rmse"] == pytest.approx(0.1)
    assert df.loc["1.5-1.0", "rmse"] == pytest.approx(0.1)


def test_vertical_skill_with_explicit_bins(simple_vertical_comparer):
    sk = simple_vertical_comparer.vertical.skill(
        bins=[(-2.1, -1.9), (-1.1, -0.9)],
        metrics="rmse",
    )

    assert sk is not None
    df = sk.to_dataframe()
    assert list(df.index) == ["2.1-1.9", "1.1-0.9"]
    assert np.all(df["n"].to_numpy() == [2, 2])
    assert np.allclose(df["rmse"].to_numpy(), [0.1, 0.1])


def test_vertical_skill_with_binsize(simple_vertical_comparer):
    sk = simple_vertical_comparer.vertical.skill(binsize=0.5, metrics="rmse")

    assert sk is not None
    df = sk.to_dataframe()
    assert df.loc["2.0-1.5", "n"] == 2
    assert df.loc["2.0-1.5", "rmse"] == pytest.approx(0.1)


def test_vertical_skill_raises_for_single_bin(simple_vertical_comparer):
    with pytest.raises(ValueError, match="Only one depth bin found"):
        simple_vertical_comparer.vertical.skill(bins=1, metrics="rmse")


def test_vertical_skill_raises_for_none_bins(simple_vertical_comparer):
    with pytest.raises(ValueError, match="bins cannot be None"):
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
