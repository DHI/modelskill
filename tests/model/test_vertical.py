from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import modelskill as ms
import mikeio


# FIXME: MAKE INTO CLASS
# ================
# Data fixtures
# ================
@pytest.fixture
def vertical_model_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "z": [-5.0, -4.0, -3.0, -5.0, -4.0, -3.0],
            "Salinity": [30.0, 31.0, 32.0, 30.5, 31.5, 32.5],
        },
        index=pd.to_datetime(
            [
                "2019-01-01 00:00:00",
                "2019-01-01 00:00:00",
                "2019-01-01 00:00:00",
                "2019-01-01 01:00:00",
                "2019-01-01 01:00:00",
                "2019-01-01 01:00:00",
            ]
        ),
    )


@pytest.fixture
def vertical_model_df_duplicates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "z": [-5.0, -5.0, -4.0, -4.0, -3.0],
            "Salinity": [30.0, 300.0, 31.0, 310.0, 32.0],
        },
        index=pd.to_datetime(
            [
                "2019-01-01 00:00:00",
                "2019-01-01 00:00:00",
                "2019-01-01 00:00:00",
                "2019-01-01 00:00:00",
                "2019-01-01 01:00:00",
            ]
        ),
    )


@pytest.fixture
def vertical_model_df_aux() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "z": [-5.0, -4.0, -3.0],
            "Salinity": [30.0, 31.0, 32.0],
            "Temperature": [8.0, 8.2, 8.4],
            "Density": [1025.0, 1025.1, 1025.2],
        },
        index=[pd.Timestamp("2019-01-01")] * 3,
    )


@pytest.fixture
def dfsu_fpath() -> str:
    return "tests/testdata/vertical/sigma_z_coast.dfsu"


@pytest.fixture
def dfsu_ds(dfsu_fpath) -> mikeio.Dataset:
    return mikeio.read(dfsu_fpath)


@pytest.fixture
def dfs0_fpath() -> str:
    return "tests/testdata/vertical/VerticalProfile_obs1.dfs0"


@pytest.fixture
def dfs0_ds(dfs0_fpath) -> mikeio.Dataset:
    return mikeio.read(dfs0_fpath)


# ================
# Test basic open for different formats
# ================
# dataframe input
def test_vertical_model_result_from_dataframe(vertical_model_df):
    mr = ms.VerticalModelResult(
        vertical_model_df,
        item="Salinity",
        z_item="z",
        x=12.0,
        y=55.0,
        name="test",
    )

    assert isinstance(mr, ms.VerticalModelResult)
    assert mr.gtype == "vertical"
    assert mr.name == "test"
    assert mr.x == pytest.approx(12.0)
    assert mr.y == pytest.approx(55.0)
    assert mr.n_points == len(vertical_model_df)
    assert mr.data[mr.name].attrs["kind"] == "model"


# dfs0 dataset path input
def test_vertical_model_result_from_dfs0_path(dfs0_fpath):
    mr = ms.VerticalModelResult(dfs0_fpath, z_item="z", item="Salinity", name="test")

    assert isinstance(mr, ms.VerticalModelResult)
    assert mr.gtype == "vertical"
    assert mr.name == "test"
    assert mr.n_points > 0
    assert not np.isnan(mr.z).any()
    assert np.isnan(mr.x)
    assert np.isnan(mr.y)

    # with x, y
    mr = ms.VerticalModelResult(
        dfs0_fpath, z_item="z", item="Salinity", name="test", x=12.0, y=55.0
    )
    assert isinstance(mr, ms.VerticalModelResult)
    assert mr.gtype == "vertical"
    assert mr.name == "test"
    assert mr.n_points > 0
    assert mr.x == pytest.approx(12.0)
    assert mr.y == pytest.approx(55.0)


# dfs0 dataset path input
def test_vertical_model_result_from_dfs0_dataset(dfs0_ds):
    mr = ms.VerticalModelResult(dfs0_ds, name="test")

    assert isinstance(mr, ms.VerticalModelResult)
    assert mr.gtype == "vertical"
    assert mr.name == "test"
    assert mr.n_points > 0
    assert not np.isnan(mr.z).any()
    assert np.isnan(mr.x)
    assert np.isnan(mr.y)
    # with x, y
    mr = ms.VerticalModelResult(dfs0_ds, name="test", x=12.0, y=55.0)
    assert isinstance(mr, ms.VerticalModelResult)
    assert mr.gtype == "vertical"
    assert mr.name == "test"
    assert mr.n_points > 0
    assert mr.x == pytest.approx(12.0)
    assert mr.y == pytest.approx(55.0)


# ================
# Test failing and optional args
# ================
# failing without z_item
def test_fail_with_3_items_no_item_arg(dfs0_ds):
    ds_test = dfs0_ds.copy()
    ds_test["extra_item"] = ds_test[1].copy()
    with pytest.raises(ValueError, match="Input has more than 2 items, but"):
        _ = ms.VerticalModelResult(ds_test)


# failing z wronge location
def test_item_named_z(dfs0_ds):
    ds_test = mikeio.Dataset(
        [dfs0_ds[1], dfs0_ds[0]],
    )
    with pytest.raises(ValueError, match="name 'z' is reserved "):
        _ = ms.VerticalModelResult(ds_test)


# ===============
# test arguments options for handling duplicates
# ===============
@pytest.mark.parametrize(
    "keep_duplicates,expected_removed,expected_z,expected_values",
    [
        ("first", 2, [-5.0, -4.0, -3.0], [30.0, 31.0, 32.0]),
        ("last", 2, [-5.0, -4.0, -3.0], [300.0, 310.0, 32.0]),
        (False, 4, [-3.0], [32.0]),
    ],
)
def test_vertical_model_keep_duplicates_modes(
    vertical_model_df_duplicates,
    keep_duplicates,
    expected_removed,
    expected_z,
    expected_values,
):
    with pytest.warns(UserWarning, match=f"Removed {expected_removed} duplicate"):
        mr = ms.VerticalModelResult(
            vertical_model_df_duplicates,
            item="Salinity",
            z_item="z",
            x=12.0,
            y=55.0,
            keep_duplicates=keep_duplicates,
        )

    assert list(mr.data["z"].values) == expected_z
    assert list(mr.data[mr.name].values) == expected_values


# aux items
def test_vertical_model_aux_items_preserved_and_tagged(vertical_model_df_aux):
    mr = ms.VerticalModelResult(
        vertical_model_df_aux,
        item="Salinity",
        z_item="z",
        x=12.0,
        y=55.0,
        aux_items=["Temperature", "Density"],
    )

    for aux_name in ["Temperature", "Density"]:
        assert aux_name in mr.data.data_vars
        assert mr.data[aux_name].attrs["kind"] == "aux"

    df = mr.to_dataframe()
    assert "z" in df.columns
    assert "Salinity" in df.columns
    assert "Temperature" in df.columns
    assert "Density" in df.columns


# ================
# Test roundtrips and identical parsing
# ================
def test_vertical_model_roundtrip_from_dataset(vertical_model_df):
    mr = ms.VerticalModelResult(
        vertical_model_df,
        item="Salinity",
        z_item="z",
        x=12.0,
        y=55.0,
        name="salt_model",
    )
    mr2 = ms.VerticalModelResult(mr.data)

    assert mr.equals(mr2)
    assert mr2.gtype == mr.gtype
    assert mr2.name == mr.name


# ================
# Test extract profile and align to obs profiles from dfsu
# ===============
# TODO: During phase 3

# def test_extract_from_dfsu(dfsu_ds):
#     dfsu_mr = ms.DfsuModelResult(dfsu_ds, item=0, name="test")
#     dummy_obs = pd.DataFrame(
#         {"z": [-5.0, -4.0, -3.0], "salt": [30.0, 31.0, 32.0]},
#         index=pd.to_datetime(["2022-06-14 00:00:00"] * 3),
#     )
#     XPOS = 6.575e5
#     YPOS = 6.55e6
#     vo = ms.VerticalObservation(dummy_obs, x=XPOS, y=YPOS, item="salt", z_item="z")
#     vmr = dfsu_mr.extract(vo, spatial_method="contained")

#     # approximate selected column coordinates from dfsu geometry
#     dfsu_col = dfsu_ds.sel(x=XPOS, y=YPOS)
#     x_mod_expected_close = dfsu_col.geometry.element_coordinates[0, 0]
#     y_mod_expected_close = dfsu_col.geometry.element_coordinates[0, 1]

#     assert isinstance(vmr, ms.VerticalModelResult)
#     assert vmr.gtype == "vertical"
#     assert vmr.name == "test"
#     assert vmr.n_points > 0
#     assert vmr.x == pytest.approx(x_mod_expected_close)
#     assert vmr.y == pytest.approx(y_mod_expected_close)


# def test_align_to_obs_profiles_matches_nearest_times_and_interpolates_depths():
#     obs_df = pd.DataFrame(
#         {
#             "z": [-5.0, -4.0, -2.0, -5.0, -4.0, -2.0, -5.0],
#             "salt": [30.0, 31.0, 33.0, 30.5, 31.5, 33.5, 34.0],
#         },
#         index=pd.to_datetime(
#             [
#                 "2019-01-01 00:10:00",
#                 "2019-01-01 00:10:00",
#                 "2019-01-01 00:10:00",
#                 "2019-01-01 01:10:00",
#                 "2019-01-01 01:10:00",
#                 "2019-01-01 01:10:00",
#                 "2019-01-01 02:00:00",
#             ]
#         ),
#     )

#     observation = ms.VerticalObservation(
#         obs_df,
#         item="salt",
#         z_item="z",
#         x=12.0,
#         y=55.0,
#     )

#     model_df = pd.DataFrame(
#         {
#             "z": [-5.0, -3.0, -1.0, -5.0, -3.0, -1.0, -5.0],
#             "Salinity": [29.0, 31.0, 33.0, 29.5, 31.5, 33.5, 35.0],
#         },
#         index=pd.to_datetime(
#             [
#                 "2019-01-01 00:00:00",
#                 "2019-01-01 00:00:00",
#                 "2019-01-01 00:00:00",
#                 "2019-01-01 01:00:00",
#                 "2019-01-01 01:00:00",
#                 "2019-01-01 01:00:00",
#                 "2019-01-01 03:00:00",
#             ]
#         ),
#     )

#     model = ms.VerticalModelResult(
#         model_df,
#         item="Salinity",
#         z_item="z",
#         x=12.0,
#         y=55.0,
#     )

#     aligned = model.align_to_obs_profiles(observation)
#     print(aligned)

#     assert isinstance(aligned, xr.Dataset)
#     assert set(aligned.data_vars) == {"obs", "mod"}
#     assert list(aligned.time.values) == [
#         np.datetime64("2019-01-01T00:10:00.000000000"),
#         np.datetime64("2019-01-01T01:10:00.000000000"),
#         np.datetime64("2019-01-01T02:00:00.000000000"),
#     ]
#     assert list(aligned.z.values) == [-5.0, -4.0, -2.0]
#     assert np.allclose(
#         aligned["obs"].sel(time="2019-01-01 00:10:00").values, [30.0, 31.0, 33.0]
#     )
#     assert np.allclose(
#         aligned["obs"].sel(time="2019-01-01 01:10:00").values, [30.5, 31.5, 33.5]
#     )
#     assert np.allclose(
#         aligned["mod"].sel(time="2019-01-01 00:10:00").values, [29.0, 30.0, 32.0]
#     )
#     assert np.allclose(
#         aligned["mod"].sel(time="2019-01-01 01:10:00").values, [29.5, 30.5, 32.5]
#     )
#     assert aligned["obs"].sel(
#         time="2019-01-01 02:00:00", z=-5.0
#     ).item() == pytest.approx(34.0)
#     assert aligned["mod"].sel(
#         time="2019-01-01 02:00:00", z=-5.0
#     ).item() == pytest.approx(35.0)
