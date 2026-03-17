import numpy as np
import pandas as pd
import pytest
import modelskill as ms
import mikeio


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
def dfs0_fpath() -> str:
    return "tests/testdata/vertical/VerticalProfile_obs1.dfs0"


@pytest.fixture
def dfs0_ds(dfs0_fpath) -> mikeio.Dataset:
    return mikeio.read(dfs0_fpath)


@pytest.fixture
def dfsu_fpath() -> str:
    return "tests/testdata/vertical/sigma_z_coast.dfsu"


@pytest.fixture
def dfsu_ds(dfsu_fpath) -> mikeio.Dataset:
    return mikeio.read(dfsu_fpath)


class TestVerticalModelResult:
    # ================
    # Test basic open for different formats
    # ================
    @pytest.mark.parametrize(
        "input_fixture", ["vertical_model_df", "dfs0_ds", "dfs0_fpath"]
    )
    def test_open_and_parse(self, request, input_fixture):
        input = request.getfixturevalue(input_fixture)
        mr = ms.VerticalModelResult(input, z_item="z", item="Salinity", name="test")

        assert isinstance(mr, ms.VerticalModelResult)
        assert mr.gtype == "vertical"
        assert mr.name == "test"
        assert mr.n_points > 0
        assert not np.isnan(mr.z).any()
        assert np.isnan(mr.x)
        assert np.isnan(mr.y)

        # with x, y
        mr = ms.VerticalModelResult(
            input, z_item="z", item="Salinity", name="test", x=12.0, y=55.0
        )
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
    def test_fail_with_3_items_no_item_arg(self, dfs0_ds):
        ds_test = dfs0_ds.copy()
        ds_test["extra_item"] = ds_test[1].copy()
        with pytest.raises(ValueError, match="Input has more than 2 items, but"):
            _ = ms.VerticalModelResult(ds_test)

    # failing z wronge location
    def test_item_named_z(self, dfs0_ds):
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
        self,
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
    def test_vertical_model_aux_items_preserved_and_tagged(self, vertical_model_df_aux):
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
    def test_vertical_model_roundtrip_from_dataset(self, vertical_model_df):
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

        # Check that changing name of original does not change roundtripped
        mr.name = "changed_name"
        assert mr2.name == "salt_model"

    # ================
    # Test extract profile and align to obs profiles from dfsu
    # ===============
    def test_extract_from_dfsu(self, dfsu_ds):
        dfsu_mr = ms.DfsuModelResult(dfsu_ds, item=0, name="test")
        dummy_obs = pd.DataFrame(
            {"z": [-5.0, -4.0, -3.0], "salt": [30.0, 31.0, 32.0]},
            index=pd.to_datetime(["2022-06-14 00:00:00"] * 3),
        )
        XPOS = 6.575e5
        YPOS = 6.55e6
        vo = ms.VerticalObservation(dummy_obs, x=XPOS, y=YPOS, item="salt", z_item="z")
        vmr = dfsu_mr.extract(vo, spatial_method="contained")

        # approximate selected column coordinates from dfsu geometry
        dfsu_col = dfsu_ds.sel(x=XPOS, y=YPOS)
        x_mod_expected_close = dfsu_col.geometry.element_coordinates[0, 0]
        y_mod_expected_close = dfsu_col.geometry.element_coordinates[0, 1]

        assert isinstance(vmr, ms.VerticalModelResult)
        assert vmr.gtype == "vertical"
        assert vmr.name == "test"
        assert vmr.n_points > 0
        assert vmr.x == pytest.approx(x_mod_expected_close)
        assert vmr.y == pytest.approx(y_mod_expected_close)

    def test_extract_from_dfsu_correct_layers(self, dfsu_ds):
        dfsu_mr = ms.DfsuModelResult(dfsu_ds, item=0, name="test")
        dummy_obs = pd.DataFrame(
            {"z": [-5.0, -4.0, -3.0], "salt": [30.0, 31.0, 32.0]},
            index=pd.to_datetime(["2022-06-14 00:00:00"] * 3),
        )
        XPOS = 6.575e5
        YPOS = 6.55e6
        vo = ms.VerticalObservation(dummy_obs, x=XPOS, y=YPOS, item="salt", z_item="z")
        vmr = dfsu_mr.extract(vo, spatial_method="contained")

        # number layers at location from sel
        dfsu_col = dfsu_ds.sel(x=XPOS, y=YPOS)
        n_layers_expected = dfsu_col.geometry.n_layers

        # number of layers in VerticalModelResult
        ntimes = len(np.unique(vmr.data.time.values))
        n_layers = int(len(vmr.data.z.values) / ntimes)

        assert n_layers == n_layers_expected

    @pytest.mark.parametrize("spatial_method", ["nearest", "inverse_distance"])
    def test_extract_from_dfsu_unsupported_spatial_methods_raise(
        self, dfsu_ds, spatial_method
    ):
        dfsu_mr = ms.DfsuModelResult(dfsu_ds, item=0, name="test")
        dummy_obs = pd.DataFrame(
            {"z": [-5.0, -4.0, -3.0], "salt": [30.0, 31.0, 32.0]},
            index=pd.to_datetime(["2022-06-14 00:00:00"] * 3),
        )
        xpos = 6.575e5
        ypos = 6.55e6
        vo = ms.VerticalObservation(
            dummy_obs,
            x=xpos,
            y=ypos,
            item="salt",
            z_item="z",
        )

        with pytest.raises(
            NotImplementedError,
            match="Only spatial_method='contained' is currently implemented",
        ):
            _ = dfsu_mr.extract(vo, spatial_method=spatial_method)

    def test_extract_from_dfsu_obs_outside_domain(self, dfsu_ds):
        dfsu_mr = ms.DfsuModelResult(dfsu_ds, item=0, name="test")
        dummy_obs = pd.DataFrame(
            {"z": [-5.0, -4.0, -3.0], "salt": [30.0, 31.0, 32.0]},
            index=pd.to_datetime(["2022-06-14 00:00:00"] * 3),
        )
        XPOS = 1
        YPOS = 1
        vo = ms.VerticalObservation(dummy_obs, x=XPOS, y=YPOS, item="salt", z_item="z")
        with pytest.raises(ValueError, match="outside model domain"):
            _ = dfsu_mr.extract(vo, spatial_method="contained")
