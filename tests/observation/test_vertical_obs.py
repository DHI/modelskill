from pathlib import Path
import pandas as pd
import pytest
import modelskill as ms


@pytest.fixture
def _vertical_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "z": [-5.0, -4.0, -3.0],
            "value": [1.0, 1.1, 1.2],
        },
        index=[pd.Timestamp("2019-01-01")] * 3,
    )


@pytest.fixture
def _vertical_df_duplicates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "z": [-5.0, -5.0, -4.0, -4.0, -3.0],
            "value": [1.0, 10.0, 2.0, 20.0, 7.0],
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
def _vertical_df_aux() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "z": [-5.0, -4.0, -3.0],
            "value": [1.0, 1.1, 1.2],
            "aux1": [10.0, 11.0, 12.0],
            "aux2": [20.0, 21.0, 22.0],
        },
        index=[pd.Timestamp("2019-01-01")] * 3,
    )


class TestVerticalObservation:
    def test_vertical_observation_factory_from_kwargs(self, _vertical_df):
        obs = ms.observation(
            _vertical_df,
            item="value",
            z_item="z",
            x=12.0,
            y=55.0,
        )
        assert isinstance(obs, ms.VerticalObservation)

    def test_name_kwarg(self, _vertical_df):
        obs = ms.observation(
            _vertical_df,
            item="value",
            z_item="z",
            x=12.0,
            y=55.0,
            name="argname",
        )
        assert obs.name == "argname"
        # check xarray name
        assert list(obs.data.data_vars.keys())[0] == "argname"

    def test_factory_from_gtype(self, _vertical_df):
        obs = ms.observation(
            _vertical_df,
            gtype="vertical",
            item="value",
            z_item="z",
            x=12.0,
            y=55.0,
        )
        assert isinstance(obs, ms.VerticalObservation)

    def test_sel_by_z_scalar(self, _vertical_df):
        obs = ms.VerticalObservation(
            _vertical_df,
            item="value",
            z_item="z",
            x=12.0,
            y=55.0,
        )

        # out = obs.sel(z=-4.0)  # only works with xarray >= 2026.01.0
        out = ms.VerticalObservation(obs.data.where(obs.data["z"] == -4.0, drop=True))

        assert isinstance(out, ms.VerticalObservation)
        assert len(out.data) == 1
        assert out.data.value == 1.1

    def test_open_dfs0_equal(self):
        fn = Path("tests/testdata/vertical/VerticalProfile_obs2.dfs0")
        obs = ms.observation(fn, z_item="z")
        obs2 = ms.VerticalObservation(fn)
        assert isinstance(obs, ms.VerticalObservation)
        assert obs.equals(obs2)

    def test_with_and_without_item_arg(self):
        fn = Path("tests/testdata/vertical/VerticalProfile_ST.dfs0")
        # no item specified, but multiple items in file
        with pytest.raises(ValueError):
            _ = ms.observation(fn, z_item="z")
        # below should be fine...only one item
        fn = Path("tests/testdata/vertical/VerticalProfile_obs1.dfs0")
        assert isinstance(ms.observation(fn, z_item="z"), ms.VerticalObservation)

    def test_duplicated_time_z_pairs(self):
        df = pd.DataFrame(
            {
                "z": [-5.0, -4.0, -4.0],
                "value": [1.0, 1.1, 1.2],
            },
            index=[pd.Timestamp("2019-01-01")] * 3,
        )
        with pytest.warns(UserWarning, match="Removed 1 duplicate"):
            _ = ms.VerticalObservation(
                df,
                item="value",
                z_item="z",
                x=12.0,
                y=55.0,
            )

    def test_keep_duplicates_last_is_applied(self, _vertical_df_duplicates):
        with pytest.warns(UserWarning, match="Removed 2 duplicate"):
            obs = ms.VerticalObservation(
                _vertical_df_duplicates,
                item="value",
                z_item="z",
                x=12.0,
                y=55.0,
                keep_duplicates="last",
            )

        assert list(obs.data["z"].values) == [-5.0, -4.0, -3.0]
        assert list(obs.data["value"].values) == [10.0, 20.0, 7.0]

    @pytest.mark.parametrize(
        "keep_duplicates,expected_removed,expected_z,expected_values",
        [
            ("first", 2, [-5.0, -4.0, -3.0], [1.0, 2.0, 7.0]),
            ("last", 2, [-5.0, -4.0, -3.0], [10.0, 20.0, 7.0]),
            (False, 4, [-3.0], [7.0]),
        ],
    )
    def test_keep_duplicates_modes(
        self,
        _vertical_df_duplicates,
        keep_duplicates,
        expected_removed,
        expected_z,
        expected_values,
    ):
        with pytest.warns(UserWarning, match=f"Removed {expected_removed} duplicate"):
            obs = ms.VerticalObservation(
                _vertical_df_duplicates,
                item="value",
                z_item="z",
                x=12.0,
                y=55.0,
                keep_duplicates=keep_duplicates,
            )

        assert list(obs.data["z"].values) == expected_z
        assert list(obs.data["value"].values) == expected_values

    def test_single_item_input_raises(self):
        df = pd.DataFrame(
            {"value": [1.0, 1.1, 1.2]},
            index=[pd.Timestamp("2019-01-01")] * 3,
        )

        with pytest.raises(ValueError, match="at least 2"):
            ms.VerticalObservation(df, item="value", z_item="z", x=12.0, y=55.0)

    def test_more_than_two_items_without_item_raises(self):
        df = pd.DataFrame(
            {
                "z": [-5.0, -4.0, -3.0],
                "value1": [1.0, 1.1, 1.2],
                "value2": [2.0, 2.1, 2.2],
            },
            index=[pd.Timestamp("2019-01-01")] * 3,
        )

        with pytest.raises(ValueError, match="item was not given"):
            ms.VerticalObservation(df, z_item="z", x=12.0, y=55.0)

    def test_duplicate_item_specification_raises(self, _vertical_df_aux):
        with pytest.raises(ValueError, match="Duplicate items"):
            ms.VerticalObservation(
                _vertical_df_aux,
                item="value",
                z_item="z",
                x=12.0,
                y=55.0,
                aux_items=["value"],
            )

    @pytest.mark.parametrize(
        "aux_items,expected_aux",
        [
            ("aux1", ["aux1"]),
            (["aux1", "aux2"], ["aux1", "aux2"]),
        ],
    )
    def test_aux_items_preserved_and_tagged(
        self, _vertical_df_aux, aux_items, expected_aux
    ):
        obs = ms.VerticalObservation(
            _vertical_df_aux,
            item="value",
            z_item="z",
            x=12.0,
            y=55.0,
            aux_items=aux_items,
        )

        for aux_name in expected_aux:
            assert aux_name in obs.data.data_vars
            assert obs.data[aux_name].attrs["kind"] == "aux"

        df = obs.to_dataframe()
        for aux_name in expected_aux:
            assert aux_name in df.columns

    def test_roundtrip_from_dataset_preserves_vertical_observation(self, _vertical_df):
        obs = ms.VerticalObservation(
            _vertical_df,
            item="value",
            z_item="z",
            x=12.0,
            y=55.0,
            attrs={"station": "A"},
        )
        obs2 = ms.VerticalObservation(obs.data)

        assert obs.equals(obs2)
        assert obs2.attrs["gtype"] == obs.attrs["gtype"]
        assert obs2.attrs["station"] == "A"
        assert obs2.name == obs.name

    def test_to_dataframe(self, _vertical_df):
        obs = ms.VerticalObservation(
            _vertical_df,
            item="value",
            z_item="z",
            x=12.0,
            y=55.0,
        )
        df = obs.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["z", "value"]
