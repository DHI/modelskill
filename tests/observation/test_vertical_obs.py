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
            obs = ms.observation(fn, z_item="z")
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
            obs = ms.VerticalObservation(
                df,
                item="value",
                z_item="z",
                x=12.0,
                y=55.0,
            )

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
