import pandas as pd
import pytest

import modelskill as ms


def test_profile_obs_from_pandas_df():
    df = pd.read_csv("tests/testdata/hano_profile.csv", parse_dates=["date"]).rename(
        columns={"date": "time"}
    )

    # this doesn't make sense for profile data, but is necessary to work with ModelSkill
    df = df.set_index("time")

    obs = ms.ProfileObservation(
        df,
        item="temperature",
        z_item="depth",
        x=55.6177,
        y=14.8672,
        name="Hano",
        quantity=ms.Quantity("Temperature", "degreeC"),
    )
    assert obs.n_points == 291
    assert len(obs.data.z) == 291
    assert obs.name == "Hano"
    assert obs.quantity.name == "Temperature"
    assert obs.quantity.unit == "degreeC"

    # Temporary assert (too intimate with the data)
    # expressed in nushell
    # $ open hano_profile.csv | where date == "2019-02-23" and depth == 142                                                                                   01/29/24 10:19:25 AM
    #    ╭───┬────────────┬───────┬─────────────┬──────────╮
    #    │ # │    date    │ depth │ temperature │ salinity │
    #    ├───┼────────────┼───────┼─────────────┼──────────┤
    #    │ 0 │ 2019-02-23 │   142 │        9.87 │    16.14 │
    #    ╰───┴────────────┴───────┴─────────────┴──────────╯

    long_df = obs.data.to_dataframe().reset_index()
    predicate = (long_df.z == 142) & (long_df.time == "2019-02-23")
    assert long_df[predicate].salinity.iloc[0] == pytest.approx(16.136002)

    # import polars as pl

    # pl_df = pl.from_pandas(long_df)

    # val = (
    #     pl_df.filter(pl.col("z") == 142)
    #          .filter(pl.col("time").dt.date() == datetime.date(2019, 2, 23))
    #          .select("salinity")[0, 0]
    # )
    # assert val == pytest.approx(16.136002)

    # same but for salinity
    obs2 = ms.ProfileObservation(
        df,
        item="salinity",
        z_item="depth",
        x=55.6177,
        y=14.8672,
        name="Hano",
        quantity=ms.Quantity("Salinity", "psu"),
    )

    assert obs2.n_points == 291
    assert obs2.quantity.name == "Salinity"
    assert obs2.quantity.unit == "psu"
