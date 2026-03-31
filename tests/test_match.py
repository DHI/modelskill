from datetime import timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import mikeio
import modelskill as ms
from modelskill.comparison._comparison import ItemSelection
from modelskill.model.dfsu import DfsuModelResult


@pytest.fixture
def o1():
    fn = "tests/testdata/SW/HKNA_Hm0.dfs0"
    return ms.PointObservation(fn, item=0, x=4.2420, y=52.6887, name="HKNA")


@pytest.fixture
def o2():
    fn = "tests/testdata/SW/eur_Hm0.dfs0"
    return ms.PointObservation(fn, item=0, x=3.2760, y=51.9990, name="EPL")


@pytest.fixture
def o2_gaps():
    fn = "tests/testdata/SW/eur_Hm0.dfs0"
    obs = mikeio.read(fn, items=0).to_dataframe().rename(columns=dict(Hm0="obs")) + 1
    dt = pd.Timedelta(180, unit="s")
    obs.index = obs.index - dt
    obs.index = obs.index.round("s")
    return ms.PointObservation(obs, item=0, x=3.2760, y=51.9990, name="EPL")


@pytest.fixture
def o3():
    fn = "tests/testdata/SW/Alti_c2_Dutch.dfs0"
    return ms.TrackObservation(fn, item=3, name="c2")


@pytest.fixture
def o4():
    fn = "tests/testdata/vertical/VerticalProfile_obs1.dfs0"
    return ms.VerticalObservation(fn, z_item="z", name="vobs", x=657500, y=6553600)


@pytest.fixture
def mr12_gaps():
    fn = "tests/testdata/SW/ts_storm_4.dfs0"
    df1 = mikeio.read(fn, items=0).to_dataframe()
    df1 = df1.resample("2h").nearest()
    df1 = df1.rename(columns={df1.columns[0]: "mr1"})
    df2 = df1.copy().rename(columns=dict(mr1="mr2")) - 1

    # keep 2017-10-28 00:00 and 2017-10-29 00:00
    # but remove the 11 steps in between
    df2.loc["2017-10-28 01:00":"2017-10-28 23:00"] = np.nan
    mr1 = ms.model_result(df1, name="mr1")
    mr2 = ms.model_result(df2, name="mr2")
    return mr1, mr2


@pytest.fixture
def mr1() -> DfsuModelResult:
    fn = "tests/testdata/SW/HKZN_local_2017_DutchCoast.dfsu"
    return ms.model_result(fn, item=0, name="SW_1")


@pytest.fixture
def mr2():
    fn = "tests/testdata/SW/HKZN_local_2017_DutchCoast_v2.dfsu"
    return ms.model_result(fn, item=0, name="SW_2")


@pytest.fixture
def mr3():
    fn = "tests/testdata/SW/HKZN_local_2017_DutchCoast_v3.dfsu"
    return ms.model_result(fn, item=0, name="SW_3")


@pytest.fixture
def mr4():
    fn = "tests/testdata/vertical/VerticalModel_at_obs.dfs0"
    return ms.model_result(fn, item="Salinity", name="vmod", gtype="vertical")


@pytest.fixture
def mr5():
    fn = "tests/testdata/vertical/sigma_z_coast.dfsu"
    return ms.model_result(fn, item="Salinity", name="3dmod")


@pytest.fixture
def simple_vo():
    ind = pd.DatetimeIndex(["2020-01-01 13:00:00"] * 2 + ["2020-01-02 11:00:00"] * 2)
    data = {
        "v": [1, 2, 1.1, 2.1],
        "z": [-1, -2, -1, -2],
        "x": [20, 20, 20, 20],
        "y": [55, 55, 55, 55],
    }
    vo = ms.VerticalObservation(
        pd.DataFrame(data, index=ind), z_item="z", item="v", name="obs", x=20, y=55
    )
    return vo


@pytest.fixture
def simple_vm():
    ind = pd.DatetimeIndex(["2020-01-01 12:00:00"] * 4 + ["2020-01-02 12:00:00"] * 4)
    data = {
        "mod": [1.1, 2.1, 3.1, 4.1, 1.2, 2.2, 3.2, 4.2],
        "z": [-1, -2, -3, -4, -1, -2, -3, -4],
    }
    vm = ms.VerticalModelResult(pd.DataFrame(data, index=ind), z_item=1, item=0)
    return vm


class TestVerticalObservation:
    # ============
    # Check variations of match with vertical data
    # ============
    def test_match_dfs0_dfs0(self, o4, mr4):
        cmp = ms.match(o4, mr4)
        assert cmp.n_models == 1
        assert cmp.n_points > 0
        assert cmp.x == pytest.approx(657500)
        assert cmp.y == pytest.approx(6553600)
        assert cmp.z is not None
        assert cmp.name == "vobs"
        assert cmp.gtype == "vertical"
        assert cmp.mod_names == ["vmod"]
        assert cmp.data["vmod"].attrs["kind"] == "model"
        assert cmp.data["Observation"].attrs["kind"] == "observation"

    def test_match_dfsu_dfs0(self, o4, mr5):
        cmp = ms.match(o4, mr5)
        assert cmp.n_models == 1
        assert cmp.n_points > 0
        assert cmp.x == pytest.approx(657500)
        assert cmp.y == pytest.approx(6553600)
        assert cmp.z is not None
        assert cmp.name == "vobs"
        assert cmp.gtype == "vertical"
        assert cmp.mod_names == ["3dmod"]
        assert cmp.data["3dmod"].attrs["kind"] == "model"
        assert cmp.data["Observation"].attrs["kind"] == "observation"

    def test_match_multiple(self, o4, mr4, mr5):
        cmp = ms.match(o4, [mr4, mr5])
        assert cmp.n_models == 2
        assert cmp.x == pytest.approx(657500)
        assert cmp.y == pytest.approx(6553600)
        assert cmp.z is not None
        assert cmp.name == "vobs"
        assert cmp.gtype == "vertical"
        assert cmp.mod_names == ["vmod", "3dmod"]
        assert cmp.data["vmod"].attrs["kind"] == "model"
        assert cmp.data["3dmod"].attrs["kind"] == "model"
        assert cmp.data["Observation"].attrs["kind"] == "observation"

    def test_failing_z_and_z_item_set(self, simple_vo, simple_vm):
        cmp = ms.match(simple_vo, simple_vm)
        with pytest.raises(ValueError, match="z and z_item"):
            ms.from_matched(cmp.to_dataframe(), z_item="z", z=cmp.z)

    # ==========
    # Test from_matched
    # ==========
    def test_round_trip_from_matched_df(self, o4, mr4):
        cmp = ms.match(o4, mr4)
        cmp2 = ms.from_matched(cmp.to_dataframe(), x=cmp.x, y=cmp.y)
        assert cmp2.gtype == "vertical"
        assert cmp2.n_models == 1
        assert cmp2.n_points == cmp.n_points
        assert cmp2.x == cmp.x  # set in from_matched
        assert cmp2.y == cmp.y  # set in from_matched
        assert np.array_equal(cmp2.z, cmp.z)
        assert cmp2.name == cmp._obs_name  # not set, defaults to obs name
        assert cmp2.mod_names == cmp.mod_names

    def test_round_trip_from_matched_dfs0(self, simple_vo, simple_vm):
        cmp = ms.match(simple_vo, simple_vm)
        # create df0 from matched data and create comparer from that again
        ds = mikeio.from_pandas(cmp.to_dataframe())
        cmp2 = ms.from_matched(ds)
        assert cmp2.gtype == cmp.gtype == "vertical"
        assert cmp2.n_models == 1
        assert cmp2.n_points == cmp.n_points
        assert np.isnan(cmp2.x)  # x and y not propagate with dataframe
        assert np.isnan(cmp2.y)  # x and y not propagate with dataframe
        assert np.array_equal(cmp2.z, cmp.z)
        assert cmp2.name == "Observation"
        assert cmp2.gtype == cmp.gtype
        assert cmp2.mod_names == cmp.mod_names

    def test_round_trip_from_matched_xr(self, simple_vo, simple_vm):
        cmp = ms.match(simple_vo, simple_vm)
        cmp2 = ms.from_matched(cmp.data)

        assert cmp2.gtype == cmp.gtype == "vertical"
        assert cmp2.n_models == 1
        assert cmp2.n_points == cmp.n_points
        assert cmp2.x == cmp.x  # propagated
        assert cmp2.y == cmp.y  # propagated
        assert np.array_equal(cmp2.z, cmp.z)
        assert cmp2.name == cmp.name
        assert cmp2.gtype == cmp.gtype
        assert cmp2.mod_names == cmp.mod_names

    # ==========
    # Test correct results
    # ==========
    def test_align_same_negative_depth(self, simple_vo, simple_vm):
        vo = simple_vo
        vm = simple_vm
        cmp = ms.match(vo, vm)
        expected_mod_values = [1.1, 2.1, 1.2, 2.2]

        assert cmp.data["mod"].to_numpy() == pytest.approx(expected_mod_values)
        assert cmp.data["z"].to_numpy() == pytest.approx(simple_vo.data["z"].to_numpy())

    def test_align_unevently_sampled_obs(self, simple_vo, simple_vm):
        # drop the last point and modify last obs depth to -4 and create new VerticalObservation
        df_o = simple_vo.to_dataframe().iloc[0:-1, :]
        df_o.iloc[-1, 0] = -4
        vo = ms.VerticalObservation(df_o)
        cmp = ms.match(vo, simple_vm)
        expected_mod_values = [1.1, 2.1, 4.2]
        expected_depths = simple_vo.data["z"].to_numpy()[0:-1]
        expected_depths[-1] = -4

        assert cmp.data["mod"].to_numpy() == pytest.approx(expected_mod_values)
        assert cmp.data["z"].to_numpy() == pytest.approx(expected_depths)

    def test_align_vertical_intepolation(self, simple_vm):
        # Create observations from model df
        vo = ms.VerticalObservation(simple_vm.to_dataframe())
        # Modify model to be inbetween the obs depths and create new VerticalModelResult
        df = simple_vm.to_dataframe()
        df["z"] = df["z"] - 0.5
        # INPUT
        # mod_z = [-1.5, -2.5, -3.5, -4.5, -1.5, -2.5, -3.5, -4.5],
        # mod_v = [1.1, 2.1, 3.1, 4.1, 1.2, 2.2, 3.2, 4.2]
        # obs_z = [-1, -2, -3, -4, -1, -2, -3, -4]
        vm = ms.VerticalModelResult(df)
        cmp = ms.match(vo, vm)

        # first depth is nan in models becasuse obs outside model domain
        # obs at -2 is between model depths -1.5 and -2.5, so mod value is = 1.6
        # ...
        expected_mod_values = [1.6, 2.6, 3.6, 1.7, 2.7, 3.7]
        assert cmp.n_points == 6
        assert cmp.data["mod"].to_numpy() == pytest.approx(expected_mod_values)

    def test_same_results(self, simple_vm):
        vo = ms.VerticalObservation(simple_vm.to_dataframe())
        cmp = ms.match(vo, simple_vm)
        assert cmp.n_points == 8
        assert cmp.data["mod"].to_numpy() == pytest.approx(
            simple_vm.data["mod"].to_numpy()
        )
        assert cmp.data["z"].to_numpy() == pytest.approx(simple_vm.data["z"].to_numpy())
        assert cmp.data["Observation"].to_numpy() == pytest.approx(
            simple_vm.data["mod"].to_numpy()
        )

    # ==========
    # Skill test (MAYBE IN SKILL)
    # ==========
    # Same models give rmse = 0

    # ==========
    # Test slicing
    # ==========
    def test_slice_vertical(self, simple_vo, simple_vm):
        cmp = ms.match(simple_vo, simple_vm)
        cmp_slice = cmp.query("z>=-1.5 & z<=-0.5")
        assert cmp_slice.n_points < cmp.n_points

    # ==========
    # Edge-cases
    # ==========
    def test_no_overlap_in_z(self, simple_vo, simple_vm):
        # shift model to be outside obs range
        df = simple_vm.to_dataframe()
        df["z"] = df["z"] - 2
        vm = ms.VerticalModelResult(df)
        cmp = ms.match(simple_vo, vm)
        assert cmp.n_points == 0

    def test_only_1_model_depth_overlap(self, simple_vo, simple_vm):
        # shift model to be outside obs range except for one point
        df = simple_vm.to_dataframe()
        df["z"] = df["z"] - 1  # only match with models at z=-2 exact
        vm = ms.VerticalModelResult(df)
        cmp = ms.match(simple_vo, vm)
        assert cmp.n_points == 2


def test_properties_after_match(o1, mr1):
    cmp = ms.match(o1, mr1)
    assert cmp.n_models == 1
    assert cmp.n_points == 386
    assert cmp.x == 4.242
    assert cmp.y == 52.6887
    assert cmp.z is None
    assert cmp.name == "HKNA"
    assert cmp.gtype == "point"
    assert cmp.mod_names == ["SW_1"]


def test_properties_after_match_ts(o1):
    fn = "tests/testdata/SW/HKNA_Hm0.dfs0"
    mr = ms.PointModelResult(fn, item=0, name="SW_1")
    cmp = ms.match(o1, mr)
    assert cmp.n_models == 1
    assert cmp.n_points == 564
    assert cmp.x == 4.242
    assert cmp.y == 52.6887
    assert cmp.z is None
    assert cmp.name == "HKNA"
    assert cmp.gtype == "point"
    assert cmp.mod_names == ["SW_1"]


def test_match_multi_obs_multi_model(o1, o2, o3, mr1, mr2):
    cc = ms.match([o1, o2, o3], [mr1, mr2])
    assert cc.n_models == 2
    assert cc.n_observations == 3
    assert cc["c2"].n_points == 113
    assert cc["HKNA"].name == "HKNA"


def test_match_dataarray(o1, o3):
    fn = "tests/testdata/SW/HKZN_local_2017_DutchCoast.dfsu"
    da = mikeio.read(fn, time=slice("2017-10-28 00:00", None))[0]  # Skip warm-up period

    # Using a mikeio.DataArray instead of a Dfs file, makes it possible to select a subset of data

    cc = ms.match([o1, o3], ms.DfsuModelResult(da))
    assert cc.n_models == 1
    assert cc["c2"].n_points == 41

    da2 = mikeio.read(fn, area=[0, 2, 52, 54], time=slice("2017-10-28 00:00", None))[
        0
    ]  # Spatio/temporal subset

    cc2 = ms.match([o1, o3], ms.DfsuModelResult(da2))
    assert cc2["c2"].n_points == 19


def test_extract_gaps1(o1, mr3):
    # obs has 278 steps (2017-10-27 18:00 to 2017-10-29 18:00) (10min data with gaps)
    # model SW_3 has 5 timesteps:
    # 2017-10-27 18:00:00  1.880594
    # 2017-10-27 21:00:00  1.781904
    # 2017-10-28 00:00:00  1.819505   (not in obs)
    # 2017-10-28 03:00:00  2.119306
    # 2017-10-29 18:00:00  3.249600

    cmp = ms.match(o1, mr3)
    assert cmp.n_points == 278

    # accept only 1 hour gaps (even though the model has 3 hour timesteps)
    # expect only exact matches (4 of 5 model timesteps are in obs)
    cmp = ms.match(o1, mr3, max_model_gap=3600)
    assert cmp.n_points == 4

    # accept only 3 hour gaps
    # should disregard everything after 2017-10-28 03:00
    # except a single point 2017-10-29 18:00 (which is hit spot on)
    cmp = ms.match(o1, mr3, max_model_gap=10800)
    assert cmp.n_points == 48 + 1

    # accept gaps up to 2 days (all points should be included)
    cmp = ms.match(o1, mr3, max_model_gap=2 * 24 * 60 * 60)
    assert cmp.n_points == 278


def test_extract_gaps2(o2_gaps, mr12_gaps):
    # mr2 has no data between 2017-10-28 00:00 and 2017-10-29 00:00
    # we therefore expect the the 24 observations in this interval to be removed
    mr1, mr2 = mr12_gaps

    cmp = ms.match(o2_gaps, [mr1, mr2])
    assert cmp.data["mr1"].count() == 66
    assert cmp.data["mr2"].count() == 66

    # no gap in mr1
    cmp = ms.match(o2_gaps, mr1, max_model_gap=7200)
    assert cmp.data["mr1"].count() == 66

    # one day gap in mr2
    cmp = ms.match(o2_gaps, mr2, max_model_gap=7200)
    assert cmp.data["mr2"].count() == 42  # 66 - 24
    assert cmp.data["mr2"].sel(time="2017-10-28").count() == 0

    # will syncronize the two models,
    # so gap in one will remove points from the other
    cmp = ms.match(o2_gaps, [mr1, mr2], max_model_gap=7200)
    assert cmp.data["mr1"].count() == 42
    assert cmp.data["mr2"].count() == 42

    # the 24 hour gap (86400 seconds) in the file cannot be filled
    # with the max_model_gap=27200
    cmp = ms.match(o2_gaps, mr2, max_model_gap=27200)
    assert cmp.data["mr2"].count() == 42
    assert cmp.data["mr2"].sel(time="2017-10-28").count() == 0


def test_extract_gaps_big(o2_gaps, mr12_gaps):
    _, mr2 = mr12_gaps
    cmp = ms.match(o2_gaps, mr2, max_model_gap=86401)  # 24 hours + 1 second
    assert cmp.data["mr2"].count() == 66  # no data removed


def test_extract_gaps_small(o2_gaps, mr12_gaps):
    _, mr2 = mr12_gaps
    # with pytest.warns(UserWarning, match="No overlapping data"):
    cmp = ms.match(o2_gaps, mr2, max_model_gap=10)  # no data with that small gap
    assert cmp.n_points == 0


def test_extract_gaps_negative(o2_gaps, mr12_gaps):
    _, mr2 = mr12_gaps
    # with pytest.warns(UserWarning, match="No overlapping data"):
    cmp = ms.match(o2_gaps, mr2, max_model_gap=-10)
    assert cmp.n_points == 0


def test_match_gaps_types(o2_gaps, mr12_gaps):
    mr1, mr2 = mr12_gaps

    gap_seconds = 7200
    gaps = [
        pd.Timedelta(gap_seconds, unit="s"),
        np.timedelta64(gap_seconds, "s"),
        timedelta(seconds=gap_seconds),
    ]
    for gap in gaps:
        cmp = ms.match(o2_gaps, [mr1, mr2], max_model_gap=gap)
        assert cmp.data["mr1"].count() == 42


def test_small_multi_model_shifted_time_match():
    obs = pd.DataFrame(
        {"HKNA": [1.1, 2.0, 3.0, 4.0]}, index=pd.date_range("2017-01-01", periods=4)
    )
    mod = pd.DataFrame(
        {"Simple": [1.1, 2.0, 3.0]}, index=pd.date_range("2017-01-01", periods=3)
    )

    mod2 = pd.DataFrame(
        {"NotSimple": [2.1, 3.0, 4.0]}, index=pd.date_range("2017-01-02", periods=3)
    )

    # observation has four timesteps, but only three of them are in the Simple model and three in the NotSimple model
    # the number of overlapping points for all three datasets are 2, but three if we look at the models individually

    cmp1 = ms.match(obs=ms.PointObservation(obs), mod=ms.PointModelResult(mod))
    cmp1 = ms.match(obs=ms.PointObservation(obs), mod=ms.PointModelResult(mod))
    assert cmp1.n_points == 3

    cmp2 = ms.match(obs=ms.PointObservation(obs), mod=ms.PointModelResult(mod2))
    assert cmp2.n_points == 3

    mcmp = ms.match(
        obs=ms.PointObservation(obs),
        mod=[
            ms.PointModelResult(mod, name="foo"),
            ms.PointModelResult(mod2, name="bar"),
        ],
    )
    assert mcmp.n_points == 2


def test_matched_data_single_model():
    df = pd.DataFrame(
        {"ts_1": [1.1, 2.0, 3.0, 4.0], "sensor_a": [0.9, 2.0, 3.0, 4.1]},
        index=pd.date_range("2017-01-01", periods=4),
    )

    cmp = ms.from_matched(df, obs_item="sensor_a")
    assert cmp.n_points == 4


def test_matched_data_quantity():
    df = pd.DataFrame(
        {"ts_1": [1.1, 2.0, 3.0, 4.0], "sensor_a": [0.9, 2.0, 3.0, 4.1]},
        index=pd.date_range("2017-01-01", periods=4),
    )
    quantity = ms.Quantity(name="Water level", unit="m")
    cmp = ms.from_matched(df, obs_item="sensor_a", quantity=quantity)

    # Model and observation have the same quantity by definition
    assert cmp.quantity == quantity


def test_matched_data_name_xyz():
    df = pd.DataFrame(
        {"ts_1": [1.1, 2.0, 3.0, 4.0], "sensor_a": [0.9, 2.0, 3.0, 4.1]},
        index=pd.date_range("2017-01-01", periods=4),
    )
    cmp = ms.from_matched(df, obs_item="sensor_a", name="MyName", x=1, y=2, z=3)

    assert cmp.name == "MyName"
    assert cmp.x == 1
    assert cmp.y == 2
    assert cmp.z == 3


def test_matched_data_not_time_index():
    df = pd.DataFrame(
        {
            "ts_1": [
                1.0,
                2.0,
                3.0,
            ],
            "sensor_a": [2.0, 3.0, 4.0],
        },
    )

    cmp = ms.from_matched(df, obs_item="sensor_a")

    # scatter plot doesn't care about time
    cmp.plot.scatter()

    # skill metrics do not care about time
    sk = cmp.skill(metrics="mae")
    assert sk.loc["sensor_a", "mae"] == pytest.approx(1.0)

    cmp.plot.timeseries()


def test_matched_data_multiple_models():
    df = pd.DataFrame(
        {
            "cal_42": [1.1, 2.0, 3.0, 4.0],
            "cal_43": [0.9, 2.0, 3.0, 4.01],
            "sensor_a": [0.9, 2.0, 3.0, 4.1],
        },
        index=pd.date_range("2017-01-01", periods=4),
    )

    cmp = ms.from_matched(df[["sensor_a", "cal_42", "cal_43"]])
    assert cmp.n_points == 4
    assert cmp.n_models == 2
    assert cmp.sel(model="cal_42").mod_names == ["cal_42"]
    assert cmp.sel(model=0).mod_names == ["cal_42"]


def test_matched_data_multiple_models_additional_columns():
    df = pd.DataFrame(
        {
            "cal_42": [1.1, 2.0, 3.0, 4.0],
            "cal_43": [0.9, 2.0, 3.0, 4.01],
            "sensor_a": [0.9, 2.0, 3.0, 4.1],
            "additional": [0.9, 2.0, 3.0, 4.1],
        },
        index=pd.date_range("2017-01-01", periods=4),
    )

    cmp = ms.from_matched(df, obs_item="sensor_a", mod_items=["cal_42", "cal_43"])
    assert cmp.n_points == 4
    assert cmp.n_models == 2


def test_from_matched_dfs0():
    fn = "tests/testdata/SW/ts_storm_4.dfs0"
    cmp = ms.from_matched(fn, obs_item=0, mod_items=[1, 2, 3, 4, 5])
    assert cmp.n_points == 397
    assert cmp.n_models == 5
    assert cmp.quantity.name == "Significant wave height"
    assert cmp.quantity.unit == "m"


def test_from_matched_Dfs0():
    # Not sure if this is really needed, but let's test it anyway
    fn = "tests/testdata/SW/ts_storm_4.dfs0"
    dfs = mikeio.open(fn)
    cmp = ms.from_matched(dfs, obs_item=0, mod_items=[1, 2, 3, 4, 5])
    assert cmp.n_points == 397
    assert cmp.n_models == 5
    assert cmp.quantity.name == "Significant wave height"
    assert cmp.quantity.unit == "m"


def test_from_matched_mikeio_dataset():
    fn = "tests/testdata/SW/ts_storm_4.dfs0"
    ds = mikeio.read(fn, time=slice("2017-10-28 00:00", "2017-10-29 00:00"))
    cmp = ms.from_matched(ds, obs_item=0, mod_items=[1, 2, 3, 4, 5])
    assert cmp.n_points == 145
    assert cmp.n_models == 5
    assert cmp.quantity.name == "Significant wave height"
    assert cmp.quantity.unit == "m"


def test_trackmodelresult_and_trackobservation_uses_model_name():
    with pytest.warns(UserWarning, match="Removed 22 duplicate"):
        mr = ms.TrackModelResult(
            "tests/testdata/NorthSeaHD_extracted_track.dfs0",
            name="MyModel",
            item="Model_surface_elevation",
        )
    assert mr.name == "MyModel"

    # reuse same data, we don't care about the data here, only the name
    with pytest.warns(UserWarning, match="Removed 22 duplicate"):
        o1 = ms.TrackObservation(
            "tests/testdata/NorthSeaHD_extracted_track.dfs0",
            item="Model_surface_elevation",
            name="MyObs",
        )
    cmp = ms.match(o1, mr)
    assert cmp.mod_names == ["MyModel"]


def test_item_selection_items_are_unique():
    with pytest.raises(ValueError):
        ItemSelection(obs="foo", model=["foo", "bar"], aux=["baz"])


def test_save_comparercollection(o1, o3, tmp_path):
    fn = "tests/testdata/SW/HKZN_local_2017_DutchCoast.dfsu"
    da = mikeio.read(fn, time=slice("2017-10-28 00:00", None))[0]

    cc = ms.match([o1, o3], ms.DfsuModelResult(da))

    fn = tmp_path / "cc.msk"
    cc.save(fn)

    assert fn.exists()


def test_wind_directions():
    df = pd.DataFrame(
        {
            "obs": [359, 91, 181, 268],
            "mod": [0, 90, 180, 270],
        },
        index=pd.date_range("2017-01-01", periods=4, freq="h"),
    )

    cc = ms.from_matched(
        df,
        obs_item="obs",
        quantity=ms.Quantity("Wind direction", unit="degree", is_directional=True),
    )
    # default metrics *are* directional
    df = cc.skill().to_dataframe()
    assert df.loc["obs", "c_rmse"] == pytest.approx(1.322875655532)


def test_obs_and_mod_can_not_have_same_aux_item_names():
    obs_df = pd.DataFrame(
        {"wl": [1.0, 2.0, 3.0], "wind_speed": [1.0, 2.0, 3.0]},
        index=pd.date_range("2017-01-01", periods=3),
    )

    mod_df = pd.DataFrame(
        {"wl": [1.1, 2.0, 3.0], "wind_speed": [0.0, 0.0, 0.0]},
        index=pd.date_range("2017-01-01", periods=3),
    )

    obs = ms.PointObservation(obs_df, item="wl", aux_items=["wind_speed"])
    mod = ms.PointModelResult(mod_df, item="wl", aux_items=["wind_speed"])

    with pytest.raises(ValueError, match="aux"):
        ms.match(obs=obs, mod=mod)


def test_mod_aux_items_overlapping_names():
    obs_df = pd.DataFrame(
        {"wl": [1.0, 2.0, 3.0], "wind_speed": [1.0, 2.0, 3.0]},
        index=pd.date_range("2017-01-01", periods=3),
    )

    mod_df = pd.DataFrame(
        {"wl": [1.1, 2.0, 3.0], "wind_speed": [0.0, 0.0, 0.0]},
        index=pd.date_range("2017-01-01", periods=3),
    )

    mod2_df = pd.DataFrame(
        {"wl": [1.2, 2.1, 3.1], "wind_speed": [0.0, 0.0, 0.0]},
        index=pd.date_range("2017-01-01", periods=3),
    )

    obs = ms.PointObservation(obs_df, item="wl")
    mod = ms.PointModelResult(mod_df, item="wl", aux_items=["wind_speed"], name="local")

    # this is ok
    mod2 = ms.PointModelResult(
        mod2_df, item="wl", aux_items=["wind_speed"], name="remote"
    )

    # we don't care which model the aux data comes from, last one wins
    cmp = ms.match(obs=obs, mod=[mod, mod2])

    assert "wind_speed" in cmp


def test_multiple_obs_not_allowed_with_non_spatial_modelresults():
    o1 = ms.PointObservation(
        pd.DataFrame(
            {"wl": [1.0, 2.0]}, index=pd.date_range("2000", freq="h", periods=2)
        ),
        name="o1",
        x=1,
        y=2,
    )
    o2 = ms.PointObservation(
        pd.DataFrame(
            {"wl": [1.0, 2.0]}, index=pd.date_range("2000", freq="h", periods=2)
        ),
        name="o2",
        x=2,
        y=3,
    )
    m1 = ms.PointModelResult(
        pd.DataFrame(
            {"wl": [1.0, 2.0]}, index=pd.date_range("2000", freq="h", periods=2)
        ),
        name="m1",
        x=1,
        y=2,
    )
    m2 = ms.PointModelResult(
        pd.DataFrame(
            {"wl": [1.0, 2.0]}, index=pd.date_range("2000", freq="h", periods=2)
        ),
        name="m2",
        x=2,
        y=3,
    )
    m3 = ms.PointModelResult(
        pd.DataFrame(
            {"wl": [1.0, 2.0]}, index=pd.date_range("2000", freq="h", periods=2)
        ),
        name="m3",
        x=3,
        y=4,
    )

    # a single observation and model is ok
    cmp = ms.match(obs=o1, mod=[m1, m2])
    assert "m1" in cmp.mod_names
    assert "m2" in cmp.mod_names

    # but this is not allowed
    with pytest.raises(ValueError, match="SpatialField type"):
        ms.match(obs=[o1, o2], mod=[m1, m2, m3])


def test_compare_model_vs_dummy(mr1, o1):
    mean_obs = o1.trim(mr1.time[0], mr1.time[-1]).values.mean()

    mr2 = ms.DummyModelResult(data=mean_obs, name="dummy")
    assert "constant" in repr(mr2)

    cmp = ms.match(obs=o1, mod=[mr1, mr2])
    assert cmp.score(metric="r2")["dummy"] == pytest.approx(0.0)


def test_compare_model_vs_dummy_for_track(mr1, o3):
    mr = ms.DummyModelResult(name="dummy", strategy="mean")
    assert "mean" in repr(mr)

    cmp = ms.match(obs=o3, mod=mr)
    assert cmp.score(metric="r2")["dummy"] == pytest.approx(0.0)

    assert cmp.score()["dummy"] == pytest.approx(1.140079520671913)

    cmp2 = ms.match(obs=o3, mod=[mr1, mr])

    # not identical to above since it is evaluated on a subset of the data
    assert cmp2.score()["dummy"] == pytest.approx(1.225945)

    # better than dummy 🙂
    assert cmp2.score()["SW_1"] == pytest.approx(0.3524703)


def test_match_obs_model_pos_args_wrong_order_helpful_error_message():
    # match is pretty helpful in converting strings or dataset
    # so we need to use a ModelResult to trigger the error
    mr = ms.PointModelResult(
        data=pd.Series([0.0, 0.0], index=pd.date_range("1970", periods=2, freq="d")),
        name="Zero",
    )
    obs = ms.PointObservation(
        data=pd.Series(
            [1.0, 2.0, 3.0], index=pd.date_range("1970", periods=3, freq="h")
        ),
        name="MyStation",
    )

    with pytest.raises(TypeError, match="order"):
        ms.match(mr, obs)


def test_multiple_models_same_name(tmp_path: Path) -> None:
    obs = ms.PointObservation(
        "tests/testdata/SW/HKNA_Hm0.dfs0", item=0, x=4.2420, y=52.6887, name="HKNA"
    )

    modelfilebase = "tests/testdata/SW/HKZN_local_2017_DutchCoast.dfsu"

    # copy model file to two new directories, representing two different calibration runs
    dir1 = tmp_path / "cal1"
    dir1.mkdir()
    dir2 = tmp_path / "cal2"
    dir2.mkdir()

    import shutil

    shutil.copy(modelfilebase, dir1)
    shutil.copy(modelfilebase, dir2)

    # named models
    mr1 = ms.DfsuModelResult(
        dir1 / "HKZN_local_2017_DutchCoast.dfsu", item=0, name="cal1"
    )
    mr2 = ms.DfsuModelResult(
        dir2 / "HKZN_local_2017_DutchCoast.dfsu", item=0, name="cal2"
    )

    cmp = ms.match(obs, [mr1, mr2])

    assert len(cmp.mod_names) == 2

    # unnamed models
    mr1 = ms.DfsuModelResult(dir1 / "HKZN_local_2017_DutchCoast.dfsu", item=0)
    mr2 = ms.DfsuModelResult(dir2 / "HKZN_local_2017_DutchCoast.dfsu", item=0)

    with pytest.raises(ValueError, match="HKZN_local_2017_DutchCoast"):
        ms.match(obs, [mr1, mr2])
