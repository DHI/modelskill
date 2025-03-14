import numpy as np
import pytest
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from modelskill.comparison import Comparer
from modelskill import __version__
import modelskill as ms


@pytest.fixture
def pt_df() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "Observation": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "m1": [1.5, 2.4, 3.6, 4.9, 5.6, 6.4],
            "m2": [1.1, 2.2, 3.1, 4.2, 4.9, 6.2],
            "time": pd.date_range("2019-01-01", periods=6, freq="D"),
        }
    ).set_index("time")
    return df


def _get_track_df() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "Observation": [1.0, 2.0, 3.0, 4.0, 5.0, np.nan],
            "x": [10.1, 10.2, 10.3, 10.4, 10.5, 10.6],
            "y": [55.1, 55.2, 55.3, 55.4, 55.5, 55.6],
            "m1": [1.5, 2.4, 3.6, 4.9, 5.6, 6.4],
            "m2": [1.1, 2.2, 3.1, 4.2, 4.9, 6.2],
        },
        index=pd.date_range("2019-01-01", periods=6, freq="D"),
    )
    df.index.name = "time"
    return df


def _set_attrs(data: xr.Dataset) -> xr.Dataset:
    data["Observation"].attrs["kind"] = "observation"
    data["Observation"].attrs["units"] = "m"
    data["Observation"].attrs["long_name"] = "fake var"
    data["m1"].attrs["kind"] = "model"
    data["m2"].attrs["kind"] = "model"
    data.attrs["modelskill_version"] = __version__
    return data


@pytest.fixture
def pc() -> Comparer:
    """A comparer with fake point data"""
    x, y = 10.0, 55.0
    df = _get_track_df().drop(columns=["x", "y"])

    data = df.to_xarray()
    data.attrs["gtype"] = "point"
    data.attrs["name"] = "fake point obs"
    data.coords["x"] = x
    data.coords["y"] = y
    data.coords["z"] = np.nan
    data = _set_attrs(data)

    raw_data = {"m1": data[["m1"]], "m2": data[["m2"]]}

    data = data.dropna(dim="time")

    return Comparer(matched_data=data, raw_mod_data=raw_data)


@pytest.fixture
def tc() -> Comparer:
    """A comparer with fake track data"""
    df = _get_track_df()

    data = df.to_xarray()
    data = data.set_coords(["x", "y"])
    data.attrs["gtype"] = "track"
    data.attrs["name"] = "fake track obs"
    data = _set_attrs(data)

    raw_data = {"m1": data[["m1"]], "m2": data[["m2"]]}

    data = data.dropna(dim="time")
    return Comparer(matched_data=data, raw_mod_data=raw_data)


def test_matched_df(pt_df):
    cmp = Comparer.from_matched_data(data=pt_df)
    assert cmp.gtype == "point"
    assert "m2" in cmp.mod_names
    assert "m1" in cmp.mod_names
    assert len(cmp.mod_names) == 2
    assert cmp.n_points == 6
    assert cmp.name == "Observation"
    assert cmp.score()["m1"] == pytest.approx(0.5916079783099617)
    assert cmp.score()["m2"] == pytest.approx(0.15811388300841905)


def test_matched_skill_geodataframe(pt_df):
    cmp = Comparer.from_matched_data(data=pt_df, x=10.0, y=55.0)
    sk = cmp.skill()
    gdf = sk.to_geodataframe()
    assert gdf.iloc[0].geometry.coords[0][0] == 10.0
    assert gdf.iloc[0].geometry.coords[0][1] == 55.0


def test_df_score():
    df = pd.DataFrame(
        {"obs": [1.0, 2.0], "not_so_good": [0.9, 2.1], "perfect": [1.0, 2.0]}
    )
    cmp = Comparer.from_matched_data(data=df, mod_items=["not_so_good", "perfect"])
    assert cmp.score("mae")["not_so_good"] == pytest.approx(0.1)
    assert cmp.score("mae")["perfect"] == pytest.approx(0.0)


def test_matched_df_int_items(pt_df):
    cmp = Comparer.from_matched_data(data=pt_df, mod_items=[1, 2])
    assert cmp.mod_names == ["m1", "m2"]
    assert cmp.n_points == 6

    cmp = Comparer.from_matched_data(data=pt_df, mod_items=[-1])
    assert cmp.mod_names == ["m2"]

    # int mod_items (not list)
    cmp = Comparer.from_matched_data(data=pt_df, mod_items=2)
    assert cmp.mod_names == ["m2"]

    # will fall because two items will have the same name "Observation"
    # cmp = Comparer.from_matched_data(data=pt_df, obs_item=1)

    pt_df = pt_df.rename(columns={"Observation": "obs"})
    cmp = Comparer.from_matched_data(data=pt_df, obs_item=1)
    assert cmp.name == "m1"
    assert "obs" in cmp.mod_names
    assert "m2" in cmp.mod_names


def test_matched_df_with_aux(pt_df):
    pt_df["wind"] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    pt_df["not_relevant"] = [0.0, 0.0, 0.0, 0.0, -1.0, 0.0]

    # by default wind is not considered and aux variable
    cmp = Comparer.from_matched_data(data=pt_df)
    assert set(cmp.mod_names) == set(["m1", "m2", "wind", "not_relevant"])
    assert cmp.n_points == 6
    assert cmp.name == "Observation"
    assert cmp.data["wind"].attrs["kind"] == "model"

    # but it can be specified
    cmp = Comparer.from_matched_data(
        data=pt_df, mod_items=["m1", "m2"], aux_items=["wind"]
    )
    assert cmp.mod_names == ["m1", "m2"]
    assert cmp.n_points == 6
    assert cmp.data["wind"].attrs["kind"] == "auxiliary"
    assert "not_relevant" not in cmp.data.data_vars

    # if aux_items is a string, it is automatically converted to a list
    cmp = Comparer.from_matched_data(
        data=pt_df, mod_items=["m1", "m2"], aux_items="wind"
    )
    assert cmp.data["wind"].attrs["kind"] == "auxiliary"

    # if models are specified, it is NOT automatically considered an aux variable
    cmp = Comparer.from_matched_data(data=pt_df, mod_items=["m1", "m2"])
    assert cmp.mod_names == ["m1", "m2"]
    assert cmp.n_points == 6
    assert "wind" not in cmp.data.data_vars


def test_aux_can_str_(pt_df):
    pt_df["area"] = ["a", "b", "c", "d", "e", "f"]

    cmp = Comparer.from_matched_data(pt_df, aux_items="area")
    assert cmp.data["area"].attrs["kind"] == "auxiliary"


def test_mod_and_obs_must_be_numeric():
    df = pd.DataFrame({"obs": ["a", "b"], "m1": [1, 2]})

    with pytest.raises(ValueError, match="numeric"):
        Comparer.from_matched_data(df)

    df2 = pd.DataFrame({"obs": [1, 2], "m1": ["c", "d"]})

    with pytest.raises(ValueError, match="numeric"):
        Comparer.from_matched_data(df2)


def test_rename_model(pt_df):
    cmp = Comparer.from_matched_data(data=pt_df, mod_items=["m1", "m2"])
    assert "m1" in cmp.mod_names
    assert "m2" in cmp.mod_names
    cmp2 = cmp.rename({"m1": "model_1", "m2": "model_2"})
    assert "model_1" in cmp2.mod_names
    assert "model_2" in cmp2.mod_names
    assert "m1" not in cmp2.mod_names
    assert "m2" not in cmp2.mod_names
    assert "model_1" in cmp2.raw_mod_data
    assert "model_2" in cmp2.raw_mod_data


def test_rename_model_conflict(pt_df):
    cmp = Comparer.from_matched_data(data=pt_df, mod_items=["m1", "m2"])
    assert cmp.mod_names == ["m1", "m2"]
    with pytest.raises(ValueError, match="conflicts"):
        cmp.rename({"m1": "m2"})


def test_rename_model_swap_names(pt_df):
    cmp = Comparer.from_matched_data(data=pt_df, mod_items=["m1", "m2"])
    assert cmp.mod_names == ["m1", "m2"]
    assert cmp.raw_mod_data["m1"].data["m1"].values[0] == 1.5
    assert cmp.raw_mod_data["m2"].data["m2"].values[0] == 1.1
    cmp2 = cmp.rename({"m1": "m2", "m2": "m1"})
    assert cmp2.mod_names == ["m2", "m1"]
    assert cmp2.raw_mod_data["m1"].data["m1"].values[0] == 1.1
    assert cmp2.raw_mod_data["m2"].data["m2"].values[0] == 1.5


def test_partial_rename_model(pt_df):
    cmp = Comparer.from_matched_data(data=pt_df, mod_items=["m1", "m2"])
    assert "m1" in cmp.mod_names
    assert "m2" in cmp.mod_names
    cmp2 = cmp.rename({"m1": "model_1"})
    assert "model_1" in cmp2.mod_names
    assert "m2" in cmp2.mod_names
    assert "m1" not in cmp2.mod_names


def test_rename_aux(pt_df):
    pt_df["wind"] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    cmp = Comparer.from_matched_data(
        data=pt_df, mod_items=["m1", "m2"], aux_items=["wind"]
    )
    assert cmp.aux_names == ["wind"]
    cmp2 = cmp.rename({"wind": "wind_speed"})
    assert cmp.aux_names == ["wind"]
    assert cmp2.aux_names == ["wind_speed"]


def test_rename_model_and_aux(pt_df):
    pt_df["wind"] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    cmp = Comparer.from_matched_data(
        data=pt_df, mod_items=["m1", "m2"], aux_items=["wind"]
    )
    cmp2 = cmp.rename({"m1": "model_1", "wind": "wind_speed"})
    assert "model_1" in cmp2.mod_names
    assert "wind_speed" in cmp2.aux_names


def test_rename_obs(pt_df):
    cmp = Comparer.from_matched_data(data=pt_df)
    assert cmp.name == "Observation"
    cmp2 = cmp.rename({"Observation": "observed"})
    assert cmp2.name == "observed"

    cmp2.name = "observed2"
    assert cmp2.name == "observed2"


def test_rename_fails_unknown_key(pt_df):
    pt_df["wind"] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    cmp = Comparer.from_matched_data(
        data=pt_df, mod_items=["m1", "m2"], aux_items=["wind"]
    )
    with pytest.raises(KeyError, match="Unknown key"):
        cmp.rename({"m1": "model_1", "wind": "wind_speed", "foo": "bar"})
    with pytest.raises(KeyError, match="Unknown key"):
        cmp.rename({"foo": "bar"})
    with pytest.raises(KeyError, match="Unknown key"):
        cmp.rename({"m1": "model_1", "foo": "bar"})
    with pytest.raises(KeyError, match="Unknown key"):
        cmp.rename({"foo": "bar", "wind": "wind_speed"})


def test_rename_errors_ignore(pt_df):
    cmp = Comparer.from_matched_data(data=pt_df)
    cmp2 = cmp.rename({"NotThere": "observed"}, errors="ignore")
    assert cmp2.name == "Observation"


def test_rename_fails_reserved_names(pt_df):
    pt_df["wind"] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    cmp = Comparer.from_matched_data(
        data=pt_df, mod_items=["m1", "m2"], aux_items=["wind"]
    )
    with pytest.raises(ValueError, match="reserved names!"):
        cmp.rename({"m1": "x"})
    with pytest.raises(ValueError, match="reserved names!"):
        cmp.rename({"m1": "y"})
    with pytest.raises(ValueError, match="reserved names!"):
        cmp.rename({"m1": "z"})
    with pytest.raises(ValueError, match="reserved names!"):
        cmp.rename({"m1": "Observation"})


def test_matched_df_illegal_items(pt_df):
    with pytest.raises(ValueError, match="data must contain at least two items"):
        # dataframe has only one column
        df = pt_df[["Observation"]]
        Comparer.from_matched_data(data=df)

    with pytest.raises(IndexError, match="out of range"):
        # non existing item
        Comparer.from_matched_data(data=pt_df, mod_items=[4])

    with pytest.raises(KeyError, match="could not be found"):
        # non existing item
        Comparer.from_matched_data(data=pt_df, mod_items=["m1", "m2", "m3"])

    with pytest.raises(ValueError, match="no model items were found"):
        # no mod_items
        Comparer.from_matched_data(data=pt_df, aux_items=["m1", "m2"])


def test_minimal_matched_data(pt_df):
    data = xr.Dataset(pt_df)
    data["Observation"].attrs["kind"] = "observation"
    data["m1"].attrs["kind"] = "model"
    data["m2"].attrs["kind"] = "model"
    data.attrs["name"] = "mini"

    cmp = Comparer.from_matched_data(data=data)  # no additional raw_mod_data

    assert cmp.data["Observation"].attrs["color"] == "black"
    assert len(cmp.raw_mod_data["m1"]) == 6

    assert cmp.mod_names == ["m1", "m2"]
    assert cmp.n_models == 2


def test_from_compared_data_doesnt_accept_missing_values_in_obs():
    df = pd.DataFrame(
        {
            "Observation": [1.0, np.nan],
            "m1": [1.5, 6.4],
            "time": pd.date_range("2019-01-01", periods=2, freq="D"),
        }
    ).set_index("time")

    data = xr.Dataset(df)
    data["Observation"].attrs["kind"] = "observation"
    data["m1"].attrs["kind"] = "model"
    data.attrs["name"] = "mini"

    with pytest.raises(ValueError):
        Comparer.from_matched_data(data=data)


def test_multiple_forecasts_matched_data():
    # an example on how a forecast dataset could be constructed
    df = pd.DataFrame(
        {
            "Observation": [1.0, 2.0, 1.1, 2.1, 3.1],
            "m1": [1.1, 2.5, 1.2, 4.9, 3.8],
            "time": pd.DatetimeIndex(
                ["2019-01-01", "2019-01-02", "2019-01-02", "2019-01-03", "2019-01-04"]
            ),
            "leadtime": [0, 24, 0, 24, 48],
        }
    ).set_index("time")
    data = xr.Dataset(df)
    data["Observation"].attrs["kind"] = "observation"
    data["m1"].attrs["kind"] = "model"
    data.attrs["name"] = "a fcst"
    cmp = Comparer.from_matched_data(data=data)  # no additional raw_mod_data
    assert len(cmp.raw_mod_data["m1"]) == 5
    assert cmp.mod_names == ["m1"]
    assert cmp.data["leadtime"].attrs["kind"] == "auxiliary"
    analysis = cmp.where(cmp.data["leadtime"] == 0)
    analysis.score()
    assert len(analysis.raw_mod_data["m1"]) == 5
    assert len(analysis.data["m1"]) == 2

    f_s = cmp.score("rmse")
    a_s = analysis.score("rmse")

    assert a_s["m1"] == pytest.approx(0.09999999999999998)
    assert f_s["m1"] == pytest.approx(1.3114877048604001)


def test_matched_aux_variables(pt_df):
    pt_df["wind"] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    data = xr.Dataset(pt_df)
    data["Observation"].attrs["kind"] = "observation"
    data["m1"].attrs["kind"] = "model"
    data["m2"].attrs["kind"] = "model"
    cmp = Comparer.from_matched_data(data=data)
    assert "wind" not in cmp.mod_names
    assert cmp.data["wind"].attrs["kind"] == "auxiliary"


def test_pc_properties(pc):
    assert pc.n_models == 2
    assert pc.n_points == 5
    assert pc.gtype == "point"
    assert pc.x == 10.0
    assert pc.y == 55.0
    assert pc.name == "fake point obs"
    assert pc.quantity.name == "fake var"
    assert pc.time[0] == pd.Timestamp("2019-01-01")
    assert pc.time[-1] == pd.Timestamp("2019-01-05")
    assert pc.mod_names == ["m1", "m2"]
    # assert pc.obs[-1] == 5.0  # TODO
    # assert pc.mod[-1, 1] == 4.9

    assert list(pc.raw_mod_data["m1"].data.data_vars) == ["m1"]
    assert np.all(pc.raw_mod_data["m1"].values == [1.5, 2.4, 3.6, 4.9, 5.6, 6.4])


def test_tc_properties(tc):
    assert tc.n_models == 2
    assert tc.n_points == 5
    assert tc.gtype == "track"
    assert np.all(tc.x == [10.1, 10.2, 10.3, 10.4, 10.5])
    assert np.all(tc.y == [55.1, 55.2, 55.3, 55.4, 55.5])
    assert tc.name == "fake track obs"
    assert tc.quantity.name == "fake var"
    assert tc.time[0] == pd.Timestamp("2019-01-01")
    assert tc.time[-1] == pd.Timestamp("2019-01-05")
    assert tc.mod_names == ["m1", "m2"]
    # assert tc.obs[-1] == 5.0   # TODO
    # assert tc.mod[-1, 1] == 4.9

    assert list(tc.raw_mod_data["m1"].data.data_vars) == ["m1"]
    assert np.all(tc.raw_mod_data["m1"].values == [1.5, 2.4, 3.6, 4.9, 5.6, 6.4])
    assert np.all(tc.raw_mod_data["m1"].x == [10.1, 10.2, 10.3, 10.4, 10.5, 10.6])


def test_attrs(pc):
    pc.attrs["a2"] = "v2"
    assert pc.attrs["a2"] == "v2"

    pc.data.attrs["version"] = 42
    assert pc.attrs["version"] == 42

    pc.attrs["version"] = 43
    assert pc.attrs["version"] == 43

    # remove all attributes and add a new one
    pc.attrs = {"version": 44}
    assert pc.attrs["version"] == 44


def test_pc_sel_time(pc):
    pc2 = pc.sel(time=slice("2019-01-03", "2019-01-04"))
    assert pc2.n_points == 2
    assert pc2.data.Observation.values.tolist() == [3.0, 4.0]


def test_pc_sel_time_empty(pc):
    pc2 = pc.sel(time=slice("2019-01-06", "2019-01-07"))
    assert pc2.n_points == 0


def test_pc_sel_model(pc):
    pc2 = pc.sel(model="m2")
    assert pc2.n_points == 5
    assert pc2.n_models == 1
    assert np.all(pc2.data.m2 == pc.data.m2)
    assert np.all(pc2.raw_mod_data["m2"] == pc.raw_mod_data["m2"])


def test_pc_sel_model_first(pc):
    pc2 = pc.sel(model=0)
    assert pc2.n_points == 5
    assert pc2.n_models == 1
    assert np.all(pc2.data.m1 == pc.data.m1)


def test_pc_sel_model_last(pc):
    pc2 = pc.sel(model=-1)
    assert pc2.n_points == 5
    assert pc2.n_models == 1
    assert np.all(pc2.data.m2 == pc.data.m2)


def test_pc_sel_models_reversed(pc):
    pc2 = pc.sel(model=["m2", "m1"])
    assert pc2.n_points == 5
    assert pc2.n_models == 2
    assert pc2.mod_names == ["m2", "m1"]
    assert np.all(pc2.data.m2 == pc.data.m2)


def test_pc_sel_model_error(pc):
    with pytest.raises(KeyError):
        pc.sel(model="m3")


def test_pc_sel_area(pc):
    bbox = [9.9, 54.9, 10.25, 55.25]
    pc2 = pc.sel(area=bbox)
    assert pc2.n_points == 5
    assert pc2.data.Observation.values.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_tc_sel_model(tc):
    tc2 = tc.sel(model="m2")
    assert tc2.n_points == 5
    assert tc2.n_models == 1
    assert np.all(tc2.data.m2 == tc.data.m2)


def test_tc_sel_area(tc):
    bbox = [9.9, 54.9, 10.25, 55.25]
    tc2 = tc.sel(area=bbox)
    assert tc2.n_points == 2
    assert tc2.data.Observation.values.tolist() == [1.0, 2.0]


def test_tc_sel_area_polygon(tc):
    area = [(9.9, 54.9), (10.25, 54.9), (10.25, 55.25), (9.9, 55.25)]
    tc2 = tc.sel(area=area)
    assert tc2.n_points == 2
    assert tc2.data.Observation.values.tolist() == [1.0, 2.0]


def test_tc_sel_time_and_area(tc):
    bbox = [9.9, 54.9, 10.25, 55.25]
    tc2 = tc.sel(time=slice("2019-01-02", "2019-01-03"), area=bbox)
    assert tc2.n_points == 1
    assert tc2.data.Observation.values.tolist() == [2.0]


def test_pc_drop_model(pc):
    pc2 = pc.drop(model="m2")
    assert isinstance(pc2, type(pc))
    assert pc2.n_models == pc.n_models - 1
    assert "m2" not in pc2.mod_names
    assert "m2" not in pc2.raw_mod_data
    assert "m2" not in pc2.data
    assert np.all(pc.data.m1 == pc2.data.m1)
    assert np.all(pc.raw_mod_data["m1"] == pc2.raw_mod_data["m1"])


def test_pc_drop_model_first(pc):
    pc2 = pc.drop(model=0)
    assert isinstance(pc2, type(pc))
    assert pc2.n_models == pc.n_models - 1
    assert "m1" not in pc2.mod_names
    assert "m1" not in pc2.raw_mod_data
    assert "m1" not in pc2.data
    assert np.all(pc.data.m2 == pc2.data.m2)
    assert np.all(pc.raw_mod_data["m2"] == pc2.raw_mod_data["m2"])


def test_pc_drop_model_last(pc):
    pc2 = pc.drop(model=-1)
    assert isinstance(pc2, type(pc))
    assert pc2.n_models == pc.n_models - 1
    assert "m2" not in pc2.mod_names
    assert "m2" not in pc2.raw_mod_data
    assert "m2" not in pc2.data
    assert np.all(pc.data.m1 == pc2.data.m1)
    assert np.all(pc.raw_mod_data["m1"] == pc2.raw_mod_data["m1"])


def test_tc_drop_model(tc):
    tc2 = tc.drop(model="m2")
    assert isinstance(tc2, type(tc))
    assert tc2.n_models == tc.n_models - 1
    assert "m2" not in tc2.mod_names
    assert "m2" not in tc2.raw_mod_data
    assert "m2" not in tc2.data
    assert np.all(tc.data.m1 == tc2.data.m1)


def test_pc_where(pc):
    pc2 = pc.where(pc.data.Observation > 2.5)
    assert pc2.n_points == 3
    assert pc2.data.Observation.values.tolist() == [3.0, 4.0, 5.0]


def test_pc_where_empty(pc):
    pc2 = pc.where(pc.data.Observation > 10.0)
    assert pc2.n_points == 0


def test_pc_where_derived(pc):
    pc.data["derived"] = pc.data.m1 + pc.data.m2
    pc2 = pc.where(pc.data.derived > 5.0)
    assert pc2.n_points == 3
    assert pc2.data.Observation.values.tolist() == [3.0, 4.0, 5.0]


def test_tc_where_derived(tc):
    x, y = 10.0, 55.0
    dist = np.sqrt((tc.data.x - x) ** 2 + (tc.data.y - y) ** 2)
    # dist = sqrt(2)*[0.1, 0.2, 0.3, 0.4, 0.5]
    tc2 = tc.where(dist > 0.4)
    assert tc2.n_points == 3
    assert tc2.data.Observation.values.tolist() == [3.0, 4.0, 5.0]


def test_tc_where_array(tc):
    cond = np.array([True, False, True, False, True])
    tc2 = tc.where(cond)
    assert tc2.n_points == 3
    assert tc2.data.Observation.values.tolist() == [1.0, 3.0, 5.0]


def test_pc_query(pc):
    pc2 = pc.query("Observation > 2.5")
    assert pc2.n_points == 3
    assert pc2.data.Observation.values.tolist() == [3.0, 4.0, 5.0]


def test_pc_query2(pc):
    pc2 = pc.query("Observation < m2")
    assert pc2.n_points == 4
    assert pc2.data.Observation.values.tolist() == [1.0, 2.0, 3.0, 4.0]


def test_pc_query_empty(pc):
    pc2 = pc.query("Observation > 10.0")
    assert pc2.n_points == 0


def test_add_pc_tc(pc, tc):
    cc = pc + tc
    assert cc.n_points == 10
    assert len(cc) == 2


def test_add_tc_pc(pc, tc):
    cc = tc + pc
    assert cc.n_points == 10
    assert len(cc) == 2


def test_pc_to_long_dataframe(pc):
    # private method testing
    df = pc._to_long_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (10, 7)
    assert "time" in df.columns
    assert "mod_val" in df.columns
    assert "obs_val" in df.columns
    assert "x" in df.columns
    assert "y" in df.columns
    assert "model" in df.columns
    assert "observation" in df.columns
    assert df.mod_val.dtype == "float64"
    assert df.obs_val.dtype == "float64"
    assert df.x.dtype == "float64"
    assert df.y.dtype == "float64"
    assert df.model.dtype == "category"
    assert df.observation.dtype == "category"
    assert df.iloc[0].x == 10.0
    assert df.iloc[0].y == 55.0
    assert df.iloc[0].model == "m1"
    assert df.iloc[9].model == "m2"


def test_pc_to_long_dataframe_add_col(pc):
    # private method testing
    pc.data["derived"] = pc.data.m1 + pc.data.m2
    df = pc._to_long_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (10, 8)
    assert "derived" in df.columns
    assert df.derived.dtype == "float64"


def test_remove_bias():
    df = pd.DataFrame({"obs": [1.0, 2.0], "mod": [1.1, 2.1]})
    cmp = Comparer.from_matched_data(data=df)
    assert cmp.score("bias")["mod"] == pytest.approx(0.1)
    ub_cmp = cmp.remove_bias()
    assert ub_cmp.score("bias")["mod"] == pytest.approx(0.0)


def test_skill_dt(pc):
    by = ["model", "dt:month"]
    sk = pc.skill(by=by)
    assert list(sk.data.index.names) == ["model", "month"]
    assert list(sk.data.index.levels[0]) == ["m1", "m2"]
    assert list(sk.data.index.levels[1]) == [1]  # only January

    # 2019-01-01 is Tuesday = 1 (Monday = 0)
    by = ["model", "dt:weekday"]
    sk = pc.skill(by=by)
    assert list(sk.data.index.names) == ["model", "weekday"]
    assert list(sk.data.index.levels[0]) == ["m1", "m2"]
    assert list(sk.data.index.levels[1]) == [1, 2, 3, 4, 5]  # Tuesday to Saturday


@pytest.mark.skipif(pd.__version__ < "2.0.0", reason="requires newer pandas")
def test_skill_freq(pc):
    assert pc.time.freq == "D"

    # aggregate to 2 days
    sk = pc.skill(by="freq:2D")
    assert len(sk.to_dataframe()) == 3

    # aggregate to 12 hours (up-sampling) doesn't interpolate
    sk2 = pc.skill(by="freq:12h")
    assert len(sk2.to_dataframe()) == 9
    assert np.isnan(sk2.to_dataframe().loc["2019-01-02 12:00:00", "rmse"])


def test_xy_in_skill_pt(pc):
    # point obs has x,y, track obs x, y are np.nan
    sk = pc.skill()
    assert "x" in sk.data.columns
    assert "y" in sk.data.columns
    df = sk.data
    assert all(df.x == pc.x)
    assert all(df.y == pc.y)

    # x, y maintained during sort_values, sort_index, sel
    sk2 = sk.sort_values("rmse")
    assert all(sk2.data.x == pc.x)
    assert all(sk2.data.y == pc.y)

    sk3 = sk.sort_index()
    assert all(sk3.data.x == pc.x)
    assert all(sk3.data.y == pc.y)

    sk4 = sk.sel(model="m1")
    assert all(sk4.data.x == pc.x)
    assert all(sk4.data.y == pc.y)

    sa = sk.rmse  # SkillArray
    assert all(sa.data.x == pc.x)
    assert all(sa.data.y == pc.y)


def test_xy_not_in_skill_tc(tc):
    # point obs has x,y, track obs x, y are np.nan
    sk = tc.skill()
    assert "x" in sk.data.columns
    assert "y" in sk.data.columns
    df = sk.data
    assert df.x.isna().all()
    assert df.y.isna().all()


def test_to_dataframe_pt(pc):
    df = pc.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (5, 3)
    assert list(df.columns) == ["Observation", "m1", "m2"]


def test_to_dataframe_tc(tc):
    df = tc.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (5, 5)
    assert list(df.columns) == ["x", "y", "Observation", "m1", "m2"]


# ======================== plotting ========================

PLOT_FUNCS_RETURNING_MANY_AX = ["scatter", "hist", "residual_hist"]


@pytest.fixture(
    params=[
        "scatter",
        "kde",
        "qq",
        "box",
        "hist",
        "timeseries",
        "taylor",
        "residual_hist",
    ]
)
def pc_plot_function(pc, request):
    func = getattr(pc.plot, request.param)
    # special cases requiring a model to be selected
    if request.param in PLOT_FUNCS_RETURNING_MANY_AX:
        func = getattr(pc.sel(model=0).plot, request.param)
    return func


@pytest.mark.parametrize("kind", PLOT_FUNCS_RETURNING_MANY_AX)
def test_plots_returning_multiple_axes(pc, kind):
    n_models = 2
    func = getattr(pc.plot, kind)
    ax = func()
    assert len(ax) == n_models
    assert all(isinstance(a, plt.Axes) for a in ax)


def test_plot_returns_an_object(pc_plot_function):
    obj = pc_plot_function()
    assert obj is not None


def test_plot_accepts_ax_if_relevant(pc_plot_function):
    _, ax = plt.subplots()
    func_name = pc_plot_function.__name__
    # plots that don't accept ax
    if func_name in ["taylor"]:
        return
    ret_ax = pc_plot_function(ax=ax)
    assert ret_ax is ax


def test_plot_accepts_title(pc_plot_function):
    expected_title = "test title"
    ret_obj = pc_plot_function(title=expected_title)

    # Handle both ax and fig titles
    title = None
    if hasattr(ret_obj, "get_title"):
        title = ret_obj.get_title()
    elif hasattr(ret_obj, "get_suptitle"):
        title = ret_obj.get_suptitle()
    elif hasattr(ret_obj, "_suptitle"):  # older versions of matplotlib
        title = ret_obj._suptitle.get_text()
    else:
        raise pytest.fail("Could not access title from return object.")

    assert title == expected_title


def test_plot_accepts_figsize(pc_plot_function):
    figsize = (10, 10)
    ax = pc_plot_function(figsize=figsize)
    a, b = ax.get_figure().get_size_inches()
    assert a, b == figsize


def test_plot_title_in_returned_ax(pc):
    # default plot is scatter
    ax = pc.sel(model="m1").plot()
    assert "m1" in ax.get_title()

    ax = pc.plot.scatter()
    assert "m1" in ax[0].get_title()
    assert "m2" in ax[1].get_title()

    ax = pc.plot.kde()
    assert "fake point obs" in ax.get_title()


def test_plots_directional(pt_df):
    data = xr.Dataset(pt_df)

    data["Observation"].attrs["kind"] = "observation"
    data["Observation"].attrs["long_name"] = "Waterlevel"
    data["Observation"].attrs["units"] = "m"
    data["m1"].attrs["kind"] = "model"
    data["m2"].attrs["kind"] = "model"
    data.attrs["name"] = "mini"
    cmp = Comparer.from_matched_data(data=data)
    cmp = cmp.sel(model="m1")

    cmp.plot.is_directional = True

    ax = cmp.plot.scatter()
    assert "m1" in ax.get_title()
    assert ax.get_xlim() == (0.0, 360.0)
    assert ax.get_ylim() == (0.0, 360.0)
    assert len(ax.get_legend().get_texts()) == 1  # no reg line or qq

    ax = cmp.plot.kde()
    assert ax is not None
    assert ax.get_xlim() == (0.0, 360.0)

    # TODO I have no idea why this fails in pandas/plotting/_matplotlib/boxplot.py:387: AssertionError
    ax = cmp.plot.box()
    assert ax is not None
    assert ax.get_ylim() == (0.0, 360.0)

    ax = cmp.plot.hist()
    assert ax is not None
    assert ax.get_xlim() == (0.0, 360.0)

    ax = cmp.plot.timeseries()
    assert ax is not None
    assert ax.get_ylim() == (0.0, 360.0)


def test_from_matched_track_data():
    df = pd.DataFrame(
        {
            "lat": [55.0, 55.1],
            "lon": [-0.1, 0.01],
            "c2": [1.2, 1.3],
            "mikeswcal5hm0": [1.22, 1.3],
        },
    )
    assert isinstance(
        df.index, pd.RangeIndex
    )  # Sometime we don't care about time only space

    cmp = ms.from_matched(
        data=df, obs_item="c2", mod_items="mikeswcal5hm0", x_item="lon", y_item="lat"
    )
    gs = cmp.gridded_skill(bins=2)
    gs.data.sel(x=-0.01, y=55.1, method="nearest").n.values == 1

    # positional args
    cmp2 = ms.from_matched(
        data=df,
        x_item=0,
        y_item=1,
        obs_item=2,
        mod_items=3,
    )

    assert len(cmp2.data.coords["x"]) == 2


def test_from_matched_dfs0():
    fn = "tests/testdata/matched_track_data.dfs0"
    # time: 2017-10-27 10:45:19 - 2017-10-29 13:10:44 (532 non-equidistant records)
    # geometry: GeometryUndefined()
    # items:
    #   0:  x <Undefined> (undefined)
    #   1:  y <Undefined> (undefined)
    #   2:  HD <Undefined> (undefined)
    #   3:  Observation <Undefined> (undefined)

    cmp = ms.from_matched(
        data=fn,
        x_item=0,
        y_item=1,
        obs_item=3,
        mod_items=2,
        quantity=ms.Quantity("Water level", "m"),
    )
    gs = cmp.gridded_skill()
    assert float(
        gs.data.sel(x=-0.01, y=55.1, method="nearest").rmse.values
    ) == pytest.approx(0.0476569069177831)


def test_from_matched_x_or_x_item_not_both():
    with pytest.raises(ValueError, match="x and x_item cannot both be specified"):
        ms.from_matched(
            data="tests/testdata/matched_track_data.dfs0",
            x=43.1,
            x_item=0,
            y_item=1,
            obs_item=3,
            mod_items=2,
        )


def test_from_matched_y_or_y_item_not_both():
    with pytest.raises(ValueError, match="y and y_item cannot both be specified"):
        ms.from_matched(
            data="tests/testdata/matched_track_data.dfs0",
            y=2.2,
            x_item=0,
            y_item=1,
            obs_item=3,
            mod_items=2,
        )


def test_from_matched_track_with_no_xy_items():
    # If no x and y items are specified and no other information is available,
    # then we cannot assume that the data is track data!
    df = pd.DataFrame(
        {
            "lat": [55.0, 55.1],
            "lon": [-0.1, 0.01],
            "c2": [1.2, 1.3],
            "mike": [1.22, 1.3],
        },
    )
    cmp = ms.from_matched(
        data=df,
        obs_item="c2",
        mod_items="mike",
    )
    assert cmp.gtype == "point"


def test_from_matched_non_scalar_xy_fails():
    # There is a risk that the user has not understood x_item and y_item
    # should be provided instead of x and y.
    df = pd.DataFrame(
        {
            "lat": [55.0, 55.1],
            "lon": [-0.1, 0.01],
            "c2": [1.2, 1.3],
            "mike": [1.22, 1.3],
        },
    )
    with pytest.raises(TypeError, match="must be scalar"):
        ms.from_matched(
            data=df,
            obs_item="c2",
            mod_items="mike",
            x=df.lon,
            y=df.lat,
        )
