import numpy as np
import pytest
import pandas as pd
import xarray as xr
import modelskill.comparison


def _get_df() -> pd.DataFrame:
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
    data.attrs["variable_name"] = "fake var"
    data["x"].attrs["kind"] = "position"
    data["y"].attrs["kind"] = "position"
    data["Observation"].attrs["kind"] = "observation"
    data["Observation"].attrs["weight"] = 1.0
    data["Observation"].attrs["unit"] = "m"
    data["m1"].attrs["kind"] = "model"
    data["m2"].attrs["kind"] = "model"
    return data


@pytest.fixture
def pc() -> modelskill.comparison.Comparer:
    """A comparer with fake point data"""
    x, y = 10.0, 55.0
    df = _get_df().drop(columns=["x", "y"])
    raw_data = {"m1": df[["m1"]], "m2": df[["m2"]]}

    data = df.dropna().to_xarray()
    data.attrs["gtype"] = "point"
    data.attrs["name"] = "fake point obs"
    data["x"] = x
    data["y"] = y
    data = _set_attrs(data)
    return modelskill.comparison.Comparer(matched_data=data, raw_mod_data=raw_data)


@pytest.fixture
def tc() -> modelskill.comparison.Comparer:
    """A comparer with fake track data"""
    df = _get_df()
    raw_data = {"m1": df[["x", "y", "m1"]], "m2": df[["x", "y", "m2"]]}

    data = df.dropna().to_xarray()
    data.attrs["gtype"] = "track"
    data.attrs["name"] = "fake track obs"
    data = _set_attrs(data)

    return modelskill.comparison.Comparer(matched_data=data, raw_mod_data=raw_data)


def test_minimal_matched_data():

    df = pd.DataFrame(
        {
            "Observation": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "m1": [1.5, 2.4, 3.6, 4.9, 5.6, 6.4],
            "m2": [1.1, 2.2, 3.1, 4.2, 4.9, 6.2],
            "time": pd.date_range("2019-01-01", periods=6, freq="D"),
        }
    ).set_index("time")

    data = xr.Dataset(df)
    data["Observation"].attrs["kind"] = "observation"
    data["m1"].attrs["kind"] = "model"
    data["m2"].attrs["kind"] = "model"
    data.attrs["name"] = "mini"

    cmp = modelskill.comparison.Comparer.from_compared_data(
        data=data
    )  # no additional raw_mod_data

    assert cmp.data["Observation"].attrs["color"] == "black"
    assert cmp.data["Observation"].attrs["unit"] == "Undefined"
    assert cmp.data.attrs["variable_name"] == "Undefined"
    assert len(cmp.raw_mod_data["m1"]) == 6

    assert cmp.mod_names == ["m1", "m2"]
    assert cmp.n_models == 2
    assert cmp.quantity.name == "Undefined"
    assert cmp.quantity.unit == "Undefined"


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
        modelskill.comparison.Comparer.from_compared_data(data=data)


def test_minimal_plots():

    df = pd.DataFrame(
        {
            "Observation": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "m1": [1.5, 2.4, 3.6, 4.9, 5.6, 6.4],
            "m2": [1.1, 2.2, 3.1, 4.2, 4.9, 6.2],
            "time": pd.date_range("2019-01-01", periods=6, freq="D"),
        }
    ).set_index("time")

    data = xr.Dataset(df)
    data.attrs["variable_name"] = "Waterlevel"
    data["Observation"].attrs["kind"] = "observation"
    data["Observation"].attrs["color"] = "pink"
    data["Observation"].attrs["unit"] = "m"
    data["m1"].attrs["kind"] = "model"
    data["m2"].attrs["kind"] = "model"
    data.attrs["name"] = "mini"
    cmp = modelskill.comparison.Comparer.from_compared_data(data=data)

    # Not very elaborate testing other than these two methods can be called without errors
    with pytest.warns(FutureWarning, match="plot.hist"):
        cmp.hist()

    with pytest.warns(FutureWarning, match="plot.kde"):
        cmp.kde()

    with pytest.warns(FutureWarning, match="plot.timeseries"):
        cmp.plot_timeseries()

    with pytest.warns(FutureWarning, match="plot.scatter"):
        cmp.scatter()

    with pytest.warns(FutureWarning, match="plot.taylor"):
        cmp.taylor()

    cmp.plot.taylor()
    # TODO should taylor also return matplotlib axes?

    # default plot is scatter
    ax = cmp.plot()
    assert "m1" in ax.get_title()

    ax = cmp.plot.scatter()
    assert "m1" in ax.get_title()

    ax = cmp.plot.kde()
    assert ax is not None

    ax = cmp.plot.hist()
    assert ax is not None

    ax = cmp.plot.timeseries()
    assert ax is not None

    ax = cmp.plot.scatter()
    assert "m1" in ax.get_title()

    ax = cmp.plot.scatter(model="m2")
    assert "m2" in ax.get_title()


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
    cmp = modelskill.comparison.Comparer.from_compared_data(
        data=data
    )  # no additional raw_mod_data
    assert len(cmp.raw_mod_data["m1"]) == 5
    assert cmp.mod_names == ["m1"]
    assert cmp.data["leadtime"].attrs["kind"] == "auxiliary"
    analysis = cmp.where(cmp.data["leadtime"] == 0)
    analysis.score()
    assert len(analysis.raw_mod_data["m1"]) == 5
    assert len(analysis.data["m1"]) == 2

    f_s = cmp.score("rmse")
    a_s = analysis.score("rmse")

    assert a_s < f_s


def test_matched_aux_variables():

    df = pd.DataFrame(
        {
            "Observation": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "m1": [1.5, 2.4, 3.6, 4.9, 5.6, 6.4],
            "m2": [1.1, 2.2, 3.1, 4.2, 4.9, 6.2],
            "wind": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "time": pd.date_range("2019-01-01", periods=6, freq="D"),
        }
    ).set_index("time")

    data = xr.Dataset(df)
    data["Observation"].attrs["kind"] = "observation"
    data["m1"].attrs["kind"] = "model"
    data["m2"].attrs["kind"] = "model"
    cmp = modelskill.comparison.Comparer.from_compared_data(data=data)
    assert "wind" not in cmp.mod_names
    assert cmp.data["wind"].attrs["kind"] == "auxiliary"


def test_pc_properties(pc):
    assert pc.n_models == 2
    assert pc.n_points == 5
    assert pc.gtype == "point"
    assert pc.x == 10.0
    assert pc.y == 55.0
    assert pc.name == "fake point obs"
    assert pc.variable_name == "fake var"
    assert pc.start == pd.Timestamp("2019-01-01")
    assert pc.end == pd.Timestamp("2019-01-05")
    assert pc.mod_names == ["m1", "m2"]
    assert pc.obs[-1] == 5.0
    assert pc.mod[-1, 1] == 4.9

    assert pc.raw_mod_data["m1"].columns.tolist() == ["m1"]
    assert np.all(pc.raw_mod_data["m1"]["m1"] == [1.5, 2.4, 3.6, 4.9, 5.6, 6.4])


def test_tc_properties(tc):
    assert tc.n_models == 2
    assert tc.n_points == 5
    assert tc.gtype == "track"
    assert np.all(tc.x == [10.1, 10.2, 10.3, 10.4, 10.5])
    assert np.all(tc.y == [55.1, 55.2, 55.3, 55.4, 55.5])
    assert tc.name == "fake track obs"
    assert tc.variable_name == "fake var"
    assert tc.start == pd.Timestamp("2019-01-01")
    assert tc.end == pd.Timestamp("2019-01-05")
    assert tc.mod_names == ["m1", "m2"]
    assert tc.obs[-1] == 5.0
    assert tc.mod[-1, 1] == 4.9

    assert tc.raw_mod_data["m1"].columns.tolist() == ["x", "y", "m1"]
    assert np.all(tc.raw_mod_data["m1"]["m1"] == [1.5, 2.4, 3.6, 4.9, 5.6, 6.4])
    assert np.all(tc.raw_mod_data["m1"]["x"] == [10.1, 10.2, 10.3, 10.4, 10.5, 10.6])


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
    assert cc.n_comparers == 2


def test_add_tc_pc(pc, tc):
    cc = tc + pc
    assert cc.n_points == 10
    assert cc.n_comparers == 2

    
def test_pc_to_dataframe(pc):
    df = pc.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (10, 6)
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
    assert df.x[0] == 10.0
    assert df.y[0] == 55.0
    assert df.model[0] == "m1"
    assert df.model[9] == "m2"


def test_pc_to_dataframe_add_col(pc):
    pc.data["derived"] = pc.data.m1 + pc.data.m2
    df = pc.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (10, 7)
    assert "derived" in df.columns
    assert df.derived.dtype == "float64"
