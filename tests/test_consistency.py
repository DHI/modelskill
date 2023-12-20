"""Test consitency across input formats and classes"""
from functools import partial
import pytest
import mikeio
import modelskill as ms


@pytest.fixture
def fn1():
    """File with 1 item"""
    return "tests/testdata/SW/eur_Hm0.dfs0"


@pytest.fixture
def fn5():
    """Track file with 5 items"""
    return "tests/testdata/SW/Alti_c2_Dutch_short.dfs0"


def _all_equal(iterable):
    """Check if all elements in iterable are equal"""
    iterator = iter(iterable)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == rest for rest in iterator)


def _obs_mod_equal(obs, mod):
    """Check if observation and modelresult are (almost) equal
    Observation have weight and color (and z)"""

    # overwrite the known differences
    mod.data[mod.name].attrs["color"] = obs._color
    mod.data[mod.name].attrs["kind"] = "observation"
    assert obs == mod


# =========== Point ================


@pytest.mark.parametrize(
    "func",
    [
        partial(ms.PointModelResult, name="alti"),
        partial(ms.PointModelResult, name="test2", x=0.0, y=1),
        partial(
            ms.PointModelResult,
            name="alti",
            quantity=ms.Quantity("Wind speed", "m/s"),
        ),
    ],
)
def test_consistency_pointmodelresult_single_item(fn1, func):
    """Only a single item in file - not necessary to specify item"""
    mr1 = func(fn1)  # dfs0 file
    mr2 = func(mikeio.open(fn1))  # mikeio.Dfs0 class
    mr3 = func(mikeio.read(fn1))  # mikeio.Dataset class
    mr4 = func(mikeio.read(fn1)[0])  # mikeio.DataArray class
    mr5 = func(mikeio.read(fn1).to_dataframe())  # pandas.DataFrame class
    mr5.quantity = mr1.quantity
    mr6 = func(mikeio.read(fn1).to_dataframe()["Hm0"])  # pandas.Series class
    mr6.quantity = mr1.quantity
    mr7 = func(mikeio.read(fn1).to_xarray())  # xarray.Dataset class
    mr7.quantity = mr1.quantity
    mr8 = func(mikeio.read(fn1).to_xarray()["Hm0"])  # xarray.DataArray class
    mr8.quantity = mr1.quantity
    assert _all_equal([mr1, mr2, mr3, mr4, mr5, mr6, mr7, mr8])


@pytest.mark.parametrize(
    "func",
    [
        partial(ms.PointModelResult, item="swh", name="alti"),
        partial(ms.PointModelResult, item="ws", name="alti", x=0.0, y=1),
        partial(
            ms.PointModelResult,
            item="adt",
            name="alti",
            quantity=ms.Quantity("Wind speed", "m/s"),
        ),
    ],
)
def test_consistency_pointmodelresult_multi_item(fn5, func):
    mr1 = func(fn5)  # dfs0 file
    mr2 = func(mikeio.open(fn5))  # mikeio.Dfs0 class
    mr3 = func(mikeio.read(fn5))  # mikeio.Dataset class
    mr4 = func(mikeio.read(fn5).to_dataframe())  # pandas.DataFrame class
    mr4.quantity = mr1.quantity
    mr5 = func(mikeio.read(fn5).to_xarray())  # xarray.Dataset class
    mr5.quantity = mr1.quantity
    assert _all_equal([mr1, mr2, mr3, mr4, mr5])


@pytest.mark.parametrize(
    "data_provider",
    [
        lambda x: x,
        lambda x: mikeio.open(x),
        lambda x: mikeio.read(x),
        lambda x: mikeio.read(x).to_dataframe(),
        lambda x: mikeio.read(x).to_xarray(),
    ],
    ids=[
        "dfs0 filename",
        "mikeio.Dfs0",
        "mikeio.Dataset",
        "pd.DataFrame",
        "xr.Dataset",
    ],
)
def test_consistency_pointmodelresult_fails(fn5, data_provider):
    """Test that it fails when item is not specified"""
    with pytest.raises(ValueError, match="but item was not given"):
        ms.PointModelResult(data_provider(fn5))


@pytest.mark.parametrize(
    "func",
    [
        partial(ms.PointObservation, name="alti"),
        partial(ms.PointObservation, name="test2", x=0.0, y=1),
        partial(
            ms.PointObservation,
            name="alti",
            quantity=ms.Quantity("Wind speed", "m/s"),
        ),
    ],
)
def test_consistency_pointobservation_single_item(fn1, func):
    """Only a single item in file - not necessary to specify item"""
    mr1 = func(fn1)  # dfs0 file
    mr2 = func(mikeio.open(fn1))  # mikeio.Dfs0 class
    mr3 = func(mikeio.read(fn1))  # mikeio.Dataset class
    mr4 = func(mikeio.read(fn1)[0])  # mikeio.DataArray class
    mr5 = func(mikeio.read(fn1).to_dataframe())  # pandas.DataFrame class
    mr5.quantity = mr1.quantity
    mr6 = func(mikeio.read(fn1).to_dataframe()["Hm0"])  # pandas.Series class
    mr6.quantity = mr1.quantity
    mr7 = func(mikeio.read(fn1).to_xarray())  # xarray.Dataset class
    mr7.quantity = mr1.quantity
    mr8 = func(mikeio.read(fn1).to_xarray()["Hm0"])  # xarray.DataArray class
    mr8.quantity = mr1.quantity
    assert _all_equal([mr1, mr2, mr3, mr4, mr5, mr6, mr7, mr8])


@pytest.mark.parametrize(
    "func",
    [
        partial(ms.PointObservation, item="swh", name="alti"),
        partial(ms.PointObservation, item="ws", name="alti", x=0.0, y=1),
        partial(
            ms.PointObservation,
            item="adt",
            name="alti",
            quantity=ms.Quantity("Wind speed", "m/s"),
        ),
    ],
)
def test_consistency_pointobservation_multi_item(fn5, func):
    mr1 = func(fn5)  # dfs0 file
    mr2 = func(mikeio.open(fn5))  # mikeio.Dfs0 class
    mr3 = func(mikeio.read(fn5))  # mikeio.Dataset class
    mr4 = func(mikeio.read(fn5).to_dataframe())  # pandas.DataFrame class
    mr4.quantity = mr1.quantity
    mr5 = func(mikeio.read(fn5).to_xarray())  # xarray.Dataset class
    mr5.quantity = mr1.quantity
    assert _all_equal([mr1, mr2, mr3, mr4, mr5])


@pytest.mark.parametrize(
    "data_provider",
    [
        lambda x: x,
        lambda x: mikeio.open(x),
        lambda x: mikeio.read(x),
        lambda x: mikeio.read(x).to_dataframe(),
        lambda x: mikeio.read(x).to_xarray(),
    ],
    ids=[
        "dfs0 filename",
        "mikeio.Dfs0",
        "mikeio.Dataset",
        "pd.DataFrame",
        "xr.Dataset",
    ],
)
def test_consistency_pointobservation_fails(fn5, data_provider):
    """Test that it fails when item is not specified"""
    with pytest.raises(ValueError, match="item was not given"):
        ms.PointObservation(data_provider(fn5))


def test_consistency_point_obs_mod(fn5):
    """Test that observation and modelresult are (almost) equal"""
    obs = ms.PointObservation(fn5, item="swh", name="alti")
    mod = ms.PointModelResult(fn5, item="swh", name="alti")
    _obs_mod_equal(obs, mod)

    obs = ms.PointObservation(fn5, item="swh", name="alti", x=0.0, y=1)
    mod = ms.PointModelResult(fn5, item="swh", name="alti", x=0.0, y=1)
    _obs_mod_equal(obs, mod)


# =========== Track ================


@pytest.mark.parametrize(
    "func",
    [
        partial(
            ms.TrackModelResult,
            item="swh",
            name="alti",
            x_item="Longitude",
            y_item="Latitude",
        ),
        partial(ms.TrackModelResult, item="ws", name="alti", x_item=1, y_item=0),
        partial(
            ms.TrackModelResult,
            item="adt",
            name="alti",
            quantity=ms.Quantity("Wind speed", "m/s"),
        ),
    ],
)
def test_consistency_trackmodelresult(fn5, func):
    mr1 = func(fn5)  # dfs0 file
    mr2 = func(mikeio.open(fn5))  # mikeio.Dfs0 class
    mr3 = func(mikeio.read(fn5))  # mikeio.Dataset class
    mr4 = func(mikeio.read(fn5).to_dataframe())  # pandas.DataFrame class
    mr4.quantity = mr1.quantity
    mr5 = func(mikeio.read(fn5).to_xarray())  # xarray.Dataset class
    mr5.quantity = mr1.quantity
    assert _all_equal([mr1, mr2, mr3, mr4, mr5])


@pytest.mark.parametrize(
    "data_provider",
    [
        lambda x: x,
        lambda x: mikeio.open(x),
        lambda x: mikeio.read(x),
        lambda x: mikeio.read(x).to_dataframe(),
        lambda x: mikeio.read(x).to_xarray(),
    ],
    ids=[
        "dfs0 filename",
        "mikeio.Dfs0",
        "mikeio.Dataset",
        "pd.DataFrame",
        "xr.Dataset",
    ],
)
def test_consistency_trackmodelresult_fails(fn5, data_provider):
    """Test that it fails when item is not specified"""
    with pytest.raises(ValueError, match="more than 3 items, but item was not given"):
        ms.TrackModelResult(data_provider(fn5))

    with pytest.raises(ValueError, match="more than 3 items, but item was not given"):
        ms.TrackModelResult(data_provider(fn5), x_item="swh")

    with pytest.raises(ValueError, match="more than 3 items, but item was not given"):
        ms.TrackModelResult(data_provider(fn5), x_item=0, y_item=1)

    # same item given twice
    with pytest.raises(ValueError, match="Duplicate items"):
        ms.TrackModelResult(data_provider(fn5), item="Longitude", x_item=0, y_item=1)


@pytest.mark.parametrize(
    "func",
    [
        partial(
            ms.TrackObservation,
            item="swh",
            name="alti",
            x_item="Longitude",
            y_item="Latitude",
        ),
        partial(ms.TrackObservation, item="ws", name="alti", x_item=1, y_item=0),
        partial(
            ms.TrackObservation,
            item="adt",
            x_item=1,
            y_item=0,
            name="alti",
            quantity=ms.Quantity("Wind speed", "m/s"),
        ),
    ],
)
def test_consistency_trackobservation(fn5, func):
    mr1 = func(fn5)  # dfs0 file
    mr2 = func(mikeio.open(fn5))  # mikeio.Dfs0 class
    mr3 = func(mikeio.read(fn5))  # mikeio.Dataset class
    mr4 = func(mikeio.read(fn5).to_dataframe())  # pandas.DataFrame class
    mr4.quantity = mr1.quantity
    mr5 = func(mikeio.read(fn5).to_xarray())  # xarray.Dataset class
    mr5.quantity = mr1.quantity
    assert _all_equal([mr1, mr2, mr3, mr4, mr5])


@pytest.mark.parametrize(
    "data_provider",
    [
        lambda x: x,
        lambda x: mikeio.open(x),
        lambda x: mikeio.read(x),
        lambda x: mikeio.read(x).to_dataframe(),
        lambda x: mikeio.read(x).to_xarray(),
    ],
    ids=[
        "dfs0 filename",
        "mikeio.Dfs0",
        "mikeio.Dataset",
        "pd.DataFrame",
        "xr.Dataset",
    ],
)
def test_consistency_trackobservation_fails(fn5, data_provider):
    """Test that it fails when item is not specified"""
    with pytest.raises(ValueError, match="more than 3 items, but item was not given"):
        ms.TrackObservation(data_provider(fn5))

    with pytest.raises(ValueError, match="more than 3 items, but item was not given"):
        ms.TrackObservation(data_provider(fn5), x_item="swh")

    # same item given twice
    with pytest.raises(ValueError, match="Duplicate items"):
        ms.TrackObservation(data_provider(fn5), item="Longitude", x_item=0, y_item=1)


def test_consistency_track_obs_mod(fn5):
    """Test that observation and modelresult are (almost) equal"""
    obs = ms.TrackObservation(fn5, item="swh", name="alti", x_item=1, y_item=0)
    mod = ms.TrackModelResult(fn5, item="swh", name="alti", x_item=1, y_item=0)
    _obs_mod_equal(obs, mod)
