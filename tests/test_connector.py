import pytest
import xarray as xr

import modelskill as ms
from modelskill.connection import SingleObsConnector


@pytest.fixture
def mr1():
    fn = "tests/testdata/SW/HKZN_local_2017_DutchCoast.dfsu"
    return ms.ModelResult(fn, item=0, name="SW_1")


@pytest.fixture
def mr2():
    fn = "tests/testdata/SW/HKZN_local_2017_DutchCoast_v2.dfsu"
    return ms.ModelResult(fn, item=0, name="SW_2")


@pytest.fixture
def mr3():
    fn = "tests/testdata/SW/HKZN_local_2017_DutchCoast_v3.dfsu"
    return ms.ModelResult(fn, item=0, name="SW_3")


@pytest.fixture
def o1():
    fn = "tests/testdata/SW/HKNA_Hm0.dfs0"
    return ms.PointObservation(fn, item=0, x=4.2420, y=52.6887, name="HKNA")


@pytest.fixture
def o2():
    fn = "tests/testdata/SW/eur_Hm0.dfs0"
    return ms.PointObservation(fn, item=0, x=3.2760, y=51.9990, name="EPL")


@pytest.fixture
def o3():
    fn = "tests/testdata/SW/Alti_c2_Dutch.dfs0"
    return ms.TrackObservation(fn, item=3, name="c2")


@pytest.fixture
def con31(o1, o2, o3, mr1):
    with pytest.warns(FutureWarning, match="modelskill.compare"):
        return ms.Connector([o1, o2, o3], mr1)


@pytest.fixture
def con32(o1, o2, o3, mr1, mr2):
    with pytest.warns(FutureWarning, match="modelskill.compare"):
        return ms.Connector([o1, o2, o3], [mr1, mr2])


def test_point_connector_repr(o1, mr1):
    with pytest.warns(FutureWarning, match="modelskill.compare"):
        con = SingleObsConnector(o1, mr1)
    txt = repr(con)
    assert "SingleObsConnector" in txt


def test_connector_add(o1, mr1):
    with pytest.warns(FutureWarning, match="modelskill.compare"):
        con = ms.Connector()
        con.add(o1, mr1, validate=False)
    assert len(con.observations) == 1


def test_connector_add_two_models(
    o1: ms.PointObservation, mr1: ms.ModelResult, mr2: ms.ModelResult
):
    with pytest.warns(FutureWarning, match="modelskill.compare"):
        con = ms.Connector(o1, [mr1, mr2])

    assert con.n_models == 2
    cc = con.extract()
    assert cc.n_models == 2

    # Alternative specification using .add() should be identical
    with pytest.warns(FutureWarning, match="modelskill.compare"):
        con2 = ms.Connector()
        con2.add(o1, mr1)
        con2.add(o1, mr2)

    assert con2.n_models == 2
    cc2 = con2.extract()
    assert cc2.n_models == 2


def test_connector_add_two_model_dataframes(
    o1: ms.PointObservation, mr1: ms.ModelResult, mr2: ms.ModelResult
):
    mr1_extr = mr1.extract(o1)
    mr2_extr = mr2.extract(o1)

    assert isinstance(mr1_extr.data, xr.Dataset)
    assert isinstance(mr2_extr.data, xr.Dataset)

    assert len(mr1_extr.data.data_vars) == 1
    assert len(mr2_extr.data.data_vars) == 1

    assert mr1_extr.n_points > 1  # Number of rows
    assert mr2_extr.n_points > 1  # Number of rows

    with pytest.warns(FutureWarning, match="modelskill.compare"):
        con = ms.Connector(o1, [mr1_extr, mr2_extr])

    assert con.n_models == 2
    cc = con.extract()
    assert cc.n_models == 2

    # Alternative specification using .add() should be identical
    with pytest.warns(FutureWarning, match="modelskill.compare"):
        con2 = ms.Connector()
        con2.add(o1, mr1_extr)
        con2.add(o1, mr2_extr)

    assert con2.n_models == 2
    cc2 = con2.extract()
    assert cc2.n_models == 2
