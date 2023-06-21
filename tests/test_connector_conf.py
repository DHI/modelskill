import pytest

import modelskill
from modelskill import ModelResult
from modelskill import PointObservation, TrackObservation
from modelskill import Connector


@pytest.fixture
def mr1():
    fn = "tests/testdata/SW/HKZN_local_2017_DutchCoast.dfsu"
    return ModelResult(fn, item=0, name="SW_1")


@pytest.fixture
def mr2():
    fn = "tests/testdata/SW/HKZN_local_2017_DutchCoast_v2.dfsu"
    return ModelResult(fn, item=0, name="SW_2")


@pytest.fixture
def o1():
    fn = "tests/testdata/SW/HKNA_Hm0.dfs0"
    return PointObservation(fn, item=0, x=4.2420, y=52.6887, name="HKNA")


@pytest.fixture
def o2():
    fn = "tests/testdata/SW/eur_Hm0.dfs0"
    return PointObservation(fn, item=0, x=3.2760, y=51.9990, name="EPL")


@pytest.fixture
def o3():
    fn = "tests/testdata/SW/Alti_c2_Dutch.dfs0"
    return TrackObservation(fn, item=3, name="c2")


@pytest.fixture
def con32(o1, o2, o3, mr1, mr2):
    return Connector([o1, o2, o3], [mr1, mr2])


@pytest.fixture
def conf_xlsx():
    return "tests/testdata/SW/conf_SW.xlsx"


# Exporting a connector to a config file, doesn't work if the data came from in memory objects

# #def test_tofrom_config_dict(con32):
#   d = con32.to_config()
#     assert "modelresults" in d
#     assert len(d["modelresults"]) == 2
#     assert "observations" in d
#     assert len(d["observations"]) == 3

#     con = Connector.from_config(d)
#     assert con.n_models == 2
#     assert con.n_observations == 3
#     assert len(con) == 3


# def test_tofrom_config_yml(tmpdir, con32):
#     filename = os.path.join(tmpdir.dirname, "testconf.yml")
#     con32.to_config(filename, relative_path=False)
#     d = Connector._yaml_to_dict(filename)
#     assert "modelresults" in d
#     assert len(d["modelresults"]) == 2
#     assert "observations" in d
#     assert len(d["observations"]) == 3

#     con = Connector.from_config(filename, relative_path=False)
#     assert con.n_models == 2
#     assert con.n_observations == 3
#     assert len(con) == 3


# def test_tofrom_config_xlsx(tmpdir, con32):
#     filename = os.path.join(tmpdir.dirname, "testconf.xlsx")
#     con32.to_config(filename, relative_path=False)
#     d = Connector._excel_to_dict(filename)
#     assert "modelresults" in d
#     assert len(d["modelresults"]) == 2
#     assert "observations" in d
#     assert len(d["observations"]) == 3

#     con = Connector.from_config(filename, relative_path=False)
#     assert con.n_models == 2
#     assert con.n_observations == 3
#     assert len(con) == 3


def test_from_excel_include(conf_xlsx):
    con = modelskill.from_config(conf_xlsx, relative_path=True)
    assert con.n_models == 1
    assert con.n_observations == 3
    assert len(con) == 3


# def test_from_excel_save_new_relative(conf_xlsx):
#     con = modelskill.from_config(conf_xlsx, relative_path=True)
#     assert con.n_models == 1
#     assert con.n_observations == 3
#     assert len(con) == 3

#     # I know you can use tmpdir, but it is not located in a very interpretable relative path...
#     os.makedirs("tests/testdata/tmp/", exist_ok=True)
#     fn = "tests/testdata/tmp/conf_SW.xlsx"
#     con.to_config(fn)
#     modelskill.from_config(fn)
#     fn = "tests/testdata/tmp/conf_SW.yml"
#     con.to_config(fn)
#     con3 = modelskill.from_config(fn)
#     assert os.path.exists(con3.observations["HKNA"].filename)
