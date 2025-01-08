import os

import pytest
import modelskill as ms


@pytest.fixture
def change_test_directory(tmp_path):
    original_directory = os.getcwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(original_directory)


def test_load_vistula(change_test_directory):
    cc = ms.data.vistula()
    assert isinstance(cc, ms.ComparerCollection)
    assert len(cc) == 8
    assert cc.mod_names == ["sim1", "sim2"]
    assert cc[0].name == "Tczew"
    assert cc[-1].n_points == 1827
    assert cc[-1].y == 52.94889

    assert cc[0].quantity.name == "Discharge"
    assert cc[0].quantity.unit == "m3/s"

    assert cc.aux_names == ["Precipitation"]
    assert float(cc[1].data.Precipitation[0]) == pytest.approx(1.18)
    assert cc[0].attrs["River"] == "Vistula"
    assert cc[0].attrs["Area"] == 193922.9

    assert cc[1].raw_mod_data["sim2"].n_points == 1827
    assert isinstance(cc[0].raw_mod_data["sim1"], ms.PointModelResult)


def test_load_oresund(change_test_directory):
    cc = ms.data.oresund()
    assert isinstance(cc, ms.ComparerCollection)
    assert len(cc) == 7
    assert cc.mod_names == ["MIKE21"]
    assert cc[-1].name == "Vedbaek"
    assert cc[0].n_points == 8422
    assert cc[0].x == pytest.approx(12.7117)

    assert cc[0].quantity.name == "Surface Elevation"
    assert cc[0].quantity.unit == "meter"

    assert cc.aux_names == ["U10", "V10"]
    assert cc[-1].attrs["Country"] == "DK"

    assert cc[1].raw_mod_data["MIKE21"].n_points == 4344
    assert isinstance(cc[0].raw_mod_data["MIKE21"], ms.PointModelResult)
