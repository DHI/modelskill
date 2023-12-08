import pytest

import modelskill as ms


@pytest.fixture
def conf_xlsx():
    return "tests/testdata/SW/conf_SW.xlsx"


def test_from_excel_include(conf_xlsx):
    with pytest.warns(FutureWarning, match="modelskill.compare"):
        con = ms.from_config(conf_xlsx, relative_path=True)
    assert con.n_models == 1
    assert con.n_observations == 3
    assert len(con) == 3
