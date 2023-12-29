import pytest

import modelskill as ms


@pytest.fixture
def conf_xlsx():
    return "tests/testdata/SW/conf_SW.xlsx"


@pytest.fixture
def conf_dict():
    configuration = dict(
        modelresults=dict(
            HD=dict(
                filename="tests/testdata/Oresund2D_subset.dfsu",
                item=0,
            ),
        ),
        observations=dict(
            klagshamn=dict(
                filename="tests/testdata/obs_two_items.dfs0",
                item=1,
                x=366844,
                y=6154291,
                name="Klagshamn2",
            ),
            Drogden=dict(
                filename="tests/testdata/dmi_30357_Drogden_Fyr.dfs0",
                item=0,
                x=355568.0,
                y=6156863.0,
            ),
        ),
    )
    return configuration


def test_from_excel_include(conf_xlsx):
    cc = ms.from_config(conf_xlsx, relative_path=True)
    assert cc.n_models == 1
    assert cc.n_observations == 3
    assert len(cc) == 3


def test_comparison_from_dict(conf_dict):
    cc = ms.from_config(conf_dict)
    assert len(cc) == 2
    assert cc.n_comparers == 2
    assert cc.n_models == 1


def test_comparison_from_yml():
    cc = ms.from_config("tests/testdata/conf.yml")

    assert len(cc) == 2
    assert cc.n_comparers == 2
    assert cc.n_models == 1
    assert cc["Klagshamn"].quantity.name == "Water Level"
