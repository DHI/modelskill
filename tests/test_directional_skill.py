import pandas as pd
import pytest

import modelskill as ms

DIRECTION = ms.Quantity("Wave direction", "degree", is_directional=True)
HEIGHT = ms.Quantity("Wave height", "m")


def _cmp(quantity: ms.Quantity, name: str) -> ms.Comparer:
    # 1° observed vs 359° modelled is a 2° error, not 358°
    df = pd.DataFrame(
        {"obs": [1.0, 2.0, 3.0], "mod": [359.0, 0.0, 1.0]},
        index=pd.date_range("2020", periods=3, freq="h"),
    )
    return ms.from_matched(df, obs_item="obs", quantity=quantity, name=name)


@pytest.fixture
def cmp() -> ms.Comparer:
    return _cmp(DIRECTION, "buoy")


@pytest.fixture
def cc(cmp) -> ms.ComparerCollection:
    return ms.ComparerCollection([cmp])


def test_comparer_skill_default_is_circular(cmp):
    assert cmp.skill().to_dataframe().loc["buoy", "c_bias"] == pytest.approx(-2.0)


def test_comparer_score_default_is_circular(cmp):
    assert cmp.score()["mod"] == pytest.approx(2.0)


def test_collection_skill_default_is_circular(cc):
    assert cc.skill().to_dataframe().loc["buoy", "c_bias"] == pytest.approx(-2.0)


def test_collection_score_default_is_circular(cc):
    assert cc.score()["mod"] == pytest.approx(2.0)


def test_collection_mean_skill_default_is_circular(cc):
    assert cc.mean_skill().to_dataframe().loc["mod", "c_rmse"] == pytest.approx(2.0)


@pytest.mark.parametrize("metric", ["bias", "rmse", "mae", "cc"])
def test_comparer_skill_with_linear_metric_raises(cmp, metric):
    with pytest.raises(ValueError, match="directional"):
        cmp.skill(metrics=[metric])


def test_error_names_the_circular_metric(cmp):
    with pytest.raises(ValueError, match="c_rmse"):
        cmp.skill(metrics=["rmse"])


def test_collection_skill_with_linear_metric_raises(cc):
    with pytest.raises(ValueError, match="directional"):
        cc.skill(metrics=["rmse"])


def test_score_with_linear_metric_raises(cmp, cc):
    with pytest.raises(ValueError, match="directional"):
        cmp.score(metric="rmse")
    with pytest.raises(ValueError, match="directional"):
        cc.score(metric="rmse")


def test_custom_metric_is_passed_through(cmp):
    def my_metric(obs, model):
        return 42.0

    assert cmp.skill(metrics=[my_metric]).to_dataframe().loc[
        "buoy", "my_metric"
    ] == pytest.approx(42.0)


def test_collection_mixing_directional_and_scalar_raises():
    cc = ms.ComparerCollection([_cmp(DIRECTION, "dir"), _cmp(HEIGHT, "hm0")])
    with pytest.raises(ValueError, match="quantity"):
        cc.skill()


def test_collection_mixing_directional_and_scalar_works_per_quantity():
    cc = ms.ComparerCollection([_cmp(DIRECTION, "dir"), _cmp(HEIGHT, "hm0")])
    sk = cc.sel(quantity="Wave direction").skill()
    assert sk.to_dataframe().loc["dir", "c_bias"] == pytest.approx(-2.0)


def test_taylor_diagram_raises_for_directional(cmp, cc):
    with pytest.raises(ValueError, match="Taylor diagram"):
        cmp.plot.taylor()
    with pytest.raises(ValueError, match="Taylor diagram"):
        cc.plot.taylor()
