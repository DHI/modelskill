from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pytest
import modelskill as ms
from modelskill.plotting._misc import format_skill_table
from modelskill.plotting._misc import sample_points


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
def mr1():
    fn = "tests/testdata/SW/HKZN_local_2017_DutchCoast.dfsu"
    return ms.model_result(fn, item=0, name="SW_1")


@pytest.fixture
def mr2():
    fn = "tests/testdata/SW/HKZN_local_2017_DutchCoast_v2.dfsu"
    return ms.model_result(fn, item=0, name="SW_2")


def test_plot_temporal_coverage_11(o1, mr1):
    ms.plotting.temporal_coverage(o1, mr1)
    plt.close()


def test_plot_temporal_coverage_12(o1, mr1, mr2):
    ms.plotting.temporal_coverage(o1, [mr1, mr2])
    plt.close()


def test_plot_temporal_coverage_31(o1, o2, o3, mr1):
    ms.plotting.temporal_coverage([o1, o2, o3], mr1)
    plt.close()


def test_plot_temporal_coverage_settings(o1, o2, o3, mr1, mr2):
    ms.plotting.temporal_coverage([o1, o2, o3], [mr1, mr2], limit_to_model_period=False)
    ms.plotting.temporal_coverage([o1, o2, o3], [mr1, mr2], marker=".")
    ms.plotting.temporal_coverage(
        [o1, o2, o3], [mr1, mr2], title="test", figsize=(3, 4)
    )
    plt.close()


def test_plot_spatial_overview(o1, o2, o3, mr1):
    ms.plotting.spatial_overview([o1, o2, o3], mr1)
    ms.plotting.spatial_overview(o1, mr1, figsize=(3, 6))
    ms.plotting.spatial_overview(obs=[], mod=mr1, title="test")
    plt.close()


def test_format_skill_table():
    #
    #    	            n	bias	rmse	urmse	mae	cc	si	r2
    # observation
    # smhi_2095_klagshamn	167	1.033099e-09	0.040645	0.040645	0.033226	0.841135	0.376413	0.706335

    skill_df = pd.DataFrame(
        {
            "n": [167],
            "bias": [1.033099e-09],
            "rmse": [0.040645],
            "urmse": [0.040645],
            "mae": [0.033226],
            "cc": [0.841135],
            "si": [0.376413],
            "r2": [0.706335],
        },
        index=["smhi_2095_klagshamn"],
    )

    skill_scores = skill_df.iloc[0].to_dict()

    df = format_skill_table(skill_scores, unit="degC")
    assert "N" in df.iloc[0, 0]
    assert "167" == str(df.iloc[0, 2]).strip()
    assert "BIAS" in df.iloc[1, 0]
    assert "0.00 degC" in df.iloc[1, 2]
    assert "RMSE" in df.iloc[2, 0]
    assert "0.04 degC" in df.iloc[2, 2]
    assert "URMSE" in df.iloc[3, 0]
    assert "0.04 degC" in df.iloc[3, 2]
    assert "MAE" in df.iloc[4, 0]
    assert "0.03 degC" in df.iloc[4, 2]
    assert "CC" in df.iloc[5, 0]
    assert "0.84" in df.iloc[5, 2]

    df_short = format_skill_table(skill_scores, unit="meter")

    assert "N" in df_short.iloc[0, 0]
    assert "167" in df_short.iloc[0, 2]
    assert "BIAS" in df_short.iloc[1, 0]
    assert "0.00 m" in df_short.iloc[1, 2]
    assert "RMSE" in df_short.iloc[2, 0]
    assert "0.04 m" in df_short.iloc[2, 2]
    assert "URMSE" in df_short.iloc[3, 0]
    assert "0.04 m" in df_short.iloc[3, 2]
    assert "MAE" in df_short.iloc[4, 0]
    assert "0.03" in df_short.iloc[4, 2]
    assert "CC" in df_short.iloc[5, 0]
    assert "0.84" in df_short.iloc[5, 2]


@pytest.fixture
def x_y():
    np.random.seed(42)
    x = np.random.rand(100000)
    y = np.random.rand(100000)
    return x, y


def test_sample_points_bool_selects_all_points(x_y):
    x, y = x_y

    x_sample, y_sample = sample_points(x, y, show_points=True)
    assert len(x_sample) == len(x)
    assert len(y_sample) == len(y)


def test_sample_points_bool_selects_no_points(x_y):
    x, y = x_y

    x_sample, y_sample = sample_points(x, y, show_points=False)
    assert len(x_sample) == 0
    assert len(y_sample) == 0


def test_sample_points_int_selects_n_points(x_y):
    x, y = x_y

    x_sample, y_sample = sample_points(x, y, show_points=10)
    assert len(x_sample) == 10
    assert len(y_sample) == 10


def test_sample_points_float_selects_fraction_points(x_y):
    x, y = x_y

    x_sample, y_sample = sample_points(x, y, show_points=0.1)
    assert len(x_sample) == 10000
    assert len(y_sample) == 10000


def test_sample_points_float_raises_error(x_y):
    x, y = x_y

    with pytest.raises(ValueError):
        sample_points(x, y, show_points=1.1)

    with pytest.raises(ValueError):
        sample_points(x, y, show_points=-0.1)


def test_sample_points_negative_int_raises_error(x_y):
    x, y = x_y

    with pytest.raises(ValueError):
        sample_points(x, y, show_points=-1)


def test_sample_points_large_int_uses_all_points(x_y):
    x, y = x_y

    x_sample, y_sample = sample_points(x, y, show_points=1000000)
    assert len(x_sample) == len(x)
    assert len(y_sample) == len(y)
