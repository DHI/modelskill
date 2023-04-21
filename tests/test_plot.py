import pytest
import fmskill as ms


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
    return ms.ModelResult(fn, item=0, name="SW_1")


@pytest.fixture
def mr2():
    fn = "tests/testdata/SW/HKZN_local_2017_DutchCoast_v2.dfsu"
    return ms.ModelResult(fn, item=0, name="SW_2")


def test_plot_temporal_coverage_11(o1, mr1):
    ms.plot_temporal_coverage(o1, mr1)


def test_plot_temporal_coverage_12(o1, mr1, mr2):
    ms.plot_temporal_coverage(o1, [mr1, mr2])

def test_plot_temporal_coverage_31(o1, o2, o3, mr1):
    ms.plot_temporal_coverage([o1, o2, o3], mr1)
