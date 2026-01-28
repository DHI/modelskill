import pytest

import modelskill as ms


@pytest.fixture
def mrmike():
    fn = "tests/testdata/SW/HKZN_local_2017_DutchCoast.dfsu"
    return ms.model_result(fn, name="SW_1", item=0)


@pytest.fixture
def mrmike2():
    fn = "tests/testdata/SW/HKZN_local_2017_DutchCoast_v2.dfsu"
    return ms.model_result(fn, name="SW_2", item=0)


@pytest.fixture
def mr2days():
    fn = "tests/testdata/SW/CMEMS_DutchCoast_*.nc"
    return ms.model_result(fn, name="CMEMS", item="VHM0")


@pytest.fixture
def mr28():
    fn = "tests/testdata/SW/CMEMS_DutchCoast_2017-10-28.nc"
    return ms.model_result(fn, name="CMEMS", item="VHM0")


@pytest.fixture
def mr29():
    fn = "tests/testdata/SW/CMEMS_DutchCoast_2017-10-29.nc"
    return ms.model_result(fn, name="CMEMS", item="VHM0")


@pytest.fixture
def o123():
    fn = "tests/testdata/SW/HKNA_Hm0.dfs0"
    o1 = ms.PointObservation(fn, item=0, x=4.2420, y=52.6887, name="HKNA")

    fn = "tests/testdata/SW/eur_Hm0.dfs0"
    o2 = ms.PointObservation(fn, item=0, x=3.2760, y=51.9990, name="EPL")

    fn = "tests/testdata/SW/Alti_c2_Dutch.dfs0"
    o3 = ms.TrackObservation(fn, item=3, name="c2")

    return o1, o2, o3


def test_merge(o123, mrmike, mrmike2):
    cc1 = ms.match(o123, mrmike)
    cc2 = ms.match(o123, mrmike2)

    cc12 = ms.match(o123, [mrmike, mrmike2])

    assert len(cc1.mod_names) == 1
    assert len(cc12.mod_names) == 2
    assert cc1.mod_names[0] == cc12.mod_names[0]
    assert cc2.mod_names[0] == cc12.mod_names[-1]
    assert cc2.end_time == cc12.end_time

    cc12b = cc1.merge(cc2)
    assert all([len(c.mod_names) == 2 for c in cc12b])
    assert cc12b.score() == cc12.score()
    assert cc12b.n_points == cc12.n_points

    # deprecated
    with pytest.warns(FutureWarning, match="deprecated"):
        cc12c = cc1 + cc2
        assert cc12c.score() == cc12.score()
        assert cc12c.n_points == cc12.n_points


def test_merge_models_different_time(o123, mrmike, mr2days):
    cc1 = ms.match(o123, mrmike)
    cc2 = ms.match(o123, mr2days)

    # note the time coverage is different for the two models
    cc12 = ms.match(o123, [mrmike, mr2days])

    assert len(cc1.mod_names) == 1
    assert len(cc12.mod_names) == 2
    assert cc1.mod_names[0] == cc12.mod_names[0]
    assert cc2.mod_names[0] == cc12.mod_names[-1]
    assert cc2.end_time == cc12.end_time

    cc12b = cc1.merge(cc2)
    assert cc12b.score() == cc12.score()
    assert cc12b.n_points == cc12.n_points


def test_merge_same_model(o123, mrmike):
    cc1 = ms.match(o123, mrmike)
    cc2 = ms.match(o123, mrmike)

    # if we add the same model multiple times it has no effect
    cc12 = cc1.merge(cc2)
    assert cc12.n_points == cc1.n_points
    assert cc12[0].data.time.to_index().is_unique
    assert cc1.score() == cc12.score()
