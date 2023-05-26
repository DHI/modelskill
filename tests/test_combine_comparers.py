import pytest

from modelskill import ModelResult
from modelskill import PointObservation, TrackObservation
from modelskill import Connector


@pytest.fixture
def mrmike():
    fn = "tests/testdata/SW/HKZN_local_2017_DutchCoast.dfsu"
    return ModelResult(fn, name="SW_1", item=0)


@pytest.fixture
def mrmike2():
    fn = "tests/testdata/SW/HKZN_local_2017_DutchCoast_v2.dfsu"
    return ModelResult(fn, name="SW_2", item=0)


@pytest.fixture
def mr2days():
    fn = "tests/testdata/SW/CMEMS_DutchCoast_*.nc"
    return ModelResult(fn, name="CMEMS", item="VHM0")


@pytest.fixture
def mr28():
    fn = "tests/testdata/SW/CMEMS_DutchCoast_2017-10-28.nc"
    return ModelResult(fn, name="CMEMS", item="VHM0")


@pytest.fixture
def mr29():
    fn = "tests/testdata/SW/CMEMS_DutchCoast_2017-10-29.nc"
    return ModelResult(fn, name="CMEMS", item="VHM0")


@pytest.fixture
def o123():
    fn = "tests/testdata/SW/HKNA_Hm0.dfs0"
    o1 = PointObservation(fn, item=0, x=4.2420, y=52.6887, name="HKNA")

    fn = "tests/testdata/SW/eur_Hm0.dfs0"
    o2 = PointObservation(fn, item=0, x=3.2760, y=51.9990, name="EPL")

    fn = "tests/testdata/SW/Alti_c2_Dutch.dfs0"
    o3 = TrackObservation(fn, item=3, name="c2")

    return o1, o2, o3


def test_concat_time(o123, mr28, mr29, mr2days):
    con1 = Connector(o123, mr28)
    with pytest.warns(UserWarning, match="No overlapping data"):
        cc1 = con1.extract()
    con2 = Connector(o123, mr29)
    cc2 = con2.extract()

    # Note: the multi-file dataset will have interpolated values
    # between 23:00 the first day and 00:00 the second day
    # that is NOT the case with the concatenated cc12b
    con12 = Connector(o123, mr2days)
    cc12 = con12.extract()

    assert cc1.start == cc12.start
    assert cc2.end == cc12.end

    cc12b = cc1 + cc2
    assert cc1.start == cc12b.start
    assert cc2.end == cc12b.end
    assert cc12b.n_points < cc12.n_points
    assert cc12b.n_points == cc1.n_points + cc2.n_points


def test_concat_model(o123, mrmike, mrmike2):
    con1 = Connector(o123, mrmike)
    cc1 = con1.extract()
    con2 = Connector(o123, mrmike2)
    cc2 = con2.extract()

    con12 = Connector(o123, [mrmike, mrmike2])
    cc12 = con12.extract()

    assert len(cc1.mod_names) == 1
    assert len(cc12.mod_names) == 2
    assert cc1.mod_names[0] == cc12.mod_names[0]
    assert cc2.mod_names[0] == cc12.mod_names[-1]
    assert cc2.end == cc12.end

    cc12b = cc1 + cc2
    assert cc12b.score() == cc12.score()
    assert cc12b.n_points == cc12.n_points


def test_concat_model_different_time(o123, mrmike, mr2days):
    con1 = Connector(o123, mrmike)
    cc1 = con1.extract()
    con2 = Connector(o123, mr2days)
    cc2 = con2.extract()

    # note the time coverage is different for the two models
    con12 = Connector(o123, [mrmike, mr2days])
    cc12 = con12.extract()

    assert len(cc1.mod_names) == 1
    assert len(cc12.mod_names) == 2
    assert cc1.mod_names[0] == cc12.mod_names[0]
    assert cc2.mod_names[0] == cc12.mod_names[-1]
    assert cc2.end == cc12.end

    cc12b = cc1 + cc2
    assert cc12b.score() == cc12.score()
    assert cc12b.n_points == cc12.n_points


def test_concat_same_model(o123, mrmike):
    con1 = Connector(o123, mrmike)
    cc1 = con1.extract()
    con2 = Connector(o123, mrmike)
    cc2 = con2.extract()

    # if we add the same model multiple times it has no effect
    cc12 = cc1 + cc2
    assert cc12.n_points == cc1.n_points
    assert cc12[0].data.time.to_index().is_unique
    assert cc1.score() == cc12.score()


def test_concat_time_overlap(o123, mrmike):
    con1 = Connector(o123, mrmike)
    cc1 = con1.extract()

    # if there they don't cover the same period...
    o1 = o123[0].copy()
    o1.data = o1.data["2017-10-26":"2017-10-27"]

    o2 = o123[1].copy()
    o2.data = o2.data["2017-10-26":"2017-10-27"]

    o3 = o123[2].copy()
    o3.data = o3.data["2017-10-26":"2017-10-27"]

    con26 = Connector([o1, o2, o3], mrmike)
    cc26 = con26.extract()

    assert cc1.start == cc26.start
    assert cc1.end > cc26.end
    assert cc1.n_points > cc26.n_points

    # cc26 completely contained in cc1
    cc12 = cc1 + cc26
    assert cc1.start == cc12.start
    assert cc1.end == cc12.end
    assert cc1.n_points == cc12.n_points
    assert cc1.score() == cc12.score()

    o1 = o123[0].copy()
    o1.data = o1.data["2017-10-27 12:00":"2017-10-29 23:00"]

    o2 = o123[1].copy()
    o2.data = o2.data["2017-10-27 12:00":"2017-10-29 23:00"]

    o3 = o123[2].copy()
    o3.data = o3.data["2017-10-27 12:00":"2017-10-29 23:00"]

    con2 = Connector([o1, o2, o3], mrmike)
    cc2 = con2.extract()

    # cc26 _not_ completely contained in cc2
    cc12 = cc26 + cc2
    assert cc2.start > cc12.start
    assert cc2.end == cc12.end
    assert cc2.n_points < cc12.n_points

    cc12a = cc2 + cc26
    assert cc12a.n_points == cc12.n_points
