import pytest

from fmskill.observation import TrackObservation


@pytest.fixture
def c2():
    return "tests/testdata/SW/Alti_c2_Dutch.dfs0"


def test_read(c2):
    o1 = TrackObservation(c2, item=2, name="c2")
    assert o1.n_points == 298
    assert len(o1.x) == o1.n_points
    assert o1.name == "c2"
    assert pytest.approx(o1.values.max()) == 17.67
