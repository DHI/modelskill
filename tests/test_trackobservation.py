import pytest

from mikefm_skill.observation import TrackObservation


@pytest.fixture
def c2():
    return "tests/testdata/SW/Alti_c2_Dutch.dfs0"


def test_read(c2):
    o1 = TrackObservation(c2, item=2, name="c2")
    assert o1.n > 1
