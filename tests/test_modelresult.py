import pytest

from mikefm_skill.model import ModelResult

@pytest.fixture
def oresund_2d():
    return "tests/testdata/Oresund2D.dfsu"

def test_dfs_object(oresund_2d):
    mr = ModelResult(oresund_2d)

    assert mr.dfs.is_2d

def test_add_observation(oresund_2d):
    mr = ModelResult(oresund_2d)

    mr.extract()

    