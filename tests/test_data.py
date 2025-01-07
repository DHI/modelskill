import os

import pytest
import modelskill as ms


@pytest.fixture
def change_test_directory(tmp_path):
    original_directory = os.getcwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(original_directory)


def test_load_vistula(change_test_directory):
    cc = ms.data.vistula()
    assert isinstance(cc, ms.ComparerCollection)


def test_load_oresund(change_test_directory):
    cc = ms.data.oresund()
    assert isinstance(cc, ms.ComparerCollection)
