import modelskill as ms


def test_load_vistula():
    cc = ms.data.vistula()
    assert cc is not None
