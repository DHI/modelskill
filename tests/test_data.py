import modelskill as ms


def test_load_vistula():
    cc = ms.data.vistula()
    assert isinstance(cc, ms.ComparerCollection)


def test_load_oresund():
    cc = ms.data.oresund()
    assert isinstance(cc, ms.ComparerCollection)
