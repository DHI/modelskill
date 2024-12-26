import modelskill as ms


def vistula():
    """10-year discharge data for Vistula catchment, Poland

    Returns
    -------
    ComparerCollection
        _description_
    """
    fn = "./data/vistula.msk"
    return ms.load(fn)


def oresund():
    """Oresund water level data for 2022

    Returns
    -------
    ComparerCollection
        _description_
    """
    fn = "./data/oresund.msk"
    return ms.load(fn)
