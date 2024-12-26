"""
Toy datasets for testing and demonstration purposes

Examples
--------
```{python}
>>> import modelskill as ms
>>> cc = ms.data.vistula()
>>> cc
```
```{python}
>>> cc = ms.data.oresund()
>>> cc
```
"""

import modelskill as ms
from pathlib import Path

_DATA_ROOT = Path(__file__).parent.parent / "data"


def vistula():
    """10-year discharge data for Vistula catchment, Poland

    Returns
    -------
    ComparerCollection
        _description_
    """
    return ms.load(_DATA_ROOT / "vistula.msk")


def oresund():
    """Oresund water level data for 2022

    Returns
    -------
    ComparerCollection
        _description_
    """
    return ms.load(_DATA_ROOT / "oresund.msk")
