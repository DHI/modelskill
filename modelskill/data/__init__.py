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

from importlib.resources import files

import modelskill as ms
from ..comparison import ComparerCollection


def vistula() -> ComparerCollection:
    """10-year discharge data for Vistula catchment, Poland

    Returns
    -------
    ComparerCollection
    """
    fn = str(files("modelskill.data") / "vistula.msk")
    return ms.load(fn)


def oresund() -> ComparerCollection:
    """Oresund water level data for 2022

    Returns
    -------
    ComparerCollection
    """
    fn = str(files("modelskill.data") / "oresund.msk")
    return ms.load(fn)
