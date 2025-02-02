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
    """5-year daily discharge data for Vistula catchment, Poland

    Contains discharge data for 8 stations along the Vistula river
    compared with two hydrological models "sim1" and "sim2".

    The dataset additionally contains precipitation data as aux data
    and metadata about the river and the catchment area in the attrs dictionary.

    Returns
    -------
    ComparerCollection
    """
    fn = str(files("modelskill.data") / "vistula.nc")
    return ms.load(fn)


def oresund() -> ComparerCollection:
    """Oresund water level data for Jan-June 2022 compared with MIKE21 model

    Contains water level data for 7 stations along the Oresund strait with
    metadata about the country in the attrs dictionary.

    The dataset contains additional ERA5 wind-components U10 and V10 aux data.

    Returns
    -------
    ComparerCollection
    """
    fn = str(files("modelskill.data") / "oresund.nc")
    return ms.load(fn)
