"""Compare model output with observations.

The `comparison` module contains different types of classes for single
observation comparison (Comparer), and collections of Comparers (ComparerCollection).
"""

from ._comparison import Comparer
from ._collection import ComparerCollection
from ._comparer_plotter import ComparerPlotter
from ._collection_plotter import ComparerCollectionPlotter


__all__ = [
    "Comparer",
    "ComparerCollection",
    "ComparerPlotter",
    "ComparerCollectionPlotter",
]
