"""Package-internal helpers shared across modelskill subpackages.

The leading underscore on the module name signals that this is internal API:
modelskill itself imports freely from here, but downstream consumers must not.
See ADR-012 for the public/private convention.
"""

from __future__ import annotations
from typing import Sequence, cast
from collections.abc import Hashable


RESERVED_COORD_NAMES = ["x", "y", "z", "time"]
RESERVED_COMPARER_VAR_NAMES = [*RESERVED_COORD_NAMES, "Observation"]


def get_name(x: int | str | None, valid_names: Sequence[Hashable]) -> str:
    """Parse name/idx from list of valid names (e.g. obs from obs_names), return name."""
    return cast(str, valid_names[get_idx(x, valid_names)])


def get_idx(x: int | str | None, valid_names: Sequence[Hashable]) -> int:
    """Parse name/idx from list of valid names (e.g. obs from obs_names), return idx."""

    if x is None:
        if len(valid_names) == 1:
            return 0
        else:
            raise ValueError(
                f"Multiple items available. Must specify name or index. Available items: {valid_names}"
            )

    n = len(valid_names)
    if n == 0:
        raise ValueError(f"Cannot select {x} from empty list!")
    elif isinstance(x, str):
        if x in valid_names:
            idx = valid_names.index(x)
        else:
            raise KeyError(f"Name {x} could not be found in {valid_names}")
    elif isinstance(x, int):
        if x < 0:
            x += n
        if x >= 0 and x < n:
            idx = x
        else:
            raise IndexError(f"Id {x} is out of range for {valid_names}")
    else:
        raise TypeError(f"Input {x} invalid! Must be None, str or int, not {type(x)}")
    return idx
