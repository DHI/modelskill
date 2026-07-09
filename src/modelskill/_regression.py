"""Linear regression helper shared by ``metrics.lin_slope`` and the scatter trendline.

Private per ADR-012: the leading underscore on the module name signals internal use.
"""

from __future__ import annotations
from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike


def linear_regression(
    obs: ArrayLike, model: ArrayLike, reg_method: str = "ols"
) -> Tuple[float, float]:
    """Fit a linear regression of ``model`` against ``obs`` and return (slope, intercept)."""
    if len(obs) == 0:  # type: ignore[arg-type]
        return np.nan, np.nan

    if reg_method == "ols":
        from scipy.stats import linregress

        reg = linregress(obs, model)
        intercept = reg.intercept
        slope = reg.slope
    elif reg_method == "odr":
        from scipy import odr

        data = odr.Data(obs, model)
        odr_obj = odr.ODR(data, odr.unilinear)
        output = odr_obj.run()

        intercept = output.beta[1]
        slope = output.beta[0]
    else:
        raise NotImplementedError(
            f"Regression method: {reg_method} not implemented, select 'ols' or 'odr'"
        )

    return slope, intercept
