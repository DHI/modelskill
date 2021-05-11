import warnings
import numpy as np


def bias(obs, model) -> float:
    """Bias (model - obs)

    .. math::
        bias=\\frac{1}{n}\\sum_{i=1}^n (model_i - obs_i)
    """

    assert obs.size == model.size
    return np.mean(model.ravel() - obs.ravel())


def mae(obs: np.ndarray, model: np.ndarray, weights: np.ndarray = None) -> float:
    """alias for mean_absolute_error"""
    assert obs.size == model.size
    return mean_absolute_error(obs, model, weights)


def mean_absolute_error(
    obs: np.ndarray, model: np.ndarray, weights: np.ndarray = None
) -> float:
    """Mean Absolute Error (MAE)

    .. math::
        MAE=\\frac{1}{n}\\sum_{i=1}^n|model_i - obs_i|
    """
    assert obs.size == model.size

    error = np.average(np.abs(model.ravel() - obs.ravel()), weights=weights)

    return error


def mape(obs: np.ndarray, model: np.ndarray) -> float:
    """alias for mean_absolute_percentage_error"""
    return mean_absolute_percentage_error(obs, model)


def mean_absolute_percentage_error(obs: np.ndarray, model: np.ndarray) -> float:
    """Mean Absolute Percentage Error (MAPE)

    .. math::
        MAPE=\\frac{1}{n}\\sum_{i=1}^n\\frac{|model_i - obs_i|}{obs_i}*100
    """

    assert obs.size == model.size

    if np.any(obs == 0.0):
        warnings.warn("Observation is zero, consider to use another metric than MAPE")
        return np.nan  # TODO is it better to return a large value +inf than NaN?

    return np.mean(np.abs((obs.ravel() - model.ravel()) / obs.ravel())) * 100


def urmse(obs: np.ndarray, model: np.ndarray, weights: np.ndarray = None) -> float:
    """Unbiased Root Mean Squared Error (uRMSE)

    See Also
    --------
    root_mean_squared_error
    """
    return root_mean_squared_error(obs, model, weights, unbiased=True)


def rmse(
    obs: np.ndarray,
    model: np.ndarray,
    weights: np.ndarray = None,
    unbiased: bool = False,
) -> float:
    """alias for root_mean_squared_error"""
    return root_mean_squared_error(obs, model, weights, unbiased)


def root_mean_squared_error(
    obs: np.ndarray,
    model: np.ndarray,
    weights: np.ndarray = None,
    unbiased: bool = False,
) -> float:
    """Root Mean Squared Error (RMSE)

    .. math::
        res_i = obs_i - model_i

        RMSE=\\sqrt{\\sum_{i=1}^n res_i^2}

    Unbiased version:

    .. math::

        res_{u,i} = res_i - \\overline res

        RMSE_u=\\sqrt{\\sum_{i=1}^n res_{u,i}^2}

    """
    assert obs.size == model.size

    residual = obs.ravel() - model.ravel()
    if unbiased:
        residual = residual - residual.mean()
    error = np.sqrt(np.average(residual ** 2, weights=weights))

    return error


def nse(obs: np.ndarray, model: np.ndarray) -> float:
    """alias for nash_sutcliffe_efficiency"""
    return nash_sutcliffe_efficiency(obs, model)


def nash_sutcliffe_efficiency(obs: np.ndarray, model: np.ndarray) -> float:
    """Nash-Sutcliffe Efficiency (NSE)

    .. math::

        NSE = 1 - \\frac {\\sum _{i=i}^{n}\\left(model_{i}-obs_{i}\\right)^{2}}
                       {\\sum_{i=1}^{n}\\left(obs_{i}-{\\overline {obs}}\\right)^{2}}

    References
    ----------
    Nash, J. E.; Sutcliffe, J. V. (1970). "River flow forecasting through conceptual models part I — A discussion of principles". Journal of Hydrology. 10 (3): 282–290.
    """
    assert obs.size == model.size

    error = 1 - (
        np.sum((obs.ravel() - model.ravel()) ** 2)
        / np.sum((model.ravel() - np.mean(model.ravel())) ** 2)
    )

    return error


def model_efficiency_factor(obs: np.ndarray, model: np.ndarray) -> float:
    """Model Efficiency Factor (MEF)

    Scale independendt RMSE, standardized by Stdev of observations

    .. math::

        MEF = \\frac{RMSE}{STDEV}=\\frac{\\sqrt{\\sum_{i=1}^n(obs_i - model_i)^2}}
                                        {\\sqrt{\\sum_{i=1}^n(obs_i - \\overline obs)^2}}=\\sqrt{1-NSE}

    See Also
    --------
    nash_sutcliffe_efficiency
    rmse

    """
    assert obs.size == model.size

    return rmse(obs, model) / obs.std()


def cc(obs: np.ndarray, model: np.ndarray, weights=None) -> float:
    """alias for corrcoef"""
    return corrcoef(obs, model)


def corrcoef(obs, model, weights=None) -> float:
    """Correlation coefficient (CC)

    .. math::
        CC=\\frac{cov(obs,model)}{\\sigma_{obs}\\sigma_{model}}

    See Also
    --------
    numpy.corrcoef
    """
    assert obs.size == model.size

    if weights is None:
        return np.corrcoef(obs.ravel(), model.ravel())[0, 1]
    else:
        C = np.cov(obs.ravel(), model.ravel(), fweights=weights)
        return C[0, 1] / np.sqrt(C[0, 0] * C[1, 1])


def si(obs: np.ndarray, model: np.ndarray) -> float:
    """alias for scatter_index"""
    return scatter_index(obs, model)


def scatter_index(obs: np.ndarray, model: np.ndarray) -> float:
    """Scatter index (SI)

    .. math::
        \\frac{\\sum_i^n \\left( (model_i - \\overline model) - (obs_i - \\overline obs) \\right)^2}
        {\\sum_i^n obs_i^2}

    """
    assert obs.size == model.size

    return np.sqrt(
        np.sum(((model.ravel() - model.mean()) - (obs.ravel() - obs.mean())) ** 2)
        / np.sum(obs.ravel() ** 2)
    )


def r2(obs: np.ndarray, model: np.ndarray) -> float:
    """Coefficient of determination (R2)

    .. math::

        R^2 = 1 - \\frac{\\sum_{i=1}^n (model_i - obs_i)^2}
                    {\\sum_{i=1}^n obs_i^2}
    """
    assert obs.size == model.size

    residual = model.ravel() - obs.ravel()
    SSr = np.sum(residual ** 2)
    SSt = np.sum(obs.ravel() ** 2)

    return 1 - SSr / SSt
