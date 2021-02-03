import numpy as np


def bias(obs, model) -> float:
    "Bias (model - obs)"
    return np.mean(model - obs)


def mean_absolute_error(
    obs: np.ndarray, model: np.ndarray, weights: np.ndarray = None
) -> float:
    """Mean Absolute Error (MAE)"""

    error = np.average(np.abs(model - obs), weights=weights)
    return error


def rmse(
    obs: np.ndarray,
    model: np.ndarray,
    weights: np.ndarray = None,
    unbiased: bool = False,
) -> float:
    return root_mean_squared_error(obs, model, weights, unbiased)


def root_mean_squared_error(
    obs: np.ndarray,
    model: np.ndarray,
    weights: np.ndarray = None,
    unbiased: bool = False,
) -> float:
    """Root Mean Squared Error (RMSE)"""
    residual = obs - model
    if unbiased:
        residual = residual - residual.mean()
    error = np.sqrt(np.average(residual ** 2, weights=weights))

    return error


def nash_sutcliffe_efficiency(obs, model) -> float:
    """Nash-Sutcliffe Efficiency (NSE)"""
    error = 1 - (np.sum((obs - model) ** 2) / np.sum((model - np.mean(model)) ** 2))

    return error


def corr_coef(obs, model) -> float:

    return np.corrcoef(obs, model)[0, 1]


def scatter_index(obs, model) -> float:

    return np.sqrt(
        np.sum(((model - model.mean()) - (obs - obs.mean())) ** 2) / np.sum(obs ** 2)
    )


def r2(obs, model) -> float:
    """Coefficient of determination"""

    residual = model - obs

    SSt = np.sum(obs ** 2)
    SSr = np.sum(residual ** 2)

    return 1 - SSr / SSt
