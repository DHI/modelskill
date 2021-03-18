import warnings
import numpy as np


def bias(obs, model) -> float:
    """Bias (model - obs)"""
    return np.mean(model - obs)


def mae(obs: np.ndarray, model: np.ndarray, weights: np.ndarray = None) -> float:
    """Mean Absolute Error (MAE)"""
    return mean_absolute_error(obs, model, weights)


def mean_absolute_error(
    obs: np.ndarray, model: np.ndarray, weights: np.ndarray = None
) -> float:
    """Mean Absolute Error (MAE)"""
    error = np.average(np.abs(model - obs), weights=weights)
    return error


def mape(obs: np.ndarray, model: np.ndarray) -> float:
    """Mean Absolute Percentage Error (MAPE)"""
    return mean_absolute_percentage_error(obs, model)


def mean_absolute_percentage_error(obs: np.ndarray, model: np.ndarray) -> float:
    """Mean Absolute Percentage Error (MAPE)"""
    if np.any(obs == 0.0):
        warnings.warn("Observation is zero, consider to use another metric than MAPE")
        return np.nan  # TODO is it better to return a large value +inf than NaN?

    return np.mean(np.abs((obs - model) / obs)) * 100


def rmse(
    obs: np.ndarray,
    model: np.ndarray,
    weights: np.ndarray = None,
    unbiased: bool = False,
) -> float:
    """Root Mean Squared Error (RMSE)"""
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


def cc(obs: np.ndarray, model: np.ndarray) -> float:
    """Correlation coefficient (CC)"""
    return corr_coef(obs, model)


def corr_coef(obs, model) -> float:
    """Correlation coefficient (CC)"""
    return np.corrcoef(obs, model)[0, 1]


def si(obs: np.ndarray, model: np.ndarray) -> float:
    """Scatter index (SI)"""
    return scatter_index(obs, model)


def scatter_index(obs, model) -> float:
    """Scatter index (SI)"""
    return np.sqrt(
        np.sum(((model - model.mean()) - (obs - obs.mean())) ** 2) / np.sum(obs ** 2)
    )


def r2(obs, model) -> float:
    """Coefficient of determination"""

    residual = model - obs

    SSt = np.sum(obs ** 2)
    SSr = np.sum(residual ** 2)

    return 1 - SSr / SSt
