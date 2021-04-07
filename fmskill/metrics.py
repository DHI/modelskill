import warnings
import numpy as np


def bias(obs, model) -> float:
    """Bias (model - obs)"""
    return np.mean(model.ravel() - obs.ravel())


def mae(obs: np.ndarray, model: np.ndarray, weights: np.ndarray = None) -> float:
    """Mean Absolute Error (MAE)"""
    return mean_absolute_error(obs, model, weights)


def mean_absolute_error(
    obs: np.ndarray, model: np.ndarray, weights: np.ndarray = None
) -> float:
    """Mean Absolute Error (MAE)"""
    error = np.average(np.abs(model.ravel() - obs.ravel()), weights=weights)
    return error


def mape(obs: np.ndarray, model: np.ndarray) -> float:
    """Mean Absolute Percentage Error (MAPE)"""
    return mean_absolute_percentage_error(obs, model)


def mean_absolute_percentage_error(obs: np.ndarray, model: np.ndarray) -> float:
    """Mean Absolute Percentage Error (MAPE)"""
    if np.any(obs == 0.0):
        warnings.warn("Observation is zero, consider to use another metric than MAPE")
        return np.nan  # TODO is it better to return a large value +inf than NaN?

    return np.mean(np.abs((obs.ravel() - model.ravel()) / obs.ravel())) * 100


def urmse(obs: np.ndarray, model: np.ndarray, weights: np.ndarray = None) -> float:
    """Unbiased Root Mean Squared Error (uRMSE)"""
    return root_mean_squared_error(obs, model, weights, unbiased=True)


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
    residual = obs.ravel() - model.ravel()
    if unbiased:
        residual = residual - residual.mean()
    error = np.sqrt(np.average(residual ** 2, weights=weights))

    return error


def nash_sutcliffe_efficiency(obs, model) -> float:
    """Nash-Sutcliffe Efficiency (NSE)"""
    error = 1 - (
        np.sum((obs.ravel() - model.ravel()) ** 2)
        / np.sum((model.ravel() - np.mean(model.ravel())) ** 2)
    )

    return error


def cc(obs: np.ndarray, model: np.ndarray, weights=None) -> float:
    """Correlation coefficient (CC)"""
    return corrcoef(obs, model)


def corrcoef(obs, model, weights=None) -> float:
    """Correlation coefficient (CC)"""
    if weights is None:
        return np.corrcoef(obs.ravel(), model.ravel())[0, 1]
    else:
        C = np.cov(obs.ravel(), model.ravel(), fweights=weights)
        return C[0, 1] / np.sqrt(C[0, 0] * C[1, 1])


def si(obs: np.ndarray, model: np.ndarray) -> float:
    """Scatter index (SI)"""
    return scatter_index(obs, model)


def scatter_index(obs, model) -> float:
    """Scatter index (SI)"""
    return np.sqrt(
        np.sum(((model.ravel() - model.mean()) - (obs.ravel() - obs.mean())) ** 2)
        / np.sum(obs.ravel() ** 2)
    )


def r2(obs, model) -> float:
    """Coefficient of determination"""

    residual = model.ravel() - obs.ravel()

    SSt = np.sum(obs.ravel() ** 2)
    SSr = np.sum(residual ** 2)

    return 1 - SSr / SSt
