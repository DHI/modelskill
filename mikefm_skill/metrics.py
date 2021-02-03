
import numpy as np

def mean_absolute_error(obs: np.ndarray, model: np.ndarray, weights: np.ndarray=None):
    """Mean Absolute Error (MAE)"""

    error = np.average(np.abs(obs - model), weights=weights)
    return error

def root_mean_squared_error(obs: np.ndarray, model: np.ndarray, weights: np.ndarray=None):
    """Root Mean Squared Error (RMSE)"""
    residual = obs - model
    error = np.sqrt(np.average(residual**2, weights=weights))

    return error

def nash_sutcliffe_efficiency(obs, model):
    """Nash-Sutcliffe Efficiency (NSE)"""
    error = 1 - (
            np.sum((obs - model)** 2)
            / np.sum((model - np.mean(model))** 2)
    )

    return error