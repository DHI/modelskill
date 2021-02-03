
import numpy as np

def bias(obs, model):
    "Bias (model - obs)"
    return np.mean(model - obs)

def mean_absolute_error(obs: np.ndarray, model: np.ndarray, weights: np.ndarray=None):
    """Mean Absolute Error (MAE)"""

    error = np.average(np.abs(model - obs), weights=weights)
    return error

def rmse(obs: np.ndarray, model: np.ndarray, weights: np.ndarray=None):
    return root_mean_squared_error(obs, model, weights)

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

def corr_coef(obs, model):

    return np.corrcoef(obs, model)[0,1]

def scatter_index(obs, model):

    return np.sqrt( np.sum(((model - model.mean()) - (obs - obs.mean()))**2) / np.sum(obs**2) )