"""The `metrics` module contains different skill metrics for evaluating the 
difference between a model and an observation. 

* bias
* max_error
* root_mean_squared_error (rmse)    
* urmse
* mean_absolute_error (mae)
* mean_absolute_percentage_error (mape)
* nash_sutcliffe_efficiency (nse)
* kling_gupta_efficiency (kge)
* r2 (r2=nse)
* model_efficiency_factor (mef)
* scatter_index (si)
* corrcoef (cc)
* spearmanr (rho)
* lin_slope
* hit_ratio

The names in parentheses are shorthand aliases for the different metrics.

Examples
--------
>>> obs = np.array([0.3, 2.1, -1.0])
>>> mod = np.array([0.0, 2.3, 1.0])
>>> bias(obs, mod)
0.6333333333333332
>>> max_error(obs, mod)
2.0
>>> rmse(obs, mod)
1.173314393786536
>>> urmse(obs, mod)
0.9877021593352702
>>> mae(obs, mod)
0.8333333333333331
>>> mape(obs, mod)
103.17460317460316
>>> nse(obs, mod)
0.14786795048143053
>>> r2(obs, mod)
0.14786795048143053
>>> mef(obs, mod)
0.9231099877688299
>>> si(obs, mod)
0.8715019052958266
>>> spearmanr(obs, mod)
0.5
>>> cc(obs, mod)
0.637783218973691
>>> lin_slope(obs, mod)
0.4724896836313617
>>> willmott(obs, mod)
0.7484604452865941
>>> hit_ratio(obs, mod, a=0.5)
0.6666666666666666
"""
import sys
from typing import Callable, Set, Tuple, Union
import warnings
import numpy as np


def bias(obs, model) -> float:
    """Bias (mean error)

    .. math::
        bias=\\frac{1}{n}\\sum_{i=1}^n (model_i - obs_i)

    Range: :math:`(-\\infty, \\infty)`; Best: 0
    """

    assert obs.size == model.size
    return np.mean(model.ravel() - obs.ravel())


def max_error(obs, model) -> float:
    """Max (absolute) error

    .. math::
        max_{error} = max(|model_i - obs_i|)

    Range: :math:`[0, \\infty)`; Best: 0
    """

    assert obs.size == model.size
    return np.max(np.abs(model.ravel() - obs.ravel()))


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

    Range: :math:`[0, \\infty)`; Best: 0
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

    Range: :math:`[0, \\infty)`; Best: 0
    """

    assert obs.size == model.size

    if len(obs) == 0:
        return np.nan
    if np.any(obs == 0.0):
        warnings.warn("Observation is zero, consider to use another metric than MAPE")
        return np.nan  # TODO is it better to return a large value +inf than NaN?

    return np.mean(np.abs((obs.ravel() - model.ravel()) / obs.ravel())) * 100


def urmse(obs: np.ndarray, model: np.ndarray, weights: np.ndarray = None) -> float:
    """Unbiased Root Mean Squared Error (uRMSE)

    .. math::

        res_i = model_i - obs_i

        res_{u,i} = res_i - \\overline {res}

        uRMSE = \\sqrt{\\frac{1}{n} \\sum_{i=1}^n res_{u,i}^2}

    Range: :math:`[0, \\infty)`; Best: 0

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
        res_i = model_i - obs_i

        RMSE=\\sqrt{\\frac{1}{n} \\sum_{i=1}^n res_i^2}

    Unbiased version:

    .. math::

        res_{u,i} = res_i - \\overline {res}

        uRMSE=\\sqrt{\\frac{1}{n} \\sum_{i=1}^n res_{u,i}^2}

    Range: :math:`[0, \\infty)`; Best: 0

    """
    assert obs.size == model.size

    residual = obs.ravel() - model.ravel()
    if unbiased:
        residual = residual - residual.mean()
    error = np.sqrt(np.average(residual**2, weights=weights))

    return error


def nse(obs: np.ndarray, model: np.ndarray) -> float:
    """alias for nash_sutcliffe_efficiency"""
    return nash_sutcliffe_efficiency(obs, model)


def nash_sutcliffe_efficiency(obs: np.ndarray, model: np.ndarray) -> float:
    """Nash-Sutcliffe Efficiency (NSE)

    .. math::

        NSE = 1 - \\frac {\\sum _{i=1}^{n}\\left(model_{i} - obs_{i}\\right)^{2}}
                       {\\sum_{i=1}^{n}\\left(obs_{i} - {\\overline{obs}}\\right)^{2}}

    Range: :math:`(-\\infty, 1]`; Best: 1

    Note
    ----
    r2 = nash_sutcliffe_efficiency

    References
    ----------
    Nash, J. E.; Sutcliffe, J. V. (1970). "River flow forecasting through conceptual models part I — A discussion of principles". Journal of Hydrology. 10 (3): 282–290.
    """
    assert obs.size == model.size

    if len(obs) == 0:
        return np.nan
    error = 1 - (
        np.sum((obs.ravel() - model.ravel()) ** 2)
        / np.sum((obs.ravel() - np.mean(obs.ravel())) ** 2)
    )

    return error


def kling_gupta_efficiency(obs: np.ndarray, model: np.ndarray) -> float:
    """
    Kling-Gupta Efficiency (KGE)

    .. math::

        KGE = 1 - \\sqrt{(r-1)^2 + \\left(\\frac{\\sigma_{mod}}{\\sigma_{obs}} - 1\\right)^2 +
                                   \\left(\\frac{\\mu_{mod}}{\\mu_{obs}} - 1\\right)^2 }

    where :math:`r` is the pearson correlation coefficient, :math:`\\mu_{obs},\\mu_{mod}` and :math:`\\sigma_{obs},\\sigma_{mod}` is the mean and standard deviation of observations and model.

    Range: :math:`(-\\infty, 1]`; Best: 1

    References
    ----------
    Gupta, H. V., Kling, H., Yilmaz, K. K. and Martinez, G. F., (2009), Decomposition of the mean squared error and NSE performance criteria: Implications for improving hydrological modelling, J. Hydrol., 377(1-2), 80-91

    Knoben, W. J. M., Freer, J. E., and Woods, R. A. (2019) Technical note: Inherent benchmark or not? Comparing Nash–Sutcliffe and Kling–Gupta efficiency scores, Hydrol. Earth Syst. Sci., 23, 4323-4331
    """
    assert obs.size == model.size

    if len(obs) == 0 or obs.std() == 0.0:
        return np.nan

    if model.std() > 1e-12:
        r = corrcoef(obs, model)
        if np.isnan(r):
            r = 0.0
    else:
        r = 0.0

    res = 1 - np.sqrt(
        (r - 1) ** 2
        + (model.std() / obs.std() - 1.0) ** 2
        + (model.mean() / obs.mean() - 1.0) ** 2
    )

    return res


def kge(obs: np.ndarray, model: np.ndarray) -> float:
    """alias for kling_gupta_efficiency"""
    return kling_gupta_efficiency(obs, model)


def r2(obs: np.ndarray, model: np.ndarray) -> float:
    """Coefficient of determination (R2)

    Pronounced 'R-squared'; the proportion of the variation in the dependent variable that is predictable from the independent variable(s), i.e. the proportion of explained variance.

    .. math::

        R^2 = 1 - \\frac{\\sum_{i=1}^n (model_i - obs_i)^2}
                    {\\sum_{i=1}^n (obs_i - \\overline {obs})^2}

    Range: :math:`(-\\infty, 1]`; Best: 1

    Note
    ----
    r2 = nash_sutcliffe_efficiency

    Examples
    --------
    >>> obs = np.array([1.0,1.1,1.2,1.3,1.4])
    >>> model = np.array([1.09, 1.16, 1.3 , 1.38, 1.49])
    >>> r2(obs,model)
    0.6379999999999998
    """
    assert obs.size == model.size
    if len(obs) == 0:
        return np.nan

    residual = model.ravel() - obs.ravel()
    SSr = np.sum(residual**2)
    SSt = np.sum((obs - obs.mean()) ** 2)

    return 1 - SSr / SSt


def mef(obs: np.ndarray, model: np.ndarray) -> float:
    """alias for model_efficiency_factor"""
    return model_efficiency_factor(obs, model)


def model_efficiency_factor(obs: np.ndarray, model: np.ndarray) -> float:
    """Model Efficiency Factor (MEF)

    Scale independent RMSE, standardized by Stdev of observations

    .. math::

        MEF = \\frac{RMSE}{STDEV}=\\frac{\\sqrt{\\frac{1}{n} \\sum_{i=1}^n(model_i - obs_i)^2}}
                                        {\\sqrt{\\frac{1}{n} \\sum_{i=1}^n(obs_i - \\overline{obs})^2}}=\\sqrt{1-NSE}

    Range: :math:`[0, \\infty)`; Best: 0

    See Also
    --------
    nash_sutcliffe_efficiency
    root_mean_squared_error

    """
    assert obs.size == model.size

    return rmse(obs, model) / obs.std()


def cc(obs: np.ndarray, model: np.ndarray, weights=None) -> float:
    """alias for corrcoef"""
    return corrcoef(obs, model, weights)


def corrcoef(obs, model, weights=None) -> float:
    """Pearson’s Correlation coefficient (CC)

    .. math::
        CC = \\frac{\\sum_{i=1}^n (model_i - \\overline{model})(obs_i - \\overline{obs}) }
                   {\\sqrt{\\sum_{i=1}^n (model_i - \\overline{model})^2}
                    \\sqrt{\\sum_{i=1}^n (obs_i - \\overline{obs})^2} }

    Range: [-1, 1]; Best: 1

    See Also
    --------
    np.corrcoef
    """
    assert obs.size == model.size
    if len(obs) <= 1:
        return np.nan

    if weights is None:
        return np.corrcoef(obs.ravel(), model.ravel())[0, 1]
    else:
        C = np.cov(obs.ravel(), model.ravel(), fweights=weights)
        return C[0, 1] / np.sqrt(C[0, 0] * C[1, 1])


def rho(obs: np.ndarray, model: np.ndarray) -> float:
    """alias for spearmanr"""
    return spearmanr(obs, model)


def spearmanr(obs: np.ndarray, model: np.ndarray) -> float:
    """Spearman rank correlation coefficient

    The rank correlation coefficient is similar to the Pearson correlation coefficient but
    applied to ranked quantities and is useful to quantify a monotonous relationship

    .. math::
        \\rho = \\frac{\\sum_{i=1}^n (rmodel_i - \\overline{rmodel})(robs_i - \\overline{robs}) }
                      {\\sqrt{\\sum_{i=1}^n (rmodel_i - \\overline{rmodel})^2}
                       \\sqrt{\\sum_{i=1}^n (robs_i - \\overline{robs})^2} }

    Range: [-1, 1]; Best: 1

    Examples
    --------
    >>> obs = np.linspace(-20, 20, 100)
    >>> mod = np.tanh(obs)
    >>> rho(obs, mod)
    0.9999759973116955
    >>> spearmanr(obs, mod)
    0.9999759973116955

    See Also
    --------
    corrcoef
    """
    import scipy.stats

    return scipy.stats.spearmanr(obs, model)[0]


def si(obs: np.ndarray, model: np.ndarray) -> float:
    """alias for scatter_index"""
    return scatter_index(obs, model)


def scatter_index(obs: np.ndarray, model: np.ndarray) -> float:
    """Scatter index (SI)

    Which is the same as the unbiased-RMSE normalized by the absolute mean of the observations.

    .. math::
        \\frac{ \\sqrt{ \\frac{1}{n} \\sum_{i=1}^n \\left( (model_i - \\overline {model}) - (obs_i - \\overline {obs}) \\right)^2} }
        {\\frac{1}{n} \\sum_{i=1}^n | obs_i | }

    Range: [0, \\infty); Best: 0
    """
    assert obs.size == model.size
    if len(obs) == 0:
        return np.nan

    residual = obs.ravel() - model.ravel()
    residual = residual - residual.mean()  # unbiased
    return np.sqrt(np.mean(residual**2)) / np.mean(np.abs(obs.ravel()))


def scatter_index2(obs: np.ndarray, model: np.ndarray) -> float:
    """Alternative formulation of the scatter index (SI)

    .. math::
        \\sqrt {\\frac{\\sum_{i=1}^n \\left( (model_i - \\overline {model}) - (obs_i - \\overline {obs}) \\right)^2}
        {\\sum_{i=1}^n obs_i^2}}

    Range: [0, 100]; Best: 0
    """
    assert obs.size == model.size
    if len(obs) == 0:
        return np.nan

    return np.sqrt(
        np.sum(((model.ravel() - model.mean()) - (obs.ravel() - obs.mean())) ** 2)
        / np.sum(obs.ravel() ** 2)
    )


def willmott(obs: np.ndarray, model: np.ndarray) -> float:
    """Willmott's Index of Agreement

    A scaled representation of the predictive accuracy of the model against observations. A value of 1 indicates a perfect match, and 0 indicates no agreement at all.

    .. math::

        willmott = 1 - \\frac{\\frac{1}{n} \\sum_{i=1}^n(model_i - obs_i)^2}
                           {\\frac{1}{n} \\sum_{i=1}^n(|model_i - \\overline{obs}| + |obs_i - \\overline{obs}|)^2}

    Range: [0, 1]; Best: 1

    Examples
    --------
    >>> obs = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.4, 1.3])
    >>> model = np.array([1.02, 1.16, 1.3, 1.38, 1.49, 1.45, 1.32])
    >>> willmott(obs, model)
    0.9501403174479723

    References
    ----------
    Willmott, C. J. 1981. "On the validation of models". Physical Geography, 2, 184–194.
    """

    assert obs.size == model.size
    if len(obs) == 0:
        return np.nan

    residual = model.ravel() - obs.ravel()
    nominator = np.sum(residual**2)
    denominator = np.sum((np.abs(model - obs.mean()) + np.abs(obs - obs.mean())) ** 2)

    return 1 - nominator / denominator


def hit_ratio(obs: np.ndarray, model: np.ndarray, a=0.1) -> float:
    """Fraction within obs ± acceptable deviation

    .. math::

        HR = \\frac{1}{n}\\sum_{i=1}^n I_{|(model_i - obs_i)|} < a

    Range: [0, 1]; Best: 1

    Examples
    --------
    >>> obs = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.4, 1.3])
    >>> model = np.array([1.02, 1.16, 1.3, 1.38, 1.49, 1.45, 1.32])
    >>> hit_ratio(obs, model, a=0.05)
    0.2857142857142857
    >>> hit_ratio(obs, model, a=0.1)
    0.8571428571428571
    >>> hit_ratio(obs, model, a=0.15)
    1.0
    """
    assert obs.size == model.size

    return np.mean(np.abs(obs.ravel() - model.ravel()) < a)


def lin_slope(obs: np.ndarray, model: np.ndarray, reg_method="ols") -> float:
    """Slope of the regression line.

    .. math::

        slope = \\frac{\\sum_{i=1}^n (model_i - \\overline {model})(obs_i - \\overline {obs})}
                      {\\sum_{i=1}^n (obs_i - \\overline {obs})^2}

    Range: :math:`(-\\infty, \\infty )`; Best: 1
    """
    assert obs.size == model.size
    return _linear_regression(obs.ravel(), model.ravel(), reg_method)[0]


def _linear_regression(
    obs: np.ndarray, model: np.ndarray, reg_method="ols"
) -> Tuple[float, float]:

    if len(obs) == 0:
        return np.nan

    if reg_method == "ols":
        from scipy.stats import linregress as _linregress

        reg = _linregress(obs, model)
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


def _std_obs(obs: np.ndarray, model: np.ndarray) -> float:
    return obs.std()


def _std_mod(obs: np.ndarray, model: np.ndarray) -> float:
    return model.std()


METRICS_WITH_DIMENSION = set(["urmse", "rmse", "bias", "mae"])  # TODO is this complete?


def metric_has_units(metric: Union[str, Callable]) -> bool:
    """Check if a metric has units (dimension).

    Some metrics are dimensionless, others have the same dimension as the observations.

    Parameters
    ----------
    metric : str or callable
        Metric name or function

    Returns
    -------
    bool
        True if metric has a dimension, False otherwise

    Examples
    --------
    >>> metric_has_units("rmse")
    True
    >>> metric_has_units("kge")
    False
    """
    if isinstance(metric, Callable):
        name = metric.__name__
    else:
        name = metric

    if name not in DEFINED_METRICS:
        raise ValueError(f"Metric {name} not defined. Choose from {DEFINED_METRICS}")

    return name in METRICS_WITH_DIMENSION


DEFINED_METRICS: Set[str] = set(
    [
        func
        for func in dir()
        if callable(getattr(sys.modules[__name__], func)) and not func.startswith("_")
    ]
)
