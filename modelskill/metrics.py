"""The `metrics` module contains different skill metrics for evaluating the 
difference between a model and an observation. 

* [bias][modelskill.metrics.bias]
* [max_error][modelskill.metrics.max_error]
* [root_mean_squared_error (rmse)][modelskill.metrics.root_mean_squared_error]    
* [urmse][modelskill.metrics.urmse]
* [mean_absolute_error (mae)][modelskill.metrics.mean_absolute_error]
* [mean_absolute_percentage_error (mape)][modelskill.metrics.mean_absolute_percentage_error]
* [kling_gupta_efficiency (kge)][modelskill.metrics.kling_gupta_efficiency]
* [nash_sutcliffe_efficiency (nse)][modelskill.metrics.nash_sutcliffe_efficiency]
* [r2 (r2=nse)][modelskill.metrics.r2]
* [model_efficiency_factor (mef)][modelskill.metrics.model_efficiency_factor]
* [wilmott][modelskill.metrics.willmott]
* [scatter_index (si)][modelskill.metrics.scatter_index]
* [scatter_index2][modelskill.metrics.scatter_index2]
* [corrcoef (cc)][modelskill.metrics.corrcoef]
* [spearmanr (rho)][modelskill.metrics.spearmanr]
* [lin_slope][modelskill.metrics.lin_slope]
* [hit_ratio][modelskill.metrics.hit_ratio]
* [explained_variance (ev)][modelskill.metrics.explained_variance]
* [peak_ratio (pr)][modelskill.metrics.peak_ratio]

Circular metrics (for directional data with units in degrees):

* [c_bias][modelskill.metrics.c_bias]
* [c_max_error][modelskill.metrics.c_max_error]
* [c_mean_absolute_error (c_mae)][modelskill.metrics.c_mean_absolute_error]
* [c_root_mean_squared_error (c_rmse)][modelskill.metrics.c_root_mean_squared_error]
* [c_unbiased_root_mean_squared_error (c_urmse)][modelskill.metrics.c_unbiased_root_mean_squared_error]

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
>>> willmott(obs, mod)
0.7484604452865941
>>> hit_ratio(obs, mod, a=0.5)
0.6666666666666666
>>> ev(obs, mod)
0.39614855570839064
"""
from __future__ import annotations
import inspect

import sys
import warnings
from typing import Callable, Iterable, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from .settings import options


def bias(obs, model) -> float:
    r"""Bias (mean error)

    $$
    bias=\frac{1}{n}\sum_{i=1}^n (model_i - obs_i)
    $$

    Range: $(-\infty, \infty)$; Best: 0
    """

    assert obs.size == model.size
    return np.mean(model - obs)


def max_error(obs, model) -> float:
    r"""Max (absolute) error

    $$
    max_{error} = max(|model_i - obs_i|)
    $$

    Range: $[0, \infty)$; Best: 0
    """

    assert obs.size == model.size
    return np.max(np.abs(model - obs))


def mae(
    obs: np.ndarray, model: np.ndarray, weights: Optional[np.ndarray] = None
) -> float:
    """alias for mean_absolute_error"""
    assert obs.size == model.size
    return mean_absolute_error(obs, model, weights)


def mean_absolute_error(
    obs: np.ndarray, model: np.ndarray, weights: Optional[np.ndarray] = None
) -> float:
    r"""Mean Absolute Error (MAE)

    $$
    MAE=\frac{1}{n}\sum_{i=1}^n|model_i - obs_i|
    $$

    Range: $[0, \infty)$; Best: 0
    """
    assert obs.size == model.size

    error = np.average(np.abs(model - obs), weights=weights)

    return error


def mape(obs: np.ndarray, model: np.ndarray) -> float:
    """alias for mean_absolute_percentage_error"""
    return mean_absolute_percentage_error(obs, model)


def mean_absolute_percentage_error(obs: np.ndarray, model: np.ndarray) -> float:
    r"""Mean Absolute Percentage Error (MAPE)

    $$
    MAPE=\frac{1}{n}\sum_{i=1}^n\frac{|model_i - obs_i|}{obs_i}*100
    $$

    Range: $[0, \infty)$; Best: 0
    """

    assert obs.size == model.size

    if len(obs) == 0:
        return np.nan
    if np.any(obs == 0.0):
        warnings.warn("Observation is zero, consider to use another metric than MAPE")
        return np.nan  # TODO is it better to return a large value +inf than NaN?

    return np.mean(np.abs((obs - model) / obs)) * 100


def urmse(
    obs: np.ndarray, model: np.ndarray, weights: Optional[np.ndarray] = None
) -> float:
    r"""Unbiased Root Mean Squared Error (uRMSE)

    $$
    res_i = model_i - obs_i
    $$

    $$
    res_{u,i} = res_i - \overline {res}
    $$

    $$
    uRMSE = \sqrt{\frac{1}{n} \sum_{i=1}^n res_{u,i}^2}
    $$

    Range: $[0, \infty)$; Best: 0

    See Also
    --------
    root_mean_squared_error
    """
    return root_mean_squared_error(obs, model, weights, unbiased=True)


def rmse(
    obs: np.ndarray,
    model: np.ndarray,
    weights: Optional[np.ndarray] = None,
    unbiased: bool = False,
) -> float:
    """alias for root_mean_squared_error"""
    return root_mean_squared_error(obs, model, weights, unbiased)


def root_mean_squared_error(
    obs: np.ndarray,
    model: np.ndarray,
    weights: Optional[np.ndarray] = None,
    unbiased: bool = False,
) -> float:
    r"""Root Mean Squared Error (RMSE)

    $$
    res_i = model_i - obs_i
    $$

    $$
    RMSE=\sqrt{\frac{1}{n} \sum_{i=1}^n res_i^2}
    $$

    Unbiased version:

    $$
    res_{u,i} = res_i - \overline {res}
    $$

    $$
    uRMSE=\sqrt{\frac{1}{n} \sum_{i=1}^n res_{u,i}^2}
    $$

    Range: $[0, \infty)$; Best: 0

    """
    assert obs.size == model.size

    residual = obs - model
    if unbiased:
        residual = residual - residual.mean()
    error = np.sqrt(np.average(residual**2, weights=weights))

    return error


def nse(obs: np.ndarray, model: np.ndarray) -> float:
    """alias for nash_sutcliffe_efficiency"""
    return nash_sutcliffe_efficiency(obs, model)


def nash_sutcliffe_efficiency(obs: np.ndarray, model: np.ndarray) -> float:
    r"""Nash-Sutcliffe Efficiency (NSE)

    $$
    NSE = 1 - \frac {\sum _{i=1}^{n}\left(model_{i} - obs_{i}\right)^{2}}
                    {\sum_{i=1}^{n}\left(obs_{i} - {\overline{obs}}\right)^{2}}
    $$

    Range: $(-\infty, 1]$; Best: 1

    Note
    ----
    r2 = nash_sutcliffe_efficiency(nse)

    References
    ----------
    Nash, J. E.; Sutcliffe, J. V. (1970). "River flow forecasting through conceptual models part I — A discussion of principles". Journal of Hydrology. 10 (3): 282–290. <https://doi.org/10.1016/0022-1694(70)90255-6>
    """
    assert obs.size == model.size

    if len(obs) == 0:
        return np.nan
    error = 1 - (np.sum((obs - model) ** 2) / np.sum((obs - np.mean(obs)) ** 2))

    return error


def kling_gupta_efficiency(obs: np.ndarray, model: np.ndarray) -> float:
    r"""
    Kling-Gupta Efficiency (KGE)

    $$
    KGE = 1 - \sqrt{(r-1)^2 + \left(\frac{\sigma_{mod}}{\sigma_{obs}} - 1\right)^2 +
                                \left(\frac{\mu_{mod}}{\mu_{obs}} - 1\right)^2 }
    $$

    where $r$ is the pearson correlation coefficient, $\mu_{obs},\mu_{mod}$ and $\sigma_{obs},\sigma_{mod}$ is the mean and standard deviation of observations and model.

    Range: $(-\infty, 1]$; Best: 1

    References
    ----------
    Gupta, H. V., Kling, H., Yilmaz, K. K. and Martinez, G. F., (2009), Decomposition of the mean squared error and NSE performance criteria: Implications for improving hydrological modelling, J. Hydrol., 377(1-2), 80-91 <https://doi.org/10.1016/j.jhydrol.2009.08.003>

    Knoben, W. J. M., Freer, J. E., and Woods, R. A. (2019) Technical note: Inherent benchmark or not? Comparing Nash–Sutcliffe and Kling–Gupta efficiency scores, Hydrol. Earth Syst. Sci., 23, 4323-4331 <https://doi.org/10.5194/hess-23-4323-2019>
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
    r"""Coefficient of determination (R2)

    Pronounced 'R-squared'; the proportion of the variation in the dependent variable that is predictable from the independent variable(s), i.e. the proportion of explained variance.

    $$
    R^2 = 1 - \frac{\sum_{i=1}^n (model_i - obs_i)^2}
                    {\sum_{i=1}^n (obs_i - \overline {obs})^2}
    $$

    Range: $(-\infty, 1]$; Best: 1

    Note
    ----
    r2 = nash_sutcliffe_efficiency(nse)

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

    residual = model - obs
    SSr = np.sum(residual**2)
    SSt = np.sum((obs - obs.mean()) ** 2)

    return 1 - SSr / SSt


def mef(obs: np.ndarray, model: np.ndarray) -> float:
    """alias for model_efficiency_factor"""
    return model_efficiency_factor(obs, model)


def model_efficiency_factor(obs: np.ndarray, model: np.ndarray) -> float:
    r"""Model Efficiency Factor (MEF)

    Scale independent RMSE, standardized by Stdev of observations

    $$
    MEF = \frac{RMSE}{STDEV}=\frac{\sqrt{\frac{1}{n} \sum_{i=1}^n(model_i - obs_i)^2}}
                                    {\sqrt{\frac{1}{n} \sum_{i=1}^n(obs_i - \overline{obs})^2}}=\sqrt{1-NSE}
    $$

    Range: $[0, \infty)$; Best: 0

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
    r"""Pearson’s Correlation coefficient (CC)

    $$
    CC = \frac{\sum_{i=1}^n (model_i - \overline{model})(obs_i - \overline{obs}) }
                   {\sqrt{\sum_{i=1}^n (model_i - \overline{model})^2}
                    \sqrt{\sum_{i=1}^n (obs_i - \overline{obs})^2} }
    $$

    Range: [-1, 1]; Best: 1

    See Also
    --------
    spearmanr
    np.corrcoef
    """
    assert obs.size == model.size
    if len(obs) <= 1:
        return np.nan

    if weights is None:
        return np.corrcoef(obs, model)[0, 1]
    else:
        C = np.cov(obs, model, fweights=weights)
        return C[0, 1] / np.sqrt(C[0, 0] * C[1, 1])


def rho(obs: np.ndarray, model: np.ndarray) -> float:
    """alias for spearmanr"""
    return spearmanr(obs, model)


def spearmanr(obs: np.ndarray, model: np.ndarray) -> float:
    r"""Spearman rank correlation coefficient

    The rank correlation coefficient is similar to the Pearson correlation coefficient but
    applied to ranked quantities and is useful to quantify a monotonous relationship

    $$
    \rho = \frac{\sum_{i=1}^n (rmodel_i - \overline{rmodel})(robs_i - \overline{robs}) }
                    {\sqrt{\sum_{i=1}^n (rmodel_i - \overline{rmodel})^2}
                    \sqrt{\sum_{i=1}^n (robs_i - \overline{robs})^2} }
    $$

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
    r"""Scatter index (SI)

    Which is the same as the unbiased-RMSE normalized by the absolute mean of the observations.

    $$
    \frac{ \sqrt{ \frac{1}{n} \sum_{i=1}^n \left( (model_i - \overline {model}) - (obs_i - \overline {obs}) \right)^2} }
    {\frac{1}{n} \sum_{i=1}^n | obs_i | }
    $$

    Range: $[0, \infty)$; Best: 0
    """
    assert obs.size == model.size
    if len(obs) == 0:
        return np.nan

    residual = obs - model
    residual = residual - residual.mean()  # unbiased
    return np.sqrt(np.mean(residual**2)) / np.mean(np.abs(obs))


def scatter_index2(obs: np.ndarray, model: np.ndarray) -> float:
    r"""Alternative formulation of the scatter index (SI)

    $$
    \sqrt {\frac{\sum_{i=1}^n \left( (model_i - \overline {model}) - (obs_i - \overline {obs}) \right)^2}
    {\sum_{i=1}^n obs_i^2}}
    $$

    Range: [0, 100]; Best: 0
    """
    assert obs.size == model.size
    if len(obs) == 0:
        return np.nan

    return np.sqrt(
        np.sum(((model - model.mean()) - (obs - obs.mean())) ** 2) / np.sum(obs**2)
    )


def ev(obs: np.ndarray, model: np.ndarray) -> float:
    """alias for explained_variance"""
    assert obs.size == model.size
    return explained_variance(obs, model)


def explained_variance(obs: np.ndarray, model: np.ndarray) -> float:
    r"""EV: Explained variance

     EV is the explained variance and measures the proportion
     [0 - 1] to which the model accounts for the variation
     (dispersion) of the observations.

     In cases with no bias, EV is equal to r2

    $$
    \frac{ \sum_{i=1}^n (obs_i - \overline{obs})^2 -
    \sum_{i=1}^n \left( (obs_i - \overline{obs}) -
    (model_i - \overline{model}) \right)^2}{\sum_{i=1}^n
    (obs_i - \overline{obs})^2}
    $$

    Range: [0, 1]; Best: 1

    See Also
    --------
    r2
    """

    assert obs.size == model.size
    if len(obs) == 0:
        return np.nan

    nominator = np.sum((obs - obs.mean()) ** 2) - np.sum(
        ((obs - obs.mean()) - (model - model.mean())) ** 2
    )
    denominator = np.sum((obs - obs.mean()) ** 2)

    return nominator / denominator


def pr(
    obs: pd.Series,
    model: np.ndarray,
    inter_event_level: float = 0.7,
    AAP: int = 2,
    inter_event_time="36h",
) -> float:
    """alias for peak_ratio"""
    assert obs.size == model.size
    return peak_ratio(obs, model, inter_event_level, AAP, inter_event_time)


def peak_ratio(
    obs: pd.Series,
    model: np.ndarray,
    inter_event_level: float = 0.7,
    AAP: int = 2,
    inter_event_time="36h",
) -> float:
    r"""Peak Ratio

    PR is the mean of the individual ratios of identified peaks in the
    model / identified peaks in the measurements. PR is calculated only for the joint-events,
    ie, events that ocurr simulateneously within a window +/- 0.5*inter_event_time.

    Parameters
    ----------
    inter_event_level (float, optional)
        Inter-event level threshold (default: 0.7).
    AAP (float, optional)
        Average Annual Peaks (ie, Number of peaks per year, on average). (default: 2)
    inter_event_time (str, optional)
            Maximum time interval between peaks (default: 36 hours).

    $$
    \frac{\sum_{i=1}^{N_{joint-peaks}} (\frac{Peak_{model_i}}{Peak_{obs_i}} )}{N_{joint-peaks}}
    $$

    Range: $[0, \infty)$; Best: 1.0
    """

    assert obs.size == model.size
    if len(obs) == 0:
        return np.nan
    assert isinstance(obs.index, pd.DatetimeIndex)
    time = obs.index

    # Calculate number of years
    dt_int = time[1:].values - time[0:-1].values
    dt_int_mode = float(stats.mode(dt_int, keepdims=False)[0]) / 1e9  # in seconds
    N_years = dt_int_mode / 24 / 3600 / 365.25 * len(time)
    found_peaks = []
    for data in [obs, model]:
        peak_index, AAP_ = _partial_duration_series(
            time,
            data,
            inter_event_level=inter_event_level,
            AAP=AAP,
            inter_event_time=inter_event_time,
        )
        peaks = data[peak_index]
        peaks_sorted = peaks.sort_values(ascending=False)
        found_peaks.append(
            peaks_sorted[0 : max(1, min(round(AAP_ * N_years), np.sum(peaks)))]
        )
    found_peaks_obs = found_peaks[0]
    found_peaks_mod = found_peaks[1]

    # Resample~ish, find peaks spread maximum Half the inter event time (if inter event =36, select data paired +/- 18h) (or inter_event) and then select
    indices_mod = (
        abs(found_peaks_obs.index.values[:, None] - found_peaks_mod.index.values)
        < pd.Timedelta(inter_event_time) / 2
    ).any(axis=0)
    indices_obs = (
        abs(found_peaks_mod.index.values[:, None] - found_peaks_obs.index.values)
        < pd.Timedelta(inter_event_time) / 2
    ).any(axis=0)
    obs_joint = found_peaks_obs.loc[indices_obs]
    mod_joint = found_peaks_mod.loc[indices_mod]

    if len(obs_joint) == 0 or len(mod_joint) == 0:
        return np.nan
    res = np.mean(mod_joint.values / obs_joint.values)
    return res


def willmott(obs: np.ndarray, model: np.ndarray) -> float:
    r"""Willmott's Index of Agreement

    A scaled representation of the predictive accuracy of the model against observations. A value of 1 indicates a perfect match, and 0 indicates no agreement at all.

    $$
    willmott = 1 - \frac{\frac{1}{n} \sum_{i=1}^n(model_i - obs_i)^2}
                        {\frac{1}{n} \sum_{i=1}^n(|model_i - \overline{obs}| + |obs_i - \overline{obs}|)^2}
    $$

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

    residual = model - obs
    nominator = np.sum(residual**2)
    denominator = np.sum((np.abs(model - obs.mean()) + np.abs(obs - obs.mean())) ** 2)

    return 1 - nominator / denominator


def hit_ratio(obs: np.ndarray, model: np.ndarray, a=0.1) -> float:
    r"""Fraction within obs ± acceptable deviation

    $$
    HR = \frac{1}{n}\sum_{i=1}^n I_{|(model_i - obs_i)|} < a
    $$

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

    return np.mean(np.abs(obs - model) < a)


def lin_slope(obs: np.ndarray, model: np.ndarray, reg_method="ols") -> float:
    r"""Slope of the regression line.

    $$
    slope = \frac{\sum_{i=1}^n (model_i - \overline {model})(obs_i - \overline {obs})}
                    {\sum_{i=1}^n (obs_i - \overline {obs})^2}
    $$

    Range: $(-\infty, \infty )$; Best: 1
    """
    assert obs.size == model.size
    return _linear_regression(obs, model, reg_method)[0]


def _linear_regression(
    obs: np.ndarray, model: np.ndarray, reg_method="ols"
) -> Tuple[float, float]:
    if len(obs) == 0:
        return np.nan, np.nan  # TODO raise error?

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


def _partial_duration_series(
    time,
    value,
    *,
    inter_event_time="36h",
    use_inter_event_level=True,
    inter_event_level=0.7,
    AAP=2,
):
    """
    Calculate the partial duration series based on the given time and value arrays.

    Parameters:
        time (array-like)
            Array of time values.
        value (array-like)
            Array of corresponding values.
        inter_event_time (str, optional)
            Maximum time interval between peaks (default: 36 hours).
        use_inter_event_level (bool, optional)
            Flag indicating whether to consider inter-event level (default: True).
        inter_event_level (float, optional)
            Inter-event level threshold (default: 0.7).
        AAP (float, optional)
            Average Annual Peaks (ie, Number of peaks per year, on average). (default: 2)

    Returns:
        tuple: (numpy.ndarray,int)
            - Array of booleans indicating the identified peaks in the partial duration series.
            - Average Annual Peaks per year

    Raises:
        None

    Notes:
        - The time values are expected to be a datetime index
        - The returned array contains True at indices corresponding to peaks in the series.
    """
    peak_list = np.zeros(len(time))

    old_peak = -1
    n = len(time)
    inter_time = pd.Timedelta(inter_event_time) / np.timedelta64(1, "h")
    inter_level = 1.0
    time = np.asarray(time)
    value = np.asarray(value)
    time = (time - time[0]).astype(float) / 1e9 / 3600  # time index in hours from t0=0

    for time_step in range(n):
        if old_peak < 0:
            old_peak = time_step
            continue

        distance = time[time_step] - time[old_peak]
        if distance > inter_time:
            peak_list[old_peak] = 1
            old_peak = time_step
            continue

        peak_val = value[old_peak]
        step_val = value[time_step]
        if peak_val < step_val:
            if time_step != n - 1:
                distance = time[time_step + 1] - time[time_step]
                if time_step != n - 1 and distance < inter_time:
                    if step_val > value[time_step + 1]:
                        old_peak = time_step
                else:
                    old_peak = time_step
            else:
                old_peak = time_step

    if peak_list[old_peak] == 0:
        peak_list[old_peak] = 1

    old_peak = -1
    for time_step in range(n - 1, -1, -1):
        if old_peak < 0:
            old_peak = time_step
            continue

        distance = time[old_peak] - time[time_step]
        if distance > inter_time:
            old_peak = time_step
            continue

        peak_val = value[old_peak]
        step_val = value[time_step]
        if peak_val < step_val:
            if time_step != 0:
                distance = time[time_step] - time[time_step - 1]
                if distance < inter_time:
                    if step_val > value[time_step]:
                        peak_list[old_peak] = 0
                        old_peak = time_step
                else:
                    peak_list[old_peak] = 0
                    old_peak = time_step
            else:
                peak_list[old_peak] = 0
                old_peak = time_step
        elif peak_val == step_val and peak_list[time_step] == 1:
            peak_list[old_peak] = 0
            old_peak = time_step
        else:
            peak_list[time_step] = 0

    peak_list[old_peak] = 1

    if use_inter_event_level:
        inter_level = inter_event_level

    i = 0
    while i < n:
        minimum = 1.0e99
        while i < n and peak_list[i] == 0:
            if value[i] < minimum:
                minimum = value[i]
            i += 1

        if i < n and peak_list[i] == 1:
            x1 = value[old_peak]
            x2 = value[i]
            distance = time[i] - time[old_peak]

            if distance > inter_time and (
                not use_inter_event_level or minimum < inter_level * min(x1, x2)
            ):
                old_peak = i
            else:
                if x1 > x2:
                    peak_list[i] = 0
                else:
                    peak_list[old_peak] = 0
                    old_peak = i
        i += 1
    return peak_list.astype(bool), AAP


## Circular metrics


def _c_residual(obs: np.ndarray, model: np.ndarray) -> np.ndarray:
    """Circular residual (0, 360) - output between -180 and 180"""
    assert obs.size == model.size
    resi = model - obs
    resi = (resi + 180) % 360 - 180
    return resi


def c_bias(obs: np.ndarray, model: np.ndarray) -> float:
    """Circular bias (mean error)

    Parameters
    ----------
    obs : np.ndarray
        Observation in degrees (0, 360)
    model : np.ndarray
        Model in degrees (0, 360)

    Range: [-180., 180.]; Best: 0.

    Returns
    -------
    float
        Circular bias

    Examples
    --------
    >>> obs = np.array([10., 355., 170.])
    >>> mod = np.array([20., 5., -180.])
    >>> c_bias(obs, mod)
    10.0
    """
    from scipy.stats import circmean

    resi = _c_residual(obs, model)
    return circmean(resi, low=-180.0, high=180.0)


def c_max_error(obs: np.ndarray, model: np.ndarray) -> float:
    """Circular max error

    Parameters
    ----------
    obs : np.ndarray
        Observation in degrees (0, 360)
    model : np.ndarray
        Model in degrees (0, 360)

    Range: :math:`[0, \\infty)`; Best: 0

    Returns
    -------
    float
        Circular max error

    Examples
    --------
    >>> obs = np.array([10., 350., 10.])
    >>> mod = np.array([20., 10., 350.])
    >>> c_max_error(obs, mod)
    20.0
    """

    resi = _c_residual(obs, model)

    # Compute the absolute differences and then
    # find the shortest distance between angles
    abs_diffs = np.abs(resi)
    circular_diffs = np.minimum(abs_diffs, 360 - abs_diffs)
    return np.max(circular_diffs)


def c_mean_absolute_error(
    obs: np.ndarray,
    model: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Circular mean absolute error

    Parameters
    ----------
    obs : np.ndarray
        Observation in degrees (0, 360)
    model : np.ndarray
        Model in degrees (0, 360)
    weights : np.ndarray, optional
        Weights, by default None

    Range: [0, 180]; Best: 0

    Returns
    -------
    float
        Circular mean absolute error
    """

    resi = _c_residual(obs, model)
    return np.average(np.abs(resi), weights=weights)


def c_mae(
    obs: np.ndarray,
    model: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """alias for circular mean absolute error"""
    return c_mean_absolute_error(obs, model, weights)


def c_root_mean_squared_error(
    obs: np.ndarray,
    model: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Circular root mean squared error

    Parameters
    ----------
    obs : np.ndarray
        Observation in degrees (0, 360)
    model : np.ndarray
        Model in degrees (0, 360)
    weights : np.ndarray, optional
        Weights, by default None

    Range: [0, 180]; Best: 0

    Returns
    -------
    float
        Circular root mean squared error
    """
    residual = _c_residual(obs, model)
    return np.sqrt(np.average(residual**2, weights=weights))


def c_rmse(
    obs: np.ndarray,
    model: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """alias for circular root mean squared error"""
    return c_root_mean_squared_error(obs, model, weights)


def c_unbiased_root_mean_squared_error(
    obs: np.ndarray,
    model: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Circular unbiased root mean squared error

    Parameters
    ----------
    obs : np.ndarray
        Observation in degrees (0, 360)
    model : np.ndarray
        Model in degrees (0, 360)
    weights : np.ndarray, optional
        Weights, by default None

    Range: [0, 180]; Best: 0

    Returns
    -------
    float
        Circular unbiased root mean squared error
    """
    from scipy.stats import circmean

    residual = _c_residual(obs, model)
    residual = residual - circmean(residual, low=-180.0, high=180.0)
    return np.sqrt(np.average(residual**2, weights=weights))


def c_urmse(
    obs: np.ndarray,
    model: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """alias for circular unbiased root mean squared error"""
    return c_unbiased_root_mean_squared_error(obs, model, weights)


METRICS_WITH_DIMENSION = set(
    [
        "bias",
        "max_error",
        "mae",
        "rmse",
        "urmse",
        "c_bias",
        "c_max_error",
        "c_mae",
        "c_rmse",
        "c_urmse",
    ]
)

default_metrics: List[Callable] = [bias, rmse, urmse, mae, cc, si, r2]
default_circular_metrics: List[Callable] = [c_bias, c_rmse, c_urmse, c_mae]


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
    if hasattr(metric, "__name__"):
        name = metric.__name__
    else:
        name = metric

    if name not in defined_metrics:
        raise ValueError(f"Metric {name} not defined. Choose from {defined_metrics}")

    return name in METRICS_WITH_DIMENSION


NON_METRICS = set(
    [
        "metric_has_units",
        "get_metric",
        "is_valid_metric",
        "add_metric",
        "Callable",
        "Optional",
        "Set",
        "Tuple",
        "List",
        "Iterable",
        "Union",
        "_c_residual",
        "_linear_regression",
        "_partial_duration_series",
    ]
)


def is_valid_metric(metric: Union[str, Callable]) -> bool:
    if hasattr(metric, "__name__"):
        name = metric.__name__
    else:
        name = metric

    return name in defined_metrics


def get_metric(metric: Union[str, Callable]) -> Callable:
    if is_valid_metric(metric):
        if isinstance(metric, str):
            return getattr(sys.modules[__name__], metric)
        else:
            return metric
    else:
        raise ValueError(
            f"Metric {metric} not defined. Choose from {defined_metrics} or use `add_metric` to add a custom metric."
        )


def add_metric(metric: Callable, has_units: bool = False) -> None:
    """Adds a metric to the metric list. Useful for custom metrics.

    Some metrics are dimensionless, others have the same dimension as the observations.

    Parameters
    ----------
    metric : str or callable
        Metric name or function
    has_units : bool
        True if metric has a dimension, False otherwise. Default:False

    Returns
    -------
    None

    Examples
    --------
    >>> add_metric(hit_ratio)
    >>> add_metric(rmse,True)
    """
    defined_metrics.add(metric.__name__)
    if has_units:
        METRICS_WITH_DIMENSION.add(metric.__name__)

    # add the function to the module
    setattr(sys.modules[__name__], metric.__name__, metric)


defined_metrics: Set[str] = (
    set([func for func in dir() if callable(getattr(sys.modules[__name__], func))])
    - NON_METRICS
)


def _parse_metric(
    metric: str | Iterable[str] | Callable | Iterable[Callable] | None,
    *,
    directional: bool = False,
) -> List[Callable]:
    if metric is None:
        if directional:
            return default_circular_metrics
        else:
            # could be a list of str!
            return [get_metric(m) for m in options.metrics.list]

    if isinstance(metric, str):
        metrics: list = [metric]
    elif callable(metric):
        metrics = [metric]
    elif isinstance(metric, Iterable):
        metrics = list(metric)

    parsed_metrics = []

    for m in metrics:
        if isinstance(m, str):
            parsed_metrics.append(get_metric(m))
        elif callable(m):
            if len(inspect.signature(m).parameters) < 2:
                raise ValueError(
                    "Metrics must have at least two arguments (obs, model)"
                )
            parsed_metrics.append(m)
        else:
            raise TypeError(f"metric {m} must be a string or callable")

    return parsed_metrics


# TODO add non-metric functions to __all__
__all__ = [str(m) for m in defined_metrics]
