"""Standardized residual diagnostics, higher-order moments, and trend drift analysis."""

import warnings
from dataclasses import asdict, dataclass
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from tropicor.core.alignment import align_common_period


@dataclass(frozen=True)
class DistributionMoments:
    """Higher-order statistical moments and normality diagnostics of residuals.

    Attributes:
        n_samples: Number of valid synchronized observations.
        mean: Sample mean of residuals.
        variance: Sample variance (ddof=1).
        std_dev: Sample standard deviation (ddof=1).
        skewness: Fisher-Pearson sample skewness.
        excess_kurtosis: Fisher sample excess kurtosis (normal distribution = 0.0).
        shapiro_stat: Shapiro-Wilk W test statistic.
        shapiro_p_value: Two-sided p-value from Shapiro-Wilk test.
        shapiro_reject_h0_alpha_05: Flag indicating p < 0.05.
        variable: Evaluated variable ('temperature' or 'precipitation').
        notes: Scientific context regarding physical boundedness and extreme
            value properties.
    """

    n_samples: int
    mean: float
    variance: float
    std_dev: float
    skewness: float
    excess_kurtosis: float
    shapiro_stat: float
    shapiro_p_value: float
    shapiro_reject_h0_alpha_05: bool
    variable: Literal["temperature", "precipitation"]
    notes: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert moments to a Python dictionary."""
        return asdict(self)

    def to_series(self) -> pd.Series:
        """Convert moments to a pandas Series for export."""
        return pd.Series(self.to_dict(), name="distribution_moments")


@dataclass(frozen=True)
class TrendDiagnostics:
    """Non-parametric monotonic trend diagnostics on deseasonalized anomalies.

    Attributes:
        n_samples: Number of timesteps evaluated.
        slope_per_decade: Sen's slope scaled to units per decade (using decimal years).
        slope_per_year: Sen's slope in units per decimal year.
        sen_intercept: Theil-Sen intercept.
        mk_stat_s: Mann-Kendall S statistic.
        mk_z_stat: Standardized Z statistic with autocorrelation correction.
        p_value: Two-sided p-value for monotonic trend significance.
        autocorr_correction: Correction applied ('hamed_rao', 'yue_wang', or 'none').
        max_lags: Maximum number of lags inspected.
        lag1_autocorr: Sample lag-1 autocorrelation of detrended residuals.
        significant_lags_used: Number of serial autocorrelation lags adjusted.
        alpha: Nominal significance threshold (default: 0.05).
        has_significant_trend: True if p_value < alpha.
    """

    n_samples: int
    slope_per_decade: float
    slope_per_year: float
    sen_intercept: float
    mk_stat_s: float
    mk_z_stat: float
    p_value: float
    autocorr_correction: Literal["hamed_rao", "yue_wang", "none"]
    max_lags: Optional[int]
    lag1_autocorr: float
    significant_lags_used: int
    alpha: float
    has_significant_trend: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert trend metrics to a Python dictionary."""
        return asdict(self)

    def to_series(self) -> pd.Series:
        """Convert trend metrics to a pandas Series for export."""
        return pd.Series(self.to_dict(), name="trend_diagnostics")


@dataclass(frozen=True)
class ResidualDiagnosticsReport:
    """Comprehensive diagnostic container for model-observation residual analysis.

    Attributes:
        moments: Higher-order distribution moments and normality test.
        trend: Monotonic trend and thermal/precipitation drift diagnostics.
        raw_residuals: Time series of raw residuals e = modeled - observed.
        monthly_climatology_residuals: 12-month mean residual cycle (months 1..12).
        deseasonalized_residuals: Residual anomalies e' = e - mean_e[month].
        standardized_residuals: Z-scores of deseasonalized anomalies.
        variable: Physical variable evaluated ('temperature' or 'precipitation').
    """

    moments: DistributionMoments
    trend: TrendDiagnostics
    raw_residuals: pd.Series
    monthly_climatology_residuals: pd.Series
    deseasonalized_residuals: pd.Series
    standardized_residuals: pd.Series
    variable: Literal["temperature", "precipitation"]

    def summary(self) -> pd.DataFrame:
        """Tabular summary of key diagnostic effect sizes and trend indicators."""
        data = {
            "mean_residual": self.moments.mean,
            "std_residual": self.moments.std_dev,
            "skewness": self.moments.skewness,
            "excess_kurtosis": self.moments.excess_kurtosis,
            "shapiro_p_value": self.moments.shapiro_p_value,
            "slope_per_decade": self.trend.slope_per_decade,
            "mk_p_value": self.trend.p_value,
            "lag1_autocorr": self.trend.lag1_autocorr,
            "has_significant_trend": self.trend.has_significant_trend,
        }
        return pd.DataFrame(data, index=[self.variable]).T

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary representation."""
        return {
            "moments": self.moments.to_dict(),
            "trend": self.trend.to_dict(),
            "variable": self.variable,
        }


def compute_deseasonalized_residuals(
    observed: pd.Series,
    modeled: pd.Series,
    min_years: int = 10,
    min_obs_per_month: int = 5,
    monthly_standardize: bool = False,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Align series on common period and extract raw error, monthly climatological mean,
    and deseasonalized anomalies.

    Args:
        observed: In-situ station observation time series with DatetimeIndex.
        modeled: Reanalysis or model time series with DatetimeIndex.
        min_years: Minimum distinct years of synchronous data required (default: 10).
        min_obs_per_month: Minimum valid observations required per calendar
            month (default: 5).
        monthly_standardize: If True, divides each month's anomaly by that month's
            residual standard deviation, mitigating precipitation heteroscedasticity.

    Returns:
        Tuple (raw_residuals, monthly_climatology_residuals, deseasonalized_residuals).

    Raises:
        ValueError: If sample size guards (min_years, min_obs_per_month) are not met.
    """
    obs_sync, mod_sync = align_common_period(observed, modeled, drop_na=True)
    if not isinstance(obs_sync.index, pd.DatetimeIndex):
        raise ValueError("Aligned series index must be a DatetimeIndex.")
    dt_index = obs_sync.index

    unique_years = len(np.unique(dt_index.year))
    if unique_years < min_years:
        raise ValueError(
            f"Insufficient synchronous years: found {unique_years}, "
            f"minimum required is {min_years}."
        )

    # Check monthly observation counts
    month_counts = obs_sync.groupby(dt_index.month).size()
    for m in range(1, 13):
        count_m = int(month_counts.get(m, 0))
        if count_m < min_obs_per_month:
            raise ValueError(
                f"Insufficient synchronous observations for month {m}: "
                f"found {count_m}, minimum required is {min_obs_per_month}."
            )

    # Raw residuals: model - observed
    raw_residuals = mod_sync - obs_sync
    raw_residuals.name = "raw_residuals"

    # Monthly climatological mean error
    monthly_clim = raw_residuals.groupby(dt_index.month).mean()
    monthly_clim.name = "monthly_clim_residuals"

    # Subtract monthly mean error
    clim_dict = monthly_clim.to_dict()
    mean_vals = pd.Series(dt_index.month.map(clim_dict), index=raw_residuals.index)
    deseasonalized = raw_residuals - mean_vals

    if monthly_standardize:
        monthly_std = raw_residuals.groupby(dt_index.month).std(ddof=1)
        std_dict = monthly_std.to_dict()
        std_vals = pd.Series(dt_index.month.map(std_dict), index=raw_residuals.index)
        std_vals = std_vals.mask(std_vals < 1e-12, np.nan)
        deseasonalized = deseasonalized / std_vals

    deseasonalized.name = "deseasonalized_residuals"
    return raw_residuals, monthly_clim, deseasonalized


def standardize_residuals(residuals: pd.Series) -> pd.Series:
    """Standardize residual series to zero mean and unit variance: Z = (x - mean) / std.

    Args:
        residuals: Time series of residuals or deseasonalized anomalies.

    Returns:
        Standardized pandas Series with identical DatetimeIndex.
    """
    valid = residuals.dropna()
    std_val = float(valid.std(ddof=1))
    if std_val < 1e-12:
        z = residuals - valid.mean()
    else:
        z = (residuals - valid.mean()) / std_val
    z.name = "standardized_residuals"
    return z


def compute_distribution_moments(
    residuals: pd.Series,
    variable: Literal["temperature", "precipitation"] = "temperature",
    alpha: float = 0.05,
    bias: bool = False,
) -> DistributionMoments:
    """Compute sample skewness, excess kurtosis, variance, and Shapiro-Wilk test.

    Args:
        residuals: Deseasonalized residual series with DatetimeIndex.
        variable: 'temperature' or 'precipitation'.
        alpha: Significance level for the Shapiro-Wilk reject flag (default: 0.05).
        bias: If False (default), calculates unbiased sample skewness and kurtosis.
            If True, uses biased moment ratios (matches scipy.stats.describe default).

    Returns:
        DistributionMoments dataclass.
    """
    vals = residuals.dropna().to_numpy(dtype=float)
    n_samples = len(vals)

    if n_samples < 3:
        raise ValueError(
            f"Insufficient samples for moment estimation: found {n_samples}, "
            "minimum is 3."
        )

    mean_val = float(np.mean(vals))
    var_val = float(np.var(vals, ddof=1)) if n_samples > 1 else 0.0
    std_val = float(np.sqrt(var_val))

    skew_val = float(stats.skew(vals, bias=bias))
    kurt_val = float(stats.kurtosis(vals, fisher=True, bias=bias))

    shapiro_stat, shapiro_p = stats.shapiro(vals)
    reject_h0 = bool(shapiro_p < alpha)

    if variable == "precipitation":
        notes = (
            "Precipitation residuals are bounded on the left by -observed and "
            "intermittent. Non-normality and Shapiro-Wilk rejection are expected from "
            "physical boundary conditions and do not constitute evidence of "
            "orographic downscaling failure."
        )
    else:
        notes = (
            "Temperature extreme residuals (TMin, TMax) follow extreme-value "
            "distributions (GEV/Gumbel) and are inherently skewed. Non-normality "
            "may indicate unresolved microclimates (thermal inversions, cold-air "
            "drainage, complex relief), but alternative drivers include extreme "
            "value distribution physics, sensor replacement, missing observation "
            "days, or ENSO interannual extremes."
        )

    return DistributionMoments(
        n_samples=n_samples,
        mean=mean_val,
        variance=var_val,
        std_dev=std_val,
        skewness=skew_val,
        excess_kurtosis=kurt_val,
        shapiro_stat=float(shapiro_stat),
        shapiro_p_value=float(shapiro_p),
        shapiro_reject_h0_alpha_05=reject_h0,
        variable=variable,
        notes=notes,
    )


def compute_mann_kendall_trend(
    series: pd.Series,
    correction: Literal["hamed_rao", "yue_wang", "none"] = "hamed_rao",
    max_lags: Optional[int] = 3,
    alpha: float = 0.05,
) -> TrendDiagnostics:
    """Calculate real-time Sen's slope (units/decade) and Mann-Kendall trend with
    autocorrelation correction.

    Args:
        series: Deseasonalized anomaly series with regular monthly DatetimeIndex.
        correction: Correction method ('hamed_rao', 'yue_wang', or 'none').
        max_lags: Maximum number of lags to include in variance correction.
            Default is 3 (empirically calibrated for monthly climate series).
            If None, evaluates all lags.
        alpha: Nominal test significance level (default: 0.05).

    Returns:
        TrendDiagnostics dataclass.

    Warns:
        UserWarning: If estimated lag-1 autocorrelation exceeds 0.70.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Series must have a pandas DatetimeIndex.")

    # 1. Decimal years for gap-invariant slope calculation
    t_years = series.index.year + (series.index.month - 0.5) / 12.0
    valid_mask = series.notna()
    y = series[valid_mask].to_numpy(dtype=float)
    t = t_years[valid_mask].to_numpy(dtype=float)
    n = len(y)

    if n < 3:
        raise ValueError(
            f"Insufficient valid timesteps for trend test: found {n}, minimum is 3."
        )

    # 2. Sen's slope via scipy theilslopes
    res_slope = stats.theilslopes(y, t)
    slope_annual = float(res_slope.slope)
    slope_decade = slope_annual * 10.0
    intercept = float(res_slope.intercept)

    # 3. Mann-Kendall score S: sum sgn(y_j - y_i) for j > i
    diffs = np.sign(y[None, :] - y[:, None])
    s = float(np.sum(np.triu(diffs, 1)))

    # 4. Theoretical variance under H0 with ties
    unique, counts = np.unique(y, return_counts=True)
    ties = counts[counts > 1]
    tie_term = np.sum(ties * (ties - 1) * (2 * ties + 5))
    var_s0 = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0

    # 5. Detrended residuals on calendar grid
    detrended_valid = y - slope_annual * t
    y_ranks = stats.rankdata(detrended_valid)

    # Sample lag-1 autocorrelation of detrended residuals on calendar grid
    detrended_series = pd.Series(detrended_valid, index=series[valid_mask].index)
    freq = getattr(series.index, "inferred_freq", None) or "MS"
    try:
        detrended_calendar = detrended_series.asfreq(freq)
        lag1_ac = detrended_calendar.autocorr(lag=1)
    except Exception:
        lag1_ac = detrended_series.autocorr(lag=1)
    lag1_autocorr = float(lag1_ac) if pd.notna(lag1_ac) else 0.0

    if lag1_autocorr > 0.70:
        warnings.warn(
            f"Lag-1 autocorrelation ({lag1_autocorr:.2f}) exceeds 0.7: "
            "corrected p-value may still be liberal (calibration sweep: "
            "~20% false positives at phi=0.85).",
            UserWarning,
            stacklevel=2,
        )

    # Autocorrelation of ranks for Hamed-Rao variance inflation
    r_mean = y_ranks.mean()
    r_centered = y_ranks - r_mean
    d = n * np.ones(2 * n - 1)
    acov = (np.correlate(r_centered, r_centered, "full") / d)[n - 1 :]
    if acov[0] < 1e-12:
        acf_ranks = np.zeros(n)
    else:
        acf_ranks = acov / acov[0]

    # 6. Variance adjustment
    lags_used = 0
    if correction == "none":
        var_s = var_s0
    elif correction == "hamed_rao":
        z_crit = stats.norm.ppf(1 - alpha / 2) / np.sqrt(n)
        upper_bound = z_crit
        lower_bound = -z_crit
        limit = n if max_lags is None else min(n, max_lags + 1)

        sni = 0.0
        for i in range(1, limit):
            if acf_ranks[i] <= upper_bound and acf_ranks[i] >= lower_bound:
                continue
            sni += (n - i) * (n - i - 1) * (n - i - 2) * acf_ranks[i]
            lags_used += 1

        n_ns = 1.0 + (2.0 / (n * (n - 1) * (n - 2))) * sni
        var_s = var_s0 * max(1.0, n_ns)
    elif correction == "yue_wang":
        limit = n if max_lags is None else min(n, max_lags + 1)
        sni = 0.0
        for i in range(1, limit):
            sni += (1.0 - i / n) * acf_ranks[i]
            lags_used += 1
        n_ns = 1.0 + 2.0 * sni
        var_s = var_s0 * max(1.0, n_ns)
    else:
        raise ValueError(f"Unknown correction method '{correction}'.")

    # 7. Standardized test statistic Z and p-value
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s == 0:
        z = 0.0
    else:
        z = (s + 1) / np.sqrt(var_s)

    p_val = float(2.0 * stats.norm.sf(abs(z)))
    has_trend = bool(p_val < alpha)

    return TrendDiagnostics(
        n_samples=n,
        slope_per_decade=slope_decade,
        slope_per_year=slope_annual,
        sen_intercept=intercept,
        mk_stat_s=s,
        mk_z_stat=float(z),
        p_value=p_val,
        autocorr_correction=correction,
        max_lags=max_lags,
        lag1_autocorr=lag1_autocorr,
        significant_lags_used=lags_used,
        alpha=alpha,
        has_significant_trend=has_trend,
    )


def compute_residual_diagnostics(
    observed: pd.Series,
    modeled: pd.Series,
    variable: Literal["temperature", "precipitation"] = "temperature",
    trend_correction: Literal["hamed_rao", "yue_wang", "none"] = "hamed_rao",
    max_lags: Optional[int] = 3,
    min_years: int = 10,
    min_obs_per_month: int = 5,
    monthly_standardize: bool = False,
    alpha: float = 0.05,
) -> ResidualDiagnosticsReport:
    """End-to-end diagnostic pipeline: temporal alignment, deseasonalization,
    higher-order moments, effect sizes, normality check, and drift analysis.

    Args:
        observed: Station observation series with DatetimeIndex.
        modeled: Reanalysis series with DatetimeIndex.
        variable: 'temperature' or 'precipitation'.
        trend_correction: 'hamed_rao', 'yue_wang', or 'none'.
        max_lags: Maximum autocorrelation lags to include (default: 3).
        min_years: Minimum years of synchronous observations (default: 10).
        min_obs_per_month: Minimum observations per calendar month (default: 5).
        monthly_standardize: If True, standardizes monthly error dispersion.
        alpha: Nominal significance level (default: 0.05).

    Returns:
        ResidualDiagnosticsReport containing moments, trend drift, and series.
    """
    raw_e, clim_e, deseased_e = compute_deseasonalized_residuals(
        observed,
        modeled,
        min_years=min_years,
        min_obs_per_month=min_obs_per_month,
        monthly_standardize=monthly_standardize,
    )

    moments = compute_distribution_moments(
        deseased_e,
        variable=variable,
        alpha=alpha,
        bias=False,
    )

    trend = compute_mann_kendall_trend(
        deseased_e,
        correction=trend_correction,
        max_lags=max_lags,
        alpha=alpha,
    )

    standardized_e = standardize_residuals(deseased_e)

    return ResidualDiagnosticsReport(
        moments=moments,
        trend=trend,
        raw_residuals=raw_e,
        monthly_climatology_residuals=clim_e,
        deseasonalized_residuals=deseased_e,
        standardized_residuals=standardized_e,
        variable=variable,
    )
