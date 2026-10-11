"""Unit tests for residual diagnostics, higher-order moments, and trend drift."""

import warnings

import numpy as np
import pandas as pd
import pymannkendall as pmk
import pytest
from scipy import stats

from tropicor.core.diagnostics import (
    DistributionMoments,
    ResidualDiagnosticsReport,
    TrendDiagnostics,
    compute_deseasonalized_residuals,
    compute_distribution_moments,
    compute_mann_kendall_trend,
    compute_residual_diagnostics,
    standardize_residuals,
)


def _make_monthly_series(
    values: np.ndarray,
    start: str = "1990-01-01",
    freq: str = "MS",
    name: str = "series",
) -> pd.Series:
    """Helper to build a pandas Series with DatetimeIndex."""
    idx = pd.date_range(start, periods=len(values), freq=freq)
    return pd.Series(values, index=idx, name=name, dtype=float)


# -----------------------------------------------------------------------------
# 1. Distribution Moments Tests
# -----------------------------------------------------------------------------


def test_distribution_moments_gaussian_known_values() -> None:
    """Verifies moments and Shapiro-Wilk for standard Gaussian series."""
    rng = np.random.default_rng(42)
    vals = rng.normal(loc=0.0, scale=1.0, size=5000)
    s = _make_monthly_series(vals)

    moments = compute_distribution_moments(s, variable="temperature", alpha=0.05)

    assert isinstance(moments, DistributionMoments)
    assert moments.n_samples == 5000
    assert moments.mean == pytest.approx(0.0, abs=0.05)
    assert moments.std_dev == pytest.approx(1.0, abs=0.05)
    assert abs(moments.skewness) < 0.1
    assert abs(moments.excess_kurtosis) < 0.1
    assert moments.shapiro_stat > 0.98
    assert moments.shapiro_p_value > 0.05
    assert moments.shapiro_reject_h0_alpha_05 is False
    assert moments.variable == "temperature"
    assert "extreme value" in moments.notes.lower()


def test_distribution_moments_skewed_gamma() -> None:
    """Verifies that skewed distribution exhibits positive skewness
    and excess kurtosis.
    """
    rng = np.random.default_rng(42)
    vals = rng.gamma(shape=2.0, scale=2.0, size=1000)
    s = _make_monthly_series(vals)

    moments = compute_distribution_moments(s, variable="precipitation")

    assert moments.skewness > 0.8
    assert moments.excess_kurtosis > 0.5
    assert moments.shapiro_p_value < 0.001
    assert moments.shapiro_reject_h0_alpha_05 is True
    assert moments.variable == "precipitation"
    assert (
        "not constitute evidence of orographic downscaling failure"
        in moments.notes.lower()
    )


def test_distribution_moments_bias_flag() -> None:
    """Verifies that bias=True exactly reproduces scipy.stats.describe default."""
    rng = np.random.default_rng(42)
    vals = rng.normal(loc=2.0, scale=3.0, size=120)
    s = _make_monthly_series(vals)

    moments_biased = compute_distribution_moments(s, bias=True)
    scipy_desc = stats.describe(vals, bias=True)

    assert moments_biased.skewness == pytest.approx(scipy_desc.skewness, rel=1e-7)
    assert moments_biased.excess_kurtosis == pytest.approx(
        scipy_desc.kurtosis, rel=1e-7
    )


def test_standardize_residuals() -> None:
    """Verifies that standardize_residuals scales to mean 0 and std 1."""
    vals = np.array([10.0, 12.0, 14.0, 16.0, 18.0])
    s = _make_monthly_series(vals)

    z = standardize_residuals(s)

    assert z.mean() == pytest.approx(0.0, abs=1e-12)
    assert z.std(ddof=1) == pytest.approx(1.0, abs=1e-12)
    assert z.index.equals(s.index)


# -----------------------------------------------------------------------------
# 2. Deseasonalization & Climatology Guards
# -----------------------------------------------------------------------------


def test_deseasonalized_residuals_removes_annual_harmonic() -> None:
    """Verifies that subtracting monthly climatological mean removes seasonal cycle."""
    n_months = 120  # 10 full years
    dates = pd.date_range("2000-01-01", periods=n_months, freq="MS")
    obs = pd.Series(20.0, index=dates, name="obs")
    months = dates.month.to_numpy()
    seasonal_bias = 5.0 * np.sin(2.0 * np.pi * months / 12.0)
    mod = pd.Series(20.0 + seasonal_bias, index=dates, name="mod")

    raw_e, clim_e, deseased_e = compute_deseasonalized_residuals(
        obs, mod, min_years=10, min_obs_per_month=5
    )

    assert len(clim_e) == 12
    for m in range(1, 13):
        expected_m = 5.0 * np.sin(2.0 * np.pi * m / 12.0)
        assert clim_e[m] == pytest.approx(expected_m, abs=1e-10)

    assert np.allclose(deseased_e.to_numpy(), 0.0, atol=1e-10)


def test_heteroscedasticity_monthly_standardize() -> None:
    """Verifies that monthly_standardize=True homogenizes per-month error variance."""
    rng = np.random.default_rng(42)
    n_months = 120  # 10 years
    dates = pd.date_range("2000-01-01", periods=n_months, freq="MS")
    obs = pd.Series(50.0, index=dates)

    errs = np.zeros(n_months)
    for idx, d in enumerate(dates):
        month_scale = float(d.month)
        errs[idx] = rng.normal(0.0, month_scale)

    mod = obs + errs

    _, _, deseased_std = compute_deseasonalized_residuals(
        obs, mod, min_years=10, min_obs_per_month=5, monthly_standardize=True
    )

    for m in range(1, 13):
        m_vals = deseased_std[deseased_std.index.month == m]
        assert m_vals.std(ddof=1) == pytest.approx(1.0, abs=0.20)


def test_climatology_sample_size_guards() -> None:
    """Raises ValueError if synchronous overlap has fewer than min_years or min_obs."""
    short_dates = pd.date_range("2000-01-01", periods=48, freq="MS")
    obs = pd.Series(20.0, index=short_dates)
    mod = pd.Series(21.0, index=short_dates)

    with pytest.raises(ValueError, match="Insufficient synchronous years"):
        compute_deseasonalized_residuals(obs, mod, min_years=10, min_obs_per_month=5)


# -----------------------------------------------------------------------------
# 3. Real-Time Sen's Slope & Gap Invariance
# -----------------------------------------------------------------------------


def test_real_time_sens_slope_and_gap_invariance() -> None:
    """Verifies that calculating Sen's slope with decimal years is robust
    to missing months.
    """
    rng = np.random.default_rng(42)
    n_months = 360  # 30 years
    dates = pd.date_range("1990-01-01", periods=n_months, freq="MS")

    t_years = dates.year + (dates.month - 0.5) / 12.0
    true_slope_annual = 0.05
    y_full = true_slope_annual * (t_years - t_years[0]) + rng.normal(0.0, 0.1, n_months)
    s_full = pd.Series(y_full, index=dates)

    trend_full = compute_mann_kendall_trend(s_full, correction="hamed_rao", max_lags=3)
    assert trend_full.slope_per_year == pytest.approx(true_slope_annual, rel=0.08)
    assert trend_full.slope_per_decade == pytest.approx(0.50, rel=0.08)

    drop_mask = rng.uniform(0, 1, n_months) < 0.20
    s_gaps = s_full[~drop_mask]

    trend_gaps = compute_mann_kendall_trend(s_gaps, correction="hamed_rao", max_lags=3)
    assert trend_gaps.slope_per_year == pytest.approx(
        trend_full.slope_per_year, rel=0.05
    )
    assert trend_gaps.slope_per_decade == pytest.approx(
        trend_full.slope_per_decade, rel=0.05
    )


# -----------------------------------------------------------------------------
# 4. Monte Carlo AR(1) Calibration & High Autocorrelation Warning
# -----------------------------------------------------------------------------


def test_autocorrelation_ar1_monte_carlo_calibration() -> None:
    """Monte Carlo test verifying that Hamed-Rao (max_lags=3) moderates
    AR(1) false positives.
    """
    rng = np.random.default_rng(42)
    n_sim = 200
    n_len = 360
    phi = 0.65

    dates = pd.date_range("1990-01-01", periods=n_len, freq="MS")
    uncorr_rejections = 0
    hr3_rejections = 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        for _ in range(n_sim):
            e = np.zeros(n_len)
            innov = rng.normal(0, 1, n_len)
            for t in range(1, n_len):
                e[t] = phi * e[t - 1] + innov[t]

            s = pd.Series(e, index=dates)
            trend_u = compute_mann_kendall_trend(s, correction="none")
            if trend_u.p_value < 0.05:
                uncorr_rejections += 1

            trend_hr3 = compute_mann_kendall_trend(
                s, correction="hamed_rao", max_lags=3
            )
            if trend_hr3.p_value < 0.05:
                hr3_rejections += 1

    uncorr_rate = uncorr_rejections / n_sim
    hr3_rate = hr3_rejections / n_sim

    assert uncorr_rate >= 0.30
    assert hr3_rate <= 0.12
    assert hr3_rate < uncorr_rate - 0.20


def test_high_autocorrelation_warning() -> None:
    """Verifies that lag1_autocorr > 0.7 triggers a UserWarning."""
    rng = np.random.default_rng(42)
    n_len = 240
    dates = pd.date_range("1990-01-01", periods=n_len, freq="MS")

    e = np.zeros(n_len)
    for t in range(1, n_len):
        e[t] = 0.85 * e[t - 1] + rng.normal(0, 1)

    s = pd.Series(e, index=dates)

    with pytest.warns(UserWarning, match=r"Lag-1 autocorrelation.*exceeds 0\.7"):
        trend = compute_mann_kendall_trend(s, correction="hamed_rao", max_lags=3)

    assert isinstance(trend, TrendDiagnostics)
    assert trend.lag1_autocorr > 0.70


# -----------------------------------------------------------------------------
# 5. Parity Test Against pymannkendall (Complete Series Only)
# -----------------------------------------------------------------------------


def test_pymannkendall_parity_on_complete_series() -> None:
    """Asserts exact numerical parity against pymannkendall on complete series."""
    rng = np.random.default_rng(123)
    n_len = 120
    dates = pd.date_range("2000-01-01", periods=n_len, freq="MS")
    vals = rng.normal(20.0, 3.0, n_len) + 0.02 * np.arange(n_len)
    s = pd.Series(vals, index=dates)

    native_res = compute_mann_kendall_trend(s, correction="hamed_rao", max_lags=3)
    pmk_res = pmk.hamed_rao_modification_test(vals, lag=3)

    assert native_res.mk_stat_s == pmk_res.s
    assert native_res.slope_per_year == pytest.approx(pmk_res.slope * 12.0, rel=1e-5)
    assert native_res.slope_per_decade == pytest.approx(pmk_res.slope * 120.0, rel=1e-5)
    assert native_res.mk_z_stat == pytest.approx(pmk_res.z, abs=1e-4)
    assert native_res.p_value == pytest.approx(pmk_res.p, abs=1e-4)


# -----------------------------------------------------------------------------
# 6. End-to-End Orchestrator & Report Container
# -----------------------------------------------------------------------------


def test_compute_residual_diagnostics_end_to_end() -> None:
    """Verifies complete report generation and tabular summary formatting."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("1990-01-01", periods=360, freq="MS")
    obs = pd.Series(25.0 + np.sin(np.arange(360)), index=dates, name="obs")
    mod = pd.Series(
        23.0 + np.sin(np.arange(360)) + rng.normal(0.0, 0.5, 360),
        index=dates,
        name="mod",
    )

    report = compute_residual_diagnostics(
        obs,
        mod,
        variable="temperature",
        trend_correction="hamed_rao",
        max_lags=3,
        min_years=10,
        min_obs_per_month=5,
    )

    assert isinstance(report, ResidualDiagnosticsReport)
    assert isinstance(report.moments, DistributionMoments)
    assert isinstance(report.trend, TrendDiagnostics)
    assert report.variable == "temperature"
    assert len(report.raw_residuals) == 360
    assert len(report.monthly_climatology_residuals) == 12
    assert len(report.deseasonalized_residuals) == 360
    assert len(report.standardized_residuals) == 360

    df_summary = report.summary()
    assert isinstance(df_summary, pd.DataFrame)
    assert "skewness" in df_summary.index
    assert "excess_kurtosis" in df_summary.index
    assert "slope_per_decade" in df_summary.index
    assert "lag1_autocorr" in df_summary.index
