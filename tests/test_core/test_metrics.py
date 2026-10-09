"""Unit tests for statistical verification metrics and ValidationReport."""

import math
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from tropicor.core.metrics import (
    TaylorStatistics,
    ValidationReport,
    compute_validation_metrics,
    taylor_statistics,
)


def test_metrics_identical_series(
    synthetic_paired_series: Tuple[pd.Series, pd.Series],
) -> None:
    """Test validation metrics on two identical series."""
    observed, _ = synthetic_paired_series
    report = compute_validation_metrics(observed, observed)

    assert isinstance(report, ValidationReport)
    assert report.n_samples == len(observed)
    assert math.isclose(report.pearson_r, 1.0, rel_tol=1e-5)
    assert math.isclose(report.p_value, 0.0, abs_tol=1e-5)
    assert math.isclose(report.rmse, 0.0, abs_tol=1e-5)
    assert math.isclose(report.mae, 0.0, abs_tol=1e-5)
    assert math.isclose(report.bias, 0.0, abs_tol=1e-5)
    assert math.isclose(report.pbias, 0.0, abs_tol=1e-5)
    assert math.isclose(report.kge, 1.0, rel_tol=1e-5)
    assert math.isclose(report.euclidean_distance, 0.0, abs_tol=1e-5)


def test_metrics_scaled_series() -> None:
    """Test scaled series: r remains 1.0, but Euclidean distance detects scale."""
    idx = pd.date_range("1990-01-01", periods=100, freq="MS")
    y_obs = pd.Series(np.linspace(10.0, 50.0, 100), index=idx)
    y_mod = y_obs * 2.0  # Double scale

    report = compute_validation_metrics(y_obs, y_mod)

    # Correlation is scale-invariant (r=1.0)
    assert math.isclose(report.pearson_r, 1.0, rel_tol=1e-5)
    # Scale discrepancy is captured by RMSE and Euclidean distance
    assert report.rmse > 0.0
    assert report.euclidean_distance > 0.0
    assert math.isclose(report.pbias, 100.0, rel_tol=1e-5)  # 100% overestimation
    assert report.kge < 1.0


def test_metrics_andean_bias_case(
    synthetic_paired_series: Tuple[pd.Series, pd.Series],
) -> None:
    """Test simulating Andean elevation cold bias (Bucaramanga UIS scenario)."""
    observed, modeled = synthetic_paired_series
    report = compute_validation_metrics(observed, modeled)

    # Phase agreement remains high
    assert report.pearson_r > 0.95
    # Severe cold bias (~ -8°C) is captured
    assert math.isclose(report.bias, -8.0, abs_tol=0.5)
    assert math.isclose(report.rmse, 8.0, abs_tol=0.5)
    assert report.pbias < -25.0


def test_metrics_near_zero_observed_mean() -> None:
    """Test KGE and PBIAS epsilon guards when observed mean is near zero."""
    idx = pd.date_range("1990-01-01", periods=20, freq="MS")
    # Observed precipitation is essentially zero (mean < 1e-6)
    y_obs = pd.Series(np.zeros(20), index=idx)
    y_mod = pd.Series(np.full(20, 5.0), index=idx)

    warn_match = "KGE is undefined when observed mean is near zero"
    with pytest.warns(UserWarning, match=warn_match):
        report = compute_validation_metrics(y_obs, y_mod)

    # KGE is nan, but report does not crash
    assert np.isnan(report.kge)
    # Other metrics compute cleanly
    assert math.isclose(report.rmse, 5.0, abs_tol=1e-5)
    assert math.isclose(report.mae, 5.0, abs_tol=1e-5)
    assert math.isclose(report.bias, 5.0, abs_tol=1e-5)
    assert np.isnan(report.pbias)  # Also guarded against sum_obs=0


def test_metrics_zero_variance() -> None:
    """Test safe handling when observed series has zero variance (constant values)."""
    idx = pd.date_range("1990-01-01", periods=10, freq="MS")
    y_obs = pd.Series(np.full(10, 15.0), index=idx)  # Mean is 15, but variance is 0
    y_mod = pd.Series(np.linspace(10.0, 20.0, 10), index=idx)

    with pytest.warns(UserWarning, match="zero-variance"):
        report = compute_validation_metrics(y_obs, y_mod)

    assert report.pearson_r == 0.0
    assert np.isnan(report.kge)
    assert report.rmse > 0.0


def test_metrics_nan_synchronization() -> None:
    """Test that mismatched NaNs are dropped synchronously."""
    idx = pd.date_range("1990-01-01", periods=10, freq="MS")
    s1 = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, np.nan, 8.0, 9.0, 10.0], index=idx)
    s2 = pd.Series([1.1, np.nan, 3.0, 4.1, 5.0, np.nan, 7.0, 8.1, 9.0, 10.0], index=idx)

    # Valid synchronous points: indices 0, 3, 4, 7, 8, 9 (6 points)
    report = compute_validation_metrics(s1, s2)
    assert report.n_samples == 6


def test_metrics_insufficient_samples() -> None:
    """Test ValueError is raised when valid synchronous points are below threshold."""
    idx = pd.date_range("1990-01-01", periods=5, freq="MS")
    s1 = pd.Series([1.0, np.nan, np.nan, np.nan, np.nan], index=idx)
    s2 = pd.Series([1.1, 2.0, 3.0, 4.0, 5.0], index=idx)

    with pytest.raises(ValueError, match="Insufficient valid synchronous samples"):
        compute_validation_metrics(s1, s2, min_valid_samples=3)


def test_validation_report_export(
    synthetic_paired_series: Tuple[pd.Series, pd.Series],
) -> None:
    """Test exporting ValidationReport to dictionary and pandas Series."""
    observed, modeled = synthetic_paired_series
    report = compute_validation_metrics(observed, modeled)

    d = report.to_dict()
    assert isinstance(d, dict)
    assert "pearson_r" in d
    assert "rmse" in d
    assert "kge" in d

    s = report.to_series()
    assert isinstance(s, pd.Series)
    assert s["n_samples"] == 360


def test_taylor_statistics_identical(
    synthetic_paired_series: Tuple[pd.Series, pd.Series],
) -> None:
    """Test Taylor statistics on identical series yield r=1 and CRMSE=0."""
    observed, _ = synthetic_paired_series
    stats = taylor_statistics(observed, observed)

    assert isinstance(stats, TaylorStatistics)
    assert not stats.normalized
    assert stats.std_obs > 0.0
    assert math.isclose(stats.std_obs, stats.std_model, rel_tol=1e-5)
    assert math.isclose(stats.correlation, 1.0, rel_tol=1e-5)
    assert math.isclose(stats.crmse, 0.0, abs_tol=1e-5)


def test_taylor_statistics_geometric_law_of_cosines() -> None:
    """Test that CRMSE rigorously satisfies Law of Cosines identity."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2000-01-01", periods=120, freq="MS")
    s_obs = pd.Series(rng.normal(20.0, 5.0, size=120), index=idx)
    s_mod = pd.Series(s_obs * 0.8 + rng.normal(0.0, 2.0, size=120), index=idx)

    stats = taylor_statistics(s_obs, s_mod)

    # Verification of Taylor (2001) geometric closure
    expected_crmse = math.sqrt(
        stats.std_obs**2
        + stats.std_model**2
        - 2.0 * stats.std_obs * stats.std_model * stats.correlation
    )
    assert math.isclose(stats.crmse, expected_crmse, rel_tol=1e-5)
    assert 0.0 < stats.correlation < 1.0


def test_taylor_statistics_normalized() -> None:
    """Test normalized Taylor statistics where std_obs is fixed to 1.0."""
    idx = pd.date_range("2000-01-01", periods=50, freq="MS")
    s_obs = pd.Series(np.linspace(10.0, 30.0, 50), index=idx)
    s_mod = pd.Series(np.linspace(5.0, 15.0, 50), index=idx)

    unnorm = taylor_statistics(s_obs, s_mod, normalize=False)
    norm = taylor_statistics(s_obs, s_mod, normalize=True)

    assert norm.normalized
    assert math.isclose(norm.std_obs, 1.0, rel_tol=1e-5)
    assert math.isclose(norm.std_model, unnorm.std_model / unnorm.std_obs, rel_tol=1e-5)
    assert math.isclose(norm.crmse, unnorm.crmse / unnorm.std_obs, rel_tol=1e-5)
    assert math.isclose(norm.correlation, unnorm.correlation, rel_tol=1e-5)


def test_taylor_statistics_export() -> None:
    """Test dictionary and pandas Series serialization."""
    idx = pd.date_range("2000-01-01", periods=10, freq="MS")
    s1 = pd.Series(np.arange(10, dtype=float), index=idx)
    s2 = pd.Series(np.arange(10, dtype=float) * 1.5, index=idx)

    stats = taylor_statistics(s1, s2)
    d = stats.to_dict()
    assert "std_obs" in d
    assert "crmse" in d
    assert "correlation" in d

    s = stats.to_series()
    assert isinstance(s, pd.Series)
    assert s.name == "taylor_statistics"
    assert math.isclose(s["correlation"], 1.0, rel_tol=1e-5)


def test_taylor_statistics_insufficient_samples() -> None:
    """Test error handling when valid pairs are below min_valid_samples."""
    idx = pd.date_range("2000-01-01", periods=3, freq="MS")
    s1 = pd.Series([1.0, np.nan, np.nan], index=idx)
    s2 = pd.Series([1.0, 2.0, 3.0], index=idx)

    with pytest.raises(ValueError, match="Insufficient valid synchronous samples"):
        taylor_statistics(s1, s2, min_valid_samples=3)
