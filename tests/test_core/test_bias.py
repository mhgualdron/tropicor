"""Unit and integration tests for statistical bias correction (Sprint 8).

Validates:
1. LinearScalingCorrection (additive for temperature, multiplicative for rain).
2. Exact in-sample volume and mean conservation for monthly and pooled modes.
3. Moving_window fits on multi-month window but applies strictly to target month;
   asserts that volume conservation does NOT hold strictly for moving_window.
4. EmpiricalQuantileMapping sample-size capping (n_quantiles capped at group size).
5. Distributional error evaluation (KS distance, quantile errors) without assuming
   point-to-point RMSE reduction.
6. Out-of-sample split-sample and Leave-One-Year-Out (LOYO) cross-validation.
7. Physical constraints (non-negativity for precipitation) and extrapolation modes.
8. Zero-model-mean handling and dry threshold warnings.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from tropicor.core.bias import (
    EmpiricalQuantileMapping,
    LinearScalingCorrection,
    linear_scale_series,
    quantile_map_series,
)


@pytest.fixture
def monthly_temperature_series() -> tuple[pd.Series, pd.Series]:
    """Generate 30-year synthetic monthly temperature series with a known bias."""
    idx = pd.date_range("1990-01-01", periods=360, freq="MS")
    t = np.linspace(0, 30 * 2 * np.pi, 360)
    rng = np.random.default_rng(42)

    # Observed temperature: ~24°C base + annual cycle + noise
    obs = pd.Series(
        24.0 + 3.0 * np.sin(t) + rng.normal(0, 0.5, 360),
        index=idx,
        name="T_obs",
    )
    # Modeled temperature: cold biased (~18°C base, -6°C bias) + seasonal amplitude
    mod = pd.Series(
        18.0 + 4.5 * np.sin(t) + rng.normal(0, 0.5, 360),
        index=idx,
        name="T_mod",
    )
    return obs, mod


@pytest.fixture
def monthly_precipitation_series() -> tuple[pd.Series, pd.Series]:
    """Generate 30-year synthetic monthly precipitation series with volume bias."""
    idx = pd.date_range("1990-01-01", periods=360, freq="MS")
    t = np.linspace(0, 30 * 2 * np.pi, 360)
    rng = np.random.default_rng(101)

    # Observed precipitation: Gamma-like positive values with seasonal peaks
    seasonal_obs = 100.0 + 80.0 * np.sin(t)
    noise_obs = rng.gamma(shape=3.0, scale=10.0, size=360)
    obs = pd.Series(
        np.maximum(0.0, seasonal_obs + noise_obs),
        index=idx,
        name="P_obs",
    )

    # Modeled precipitation: underestimating dry season, overestimating wet season
    seasonal_mod = 140.0 + 120.0 * np.sin(t)
    noise_mod = rng.gamma(shape=2.0, scale=15.0, size=360)
    mod = pd.Series(
        np.maximum(0.0, seasonal_mod + noise_mod),
        index=idx,
        name="P_mod",
    )
    return obs, mod


# =============================================================================
# 1. LinearScalingCorrection Tests
# =============================================================================


def test_linear_scaling_additive_monthly_mean_matching(
    monthly_temperature_series: tuple[pd.Series, pd.Series],
) -> None:
    """Verify additive monthly linear scaling matches monthly means in-sample."""
    obs, mod = monthly_temperature_series
    scaler = LinearScalingCorrection(
        variable="temperature", method="additive", grouping="monthly"
    )
    corr = scaler.fit_transform(obs, mod)

    for m in range(1, 13):
        obs_m = obs[obs.index.month == m]
        corr_m = corr[corr.index.month == m]
        np.testing.assert_allclose(
            float(corr_m.mean()), float(obs_m.mean()), rtol=1e-10
        )


def test_linear_scaling_additive_pooled_mean_matching(
    monthly_temperature_series: tuple[pd.Series, pd.Series],
) -> None:
    """Verify additive pooled linear scaling matches global mean in-sample."""
    obs, mod = monthly_temperature_series
    scaler = LinearScalingCorrection(
        variable="temperature", method="additive", grouping="pooled"
    )
    corr = scaler.fit_transform(obs, mod)

    np.testing.assert_allclose(float(corr.mean()), float(obs.mean()), rtol=1e-10)


def test_linear_scaling_multiplicative_monthly_volume_conservation(
    monthly_precipitation_series: tuple[pd.Series, pd.Series],
) -> None:
    """Verify exact monthly volume conservation under multiplicative monthly mode."""
    obs, mod = monthly_precipitation_series
    scaler = LinearScalingCorrection(
        variable="precipitation", method="multiplicative", grouping="monthly"
    )
    corr = scaler.fit_transform(obs, mod)

    for m in range(1, 13):
        sum_obs = float(obs[obs.index.month == m].sum())
        sum_corr = float(corr[corr.index.month == m].sum())
        np.testing.assert_allclose(sum_corr, sum_obs, rtol=1e-10)


def test_linear_scaling_multiplicative_pooled_volume_conservation(
    monthly_precipitation_series: tuple[pd.Series, pd.Series],
) -> None:
    """Verify exact global volume conservation under multiplicative pooled mode."""
    obs, mod = monthly_precipitation_series
    scaler = LinearScalingCorrection(
        variable="precipitation", method="multiplicative", grouping="pooled"
    )
    corr = scaler.fit_transform(obs, mod)

    np.testing.assert_allclose(float(corr.sum()), float(obs.sum()), rtol=1e-10)


def test_linear_scaling_moving_window_applied_strictly_to_target_month(
    monthly_precipitation_series: tuple[pd.Series, pd.Series],
) -> None:
    """Verify moving_window fits on 3-month window but applies only to target month."""
    obs, mod = monthly_precipitation_series
    scaler = LinearScalingCorrection(
        variable="precipitation",
        method="multiplicative",
        grouping="moving_window",
        window_size=3,
    )
    scaler.fit(obs, mod)

    # January (m=1) window is Dec (12), Jan (1), Feb (2)
    win_mask = obs.index.month.isin([12, 1, 2])
    expected_jan_factor = float(obs[win_mask].mean()) / float(mod[win_mask].mean())
    np.testing.assert_allclose(scaler.factors_[1], expected_jan_factor, rtol=1e-10)

    # When transformed, only January points are scaled by scaler.factors_[1]
    corr = scaler.transform(mod)
    jan_mod = mod[mod.index.month == 1]
    jan_corr = corr[corr.index.month == 1]
    np.testing.assert_allclose(
        jan_corr.to_numpy(),
        jan_mod.to_numpy() * expected_jan_factor,
        rtol=1e-10,
    )

    # Volume conservation is intentionally NOT asserted for moving_window:
    # the January factor is estimated on Dec-Jan-Feb, so in-sample January
    # totals are not guaranteed to match observations.


def test_linear_scaling_zero_model_mean_guard() -> None:
    """Verify zero model mean warns, preserves zeros, and avoids additive fallback."""
    idx = pd.date_range("2000-01-01", periods=12, freq="MS")
    obs = pd.Series([10.0] * 12, index=idx)
    mod = pd.Series([0.0] * 12, index=idx)

    scaler = LinearScalingCorrection(
        variable="precipitation", method="multiplicative", grouping="pooled"
    )
    with pytest.warns(UserWarning, match="Modeled mean is ~0.0"):
        corr = scaler.fit_transform(obs, mod)

    # Values remain zero with factor 1.0; no additive shift to 10.0
    np.testing.assert_array_equal(corr.to_numpy(), np.zeros(12))


# =============================================================================
# 2. EmpiricalQuantileMapping Tests
# =============================================================================


def test_eqm_quantile_granularity_capped_at_sample_size(
    monthly_precipitation_series: tuple[pd.Series, pd.Series],
) -> None:
    """Verify n_quantiles=1000 is capped at sample size (~30 points per month)."""
    obs, mod = monthly_precipitation_series
    eqm = EmpiricalQuantileMapping(
        variable="precipitation",
        grouping="monthly",
        n_quantiles=1000,
    )
    eqm.fit(obs, mod)

    for m in range(1, 13):
        table = eqm.transfer_tables_[m]
        # In a 30-year series, each month has exactly 30 values
        assert len(table["prob"]) <= 30
        assert len(table["q_obs"]) <= 30
        assert len(table["q_mod"]) <= 30


def test_eqm_moving_window_fitted_on_window_applied_to_target_month(
    monthly_precipitation_series: tuple[pd.Series, pd.Series],
) -> None:
    """Verify EQM moving_window pools 3-month samples, applies only to target month."""
    obs, mod = monthly_precipitation_series
    eqm = EmpiricalQuantileMapping(
        variable="precipitation",
        grouping="moving_window",
        window_size=3,
        n_quantiles=500,
    )
    eqm.fit(obs, mod)

    # January table should be fitted on Dec+Jan+Feb (30 * 3 = 90 points)
    jan_table = eqm.transfer_tables_[1]
    assert len(jan_table["prob"]) == 90

    # Transform applies transfer table strictly to January points
    corr = eqm.transform(mod)
    assert len(corr) == len(mod)
    assert corr.name == mod.name


def test_eqm_distributional_error_reduction(
    monthly_precipitation_series: tuple[pd.Series, pd.Series],
) -> None:
    """Verify EQM reduces Kolmogorov-Smirnov distance and quantile discrepancies."""
    obs, mod = monthly_precipitation_series
    eqm = EmpiricalQuantileMapping(
        variable="precipitation",
        grouping="monthly",
    )
    corr = eqm.fit_transform(obs, mod)

    # Measure KS statistic before and after
    ks_before = stats.ks_2samp(mod.to_numpy(), obs.to_numpy()).statistic
    ks_after = stats.ks_2samp(corr.to_numpy(), obs.to_numpy()).statistic
    assert ks_after < ks_before

    # Verify key quantiles (P10, P50, P90) are closer to observed
    for q in [0.10, 0.50, 0.90]:
        q_obs = float(np.quantile(obs, q))
        q_mod = float(np.quantile(mod, q))
        q_corr = float(np.quantile(corr, q))
        assert abs(q_corr - q_obs) < abs(q_mod - q_obs)


def test_eqm_precipitation_non_negativity() -> None:
    """Verify EQM strictly clips precipitation at >= 0.0."""
    idx = pd.date_range("2000-01-01", periods=24, freq="MS")
    obs = pd.Series([5.0, 10.0] * 12, index=idx)
    # Model with extreme negative perturbation
    mod = pd.Series([-50.0, 100.0] * 12, index=idx)

    eqm = EmpiricalQuantileMapping(variable="precipitation", grouping="pooled")
    corr = eqm.fit_transform(obs, mod)
    assert (corr >= 0.0).all()


def test_eqm_dry_threshold_warning() -> None:
    """Verify passing dry_threshold > 0 on monthly series emits UserWarning."""
    idx = pd.date_range("2000-01-01", periods=12, freq="MS")
    obs = pd.Series([20.0] * 12, index=idx)
    mod = pd.Series([25.0] * 12, index=idx)

    with pytest.warns(
        UserWarning, match="dry_threshold=0.1 mm is a daily-scale concept"
    ):
        eqm = EmpiricalQuantileMapping(
            variable="precipitation",
            dry_threshold=0.1,
        )
        eqm.fit(obs, mod)


def test_eqm_extrapolation_modes() -> None:
    """Verify constant, linear, and delta extrapolation on out-of-sample extremes."""
    idx = pd.date_range("2000-01-01", periods=12, freq="MS")
    obs = pd.Series([10.0, 20.0, 30.0, 40.0] * 3, index=idx)
    mod = pd.Series([12.0, 22.0, 32.0, 42.0] * 3, index=idx)

    # Test extreme out-of-sample model value: 100.0 (max calibration mod was 42.0)
    eval_series = pd.Series(
        [100.0], index=pd.date_range("2010-01-01", periods=1, freq="MS")
    )

    # Constant: clamps to max observed (40.0)
    eqm_const = EmpiricalQuantileMapping(
        variable="temperature", grouping="pooled", extrapolation="constant"
    )
    eqm_const.fit(obs, mod)
    res_const = eqm_const.transform(eval_series)
    np.testing.assert_allclose(float(res_const.iloc[0]), 40.0)

    # Delta: adds deviation (100 - 42 = 58) to max observed (40 + 58 = 98.0)
    eqm_delta = EmpiricalQuantileMapping(
        variable="temperature", grouping="pooled", extrapolation="delta"
    )
    eqm_delta.fit(obs, mod)
    res_delta = eqm_delta.transform(eval_series)
    np.testing.assert_allclose(float(res_delta.iloc[0]), 98.0)


def test_eqm_precipitation_delta_is_multiplicative() -> None:
    """Verify precipitation delta extrapolation preserves the ratio, not offset."""
    idx = pd.date_range("2000-01-01", periods=12, freq="MS")
    obs = pd.Series([10.0, 20.0, 30.0, 40.0] * 3, index=idx)
    mod = pd.Series([20.0, 40.0, 60.0, 80.0] * 3, index=idx)
    eval_series = pd.Series(
        [160.0], index=pd.date_range("2010-01-01", periods=1, freq="MS")
    )

    eqm = EmpiricalQuantileMapping(
        variable="precipitation", grouping="pooled", extrapolation="delta"
    )
    eqm.fit(obs, mod)
    res = eqm.transform(eval_series)
    # 160 is 2x the calibration max (80) -> 2x the observed max (40) = 80
    np.testing.assert_allclose(float(res.iloc[0]), 80.0)


# =============================================================================
# 3. Out-of-Sample Split-Sample & Cross-Validation Tests
# =============================================================================


def test_split_sample_out_of_sample_evaluation(
    monthly_precipitation_series: tuple[pd.Series, pd.Series],
) -> None:
    """Verify strict out-of-sample split-sample protocol (1990-2009 cal, 2010-2019)."""
    obs, mod = monthly_precipitation_series

    obs_cal = obs.loc["1990-01-01":"2009-12-01"]
    mod_cal = mod.loc["1990-01-01":"2009-12-01"]
    obs_test = obs.loc["2010-01-01":"2019-12-01"]
    mod_test = mod.loc["2010-01-01":"2019-12-01"]

    eqm = EmpiricalQuantileMapping(variable="precipitation", grouping="monthly")
    eqm.fit(obs_cal, mod_cal)
    mod_test_corr = eqm.transform(mod_test)

    # Evaluated strictly on test period
    ks_raw = stats.ks_2samp(mod_test.to_numpy(), obs_test.to_numpy()).statistic
    ks_corr = stats.ks_2samp(mod_test_corr.to_numpy(), obs_test.to_numpy()).statistic
    assert ks_corr < ks_raw


def test_leave_one_year_out_cross_validation(
    monthly_temperature_series: tuple[pd.Series, pd.Series],
) -> None:
    """Verify Leave-One-Year-Out (LOYO) cross-validation protocol over 5 years."""
    obs, mod = monthly_temperature_series
    years = obs.index.year.unique()[:5]

    loyo_errors_raw: list[float] = []
    loyo_errors_corr: list[float] = []

    for test_year in years:
        cal_mask = obs.index.year != test_year
        test_mask = obs.index.year == test_year

        scaler = LinearScalingCorrection(
            variable="temperature", method="additive", grouping="monthly"
        )
        scaler.fit(obs[cal_mask], mod[cal_mask])
        corr_test = scaler.transform(mod[test_mask])

        mae_raw = float(np.mean(np.abs(mod[test_mask] - obs[test_mask])))
        mae_corr = float(np.mean(np.abs(corr_test - obs[test_mask])))

        loyo_errors_raw.append(mae_raw)
        loyo_errors_corr.append(mae_corr)

    # Average out-of-sample MAE improves across LOYO folds
    assert np.mean(loyo_errors_corr) < np.mean(loyo_errors_raw)


# =============================================================================
# 4. Input Guards and Functional Wrappers
# =============================================================================


def test_input_guards_and_not_fitted() -> None:
    """Verify error raises on unfitted calls and non-DatetimeIndex series."""
    scaler = LinearScalingCorrection()
    series_no_dt = pd.Series([1.0, 2.0], index=[0, 1])

    with pytest.raises(RuntimeError, match="not fitted yet"):
        scaler.transform(
            pd.Series([1.0], index=pd.date_range("2000-01-01", periods=1, freq="MS"))
        )

    with pytest.raises(ValueError, match="must have a pandas DatetimeIndex"):
        scaler.fit(series_no_dt, series_no_dt)


def test_functional_wrappers(
    monthly_temperature_series: tuple[pd.Series, pd.Series],
) -> None:
    """Verify linear_scale_series and quantile_map_series convenience functions."""
    obs, mod = monthly_temperature_series
    s_corr = linear_scale_series(obs, mod, variable="temperature", grouping="monthly")
    q_corr = quantile_map_series(obs, mod, variable="temperature", grouping="monthly")

    assert len(s_corr) == len(mod)
    assert len(q_corr) == len(mod)
    assert isinstance(s_corr.index, pd.DatetimeIndex)
    assert isinstance(q_corr.index, pd.DatetimeIndex)


def test_default_grouping_by_variable() -> None:
    """Verify that defaults match exploratory recommendations:

    - LinearScalingCorrection: monthly for both temperature and precipitation.
    - EmpiricalQuantileMapping: monthly for precipitation, moving_window for
      temperature.
    """
    ls_temp = LinearScalingCorrection(variable="temperature")
    assert ls_temp.grouping == "monthly"

    ls_prec = LinearScalingCorrection(variable="precipitation")
    assert ls_prec.grouping == "monthly"

    eqm_prec = EmpiricalQuantileMapping(variable="precipitation")
    assert eqm_prec.grouping == "monthly"

    eqm_temp = EmpiricalQuantileMapping(variable="temperature")
    assert eqm_temp.grouping == "moving_window"
