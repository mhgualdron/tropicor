import warnings
from pathlib import Path
from typing import List, Literal, cast

import netCDF4  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import pytest

from tropicor.core.protocol import temperature_tuning_loss
from tropicor.downscale.temperature import (
    Level1MonthlyAdditiveTemperature,
    Level2MeanStateDTRScalingTemperature,
    Level3QuantileMappingTemperature,
    Level3TemperatureTuningResult,
    _doy_window_mask,
    date_to_leap_doy,
    load_calibration_temperature_pair,
    select_best_level3_temperature_result,
    tune_level3_temperature_hyperparameters,
)

# ==============================================================================
# Helper to generate synthetic multi-year temperature series
# ==============================================================================


def _make_daily_temp_series(
    start: str,
    end: str,
    base_tmax: float,
    base_tmin: float,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic daily temperature DataFrame with realistic variability."""
    idx = pd.date_range(start, end, freq="D")
    rng = np.random.default_rng(seed)
    n = len(idx)

    months = idx.month.to_numpy()
    seasonal_cycle = 2.0 * np.sin(2.0 * np.pi * (months - 1) / 12.0)

    noise_tmax = rng.normal(0.0, 1.0, size=n)
    noise_tmin = rng.normal(0.0, 1.0, size=n)

    tmax_vals = base_tmax + seasonal_cycle + noise_tmax
    tmin_vals = base_tmin + seasonal_cycle + noise_tmin

    # Ensure physical consistency: Tmax >= Tmin + 5.0
    inconsistent = tmin_vals >= tmax_vals
    if inconsistent.any():
        tmax_vals[inconsistent] = tmin_vals[inconsistent] + 5.0

    return pd.DataFrame(
        {"tmax": tmax_vals, "tmin": tmin_vals},
        index=idx,
    )


# ==============================================================================
# 1. Level 1: Hand-Calculated Monthly Additive Bias
# ==============================================================================


def test_level1_monthly_additive_hand_calculated() -> None:
    """Verify Level 1 monthly additive bias correction (hand-calculated)."""
    idx = pd.date_range("1990-01-01", "1990-12-31", freq="D")

    obs_tmax = pd.Series(index=idx, dtype=float)
    obs_tmin = pd.Series(index=idx, dtype=float)
    mod_tmax = pd.Series(index=idx, dtype=float)
    mod_tmin = pd.Series(index=idx, dtype=float)

    for m in range(1, 13):
        m_mask = idx.month == m
        obs_tmax.loc[m_mask] = 20.0
        obs_tmin.loc[m_mask] = 10.0
        mod_tmax.loc[m_mask] = 23.0  # bias = +3.0
        mod_tmin.loc[m_mask] = 12.0  # bias = +2.0

    model = Level1MonthlyAdditiveTemperature()
    model.fit(obs_tmax, obs_tmin, mod_tmax, mod_tmin)

    assert np.isclose(model.tmax_biases_[1], 3.0, atol=1e-6)
    assert np.isclose(model.tmin_biases_[1], 2.0, atol=1e-6)

    # Test day in Jan 1991: mod=(26.0, 12.0) -> corr=(26-3=23, 12-2=10)
    test_idx = pd.to_datetime(["1991-01-15"])
    test_mod_tmax = pd.Series([26.0], index=test_idx)
    test_mod_tmin = pd.Series([12.0], index=test_idx)

    corr_tmax, corr_tmin = model.transform(test_mod_tmax, test_mod_tmin)

    assert np.isclose(corr_tmax.iloc[0], 23.0, atol=1e-6)
    assert np.isclose(corr_tmin.iloc[0], 10.0, atol=1e-6)


# ==============================================================================
# 2. Level 2: Mean State + DTR Scaling Hand-Calculated
# ==============================================================================


def test_level2_mean_state_dtr_scaling_hand_calculated() -> None:
    """Verify Level 2 decomposition, scaling, and reconstruction (hand-calculated)."""
    idx = pd.date_range("1990-01-01", "1990-12-31", freq="D")

    obs_tmax = pd.Series(index=idx, dtype=float)
    obs_tmin = pd.Series(index=idx, dtype=float)
    mod_tmax = pd.Series(index=idx, dtype=float)
    mod_tmin = pd.Series(index=idx, dtype=float)

    # For Month 1:
    #   mod: tmax=30.0, tmin=20.0 -> Tmean=25.0, DTR=10.0
    #   obs: tmax=28.0, tmin=16.0 -> Tmean=22.0, DTR=12.0
    #   Tmean_bias = 25.0 - 22.0 = +3.0
    #   s_dtr = 12.0 / 10.0 = 1.2
    for m in range(1, 13):
        m_mask = idx.month == m
        obs_tmax.loc[m_mask] = 28.0
        obs_tmin.loc[m_mask] = 16.0
        mod_tmax.loc[m_mask] = 30.0
        mod_tmin.loc[m_mask] = 20.0

    model = Level2MeanStateDTRScalingTemperature()
    model.fit(obs_tmax, obs_tmin, mod_tmax, mod_tmin)

    assert np.isclose(model.tmean_biases_[1], 3.0, atol=1e-6)
    assert np.isclose(model.dtr_scaling_factors_[1], 1.2, atol=1e-6)

    # Test day:
    #   mod: tmax=32.0, tmin=18.0 -> Tmean=25.0, DTR=14.0
    #   Tmean_corr = 25.0 - 3.0 = 22.0
    #   DTR_corr = 14.0 * 1.2 = 16.8
    #   Tmax_corr = 22.0 + 16.8 / 2 = 30.4
    #   Tmin_corr = 22.0 - 16.8 / 2 = 13.6
    test_idx = pd.to_datetime(["1991-01-10"])
    test_tmax = pd.Series([32.0], index=test_idx)
    test_tmin = pd.Series([18.0], index=test_idx)

    corr_tmax, corr_tmin = model.transform(test_tmax, test_tmin)

    assert np.isclose(corr_tmax.iloc[0], 30.4, atol=1e-6)
    assert np.isclose(corr_tmin.iloc[0], 13.6, atol=1e-6)


def test_level2_dtr_mod_near_zero_raises_value_error() -> None:
    """Verify that Level 2 raises ValueError when mod DTR <= 1e-4 °C (Decision A.4)."""
    idx = pd.date_range("1990-01-01", "1990-12-31", freq="D")
    obs_tmax = pd.Series(25.0, index=idx)
    obs_tmin = pd.Series(15.0, index=idx)

    # Modeled tmax == tmin (DTR = 0.0 <= 1e-4)
    mod_tmax = pd.Series(20.0, index=idx)
    mod_tmin = pd.Series(20.0, index=idx)

    model = Level2MeanStateDTRScalingTemperature()
    with pytest.raises(ValueError, match="Level 2 DTR scaling undefined.*<= 1e-4"):
        model.fit(obs_tmax, obs_tmin, mod_tmax, mod_tmin)


def test_universal_crossing_rule_tmid_offset_hand_calculated() -> None:
    """Verify universal deterministic crossing rule with hand-calculated Tmid offset."""
    idx = pd.date_range("1990-01-01", periods=2, freq="D")
    tmax_raw = pd.Series([25.0, 20.0], index=idx)
    tmin_raw = pd.Series([15.0, 22.0], index=idx)

    model = Level1MonthlyAdditiveTemperature()
    model.tmax_biases_ = {m: 0.0 for m in range(1, 13)}
    model.tmin_biases_ = {m: 0.0 for m in range(1, 13)}

    tmax_corr, tmin_corr = model.transform(tmax_raw, tmin_raw)

    assert tmax_corr.iloc[0] == pytest.approx(25.0)
    assert tmin_corr.iloc[0] == pytest.approx(15.0)

    # Day 2 resolved: Tmid = 21.0 -> Tmax = 21.025, Tmin = 20.975
    assert tmax_corr.iloc[1] == pytest.approx(21.025)
    assert tmin_corr.iloc[1] == pytest.approx(20.975)
    assert (tmax_corr.iloc[1] - tmin_corr.iloc[1]) == pytest.approx(0.05)

    assert model.n_crossings_before_rule_ == 1
    assert model.n_crossings_after_rule_ == 0
    assert model.last_crossing_diagnostic_ is not None
    assert model.last_crossing_diagnostic_.n_crossings == 1
    assert model.last_crossing_diagnostic_.fraction_crossings == pytest.approx(0.5)


def test_temperature_out_of_plausible_range_raises_value_error() -> None:
    """Verify temperatures outside [-60, 60] °C raise ValueError."""
    idx = pd.date_range("1990-01-01", periods=10, freq="D")
    t_kelvin = pd.Series(293.15, index=idx)
    t_normal = pd.Series(20.0, index=idx)

    model = Level1MonthlyAdditiveTemperature()
    with pytest.raises(ValueError, match="physically plausible range"):
        model.fit(t_kelvin, t_normal, t_normal, t_normal)

    with pytest.raises(ValueError, match="physically plausible range"):
        model.fit(t_normal, t_normal, t_kelvin, t_normal)

    t_fitted = Level1MonthlyAdditiveTemperature()
    t_fitted.tmax_biases_ = {m: 0.0 for m in range(1, 13)}
    t_fitted.tmin_biases_ = {m: 0.0 for m in range(1, 13)}
    t_extreme_cold = pd.Series(-70.0, index=idx)
    with pytest.raises(ValueError, match="physically plausible range"):
        t_fitted.transform(t_extreme_cold, t_normal)


def test_zero_valid_pairs_in_month_raises_value_error() -> None:
    """Verify that an entire calendar month with zero valid pairs raises ValueError."""
    idx = pd.date_range("1990-01-01", "1990-12-31", freq="D")
    obs_tmax = pd.Series(20.0, index=idx)
    obs_tmin = pd.Series(10.0, index=idx)
    mod_tmax = pd.Series(22.0, index=idx)
    mod_tmin = pd.Series(12.0, index=idx)

    mar_mask = idx.month == 3
    obs_tmax.loc[mar_mask] = np.nan

    l1 = Level1MonthlyAdditiveTemperature()
    with pytest.raises(
        ValueError, match="Month 3 has zero valid observation/model pairs"
    ):
        l1.fit(obs_tmax, obs_tmin, mod_tmax, mod_tmin)

    l2 = Level2MeanStateDTRScalingTemperature()
    with pytest.raises(
        ValueError, match="Month 3 has zero valid observation/model pairs"
    ):
        l2.fit(obs_tmax, obs_tmin, mod_tmax, mod_tmin)

    l3 = Level3QuantileMappingTemperature(grouping="monthly")
    with pytest.raises(ValueError, match="Month 3 has insufficient valid samples"):
        l3.fit(obs_tmax, obs_tmin, mod_tmax, mod_tmin)


# ==============================================================================
# 3. Level 3 EQM: Interior Piecewise Linear & Exact Node Counts
# ==============================================================================


def test_eqm_interior_mapping_hand_calculated() -> None:
    """Verify EQM interior piecewise linear mapping against hand-calculated values."""
    q_mod = np.array([10.0, 20.0, 30.0])
    q_obs = np.array([12.0, 24.0, 36.0])

    # Interior value 15.0: halfway between 10 and 20 -> 12 + 0.5 * 12 = 18.0
    # Interior value 25.0: halfway between 20 and 30 -> 24 + 0.5 * 12 = 30.0
    vals = np.array([15.0, 25.0])
    res, extrap_mask, abs_corr = Level3QuantileMappingTemperature._apply_eqm_to_array(
        vals, q_mod, q_obs, s_lower=1.0, s_upper=1.0
    )

    assert np.isclose(res[0], 18.0, atol=1e-6)
    assert np.isclose(res[1], 30.0, atol=1e-6)
    assert not extrap_mask.any()
    assert np.allclose(abs_corr, 0.0)


def test_eqm_quantile_nodes_count_exact_49_and_99() -> None:
    """Verify that 50 bins produces 49 nodes and 100 bins produces 99 nodes."""
    idx = pd.date_range("1990-01-01", "1990-12-31", freq="D")
    jan_vals = np.linspace(10.0, 40.0, 31)
    obs_tmax = pd.Series(20.0, index=idx)
    obs_tmax.loc[idx[:31]] = jan_vals
    obs_tmin = obs_tmax - 5.0
    mod_tmax = obs_tmax + 1.0
    mod_tmin = obs_tmin + 1.0

    model_50 = Level3QuantileMappingTemperature(grouping="monthly", n_quantiles=50)
    model_50.fit(obs_tmax, obs_tmin, mod_tmax, mod_tmin)
    # Exactly 49 empirical quantile nodes
    assert len(model_50.tmax_quantiles_mod_[1]) == 49
    assert len(model_50.tmax_quantiles_obs_[1]) == 49
    # B2: q_obs[0] and q_obs[-1] match np.percentile at 2 and 98
    assert np.isclose(
        model_50.tmax_quantiles_obs_[1][0],
        np.percentile(jan_vals, 2, method="linear"),
    )
    assert np.isclose(
        model_50.tmax_quantiles_obs_[1][-1],
        np.percentile(jan_vals, 98, method="linear"),
    )

    model_100 = Level3QuantileMappingTemperature(grouping="monthly", n_quantiles=100)
    model_100.fit(obs_tmax, obs_tmin, mod_tmax, mod_tmin)
    # Exactly 99 empirical quantile nodes
    assert len(model_100.tmax_quantiles_mod_[1]) == 99
    assert len(model_100.tmax_quantiles_obs_[1]) == 99
    # B2: q_obs[0] and q_obs[-1] match np.percentile at 1 and 99
    assert np.isclose(
        model_100.tmax_quantiles_obs_[1][0],
        np.percentile(jan_vals, 1, method="linear"),
    )
    assert np.isclose(
        model_100.tmax_quantiles_obs_[1][-1],
        np.percentile(jan_vals, 99, method="linear"),
    )


# ==============================================================================
# 4. Level 3 Tail Extrapolations: Constant vs Linear (Both Tails + Fallback)
# ==============================================================================


def test_eqm_tail_extrapolation_constant_and_linear_hand_calculated() -> None:
    """Verify tail extrapolation for constant and linear schemes (both tails)."""
    q_mod = np.array([10.0, 20.0, 30.0])
    q_obs = np.array([12.0, 24.0, 39.0])

    # Extreme interval slopes:
    #   s_upper = (39.0 - 24.0) / (30.0 - 20.0) = 15.0 / 10.0 = 1.5
    #   s_lower = (24.0 - 12.0) / (20.0 - 10.0) = 12.0 / 10.0 = 1.2
    s_low, s_up = Level3QuantileMappingTemperature._compute_slopes(
        q_mod, q_obs, "linear"
    )
    assert np.isclose(s_low, 1.2, atol=1e-6)
    assert np.isclose(s_up, 1.5, atol=1e-6)

    # Constant slopes are strictly 1.0
    s_low_c, s_up_c = Level3QuantileMappingTemperature._compute_slopes(
        q_mod, q_obs, "constant"
    )
    assert s_low_c == 1.0
    assert s_up_c == 1.0

    # Test values: 5.0 (below Q1=10) and 35.0 (above Q3=30)
    vals = np.array([5.0, 35.0])

    # 1. Constant Extrapolation:
    # Lower (val=5.0): delta = 5 - 10 = -5.0 -> corr = 12.0 + 1.0 * (-5.0) = 7.0
    # Upper (val=35.0): delta = 35 - 30 = +5.0 -> corr = 39.0 + 1.0 * (+5.0) = 44.0
    res_c, mask_c, corr_c = Level3QuantileMappingTemperature._apply_eqm_to_array(
        vals, q_mod, q_obs, s_lower=1.0, s_upper=1.0
    )
    assert np.isclose(res_c[0], 7.0, atol=1e-6)
    assert np.isclose(res_c[1], 44.0, atol=1e-6)
    assert mask_c.all()
    assert np.isclose(corr_c[0], abs(7.0 - 5.0), atol=1e-6)  # 2.0
    assert np.isclose(corr_c[1], abs(44.0 - 35.0), atol=1e-6)  # 9.0

    # 2. Linear Extrapolation:
    # Lower (val=5.0): delta = -5.0 -> corr = 12.0 + 1.2 * (-5.0) = 12.0 - 6.0 = 6.0
    # Upper (val=35.0): delta = +5.0 -> corr = 39.0 + 1.5 * (+5.0) = 39.0 + 7.5 = 46.5
    res_l, mask_l, corr_l = Level3QuantileMappingTemperature._apply_eqm_to_array(
        vals, q_mod, q_obs, s_lower=1.2, s_upper=1.5
    )
    assert np.isclose(res_l[0], 6.0, atol=1e-6)
    assert np.isclose(res_l[1], 46.5, atol=1e-6)
    assert mask_l.all()
    assert np.isclose(corr_l[0], abs(6.0 - 5.0), atol=1e-6)  # 1.0
    assert np.isclose(corr_l[1], abs(46.5 - 35.0), atol=1e-6)  # 11.5


def test_eqm_tail_extrapolation_denominator_fallback() -> None:
    """Verify linear extrapolation falls back when denom <= 1e-9."""
    # Near identical nodes in model: denom = 1e-10 <= 1e-9
    q_mod = np.array([10.0, 10.0 + 1e-10, 30.0])
    q_obs = np.array([12.0, 15.0, 35.0])

    s_low, s_up = Level3QuantileMappingTemperature._compute_slopes(
        q_mod, q_obs, "linear"
    )
    # Lower denominator <= 1e-9 falls back to 1.0
    assert s_low == 1.0
    # Upper denominator = 30.0 - 10.0000000001 = ~20.0 > 1e-9 -> linear slope
    assert np.isclose(s_up, (35.0 - 15.0) / (30.0 - 10.0), atol=1e-5)


# ==============================================================================
# 5. Moving Window Group Assignment (Jan 1 and Dec 17)
# ==============================================================================


def test_doy_window_mask_direct() -> None:
    """Verify _doy_window_mask directly on circular 366-day leap calendar."""
    all_doys = np.arange(1, 367)

    # DOY 1: 31 elements, includes 352 and 16, excludes 351 and 17
    mask_1 = _doy_window_mask(all_doys, 1, half_window=15)
    assert mask_1.sum() == 31
    assert mask_1[0]  # DOY 1
    assert mask_1[15]  # DOY 16
    assert not mask_1[16]  # DOY 17
    assert mask_1[351]  # DOY 352
    assert not mask_1[350]  # DOY 351

    # DOY 352: 31 elements, includes 337 and 1, excludes 336 and 2
    mask_352 = _doy_window_mask(all_doys, 352, half_window=15)
    assert mask_352.sum() == 31
    assert mask_352[351]  # DOY 352
    assert mask_352[336]  # DOY 337
    assert not mask_352[335]  # DOY 336
    assert mask_352[0]  # DOY 1
    assert not mask_352[1]  # DOY 2


def test_level3_moving_window_fit_group_isolation() -> None:
    """Verify moving window fit isolates target window and wraps boundaries."""
    idx = pd.date_range("1990-01-01", "1992-12-31", freq="D")  # 3 full years
    leap_doys = np.array([date_to_leap_doy(ts) for ts in idx])

    # Case 1: Mid-year window (center DOY 100, span DOY 85..115)
    obs_tmax = pd.Series(10.0, index=idx)
    obs_tmax.loc[(leap_doys >= 85) & (leap_doys <= 115)] = 20.0
    obs_tmin = obs_tmax - 5.0
    mod_tmax = pd.Series(15.0, index=idx)
    mod_tmin = mod_tmax - 5.0

    m_mw = Level3QuantileMappingTemperature(grouping="moving_window", n_quantiles=50)
    m_mw.fit(obs_tmax, obs_tmin, mod_tmax, mod_tmin)

    # Group DOY 100 window is strictly 85..115, where obs is 20 everywhere
    assert np.all(m_mw.tmax_quantiles_obs_[100] == 20.0)
    # Group DOY 99 window is 84..114, includes DOY 84 (which is 10.0) -> not all 20
    assert not np.all(m_mw.tmax_quantiles_obs_[99] == 20.0)
    # Group DOY 101 window is 86..116, includes DOY 116 (which is 10.0) -> not all 20
    assert not np.all(m_mw.tmax_quantiles_obs_[101] == 20.0)

    # Case 2: Year-wrapping window (center DOY 1, span DOY 352..366 and 1..16)
    obs_wrap = pd.Series(10.0, index=idx)
    wrap_mask = ((leap_doys >= 352) & (leap_doys <= 366)) | (
        (leap_doys >= 1) & (leap_doys <= 16)
    )
    obs_wrap.loc[wrap_mask] = 20.0
    obs_wrap_min = obs_wrap - 5.0

    m_wrap = Level3QuantileMappingTemperature(grouping="moving_window", n_quantiles=50)
    m_wrap.fit(obs_wrap, obs_wrap_min, mod_tmax, mod_tmin)

    # Group DOY 1 window is 352..366 and 1..16, where obs is 20 everywhere
    assert np.all(m_wrap.tmax_quantiles_obs_[1] == 20.0)
    # Group DOY 2 window is 353..366 and 1..17, includes DOY 17 (which is 10.0)
    assert not np.all(m_wrap.tmax_quantiles_obs_[2] == 20.0)
    # Group DOY 366 window is 351..366 and 1..15, includes DOY 351 (which is 10.0)
    assert not np.all(m_wrap.tmax_quantiles_obs_[366] == 20.0)


# ==============================================================================
# 6. Minimum Sample Size Error & Model Attribution
# ==============================================================================


def test_minimum_group_sample_size_exact_boundary() -> None:
    """Verify exact boundary of MIN_SAMPLES_PER_GROUP (9 fails, 10 passes)."""
    idx = pd.date_range("1990-01-01", "1990-12-31", freq="D")
    # All months have 15 samples except Month 1
    obs_tmax = pd.Series(np.nan, index=idx)
    for month_num in range(2, 13):
        m_dates = idx[idx.month == month_num][:15]
        obs_tmax.loc[m_dates] = 20.0

    # In month 1: give exactly 9 valid samples -> must fail
    m1_dates_9 = idx[idx.month == 1][:9]
    obs_tmax_9 = obs_tmax.copy()
    obs_tmax_9.loc[m1_dates_9] = 20.0
    obs_tmin_9 = obs_tmax_9 - 5.0
    mod_tmax_9 = obs_tmax_9 + 1.0
    mod_tmin_9 = obs_tmin_9 + 1.0

    m = Level3QuantileMappingTemperature(grouping="monthly", n_quantiles=50)
    with pytest.raises(
        ValueError, match=r"Month 1 has insufficient valid samples.*9 \(< 10\)"
    ):
        m.fit(obs_tmax_9, obs_tmin_9, mod_tmax_9, mod_tmin_9)

    # In month 1: give exactly 10 valid samples -> must pass
    m1_dates_10 = idx[idx.month == 1][:10]
    obs_tmax_10 = obs_tmax.copy()
    obs_tmax_10.loc[m1_dates_10] = 20.0
    obs_tmin_10 = obs_tmax_10 - 5.0
    mod_tmax_10 = obs_tmax_10 + 1.0
    mod_tmin_10 = obs_tmin_10 + 1.0

    m.fit(obs_tmax_10, obs_tmin_10, mod_tmax_10, mod_tmin_10)
    assert 1 in m.tmax_quantiles_obs_


# ==============================================================================
# 7. NaN Propagation across Levels 1, 2, and 3
# ==============================================================================


def test_nan_propagation_level1_level2_level3() -> None:
    """Verify NaN propagation policies across Levels 1-3."""
    idx = pd.date_range("1990-01-01", "1990-12-31", freq="D")
    obs_tmax = pd.Series(20.0, index=idx)
    obs_tmin = pd.Series(10.0, index=idx)
    mod_tmax = pd.Series(20.0, index=idx)
    mod_tmin = pd.Series(10.0, index=idx)

    l1 = Level1MonthlyAdditiveTemperature().fit(obs_tmax, obs_tmin, mod_tmax, mod_tmin)
    l2 = Level2MeanStateDTRScalingTemperature().fit(
        obs_tmax, obs_tmin, mod_tmax, mod_tmin
    )
    l3 = Level3QuantileMappingTemperature(grouping="monthly", n_quantiles=50).fit(
        obs_tmax, obs_tmin, mod_tmax, mod_tmin
    )

    # Day 1: Tmax is NaN, Tmin is valid (15.0)
    test_idx = pd.to_datetime(["1991-01-01"])
    tmax_in = pd.Series([np.nan], index=test_idx)
    tmin_in = pd.Series([15.0], index=test_idx)

    # Level 1: Tmax is NaN, Tmin is corrected
    l1_tmax, l1_tmin = l1.transform(tmax_in, tmin_in)
    assert pd.isna(l1_tmax.iloc[0])
    assert not pd.isna(l1_tmin.iloc[0])

    # Level 2: both are NaN because DTR cannot be decomposed
    l2_tmax, l2_tmin = l2.transform(tmax_in, tmin_in)
    assert pd.isna(l2_tmax.iloc[0])
    assert pd.isna(l2_tmin.iloc[0])

    # Level 3: Tmax is NaN, Tmin is corrected
    l3_tmax, l3_tmin = l3.transform(tmax_in, tmin_in)
    assert pd.isna(l3_tmax.iloc[0])
    assert not pd.isna(l3_tmin.iloc[0])


# ==============================================================================
# 8. Masking with Partial Days (Level 1 per variable, Level 2 requiring 4)
# ==============================================================================


def test_masking_partial_days_level1_and_level2() -> None:
    """Verify Level 1 fits per variable while Level 2 requires all four values."""
    idx = pd.date_range("1990-01-01", "1990-12-31", freq="D")

    # Baseline: model is 20.0 for Tmax, 10.0 for Tmin throughout the entire year
    mod_tmax = pd.Series(20.0, index=idx)
    mod_tmin = pd.Series(10.0, index=idx)

    # In months 2-12: observations match model exactly (0 bias)
    obs_tmax = pd.Series(20.0, index=idx)
    obs_tmin = pd.Series(10.0, index=idx)

    # In Month 1 (January 1990):
    # Days 1-10: all four values valid.
    # obs_tmax = 22.0 (diff = +2.0), obs_tmin = 13.0 (diff = +3.0)
    m1_days_1_10 = idx[(idx.month == 1) & (idx.day <= 10)]
    obs_tmax.loc[m1_days_1_10] = 22.0
    obs_tmin.loc[m1_days_1_10] = 13.0

    # Days 11-20: only Tmax valid with distinct value 26.0 (diff = +6.0). Tmin is NaN.
    m1_days_11_20 = idx[(idx.month == 1) & (idx.day >= 11) & (idx.day <= 20)]
    obs_tmax.loc[m1_days_11_20] = 26.0
    obs_tmin.loc[m1_days_11_20] = np.nan

    # Days 21-31: both missing in observations
    m1_days_21_31 = idx[(idx.month == 1) & (idx.day >= 21)]
    obs_tmax.loc[m1_days_21_31] = np.nan
    obs_tmin.loc[m1_days_21_31] = np.nan

    # Fit Level 1
    l1 = Level1MonthlyAdditiveTemperature().fit(obs_tmax, obs_tmin, mod_tmax, mod_tmin)

    # Hand calculation for Month 1 Level 1 (bias is model - obs):
    # Tmax bias uses 20 valid days: (10*(-2.0) + 10*(-6.0))/20 = -80.0/20 = -4.0 °C
    assert np.isclose(l1.tmax_biases_[1], -4.0)
    # Tmin bias uses only the 10 valid days: (10 * (-3.0)) / 10 = -3.0 °C
    assert np.isclose(l1.tmin_biases_[1], -3.0)

    # Fit Level 2
    l2 = Level2MeanStateDTRScalingTemperature().fit(
        obs_tmax, obs_tmin, mod_tmax, mod_tmin
    )

    # Hand calculation for Month 1 Level 2:
    # Requires all four values valid -> strictly evaluated on days 1-10 (10 days).
    # obs_tmean = (22.0 + 13.0) / 2 = 17.5 °C
    # mod_tmean = (20.0 + 10.0) / 2 = 15.0 °C
    # bias_tmean = 15.0 - 17.5 = -2.5 °C
    # obs_dtr = 22.0 - 13.0 = 9.0 °C
    # mod_dtr = 20.0 - 10.0 = 10.0 °C
    # s_dtr = 9.0 / 10.0 = 0.9
    assert np.isclose(l2.tmean_biases_[1], -2.5)
    assert np.isclose(l2.dtr_scaling_factors_[1], 0.9)


# ==============================================================================
# 9. Mismatched DatetimeIndex Validation
# ==============================================================================


def test_mismatched_datetime_index_raises_value_error() -> None:
    """Verify that series with non-identical DatetimeIndex raise ValueError."""
    idx1 = pd.date_range("1990-01-01", "1990-01-31", freq="D")
    idx2 = pd.date_range("1990-01-02", "1990-02-01", freq="D")  # Offset by 1 day

    s1 = pd.Series(20.0, index=idx1)
    s2 = pd.Series(10.0, index=idx2)

    l1 = Level1MonthlyAdditiveTemperature()
    with pytest.raises(ValueError, match="DatetimeIndex of 'obs_tmin' does not match"):
        l1.fit(s1, s2, s1, s1)

    # Fit on a full year so transform can be evaluated
    idx_yr = pd.date_range("1990-01-01", "1990-12-31", freq="D")
    s_yr = pd.Series(20.0, index=idx_yr)
    l1.fit(s_yr, s_yr, s_yr, s_yr)
    with pytest.raises(ValueError, match="DatetimeIndex of 'mod_tmin' does not match"):
        l1.transform(s1, s2)


# ==============================================================================
# 10. Parsimony Tie-Breaking Hierarchy, NaN Handling, and Final Tie-Break
# ==============================================================================


def test_parsimony_tie_breaking_hierarchy_and_edge_cases() -> None:
    """Verify parsimony hierarchy: monthly>window, 50>100, const>lin, config_id."""
    # Hierarchy test with identical loss (within 1e-6)
    r_window_50_c = Level3TemperatureTuningResult(
        config_id=4,
        grouping="moving_window",
        n_quantiles=50,
        extrapolation="constant",
        overall_tuning_loss=0.1000000,
        annual_fold_losses=[0.1],
        crossings_before_rule=0,
        crossings_after_rule=0,
        crossing_fraction=0.0,
        n_extrapolated_days=0,
        max_abs_extrapolation_correction=0.0,
        min_n_per_group=20,
    )
    r_month_100_c = Level3TemperatureTuningResult(
        config_id=2,
        grouping="monthly",
        n_quantiles=100,
        extrapolation="constant",
        overall_tuning_loss=0.1000000,
        annual_fold_losses=[0.1],
        crossings_before_rule=0,
        crossings_after_rule=0,
        crossing_fraction=0.0,
        n_extrapolated_days=0,
        max_abs_extrapolation_correction=0.0,
        min_n_per_group=20,
    )
    r_month_50_l = Level3TemperatureTuningResult(
        config_id=1,
        grouping="monthly",
        n_quantiles=50,
        extrapolation="linear",
        overall_tuning_loss=0.1000000,
        annual_fold_losses=[0.1],
        crossings_before_rule=0,
        crossings_after_rule=0,
        crossing_fraction=0.0,
        n_extrapolated_days=0,
        max_abs_extrapolation_correction=0.0,
        min_n_per_group=20,
    )
    r_month_50_c = Level3TemperatureTuningResult(
        config_id=0,
        grouping="monthly",
        n_quantiles=50,
        extrapolation="constant",
        overall_tuning_loss=0.1000004,  # Within 1e-6 of 0.1000000
        annual_fold_losses=[0.1],
        crossings_before_rule=0,
        crossings_after_rule=0,
        crossing_fraction=0.0,
        n_extrapolated_days=0,
        max_abs_extrapolation_correction=0.0,
        min_n_per_group=20,
    )

    # Monthly 50 constant should win over all others due to parsimony hierarchy
    winner = select_best_level3_temperature_result(
        [r_window_50_c, r_month_100_c, r_month_50_l, r_month_50_c],
        tolerance=1e-6,
    )
    assert winner.config_id == 0

    # Outside tolerance (> 1e-6): strictly lowest loss wins
    r_inferior_parsimonious = Level3TemperatureTuningResult(
        config_id=0,
        grouping="monthly",
        n_quantiles=50,
        extrapolation="constant",
        overall_tuning_loss=0.1000020,  # 2e-6 worse than 0.1000000
        annual_fold_losses=[0.1],
        crossings_before_rule=0,
        crossings_after_rule=0,
        crossing_fraction=0.0,
        n_extrapolated_days=0,
        max_abs_extrapolation_correction=0.0,
        min_n_per_group=20,
    )
    winner_outside = select_best_level3_temperature_result(
        [r_inferior_parsimonious, r_window_50_c], tolerance=1e-6
    )
    assert winner_outside.config_id == 4

    # Final tie-breaking by config_id: identical parsimony level
    r_a = Level3TemperatureTuningResult(
        config_id=10,
        grouping="monthly",
        n_quantiles=50,
        extrapolation="constant",
        overall_tuning_loss=0.1,
        annual_fold_losses=[0.1],
        crossings_before_rule=0,
        crossings_after_rule=0,
        crossing_fraction=0.0,
        n_extrapolated_days=0,
        max_abs_extrapolation_correction=0.0,
        min_n_per_group=20,
    )
    r_b = Level3TemperatureTuningResult(
        config_id=5,
        grouping="monthly",
        n_quantiles=50,
        extrapolation="constant",
        overall_tuning_loss=0.1,
        annual_fold_losses=[0.1],
        crossings_before_rule=0,
        crossings_after_rule=0,
        crossing_fraction=0.0,
        n_extrapolated_days=0,
        max_abs_extrapolation_correction=0.0,
        min_n_per_group=20,
    )
    winner_cfg = select_best_level3_temperature_result([r_a, r_b])
    assert winner_cfg.config_id == 5

    # NaN handling: NaN loss ignored
    r_nan = Level3TemperatureTuningResult(
        config_id=99,
        grouping="monthly",
        n_quantiles=50,
        extrapolation="constant",
        overall_tuning_loss=np.nan,
        annual_fold_losses=[np.nan],
        crossings_before_rule=0,
        crossings_after_rule=0,
        crossing_fraction=0.0,
        n_extrapolated_days=0,
        max_abs_extrapolation_correction=0.0,
        min_n_per_group=20,
    )
    assert select_best_level3_temperature_result([r_nan, r_a]).config_id == 10

    # All NaN raises ValueError
    with pytest.raises(
        ValueError, match="No candidate configuration produced a finite"
    ):
        select_best_level3_temperature_result([r_nan])


# ==============================================================================
# 11. Cross-Validation: Unpaired Years Omission & Crossing Diagnostics
# ==============================================================================


def test_tune_level3_cross_validation_comprehensive() -> None:
    """Verify CV tuning loss matches hand-fitted concatenated OOS predictions
    and crossings.
    """
    # Create 3 full years: 1990, 1991, 1992
    idx = pd.date_range("1990-01-01", "1992-12-31", freq="D")

    # Base model series with distinct yearly distributions
    mod_tmax = pd.Series(20.0, index=idx)
    mod_tmin = pd.Series(10.0, index=idx)

    for y in [1990, 1991, 1992]:
        mask_y = idx.year == y
        mod_tmax.loc[mask_y] += (y - 1990) * 2.0
        mod_tmin.loc[mask_y] += (y - 1990) * 1.0

    obs_tmax = mod_tmax + 1.0
    obs_tmin = mod_tmin - 1.0

    # Construct crossing condition on DOY 100 in 1990 and 1991 (train) vs 1992 (test):
    # On DOY 100 across all years, model has very small gap
    doy_100_mask = idx.dayofyear == 100
    mod_tmax.loc[doy_100_mask] = 16.0
    mod_tmin.loc[doy_100_mask] = 15.0

    # In 1990 and 1991 (train folds), observations strongly invert the order:
    train_doy_100 = doy_100_mask & (idx.year.isin([1990, 1991]))
    obs_tmax.loc[train_doy_100] = 12.0  # mod 16 maps downwards to 12
    obs_tmin.loc[train_doy_100] = 17.0  # mod 15 maps upwards to 17

    # In 1992 (test fold), observations exist
    test_doy_100 = doy_100_mask & (idx.year == 1992)
    obs_tmax.loc[test_doy_100] = 14.0
    obs_tmin.loc[test_doy_100] = 15.0

    # Add an extreme out-of-bounds point to produce extrapolation
    # In 1992 DOY 200: mod_tmax = 35.0 (well above training max ~25.0)
    doy_200_1992 = (idx.year == 1992) & (idx.dayofyear == 200)
    mod_tmax.loc[doy_200_1992] = 35.0

    configs = [
        {
            "config_id": 0,
            "grouping": "monthly",
            "n_quantiles": 50,
            "extrapolation": "constant",
        },
        {
            "config_id": 4,
            "grouping": "moving_window",
            "n_quantiles": 50,
            "extrapolation": "constant",
        },
    ]

    best_res, final_model, all_res = tune_level3_temperature_hyperparameters(
        obs_tmax=obs_tmax,
        obs_tmin=obs_tmin,
        mod_tmax=mod_tmax,
        mod_tmin=mod_tmin,
        configs=configs,
    )

    valid_years = [1990, 1991, 1992]

    for res in all_res:
        # Hand-fit each fold for this config
        fold_oos_tmax: List[pd.Series] = []
        fold_oos_tmin: List[pd.Series] = []
        fold_crossings_before: List[int] = []
        fold_extrap_days: List[int] = []

        for val_year in valid_years:
            val_mask = idx.year == val_year
            train_mask = ~val_mask

            m_fold = Level3QuantileMappingTemperature(
                grouping=cast(Literal["monthly", "moving_window"], res.grouping),
                n_quantiles=res.n_quantiles,
                extrapolation=cast(Literal["constant", "linear"], res.extrapolation),
            )
            m_fold.fit(
                obs_tmax.loc[train_mask],
                obs_tmin.loc[train_mask],
                mod_tmax.loc[train_mask],
                mod_tmin.loc[train_mask],
            )
            p_tmax, p_tmin = m_fold.transform(
                mod_tmax.loc[val_mask],
                mod_tmin.loc[val_mask],
            )
            fold_oos_tmax.append(p_tmax)
            fold_oos_tmin.append(p_tmin)
            fold_crossings_before.append(m_fold.n_crossings_before_rule_)
            fold_extrap_days.append(m_fold.n_extrapolated_days_)

        hand_pred_tmax = pd.concat(fold_oos_tmax).sort_index()
        hand_pred_tmin = pd.concat(fold_oos_tmin).sort_index()

        # (i) Selected loss == temperature_tuning_loss on concatenated OOS predictions
        hand_loss = temperature_tuning_loss(
            obs_tmax.loc[hand_pred_tmax.index],
            hand_pred_tmax,
            obs_tmin.loc[hand_pred_tmin.index],
            hand_pred_tmin,
        )
        assert np.isclose(res.overall_tuning_loss, hand_loss)

        # (i) Selected loss is distinct from mean of fold losses
        mean_fold_loss = float(np.mean(res.annual_fold_losses))
        assert not np.isclose(res.overall_tuning_loss, mean_fold_loss)

        # (ii) crossings_before_rule == sum of fold n_crossings_before_rule_
        # with > 0 crossings
        assert res.crossings_before_rule == sum(fold_crossings_before)
        assert res.crossings_before_rule > 0
        assert res.crossing_fraction == pytest.approx(
            res.crossings_before_rule / len(idx)
        )

        # (iii) crossings_after_rule measured == 0
        assert res.crossings_after_rule == 0

        # (v) n_extrapolated_days == sum of folds
        assert res.n_extrapolated_days == sum(fold_extrap_days)
        assert res.n_extrapolated_days > 0

    # (iv) Verified for both monthly and moving_window
    groupings = {r.grouping for r in all_res}
    assert "monthly" in groupings
    assert "moving_window" in groupings


# ==============================================================================
# 12. Slope Ranges Property (C10)
# ==============================================================================


def test_level3_slope_range_property() -> None:
    """Verify slope_ranges and slope_range properties on Level 3."""
    idx = pd.date_range("1990-01-01", "1992-12-31", freq="D")
    obs_tmax = pd.Series(20.0, index=idx)
    obs_tmin = pd.Series(10.0, index=idx)
    mod_tmax = pd.Series(22.0, index=idx)
    mod_tmin = pd.Series(12.0, index=idx)

    # Constant extrapolation
    m_const = Level3QuantileMappingTemperature(
        grouping="monthly", n_quantiles=50, extrapolation="constant"
    ).fit(obs_tmax, obs_tmin, mod_tmax, mod_tmin)
    assert m_const.slope_range == (1.0, 1.0)
    assert m_const.slope_ranges == {
        "tmax_upper": (1.0, 1.0),
        "tmax_lower": (1.0, 1.0),
        "tmin_upper": (1.0, 1.0),
        "tmin_lower": (1.0, 1.0),
    }

    # Linear extrapolation: slopes calculated from adjacent quantile nodes
    obs_tmax_var = pd.Series(20.0 + np.sin(np.arange(len(idx))), index=idx)
    obs_tmin_var = pd.Series(10.0 + np.cos(np.arange(len(idx))), index=idx)
    mod_tmax_var = pd.Series(22.0 + 1.5 * np.sin(np.arange(len(idx))), index=idx)
    mod_tmin_var = pd.Series(12.0 + 1.5 * np.cos(np.arange(len(idx))), index=idx)

    m_lin = Level3QuantileMappingTemperature(
        grouping="monthly", n_quantiles=50, extrapolation="linear"
    ).fit(obs_tmax_var, obs_tmin_var, mod_tmax_var, mod_tmin_var)
    s_min, s_max = m_lin.slope_range
    assert s_min <= s_max
    assert len(m_lin.slope_ranges) == 4
    for key in ["tmax_upper", "tmax_lower", "tmin_upper", "tmin_lower"]:
        assert key in m_lin.slope_ranges
        assert m_lin.slope_ranges[key][0] <= m_lin.slope_ranges[key][1]


# ==============================================================================
# 13. Real Loader with Synthetic Files: Strict Checks (B6)
# ==============================================================================


def _create_synthetic_era5_nc(
    path: Path, start_date: str, end_date: str, t2m_k: float = 293.15
) -> None:
    """Helper to create a valid synthetic ERA5 NetCDF file without warnings."""
    hourly_dates = pd.date_range(start_date, end_date, freq="h")
    n_hours = len(hourly_dates)
    with netCDF4.Dataset(str(path), "w", format="NETCDF4") as nc:
        nc.createDimension("time", n_hours)
        nc.createDimension("latitude", 1)
        nc.createDimension("longitude", 1)
        time_var = nc.createVariable("time", "f8", ("time",))
        time_var.units = "hours since 1990-01-01 00:00:00"
        lat_var = nc.createVariable("latitude", "f4", ("latitude",))
        lon_var = nc.createVariable("longitude", "f4", ("longitude",))
        t2m_var = nc.createVariable("t2m", "f4", ("time", "latitude", "longitude"))
        t2m_var.units = "K"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            time_var[:] = np.arange(n_hours)
            lat_var[:] = [4.6]
            lon_var[:] = [-74.1]
            t2m_var[:] = np.full((n_hours, 1, 1), t2m_k, dtype=np.float32)


def test_load_calibration_temperature_pair_synthetic_io(tmp_path: Path) -> None:
    """Verify loader reads synthetic DHIME CSV and ERA5 NetCDF strictly."""
    # 1. Create synthetic observation CSV spanning 1985 to 2010 (cal is 1990-2004)
    obs_dates = pd.date_range("1985-01-01", "2010-12-31", freq="D")
    df_obs = pd.DataFrame(
        {
            "CodigoEstacion": ["21205710"] * len(obs_dates),
            "Fecha": obs_dates.strftime("%Y-%m-%d"),
            "Valor": [20.0] * len(obs_dates),
        }
    )
    tmax_path = tmp_path / "tmax_obs.csv"
    tmin_path = tmp_path / "tmin_obs.csv"
    df_obs.to_csv(tmax_path, index=False)
    df_obs.to_csv(tmin_path, index=False)

    # 2. Create synthetic ERA5 NetCDF with t2m in Kelvin
    era5_file = tmp_path / "era5_t2m.nc"
    _create_synthetic_era5_nc(era5_file, "1990-01-01", "2005-01-02", 293.15)

    # Test missing mandatory arguments (window_start_utc, lag_days, min_hours)
    # raises TypeError
    with pytest.raises(TypeError):
        load_calibration_temperature_pair(  # type: ignore[call-arg]
            station_id="21205710",
            obs_tmax_path=tmax_path,
            obs_tmin_path=tmin_path,
            era5_target=era5_file,
        )

    # Successful load: truncated strictly to 1990-01-01..2004-12-31
    o_tmax, o_tmin, m_tmax, m_tmin = load_calibration_temperature_pair(
        station_id="21205710",
        obs_tmax_path=tmax_path,
        obs_tmin_path=tmin_path,
        era5_target=era5_file,
        window_start_utc=5,
        lag_days=0,
        min_hours=24,
    )
    assert o_tmax.index[0] == pd.Timestamp("1990-01-01")
    assert o_tmax.index[-1] == pd.Timestamp("2004-12-31")
    assert np.isclose(m_tmax.iloc[0], 20.0, atol=1e-4)

    # Test wrong units attribute raises ValueError
    era5_wrong_units = tmp_path / "era5_wrong_units.nc"
    with netCDF4.Dataset(str(era5_wrong_units), "w", format="NETCDF4") as nc:
        nc.createDimension("time", 24)
        nc.createDimension("latitude", 1)
        nc.createDimension("longitude", 1)
        time_var = nc.createVariable("time", "f8", ("time",))
        time_var.units = "hours since 1990-01-01 00:00:00"
        lat_var = nc.createVariable("latitude", "f4", ("latitude",))
        lon_var = nc.createVariable("longitude", "f4", ("longitude",))
        t2m_var = nc.createVariable("t2m", "f4", ("time", "latitude", "longitude"))
        t2m_var.units = "degC"  # NOT K
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            time_var[:] = np.arange(24)
            lat_var[:] = [4.6]
            lon_var[:] = [-74.1]
            t2m_var[:] = np.full((24, 1, 1), 20.0, dtype=np.float32)

    with pytest.raises(ValueError, match="must have units 'K'.*found 'degC'"):
        load_calibration_temperature_pair(
            station_id="21205710",
            obs_tmax_path=tmax_path,
            obs_tmin_path=tmin_path,
            era5_target=era5_wrong_units,
            window_start_utc=5,
            lag_days=0,
            min_hours=24,
        )

    # Test wrong station code raises ValueError
    df_wrong_st = df_obs.copy()
    df_wrong_st["CodigoEstacion"] = "99999999"
    wrong_st_path = tmp_path / "wrong_station.csv"
    df_wrong_st.to_csv(wrong_st_path, index=False)

    with pytest.raises(ValueError, match="Station ID '21205710' not found"):
        load_calibration_temperature_pair(
            station_id="21205710",
            obs_tmax_path=wrong_st_path,
            obs_tmin_path=tmin_path,
            era5_target=era5_file,
            window_start_utc=5,
            lag_days=0,
            min_hours=24,
        )


def test_load_calibration_temperature_pair_unregistered_station_raises_value_error(
    tmp_path: Path,
) -> None:
    """Verify that requesting an unregistered station raises ValueError."""
    dummy_csv = tmp_path / "obs.csv"
    dummy_csv.write_text("CodigoEstacion,Fecha,Valor\n54085010,1990-01-01,20.0\n")
    dummy_nc = tmp_path / "era5.nc"

    with pytest.raises(
        ValueError, match="has no pre-specified temperature calibration partition"
    ):
        load_calibration_temperature_pair(
            station_id="54085010",
            obs_tmax_path=dummy_csv,
            obs_tmin_path=dummy_csv,
            era5_target=dummy_nc,
            window_start_utc=5,
            lag_days=0,
            min_hours=24,
        )


def test_load_calibration_temperature_pair_missing_units_attribute_raises_value_error(
    tmp_path: Path,
) -> None:
    """Verify that an ERA5 NetCDF file without units attribute raises ValueError."""
    df_obs = pd.DataFrame(
        {
            "CodigoEstacion": ["21205710"] * 24,
            "Fecha": pd.date_range("1990-01-01", periods=24, freq="D").strftime(
                "%Y-%m-%d"
            ),
            "Valor": [20.0] * 24,
        }
    )
    csv_path = tmp_path / "obs.csv"
    df_obs.to_csv(csv_path, index=False)

    nc_path = tmp_path / "era5_no_units.nc"
    with netCDF4.Dataset(str(nc_path), "w", format="NETCDF4") as nc:
        nc.createDimension("time", 24)
        nc.createDimension("latitude", 1)
        nc.createDimension("longitude", 1)
        time_var = nc.createVariable("time", "f8", ("time",))
        time_var.units = "hours since 1990-01-01 00:00:00"
        lat_var = nc.createVariable("latitude", "f4", ("latitude",))
        lon_var = nc.createVariable("longitude", "f4", ("longitude",))
        t2m_var = nc.createVariable("t2m", "f4", ("time", "latitude", "longitude"))
        # Crucial: NO units attribute set
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            time_var[:] = np.arange(24)
            lat_var[:] = [4.6]
            lon_var[:] = [-74.1]
            t2m_var[:] = np.full((24, 1, 1), 293.15, dtype=np.float32)

    with pytest.raises(ValueError, match="must have units 'K'.*found ''"):
        load_calibration_temperature_pair(
            station_id="21205710",
            obs_tmax_path=csv_path,
            obs_tmin_path=csv_path,
            era5_target=nc_path,
            window_start_utc=5,
            lag_days=0,
            min_hours=24,
        )


def test_loader_validation_year_out_of_range_does_not_fail(tmp_path: Path) -> None:
    """B6(i): A value out of physical range (99.0) in a validation year of the
    CSV causes no error.
    """
    # Calibration is 1990-2004; put 99.0 in 2005 (validation year)
    obs_dates = pd.date_range("1990-01-01", "2006-12-31", freq="D")
    df_obs = pd.DataFrame(
        {
            "CodigoEstacion": ["21205710"] * len(obs_dates),
            "Fecha": obs_dates.strftime("%Y-%m-%d"),
            "Valor": [20.0] * len(obs_dates),
        }
    )
    # Set 99.0 on a validation date
    val_date_mask = df_obs["Fecha"] == "2005-06-15"
    df_obs.loc[val_date_mask, "Valor"] = 99.0

    tmax_path = tmp_path / "tmax.csv"
    tmin_path = tmp_path / "tmin.csv"
    df_obs.to_csv(tmax_path, index=False)
    df_obs.to_csv(tmin_path, index=False)

    era5_file = tmp_path / "era5_cal.nc"
    _create_synthetic_era5_nc(era5_file, "1990-01-01", "2005-01-02", 293.15)

    # Loader must succeed without raising ValueError for 99.0
    o_tmax, o_tmin, m_tmax, m_tmin = load_calibration_temperature_pair(
        station_id="21205710",
        obs_tmax_path=tmax_path,
        obs_tmin_path=tmin_path,
        era5_target=era5_file,
        window_start_utc=5,
        lag_days=0,
        min_hours=24,
    )
    assert o_tmax.index[-1] == pd.Timestamp("2004-12-31")


def test_loader_corrupted_validation_era5_file_not_opened(tmp_path: Path) -> None:
    """B6(ii): A corrupted ERA5 file from a validation year is not opened by the
    calibration loader.
    """
    era5_dir = tmp_path / "era5_dir"
    era5_dir.mkdir()

    # Valid file for calibration years
    _create_synthetic_era5_nc(
        era5_dir / "era5_2m_temperature_1990_2004.nc",
        "1990-01-01",
        "2005-01-02",
        293.15,
    )

    # Corrupted file for validation year 2008 (would crash if opened)
    corrupted_file = era5_dir / "era5_2m_temperature_2008_corrupt.nc"
    corrupted_file.write_bytes(b"NOT_A_VALID_NETCDF_FILE_CORRUPTED_BYTES")

    obs_dates = pd.date_range("1990-01-01", "2004-12-31", freq="D")
    df_obs = pd.DataFrame(
        {
            "CodigoEstacion": ["21205710"] * len(obs_dates),
            "Fecha": obs_dates.strftime("%Y-%m-%d"),
            "Valor": [20.0] * len(obs_dates),
        }
    )
    tmax_path = tmp_path / "tmax.csv"
    tmin_path = tmp_path / "tmin.csv"
    df_obs.to_csv(tmax_path, index=False)
    df_obs.to_csv(tmin_path, index=False)

    # Must succeed without error because 2008 file is excluded by year regex
    o_tmax, o_tmin, m_tmax, m_tmin = load_calibration_temperature_pair(
        station_id="21205710",
        obs_tmax_path=tmax_path,
        obs_tmin_path=tmin_path,
        era5_target=era5_dir,
        window_start_utc=5,
        lag_days=0,
        min_hours=24,
    )
    assert len(o_tmax) == len(obs_dates)


def test_loader_directory_without_matching_files_raises_value_error(
    tmp_path: Path,
) -> None:
    """B6(iii): Directory without matching calibration files raises ValueError."""
    era5_dir = tmp_path / "era5_empty_match"
    era5_dir.mkdir()

    # Case A: Directory without any 2m_temperature files raises ValueError
    _create_synthetic_era5_nc(
        era5_dir / "era5_total_precipitation_2000.nc",
        "2000-01-01",
        "2000-02-01",
        293.15,
    )
    tmax_path = tmp_path / "tmax.csv"
    tmin_path = tmp_path / "tmin.csv"
    obs_dates = pd.date_range("1990-01-01", "2004-12-31", freq="D")
    df_obs = pd.DataFrame(
        {
            "CodigoEstacion": ["21205710"] * len(obs_dates),
            "Fecha": obs_dates.strftime("%Y-%m-%d"),
            "Valor": [20.0] * len(obs_dates),
        }
    )
    df_obs.to_csv(tmax_path, index=False)
    df_obs.to_csv(tmin_path, index=False)

    with pytest.raises(
        ValueError, match="No '\\*2m_temperature\\*\\.nc' files found in directory"
    ):
        load_calibration_temperature_pair(
            station_id="21205710",
            obs_tmax_path=tmax_path,
            obs_tmin_path=tmin_path,
            era5_target=era5_dir,
            window_start_utc=5,
            lag_days=0,
            min_hours=24,
        )

    # Case B: Create 2m_temperature file for year 2020 only (calibration is 1990-2004)
    _create_synthetic_era5_nc(
        era5_dir / "era5_2m_temperature_2020.nc", "2020-01-01", "2020-02-01", 293.15
    )

    with pytest.raises(
        ValueError, match="No ERA5 NetCDF files matching calibration years 1990-2004"
    ):
        load_calibration_temperature_pair(
            station_id="21205710",
            obs_tmax_path=tmax_path,
            obs_tmin_path=tmin_path,
            era5_target=era5_dir,
            window_start_utc=5,
            lag_days=0,
            min_hours=24,
        )


def test_loader_missing_calibration_era5_year_raises_nan_value_error(
    tmp_path: Path,
) -> None:
    """B6(iv): Calibration year missing from ERA5 files raises ValueError
    for model NaNs.
    """
    era5_dir = tmp_path / "era5_partial"
    era5_dir.mkdir()

    # Provide only 1990 (calibration is 1990-2004)
    _create_synthetic_era5_nc(
        era5_dir / "era5_2m_temperature_1990.nc", "1990-01-01", "1990-12-31", 293.15
    )

    obs_dates = pd.date_range("1990-01-01", "2004-12-31", freq="D")
    df_obs = pd.DataFrame(
        {
            "CodigoEstacion": ["21205710"] * len(obs_dates),
            "Fecha": obs_dates.strftime("%Y-%m-%d"),
            "Valor": [20.0] * len(obs_dates),
        }
    )
    tmax_path = tmp_path / "tmax.csv"
    tmin_path = tmp_path / "tmin.csv"
    df_obs.to_csv(tmax_path, index=False)
    df_obs.to_csv(tmin_path, index=False)

    with pytest.raises(
        ValueError,
        match=r"contains missing values \(NaN\) within calibration partition 1990-2004",
    ):
        load_calibration_temperature_pair(
            station_id="21205710",
            obs_tmax_path=tmax_path,
            obs_tmin_path=tmin_path,
            era5_target=era5_dir,
            window_start_utc=5,
            lag_days=0,
            min_hours=24,
        )


def test_loader_csv_missing_valor_column_raises_value_error(tmp_path: Path) -> None:
    """B6(v): Observation CSV missing 'Valor' column raises ValueError."""
    df_no_valor = pd.DataFrame(
        {
            "CodigoEstacion": ["21205710"] * 10,
            "Fecha": pd.date_range("1990-01-01", periods=10, freq="D").strftime(
                "%Y-%m-%d"
            ),
            "Temperatura": [20.0] * 10,  # Missing 'Valor'
        }
    )
    csv_path = tmp_path / "no_valor.csv"
    df_no_valor.to_csv(csv_path, index=False)

    era5_file = tmp_path / "era5.nc"
    _create_synthetic_era5_nc(era5_file, "1990-01-01", "1991-01-02", 293.15)

    with pytest.raises(
        ValueError, match="Required columns 'Fecha' and 'Valor' must be present"
    ):
        load_calibration_temperature_pair(
            station_id="21205710",
            obs_tmax_path=csv_path,
            obs_tmin_path=csv_path,
            era5_target=era5_file,
            window_start_utc=5,
            lag_days=0,
            min_hours=24,
        )


def test_loader_csv_duplicate_dates_with_distinct_values_raises_value_error(
    tmp_path: Path,
) -> None:
    """B6(vi): Observation CSV with duplicate dates with distinct values raises
    ValueError.
    """
    df_dup = pd.DataFrame(
        {
            "CodigoEstacion": ["21205710", "21205710"],
            "Fecha": ["1990-01-01", "1990-01-01"],
            "Valor": [20.0, 25.0],  # Distinct values on same date
        }
    )
    csv_path = tmp_path / "dup_distinct.csv"
    df_dup.to_csv(csv_path, index=False)

    era5_file = tmp_path / "era5.nc"
    _create_synthetic_era5_nc(era5_file, "1990-01-01", "1991-01-02", 293.15)

    with pytest.raises(
        ValueError, match=r"Found 1 duplicate date\(s\) with distinct values"
    ):
        load_calibration_temperature_pair(
            station_id="21205710",
            obs_tmax_path=csv_path,
            obs_tmin_path=csv_path,
            era5_target=era5_file,
            window_start_utc=5,
            lag_days=0,
            min_hours=24,
        )


def test_level3_transform_hand_counted_extrapolation() -> None:
    """B3: Verify Level 3 transform extrapolation diagnostics with exact hand counts.

    obs = mod + 1 (constant extrapolation => tail correction exactly 1.0 °C).
    Injects 3 out-of-range days:
      - Day 0: out of range only in Tmax (upper tail)
      - Day 1: out of range only in Tmin (lower tail)
      - Day 2: out of range in both Tmax and Tmin
    Asserts n_extrapolated_days_ == 3 and max_abs_extrapolation_correction_ == 1.0.
    """
    dates = pd.date_range("2000-01-01", periods=365, freq="D")
    mod_tmax = pd.Series(np.linspace(20.0, 30.0, 365), index=dates)
    mod_tmin = pd.Series(np.linspace(10.0, 20.0, 365), index=dates)
    obs_tmax = mod_tmax + 1.0
    obs_tmin = mod_tmin + 1.0

    model = Level3QuantileMappingTemperature(
        grouping="monthly", n_quantiles=50, extrapolation="constant"
    )
    model.fit(obs_tmax, obs_tmin, mod_tmax, mod_tmin)

    test_dates = pd.date_range("2001-01-01", periods=10, freq="D")
    idx_train = cast(pd.DatetimeIndex, mod_tmax.index)
    jan_mod_tmax_mid = float(mod_tmax.loc[idx_train.month == 1].median())
    jan_mod_tmin_mid = float(mod_tmin.loc[idx_train.month == 1].median())

    test_tmax = pd.Series(jan_mod_tmax_mid, index=test_dates)
    test_tmin = pd.Series(jan_mod_tmin_mid, index=test_dates)

    # Day 0: only Tmax out of range (upper: > max of Jan training ~ 20.8 °C)
    test_tmax.iloc[0] = 35.0
    # Day 1: only Tmin out of range (lower: < min of Jan training ~ 10.0 °C)
    test_tmin.iloc[1] = 5.0
    # Day 2: both out of range
    test_tmax.iloc[2] = 35.0
    test_tmin.iloc[2] = 5.0

    pred_tmax, pred_tmin = model.transform(test_tmax, test_tmin)

    assert model.n_extrapolated_days_ == 3
    assert np.isclose(model.max_abs_extrapolation_correction_, 1.0, atol=1e-6)


def test_loader_non_iso_date_format_raises_value_error(tmp_path: Path) -> None:
    """A4: Observation CSV with non-ISO dates (e.g. DD/MM/YYYY) raises ValueError."""
    era5_file = tmp_path / "era5_2m_temperature.nc"
    _create_synthetic_era5_nc(era5_file, "1990-01-01", "2005-01-02", 293.15)

    # Create CSV with non-ISO dates: "15/01/1990" instead of "1990-01-15"
    df_obs = pd.DataFrame(
        {
            "CodigoEstacion": ["21205710"] * 5,
            "Fecha": [
                "15/01/1990",
                "16/01/1990",
                "17/01/1990",
                "18/01/1990",
                "19/01/1990",
            ],
            "Valor": [20.0, 21.0, 22.0, 23.0, 24.0],
        }
    )
    tmax_path = tmp_path / "tmax_non_iso.csv"
    tmin_path = tmp_path / "tmin_valid.csv"
    df_obs.to_csv(tmax_path, index=False)

    df_valid = pd.DataFrame(
        {
            "CodigoEstacion": ["21205710"] * 5,
            "Fecha": [
                "1990-01-15",
                "1990-01-16",
                "1990-01-17",
                "1990-01-18",
                "1990-01-19",
            ],
            "Valor": [10.0, 11.0, 12.0, 13.0, 14.0],
        }
    )
    df_valid.to_csv(tmin_path, index=False)

    with pytest.raises(ValueError, match="unparseable date"):
        load_calibration_temperature_pair(
            station_id="21205710",
            obs_tmax_path=tmax_path,
            obs_tmin_path=tmin_path,
            era5_target=era5_file,
            window_start_utc=5,
            lag_days=0,
            min_hours=24,
        )
