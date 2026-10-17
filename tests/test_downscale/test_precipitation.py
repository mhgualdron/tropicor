"""Comprehensive unit test suite for daily precipitation downscaling hierarchy.

Tests verify:
- Level 1: exact mass balance, hand calculations, NaN preservation, ValueError on
  missing month.
- Level 2: Type 7 threshold matching (0.76 mm), mass balance, ValueError on zero
  denominator, preservation of sub-1mm values, NaN preservation, ValueError on
  missing month, and D.3 zero observed wet days handling.
- Level 3: 366-day leap circular window (Dec 17 = day 352, Jan 1 window =
  [352..366] U [1..16] = 31 days), all 8 discrete configurations, constant vs.
  linear extrapolation with transfer function extension, defensive zero-floor
  clipping path, NaN preservation, ValueError on groups with < 2 wet days,
  and equivalence of level2_thresholds with Level 2.
- Units validation: ValueError in fit() if series mean < 0.01 mm.
- Cross-validation: nested annual block CV skipping empty years, overall loss
  on concatenated series.
- Tie-breaking: select_best_level3_result module function adhering to D.17
  parsimony hierarchy with 1e-6 absolute numerical tolerance.
- Calibration loader: immediate trimming of validation years with synthetic CSV,
  pre-specified partition lookup for known stations, and explicit ValueError
  for unknown stations without cal_range.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tropicor.downscale.precipitation import (
    Level1LinearScalingPrecipitation,
    Level2ThresholdMatchingPrecipitation,
    Level3QuantileMappingPrecipitation,
    Level3TuningResult,
    date_to_leap_doy,
    leap_doy_circular_distance,
    load_calibration_precipitation_pair,
    select_best_level3_result,
    tune_level3_precipitation_hyperparameters,
)

# ==============================================================================
# Level 1 Tests
# ==============================================================================


def test_level1_exact_calibration_mass_balance():
    """Verify exact calibration mass balance in Level 1 per calendar month."""
    dates = pd.date_range("1980-01-01", "1980-12-31", freq="D")
    rng = np.random.default_rng(42)
    obs_vals = rng.exponential(scale=5.0, size=len(dates)) + 0.1
    mod_vals = rng.exponential(scale=3.0, size=len(dates)) + 0.1

    s_obs = pd.Series(obs_vals, index=dates)
    s_mod = pd.Series(mod_vals, index=dates)

    scaler = Level1LinearScalingPrecipitation()
    s_corr = scaler.fit_transform(s_obs, s_mod)

    # Check monthly sums match to floating point precision
    for m in range(1, 13):
        mask = dates.month == m
        sum_obs = s_obs[mask].sum()
        sum_corr = s_corr[mask].sum()
        assert np.isclose(sum_corr, sum_obs, rtol=1e-12)


def test_level1_hand_calculated_factor():
    """Hand-calculated monthly factor and transformation.

    Jan obs: [10.0, 20.0, 30.0] -> sum = 60.0
    Jan raw: [5.0, 5.0, 10.0]   -> sum = 20.0
    Expected factor: s_1 = 3.0.
    Transform input [2.0, 4.0] -> [6.0, 12.0].
    Months 2..12 padded with 1 valid day each.
    """
    cal_dates = pd.to_datetime(["1980-01-01", "1980-01-02", "1980-01-03"])
    s_obs = pd.Series([10.0, 20.0, 30.0], index=cal_dates)
    s_mod = pd.Series([5.0, 5.0, 10.0], index=cal_dates)

    # Pad months 2..12
    pad_dates = pd.date_range("1980-02-01", "1980-12-01", freq="MS")
    s_obs = pd.concat([s_obs, pd.Series([5.0] * 11, index=pad_dates)])
    s_mod = pd.concat([s_mod, pd.Series([5.0] * 11, index=pad_dates)])

    scaler = Level1LinearScalingPrecipitation()
    scaler.fit(s_obs, s_mod)
    assert np.isclose(scaler.monthly_factors[1], 3.0)

    eval_dates = pd.to_datetime(["1981-01-10", "1981-01-11"])
    s_eval = pd.Series([2.0, 4.0], index=eval_dates)
    s_out = scaler.transform(s_eval)

    expected = np.array([6.0, 12.0])
    np.testing.assert_allclose(s_out.values, expected)


def test_level1_nan_preservation():
    """Preserve NaN values in Level 1 transform."""
    cal_dates = pd.to_datetime(["1980-01-01", "1980-01-02"])
    s_obs = pd.Series([10.0, 20.0], index=cal_dates)
    s_mod = pd.Series([5.0, 5.0], index=cal_dates)

    pad_dates = pd.date_range("1980-02-01", "1980-12-01", freq="MS")
    s_obs = pd.concat([s_obs, pd.Series([5.0] * 11, index=pad_dates)])
    s_mod = pd.concat([s_mod, pd.Series([5.0] * 11, index=pad_dates)])

    scaler = Level1LinearScalingPrecipitation().fit(s_obs, s_mod)

    eval_dates = pd.to_datetime(["1981-01-01", "1981-01-02", "1981-01-03"])
    s_eval = pd.Series([5.0, np.nan, 10.0], index=eval_dates)
    s_out = scaler.transform(s_eval)

    assert not np.isnan(s_out.iloc[0])
    assert np.isnan(s_out.iloc[1])
    assert not np.isnan(s_out.iloc[2])


def test_level1_fit_raises_value_error_on_missing_month():
    """Level 1 fit raises ValueError if a month lacks paired calibration days."""
    dates = pd.date_range("1980-01-01", "1980-11-30", freq="D")  # Month 12 missing
    s_obs = pd.Series(np.full(len(dates), 5.0), index=dates)
    s_mod = pd.Series(np.full(len(dates), 5.0), index=dates)

    scaler = Level1LinearScalingPrecipitation()
    with pytest.raises(
        ValueError, match="no paired valid calibration days for month 12"
    ):
        scaler.fit(s_obs, s_mod)


# ==============================================================================
# Level 2 Tests
# ==============================================================================


def test_level2_hand_calculated_threshold_matching_type7():
    """Hand-calculated threshold matching with Type 7 linear interpolation.

    Month with 10 days (N = 10).
    Obs: 4 wet days (>= 1.0), 6 dry days (< 1.0). f_obs^wet = 0.40.
    Raw ERA5 sorted: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0, 2.0, 3.0, 4.0].
    (1 - 0.40) = 0.60 quantile under Type 7:
        Index i = (10 - 1) * 0.60 = 5.4.
        arr[5] = 0.6, arr[6] = 1.0.
        P_th = 0.6 + 0.4 * (1.0 - 0.6) = 0.76 mm.
    Truncation: values < 0.76 become 0.0.
    Retained: [1.0, 2.0, 3.0, 4.0] -> sum censored = 10.0 mm.
    Obs wet sum = 15.0 mm.
    Scaling factor s = 15.0 / 10.0 = 1.5.
    Corrected: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5, 3.0, 4.5, 6.0].
    Sum = 15.0 mm (exact mass balance).
    Months 2..12 padded with 1 valid day each.
    """
    cal_dates = pd.date_range("1980-01-01", periods=10, freq="D")
    obs_vals = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 5.0, 6.0]  # sum = 15.0
    mod_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0, 2.0, 3.0, 4.0]

    s_obs = pd.Series(obs_vals, index=cal_dates)
    s_mod = pd.Series(mod_vals, index=cal_dates)

    pad_dates = pd.date_range("1980-02-01", "1980-12-01", freq="MS")
    s_obs = pd.concat([s_obs, pd.Series([5.0] * 11, index=pad_dates)])
    s_mod = pd.concat([s_mod, pd.Series([5.0] * 11, index=pad_dates)])

    scaler = Level2ThresholdMatchingPrecipitation()
    scaler.fit(s_obs, s_mod)

    # Threshold P_th must be exactly 0.76 mm
    assert np.isclose(scaler.monthly_thresholds[1], 0.76)
    # Scaling factor must be 1.5
    assert np.isclose(scaler.monthly_scaling_factors[1], 1.5)

    s_corr = scaler.transform(s_mod.iloc[:10])
    expected = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5, 3.0, 4.5, 6.0])
    np.testing.assert_allclose(s_corr.values, expected)
    assert np.isclose(s_corr.sum(), 15.0)


def test_level2_value_error_on_zero_censored_sum():
    """Explicit ValueError when model censored sum is zero with positive obs."""
    dates = pd.date_range("1980-01-01", "1980-12-31", freq="D")
    obs_vals = [2.0 if d.month == 1 else 5.0 for d in dates]
    mod_vals = [0.0 if d.month == 1 else 5.0 for d in dates]

    s_obs = pd.Series(obs_vals, index=dates)
    s_mod = pd.Series(mod_vals, index=dates)

    scaler = Level2ThresholdMatchingPrecipitation()
    with pytest.raises(ValueError, match="zero model censored precipitation"):
        scaler.fit(s_obs, s_mod)


def test_level2_nan_preservation():
    """Preserve NaN values in Level 2 transform."""
    cal_dates = pd.date_range("1980-01-01", periods=10, freq="D")
    s_obs = pd.Series([0.0] * 6 + [5.0] * 4, index=cal_dates)
    s_mod = pd.Series(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0, 2.0, 3.0, 4.0], index=cal_dates
    )

    pad_dates = pd.date_range("1980-02-01", "1980-12-01", freq="MS")
    s_obs = pd.concat([s_obs, pd.Series([5.0] * 11, index=pad_dates)])
    s_mod = pd.concat([s_mod, pd.Series([5.0] * 11, index=pad_dates)])

    scaler = Level2ThresholdMatchingPrecipitation().fit(s_obs, s_mod)

    eval_dates = pd.to_datetime(["1981-01-01", "1981-01-02", "1981-01-03"])
    s_eval = pd.Series([0.5, np.nan, 2.0], index=eval_dates)
    s_out = scaler.transform(s_eval)

    # 0.5 < P_th (0.76) -> 0.0
    assert s_out.iloc[0] == 0.0
    assert np.isnan(s_out.iloc[1])
    # 2.0 >= 0.76 -> scaled
    assert not np.isnan(s_out.iloc[2])
    assert s_out.iloc[2] > 0.0


def test_level2_no_second_truncation_preserves_sub_1mm():
    """Level 2 has no second truncation: s_m * P_censored < 1.0 remains > 0."""
    cal_dates = pd.date_range("1980-01-01", periods=10, freq="D")
    obs_vals = [0.0] * 6 + [0.5, 0.5, 0.5, 0.5]  # total = 2.0
    mod_vals = [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        1.0,
        1.0,
        1.0,
        1.0,
    ]  # total censored = 4.0

    s_obs = pd.Series(obs_vals, index=cal_dates)
    s_mod = pd.Series(mod_vals, index=cal_dates)

    pad_dates = pd.date_range("1980-02-01", "1980-12-01", freq="MS")
    s_obs = pd.concat([s_obs, pd.Series([5.0] * 11, index=pad_dates)])
    s_mod = pd.concat([s_mod, pd.Series([5.0] * 11, index=pad_dates)])

    l2 = Level2ThresholdMatchingPrecipitation().fit(s_obs, s_mod)
    assert np.isclose(l2.monthly_scaling_factors[1], 0.50)

    # Evaluate on day with P = 1.4 mm >= P_th
    # P_corr = 0.5 * 1.4 = 0.70 mm. It must NOT be truncated to 0!
    eval_dates = pd.to_datetime(["1981-01-05"])
    s_eval = pd.Series([1.4], index=eval_dates)
    s_out = l2.transform(s_eval)

    assert np.isclose(s_out.iloc[0], 0.70)
    assert s_out.iloc[0] > 0.0
    assert s_out.iloc[0] < 1.0


def test_level2_fit_raises_value_error_on_missing_month():
    """Level 2 fit raises ValueError if a month lacks paired calibration days."""
    dates = pd.date_range("1980-01-01", "1980-11-30", freq="D")  # Month 12 missing
    s_obs = pd.Series(np.full(len(dates), 5.0), index=dates)
    s_mod = pd.Series(np.full(len(dates), 5.0), index=dates)

    scaler = Level2ThresholdMatchingPrecipitation()
    with pytest.raises(
        ValueError, match="no paired valid calibration days for month 12"
    ):
        scaler.fit(s_obs, s_mod)


def test_level2_zero_observed_wet_days_threshold_is_model_max_and_factor_zero():
    """D.3: Zero observed wet days sets P_th to max model precipitation and s_m = 0.

    Hand-calculated values for January:
    Obs: 5 days with [0.0, 0.0, 0.0, 0.0, 0.0] -> 0 wet days (>= 1.0 mm), f_wet = 0.0.
    Model: 5 days with [1.0, 2.0, 3.0, 4.0, 5.0] -> max model precipitation = 5.0 mm.
    Under D.3, q = 1 - 0.0 = 1.0 (100th percentile):
        P_th,1 = max(P_mod) = 5.0 mm.
    Censored model: values < 5.0 are 0.0 -> [0.0, 0.0, 0.0, 0.0, 5.0], sum = 5.0 mm.
    Observed sum = 0.0 mm.
    Under D.5 zero-sum rules: s_1 = 0.0 / 5.0 = 0.0.
    Transformation of [2.0, 5.0, 10.0] yields [0.0, 0.0, 0.0].
    """
    jan_dates = pd.date_range("1980-01-01", periods=5, freq="D")
    s_obs_jan = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0], index=jan_dates)
    s_mod_jan = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=jan_dates)

    pad_dates = pd.date_range("1980-02-01", "1980-12-01", freq="MS")
    s_obs = pd.concat([s_obs_jan, pd.Series([5.0] * 11, index=pad_dates)])
    s_mod = pd.concat([s_mod_jan, pd.Series([5.0] * 11, index=pad_dates)])

    l2 = Level2ThresholdMatchingPrecipitation().fit(s_obs, s_mod)

    assert l2.monthly_observed_f_wet[1] == 0.0
    assert np.isclose(l2.monthly_thresholds[1], 5.0)
    assert np.isclose(l2.monthly_scaling_factors[1], 0.0)

    eval_dates = pd.to_datetime(["1981-01-01", "1981-01-02", "1981-01-03"])
    s_eval = pd.Series([2.0, 5.0, 10.0], index=eval_dates)
    s_out = l2.transform(s_eval)
    np.testing.assert_allclose(s_out.values, [0.0, 0.0, 0.0])


def test_synchronous_paired_mask_with_isolated_nans():
    """Fit with NaN in obs/mod (pairwise synchronous deletion)."""
    dates = pd.to_datetime(["1980-01-01", "1980-01-02", "1980-01-03", "1980-01-04"])
    s_obs = pd.Series([10.0, np.nan, 30.0, 20.0], index=dates)
    s_mod = pd.Series([5.0, 20.0, np.nan, 10.0], index=dates)

    pad_dates = pd.date_range("1980-02-01", "1980-12-01", freq="MS")
    s_obs = pd.concat([s_obs, pd.Series([5.0] * 11, index=pad_dates)])
    s_mod = pd.concat([s_mod, pd.Series([5.0] * 11, index=pad_dates)])

    # Retained days in Jan: Day 1 and Day 4.
    # sum_obs = 10.0 + 20.0 = 30.0
    # sum_mod = 5.0 + 10.0 = 15.0
    # Level 1 factor s_1 = 30.0 / 15.0 = 2.0
    l1 = Level1LinearScalingPrecipitation().fit(s_obs, s_mod)
    assert np.isclose(l1.monthly_factors[1], 2.0)

    # Level 2 factor s_1 = 30.0 / 15.0 = 2.0
    l2 = Level2ThresholdMatchingPrecipitation().fit(s_obs, s_mod)
    assert np.isclose(l2.monthly_scaling_factors[1], 2.0)


# ==============================================================================
# Level 3 Tests
# ==============================================================================


def test_level3_circular_leap_window_dec17_and_jan1():
    """366-day circular reference calendar.

    Verification of:
    - Feb 28 is day 59 in both leap (2000) and non-leap (2001).
    - Feb 29 is day 60 in leap year 2000.
    - Mar 1 is day 61 in both leap (2000) and non-leap (2001).
    - Dec 17 is day 352 (LEAP_CUMULATIVE_DAYS[11] + 17 = 335 + 17 = 352).
    - Dec 31 is day 366.
    - Jan 1 is day 1.
    - Jan 16 is day 16.
    - The 31-day window centered on Jan 1 (target DOY = 1) comprises circular
      distance <= 15, matching exactly days >= 352 and days <= 16 (15 + 16 = 31 days).
    """
    assert date_to_leap_doy(pd.Timestamp("2000-02-28")) == 59
    assert date_to_leap_doy(pd.Timestamp("2001-02-28")) == 59

    assert date_to_leap_doy(pd.Timestamp("2000-02-29")) == 60

    assert date_to_leap_doy(pd.Timestamp("2000-03-01")) == 61
    assert date_to_leap_doy(pd.Timestamp("2001-03-01")) == 61

    assert date_to_leap_doy(pd.Timestamp("2000-12-17")) == 352
    assert date_to_leap_doy(pd.Timestamp("2001-12-17")) == 352

    assert date_to_leap_doy(pd.Timestamp("2000-12-31")) == 366
    assert date_to_leap_doy(pd.Timestamp("2001-12-31")) == 366

    assert date_to_leap_doy(pd.Timestamp("2000-01-01")) == 1
    assert date_to_leap_doy(pd.Timestamp("2001-01-01")) == 1

    target_doy = 1
    window_days = [
        d for d in range(1, 367) if leap_doy_circular_distance(d, target_doy) <= 15
    ]
    assert len(window_days) == 31
    expected_set = set(range(352, 367)).union(set(range(1, 17)))
    assert set(window_days) == expected_set


def test_level3_discrete_8_configurations():
    """Verify all 8 discrete configurations instantiate, fit, and transform."""
    dates = pd.date_range("1980-01-01", "1983-12-31", freq="D")
    rng = np.random.default_rng(123)
    obs = pd.Series(rng.exponential(scale=4.0, size=len(dates)) + 0.05, index=dates)
    mod = pd.Series(rng.exponential(scale=3.5, size=len(dates)) + 0.05, index=dates)

    configs = [
        ("monthly", 50, "constant"),
        ("monthly", 50, "linear"),
        ("monthly", 100, "constant"),
        ("monthly", 100, "linear"),
        ("moving_window_31", 50, "constant"),
        ("moving_window_31", 50, "linear"),
        ("moving_window_31", 100, "constant"),
        ("moving_window_31", 100, "linear"),
    ]

    for win, nq, ext in configs:
        model = Level3QuantileMappingPrecipitation(
            window=win, n_quantiles=nq, extrapolation=ext
        )
        model.fit(obs, mod)
        pred = model.transform(mod)

        assert len(pred) == len(mod)
        assert (pred >= 0.0).all()
        assert np.isfinite(pred).all()


def test_level3_requires_at_least_2_wet_days_monthly():
    """Level 3 raises ValueError if any monthly group has < 2 wet days."""
    dates = pd.date_range("1980-01-01", "1980-12-31", freq="D")
    # In January, only 1 wet day in observations
    obs_vals = [
        10.0 if d == pd.Timestamp("1980-01-05") else (0.1 if d.month == 1 else 5.0)
        for d in dates
    ]
    mod_vals = [5.0 for _ in dates]

    s_obs = pd.Series(obs_vals, index=dates)
    s_mod = pd.Series(mod_vals, index=dates)

    model = Level3QuantileMappingPrecipitation(window="monthly")
    with pytest.raises(ValueError, match="fewer than 2 wet days"):
        model.fit(s_obs, s_mod)


def test_level3_requires_at_least_2_wet_days_moving_window():
    """Level 3 raises ValueError if any 31-day circular window has < 2 wet days."""
    dates = pd.date_range("1980-01-01", "1980-12-31", freq="D")
    # All year dry except two days in July
    obs_vals = [
        10.0 if d in [pd.Timestamp("1980-07-01"), pd.Timestamp("1980-07-02")] else 0.0
        for d in dates
    ]
    mod_vals = [5.0 for _ in dates]

    s_obs = pd.Series(obs_vals, index=dates)
    s_mod = pd.Series(mod_vals, index=dates)

    model = Level3QuantileMappingPrecipitation(window="moving_window_31")
    with pytest.raises(ValueError, match="fewer than 2 wet days"):
        model.fit(s_obs, s_mod)


def test_level3_nan_preservation():
    """Level 3 transform preserves NaN values in input model series."""
    dates = pd.date_range("1980-01-01", "1981-12-31", freq="D")
    rng = np.random.default_rng(42)
    s_obs = pd.Series(rng.exponential(scale=5.0, size=len(dates)) + 0.1, index=dates)
    s_mod = pd.Series(rng.exponential(scale=4.0, size=len(dates)) + 0.1, index=dates)

    model = Level3QuantileMappingPrecipitation().fit(s_obs, s_mod)

    eval_dates = pd.date_range("1982-01-01", periods=5, freq="D")
    s_eval = pd.Series([2.0, np.nan, 5.0, np.nan, 10.0], index=eval_dates)
    s_out = model.transform(s_eval)

    assert not np.isnan(s_out.iloc[0])
    assert np.isnan(s_out.iloc[1])
    assert not np.isnan(s_out.iloc[2])
    assert np.isnan(s_out.iloc[3])
    assert not np.isnan(s_out.iloc[4])
    assert (s_out.dropna() >= 0.0).all()
    assert np.isfinite(s_out.dropna()).all()


def test_level3_defensive_clipping_below_zero():
    """Defensive zero floor triggers when nodes have declining upper quantiles.

    Artificial nodes for month 1 with extrapolation='linear':
    Q_mod: [10.0, 20.0, 30.0]
    Q_obs: [15.0, 25.0, 5.0] (declining at top: 25.0 -> 5.0)
    Top slope: s = (5.0 - 25.0) / (30.0 - 20.0) = -20.0 / 10.0 = -2.0.
    For P = 50.0 > 30.0:
    P_corr = 5.0 + (-2.0) * (50.0 - 30.0) = 5.0 - 40.0 = -35.0 mm.
    The defensive clamp clamps P_corr to 0.0 and increments n_clipped_below_zero.
    """
    m = Level3QuantileMappingPrecipitation(window="monthly", extrapolation="linear")
    m.level2_thresholds = {1: 1.0}
    q_mod = np.array([10.0, 20.0, 30.0])
    q_obs = np.array([15.0, 25.0, 5.0])
    r = q_obs / q_mod
    m.quantile_transfer_nodes = {1: (q_mod, q_obs, r)}

    dates = pd.to_datetime(["1981-01-15"])
    s_in = pd.Series([50.0], index=dates)
    s_out = m.transform(s_in)

    assert s_out.iloc[0] == 0.0
    assert m.n_clipped_below_zero > 0
    assert m.n_clipped_below_zero == 1


def test_level3_level2_thresholds_match_level2_monthly_thresholds():
    """Level 3 level2_thresholds match Level 2 monthly_thresholds on identical data."""
    dates = pd.date_range("1980-01-01", "1981-12-31", freq="D")
    rng = np.random.default_rng(99)
    s_obs = pd.Series(rng.exponential(scale=5.0, size=len(dates)) + 0.1, index=dates)
    s_mod = pd.Series(rng.exponential(scale=4.0, size=len(dates)) + 0.1, index=dates)

    l2 = Level2ThresholdMatchingPrecipitation().fit(s_obs, s_mod)
    l3 = Level3QuantileMappingPrecipitation().fit(s_obs, s_mod)

    for m in range(1, 13):
        assert np.isclose(l3.level2_thresholds[m], l2.monthly_thresholds[m])


def test_hand_calculated_quantile_node_levels():
    """Quantile levels of nodes: 99 nodes for 100 bins, 49 for 50 bins."""
    dates = pd.date_range("1980-01-01", "1980-12-31", freq="D")
    rng = np.random.default_rng(42)
    obs = pd.Series(rng.uniform(1.0, 20.0, size=len(dates)), index=dates)
    mod = pd.Series(rng.uniform(1.0, 20.0, size=len(dates)), index=dates)

    m100 = Level3QuantileMappingPrecipitation(n_quantiles=100).fit(obs, mod)
    m50 = Level3QuantileMappingPrecipitation(n_quantiles=50).fit(obs, mod)

    # 100 bins: exactly 99 nodes
    q_mod_100, q_obs_100, r_100 = m100.quantile_transfer_nodes[1]
    assert len(q_mod_100) == 99
    assert len(q_obs_100) == 99
    assert len(r_100) == 99

    # 50 bins: exactly 49 nodes
    q_mod_50, q_obs_50, r_50 = m50.quantile_transfer_nodes[1]
    assert len(q_mod_50) == 49
    assert len(q_obs_50) == 49
    assert len(r_50) == 49

    expected_probs_100 = np.linspace(0.01, 0.99, 99)
    expected_probs_50 = np.linspace(0.02, 0.98, 49)
    assert np.isclose(expected_probs_100[0], 0.01)
    assert np.isclose(expected_probs_100[-1], 0.99)
    assert np.isclose(expected_probs_50[0], 0.02)
    assert np.isclose(expected_probs_50[-1], 0.98)


def test_hand_calculated_interior_node_interpolation():
    """Piecewise linear interpolation between nodes (np.interp)."""
    m = Level3QuantileMappingPrecipitation(window="monthly", n_quantiles=50)
    m.level2_thresholds = {1: 1.0}

    q_mod = np.array([10.0, 20.0, 30.0])
    q_obs = np.array([14.0, 26.0, 38.0])
    r = q_obs / q_mod
    m.quantile_transfer_nodes = {1: (q_mod, q_obs, r)}

    # P = 15.0 -> 14.0 + 0.5 * 12.0 = 20.0
    dates = pd.to_datetime(["1981-01-15"])
    s_in = pd.Series([15.0], index=dates)
    s_out = m.transform(s_in)
    assert np.isclose(s_out.iloc[0], 20.0)

    # P = 27.5 -> 26.0 + 0.75 * 12.0 = 35.0
    s_in2 = pd.Series([27.5], index=dates)
    s_out2 = m.transform(s_in2)
    assert np.isclose(s_out2.iloc[0], 35.0)


def test_hand_calculated_below_first_node_interpolation():
    """Hand-calculated values below first node: P_corr = r_1 * P."""
    m = Level3QuantileMappingPrecipitation(window="monthly", n_quantiles=50)
    m.level2_thresholds = {1: 1.0}

    q_mod = np.array([10.0, 20.0])
    q_obs = np.array([6.0, 16.0])
    r = q_obs / q_mod
    m.quantile_transfer_nodes = {1: (q_mod, q_obs, r)}

    # P = 5.0 (under first node 10.0, but >= threshold 1.0) -> (6.0 / 10.0) * 5.0 = 3.0
    dates = pd.to_datetime(["1981-01-10"])
    s_in = pd.Series([5.0], index=dates)
    s_out = m.transform(s_in)
    assert np.isclose(s_out.iloc[0], 3.0)


def test_hand_calculated_constant_vs_linear_with_transfer_extension():
    """Constant vs. Linear with extension s = Delta Q_obs / Delta Q_mod."""
    q_mod = np.array([10.0, 30.0, 50.0])
    q_obs = np.array([10.0, 40.0, 50.0])
    r = q_obs / q_mod

    m_const = Level3QuantileMappingPrecipitation(
        window="monthly", extrapolation="constant"
    )
    m_const.level2_thresholds = {1: 1.0}
    m_const.quantile_transfer_nodes = {1: (q_mod, q_obs, r)}

    m_lin = Level3QuantileMappingPrecipitation(window="monthly", extrapolation="linear")
    m_lin.level2_thresholds = {1: 1.0}
    m_lin.quantile_transfer_nodes = {1: (q_mod, q_obs, r)}

    # Evaluate at P = 70.0 (above Q_mod,top = 50.0)
    # Constant: P_corr = r_top * P = 1.0 * 70.0 = 70.0
    # Linear:   P_corr = Q_obs,top + s * (P - Q_mod,top) = 50.0 + 0.5 * 20.0 = 60.0
    dates = pd.to_datetime(["1981-01-20"])
    s_in = pd.Series([70.0], index=dates)

    out_const = m_const.transform(s_in).iloc[0]
    out_lin = m_lin.transform(s_in).iloc[0]

    assert np.isclose(out_const, 70.0)
    assert np.isclose(out_lin, 60.0)
    assert m_lin.n_clipped_below_zero == 0

    # Test denominator <= 1e-9 fallback to r_top in linear
    q_mod_flat = np.array([10.0, 50.0, 50.0])
    q_obs_flat = np.array([10.0, 40.0, 50.0])
    r_flat = np.where(q_mod_flat > 0, q_obs_flat / q_mod_flat, 1.0)
    m_flat = Level3QuantileMappingPrecipitation(
        window="monthly", extrapolation="linear"
    )
    m_flat.level2_thresholds = {1: 1.0}
    m_flat.quantile_transfer_nodes = {1: (q_mod_flat, q_obs_flat, r_flat)}

    out_flat = m_flat.transform(s_in).iloc[0]
    assert np.isclose(out_flat, 70.0)


def test_level3_extrapolation_with_fitted_nodes_above_calibration_maximum():
    """Extrapolation with nodes fitted via fit() on events exceeding calibration max.

    Verifies for both constant and linear extrapolation:
    - Output is strictly non-negative (>= 0).
    - Constant extrapolation equals r_top * P.
    - n_clipped_below_zero evaluates strictly to 0 in both.
    """
    dates = pd.date_range("1980-01-01", "1981-12-31", freq="D")
    rng = np.random.default_rng(101)
    s_obs = pd.Series(rng.exponential(scale=5.0, size=len(dates)) + 0.1, index=dates)
    s_mod = pd.Series(rng.exponential(scale=4.0, size=len(dates)) + 0.1, index=dates)

    m_const = Level3QuantileMappingPrecipitation(
        window="monthly", n_quantiles=50, extrapolation="constant"
    ).fit(s_obs, s_mod)

    m_lin = Level3QuantileMappingPrecipitation(
        window="monthly", n_quantiles=50, extrapolation="linear"
    ).fit(s_obs, s_mod)

    # Extreme storm event in January well above maximum calibration model value
    cal_max = s_mod.max()
    extreme_val = cal_max * 2.5 + 50.0
    eval_dates = pd.to_datetime(["1982-01-15"])
    s_eval = pd.Series([extreme_val], index=eval_dates)

    out_const = m_const.transform(s_eval).iloc[0]
    out_lin = m_lin.transform(s_eval).iloc[0]

    # Verify constant equals r_top * P
    q_mod, q_obs, r = m_const.quantile_transfer_nodes[1]
    r_top = r[-1]
    assert np.isclose(out_const, r_top * extreme_val)

    assert out_const >= 0.0
    assert out_lin >= 0.0
    assert m_const.n_clipped_below_zero == 0
    assert m_lin.n_clipped_below_zero == 0


# ==============================================================================
# Group Assignment and Threshold Mapping Verification Tests
# ==============================================================================


def test_group_assignment_moving_window_31_and_threshold_month():
    """moving_window_31 assigns DOY group and month-specific threshold."""
    assert date_to_leap_doy(pd.Timestamp("1981-01-01")) == 1
    assert date_to_leap_doy(pd.Timestamp("1981-07-15")) == 197
    assert date_to_leap_doy(pd.Timestamp("1981-12-17")) == 352
    assert date_to_leap_doy(pd.Timestamp("1981-01-16")) == 16
    assert leap_doy_circular_distance(1, 352) == 15
    assert leap_doy_circular_distance(1, 16) == 15

    m = Level3QuantileMappingPrecipitation(window="moving_window_31", n_quantiles=50)

    # Thresholds are strictly monthly (indexed by calendar month 1..12)
    m.level2_thresholds = {
        1: 4.0,
        7: 2.0,
        12: 1.0,
    }

    q_mod_1 = np.array([10.0, 20.0])
    q_obs_1 = np.array([15.0, 30.0])
    r_1 = q_obs_1 / q_mod_1

    q_mod_197 = np.array([10.0, 20.0])
    q_obs_197 = np.array([5.0, 10.0])
    r_197 = q_obs_197 / q_mod_197

    m.quantile_transfer_nodes = {
        1: (q_mod_1, q_obs_1, r_1),
        197: (q_mod_197, q_obs_197, r_197),
    }

    dates = pd.to_datetime(["1981-01-01", "1981-07-15"])
    s_in = pd.Series([10.0, 10.0], index=dates)
    s_out = m.transform(s_in)

    assert np.isclose(s_out.iloc[0], 15.0)
    assert np.isclose(s_out.iloc[1], 5.0)

    # Threshold check: Jan 1 with P=2.5 mm < P_th,1 (4.0 mm) -> truncated to 0.0
    s_sub_th = pd.Series([2.5], index=pd.to_datetime(["1981-01-01"]))
    s_out_th = m.transform(s_sub_th)
    assert s_out_th.iloc[0] == 0.0


def test_group_assignment_monthly_and_threshold_month():
    """monthly window assigns calendar month group and month threshold."""
    m = Level3QuantileMappingPrecipitation(window="monthly", n_quantiles=50)

    m.level2_thresholds = {1: 3.0, 7: 1.0}

    q_mod_1 = np.array([10.0, 20.0])
    q_obs_1 = np.array([20.0, 40.0])
    r_1 = q_obs_1 / q_mod_1

    q_mod_7 = np.array([10.0, 20.0])
    q_obs_7 = np.array([5.0, 10.0])
    r_7 = q_obs_7 / q_mod_7

    m.quantile_transfer_nodes = {
        1: (q_mod_1, q_obs_1, r_1),
        7: (q_mod_7, q_obs_7, r_7),
    }

    dates = pd.to_datetime(["1981-01-10", "1981-07-10"])
    s_in = pd.Series([10.0, 10.0], index=dates)
    s_out = m.transform(s_in)

    assert np.isclose(s_out.iloc[0], 20.0)
    assert np.isclose(s_out.iloc[1], 5.0)

    # Threshold check: P = 2.0 mm
    # Jan: 2.0 < 3.0 -> 0.0
    # Jul: 2.0 >= 1.0 -> scaled: (5.0 / 10.0) * 2.0 = 1.0
    s_in_th = pd.Series([2.0, 2.0], index=dates)
    s_out_th = m.transform(s_in_th)
    assert s_out_th.iloc[0] == 0.0
    assert np.isclose(s_out_th.iloc[1], 1.0)


# ==============================================================================
# Cross-Validation Tests
# ==============================================================================


def test_cv_nested_annual_folds_skips_years_without_valid_pairs():
    """Nested annual block CV builds folds strictly for valid calendar years."""
    dates_80 = pd.date_range("1980-01-01", "1980-12-31", freq="D")
    dates_81 = pd.date_range("1981-01-01", "1981-12-31", freq="D")
    dates_82 = pd.date_range("1982-01-01", "1982-12-31", freq="D")

    rng = np.random.default_rng(42)
    s_obs = pd.concat(
        [
            pd.Series(rng.uniform(1.0, 10.0, size=len(dates_80)), index=dates_80),
            pd.Series(np.nan, index=dates_81),
            pd.Series(rng.uniform(1.0, 10.0, size=len(dates_82)), index=dates_82),
        ]
    )
    s_mod = pd.concat(
        [
            pd.Series(rng.uniform(1.0, 10.0, size=len(dates_80)), index=dates_80),
            pd.Series(rng.uniform(1.0, 10.0, size=len(dates_81)), index=dates_81),
            pd.Series(rng.uniform(1.0, 10.0, size=len(dates_82)), index=dates_82),
        ]
    )

    winner, results, best = tune_level3_precipitation_hyperparameters(s_obs, s_mod)
    assert len(best.fold_tuning_losses) == 2


def test_cv_overall_loss_is_concatenated_not_mean_of_folds():
    """Overall CV loss is evaluated on concatenated series."""
    from tropicor.core.protocol import precipitation_tuning_loss

    dates_80 = pd.date_range("1980-01-01", "1980-12-31", freq="D")
    dates_81 = pd.date_range("1981-01-01", "1981-12-31", freq="D")
    rng = np.random.default_rng(123)

    all_dates = dates_80.union(dates_81)
    s_obs = pd.Series(rng.uniform(1.0, 25.0, size=len(all_dates)), index=all_dates)
    s_mod = pd.Series(rng.uniform(1.0, 20.0, size=len(all_dates)), index=all_dates)

    winner, results, best = tune_level3_precipitation_hyperparameters(s_obs, s_mod)

    y_oos = pd.Series(np.nan, index=s_mod.index)
    for yr in [1980, 1981]:
        val_mask = s_mod.index.year == yr
        train_mask = s_mod.index.year != yr
        m = Level3QuantileMappingPrecipitation(
            window=best.window,
            n_quantiles=best.n_quantiles,
            extrapolation=best.extrapolation,
        ).fit(s_obs[train_mask], s_mod[train_mask])
        y_oos.loc[val_mask] = m.transform(s_mod[val_mask])

    expected_overall = precipitation_tuning_loss(s_obs, y_oos)
    assert np.isclose(best.overall_tuning_loss, expected_overall)


# ==============================================================================
# Parsimony Tie-Breaker (select_best_level3_result) Direct Tests
# ==============================================================================


def test_select_best_level3_result_tie_within_5e7_picks_more_parsimonious():
    """Loss diff 5e-7 (<= 1e-6) ties and selects more parsimonious configuration."""
    r_lin = Level3TuningResult(
        config_id=1,
        window="monthly",
        n_quantiles=50,
        extrapolation="linear",
        overall_tuning_loss=5.0000000,
        fold_tuning_losses=[5.0],
        mean_fold_loss=5.0,
        std_fold_loss=0.0,
        min_n_wet_per_group=10,
        n_clipped_below_zero=0,
    )
    r_const = Level3TuningResult(
        config_id=0,
        window="monthly",
        n_quantiles=50,
        extrapolation="constant",
        overall_tuning_loss=5.0000005,
        fold_tuning_losses=[5.0],
        mean_fold_loss=5.0,
        std_fold_loss=0.0,
        min_n_wet_per_group=10,
        n_clipped_below_zero=0,
    )

    best = select_best_level3_result([r_lin, r_const], tol=1e-6)
    # Difference is 5e-7 <= 1e-6 -> tied! constant beats linear
    assert best.config_id == 0
    assert best.extrapolation == "constant"


def test_select_best_level3_result_difference_2e6_picks_lower_loss():
    """Loss difference 2e-6 (> 1e-6) picks lower loss even if less parsimonious."""
    r_lin = Level3TuningResult(
        config_id=1,
        window="monthly",
        n_quantiles=50,
        extrapolation="linear",
        overall_tuning_loss=5.0000000,
        fold_tuning_losses=[5.0],
        mean_fold_loss=5.0,
        std_fold_loss=0.0,
        min_n_wet_per_group=10,
        n_clipped_below_zero=0,
    )
    r_const = Level3TuningResult(
        config_id=0,
        window="monthly",
        n_quantiles=50,
        extrapolation="constant",
        overall_tuning_loss=5.0000020,
        fold_tuning_losses=[5.0],
        mean_fold_loss=5.0,
        std_fold_loss=0.0,
        min_n_wet_per_group=10,
        n_clipped_below_zero=0,
    )

    best = select_best_level3_result([r_lin, r_const], tol=1e-6)
    # Difference is 2e-6 > 1e-6 -> not tied! lower loss wins
    assert best.config_id == 1
    assert best.extrapolation == "linear"


def test_select_best_level3_result_excludes_nan_loss():
    """Non-finite losses are excluded from eligibility."""
    r_nan = Level3TuningResult(
        config_id=0,
        window="monthly",
        n_quantiles=50,
        extrapolation="constant",
        overall_tuning_loss=float("nan"),
        fold_tuning_losses=[float("nan")],
        mean_fold_loss=float("nan"),
        std_fold_loss=float("nan"),
        min_n_wet_per_group=10,
        n_clipped_below_zero=0,
    )
    r_valid = Level3TuningResult(
        config_id=1,
        window="monthly",
        n_quantiles=50,
        extrapolation="linear",
        overall_tuning_loss=10.0,
        fold_tuning_losses=[10.0],
        mean_fold_loss=10.0,
        std_fold_loss=0.0,
        min_n_wet_per_group=10,
        n_clipped_below_zero=0,
    )

    best = select_best_level3_result([r_nan, r_valid])
    assert best.config_id == 1


def test_select_best_level3_result_all_nan_raises_value_error():
    """Raises ValueError if all candidates have non-finite tuning losses."""
    r_nan = Level3TuningResult(
        config_id=0,
        window="monthly",
        n_quantiles=50,
        extrapolation="constant",
        overall_tuning_loss=float("nan"),
        fold_tuning_losses=[float("nan")],
        mean_fold_loss=float("nan"),
        std_fold_loss=float("nan"),
        min_n_wet_per_group=10,
        n_clipped_below_zero=0,
    )
    with pytest.raises(ValueError, match="No finite tuning losses"):
        select_best_level3_result([r_nan])


def test_select_best_level3_result_full_parsimony_hierarchy():
    """Parsimony hierarchy: window > n_quantiles > extrapolation > config_id."""
    tied_loss = 5.0
    r_m_50_const = Level3TuningResult(
        0, "monthly", 50, "constant", tied_loss, [5.0], 5.0, 0.0, 10, 0
    )
    r_m_50_lin = Level3TuningResult(
        1, "monthly", 50, "linear", tied_loss, [5.0], 5.0, 0.0, 10, 0
    )
    r_m_100_const = Level3TuningResult(
        2, "monthly", 100, "constant", tied_loss, [5.0], 5.0, 0.0, 10, 0
    )
    r_w_50_const = Level3TuningResult(
        4, "moving_window_31", 50, "constant", tied_loss, [5.0], 5.0, 0.0, 10, 0
    )

    # 1. monthly beats moving window
    assert select_best_level3_result([r_w_50_const, r_m_50_const]).config_id == 0
    # 2. 50 bins beats 100 bins
    assert select_best_level3_result([r_m_100_const, r_m_50_const]).config_id == 0
    # 3. constant beats linear
    assert select_best_level3_result([r_m_50_lin, r_m_50_const]).config_id == 0


def test_tune_level3_hyperparameters_parsimony_tie_breaker(
    monkeypatch: pytest.MonkeyPatch,
):
    """Tuning tie-breaker selects config_id 0 when all losses tie under monkeypatch."""
    dates = pd.date_range("1980-01-01", "1982-12-31", freq="D")
    rng = np.random.default_rng(42)
    s_obs = pd.Series(rng.uniform(1.0, 20.0, size=len(dates)), index=dates)
    s_mod = pd.Series(rng.uniform(1.0, 20.0, size=len(dates)), index=dates)

    import tropicor.downscale.precipitation as p_mod

    monkeypatch.setattr(p_mod, "precipitation_tuning_loss", lambda o, m: 1.0)

    winner, results, best_res = tune_level3_precipitation_hyperparameters(s_obs, s_mod)

    assert best_res.config_id == 0
    assert best_res.window == "monthly"
    assert best_res.n_quantiles == 50
    assert best_res.extrapolation == "constant"


# ==============================================================================
# Units Validation Tests
# ==============================================================================


def test_units_validation_raises_value_error_if_mean_under_0_01_mm():
    """Series mean < 0.01 mm raises ValueError (detecting meters instead of mm)."""
    dates = pd.date_range("1980-01-01", "1980-12-31", freq="D")
    s_obs = pd.Series(np.full(len(dates), 5.0), index=dates)
    s_mod_meters = pd.Series(np.full(len(dates), 0.002), index=dates)

    scaler = Level1LinearScalingPrecipitation()
    with pytest.raises(ValueError, match=r"< 0\.01 mm"):
        scaler.fit(s_obs, s_mod_meters)


# ==============================================================================
# Calibration Loader Tests
# ==============================================================================


def test_load_calibration_precipitation_pair_synthetic_csv(tmp_path: Path):
    """Synthetic CSV extending beyond calibration affirms immediate discarding."""
    full_dates = pd.date_range("1970-01-01", "2010-12-31", freq="D")
    rng = np.random.default_rng(77)
    df_obs = pd.DataFrame(
        {
            "CodigoEstacion": ["21205710"] * len(full_dates),
            "Fecha": full_dates.strftime("%Y-%m-%d"),
            "Valor": rng.uniform(0.0, 10.0, size=len(full_dates)),
        }
    )
    obs_file = tmp_path / "synthetic_obs.csv"
    df_obs.to_csv(obs_file, index=False)

    df_mod = pd.DataFrame(
        {
            "Fecha": full_dates.strftime("%Y-%m-%d"),
            "Valor": rng.uniform(0.0, 10.0, size=len(full_dates)),
        }
    )
    mod_file = tmp_path / "synthetic_mod.csv"
    df_mod.to_csv(mod_file, index=False)

    s_obs, s_mod = load_calibration_precipitation_pair(
        station_id="21205710",
        dhime_path=obs_file,
        era5_path_or_dir=mod_file,
        cal_range=(1975, 1994),
    )

    assert s_obs.index.min() == pd.Timestamp("1975-01-01")
    assert s_obs.index.max() == pd.Timestamp("1994-12-31")
    assert s_mod.index.min() == pd.Timestamp("1975-01-01")
    assert s_mod.index.max() == pd.Timestamp("1994-12-31")
    assert len(s_obs) == 7305


def test_load_calibration_precipitation_pair_known_station_no_cal_range_synthetic(
    tmp_path: Path,
):
    """Station '23215030' without cal_range returns 1975-1987 (4748 days)."""
    full_dates = pd.date_range("1970-01-01", "2010-12-31", freq="D")
    rng = np.random.default_rng(88)
    df_obs = pd.DataFrame(
        {
            "CodigoEstacion": ["23215030"] * len(full_dates),
            "Fecha": full_dates.strftime("%Y-%m-%d"),
            "Valor": rng.uniform(0.0, 10.0, size=len(full_dates)),
        }
    )
    obs_file = tmp_path / "obs_23215030.csv"
    df_obs.to_csv(obs_file, index=False)

    df_mod = pd.DataFrame(
        {
            "Fecha": full_dates.strftime("%Y-%m-%d"),
            "Valor": rng.uniform(0.0, 10.0, size=len(full_dates)),
        }
    )
    mod_file = tmp_path / "mod_23215030.csv"
    df_mod.to_csv(mod_file, index=False)

    s_obs, s_mod = load_calibration_precipitation_pair(
        station_id="23215030",
        dhime_path=obs_file,
        era5_path_or_dir=mod_file,
        cal_range=None,
    )

    assert s_obs.index.min() == pd.Timestamp("1975-01-01")
    assert s_obs.index.max() == pd.Timestamp("1987-12-31")
    assert s_mod.index.min() == pd.Timestamp("1975-01-01")
    assert s_mod.index.max() == pd.Timestamp("1987-12-31")
    assert len(s_obs) == 4748
    assert len(s_mod) == 4748


def test_load_calibration_pair_unknown_station_no_cal_range_raises(
    tmp_path: Path,
):
    """Unknown station without cal_range raises explicit ValueError without fallback."""
    full_dates = pd.date_range("1975-01-01", "1980-12-31", freq="D")
    df_obs = pd.DataFrame(
        {
            "CodigoEstacion": ["99999999"] * len(full_dates),
            "Fecha": full_dates.strftime("%Y-%m-%d"),
            "Valor": np.full(len(full_dates), 5.0),
        }
    )
    obs_file = tmp_path / "obs_unknown.csv"
    df_obs.to_csv(obs_file, index=False)

    df_mod = pd.DataFrame(
        {
            "Fecha": full_dates.strftime("%Y-%m-%d"),
            "Valor": np.full(len(full_dates), 5.0),
        }
    )
    mod_file = tmp_path / "mod_unknown.csv"
    df_mod.to_csv(mod_file, index=False)

    err_msg = (
        "no pre-specified calibration partition for station 99999999; "
        "pass cal_range explicitly"
    )
    with pytest.raises(ValueError, match=err_msg):
        load_calibration_precipitation_pair(
            station_id="99999999",
            dhime_path=obs_file,
            era5_path_or_dir=mod_file,
            cal_range=None,
        )


@pytest.mark.skipif(
    not Path("local/newFiles/DiaPluviometrico_JardinBotanico_21205710.csv").exists()
    or not Path("local/era5_hourly/21205710").exists(),
    reason="Real data files in local/ directory not available in current environment",
)
def test_load_calibration_precipitation_pair_real_data():
    """Optional test on real calibration files for Jardín Botánico."""
    s_obs, s_mod = load_calibration_precipitation_pair("21205710")
    assert s_obs.index.min() == pd.Timestamp("1975-01-01")
    assert s_obs.index.max() == pd.Timestamp("1994-12-31")
    assert s_mod.index.min() == pd.Timestamp("1975-01-01")
    assert s_mod.index.max() == pd.Timestamp("1994-12-31")
