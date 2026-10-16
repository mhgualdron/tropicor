"""Comprehensive unit tests for the protocol verification engine.

Verifies exact adherence to Protocol v1.0 specifications using synthetic data,
including decisive/descriptive metrics, annual block bootstrap uncertainty,
symmetric three-tier mapping, 70% threshold rounding, physical gates,
cross-validation partitioners, and validation access audit guard.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tropicor.core.protocol import (
    BootstrapConfidenceInterval,
    CDDDiagnostic,
    GlobalOutcome,
    MetricOutcome,
    apply_paired_mask,
    apply_universal_crossing_rule,
    apply_weaker_of_two_concordance,
    check_precipitation_physical_gate,
    check_temperature_physical_gate,
    compute_annual_block_bootstrap_precipitation,
    compute_annual_block_bootstrap_temperature,
    compute_annual_cdd_max,
    compute_daily_pearson,
    compute_precipitation_contingency,
    compute_precipitation_decisive_metrics,
    compute_required_passes,
    compute_rx1day,
    compute_rx5day,
    compute_sdii,
    compute_temperature_decisive_metrics,
    compute_wet_day_frequency,
    compute_wet_day_percentiles,
    contiguous_5year_block_cv,
    evaluate_precipitation_scorecard,
    evaluate_symmetric_metric_outcome,
    evaluate_temperature_scorecard,
    get_benchmark_partitions,
    nested_annual_block_cv,
    precipitation_tuning_loss,
    split_calibration_validation,
    temperature_tuning_loss,
)

# ==============================================================================
# 1. Masking & Universal Crossing Rule Tests
# ==============================================================================


def test_paired_mask_with_nans_and_faltante_flag() -> None:
    """Verify synchronous paired mask excludes NaNs and Faltante (Section 2.4)."""
    dates = pd.date_range("2000-01-01", periods=6, freq="D")
    obs = pd.Series([10.0, np.nan, 5.0, 12.0, 0.0, 8.0], index=dates)
    mod = pd.Series([9.5, 3.2, 5.1, 11.0, 0.2, 7.9], index=dates)
    flags = pd.Series(
        [
            "Definitivo",
            "Definitivo",
            "Faltante",
            "Definitivo",
            "Preliminar",
            "Definitivo",
        ],
        index=dates,
    )

    clean_obs, clean_mod = apply_paired_mask(obs, mod, quality_flags=flags)
    assert len(clean_obs) == 4
    assert len(clean_mod) == 4
    assert list(clean_obs.values) == [10.0, 12.0, 0.0, 8.0]
    assert list(clean_mod.values) == [9.5, 11.0, 0.2, 7.9]
    assert (clean_obs.index == clean_mod.index).all()


def test_universal_crossing_rule_and_alert() -> None:
    """Verify midpoint +/- 0.025 C rule, DTR=0.05, counts, alert (Section 3.5)."""
    dates = pd.date_range("2000-01-01", periods=100, freq="D")
    tmax = pd.Series(np.full(100, 25.0), index=dates)
    tmin = pd.Series(np.full(100, 15.0), index=dates)

    tmax.iloc[10] = 18.0
    tmin.iloc[10] = 22.0  # crossing: midpoint 20.0
    tmax.iloc[20] = 14.0
    tmin.iloc[20] = 16.0  # crossing: midpoint 15.0

    assert not check_temperature_physical_gate(tmax, tmin)
    corr_tmax, corr_tmin, diag = apply_universal_crossing_rule(tmax, tmin)

    assert diag.n_crossings == 2
    assert diag.n_total_days == 100
    assert pytest.approx(diag.fraction_crossings) == 0.02
    assert diag.alert_triggered is True

    # Midpoint checks
    assert pytest.approx(corr_tmax.iloc[10]) == 20.025
    assert pytest.approx(corr_tmin.iloc[10]) == 19.975
    assert pytest.approx(corr_tmax.iloc[10] - corr_tmin.iloc[10]) == 0.05
    assert check_temperature_physical_gate(corr_tmax, corr_tmin)


def test_physical_consistency_gates() -> None:
    """Verify physical consistency gates for temperature and precipitation."""
    dates = pd.date_range("2000-01-01", periods=5, freq="D")

    tmax = pd.Series([20.0, 21.0, 22.0, 23.0, 24.0], index=dates)
    tmin = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0], index=dates)
    assert check_temperature_physical_gate(tmax, tmin) is True

    tmin_viol = pd.Series([10.0, 11.0, 25.0, 13.0, 14.0], index=dates)
    assert check_temperature_physical_gate(tmax, tmin_viol) is False

    p_valid = pd.Series([0.0, 1.2, 0.0, 15.4, 0.5], index=dates)
    assert check_precipitation_physical_gate(p_valid) is True

    p_neg = pd.Series([0.0, 1.2, -0.1, 15.4, 0.5], index=dates)
    assert check_precipitation_physical_gate(p_neg) is False

    p_inf = pd.Series([0.0, 1.2, np.inf, 15.4, 0.5], index=dates)
    assert check_precipitation_physical_gate(p_inf) is False


# ==============================================================================
# 2. Decisive & Descriptive Metrics Hand Calculations
# ==============================================================================


def test_precipitation_decisive_metrics_hand_calculated() -> None:
    """Verify precipitation decisive metrics with exact hand calculations."""
    dates = pd.date_range("2000-01-01", "2000-12-31", freq="D")
    n = len(dates)

    obs_vals = np.zeros(n)
    obs_vals[:100] = np.arange(1.0, 101.0)
    obs = pd.Series(obs_vals, index=dates)

    mod_vals = np.zeros(n)
    mod_vals[:120] = np.arange(1.0, 121.0)
    mod = pd.Series(mod_vals, index=dates)

    raw_vals = np.zeros(n)
    raw_vals[:150] = np.linspace(1.0, 100.0, 150)
    raw = pd.Series(raw_vals, index=dates)

    f_obs = compute_wet_day_frequency(obs)
    f_mod = compute_wet_day_frequency(mod)
    assert pytest.approx(f_obs) == 100.0 / 366.0
    assert pytest.approx(f_mod) == 120.0 / 366.0

    p_obs = compute_wet_day_percentiles(obs, percentiles=(95.0, 99.0))
    p_mod = compute_wet_day_percentiles(mod, percentiles=(95.0, 99.0))
    assert pytest.approx(p_obs[95.0], rel=1e-2) == 95.05
    assert pytest.approx(p_mod[95.0], rel=1e-2) == 114.05

    metrics = compute_precipitation_decisive_metrics(obs, mod, raw=raw)

    assert pytest.approx(metrics["f_wet_total_pbias"]) == 20.0
    assert pytest.approx(metrics["pbias_monthly"], rel=1e-3) == 43.762376

    sum_raw = float(raw.sum())
    pbias_raw = 100.0 * (sum_raw - 5050.0) / 5050.0
    expected_delta = abs(pbias_raw) - abs(metrics["pbias_monthly"])
    assert pytest.approx(metrics["delta_pbias_volume"]) == expected_delta


def test_cdd_completeness_rules_hand_calculated() -> None:
    """Verify CDD rules: year <330d excluded, crossing streak discarded, boundary."""
    dates_yr1 = pd.date_range("2001-01-01", "2001-12-31", freq="D")  # 365 days
    dates_yr2 = pd.date_range("2002-01-01", "2002-12-31", freq="D")  # 365 days
    dates_yr3 = pd.date_range("2003-01-01", "2003-01-10", freq="D")  # 10 days
    full_dates = dates_yr1.union(dates_yr2).union(dates_yr3)

    # Initialize all with rain (10.0 mm) so we control dry spells precisely
    obs = pd.Series(10.0, index=full_dates)
    mod = pd.Series(10.0, index=full_dates)

    # 1. Year 1: introduce 50 NaNs -> 315 valid days (< 330) -> Year 1 EXCLUDED
    obs.iloc[:50] = np.nan

    # 2. Year 2: March has 8 NaNs out of 31 days (25.8% > 20% missing) -> March EXCLUDED
    mar_mask = (full_dates.year == 2002) & (full_dates.month == 3)
    mar_dates = full_dates[mar_mask][:8]
    obs.loc[mar_dates] = np.nan
    mod.loc[mar_dates] = np.nan

    # Streak A (racha adyacente): Jan 10 to Jan 25 = 16 days dry in obs
    # For mod, Jan 8 to Jan 25 = 18 days dry
    jan_streak_obs = (
        (full_dates.year == 2002)
        & (full_dates.month == 1)
        & (full_dates.day >= 10)
        & (full_dates.day <= 25)
    )
    obs.loc[jan_streak_obs] = 0.0

    jan_streak_mod = (
        (full_dates.year == 2002)
        & (full_dates.month == 1)
        & (full_dates.day >= 8)
        & (full_dates.day <= 25)
    )
    mod.loc[jan_streak_mod] = 0.0

    # Streak B (racha que cruza mes excluido): Feb 20 to March 5 dry (14 days)
    # Touches March (an excluded month) -> MUST BE DISCARDED (not counted as 9)
    feb_mar_streak = (full_dates.year == 2002) & (
        ((full_dates.month == 2) & (full_dates.day >= 20))
        | ((full_dates.month == 3) & (full_dates.day <= 5))
    )
    obs.loc[feb_mar_streak] = 0.0
    mod.loc[feb_mar_streak] = 0.0

    # Streak C (racha en la frontera de año): Dec 20 to Dec 31 dry in Year 2 (12 days)
    # and continues Jan 1 to Jan 5 in Year 3.
    # Must be evaluated strictly within the calendar year (length = 12 for Year 2)
    dec_jan_streak = (
        (full_dates.year == 2002) & (full_dates.month == 12) & (full_dates.day >= 20)
    ) | ((full_dates.year == 2003) & (full_dates.month == 1) & (full_dates.day <= 5))
    obs.loc[dec_jan_streak] = 0.0
    mod.loc[dec_jan_streak] = 0.0

    mean_obs, mean_mod, pbias, diag = compute_annual_cdd_max(obs, mod)

    # Valid streaks in Year 2:
    # obs: Streak A = 16, Streak B = discarded, Streak C = 12 -> max = 16.0
    # mod: Streak A = 18, Streak B = discarded, Streak C = 12 -> max = 18.0
    assert mean_obs == 16.0
    assert mean_mod == 18.0
    assert pytest.approx(pbias) == 100.0 * (18.0 - 16.0) / 16.0  # 12.5%
    assert diag.n_discarded_streaks_obs == 1
    assert diag.n_discarded_streaks_mod == 1
    assert diag.n_discarded_years_obs == 2
    assert diag.n_discarded_years_mod == 2


def test_temperature_decisive_metrics_hand_calculated() -> None:
    """Verify temperature decisive metrics with exact hand-calculated values."""
    dates = pd.date_range("2000-01-01", "2000-12-31", freq="D")
    n = len(dates)

    tmax_obs = pd.Series(np.full(n, 20.0), index=dates)
    tmin_obs = pd.Series(np.full(n, 10.0), index=dates)
    tmax_mod = pd.Series(np.full(n, 21.0), index=dates)
    tmin_mod = pd.Series(np.full(n, 11.5), index=dates)

    metrics = compute_temperature_decisive_metrics(
        tmax_obs, tmax_mod, tmin_obs, tmin_mod
    )

    assert pytest.approx(metrics["max_monthly_bias"]) == 1.5
    assert pytest.approx(metrics["dtr_mean_bias"]) == 0.5
    assert pytest.approx(metrics["extreme_percentiles_bias"]) == 1.5
    assert pytest.approx(metrics["delta_p5_tmin"]) == 1.5
    assert pytest.approx(metrics["delta_p95_tmax"]) == 1.0


def test_descriptive_metrics() -> None:
    """Verify descriptive metrics: Rx1day, Rx5day, SDII, POD, FAR, Pearson r."""
    dates = pd.date_range("2000-01-01", periods=10, freq="D")
    p = pd.Series([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 0.0, 0.0, 0.0, 0.0], index=dates)

    assert pytest.approx(compute_rx1day(p)) == 10.0
    assert pytest.approx(compute_rx5day(p)) == 30.0
    assert pytest.approx(compute_sdii(p)) == 6.0

    obs = pd.Series([0.0, 2.0, 0.0, 5.0], index=dates[:4])
    mod = pd.Series([0.0, 3.0, 2.0, 0.0], index=dates[:4])
    cont = compute_precipitation_contingency(obs, mod)
    assert pytest.approx(cont["pod"]) == 0.5
    assert pytest.approx(cont["far"]) == 0.5

    r = compute_daily_pearson(obs, mod)
    assert not np.isnan(r)


# ==============================================================================
# 3. Three-Tier Symmetric Mapping & State Concordance Tests (Section 5.2 - 5.4)
# ==============================================================================


def test_symmetric_three_tier_metric_outcomes() -> None:
    """Verify symmetric mapping: Pass, Fail, and Inconclusive."""
    thresh = 5.0  # threshold +/- 5.0%

    # Two-sided tests
    # 1. Entirely inside -> Pass
    assert evaluate_symmetric_metric_outcome(-3.0, 2.5, thresh) == MetricOutcome.PASS
    assert (
        evaluate_symmetric_metric_outcome(-5.0, 5.0, thresh) == MetricOutcome.PASS
    )  # boundary

    # 2. Entirely outside on positive side -> Fail
    assert evaluate_symmetric_metric_outcome(5.1, 10.0, thresh) == MetricOutcome.FAIL

    # 3. Entirely outside on negative side -> Fail (symmetry check)
    assert evaluate_symmetric_metric_outcome(-12.0, -5.1, thresh) == MetricOutcome.FAIL

    # 4. Crossing upper threshold boundary -> Inconclusive
    assert (
        evaluate_symmetric_metric_outcome(2.0, 7.0, thresh)
        == MetricOutcome.INCONCLUSIVE
    )

    # 5. Crossing lower threshold boundary -> Inconclusive (symmetry check)
    assert (
        evaluate_symmetric_metric_outcome(-7.0, -2.0, thresh)
        == MetricOutcome.INCONCLUSIVE
    )

    # 6. Point estimate inside, but CI spans outside -> Inconclusive
    assert (
        evaluate_symmetric_metric_outcome(-1.0, 6.0, thresh)
        == MetricOutcome.INCONCLUSIVE
    )

    # 7. Point estimate outside, but CI spans inside -> Inconclusive (NOT Fail)
    assert (
        evaluate_symmetric_metric_outcome(4.0, 8.0, thresh)
        == MetricOutcome.INCONCLUSIVE
    )

    # One-sided tests (e.g. absolute bias <= 0.5 C)
    t_one = 0.5
    assert (
        evaluate_symmetric_metric_outcome(0.1, 0.4, t_one, is_two_sided=False)
        == MetricOutcome.PASS
    )
    assert (
        evaluate_symmetric_metric_outcome(0.6, 1.2, t_one, is_two_sided=False)
        == MetricOutcome.FAIL
    )
    assert (
        evaluate_symmetric_metric_outcome(0.3, 0.7, t_one, is_two_sided=False)
        == MetricOutcome.INCONCLUSIVE
    )


def test_global_scorecard_precipitation_rounding_rule_and_gates() -> None:
    """Verify precip global outcome: N=6 decisive -> 5 passes required."""

    # Create mock bootstrap results for all 6 metrics
    def _make_ci(pt: float, low: float, high: float) -> BootstrapConfidenceInterval:
        return BootstrapConfidenceInterval(
            point_estimate=pt, ci_lower=low, ci_upper=high, n_replicas=1000, seed=42
        )

    # 1. 5 passes, 1 inconclusive, 0 fails, gates pass -> Success (ceil(0.70*6)=5)
    boot_pass5 = {
        "f_wet_total_pbias": _make_ci(1.0, -2.0, 2.0),  # Pass (<=5)
        "f_wet_seasonal_max_pbias": _make_ci(5.0, 2.0, 7.0),  # Pass (<=10)
        "p95_wet_pbias": _make_ci(3.0, -5.0, 5.0),  # Pass (<=15)
        "p99_wet_pbias": _make_ci(4.0, -8.0, 8.0),  # Pass (<=25)
        "cdd_max_pbias": _make_ci(2.0, -4.0, 4.0),  # Pass (<=15)
        "pbias_monthly": _make_ci(6.0, 2.0, 12.0),  # Inconclusive (spans 10)
        "delta_pbias_volume": _make_ci(1.0, -0.5, 2.5),  # CI_upper >= 0 -> Pass
    }
    sc1 = evaluate_precipitation_scorecard(
        "21205710", "Level 2", boot_pass5, physical_gate_passed=True
    )
    assert sc1.n_passes == 5
    assert sc1.global_outcome == GlobalOutcome.SUCCESS

    # 2. 4 passes, 2 inconclusive, 0 fails -> Not Established (4 < 5 required)
    boot_pass4 = dict(boot_pass5)
    boot_pass4["cdd_max_pbias"] = _make_ci(10.0, 5.0, 18.0)  # Inconclusive
    sc2 = evaluate_precipitation_scorecard(
        "21205710", "Level 2", boot_pass4, physical_gate_passed=True
    )
    assert sc2.n_passes == 4
    assert sc2.global_outcome == GlobalOutcome.NOT_ESTABLISHED

    # 3. 5 passes, but 1 FAIL -> Falsified
    boot_fail1 = dict(boot_pass5)
    boot_fail1["pbias_monthly"] = _make_ci(15.0, 12.0, 20.0)  # FAIL (>10)
    sc3 = evaluate_precipitation_scorecard(
        "21205710", "Level 2", boot_fail1, physical_gate_passed=True
    )
    assert sc3.global_outcome == GlobalOutcome.FALSIFIED

    # 4. 5 passes, but physical gate failed (negative rain) -> Falsified
    sc4 = evaluate_precipitation_scorecard(
        "21205710", "Level 2", boot_pass5, physical_gate_passed=False
    )
    assert sc4.global_outcome == GlobalOutcome.FALSIFIED

    # 5. 5 passes, but volume outperformance fails (CI_upper < 0) -> Falsified
    boot_vol_fail = dict(boot_pass5)
    boot_vol_fail["delta_pbias_volume"] = _make_ci(-2.0, -4.0, -0.1)  # upper < 0
    sc5 = evaluate_precipitation_scorecard(
        "21205710", "Level 2", boot_vol_fail, physical_gate_passed=True
    )
    assert sc5.global_outcome == GlobalOutcome.FALSIFIED


def test_global_scorecard_temperature_rounding_rule_and_gates() -> None:
    """Verify temp global outcome: N=3 decisive -> 3 passes required."""

    def _make_ci(pt: float, low: float, high: float) -> BootstrapConfidenceInterval:
        return BootstrapConfidenceInterval(
            point_estimate=pt, ci_lower=low, ci_upper=high, n_replicas=1000, seed=42
        )

    # All 3 pass -> Success
    boot_pass3 = {
        "max_monthly_bias": _make_ci(0.2, 0.1, 0.4),  # Pass (<=0.5)
        "dtr_mean_bias": _make_ci(0.3, 0.1, 0.6),  # Pass (<=0.8)
        "extreme_percentiles_bias": _make_ci(0.4, 0.2, 0.7),  # Pass (<=0.8)
        "delta_p5_tmin": _make_ci(0.3, 0.1, 0.5),  # Pass
        "delta_p95_tmax": _make_ci(0.4, 0.2, 0.6),  # Pass
    }
    sc_t1 = evaluate_temperature_scorecard(
        "21205710", "Level 2", boot_pass3, physical_gate_passed=True
    )
    assert sc_t1.n_passes == 3
    assert sc_t1.global_outcome == GlobalOutcome.SUCCESS

    # 2 pass, 1 inconclusive -> Not Established (3 required)
    boot_pass2 = dict(boot_pass3)
    boot_pass2["max_monthly_bias"] = _make_ci(0.4, 0.2, 0.6)  # Inconclusive
    sc_t2 = evaluate_temperature_scorecard(
        "21205710", "Level 2", boot_pass2, physical_gate_passed=True
    )
    assert sc_t2.n_passes == 2
    assert sc_t2.global_outcome == GlobalOutcome.NOT_ESTABLISHED

    # 1 fail -> Falsified
    boot_fail1 = dict(boot_pass3)
    boot_fail1["dtr_mean_bias"] = _make_ci(1.0, 0.9, 1.2)  # FAIL
    sc_t3 = evaluate_temperature_scorecard(
        "21205710", "Level 2", boot_fail1, physical_gate_passed=True
    )
    assert sc_t3.global_outcome == GlobalOutcome.FALSIFIED


def test_temperature_extreme_component_degradation_cases() -> None:
    """Verify extreme degradation: Case A (FAIL retained), Case B (INCONCLUSIVE)."""

    def _make_ci(pt: float, low: float, high: float) -> BootstrapConfidenceInterval:
        return BootstrapConfidenceInterval(
            point_estimate=pt, ci_lower=low, ci_upper=high, n_replicas=1000, seed=42
        )

    base_boot = {
        "max_monthly_bias": _make_ci(0.2, 0.1, 0.4),  # Pass
        "dtr_mean_bias": _make_ci(0.3, 0.1, 0.6),  # Pass
    }

    # Case A: One component has CI outside (>0.8) -> MUST BE FAIL (never softened)
    boot_case_a = dict(base_boot)
    boot_case_a["extreme_percentiles_bias"] = _make_ci(0.9, 0.85, 1.10)
    boot_case_a["delta_p5_tmin"] = _make_ci(0.95, 0.85, 1.10)  # FAIL (>0.8)
    boot_case_a["delta_p95_tmax"] = _make_ci(0.30, 0.10, 0.50)  # Pass (<=0.8)

    sc_a = evaluate_temperature_scorecard(
        "21205710", "Level 2", boot_case_a, physical_gate_passed=True
    )
    assert sc_a.decisive_outcomes["extreme_percentiles_bias"] == MetricOutcome.FAIL
    assert sc_a.global_outcome == GlobalOutcome.FALSIFIED

    # Case B: Aggregate max within acceptable (<0.8), but component crosses 0.8
    # -> degrades to INCONCLUSIVE, does NOT pass
    boot_case_b = dict(base_boot)
    boot_case_b["extreme_percentiles_bias"] = _make_ci(
        0.60, 0.40, 0.75
    )  # Looks like Pass
    boot_case_b["delta_p5_tmin"] = _make_ci(0.60, 0.40, 0.75)  # Pass
    boot_case_b["delta_p95_tmax"] = _make_ci(
        0.70, 0.50, 0.90
    )  # Inconclusive (crosses 0.8)

    sc_b = evaluate_temperature_scorecard(
        "21205710", "Level 2", boot_case_b, physical_gate_passed=True
    )
    assert (
        sc_b.decisive_outcomes["extreme_percentiles_bias"] == MetricOutcome.INCONCLUSIVE
    )
    # Since only 2 passes out of 3, global outcome is NOT_ESTABLISHED
    assert sc_b.global_outcome == GlobalOutcome.NOT_ESTABLISHED


def test_weaker_of_two_concordance() -> None:
    """Verify concordance rule: Fail < Inconclusive/Not Established < Success."""
    assert (
        apply_weaker_of_two_concordance(GlobalOutcome.SUCCESS, GlobalOutcome.SUCCESS)
        == GlobalOutcome.SUCCESS
    )
    assert (
        apply_weaker_of_two_concordance(
            GlobalOutcome.SUCCESS, GlobalOutcome.NOT_ESTABLISHED
        )
        == GlobalOutcome.NOT_ESTABLISHED
    )
    assert (
        apply_weaker_of_two_concordance(
            GlobalOutcome.NOT_ESTABLISHED, GlobalOutcome.SUCCESS
        )
        == GlobalOutcome.NOT_ESTABLISHED
    )
    assert (
        apply_weaker_of_two_concordance(GlobalOutcome.SUCCESS, GlobalOutcome.FALSIFIED)
        == GlobalOutcome.FALSIFIED
    )
    assert (
        apply_weaker_of_two_concordance(
            GlobalOutcome.NOT_ESTABLISHED, GlobalOutcome.FALSIFIED
        )
        == GlobalOutcome.FALSIFIED
    )


# ==============================================================================
# 4. Bootstrap Reproducibility, Pairing & Coverage Tests (Section 5.1)
# ==============================================================================


def test_annual_block_bootstrap_reproducibility_and_pairing() -> None:
    """Verify annual block bootstrap is reproducible with seed and correctly paired."""
    dates = pd.date_range("1990-01-01", "1994-12-31", freq="D")
    rng = np.random.default_rng(123)
    n = len(dates)

    obs = pd.Series(rng.uniform(0.0, 15.0, size=n), index=dates)
    mod = pd.Series(obs * 1.1 + rng.normal(0, 0.5, size=n), index=dates)
    raw = pd.Series(obs * 1.3, index=dates)

    # 1. Reproducibility with identical seed
    res1 = compute_annual_block_bootstrap_precipitation(
        obs, mod, raw=raw, n_replicas=100, seed=42
    )
    res2 = compute_annual_block_bootstrap_precipitation(
        obs, mod, raw=raw, n_replicas=100, seed=42
    )

    for k in res1:
        assert pytest.approx(res1[k].point_estimate) == res2[k].point_estimate
        assert pytest.approx(res1[k].ci_lower) == res2[k].ci_lower
        assert pytest.approx(res1[k].ci_upper) == res2[k].ci_upper

    # 2. Temperature bootstrap reproducibility
    t_obs = pd.Series(rng.uniform(15.0, 25.0, size=n), index=dates)
    t_mod = pd.Series(t_obs + 1.0, index=dates)
    t_min_obs = t_obs - 5.0
    t_min_mod = t_mod - 4.5

    t_res1 = compute_annual_block_bootstrap_temperature(
        t_obs, t_mod, t_min_obs, t_min_mod, n_replicas=100, seed=42
    )
    t_res2 = compute_annual_block_bootstrap_temperature(
        t_obs, t_mod, t_min_obs, t_min_mod, n_replicas=100, seed=42
    )

    for k in t_res1:
        assert pytest.approx(t_res1[k].point_estimate) == t_res2[k].point_estimate
        assert pytest.approx(t_res1[k].ci_lower) == t_res2[k].ci_lower
        assert pytest.approx(t_res1[k].ci_upper) == t_res2[k].ci_upper


# ==============================================================================
# 5. Partitioners, Leakage Prevention & Remainder Fusion (Sections 2.2, 4.1, 4.2)
# ==============================================================================


def test_benchmark_partitions_configuration() -> None:
    """Verify pre-specified partition ranges match Table 2.2 exactly."""
    p_jb_cal, p_jb_val = get_benchmark_partitions("precipitation", "21205710")
    assert p_jb_cal == (1975, 1994)
    assert p_jb_val == (1995, 2015)

    p_ac_cal, p_ac_val = get_benchmark_partitions("precipitation", "23215030")
    assert p_ac_cal == (1975, 1987)
    assert p_ac_val == (1988, 2000)

    t_jb_cal, t_jb_val = get_benchmark_partitions("temperature", "21205710")
    assert t_jb_cal == (1990, 2004)
    assert t_jb_val == (2005, 2020)

    t_ac_cal, t_ac_val = get_benchmark_partitions("temperature", "23215030")
    assert t_ac_cal == (1990, 2000)
    assert t_ac_val == (2001, 2011)


def test_nested_annual_block_cv_leakage_and_tuning_losses() -> None:
    """Verify nested annual block CV has zero year leakage between train and val."""
    dates = pd.date_range("1975-01-01", "1994-12-31", freq="D")  # 20 years
    series = pd.Series(np.ones(len(dates)), index=dates)

    folds = list(nested_annual_block_cv(series))
    assert len(folds) == 20

    for train_s, val_s in folds:
        train_years = set(train_s.index.year)
        val_years = set(val_s.index.year)
        # ZERO overlap
        assert len(train_years.intersection(val_years)) == 0
        assert len(val_years) == 1
        assert len(train_years) == 19

    # Verify tuning losses
    p_loss = precipitation_tuning_loss(series, series * 1.1)
    assert p_loss >= 0.0

    t_loss = temperature_tuning_loss(
        series * 20.0, series * 21.0, series * 10.0, series * 11.0
    )
    assert t_loss >= 0.0


def test_contiguous_5year_block_cv_with_remainder_fusion(tmp_path: Path) -> None:
    """Verify 41-yr record produces 8 blocks (seven 5-yr, one 6-yr) without leakage."""
    dates = pd.date_range("1975-01-01", "2015-12-31", freq="D")  # 41 years
    series = pd.Series(np.ones(len(dates)), index=dates)
    ledger = tmp_path / "ledger.jsonl"

    folds = list(
        contiguous_5year_block_cv(
            series,
            open_validation=True,
            station_id="21205710",
            variable="precipitation",
            ledger_path=ledger,
        )
    )

    # 41 years -> 8 blocks (7 of 5 years: 1975..2009; 1 of 6 years: 2010..2015)
    assert len(folds) == 8

    test_ranges = [r for _, _, r in folds]
    assert test_ranges[0] == (1975, 1979)
    assert test_ranges[1] == (1980, 1984)
    assert test_ranges[2] == (1985, 1989)
    assert test_ranges[3] == (1990, 1994)
    assert test_ranges[4] == (1995, 1999)
    assert test_ranges[5] == (2000, 2004)
    assert test_ranges[6] == (2005, 2009)
    assert test_ranges[7] == (2010, 2015)  # 6-year fused remainder block!

    # Check zero leakage in each fold
    for train_s, test_s, r in folds:
        train_y = set(train_s.index.year)
        test_y = set(test_s.index.year)
        assert len(train_y.intersection(test_y)) == 0


# ==============================================================================
# 6. Validation Access Guard Tests (Section 2.2 and requirement g)
# ==============================================================================


def test_validation_access_guard_blocks_and_records(tmp_path: Path) -> None:
    """Verify guard raises PermissionError without flag and logs with flag."""
    ledger = tmp_path / "openings.jsonl"
    dates = pd.date_range("1975-01-01", "2015-12-31", freq="D")
    series = pd.Series(np.ones(len(dates)), index=dates)

    # 1. Blocking without flag
    with pytest.raises(PermissionError, match="Validation access blocked"):
        split_calibration_validation(
            series,
            cal_range=(1975, 1994),
            val_range=(1995, 2015),
            open_validation=False,
            station_id="21205710",
            variable="precipitation",
            ledger_path=ledger,
        )

    # 2. 5-year block CV also blocks without flag
    with pytest.raises(PermissionError, match="Validation access blocked"):
        list(
            contiguous_5year_block_cv(
                series,
                open_validation=False,
                station_id="21205710",
                variable="precipitation",
                ledger_path=ledger,
            )
        )

    # 3. Successful access with flag records in ledger
    cal, val = split_calibration_validation(
        series,
        cal_range=(1975, 1994),
        val_range=(1995, 2015),
        open_validation=True,
        station_id="21205710",
        variable="precipitation",
        ledger_path=ledger,
    )
    assert len(cal) > 0
    assert len(val) > 0
    assert ledger.exists()

    # Read ledger lines
    lines = ledger.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["station_id"] == "21205710"
    assert record["variable"] == "precipitation"
    assert "timestamp" in record
    assert "commit_hash" in record


def test_annual_block_bootstrap_empirical_coverage() -> None:
    """Verify empirical coverage of annual block bootstrap on synthetic series."""
    dates = pd.date_range("1980-01-01", "1994-12-31", freq="D")  # 15 years
    rng = np.random.default_rng(999)
    n = len(dates)

    # True observed: uniform [0, 20]
    obs_vals = rng.uniform(0.0, 20.0, size=n)
    # True modeled has exactly +10% volume scaling
    mod_vals = obs_vals * 1.10

    obs = pd.Series(obs_vals, index=dates)
    mod = pd.Series(mod_vals, index=dates)

    boot = compute_annual_block_bootstrap_precipitation(
        obs, mod, n_replicas=200, seed=123
    )
    ci_pbias = boot["pbias_monthly"]

    # True monthly pbias is exactly 10.0%
    true_pbias = 10.0
    assert ci_pbias.ci_lower <= true_pbias <= ci_pbias.ci_upper
    assert pytest.approx(ci_pbias.point_estimate, rel=1e-2) == true_pbias


def test_cdd_nan_interrupted_streak() -> None:
    """Verify CDD_max discards any dry streak interrupted by a NaN (hand-calculated)."""
    dates = pd.date_range("2000-01-01", "2000-12-31", freq="D")  # 366 days (leap)
    obs = pd.Series(10.0, index=dates)  # baseline rain
    mod = pd.Series(10.0, index=dates)

    # Streak 1 (interrupted by NaN): May 10-14 dry (5d), May 15 NaN, May 16-19 dry (4d).
    # Bounded by May 9 (rain 10mm) and May 20 (rain 10mm).
    # Entire streak touches/contains NaN -> DISCARDED (not counted as 5 or 4).
    obs.loc["2000-05-10":"2000-05-14"] = 0.0
    obs.loc["2000-05-15"] = np.nan
    obs.loc["2000-05-16":"2000-05-19"] = 0.0

    mod.loc["2000-05-10":"2000-05-14"] = 0.0
    mod.loc["2000-05-15"] = np.nan
    mod.loc["2000-05-16":"2000-05-19"] = 0.0

    # Streak 2 (valid): July 1 to July 10 dry (10 days).
    obs.loc["2000-07-01":"2000-07-10"] = 0.0
    mod.loc["2000-07-01":"2000-07-12"] = 0.0  # 12 days in mod

    mean_obs, mean_mod, pbias, diag = compute_annual_cdd_max(obs, mod)

    assert mean_obs == 10.0
    assert mean_mod == 12.0
    assert pytest.approx(pbias) == 100.0 * (12.0 - 10.0) / 10.0  # 20%
    assert diag.n_discarded_streaks_obs == 1
    assert diag.n_discarded_streaks_mod == 1
    assert diag.n_discarded_years_obs == 0
    assert diag.n_discarded_years_mod == 0


def test_cdd_adjacent_nan() -> None:
    """Verify CDD_max discards streaks adjacent to NaNs (hand-calculated)."""
    dates = pd.date_range("2001-01-01", "2001-12-31", freq="D")  # 365 days
    obs = pd.Series(10.0, index=dates)
    mod = pd.Series(10.0, index=dates)

    # Streak 1: NaN on left (April 5 is NaN, April 6 to 12 dry = 7d, April 13 rain)
    # Touches NaN on left -> DISCARDED.
    obs.loc["2001-04-05"] = np.nan
    obs.loc["2001-04-06":"2001-04-12"] = 0.0
    mod.loc["2001-04-05"] = np.nan
    mod.loc["2001-04-06":"2001-04-12"] = 0.0

    # Streak 2: NaN on right (August 1 to 6 dry = 6d, August 7 is NaN, August 8 rain)
    # Touches NaN on right -> DISCARDED.
    obs.loc["2001-08-01":"2001-08-06"] = 0.0
    obs.loc["2001-08-07"] = np.nan
    mod.loc["2001-08-01":"2001-08-06"] = 0.0
    mod.loc["2001-08-07"] = np.nan

    # Streak 3: Valid clean streak (October 1 to 8 dry = 8 days)
    obs.loc["2001-10-01":"2001-10-08"] = 0.0
    mod.loc["2001-10-01":"2001-10-08"] = 0.0

    mean_obs, mean_mod, pbias, diag = compute_annual_cdd_max(obs, mod)

    assert mean_obs == 8.0
    assert mean_mod == 8.0
    assert pbias == 0.0
    assert diag.n_discarded_streaks_obs == 2
    assert diag.n_discarded_streaks_mod == 2
    assert diag.n_discarded_years_obs == 0
    assert diag.n_discarded_years_mod == 0


def test_cdd_year_boundary_nan() -> None:
    """Verify CDD_max discards streaks touching NaNs at calendar year boundaries."""
    dates = pd.date_range("2002-01-01", "2002-12-31", freq="D")  # 365 days
    obs = pd.Series(10.0, index=dates)
    mod = pd.Series(10.0, index=dates)

    # Streak 1: Jan 1 is NaN, Jan 2 to Jan 6 are dry (5 days), Jan 7 rain.
    # Touches NaN at Jan 1 boundary -> DISCARDED.
    obs.loc["2002-01-01"] = np.nan
    obs.loc["2002-01-02":"2002-01-06"] = 0.0
    mod.loc["2002-01-01"] = np.nan
    mod.loc["2002-01-02":"2002-01-06"] = 0.0

    # Streak 2: Dec 25 rain, Dec 26 to Dec 30 dry (5 days), Dec 31 is NaN.
    # Touches NaN at Dec 31 boundary -> DISCARDED.
    obs.loc["2002-12-26":"2002-12-30"] = 0.0
    obs.loc["2002-12-31"] = np.nan
    mod.loc["2002-12-26":"2002-12-30"] = 0.0
    mod.loc["2002-12-31"] = np.nan

    # Streak 3: Clean valid streak in June (June 1 to 7 dry = 7 days)
    obs.loc["2002-06-01":"2002-06-07"] = 0.0
    mod.loc["2002-06-01":"2002-06-07"] = 0.0

    mean_obs, mean_mod, pbias, diag = compute_annual_cdd_max(obs, mod)

    assert mean_obs == 7.0
    assert mean_mod == 7.0
    assert pbias == 0.0
    assert diag.n_discarded_streaks_obs == 2
    assert diag.n_discarded_streaks_mod == 2
    assert diag.n_discarded_years_obs == 0
    assert diag.n_discarded_years_mod == 0


def test_compute_required_passes_integer_arithmetic() -> None:
    """Verify 70% threshold ceiling strictly uses integer arithmetic: -(-7*N // 10)."""
    # N = 3 -> ceil(2.1) = 3
    assert compute_required_passes(3) == 3
    assert compute_required_passes(3) == -(-7 * 3 // 10)

    # N = 6 -> ceil(4.2) = 5
    assert compute_required_passes(6) == 5
    assert compute_required_passes(6) == -(-7 * 6 // 10)

    # N = 10 -> ceil(7.0) = 7
    assert compute_required_passes(10) == 7
    assert compute_required_passes(10) == -(-7 * 10 // 10)

    # Type check: result is strictly int
    assert isinstance(compute_required_passes(3), int)
    assert isinstance(compute_required_passes(6), int)
    assert isinstance(compute_required_passes(10), int)


def test_bootstrap_discarded_replicas_tracking() -> None:
    """Verify that annual block bootstrap reports discarded replicas count."""
    dates = pd.date_range("1980-01-01", "1983-12-31", freq="D")  # 4 years
    rng = np.random.default_rng(42)
    obs = pd.Series(rng.uniform(0.0, 10.0, size=len(dates)), index=dates)
    mod = obs * 1.05

    boot = compute_annual_block_bootstrap_precipitation(
        obs, mod, n_replicas=50, seed=42
    )

    for metric_name, ci in boot.items():
        assert isinstance(ci.n_discarded_replicas, int)
        assert ci.n_discarded_replicas >= 0
        assert ci.n_discarded_replicas <= ci.n_replicas


def test_scorecard_exposes_cdd_diagnostic() -> None:
    """Verify EvaluationScorecard exposes CDDDiagnostic for precipitation."""
    cdd_diag = CDDDiagnostic(
        n_discarded_streaks_obs=2,
        n_discarded_streaks_mod=1,
        n_discarded_years_obs=0,
        n_discarded_years_mod=0,
    )
    boot_res = {
        m: BootstrapConfidenceInterval(
            point_estimate=1.0,
            ci_lower=0.5,
            ci_upper=1.5,
            n_replicas=100,
            seed=42,
            n_discarded_replicas=0,
        )
        for m in [
            "f_wet_total_pbias",
            "f_wet_seasonal_max_pbias",
            "p95_wet_pbias",
            "p99_wet_pbias",
            "cdd_max_pbias",
            "pbias_monthly",
            "delta_pbias_volume",
        ]
    }

    card = evaluate_precipitation_scorecard(
        station_id="21205710",
        level="Level1",
        boot_results=boot_res,
        physical_gate_passed=True,
        cdd_diagnostic=cdd_diag,
    )

    assert card.cdd_diagnostic is not None
    assert card.cdd_diagnostic.n_discarded_streaks_obs == 2
    assert card.cdd_diagnostic.n_discarded_streaks_mod == 1

