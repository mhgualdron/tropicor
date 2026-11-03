"""Unit tests for benchmark evaluator pure helpers (Appendix J and K)."""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pytest

import tropicor.core.benchmark_evaluator as be
from tropicor.core.benchmark_evaluator import (
    BENCHMARK_STATION_IDS,
    SAMPLE_STATUS_ABORTED,
    SAMPLE_STATUS_INSUFFICIENT,
    SAMPLE_STATUS_OK,
    FittedLevels,
    LevelPrediction,
    TrackEvaluation,
    _block_cv_predictions,
    combine_tracks,
    evaluate_block_cv,
    evaluate_primary_split,
    evaluate_track_levels,
    fit_levels,
    five_year_blocks,
    levels_with_global_success,
    mask_to_valid_years,
    predict_levels,
    track_sample_status,
    valid_years_j71,
)
from tropicor.core.protocol import (
    CrossingDiagnostic,
    GlobalOutcome,
    compute_annual_block_bootstrap_temperature,
    contiguous_5year_block_cv,
)
from tropicor.downscale.temperature import (
    Level1MonthlyAdditiveTemperature,
    Level2MeanStateDTRScalingTemperature,
)


def test_benchmark_station_ids() -> None:
    """Verify benchmark station IDs match Appendix J specification."""
    assert BENCHMARK_STATION_IDS == ("21205710", "23215030")


def test_valid_years_j71_single_day_counts_as_valid() -> None:
    """Requirement 2a: Single day with all 4 values counts as valid (J.12.3)."""
    dates = pd.date_range("2001-01-01", "2001-12-31", freq="D")
    tmax_obs = pd.Series(np.nan, index=dates)
    tmin_obs = pd.Series(np.nan, index=dates)
    tmax_mod = pd.Series(np.nan, index=dates)
    tmin_mod = pd.Series(np.nan, index=dates)

    # Exactly one calendar day has all 4 series non-NaN
    target_day = pd.Timestamp("2001-06-15")
    tmax_obs.loc[target_day] = 25.0
    tmin_obs.loc[target_day] = 15.0
    tmax_mod.loc[target_day] = 24.0
    tmin_mod.loc[target_day] = 14.0

    valid_years = valid_years_j71(tmax_obs, tmin_obs, tmax_mod, tmin_mod)
    assert valid_years == [2001]


def test_valid_years_j71_unpaired_days_not_valid() -> None:
    """Requirement 2b: Unpaired observations across days do not make a year valid."""
    dates = pd.date_range("2001-01-01", "2001-12-31", freq="D")
    tmax_obs = pd.Series(np.nan, index=dates)
    tmin_obs = pd.Series(np.nan, index=dates)
    tmax_mod = pd.Series(20.0, index=dates)
    tmin_mod = pd.Series(10.0, index=dates)

    # Day A has valid Tmax obs, but NaN Tmin obs
    day_a = pd.Timestamp("2001-03-01")
    tmax_obs.loc[day_a] = 25.0

    # Day B has valid Tmin obs, but NaN Tmax obs
    day_b = pd.Timestamp("2001-03-02")
    tmin_obs.loc[day_b] = 15.0

    valid_years = valid_years_j71(tmax_obs, tmin_obs, tmax_mod, tmin_mod)
    assert valid_years == []


def test_mask_to_valid_years_does_not_modify_inputs() -> None:
    """Requirement 2c: mask_to_valid_years preserves inputs and masks invalid years."""
    dates = pd.date_range("2001-01-01", "2003-12-31", freq="D")
    tmax_obs = pd.Series(25.0, index=dates)
    tmin_obs = pd.Series(15.0, index=dates)
    tmax_mod = pd.Series(24.0, index=dates)
    tmin_mod = pd.Series(14.0, index=dates)

    orig_tmax_obs = tmax_obs.copy()
    orig_tmin_obs = tmin_obs.copy()
    orig_tmax_mod = tmax_mod.copy()
    orig_tmin_mod = tmin_mod.copy()

    # Preserve only 2002
    m1, m2, m3, m4 = mask_to_valid_years(
        tmax_obs, tmin_obs, tmax_mod, tmin_mod, valid_years=[2002]
    )

    # Verify originals were untouched
    assert tmax_obs.equals(orig_tmax_obs)
    assert tmin_obs.equals(orig_tmin_obs)
    assert tmax_mod.equals(orig_tmax_mod)
    assert tmin_mod.equals(orig_tmin_mod)

    # Verify non-valid years (2001, 2003) are NaN across all four series
    for masked in (m1, m2, m3, m4):
        invalid_years_data = masked.loc[masked.index.year.isin([2001, 2003])]
        assert invalid_years_data.isna().all()

        # Valid year 2002 is preserved
        valid_year_data = masked.loc[masked.index.year == 2002]
        assert not valid_year_data.isna().any()


def test_five_year_blocks_exact_ranges() -> None:
    """Requirement 2d: Test exact block ranges for 1990-2020 and 1990-2011."""
    blocks_31 = five_year_blocks(range(1990, 2021))
    assert blocks_31 == [
        (1990, 1994),
        (1995, 1999),
        (2000, 2004),
        (2005, 2009),
        (2010, 2014),
        (2015, 2020),
    ]

    blocks_22 = five_year_blocks(range(1990, 2012))
    assert blocks_22 == [
        (1990, 1994),
        (1995, 1999),
        (2000, 2004),
        (2005, 2011),
    ]


def test_five_year_blocks_agrees_with_contiguous_cv(tmp_path: Path) -> None:
    """Requirement 2e: Agrees with contiguous_5year_block_cv and writes tmp ledger."""
    dates = pd.date_range("1990-01-01", "2020-12-31", freq="D")
    series = pd.Series(1.0, index=dates)
    tmp_ledger = tmp_path / "test_validation_ledger.jsonl"

    cv_gen = contiguous_5year_block_cv(
        series=series,
        open_validation=True,
        station_id="TEST_STATION",
        variable="tmax",
        ledger_path=tmp_ledger,
    )
    cv_ranges = [fold[2] for fold in cv_gen]

    years = sorted(series.index.year.unique())
    helper_blocks = five_year_blocks(years)

    assert helper_blocks == cv_ranges
    assert tmp_ledger.exists()
    assert tmp_ledger.stat().st_size > 0
    assert tmp_ledger.parent == tmp_path


def test_masking_effect_on_annual_block_bootstrap() -> None:
    """Requirement 2f: Documents why J.7.1 masking is required before bootstrap.

    Synthetic series has 6 years valid under J.7.1 plus one extra year with
    unpaired observations. Without masking, compute_annual_block_bootstrap_temperature
    finds 7 years (insufficient_blocks=False). After masking, only 6 valid years
    remain, resulting in insufficient_blocks=True.
    """
    dates = pd.date_range("2000-01-01", "2006-12-31", freq="D")
    tmax_obs = pd.Series(np.nan, index=dates)
    tmin_obs = pd.Series(np.nan, index=dates)
    tmax_mod = pd.Series(20.0, index=dates)
    tmin_mod = pd.Series(10.0, index=dates)

    # 6 years (2000-2005) with valid data on all days
    for y in range(2000, 2006):
        mask = dates.year == y
        tmax_obs.loc[mask] = 25.0
        tmin_obs.loc[mask] = 15.0

    # 7th year (2006) has unpaired observations only:
    # tmax_obs on day 1, tmin_obs on day 2; no day has both valid
    tmax_obs.loc[pd.Timestamp("2006-01-01")] = 25.0
    tmin_obs.loc[pd.Timestamp("2006-01-02")] = 15.0

    # 1. Unmasked: compute_annual_block_bootstrap_temperature counts 7 years
    res_unmasked = compute_annual_block_bootstrap_temperature(
        tmax_obs=tmax_obs,
        tmax_mod=tmax_mod,
        tmin_obs=tmin_obs,
        tmin_mod=tmin_mod,
        n_replicas=10,
        seed=42,
    )
    assert not any(v.insufficient_blocks for v in res_unmasked.values())

    # 2. Masked: J.7.1 valid years identifies only 6 valid years
    valid_years = valid_years_j71(tmax_obs, tmin_obs, tmax_mod, tmin_mod)
    assert valid_years == list(range(2000, 2006))

    m_tmax_obs, m_tmin_obs, m_tmax_mod, m_tmin_mod = mask_to_valid_years(
        tmax_obs, tmin_obs, tmax_mod, tmin_mod, valid_years
    )
    res_masked = compute_annual_block_bootstrap_temperature(
        tmax_obs=m_tmax_obs,
        tmax_mod=m_tmax_mod,
        tmin_obs=m_tmin_obs,
        tmin_mod=m_tmin_mod,
        n_replicas=10,
        seed=42,
    )
    assert all(v.insufficient_blocks for v in res_masked.values())


def test_track_sample_status() -> None:
    """Requirement 2g: Test track_sample_status at 6 and 7 (and edge values)."""
    assert track_sample_status(6) == SAMPLE_STATUS_INSUFFICIENT
    assert track_sample_status(7) == SAMPLE_STATUS_OK
    assert track_sample_status(0) == SAMPLE_STATUS_INSUFFICIENT
    assert track_sample_status(8) == SAMPLE_STATUS_OK


def test_index_validation_errors() -> None:
    """Test validation errors for non-DatetimeIndex and mismatched indexes."""
    non_dt_series = pd.Series([1.0, 2.0], index=[0, 1])
    dt_series1 = pd.Series([1.0, 2.0], index=pd.date_range("2000-01-01", periods=2))
    dt_series2 = pd.Series([1.0, 2.0], index=pd.date_range("2001-01-01", periods=2))

    with pytest.raises(ValueError, match="pd.DatetimeIndex"):
        valid_years_j71(non_dt_series, dt_series1, dt_series1, dt_series1)

    with pytest.raises(ValueError, match="identical indexes"):
        valid_years_j71(dt_series1, dt_series2, dt_series1, dt_series1)

    with pytest.raises(ValueError, match="pd.DatetimeIndex"):
        mask_to_valid_years(non_dt_series, dt_series1, dt_series1, dt_series1, [2000])

    with pytest.raises(ValueError, match="identical indexes"):
        mask_to_valid_years(dt_series1, dt_series2, dt_series1, dt_series1, [2000])


def test_five_year_blocks_insufficient_years() -> None:
    """Test ValueError when fewer than 5 unique years are supplied."""
    with pytest.raises(ValueError, match="at least 5 years"):
        five_year_blocks([2000, 2001, 2002, 2003])


def test_track_levels_synthetic_monthly_bias_l0_fails_l1_passes() -> None:
    """Requirement 2a: 7 full years with constant bias. L0 fails, L1+ passes.

    Raw has constant +3 C bias on Tmax and -2 C on Tmin relative to obs.
    Downscalers fit on raw correct the bias.
    Verify:
      - sample_status == 'ok'
      - level0 outcome is FALSIFIED
      - level1, level2, level3_winner, level3_constant all SUCCESS
      - level3_constant scorecard has identical metrics to level3_winner.
    """
    dates = pd.date_range("2010-01-01", "2016-12-31", freq="D")
    doy = dates.dayofyear.to_numpy()
    tmax_obs = pd.Series(25.0 + 3.0 * np.sin(2 * np.pi * doy / 365.25), index=dates)
    tmin_obs = pd.Series(15.0 + 2.0 * np.sin(2 * np.pi * doy / 365.25), index=dates)
    raw_tmax = tmax_obs + 3.0
    raw_tmin = tmin_obs - 2.0

    m1 = Level1MonthlyAdditiveTemperature()
    m1.fit(tmax_obs, tmin_obs, raw_tmax, raw_tmin)
    l1_tmax, l1_tmin = m1.transform(raw_tmax, raw_tmin)
    pred_l1 = LevelPrediction(
        tmax=l1_tmax,
        tmin=l1_tmin,
        crossing_diagnostic=m1.last_crossing_diagnostic_,
    )

    m2 = Level2MeanStateDTRScalingTemperature()
    m2.fit(tmax_obs, tmin_obs, raw_tmax, raw_tmin)
    l2_tmax, l2_tmin = m2.transform(raw_tmax, raw_tmin)
    pred_l2 = LevelPrediction(
        tmax=l2_tmax,
        tmin=l2_tmin,
        crossing_diagnostic=m2.last_crossing_diagnostic_,
    )

    predictions = {
        "level1": pred_l1,
        "level2": pred_l2,
        "level3_winner": pred_l1,
    }

    result = evaluate_track_levels(
        station_id="TEST_STATION",
        obs_tmax=tmax_obs,
        obs_tmin=tmin_obs,
        raw_tmax=raw_tmax,
        raw_tmin=raw_tmin,
        predictions=predictions,
        identical_to_winner=True,
        n_replicas=20,
        seed=42,
    )

    assert isinstance(result, TrackEvaluation)
    assert result.sample_status == SAMPLE_STATUS_OK
    assert result.valid_years == list(range(2010, 2017))
    assert result.reason is None

    assert result.outcomes["level0"] == GlobalOutcome.FALSIFIED
    assert result.outcomes["level1"] == GlobalOutcome.SUCCESS
    assert result.outcomes["level2"] == GlobalOutcome.SUCCESS
    assert result.outcomes["level3_winner"] == GlobalOutcome.SUCCESS
    assert result.outcomes["level3_constant"] == GlobalOutcome.SUCCESS

    sc_l3w = result.scorecards["level3_winner"]
    sc_l3c = result.scorecards["level3_constant"]
    assert sc_l3w is not None and sc_l3c is not None
    assert sc_l3c.level == "level3_constant"
    assert sc_l3c.decisive_cis == sc_l3w.decisive_cis


def test_track_levels_insufficient_sample_short_circuit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Requirement 2b: 3 full years only short-circuits to NOT_ESTABLISHED.

    compute_annual_block_bootstrap_temperature is never called.
    """
    dates = pd.date_range("2010-01-01", "2012-12-31", freq="D")
    tmax_obs = pd.Series(25.0, index=dates)
    tmin_obs = pd.Series(15.0, index=dates)
    raw_tmax = pd.Series(28.0, index=dates)
    raw_tmin = pd.Series(13.0, index=dates)

    pred = LevelPrediction(tmax=tmax_obs, tmin=tmin_obs)
    predictions = {
        "level1": pred,
        "level2": pred,
        "level3_winner": pred,
    }

    def _fail_bootstrap(*args: object, **kwargs: object) -> object:
        raise RuntimeError("Bootstrap should not be called!")

    monkeypatch.setattr(
        be,
        "compute_annual_block_bootstrap_temperature",
        _fail_bootstrap,
    )

    result = evaluate_track_levels(
        station_id="TEST_STATION",
        obs_tmax=tmax_obs,
        obs_tmin=tmin_obs,
        raw_tmax=raw_tmax,
        raw_tmin=raw_tmin,
        predictions=predictions,
        identical_to_winner=True,
    )

    assert result.sample_status == SAMPLE_STATUS_INSUFFICIENT
    assert result.valid_years == [2010, 2011, 2012]
    assert all(sc is None for sc in result.scorecards.values())
    assert all(out == GlobalOutcome.NOT_ESTABLISHED for out in result.outcomes.values())
    assert result.reason is not None
    assert "insufficient sample" in result.reason


def test_track_levels_unpaired_seventh_year_short_circuit() -> None:
    """Requirement 2c: 6 paired years + 1 unpaired year -> insufficient blocks.

    6 full paired years (2010..2015) + 1 year (2016) where obs has 50 days
    and raw has 50 missing days on those exact days, resulting in 0 paired days.
    """
    dates = pd.date_range("2010-01-01", "2016-12-31", freq="D")
    obs_tmax = pd.Series(25.0, index=dates)
    obs_tmin = pd.Series(15.0, index=dates)
    raw_tmax = pd.Series(28.0, index=dates)
    raw_tmin = pd.Series(13.0, index=dates)

    # In 2016, obs is present only on first 50 days, but raw has 50 missing days there
    mask_obs_absent_2016 = (dates.year == 2016) & (dates.dayofyear > 50)
    obs_tmax.loc[mask_obs_absent_2016] = np.nan
    obs_tmin.loc[mask_obs_absent_2016] = np.nan

    mask_raw_missing_2016 = (dates.year == 2016) & (dates.dayofyear <= 50)
    raw_tmax.loc[mask_raw_missing_2016] = np.nan
    raw_tmin.loc[mask_raw_missing_2016] = np.nan

    pred = LevelPrediction(tmax=raw_tmax, tmin=raw_tmin)
    predictions = {
        "level1": pred,
        "level2": pred,
        "level3_winner": pred,
    }

    result = evaluate_track_levels(
        station_id="TEST_STATION",
        obs_tmax=obs_tmax,
        obs_tmin=obs_tmin,
        raw_tmax=raw_tmax,
        raw_tmin=raw_tmin,
        predictions=predictions,
        identical_to_winner=True,
    )

    assert len(result.valid_years) == 6
    assert result.valid_years == list(range(2010, 2016))
    assert result.sample_status == SAMPLE_STATUS_INSUFFICIENT
    assert all(sc is None for sc in result.scorecards.values())
    assert all(out == GlobalOutcome.NOT_ESTABLISHED for out in result.outcomes.values())


def test_track_levels_identical_to_winner_handling() -> None:
    """Requirement 2d: Test identical_to_winner handling and level3_constant."""
    dates = pd.date_range("2010-01-01", "2016-12-31", freq="D")
    obs_tmax = pd.Series(25.0, index=dates)
    obs_tmin = pd.Series(15.0, index=dates)
    raw_tmax = pd.Series(25.0, index=dates)
    raw_tmin = pd.Series(15.0, index=dates)

    pred = LevelPrediction(tmax=obs_tmax, tmin=obs_tmin)
    preds_no_l3c = {
        "level1": pred,
        "level2": pred,
        "level3_winner": pred,
    }

    # 1. identical_to_winner=True: does not require 'level3_constant'
    res_true = evaluate_track_levels(
        station_id="TEST_STATION",
        obs_tmax=obs_tmax,
        obs_tmin=obs_tmin,
        raw_tmax=raw_tmax,
        raw_tmin=raw_tmin,
        predictions=preds_no_l3c,
        identical_to_winner=True,
        n_replicas=20,
        seed=42,
    )
    assert res_true.scorecards["level3_constant"] is not None
    assert (
        res_true.scorecards["level3_constant"].decisive_cis
        == res_true.scorecards["level3_winner"].decisive_cis
    )

    # 2. identical_to_winner=False: raises ValueError when missing 'level3_constant'
    with pytest.raises(ValueError, match="level3_constant"):
        evaluate_track_levels(
            station_id="TEST_STATION",
            obs_tmax=obs_tmax,
            obs_tmin=obs_tmin,
            raw_tmax=raw_tmax,
            raw_tmin=raw_tmin,
            predictions=preds_no_l3c,
            identical_to_winner=False,
            n_replicas=20,
            seed=42,
        )

    # 3. identical_to_winner=False with 'level3_constant' provided: runs bootstrap
    pred_l3c = LevelPrediction(tmax=obs_tmax + 0.1, tmin=obs_tmin)
    preds_with_l3c = dict(preds_no_l3c, level3_constant=pred_l3c)
    res_false = evaluate_track_levels(
        station_id="TEST_STATION",
        obs_tmax=obs_tmax,
        obs_tmin=obs_tmin,
        raw_tmax=raw_tmax,
        raw_tmin=raw_tmin,
        predictions=preds_with_l3c,
        identical_to_winner=False,
        n_replicas=20,
        seed=42,
    )
    assert res_false.scorecards["level3_constant"] is not None
    assert res_false.scorecards["level3_constant"].level == "level3_constant"


def test_track_levels_crossing_alert_triggers_falsified() -> None:
    """Requirement 2e: Crossing alert triggers FALSIFIED scorecard.

    A level prediction with crossing_diagnostic.alert_triggered=True and
    uncorrected crossings fails the physical gate, resulting in FALSIFIED
    even though overall biases are small.
    """
    dates = pd.date_range("2010-01-01", "2016-12-31", freq="D")
    doy = dates.dayofyear.to_numpy()
    obs_tmax = pd.Series(25.0 + 3.0 * np.sin(2 * np.pi * doy / 365.25), index=dates)
    obs_tmin = pd.Series(15.0 + 2.0 * np.sin(2 * np.pi * doy / 365.25), index=dates)
    raw_tmax = obs_tmax.copy()
    raw_tmin = obs_tmin.copy()

    pred_pass = LevelPrediction(tmax=obs_tmax, tmin=obs_tmin)

    # Crossing on 50 days (approx 2% of ~2556 days > 1% threshold)
    tmax_alert = obs_tmax.copy()
    tmin_alert = obs_tmin.copy()
    idx_cross = dates[:50]
    tmax_alert.loc[idx_cross] = 14.0
    tmin_alert.loc[idx_cross] = 16.0
    diag = CrossingDiagnostic(
        n_crossings=50,
        n_total_days=len(dates),
        fraction_crossings=50 / len(dates),
        alert_triggered=True,
    )
    pred_alert = LevelPrediction(
        tmax=tmax_alert,
        tmin=tmin_alert,
        crossing_diagnostic=diag,
    )

    preds = {
        "level1": pred_alert,
        "level2": pred_pass,
        "level3_winner": pred_pass,
    }

    res = evaluate_track_levels(
        station_id="TEST_STATION",
        obs_tmax=obs_tmax,
        obs_tmin=obs_tmin,
        raw_tmax=raw_tmax,
        raw_tmin=raw_tmin,
        predictions=preds,
        identical_to_winner=True,
        n_replicas=20,
        seed=42,
    )

    sc_l1 = res.scorecards["level1"]
    assert sc_l1 is not None
    assert sc_l1.crossing_diagnostic is not None
    assert sc_l1.crossing_diagnostic.alert_triggered is True
    assert sc_l1.physical_gate_passed is False
    assert res.outcomes["level1"] == GlobalOutcome.FALSIFIED


def test_track_levels_determinism() -> None:
    """Requirement 2f: Identical seed produces identical metric values."""
    dates = pd.date_range("2010-01-01", "2016-12-31", freq="D")
    doy = dates.dayofyear.to_numpy()
    obs_tmax = pd.Series(25.0 + 3.0 * np.sin(2 * np.pi * doy / 365.25), index=dates)
    obs_tmin = pd.Series(15.0 + 2.0 * np.sin(2 * np.pi * doy / 365.25), index=dates)
    raw_tmax = obs_tmax + 3.0
    raw_tmin = obs_tmin - 2.0

    pred = LevelPrediction(tmax=obs_tmax, tmin=obs_tmin)
    preds = {
        "level1": pred,
        "level2": pred,
        "level3_winner": pred,
    }

    res1 = evaluate_track_levels(
        station_id="TEST_STATION",
        obs_tmax=obs_tmax,
        obs_tmin=obs_tmin,
        raw_tmax=raw_tmax,
        raw_tmin=raw_tmin,
        predictions=preds,
        identical_to_winner=True,
        n_replicas=20,
        seed=42,
    )
    res2 = evaluate_track_levels(
        station_id="TEST_STATION",
        obs_tmax=obs_tmax,
        obs_tmin=obs_tmin,
        raw_tmax=raw_tmax,
        raw_tmin=raw_tmin,
        predictions=preds,
        identical_to_winner=True,
        n_replicas=20,
        seed=42,
    )

    for lvl in ("level0", "level1", "level2", "level3_winner", "level3_constant"):
        sc1 = res1.scorecards[lvl]
        sc2 = res2.scorecards[lvl]
        assert sc1 is not None and sc2 is not None
        assert sc1.global_outcome == sc2.global_outcome
        for metric_name in sc1.decisive_cis:
            ci1 = sc1.decisive_cis[metric_name]
            ci2 = sc2.decisive_cis[metric_name]
            assert ci1.point_estimate == ci2.point_estimate
            assert ci1.ci_lower == ci2.ci_lower
            assert ci1.ci_upper == ci2.ci_upper


def _make_synthetic_temperature_series(
    start_year: int,
    end_year: int,
    seed: int = 42,
) -> Tuple[pd.Series, pd.Series]:
    """Build deterministic synthetic daily series with seasonal cycle and noise.

    Guarantees Tmin < Tmax strictly on every day.
    """
    dates = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="D")
    rng = np.random.default_rng(seed)
    n = len(dates)
    doy = dates.dayofyear.to_numpy()

    seasonal = 3.0 * np.sin(2 * np.pi * doy / 365.25)
    noise_tmax = rng.normal(0.0, 0.5, size=n)
    noise_dtr = rng.uniform(0.0, 0.5, size=n)

    tmax = pd.Series(25.0 + seasonal + noise_tmax, index=dates, name="tmax")
    dtr = 8.0 + noise_dtr
    tmin = pd.Series(tmax - dtr, index=dates, name="tmin")

    return tmax, tmin


def test_fit_and_predict_levels_round_trip() -> None:
    """Requirement 5a: fit_levels/predict_levels round trip on seasonal series.

    When model = obs + 3 C, Level 1 subtracts 3 C exactly so predictions equal obs.
    identical_to_winner is True for 'constant' extrapolation, False for 'linear'.
    """
    obs_tmax, obs_tmin = _make_synthetic_temperature_series(1990, 2000)
    mod_tmax = obs_tmax + 3.0
    mod_tmin = obs_tmin + 3.0

    # 1. Extrapolation == "constant"
    cfg_const = {
        "config_id": 1,
        "grouping": "monthly",
        "n_quantiles": 50,
        "extrapolation": "constant",
    }
    fitted_const = fit_levels(obs_tmax, obs_tmin, mod_tmax, mod_tmin, cfg_const)
    assert isinstance(fitted_const, FittedLevels)
    assert fitted_const.identical_to_winner is True
    assert fitted_const.winner_config_id == 1
    assert fitted_const.level3_constant is fitted_const.level3_winner

    preds_const = predict_levels(fitted_const, mod_tmax, mod_tmin)
    assert "level3_constant" not in preds_const
    np.testing.assert_allclose(
        preds_const["level1"].tmax.to_numpy(), obs_tmax.to_numpy(), atol=1e-9
    )
    np.testing.assert_allclose(
        preds_const["level1"].tmin.to_numpy(), obs_tmin.to_numpy(), atol=1e-9
    )

    # 2. Extrapolation == "linear"
    cfg_linear = {
        "config_id": 2,
        "grouping": "monthly",
        "n_quantiles": 50,
        "extrapolation": "linear",
    }
    fitted_linear = fit_levels(obs_tmax, obs_tmin, mod_tmax, mod_tmin, cfg_linear)
    assert fitted_linear.identical_to_winner is False
    assert fitted_linear.winner_config_id == 2
    assert fitted_linear.level3_constant is not fitted_linear.level3_winner

    preds_linear = predict_levels(fitted_linear, mod_tmax, mod_tmin)
    assert "level3_constant" in preds_linear


def test_block_cv_leakage_by_construction() -> None:
    """Requirement 5b: Block CV leakage test by construction (1990-2011).

    bias is 1.0 only in 1995-1999 and 0.0 elsewhere.
    Out-of-sample prediction in 1995-1999 still carries +1.0 bias (it was held out).
    Level 1 fitted on the whole record reduces the bias (differs by > 0.1).
    """
    obs_tmax, obs_tmin = _make_synthetic_temperature_series(1990, 2011)
    mod_tmax = obs_tmax.copy()
    mod_tmin = obs_tmin.copy()

    mask_bias = (obs_tmax.index.year >= 1995) & (obs_tmax.index.year <= 1999)
    mod_tmax.loc[mask_bias] += 1.0
    mod_tmin.loc[mask_bias] += 1.0

    cfg_const = {
        "config_id": 1,
        "grouping": "monthly",
        "n_quantiles": 50,
        "extrapolation": "constant",
    }

    # 1. Out-of-sample block CV predictions
    preds_cv, _ = _block_cv_predictions(
        obs_tmax=obs_tmax,
        obs_tmin=obs_tmin,
        raw_tmax=mod_tmax,
        raw_tmin=mod_tmin,
        winner_config=cfg_const,
    )
    diff_cv = preds_cv["level1"].tmax.loc[mask_bias] - obs_tmax.loc[mask_bias]
    # Because training years had 0.0 bias, L1 correction was 0.0, holding +1.0 bias
    np.testing.assert_allclose(diff_cv.to_numpy(), 1.0, atol=1e-6)

    # 2. Whole-record fit
    m1_whole = Level1MonthlyAdditiveTemperature()
    m1_whole.fit(obs_tmax, obs_tmin, mod_tmax, mod_tmin)
    pred_whole_tmax, _ = m1_whole.transform(mod_tmax, mod_tmin)
    diff_whole = pred_whole_tmax.loc[mask_bias] - obs_tmax.loc[mask_bias]

    # Whole-record model learned 5/22 * 1.0 bias, so diff_whole is ~0.773
    assert float(np.abs(diff_cv.mean() - diff_whole.mean())) > 0.1


def test_primary_split_anti_leakage() -> None:
    """Requirement 5c: Primary split anti-leakage with contaminated validation.

    Evaluating two datasets with identical calibration but different validation
    contents produces identical Level 1 corrections and predictions.
    """
    obs_tmax, obs_tmin = _make_synthetic_temperature_series(1990, 2010)
    raw_tmax = obs_tmax + 2.0
    raw_tmin = obs_tmin - 1.0

    cal_range = (1990, 2004)
    val_range = (2005, 2010)

    # Dataset 1: clean
    # Dataset 2: contaminated validation
    raw2_tmax = raw_tmax.copy()
    raw2_tmin = raw_tmin.copy()
    val_mask = raw2_tmax.index.year.isin(range(val_range[0], val_range[1] + 1))
    raw2_tmax.loc[val_mask] += 50.0
    raw2_tmin.loc[val_mask] += 50.0

    cfg_const = {
        "config_id": 1,
        "grouping": "monthly",
        "n_quantiles": 50,
        "extrapolation": "constant",
    }

    cal_mask = obs_tmax.index.year.isin(range(cal_range[0], cal_range[1] + 1))
    fitted1 = fit_levels(
        obs_tmax[cal_mask],
        obs_tmin[cal_mask],
        raw_tmax[cal_mask],
        raw_tmin[cal_mask],
        cfg_const,
    )
    fitted2 = fit_levels(
        obs_tmax[cal_mask],
        obs_tmin[cal_mask],
        raw2_tmax[cal_mask],
        raw2_tmin[cal_mask],
        cfg_const,
    )

    pred1 = predict_levels(fitted1, raw_tmax[val_mask], raw_tmin[val_mask])
    pred2 = predict_levels(fitted2, raw_tmax[val_mask], raw_tmin[val_mask])

    np.testing.assert_allclose(
        pred1["level1"].tmax.to_numpy(), pred2["level1"].tmax.to_numpy(), atol=1e-9
    )
    np.testing.assert_allclose(
        pred1["level1"].tmin.to_numpy(), pred2["level1"].tmin.to_numpy(), atol=1e-9
    )


def test_block_cv_fold_abort_safeguard(monkeypatch: pytest.MonkeyPatch) -> None:
    """Requirement 5d: Fold abort returns SAMPLE_STATUS_ABORTED."""
    obs_tmax, obs_tmin = _make_synthetic_temperature_series(1990, 2011)
    raw_tmax = obs_tmax + 2.0
    raw_tmin = obs_tmin - 1.0

    cfg_const = {
        "config_id": 1,
        "grouping": "monthly",
        "n_quantiles": 50,
        "extrapolation": "constant",
    }

    orig_fit = Level1MonthlyAdditiveTemperature.fit

    def _mock_fit_fail(
        self: Level1MonthlyAdditiveTemperature,
        o_tmax: pd.Series,
        o_tmin: pd.Series,
        m_tmax: pd.Series,
        m_tmin: pd.Series,
    ) -> None:
        # Fail when test fold is 1995-1999 (meaning training excludes 1995-1999)
        if not o_tmax.index.year.isin([1995]).any():
            raise ValueError("Safeguard triggered in fold 1995-1999")
        orig_fit(self, o_tmax, o_tmin, m_tmax, m_tmin)

    monkeypatch.setattr(Level1MonthlyAdditiveTemperature, "fit", _mock_fit_fail)

    result = evaluate_block_cv(
        station_id="TEST_STATION",
        obs_tmax=obs_tmax,
        obs_tmin=obs_tmin,
        raw_tmax=raw_tmax,
        raw_tmin=raw_tmin,
        winner_config=cfg_const,
        n_replicas=20,
    )

    assert result.sample_status == SAMPLE_STATUS_ABORTED
    assert result.reason is not None
    assert "1995-1999" in result.reason
    assert all(sc is None for sc in result.scorecards.values())
    assert all(out == GlobalOutcome.NOT_ESTABLISHED for out in result.outcomes.values())


def test_block_cv_gap_in_calendar_years_raises() -> None:
    """Requirement 5e: Gap in calendar years raises ValueError in block CV."""
    obs_tmax, obs_tmin = _make_synthetic_temperature_series(1990, 2011)
    raw_tmax = obs_tmax + 2.0
    raw_tmin = obs_tmin - 1.0

    # Drop year 2000 to create a gap
    mask_gap = obs_tmax.index.year != 2000
    cfg_const = {
        "config_id": 1,
        "grouping": "monthly",
        "n_quantiles": 50,
        "extrapolation": "constant",
    }

    with pytest.raises(ValueError, match="gap"):
        evaluate_block_cv(
            station_id="TEST_STATION",
            obs_tmax=obs_tmax[mask_gap],
            obs_tmin=obs_tmin[mask_gap],
            raw_tmax=raw_tmax[mask_gap],
            raw_tmin=raw_tmin[mask_gap],
            winner_config=cfg_const,
            n_replicas=20,
        )


def test_level0_and_level3_constant_crossing_alert_internal() -> None:
    """Requirement 5f: Crossing alert computed internally for L0 and L3 constant."""
    obs_tmax, obs_tmin = _make_synthetic_temperature_series(2010, 2016)
    n_days = len(obs_tmax)
    cross_count = int(0.05 * n_days) + 1  # 5% crossings

    # 1. Raw series with crossings gives level0 crossing alert and FALSIFIED
    raw_cross_tmax = obs_tmax.copy()
    raw_cross_tmin = obs_tmin.copy()
    idx_cross = obs_tmax.index[:cross_count]
    raw_cross_tmax.loc[idx_cross] = 10.0
    raw_cross_tmin.loc[idx_cross] = 20.0

    pred_pass = LevelPrediction(tmax=obs_tmax, tmin=obs_tmin)
    preds = {
        "level1": pred_pass,
        "level2": pred_pass,
        "level3_winner": pred_pass,
    }

    res1 = evaluate_track_levels(
        station_id="TEST_STATION",
        obs_tmax=obs_tmax,
        obs_tmin=obs_tmin,
        raw_tmax=raw_cross_tmax,
        raw_tmin=raw_cross_tmin,
        predictions=preds,
        identical_to_winner=True,
        n_replicas=20,
        seed=42,
    )
    sc_l0 = res1.scorecards["level0"]
    assert sc_l0 is not None
    assert sc_l0.crossing_diagnostic is not None
    assert sc_l0.crossing_diagnostic.alert_triggered is True
    assert res1.outcomes["level0"] == GlobalOutcome.FALSIFIED

    # 2. Level 3 constant with crossing_diagnostic=None computes internally
    pred_l3c = LevelPrediction(
        tmax=raw_cross_tmax,
        tmin=raw_cross_tmin,
        crossing_diagnostic=None,
    )
    preds_l3c = dict(preds, level3_constant=pred_l3c)
    res2 = evaluate_track_levels(
        station_id="TEST_STATION",
        obs_tmax=obs_tmax,
        obs_tmin=obs_tmin,
        raw_tmax=obs_tmax,
        raw_tmin=obs_tmin,
        predictions=preds_l3c,
        identical_to_winner=False,
        n_replicas=20,
        seed=42,
    )
    sc_l3c = res2.scorecards["level3_constant"]
    assert sc_l3c is not None
    assert sc_l3c.crossing_diagnostic is not None
    assert sc_l3c.crossing_diagnostic.alert_triggered is True
    assert res2.outcomes["level3_constant"] == GlobalOutcome.FALSIFIED


def test_combine_tracks_truth_table() -> None:
    """Requirement 5g: combine_tracks truth table under weaker-of-the-two rule."""

    def _make_eval(outcomes: Dict[str, GlobalOutcome]) -> TrackEvaluation:
        return TrackEvaluation(
            station_id="TEST_STATION",
            valid_years=[2010, 2011, 2012, 2013, 2014, 2015, 2016],
            sample_status=SAMPLE_STATUS_OK,
            scorecards={k: None for k in outcomes},
            outcomes=outcomes,
            identical_to_winner=True,
            reason=None,
        )

    # 1. SUCCESS / SUCCESS -> SUCCESS
    p1 = _make_eval(
        {
            "level0": GlobalOutcome.SUCCESS,
            "level1": GlobalOutcome.SUCCESS,
            "level2": GlobalOutcome.SUCCESS,
            "level3_winner": GlobalOutcome.SUCCESS,
            "level3_constant": GlobalOutcome.SUCCESS,
        }
    )
    c1 = _make_eval(
        {
            "level0": GlobalOutcome.SUCCESS,
            "level1": GlobalOutcome.SUCCESS,
            "level2": GlobalOutcome.SUCCESS,
            "level3_winner": GlobalOutcome.SUCCESS,
            "level3_constant": GlobalOutcome.SUCCESS,
        }
    )
    assert combine_tracks(p1, c1)["level1"] == GlobalOutcome.SUCCESS

    # 2. SUCCESS / NOT_ESTABLISHED -> NOT_ESTABLISHED
    p2 = _make_eval(
        {
            "level0": GlobalOutcome.SUCCESS,
            "level1": GlobalOutcome.SUCCESS,
            "level2": GlobalOutcome.SUCCESS,
            "level3_winner": GlobalOutcome.SUCCESS,
            "level3_constant": GlobalOutcome.SUCCESS,
        }
    )
    c2 = _make_eval(
        {
            "level0": GlobalOutcome.NOT_ESTABLISHED,
            "level1": GlobalOutcome.NOT_ESTABLISHED,
            "level2": GlobalOutcome.NOT_ESTABLISHED,
            "level3_winner": GlobalOutcome.NOT_ESTABLISHED,
            "level3_constant": GlobalOutcome.NOT_ESTABLISHED,
        }
    )
    assert combine_tracks(p2, c2)["level1"] == GlobalOutcome.NOT_ESTABLISHED

    # 3. NOT_ESTABLISHED / FALSIFIED -> FALSIFIED
    p3 = _make_eval(
        {
            "level0": GlobalOutcome.NOT_ESTABLISHED,
            "level1": GlobalOutcome.NOT_ESTABLISHED,
            "level2": GlobalOutcome.NOT_ESTABLISHED,
            "level3_winner": GlobalOutcome.NOT_ESTABLISHED,
            "level3_constant": GlobalOutcome.NOT_ESTABLISHED,
        }
    )
    c3 = _make_eval(
        {
            "level0": GlobalOutcome.FALSIFIED,
            "level1": GlobalOutcome.FALSIFIED,
            "level2": GlobalOutcome.FALSIFIED,
            "level3_winner": GlobalOutcome.FALSIFIED,
            "level3_constant": GlobalOutcome.FALSIFIED,
        }
    )
    assert combine_tracks(p3, c3)["level1"] == GlobalOutcome.FALSIFIED


def test_levels_with_global_success() -> None:
    """Requirement 5h: Candidate levels with Global Success at every station."""
    outcomes = {
        "station_1": {
            "level0": GlobalOutcome.SUCCESS,
            "level1": GlobalOutcome.SUCCESS,
            "level2": GlobalOutcome.SUCCESS,
            "level3_winner": GlobalOutcome.SUCCESS,
            "level3_constant": GlobalOutcome.SUCCESS,
        },
        "station_2": {
            "level0": GlobalOutcome.SUCCESS,
            "level1": GlobalOutcome.SUCCESS,
            "level2": GlobalOutcome.FALSIFIED,
            "level3_winner": GlobalOutcome.SUCCESS,
            "level3_constant": GlobalOutcome.SUCCESS,
        },
    }

    successful = levels_with_global_success(outcomes)
    assert successful == ["level1", "level3_winner"]
    assert "level0" not in successful
    assert "level2" not in successful
    assert "level3_constant" not in successful


def test_primary_split_propagation_on_safeguard_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Requirement 5i: Primary split propagates ValueError immediately."""
    obs_tmax, obs_tmin = _make_synthetic_temperature_series(1990, 2010)
    raw_tmax = obs_tmax + 2.0
    raw_tmin = obs_tmin - 1.0

    cfg_const = {
        "config_id": 1,
        "grouping": "monthly",
        "n_quantiles": 50,
        "extrapolation": "constant",
    }

    def _mock_boom(*args: object, **kwargs: object) -> None:
        raise ValueError("boom")

    monkeypatch.setattr(Level1MonthlyAdditiveTemperature, "fit", _mock_boom)

    with pytest.raises(ValueError, match="boom"):
        evaluate_primary_split(
            station_id="TEST_STATION",
            obs_tmax=obs_tmax,
            obs_tmin=obs_tmin,
            raw_tmax=raw_tmax,
            raw_tmin=raw_tmin,
            cal_range=(1990, 2004),
            val_range=(2005, 2010),
            winner_config=cfg_const,
            n_replicas=20,
        )


def test_sample_status_constants() -> None:
    """Requirement 5j: SAMPLE_STATUS constants match track_sample_status."""
    assert SAMPLE_STATUS_INSUFFICIENT == track_sample_status(6)
    assert SAMPLE_STATUS_OK == track_sample_status(7)
    assert SAMPLE_STATUS_INSUFFICIENT == "not_established_insufficient_sample"
    assert SAMPLE_STATUS_OK == "ok"
    assert SAMPLE_STATUS_ABORTED == "aborted_safeguard"
