"""Unit tests for benchmark evaluator pure helpers (Appendix J and K)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import tropicor.core.benchmark_evaluator as be
from tropicor.core.benchmark_evaluator import (
    BENCHMARK_STATION_IDS,
    LevelPrediction,
    TrackEvaluation,
    evaluate_track_levels,
    five_year_blocks,
    mask_to_valid_years,
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
    assert track_sample_status(6) == "not_established_insufficient_sample"
    assert track_sample_status(7) == "ok"
    assert track_sample_status(0) == "not_established_insufficient_sample"
    assert track_sample_status(8) == "ok"


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
    assert result.sample_status == "ok"
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

    assert result.sample_status == "insufficient_blocks"
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
    assert result.sample_status == "insufficient_blocks"
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
