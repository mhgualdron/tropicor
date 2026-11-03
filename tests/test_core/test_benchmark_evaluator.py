"""Unit tests for benchmark evaluator pure helpers (Appendix J and K)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tropicor.core.benchmark_evaluator import (
    BENCHMARK_STATION_IDS,
    five_year_blocks,
    mask_to_valid_years,
    track_sample_status,
    valid_years_j71,
)
from tropicor.core.protocol import (
    compute_annual_block_bootstrap_temperature,
    contiguous_5year_block_cv,
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
