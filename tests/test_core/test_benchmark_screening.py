"""Tests for J.3.3 screening diagnostic counts and J.3.4 approval masking."""

import inspect
from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from tropicor.core.benchmark_screening import (
    MonthlyStats,
    ScreeningCounts,
    approval_label_counts,
    mask_to_definitive,
    monthly_stats,
    screening_counts,
)


def test_screening_outside_e42_limits() -> None:
    # E.4.2: Tmax in [5, 45] °C, Tmin in [-5, 35] °C (inclusive)
    # Outside: Tmax < 5.0 or > 45.0; Tmin < -5.0 or > 35.0
    dates = pd.date_range("2020-01-01", periods=6, freq="D")
    # Tmax: 5.0 (at lower limit, inside), 45.0 (at upper limit, inside),
    #       4.9 (below lower, outside), 45.1 (above upper, outside),
    #       25.0 (inside), NaN (ignored) -> outside count = 2
    tmax = pd.Series([5.0, 45.0, 4.9, 45.1, 25.0, np.nan], index=dates, name="Tmax")
    # Tmin: -5.0 (at lower limit, inside), 35.0 (at upper limit, inside),
    #       -5.1 (below lower, outside), 35.1 (above upper, outside),
    #       15.0 (inside), NaN (ignored) -> outside count = 2
    tmin = pd.Series([-5.0, 35.0, -5.1, 35.1, 15.0, np.nan], index=dates, name="Tmin")

    dummy_stats = MonthlyStats(
        mean=tuple([20.0] * 12),
        std=tuple([float("nan")] * 12),
        n=tuple([0] * 12),
    )
    res = screening_counts(tmax, tmin, dummy_stats, dummy_stats)

    assert res.n_valid_tmax == 5
    assert res.n_valid_tmin == 5
    assert res.n_outside_e42_tmax == 2
    assert res.n_outside_e42_tmin == 2


def test_screening_tmax_below_tmin() -> None:
    # Strict inequality Tmax < Tmin on paired days only
    dates = pd.date_range("2020-01-01", periods=6, freq="D")
    # Day 1: 20.0 < 25.0 -> paired and Tmax < Tmin (+1)
    # Day 2: 20.0 == 20.0 -> paired and equal (0, strict)
    # Day 3: 25.0 > 20.0 -> paired and Tmax > Tmin (0)
    # Day 4: NaN, 25.0 -> unpaired (0)
    # Day 5: 20.0, NaN -> unpaired (0)
    # Day 6: NaN, NaN -> unpaired (0)
    tmax = pd.Series([20.0, 20.0, 25.0, np.nan, 20.0, np.nan], index=dates, name="Tmax")
    tmin = pd.Series([25.0, 20.0, 20.0, 25.0, np.nan, np.nan], index=dates, name="Tmin")

    dummy_stats = MonthlyStats(
        mean=tuple([20.0] * 12),
        std=tuple([float("nan")] * 12),
        n=tuple([0] * 12),
    )
    res = screening_counts(tmax, tmin, dummy_stats, dummy_stats)

    assert res.n_paired == 3
    assert res.n_tmax_below_tmin == 1


def test_screening_runs_ge7_counts() -> None:
    # 1. Exactly 7 identical values: 1 run, 7 days
    d7 = pd.date_range("2020-01-01", periods=7, freq="D")
    s7 = pd.Series([20.0] * 7, index=d7)
    dummy_stats = MonthlyStats(
        mean=tuple([20.0] * 12),
        std=tuple([float("nan")] * 12),
        n=tuple([0] * 12),
    )
    r7 = screening_counts(s7, s7, dummy_stats, dummy_stats)
    assert r7.n_days_in_runs_ge7_tmax == 7

    # 2. Exactly 6 identical values: 0 runs, 0 days
    d6 = pd.date_range("2020-01-01", periods=6, freq="D")
    s6 = pd.Series([20.0] * 6, index=d6)
    r6 = screening_counts(s6, s6, dummy_stats, dummy_stats)
    assert r6.n_days_in_runs_ge7_tmax == 0

    # 3. NaN in the middle splits run (4 values, 1 NaN, 4 values): 0 runs, 0 days
    d_split = pd.date_range("2020-01-01", periods=9, freq="D")
    s_split = pd.Series(
        [20.0, 20.0, 20.0, 20.0, np.nan, 20.0, 20.0, 20.0, 20.0],
        index=d_split,
    )
    r_split = screening_counts(s_split, s_split, dummy_stats, dummy_stats)
    assert r_split.n_days_in_runs_ge7_tmax == 0

    # 4. Two separate runs (7 values of 20.0, 1 value of 25.0, 8 values of 30.0)
    d_two = pd.date_range("2020-01-01", periods=16, freq="D")
    s_two = pd.Series([20.0] * 7 + [25.0] + [30.0] * 8, index=d_two)
    r_two = screening_counts(s_two, s_two, dummy_stats, dummy_stats)
    assert r_two.n_days_in_runs_ge7_tmax == 15

    # 5. A run of 10 counts once (1 run, 10 days)
    d10 = pd.date_range("2020-01-01", periods=10, freq="D")
    s10 = pd.Series([22.0] * 10, index=d10)
    r10 = screening_counts(s10, s10, dummy_stats, dummy_stats)
    assert r10.n_days_in_runs_ge7_tmax == 10

    # 6. Calendar gaps (missing dates in index) break the run
    gap_dates = pd.DatetimeIndex(
        [
            "2020-01-01",
            "2020-01-02",
            "2020-01-03",
            "2020-01-04",
            "2020-01-06",
            "2020-01-07",
            "2020-01-08",
            "2020-01-09",
        ]
    )
    s_gap = pd.Series([20.0] * 8, index=gap_dates)
    r_gap = screening_counts(s_gap, s_gap, dummy_stats, dummy_stats)
    assert r_gap.n_days_in_runs_ge7_tmax == 0

    # 7. 7 identical values + 1 NaN + 7 identical values, contiguous dates -> 14 days
    d7_nan_7 = pd.date_range("2020-01-01", periods=15, freq="D")
    s7_nan_7 = pd.Series([20.0] * 7 + [np.nan] + [20.0] * 7, index=d7_nan_7)
    r7_nan_7 = screening_counts(s7_nan_7, s7_nan_7, dummy_stats, dummy_stats)
    assert r7_nan_7.n_days_in_runs_ge7_tmax == 14

    # 8. 7 identical values + date absent + 7 identical values -> 14 days
    d_gap_7_7 = pd.DatetimeIndex(
        [
            "2020-01-01",
            "2020-01-02",
            "2020-01-03",
            "2020-01-04",
            "2020-01-05",
            "2020-01-06",
            "2020-01-07",
            # 2020-01-08 absent
            "2020-01-09",
            "2020-01-10",
            "2020-01-11",
            "2020-01-12",
            "2020-01-13",
            "2020-01-14",
            "2020-01-15",
        ]
    )
    s_gap_7_7 = pd.Series([20.0] * 14, index=d_gap_7_7)
    r_gap_7_7 = screening_counts(s_gap_7_7, s_gap_7_7, dummy_stats, dummy_stats)
    assert r_gap_7_7.n_days_in_runs_ge7_tmax == 14

    # 9. 6 identical + date absent + 6 identical -> 0 days
    d_gap_6_6 = pd.DatetimeIndex(
        [
            "2020-01-01",
            "2020-01-02",
            "2020-01-03",
            "2020-01-04",
            "2020-01-05",
            "2020-01-06",
            # 2020-01-07 absent
            "2020-01-08",
            "2020-01-09",
            "2020-01-10",
            "2020-01-11",
            "2020-01-12",
            "2020-01-13",
        ]
    )
    s_gap_6_6 = pd.Series([20.0] * 12, index=d_gap_6_6)
    r_gap_6_6 = screening_counts(s_gap_6_6, s_gap_6_6, dummy_stats, dummy_stats)
    assert r_gap_6_6.n_days_in_runs_ge7_tmax == 0


def test_screening_6sigma_counts() -> None:
    # Build stats by hand with literals:
    # Month 1 (Jan): mean = 20.0, std = 2.0 -> 6*sigma = 12.0; bounds [8.0, 32.0]
    # Month 2 (Feb): n = 1, std = NaN (undefined)
    # Month 3 (Mar): n = 5, std = 0.0 (zero sigma)
    # Months 4-12: std = NaN
    means = [20.0, 15.0, 10.0] + [20.0] * 9
    stds = [2.0, float("nan"), 0.0] + [float("nan")] * 9
    ns = [100, 1, 5] + [0] * 9
    stats = MonthlyStats(mean=tuple(means), std=tuple(stds), n=tuple(ns))

    # Dates description:
    # Jan 1: 32.0 (exactly 6*sigma, |32-20| == 12) -> NOT counted (strict >)
    # Jan 2: 32.01 (just beyond, |32.01-20| == 12.01 > 12) -> COUNTED (+1)
    # Jan 3: 8.0 (exactly 6*sigma, |8-20| == 12) -> NOT counted
    # Jan 4: 7.99 (just beyond, |7.99-20| == 12.01 > 12) -> COUNTED (+1)
    # Jan 5: 20.0 (mean) -> NOT counted
    # Feb 1: Month 2 has NaN sigma -> NOT counted, month listed in undefined
    # Mar 1: Month 3 has 0.0 sigma -> NOT counted, month listed in undefined
    dates = pd.DatetimeIndex(
        [
            "2020-01-01",
            "2020-01-02",
            "2020-01-03",
            "2020-01-04",
            "2020-01-05",
            "2020-02-01",
            "2020-03-01",
        ]
    )
    vals = pd.Series([32.0, 32.01, 8.0, 7.99, 20.0, 100.0, 100.0], index=dates)

    res = screening_counts(vals, vals, stats, stats)

    assert res.n_beyond_6sigma_tmax == 2
    assert res.n_beyond_6sigma_tmin == 2
    expected_undefined = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
    assert res.sigma_undefined_months_tmax == expected_undefined
    assert res.sigma_undefined_months_tmin == expected_undefined


def test_monthly_stats_hand_calculation() -> None:
    # Month 1 (Jan): [10.0, 20.0, 30.0] -> mean = 20.0, std = 10.0 (ddof=1), n = 3
    # Month 2 (Feb): [15.0] -> mean = 15.0, std = NaN (n < 2), n = 1
    # Month 3 (Mar): [25.0, np.nan] -> mean = 25.0, std = NaN (n < 2), n = 1
    # Months 4-12: no records -> mean = NaN, std = NaN, n = 0
    dates = pd.DatetimeIndex(
        [
            "2020-01-01",
            "2020-01-02",
            "2020-01-03",
            "2020-02-01",
            "2020-03-01",
            "2020-03-02",
        ]
    )
    s = pd.Series([10.0, 20.0, 30.0, 15.0, 25.0, np.nan], index=dates)

    stats = monthly_stats(s)

    assert stats.mean[0] == pytest.approx(20.0)
    assert stats.std[0] == pytest.approx(10.0)
    assert stats.n[0] == 3

    assert stats.mean[1] == pytest.approx(15.0)
    assert np.isnan(stats.std[1])
    assert stats.n[1] == 1

    assert stats.mean[2] == pytest.approx(25.0)
    assert np.isnan(stats.std[2])
    assert stats.n[2] == 1

    for m in range(3, 12):
        assert np.isnan(stats.mean[m])
        assert np.isnan(stats.std[m])
        assert stats.n[m] == 0


def test_no_mutation_and_all_nan_input() -> None:
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    tmax = pd.Series([20.0, 22.0, 21.0, 23.0, 24.0], index=dates, name="Tmax")
    tmin = pd.Series([10.0, 12.0, 11.0, 13.0, 14.0], index=dates, name="Tmin")
    app = pd.Series(
        ["Definitivo", "Preliminar", "Definitivo", "Definitivo", "Definitivo"],
        index=dates,
    )

    tmax_copy = tmax.copy()
    tmin_copy = tmin.copy()
    app_copy = app.copy()

    stats_tmax = monthly_stats(tmax)
    stats_tmin = monthly_stats(tmin)
    screening_counts(tmax, tmin, stats_tmax, stats_tmin)
    approval_label_counts(tmax, app)
    mask_to_definitive(tmax, app)

    # Verify inputs are identical before and after every function
    assert tmax.equals(tmax_copy)
    assert tmin.equals(tmin_copy)
    assert app.equals(app_copy)

    # All-NaN input does not raise; counts are zero
    nan_s = pd.Series([np.nan] * 5, index=dates, name="AllNaN")
    nan_app = pd.Series([np.nan] * 5, index=dates, name="AllNaNApp")

    stats_nan = monthly_stats(nan_s)
    assert all(n == 0 for n in stats_nan.n)
    assert all(np.isnan(m) for m in stats_nan.mean)
    assert all(np.isnan(s) for s in stats_nan.std)

    res_nan = screening_counts(nan_s, nan_s, stats_nan, stats_nan)
    assert res_nan.n_valid_tmax == 0
    assert res_nan.n_valid_tmin == 0
    assert res_nan.n_paired == 0
    assert res_nan.n_outside_e42_tmax == 0
    assert res_nan.n_outside_e42_tmin == 0
    assert res_nan.n_tmax_below_tmin == 0
    assert res_nan.n_days_in_runs_ge7_tmax == 0
    assert res_nan.n_days_in_runs_ge7_tmin == 0
    assert res_nan.n_beyond_6sigma_tmax == 0
    assert res_nan.n_beyond_6sigma_tmin == 0

    app_nan_counts = approval_label_counts(nan_s, nan_app)
    assert app_nan_counts == {}

    masked_nan = mask_to_definitive(nan_s, nan_app)
    assert masked_nan.isna().all()
    assert masked_nan.name == "AllNaN"


def test_contracts_and_sentinel_invariance() -> None:
    dates1 = pd.date_range("2020-01-01", periods=5, freq="D")
    dates2 = pd.date_range("2020-01-02", periods=5, freq="D")

    sentinel_val = 25.5
    sentinel_label = "SECRET_SENTINEL_LABEL"

    s1 = pd.Series([sentinel_val, 20.0, 21.0, 22.0, 23.0], index=dates1)
    s2 = pd.Series([sentinel_val, 20.0, 21.0, 22.0, 23.0], index=dates2)

    # 1. Non-DatetimeIndex raises ValueError
    s_int = pd.Series([sentinel_val, 20.0, 21.0, 22.0, 23.0], index=pd.RangeIndex(5))
    with pytest.raises(ValueError) as exc1:
        monthly_stats(s_int)
    assert "25.5" not in str(exc1.value)

    # 2. Different indices raises ValueError
    stats1 = monthly_stats(s1)
    with pytest.raises(ValueError) as exc2:
        screening_counts(s1, s2, stats1, stats1)
    assert "25.5" not in str(exc2.value)

    with pytest.raises(ValueError) as exc3:
        approval_label_counts(s1, s2)
    assert "25.5" not in str(exc3.value)

    with pytest.raises(ValueError) as exc4:
        mask_to_definitive(s1, s2)
    assert "25.5" not in str(exc4.value)

    # 3. Sentinel label not in error message
    with pytest.raises(ValueError) as exc5:
        approval_label_counts(s1, pd.Series([sentinel_label] * 5, index=dates2))
    assert sentinel_label not in str(exc5.value)


def test_approval_label_counts() -> None:
    dates = pd.date_range("2020-01-01", periods=8, freq="D")
    vals = pd.Series([20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, np.nan], index=dates)
    apps = pd.Series(
        [
            "Preliminar",
            "Definitivo",
            " Definitivo ",
            "definitivo",
            np.nan,
            "",
            "   ",
            "Definitivo",  # Corresponding value is NaN -> not counted
        ],
        index=dates,
    )

    counts = approval_label_counts(vals, apps)

    # Hand-counted expected dictionary:
    expected = {
        "Preliminar": 1,
        "Definitivo": 2,
        "definitivo": 1,
        "<missing>": 3,
    }
    assert counts == expected


def test_mask_to_definitive() -> None:
    dates = pd.date_range("2020-01-01", periods=8, freq="D")
    vals = pd.Series(
        [20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, np.nan],
        index=dates,
        name="ObsTemp",
    )
    apps = pd.Series(
        [
            "Preliminar",
            "Definitivo",
            " Definitivo ",
            "definitivo",
            np.nan,
            "",
            "   ",
            "Definitivo",
        ],
        index=dates,
    )

    vals_copy = vals.copy()
    masked = mask_to_definitive(vals, apps)

    # Input unchanged and name preserved
    assert vals.equals(vals_copy)
    assert masked.name == "ObsTemp"

    # Only exact "Definitivo" survive (stripped " Definitivo " also matches)
    # Day 2 (21.0) and Day 3 (22.0) survive; all others are NaN
    assert masked.iloc[0] is np.nan or np.isnan(masked.iloc[0])
    assert masked.iloc[1] == pytest.approx(21.0)
    assert masked.iloc[2] == pytest.approx(22.0)
    assert masked.iloc[3] is np.nan or np.isnan(masked.iloc[3])
    assert masked.iloc[4] is np.nan or np.isnan(masked.iloc[4])
    assert masked.iloc[5] is np.nan or np.isnan(masked.iloc[5])
    assert masked.iloc[6] is np.nan or np.isnan(masked.iloc[6])
    assert masked.iloc[7] is np.nan or np.isnan(masked.iloc[7])

    # Number of survivors equals approval_label_counts()["Definitivo"]
    survivors = int(masked.dropna().count())
    app_counts = approval_label_counts(vals, apps)
    assert survivors == app_counts["Definitivo"]
    assert survivors == 2


def test_mask_to_definitive_empty_input() -> None:
    empty_idx = pd.DatetimeIndex([])
    v = pd.Series([], dtype=float, index=empty_idx, name="EmptyTemp")
    a = pd.Series([], dtype=object, index=empty_idx, name="EmptyApp")
    res = mask_to_definitive(v, a)
    assert len(res) == 0
    assert res.name == "EmptyTemp"
    assert isinstance(res.index, pd.DatetimeIndex)


def test_source_guard_pure_module() -> None:
    import tropicor.core.benchmark_screening as mod

    src = inspect.getsource(mod)
    forbidden = [
        "print(",
        "open(",
        "read_csv",
        "validation_openings",
        "validate_validation_access",
        "record_validation_opening",
        "StationCatalog",
        "logging",
    ]
    for term in forbidden:
        assert term not in src, f"Forbidden term '{term}' found in module source"


def test_dataclasses_frozen_and_repr() -> None:
    stats = MonthlyStats(
        mean=tuple([20.0] * 12),
        std=tuple([2.0] * 12),
        n=tuple([30] * 12),
    )
    counts = ScreeningCounts(
        n_valid_tmax=10,
        n_valid_tmin=10,
        n_paired=10,
        n_outside_e42_tmax=0,
        n_outside_e42_tmin=0,
        n_tmax_below_tmin=0,
        n_days_in_runs_ge7_tmax=0,
        n_days_in_runs_ge7_tmin=0,
        n_beyond_6sigma_tmax=0,
        n_beyond_6sigma_tmin=0,
        sigma_undefined_months_tmax=(),
        sigma_undefined_months_tmin=(),
    )

    # Immutability
    with pytest.raises(FrozenInstanceError):
        stats.mean = tuple([0.0] * 12)  # type: ignore[misc]

    with pytest.raises(FrozenInstanceError):
        counts.n_valid_tmax = 999  # type: ignore[misc]

    # Repr check: ints/floats/tuples, no Series
    r_stats = repr(stats)
    assert "Series" not in r_stats
    assert "pd." not in r_stats

    r_counts = repr(counts)
    assert "Series" not in r_counts
    assert "pd." not in r_counts
