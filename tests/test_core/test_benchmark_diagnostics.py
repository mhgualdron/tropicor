"""Unit tests for Benchmark Diagnostic helpers under Appendix K.5.2 (D1, D2, D3).

All tests use synthetic in-memory data only.
"""

import inspect
from typing import Any

import numpy as np
import pandas as pd
import pytest

import tropicor.core.benchmark_diagnostics as bdiag
from tropicor.core.benchmark_diagnostics import (
    DIAG_STATUS_INSUFFICIENT,
    DIAG_STATUS_OK,
    annual_series_d1,
    level_change_d2,
    trend_d3,
)


def _make_daily_index(start_year: int, end_year: int) -> pd.DatetimeIndex:
    """Create a daily DatetimeIndex spanning inclusive calendar years."""
    return pd.date_range(
        start=f"{start_year}-01-01",
        end=f"{end_year}-12-31",
        freq="D",
    )


def test_d1_inclusion_rule_boundaries() -> None:
    """Test 3a: D1 inclusion rule at boundaries (days per month, months per year)."""
    # 2001 (cal): 9 valid months (months 1-8 have 10 days, month 9 has 10 days,
    # month 10 has 9 days, months 11-12 have 0 days) -> included == True.
    # 2002 (cal): 8 valid months (months 1-8 have 10 days, month 9 has 9 days,
    # months 10-12 have 0 days) -> included == False.
    # 2003 (val): 0 paired days -> included == False.
    idx = _make_daily_index(2001, 2003)
    obs = pd.Series(np.nan, index=idx)
    mod = pd.Series(np.nan, index=idx)

    # 2001:
    for m in range(1, 9):
        # 10 paired days
        m_idx = idx[(idx.year == 2001) & (idx.month == m)][:10]
        obs.loc[m_idx] = 15.0
        mod.loc[m_idx] = 16.0
    # month 9: exactly 10 paired days (valid month boundary)
    m9_idx = idx[(idx.year == 2001) & (idx.month == 9)][:10]
    obs.loc[m9_idx] = 15.0
    mod.loc[m9_idx] = 16.0
    # month 10: exactly 9 paired days (invalid month boundary)
    m10_idx = idx[(idx.year == 2001) & (idx.month == 10)][:9]
    obs.loc[m10_idx] = 15.0
    mod.loc[m10_idx] = 16.0

    # 2002:
    for m in range(1, 9):
        m_idx = idx[(idx.year == 2002) & (idx.month == m)][:10]
        obs.loc[m_idx] = 15.0
        mod.loc[m_idx] = 16.0
    # month 9: exactly 9 paired days (invalid) -> total 8 valid months (year boundary)
    m9_02 = idx[(idx.year == 2002) & (idx.month == 9)][:9]
    obs.loc[m9_02] = 15.0
    mod.loc[m9_02] = 16.0

    bias = [0.0] * 12
    df = annual_series_d1(obs, mod, bias, cal_years=[2001, 2002], val_years=[2003])

    # Every year appears
    assert list(df["year"]) == [2001, 2002, 2003]
    assert list(df["partition"]) == ["calibration", "calibration", "validation"]

    # 2001: 9 valid months -> included
    row_2001 = df[df["year"] == 2001].iloc[0]
    assert row_2001["valid_months"] == 9
    assert row_2001["included"] is True or row_2001["included"] == 1
    assert row_2001["exclusion_reason"] is None or pd.isna(row_2001["exclusion_reason"])
    assert np.isfinite(row_2001["d_y"])
    assert np.isfinite(row_2001["r_y"])

    # 2002: 8 valid months -> excluded
    row_2002 = df[df["year"] == 2002].iloc[0]
    assert row_2002["valid_months"] == 8
    assert not row_2002["included"]
    assert row_2002["exclusion_reason"] is not None
    assert "8" in row_2002["exclusion_reason"]
    assert np.isnan(row_2002["d_y"])
    assert np.isnan(row_2002["r_y"])

    # 2003: 0 valid months -> excluded
    row_2003 = df[df["year"] == 2003].iloc[0]
    assert row_2003["valid_months"] == 0
    assert not row_2003["included"]
    assert row_2003["exclusion_reason"] is not None
    assert np.isnan(row_2003["d_y"])
    assert np.isnan(row_2003["r_y"])


def test_d1_sign_and_residual() -> None:
    """Test 3b: D1 sign convention (mod - obs) and residual r_y."""
    idx = _make_daily_index(2001, 2002)
    obs = pd.Series(10.0, index=idx)
    mod = pd.Series(12.0, index=idx)  # mod = obs + 2.0

    bias_2 = [2.0] * 12
    df1 = annual_series_d1(obs, mod, bias_2, cal_years=[2001], val_years=[2002])
    for _, row in df1.iterrows():
        assert row["included"] is True or row["included"] == 1
        assert abs(row["d_y"] - 2.0) < 1e-12
        assert abs(row["r_y"] - 0.0) < 1e-12

    bias_half = [0.5] * 12
    df2 = annual_series_d1(obs, mod, bias_half, cal_years=[2001], val_years=[2002])
    for _, row in df2.iterrows():
        assert abs(row["r_y"] - 1.5) < 1e-12


def test_d1_uses_provided_calibration_bias() -> None:
    """Test 3c: D1 uses provided calibration bias without recomputing."""
    idx = _make_daily_index(2001, 2001)
    obs = pd.Series(10.0, index=idx)
    # Months 1-10 present (10 valid months), months 11-12 NaN
    m11_12 = idx[idx.month >= 11]
    obs.loc[m11_12] = np.nan
    mod = obs + 1.0  # diff = 1.0 for valid months

    b_base = [1.0] * 12
    df_base = annual_series_d1(obs, mod, b_base, cal_years=[2001], val_years=[])
    r_base = df_base.iloc[0]["r_y"]
    assert abs(r_base - 0.0) < 1e-12
    assert df_base.iloc[0]["valid_months"] == 10

    # Modify month 1 bias by delta = 0.5
    delta = 0.5
    b_mod = b_base.copy()
    b_mod[0] += delta  # b_1 becomes 1.5 -> diff_1 - b_1 = 1.0 - 1.5 = -0.5
    df_mod = annual_series_d1(obs, mod, b_mod, cal_years=[2001], val_years=[])
    r_mod = df_mod.iloc[0]["r_y"]

    expected_change = delta / 10.0
    assert abs(r_base - r_mod - expected_change) < 1e-12


def test_d2_point_estimate_and_draw_order() -> None:
    """Test 3d: D2 point estimate on known series and pinned draw order."""
    r_cal = [0.1, 0.2, 0.3]
    r_val = [0.5, 0.7, 0.9, 1.1]

    # Point estimate: mean(val) - mean(cal) = 0.8 - 0.2 = 0.6
    res = level_change_d2(r_cal, r_val, n_replicas=200, seed=42)
    assert abs(res["point"] - 0.6) < 1e-12
    assert res["status"] == DIAG_STATUS_OK
    assert res["n_cal"] == 3
    assert res["n_val"] == 4

    # Recompute with exact documented draw order: cal draw then val draw
    rng_doc = np.random.default_rng(42)
    diffs_doc = []
    for _ in range(200):
        c = rng_doc.choice(r_cal, size=len(r_cal), replace=True)
        v = rng_doc.choice(r_val, size=len(r_val), replace=True)
        diffs_doc.append(float(np.mean(v) - np.mean(c)))
    expected_low = float(np.percentile(diffs_doc, 2.5))
    expected_high = float(np.percentile(diffs_doc, 97.5))

    assert res["ci_low"] == expected_low
    assert res["ci_high"] == expected_high

    # Swapped draw order: val draw then cal draw
    rng_swap = np.random.default_rng(42)
    diffs_swap = []
    for _ in range(200):
        v = rng_swap.choice(r_val, size=len(r_val), replace=True)
        c = rng_swap.choice(r_cal, size=len(r_cal), replace=True)
        diffs_swap.append(float(np.mean(v) - np.mean(c)))
    swap_low = float(np.percentile(diffs_swap, 2.5))
    swap_high = float(np.percentile(diffs_swap, 97.5))

    # Assert that swapping gives a different interval (order is pinned)
    assert (res["ci_low"] != swap_low) or (res["ci_high"] != swap_high)


def test_d2_determinism() -> None:
    """Test 3e: D2 determinism across identical runs and variance across seeds."""
    r_cal = [0.1, 0.2, 0.3]
    r_val = [0.5, 0.7, 0.9, 1.1]

    res1 = level_change_d2(r_cal, r_val, n_replicas=500, seed=42)
    res2 = level_change_d2(r_cal, r_val, n_replicas=500, seed=42)
    assert res1 == res2

    res_other = level_change_d2(r_cal, r_val, n_replicas=500, seed=99)
    assert (res1["ci_low"] != res_other["ci_low"]) or (
        res1["ci_high"] != res_other["ci_high"]
    )


def test_d3_slope_and_draw_check() -> None:
    """Test 3f (part 1): D3 slope on exact linear data and draw order check."""
    years = np.array([2000, 2001, 2002, 2003, 2004, 2005], dtype=float)
    # r = 0.05 * (year - 2000) -> slope per year = 0.05 -> slope per decade = 0.5
    r_values = 0.05 * (years - 2000.0)

    res = trend_d3(years, r_values, n_replicas=200, seed=42)
    assert abs(res["slope_k_per_decade"] - 0.5) < 1e-12
    assert res["ci_low"] <= 0.5 <= res["ci_high"]
    assert res["status"] == DIAG_STATUS_OK
    assert res["n_years"] == 6
    assert res["n_degenerate_replicas"] == 0

    # Draw check with n_replicas=200
    rng = np.random.default_rng(42)
    slopes = []
    for _ in range(200):
        idx = rng.choice(len(years), size=len(years), replace=True)
        sy = years[idx]
        sr = r_values[idx]
        if len(set(sy)) < 2:
            slopes.append(float("nan"))
        else:
            sdy = sy - np.mean(sy)
            sdenom = float(np.sum(sdy * sdy))
            slopes.append(float(np.sum(sdy * sr) / sdenom * 10.0))

    expected_low = float(np.nanpercentile(slopes, 2.5))
    expected_high = float(np.nanpercentile(slopes, 97.5))
    assert res["ci_low"] == expected_low
    assert res["ci_high"] == expected_high


@pytest.mark.filterwarnings("error")
def test_d3_degenerate_replicas_counting(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test 3f (part 2): Degenerate replica counting with warnings treated as errors."""
    years = [2000, 2001, 2002, 2003]
    r_values = [0.1, 0.2, 0.3, 0.4]

    class FakeConstantRNG:
        def choice(self, a: Any, size: Any = None, replace: bool = True) -> np.ndarray:
            return np.zeros(size, dtype=int)

    monkeypatch.setattr(
        bdiag.np.random,
        "default_rng",
        lambda seed=None: FakeConstantRNG(),
    )

    res = trend_d3(years, r_values, n_replicas=50)
    assert res["n_degenerate_replicas"] == 50
    assert np.isnan(res["ci_low"])
    assert np.isnan(res["ci_high"])
    assert res["status"] == DIAG_STATUS_OK


def test_error_policy_contract_violations() -> None:
    """Test 3g: Contract violations raise ValueError."""
    idx = _make_daily_index(2000, 2001)
    obs = pd.Series(10.0, index=idx)
    mod = pd.Series(10.0, index=idx)
    valid_bias = [0.0] * 12

    # 1. Overlapping partitions in D1
    with pytest.raises(ValueError, match="overlap"):
        annual_series_d1(obs, mod, valid_bias, cal_years=[2000, 2001], val_years=[2001])

    # 2. Year in data outside both partitions in D1
    with pytest.raises(ValueError, match="outside both partitions"):
        annual_series_d1(obs, mod, valid_bias, cal_years=[2000], val_years=[])

    # 3. Non-identical indices in D1 (dates shifted one day)
    shifted_mod = mod.copy()
    shifted_mod.index = shifted_mod.index + pd.Timedelta(days=1)
    with pytest.raises(ValueError):
        annual_series_d1(
            obs, shifted_mod, valid_bias, cal_years=[2000, 2001], val_years=[]
        )

    # 4. Calibration monthly bias length / finite violations (four sub-cases)
    # 4a. 11 values
    with pytest.raises(ValueError, match="length 12"):
        annual_series_d1(obs, mod, [0.0] * 11, cal_years=[2000, 2001], val_years=[])
    # 4b. 13 values
    with pytest.raises(ValueError, match="length 12"):
        annual_series_d1(obs, mod, [0.0] * 13, cal_years=[2000, 2001], val_years=[])
    # 4c. 12 values with NaN
    with pytest.raises(ValueError, match="non-finite"):
        annual_series_d1(
            obs,
            mod,
            [0.0] * 11 + [float("nan")],
            cal_years=[2000, 2001],
            val_years=[],
        )
    # 4d. 12 values with inf
    with pytest.raises(ValueError, match="non-finite"):
        annual_series_d1(
            obs,
            mod,
            [0.0] * 11 + [float("inf")],
            cal_years=[2000, 2001],
            val_years=[],
        )

    # 5. NaN in input of level_change_d2
    with pytest.raises(ValueError, match="NaN"):
        level_change_d2([float("nan")], [0.1])
    with pytest.raises(ValueError, match="NaN"):
        level_change_d2([0.1], [float("nan")])

    # 6. NaN in input of trend_d3
    with pytest.raises(ValueError, match="NaN"):
        trend_d3([2000, float("nan"), 2002], [0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="NaN"):
        trend_d3([2000, 2001, 2002], [0.1, float("nan"), 0.3])

    # 7. Mismatched lengths in trend_d3
    with pytest.raises(ValueError, match="Mismatched lengths"):
        trend_d3([2000, 2001], [0.1, 0.2, 0.3])


def test_error_policy_insufficient_data() -> None:
    """Test 3g: Insufficient data returns NaN with status (no exception)."""
    # Empty calibration set in D2
    res_cal_empty = level_change_d2([], [0.5, 0.6])
    assert np.isnan(res_cal_empty["point"])
    assert np.isnan(res_cal_empty["ci_low"])
    assert np.isnan(res_cal_empty["ci_high"])
    assert res_cal_empty["status"] == DIAG_STATUS_INSUFFICIENT
    assert res_cal_empty["n_cal"] == 0
    assert res_cal_empty["n_val"] == 2

    # Empty validation set in D2
    res_val_empty = level_change_d2([0.1, 0.2], [])
    assert np.isnan(res_val_empty["point"])
    assert np.isnan(res_val_empty["ci_low"])
    assert np.isnan(res_val_empty["ci_high"])
    assert res_val_empty["status"] == DIAG_STATUS_INSUFFICIENT
    assert res_val_empty["n_cal"] == 2
    assert res_val_empty["n_val"] == 0

    # Positive case for D2
    res_d2_ok = level_change_d2([0.1], [0.2], n_replicas=10)
    assert res_d2_ok["status"] == DIAG_STATUS_OK

    # 2 included years in D3 (< 3)
    res_d3_few = trend_d3([2000, 2001], [0.1, 0.2])
    assert np.isnan(res_d3_few["slope_k_per_decade"])
    assert np.isnan(res_d3_few["ci_low"])
    assert np.isnan(res_d3_few["ci_high"])
    assert res_d3_few["status"] == DIAG_STATUS_INSUFFICIENT
    assert res_d3_few["n_years"] == 2
    assert res_d3_few["n_degenerate_replicas"] == 0

    # Positive case for D3
    res_d3_ok = trend_d3([2000, 2001, 2002], [0.1, 0.2, 0.3], n_replicas=10)
    assert res_d3_ok["status"] == DIAG_STATUS_OK


def test_source_guard_benchmark_diagnostics() -> None:
    """Test 3h: Source guard against read_csv and ledger imports."""
    src = inspect.getsource(bdiag)
    assert "read_csv" not in src
    assert "validate_validation_access" not in src
    assert "record_validation_opening" not in src
