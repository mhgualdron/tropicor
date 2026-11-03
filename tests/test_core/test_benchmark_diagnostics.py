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
    annual_dtr_bias_series,
    annual_series_d1,
    level_change_d2,
    step_description_d4,
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


def test_dtr_series_four_valid_days_only() -> None:
    """Test 3a: DTR series uses only days where all four values are valid."""
    # 2001 (cal): Month 1 has 10 days with all 4 values valid, plus 10 days where
    # mod_tmax is NaN (partially valid). Months 2-9 have 10 four-valid days each.
    # Total valid months = 9 -> included.
    # 2002 (val): Month 1 has 9 four-valid days plus 15 partially valid days.
    # Months 2-9 have 10 four-valid days each (8 valid months).
    # Total valid months = 8 -> excluded.
    idx = _make_daily_index(2001, 2002)
    obs_tmax = pd.Series(np.nan, index=idx)
    obs_tmin = pd.Series(np.nan, index=idx)
    mod_tmax = pd.Series(np.nan, index=idx)
    mod_tmin = pd.Series(np.nan, index=idx)

    # 2001 month 1: 10 four-valid days
    m1_01_all = idx[(idx.year == 2001) & (idx.month == 1)][:10]
    obs_tmax.loc[m1_01_all] = 25.0
    obs_tmin.loc[m1_01_all] = 15.0
    mod_tmax.loc[m1_01_all] = 26.0
    mod_tmin.loc[m1_01_all] = 15.0

    # 2001 month 1: 10 partially valid days (mod_tmax is NaN)
    m1_01_part = idx[(idx.year == 2001) & (idx.month == 1)][10:20]
    obs_tmax.loc[m1_01_part] = 25.0
    obs_tmin.loc[m1_01_part] = 15.0
    mod_tmin.loc[m1_01_part] = 15.0
    # mod_tmax is left NaN

    # 2001 months 2-9: 10 four-valid days each
    for m in range(2, 10):
        m_idx = idx[(idx.year == 2001) & (idx.month == m)][:10]
        obs_tmax.loc[m_idx] = 25.0
        obs_tmin.loc[m_idx] = 15.0
        mod_tmax.loc[m_idx] = 26.0
        mod_tmin.loc[m_idx] = 15.0

    # 2002 month 1: exactly 9 four-valid days
    m1_02_all = idx[(idx.year == 2002) & (idx.month == 1)][:9]
    obs_tmax.loc[m1_02_all] = 25.0
    obs_tmin.loc[m1_02_all] = 15.0
    mod_tmax.loc[m1_02_all] = 26.0
    mod_tmin.loc[m1_02_all] = 15.0

    # 2002 month 1: 15 partially valid days (mod_tmax is NaN)
    m1_02_part = idx[(idx.year == 2002) & (idx.month == 1)][9:24]
    obs_tmax.loc[m1_02_part] = 25.0
    obs_tmin.loc[m1_02_part] = 15.0
    mod_tmin.loc[m1_02_part] = 15.0

    # 2002 months 2-9: 10 four-valid days each
    for m in range(2, 10):
        m_idx = idx[(idx.year == 2002) & (idx.month == m)][:10]
        obs_tmax.loc[m_idx] = 25.0
        obs_tmin.loc[m_idx] = 15.0
        mod_tmax.loc[m_idx] = 26.0
        mod_tmin.loc[m_idx] = 15.0

    df = annual_dtr_bias_series(
        obs_tmax, obs_tmin, mod_tmax, mod_tmin, cal_years=[2001], val_years=[2002]
    )

    row_2001 = df[df["year"] == 2001].iloc[0]
    assert row_2001["valid_months"] == 9
    assert row_2001["included"] is True or row_2001["included"] == 1
    assert row_2001["exclusion_reason"] is None or pd.isna(row_2001["exclusion_reason"])
    assert np.isfinite(row_2001["dtr_bias_y"])

    row_2002 = df[df["year"] == 2002].iloc[0]
    assert row_2002["valid_months"] == 8
    assert not row_2002["included"]
    assert pd.notna(row_2002["exclusion_reason"])
    assert np.isnan(row_2002["dtr_bias_y"])


def test_dtr_sign_and_raw_bias() -> None:
    """Test 3b: DTR sign and raw bias without calibration bias subtraction."""
    idx = _make_daily_index(2001, 2001)
    obs_tmax = pd.Series(20.0, index=idx)
    obs_tmin = pd.Series(10.0, index=idx)  # obs DTR = 10.0
    mod_tmax = pd.Series(21.0, index=idx)  # mod_tmax = obs_tmax + 1.0
    mod_tmin = pd.Series(10.0, index=idx)  # mod_tmin = obs_tmin -> mod DTR = 11.0

    df = annual_dtr_bias_series(
        obs_tmax, obs_tmin, mod_tmax, mod_tmin, cal_years=[2001], val_years=[]
    )
    row = df.iloc[0]
    assert row["included"] is True or row["included"] == 1
    # ERA5 DTR minus observed DTR = 11.0 - 10.0 = 1.0
    assert abs(row["dtr_bias_y"] - 1.0) < 1e-12


def test_dtr_series_contract_errors() -> None:
    """Test 3c: Contract errors for DTR series (index, overlap, outside years)."""
    idx = _make_daily_index(2001, 2002)
    obs_tmax = pd.Series(20.0, index=idx)
    obs_tmin = pd.Series(10.0, index=idx)
    mod_tmax = pd.Series(20.0, index=idx)
    mod_tmin = pd.Series(10.0, index=idx)

    # 1. Shifted index
    shifted_mod_tmax = mod_tmax.copy()
    shifted_mod_tmax.index = shifted_mod_tmax.index + pd.Timedelta(days=1)
    with pytest.raises(ValueError):
        annual_dtr_bias_series(
            obs_tmax, obs_tmin, shifted_mod_tmax, mod_tmin, [2001], [2002]
        )

    # 2. Overlapping partitions
    with pytest.raises(ValueError, match="overlap"):
        annual_dtr_bias_series(
            obs_tmax, obs_tmin, mod_tmax, mod_tmin, [2001, 2002], [2002]
        )

    # 3. Year outside both partitions
    with pytest.raises(ValueError, match="outside both partitions"):
        annual_dtr_bias_series(obs_tmax, obs_tmin, mod_tmax, mod_tmin, [2001], [])


def test_d4_exact_step() -> None:
    """Test 3d: D4 exact step with known step structure and 100% rank."""
    years = list(range(2000, 2011))  # 11 years: 2000..2010
    values = [0.0] * 5 + [1.0] * 6  # 5 zeros, 6 ones

    res = step_description_d4(years, values, n_permutations=100)
    assert res["n_years"] == 11
    assert res["break_year"] == years[5]  # 6th year: 2005
    assert res["segment1"] == (2000, 2004)
    assert res["segment2"] == (2005, 2010)
    assert abs(res["mean1"] - 0.0) < 1e-12
    assert abs(res["mean2"] - 1.0) < 1e-12
    assert abs(res["step"] - 1.0) < 1e-12
    assert abs(res["sse_step"] - 0.0) < 1e-12
    assert abs(res["sse_reduction_pct"] - 100.0) < 1e-12
    assert abs(res["rank_pct"] - 100.0) < 1e-12
    assert res["status"] == DIAG_STATUS_OK


def test_d4_independent_brute_force() -> None:
    """Test 3e: D4 independent brute force verification on fixed random data."""
    years = list(range(2000, 2014))  # 14 values
    rng = np.random.default_rng(7)
    values = rng.normal(size=14)

    mu_0 = float(np.mean(values))
    sse_0 = float(np.sum((values - mu_0) ** 2))
    best_sse = float("inf")
    best_k = 3
    best_m1, best_m2 = 0.0, 0.0

    for k in range(3, 14 - 2):
        g1 = values[:k]
        g2 = values[k:]
        m1 = float(np.mean(g1))
        m2 = float(np.mean(g2))
        sse = float(np.sum((g1 - m1) ** 2) + np.sum((g2 - m2) ** 2))
        if sse < best_sse:
            best_sse = sse
            best_k = k
            best_m1 = m1
            best_m2 = m2

    expected_break_year = years[best_k]
    expected_step = best_m2 - best_m1
    expected_reduction = sse_0 - best_sse

    res = step_description_d4(years, values, n_permutations=10)
    assert res["break_year"] == expected_break_year
    assert abs(res["step"] - expected_step) < 1e-12
    assert abs(res["sse_reduction"] - expected_reduction) < 1e-12


def test_d4_minimum_segment_pinned() -> None:
    """Test 3f: D4 minimum segment constraints (>= 3 on each side)."""
    years = list(range(2000, 2008))  # 8 years
    # Extreme jump at index 2 (unconstrained best break would be at index 2)
    values = [0.0, 0.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]

    res = step_description_d4(years, values, min_segment=3)
    assert res["break_year"] == 2003  # index 3, leaving 3 on left
    assert res["segment1"] == (2000, 2002)
    assert res["segment2"] == (2003, 2007)
    # Check both segments have at least 3 years
    assert res["segment1"][1] - res["segment1"][0] + 1 >= 3
    assert res["segment2"][1] - res["segment2"][0] + 1 >= 3


def test_d4_permutation_draw_check_and_determinism() -> None:
    """Test 3g: D4 permutation draw order check and determinism."""
    rng_init = np.random.default_rng(123)
    years = list(range(2000, 2012))
    val_arr = rng_init.normal(size=len(years))

    res = step_description_d4(years, val_arr, n_permutations=50, seed=42)

    # Independent computation with same sequence of draws
    n = len(val_arr)
    mu_0 = float(np.mean(val_arr))
    sse_0 = float(np.sum((val_arr - mu_0) ** 2))
    best_sse = float("inf")
    for k in range(3, n - 2):
        g1, g2 = val_arr[:k], val_arr[k:]
        sse = float(np.sum((g1 - np.mean(g1)) ** 2) + np.sum((g2 - np.mean(g2)) ** 2))
        if sse < best_sse:
            best_sse = sse
    real_red = sse_0 - best_sse

    rng = np.random.default_rng(42)
    p_reds = []
    for _ in range(50):
        p = rng.permutation(val_arr)
        p_best = float("inf")
        for k in range(3, n - 2):
            g1, g2 = p[:k], p[k:]
            sse_p = float(
                np.sum((g1 - np.mean(g1)) ** 2) + np.sum((g2 - np.mean(g2)) ** 2)
            )
            if sse_p < p_best:
                p_best = sse_p
        p_reds.append(float(sse_0 - p_best))

    expected_rank = float(np.mean(np.array(p_reds) <= real_red) * 100.0)
    assert res["rank_pct"] == expected_rank

    # Determinism
    res2 = step_description_d4(years, val_arr, n_permutations=50, seed=42)
    assert res == res2

    # Different seed gives different permutation sequence
    res_other = step_description_d4(years, val_arr, n_permutations=50, seed=99)
    assert res["rank_pct"] != res_other["rank_pct"]


def test_d4_insufficient_data() -> None:
    """Test 3h: D4 insufficient data policy (< 6 years -> NaN and status)."""
    years5 = list(range(2000, 2005))  # 5 years (< 2 * 3)
    values5 = [0.1, 0.2, 0.3, 0.4, 0.5]

    res5 = step_description_d4(years5, values5)
    assert res5["status"] == DIAG_STATUS_INSUFFICIENT
    assert res5["n_years"] == 5
    assert np.isnan(res5["break_year"])
    assert np.isnan(res5["step"])
    assert np.isnan(res5["mean1"])
    assert np.isnan(res5["mean2"])
    assert np.isnan(res5["sse_constant"])
    assert np.isnan(res5["sse_step"])
    assert np.isnan(res5["sse_reduction"])
    assert np.isnan(res5["sse_reduction_pct"])
    assert np.isnan(res5["rank_pct"])
    assert np.isnan(res5["segment1"][0]) and np.isnan(res5["segment1"][1])
    assert np.isnan(res5["segment2"][0]) and np.isnan(res5["segment2"][1])

    # 6 years -> OK
    years6 = list(range(2000, 2006))
    values6 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    res6 = step_description_d4(years6, values6)
    assert res6["status"] == DIAG_STATUS_OK
    assert res6["n_years"] == 6
    assert not np.isnan(res6["step"])


def test_d4_contract_errors() -> None:
    """Test 3i: Contract errors in D4 (NaN, mismatched lengths, invalid years)."""
    # NaN in values
    with pytest.raises(ValueError, match="NaN"):
        step_description_d4(
            list(range(2000, 2006)), [0.1, float("nan"), 0.3, 0.4, 0.5, 0.6]
        )

    # Mismatched lengths
    with pytest.raises(ValueError, match="Mismatched lengths"):
        step_description_d4([2000, 2001], [0.1, 0.2, 0.3])

    # Duplicate years
    with pytest.raises(ValueError, match="strictly increasing"):
        step_description_d4([2000, 2001, 2001, 2003, 2004, 2005], [0.1] * 6)

    # Decreasing years
    with pytest.raises(ValueError, match="strictly increasing"):
        step_description_d4([2000, 2002, 2001, 2003, 2004, 2005], [0.1] * 6)


def test_source_guard_benchmark_diagnostics_extended() -> None:
    """Test 3j: Source guard extended against read_csv and ledger imports."""
    src = inspect.getsource(bdiag)
    assert "read_csv" not in src
    assert "validate_validation_access" not in src
    assert "record_validation_opening" not in src
