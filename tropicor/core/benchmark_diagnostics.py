"""Benchmark diagnostic helpers under Protocol Appendix K.5.2 (D1, D2, D3, D4, D5, D6).

Provides pure diagnostic functions for Level 1 residual analysis:
- D1: Annual bias and Level 1 residual series
- D2: Level change between calibration and validation partitions
- D3: OLS trend of Level 1 residuals over included years
- D4: Best single step with permutation reference
- D5: Noise floor via random reassignment or full enumeration
- D6: Decomposition into offset and seasonal amplitude
"""

import itertools
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from tropicor.core.benchmark_evaluator import _require_aligned_index

DIAG_MIN_PAIRED_DAYS_PER_MONTH: int = 10
DIAG_MIN_VALID_MONTHS_PER_YEAR: int = 9
DIAG_BOOTSTRAP_REPLICAS: int = 10000
DIAG_NOISE_FLOOR_DRAWS: int = 999
DIAG_SEED: int = 42

DIAG_STATUS_OK: str = "ok"
DIAG_STATUS_INSUFFICIENT: str = "insufficient_included_years"
DIAG_STATUS_MONTH_WITHOUT_DAYS: str = "month_without_paired_days"


def annual_series_d1(
    obs: pd.Series,
    mod: pd.Series,
    calibration_monthly_bias: Sequence[float],
    cal_years: Sequence[int],
    val_years: Sequence[int],
) -> pd.DataFrame:
    """Compute annual bias and Level 1 residual series D1 for a single variable.

    Under Protocol Appendix K.5.1 and K.5.2:
    - Sign convention: ERA5 minus observation (mod - obs).
    - Inclusion rule: A month is valid if it contains >= 10 paired days. A year
      is included if it contains >= 9 valid months.
    - d_y: Mean over valid months of the monthly mean of (mod - obs).
    - r_y: Mean over valid months of (monthly mean of (mod - obs) minus b_m),
      where b_m is the calibration monthly bias of the official Level 1 fit.

    Parameters
    ----------
    obs : pd.Series
        Observed daily temperature series for a single variable.
    mod : pd.Series
        Model/reanalysis daily temperature series for a single variable.
    calibration_monthly_bias : Sequence[float]
        Exactly 12 finite monthly bias values b_m = mean(mod - obs) over the
        calibration partition, starting from month 1 (January).
    cal_years : Sequence[int]
        Calendar years belonging to the calibration partition.
    val_years : Sequence[int]
        Calendar years belonging to the validation partition.

    Returns
    -------
    pd.DataFrame
        DataFrame sorted by year with columns:
        - year (int)
        - partition (str: "calibration" or "validation")
        - n_paired_days (int)
        - valid_months (int)
        - included (bool)
        - exclusion_reason (str or None)
        - d_y (float or NaN)
        - r_y (float or NaN)

    Raises
    ------
    ValueError
        If series indexes are unaligned, partitions overlap, data contains
        years outside both partitions, or calibration_monthly_bias is invalid.
    """
    idx = _require_aligned_index(obs=obs, mod=mod)

    cal_set = set(cal_years)
    val_set = set(val_years)
    overlap = cal_set & val_set
    if overlap:
        raise ValueError(
            f"Calibration and validation partitions overlap: {sorted(overlap)}"
        )

    partition_years = cal_set | val_set
    data_years = set(idx.year)
    outside_years = data_years - partition_years
    if outside_years:
        raise ValueError(
            f"Years present in data outside both partitions: {sorted(outside_years)}"
        )

    if len(calibration_monthly_bias) != 12:
        raise ValueError(
            f"calibration_monthly_bias must have length 12, "
            f"got {len(calibration_monthly_bias)}."
        )
    bias_arr = np.asarray(calibration_monthly_bias, dtype=float)
    if not np.all(np.isfinite(bias_arr)):
        raise ValueError("calibration_monthly_bias contains non-finite values.")

    all_years = sorted(partition_years)
    rows: List[Dict[str, Any]] = []

    paired_valid = obs.notna() & mod.notna()

    for y in all_years:
        part = "calibration" if y in cal_set else "validation"
        year_mask = idx.year == y

        n_paired_days = int(paired_valid[year_mask].sum())

        valid_months = 0
        valid_d: List[float] = []
        valid_r: List[float] = []

        for m in range(1, 13):
            m_mask = year_mask & (idx.month == m) & paired_valid
            n_m_days = int(m_mask.sum())
            if n_m_days >= DIAG_MIN_PAIRED_DAYS_PER_MONTH:
                valid_months += 1
                diff_m = float((mod[m_mask] - obs[m_mask]).mean())
                b_m = bias_arr[m - 1]
                valid_d.append(diff_m)
                valid_r.append(diff_m - b_m)

        included = valid_months >= DIAG_MIN_VALID_MONTHS_PER_YEAR
        exclusion_reason: Optional[str]
        if included:
            exclusion_reason = None
            d_y = float(np.mean(valid_d))
            r_y = float(np.mean(valid_r))
        else:
            exclusion_reason = (
                f"fewer than {DIAG_MIN_VALID_MONTHS_PER_YEAR} valid months "
                f"({valid_months})"
            )
            d_y = float("nan")
            r_y = float("nan")

        rows.append(
            {
                "year": y,
                "partition": part,
                "n_paired_days": n_paired_days,
                "valid_months": valid_months,
                "included": included,
                "exclusion_reason": exclusion_reason,
                "d_y": d_y,
                "r_y": r_y,
            }
        )

    return pd.DataFrame(rows)


def level_change_d2(
    r_cal: Sequence[float] | np.ndarray,
    r_val: Sequence[float] | np.ndarray,
    n_replicas: int = DIAG_BOOTSTRAP_REPLICAS,
    seed: int = DIAG_SEED,
) -> Dict[str, Any]:
    """Compute Level 1 level change diagnostic D2 under Protocol Appendix K.5.2.

    Computes the difference in mean Level 1 residuals (validation minus calibration)
    over included years:
        point = mean(r_val) - mean(r_cal)

    The confidence interval is obtained via bootstrap resampling of years
    with replacement: drawing calibration years first, then validation years,
    using a single generator.

    Parameters
    ----------
    r_cal : Sequence[float] or np.ndarray
        Level 1 residuals r_y for included calibration years.
    r_val : Sequence[float] or np.ndarray
        Level 1 residuals r_y for included validation years.
    n_replicas : int, default=10000
        Number of bootstrap replicas.
    seed : int, default=42
        Random seed for the generator.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:
        - point: mean(r_val) - mean(r_cal)
        - ci_low: 2.5th percentile of bootstrap differences
        - ci_high: 97.5th percentile of bootstrap differences
        - n_cal: number of included calibration years
        - n_val: number of included validation years
        - status: DIAG_STATUS_OK or DIAG_STATUS_INSUFFICIENT

    Raises
    ------
    ValueError
        If r_cal or r_val contains NaN or non-finite values.
    """
    cal_arr = np.asarray(r_cal, dtype=float)
    val_arr = np.asarray(r_val, dtype=float)

    if np.isnan(cal_arr).any() or np.isnan(val_arr).any():
        raise ValueError("r_cal and r_val must not contain NaN values.")

    n_cal = len(cal_arr)
    n_val = len(val_arr)

    if n_cal == 0 or n_val == 0:
        # OPEN ITEM v1.10: K.5.2 does not specify the minimum number of included years for D2/D3; NaN returned with status  # noqa: E501
        return {
            "point": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_cal": n_cal,
            "n_val": n_val,
            "status": DIAG_STATUS_INSUFFICIENT,
        }

    point = float(np.mean(val_arr) - np.mean(cal_arr))

    rng = np.random.default_rng(seed)
    diffs: List[float] = []
    for _ in range(n_replicas):
        c_draw = rng.choice(cal_arr, size=n_cal, replace=True)
        v_draw = rng.choice(val_arr, size=n_val, replace=True)
        diffs.append(float(np.mean(v_draw) - np.mean(c_draw)))

    ci_low = float(np.percentile(diffs, 2.5))
    ci_high = float(np.percentile(diffs, 97.5))

    return {
        "point": point,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_cal": n_cal,
        "n_val": n_val,
        "status": DIAG_STATUS_OK,
    }


def trend_d3(
    years: Sequence[int | float] | np.ndarray,
    r_values: Sequence[float] | np.ndarray,
    n_replicas: int = DIAG_BOOTSTRAP_REPLICAS,
    seed: int = DIAG_SEED,
) -> Dict[str, Any]:
    """Compute Level 1 residual trend diagnostic D3 under Protocol Appendix K.5.2.

    Computes the ordinary least squares (OLS) slope of r_y on calendar year,
    expressed in K per decade (slope * 10).

    The confidence interval is obtained via paired bootstrap resampling of
    (year, r_y) pairs with replacement using a single generator. Replicas with
    fewer than 2 distinct years are counted as degenerate and skipped when
    computing percentiles.

    Parameters
    ----------
    years : Sequence[int or float] or np.ndarray
        Calendar years for included years across both partitions.
    r_values : Sequence[float] or np.ndarray
        Level 1 residuals r_y for included years across both partitions.
    n_replicas : int, default=10000
        Number of bootstrap replicas.
    seed : int, default=42
        Random seed for the generator.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:
        - slope_k_per_decade: OLS slope * 10
        - ci_low: 2.5th percentile of non-degenerate replica slopes
        - ci_high: 97.5th percentile of non-degenerate replica slopes
        - n_years: number of included years
        - n_degenerate_replicas: count of degenerate bootstrap draws
        - status: DIAG_STATUS_OK or DIAG_STATUS_INSUFFICIENT

    Raises
    ------
    ValueError
        If years and r_values have different lengths or contain NaN values.
    """
    y_arr = np.asarray(years, dtype=float)
    r_arr = np.asarray(r_values, dtype=float)

    if len(y_arr) != len(r_arr):
        raise ValueError(
            f"Mismatched lengths: {len(y_arr)} years vs {len(r_arr)} r_values."
        )

    if np.isnan(y_arr).any() or np.isnan(r_arr).any():
        raise ValueError("years and r_values must not contain NaN values.")

    n_years = len(y_arr)

    if n_years < 3:
        # OPEN ITEM v1.10: K.5.2 does not specify the minimum number of included years for D2/D3; NaN returned with status  # noqa: E501
        return {
            "slope_k_per_decade": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_years": n_years,
            "n_degenerate_replicas": 0,
            "status": DIAG_STATUS_INSUFFICIENT,
        }

    # OLS point slope: sum((y - y_bar) * r) / sum((y - y_bar)^2)
    dy = y_arr - np.mean(y_arr)
    denom = float(np.sum(dy * dy))
    if denom <= 0.0:
        point_slope = float("nan")
    else:
        point_slope = float(np.sum(dy * r_arr) / denom * 10.0)

    rng = np.random.default_rng(seed)
    slopes: List[float] = []
    n_degenerate = 0

    for _ in range(n_replicas):
        idx = rng.choice(n_years, size=n_years, replace=True)
        sub_y = y_arr[idx]
        sub_r = r_arr[idx]
        if len(set(sub_y)) < 2:
            n_degenerate += 1
            slopes.append(float("nan"))
        else:
            sub_dy = sub_y - np.mean(sub_y)
            sub_denom = float(np.sum(sub_dy * sub_dy))
            if sub_denom <= 0.0:
                n_degenerate += 1
                slopes.append(float("nan"))
            else:
                slopes.append(float(np.sum(sub_dy * sub_r) / sub_denom * 10.0))

    valid_slopes = [s for s in slopes if np.isfinite(s)]
    if not valid_slopes:
        ci_low = float("nan")
        ci_high = float("nan")
    else:
        # OPEN ITEM v1.10: K.5.2 D3 does not specify the handling of degenerate bootstrap replicas; NaN skipped and counted  # noqa: E501
        ci_low = float(np.nanpercentile(slopes, 2.5))
        ci_high = float(np.nanpercentile(slopes, 97.5))

    return {
        "slope_k_per_decade": point_slope,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_years": n_years,
        "n_degenerate_replicas": n_degenerate,
        "status": DIAG_STATUS_OK,
    }


def annual_dtr_bias_series(
    obs_tmax: pd.Series,
    obs_tmin: pd.Series,
    mod_tmax: pd.Series,
    mod_tmin: pd.Series,
    cal_years: Sequence[int],
    val_years: Sequence[int],
) -> pd.DataFrame:
    """Compute annual diurnal temperature range (DTR) bias series.

    Under Protocol Section K.4.4 and Appendix K.5.2:
    - This is NOT a reuse of annual_series_d1: it operates strictly on days
      where ALL FOUR values (obs_tmax, obs_tmin, mod_tmax, mod_tmin) are valid.
    - Sign convention: ERA5 minus observation
      ((mod_tmax - mod_tmin) - (obs_tmax - obs_tmin)).
    - Inclusion rule: A month is valid if it contains >= 10 four-valid days. A year
      is included if it contains >= 9 valid months.
    - Annual value: Mean over valid months of the monthly mean DTR bias.
      Raw bias only; no calibration monthly bias is subtracted.

    Parameters
    ----------
    obs_tmax : pd.Series
        Observed daily maximum temperature series.
    obs_tmin : pd.Series
        Observed daily minimum temperature series.
    mod_tmax : pd.Series
        Model/reanalysis daily maximum temperature series.
    mod_tmin : pd.Series
        Model/reanalysis daily minimum temperature series.
    cal_years : Sequence[int]
        Calendar years belonging to the calibration partition.
    val_years : Sequence[int]
        Calendar years belonging to the validation partition.

    Returns
    -------
    pd.DataFrame
        DataFrame sorted by year with columns:
        - year (int)
        - partition (str: "calibration" or "validation")
        - n_days_all_four (int)
        - valid_months (int)
        - included (bool)
        - exclusion_reason (str or None)
        - dtr_bias_y (float or NaN)

    Raises
    ------
    ValueError
        If series indexes are unaligned, partitions overlap, or data contains
        years outside both partitions.
    """
    idx = _require_aligned_index(
        obs_tmax=obs_tmax,
        obs_tmin=obs_tmin,
        mod_tmax=mod_tmax,
        mod_tmin=mod_tmin,
    )

    cal_set = set(cal_years)
    val_set = set(val_years)
    overlap = cal_set & val_set
    if overlap:
        raise ValueError(
            f"Calibration and validation partitions overlap: {sorted(overlap)}"
        )

    partition_years = cal_set | val_set
    data_years = set(idx.year)
    outside_years = data_years - partition_years
    if outside_years:
        raise ValueError(
            f"Years present in data outside both partitions: {sorted(outside_years)}"
        )

    all_years = sorted(partition_years)
    rows: List[Dict[str, Any]] = []

    all_four_valid = (
        obs_tmax.notna() & obs_tmin.notna() & mod_tmax.notna() & mod_tmin.notna()
    )

    dtr_mod = mod_tmax - mod_tmin
    dtr_obs = obs_tmax - obs_tmin
    dtr_diff = dtr_mod - dtr_obs

    for y in all_years:
        part = "calibration" if y in cal_set else "validation"
        year_mask = idx.year == y

        n_days_all_four = int(all_four_valid[year_mask].sum())

        valid_months = 0
        valid_dtr: List[float] = []

        for m in range(1, 13):
            m_mask = year_mask & (idx.month == m) & all_four_valid
            n_m_days = int(m_mask.sum())
            if n_m_days >= DIAG_MIN_PAIRED_DAYS_PER_MONTH:
                valid_months += 1
                diff_m = float(dtr_diff[m_mask].mean())
                valid_dtr.append(diff_m)

        included = valid_months >= DIAG_MIN_VALID_MONTHS_PER_YEAR
        exclusion_reason: Optional[str]
        if included:
            exclusion_reason = None
            dtr_bias_y = float(np.mean(valid_dtr))
        else:
            exclusion_reason = (
                f"fewer than {DIAG_MIN_VALID_MONTHS_PER_YEAR} valid months "
                f"({valid_months})"
            )
            dtr_bias_y = float("nan")

        rows.append(
            {
                "year": y,
                "partition": part,
                "n_days_all_four": n_days_all_four,
                "valid_months": valid_months,
                "included": included,
                "exclusion_reason": exclusion_reason,
                "dtr_bias_y": dtr_bias_y,
            }
        )

    return pd.DataFrame(rows)


def step_description_d4(
    years: Sequence[int | float] | np.ndarray,
    values: Sequence[float] | np.ndarray,
    n_permutations: int = 999,
    seed: int = DIAG_SEED,
    min_segment: int = 3,
) -> Dict[str, Any]:
    """Fit best single step model with permutation reference under Appendix K.5.2 D4.

    Finds candidate break minimizing total sum-of-squared errors (SSE) of
    two segment means, leaving at least min_segment years on each side.
    A permutation reference of n_permutations random orderings of values
    is computed using a single generator.

    Descriptive only: no test decision, no comparison with the D3 linear fit,
    no ranking of the two descriptions.

    Parameters
    ----------
    years : Sequence[int or float] or np.ndarray
        Strictly increasing calendar years for included years.
    values : Sequence[float] or np.ndarray
        Series values (e.g. r_y or dtr_bias_y) for included years.
    n_permutations : int, default=999
        Number of permutation iterations for reference rank.
    seed : int, default=42
        Random seed for the generator.
    min_segment : int, default=3
        Minimum number of included years required on each side of the break.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:
        - n_years: number of included years
        - break_year: first year of the second segment
        - segment1: tuple of (first_year, last_year) of first segment
        - segment2: tuple of (first_year, last_year) of second segment
        - mean1: mean of first segment
        - mean2: mean of second segment
        - step: mean2 - mean1
        - sse_constant: SSE of constant mean model
        - sse_step: SSE of two-segment step model
        - sse_reduction: sse_constant - sse_step
        - sse_reduction_pct: 100 * sse_reduction / sse_constant
        - rank_pct: percentage of permutation reductions <= real reduction
        - n_permutations: number of permutations evaluated
        - status: DIAG_STATUS_OK or DIAG_STATUS_INSUFFICIENT

    Raises
    ------
    ValueError
        If years and values have mismatched lengths, contain NaN values,
        or years are not strictly increasing without duplicates.
    """
    y_arr = np.asarray(years)
    v_arr = np.asarray(values, dtype=float)

    if len(y_arr) != len(v_arr):
        raise ValueError(
            f"Mismatched lengths: {len(y_arr)} years vs {len(v_arr)} values."
        )

    if np.isnan(v_arr).any() or pd.isna(y_arr).any():
        raise ValueError("years and values must not contain NaN values.")

    if len(y_arr) > 1:
        diffs = np.diff(y_arr)
        if np.any(diffs <= 0):
            raise ValueError("years must be strictly increasing without duplicates.")

    n_years = len(y_arr)

    if n_years < 2 * min_segment:
        # OPEN ITEM v1.10: K.5.2 does not specify the output of D4 with fewer than 6 included years; NaN returned with status  # noqa: E501
        return {
            "n_years": n_years,
            "break_year": float("nan"),
            "segment1": (float("nan"), float("nan")),
            "segment2": (float("nan"), float("nan")),
            "mean1": float("nan"),
            "mean2": float("nan"),
            "step": float("nan"),
            "sse_constant": float("nan"),
            "sse_step": float("nan"),
            "sse_reduction": float("nan"),
            "sse_reduction_pct": float("nan"),
            "rank_pct": float("nan"),
            "n_permutations": n_permutations,
            "status": DIAG_STATUS_INSUFFICIENT,
        }

    mu_0 = float(np.mean(v_arr))
    sse_0 = float(np.sum((v_arr - mu_0) ** 2))

    best_idx = min_segment
    best_sse = float("inf")
    best_m1 = float("nan")
    best_m2 = float("nan")

    for k in range(min_segment, n_years - min_segment + 1):
        g1, g2 = v_arr[:k], v_arr[k:]
        m1, m2 = float(np.mean(g1)), float(np.mean(g2))
        sse = float(np.sum((g1 - m1) ** 2) + np.sum((g2 - m2) ** 2))
        if sse < best_sse:
            best_sse = sse
            best_idx = k
            best_m1 = m1
            best_m2 = m2

    step_size = float(best_m2 - best_m1)
    red_step = float(sse_0 - best_sse)
    if sse_0 == 0.0:
        sse_red_pct = float("nan")
    else:
        sse_red_pct = float(100.0 * red_step / sse_0)

    break_year = y_arr[best_idx]
    segment1 = (y_arr[0], y_arr[best_idx - 1])
    segment2 = (y_arr[best_idx], y_arr[-1])

    # Permutation test
    rng_p = np.random.default_rng(seed)
    p_reds: List[float] = []
    for _ in range(n_permutations):
        p_vals = rng_p.permutation(v_arr)
        p_best = float("inf")
        for k in range(min_segment, n_years - min_segment + 1):
            g1, g2 = p_vals[:k], p_vals[k:]
            sse_p = float(
                np.sum((g1 - np.mean(g1)) ** 2) + np.sum((g2 - np.mean(g2)) ** 2)
            )
            if sse_p < p_best:
                p_best = sse_p
        p_reds.append(float(sse_0 - p_best))

    if n_permutations > 0:
        rank_pct = float(np.mean(np.array(p_reds) <= red_step) * 100.0)
    else:
        rank_pct = float("nan")

    return {
        "n_years": n_years,
        "break_year": break_year,
        "segment1": segment1,
        "segment2": segment2,
        "mean1": best_m1,
        "mean2": best_m2,
        "step": step_size,
        "sse_constant": sse_0,
        "sse_step": best_sse,
        "sse_reduction": red_step,
        "sse_reduction_pct": sse_red_pct,
        "rank_pct": rank_pct,
        "n_permutations": n_permutations,
        "status": DIAG_STATUS_OK,
    }


def _monthly_diff_sums_counts(
    obs: pd.Series,
    mod: pd.Series,
    years: Sequence[int] | np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-year, per-month sum of (mod - obs) and paired day counts.

    Parameters
    ----------
    obs : pd.Series
        Observed daily temperature series for a single variable.
    mod : pd.Series
        Model/reanalysis daily temperature series for a single variable.
    years : Sequence[int] or np.ndarray
        Years to evaluate. Other years present in the series are ignored.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Two arrays of shape (n_years, 12):
        - sums: sum of (mod - obs) for valid paired days per year and month.
        - counts: count of valid paired days per year and month.
    """
    n_years = len(years)
    sums = np.zeros((n_years, 12), dtype=float)
    counts = np.zeros((n_years, 12), dtype=int)
    if n_years == 0:
        return sums, counts

    idx = obs.index
    assert isinstance(idx, pd.DatetimeIndex)
    valid = obs.notna() & mod.notna()
    diff = (mod - obs).where(valid, 0.0)

    for i, y in enumerate(years):
        y_mask = (idx.year == y) & valid
        if not y_mask.any():
            continue
        sub_diff = diff[y_mask]
        sub_months = idx.month[y_mask]
        for m in range(1, 13):
            m_mask = sub_months == m
            n_days = int(m_mask.sum())
            counts[i, m - 1] = n_days
            if n_days > 0:
                sums[i, m - 1] = float(sub_diff[m_mask].sum())

    return sums, counts


def _level1_bias_from_years(
    sums: np.ndarray,
    counts: np.ndarray,
    row_idx: Sequence[int] | np.ndarray,
) -> np.ndarray:
    """Compute Level 1 additive monthly bias pooled over selected years.

    Parameters
    ----------
    sums : np.ndarray
        Array of shape (n_years, 12) with sums of differences.
    counts : np.ndarray
        Array of shape (n_years, 12) with valid day counts.
    row_idx : Sequence[int] or np.ndarray
        Row indices corresponding to the selected calibration years.

    Returns
    -------
    np.ndarray
        Array of 12 monthly bias values b_m = pooled mean(mod - obs).
        NaN for any calendar month with zero paired days.
    """
    if len(row_idx) == 0:
        return np.full(12, np.nan, dtype=float)
    sub_sums = sums[row_idx].sum(axis=0)
    sub_counts = counts[row_idx].sum(axis=0)
    b_m = np.full(12, np.nan, dtype=float)
    nonzero = sub_counts > 0
    b_m[nonzero] = sub_sums[nonzero] / sub_counts[nonzero]
    return b_m


def _validation_monthly_bias(
    sums: np.ndarray,
    counts: np.ndarray,
    row_idx: Sequence[int] | np.ndarray,
    b_cal: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Compute monthly validation bias after subtracting calibration monthly bias.

    Parameters
    ----------
    sums : np.ndarray
        Array of shape (n_years, 12) with sums of differences.
    counts : np.ndarray
        Array of shape (n_years, 12) with valid day counts.
    row_idx : Sequence[int] or np.ndarray
        Row indices corresponding to the selected validation years.
    b_cal : Sequence[float] or np.ndarray
        12 calibration monthly bias values.

    Returns
    -------
    np.ndarray
        Array of 12 monthly validation bias values: pooled mean(mod - obs)
        minus b_cal[m]. NaN for any calendar month with zero paired days
        or where b_cal[m] is NaN.
    """
    if len(row_idx) == 0:
        return np.full(12, np.nan, dtype=float)
    sub_sums = sums[row_idx].sum(axis=0)
    sub_counts = counts[row_idx].sum(axis=0)
    v_m = np.full(12, np.nan, dtype=float)
    b_arr = np.asarray(b_cal, dtype=float)
    for m in range(12):
        if sub_counts[m] == 0 or np.isnan(b_arr[m]):
            v_m[m] = np.nan
        else:
            val_mean = sub_sums[m] / sub_counts[m]
            v_m[m] = float(val_mean - b_arr[m])
    return v_m


def _decompose(bias_m: Sequence[float] | np.ndarray) -> Dict[str, float]:
    """Decompose 12 monthly validation biases into offset and seasonal amplitude.

    Parameters
    ----------
    bias_m : Sequence[float] or np.ndarray
        12 monthly validation bias values.

    Returns
    -------
    Dict[str, float]
        Dictionary with keys:
        - max_abs_bias: max over months of |bias_m|
        - offset: mean of the 12 monthly biases
        - abs_offset: |offset|
        - amplitude: max over months of |bias_m - offset|
        All values are NaN if any monthly bias is NaN.
    """
    arr = np.asarray(bias_m, dtype=float)
    if len(arr) != 12 or np.isnan(arr).any():
        return {
            "max_abs_bias": float("nan"),
            "offset": float("nan"),
            "abs_offset": float("nan"),
            "amplitude": float("nan"),
        }
    max_abs_bias = float(np.max(np.abs(arr)))
    offset = float(np.mean(arr))
    abs_offset = float(abs(offset))
    amplitude = float(np.max(np.abs(arr - offset)))
    return {
        "max_abs_bias": max_abs_bias,
        "offset": offset,
        "abs_offset": abs_offset,
        "amplitude": amplitude,
    }


def noise_floor_d5_d6(
    obs: pd.Series,
    mod: pd.Series,
    included_cal_years: Sequence[int],
    included_val_years: Sequence[int],
    n_draws: int = DIAG_NOISE_FLOOR_DRAWS,
    seed: int = DIAG_SEED,
) -> Dict[str, Any]:
    """Compute noise floor D5 and decomposition D6 for a single variable.

    Under Protocol Appendix K.5.1 and K.5.2:
    - Sign convention: ERA5 minus observation (mod - obs).
    - Excluded years present in the series are ignored; only the included years
      from the calibration and validation partitions are used.
    - The official decisive value computed under J.7.1 is NOT part of this
      function (the report assembler places it next to the recomputed real
      value with the number of years used in each, K.5.2 D5).
    - Diagnostics are descriptive: nothing here returns an outcome or can
      feed a GlobalOutcome, and it makes no comparison with the decisive
      threshold of Section 3.4.

    Parameters
    ----------
    obs : pd.Series
        Observed daily temperature series for a single variable.
    mod : pd.Series
        Model/reanalysis daily temperature series for a single variable.
    included_cal_years : Sequence[int]
        Included years from the calibration partition.
    included_val_years : Sequence[int]
        Included years from the validation partition.
    n_draws : int, default=DIAG_NOISE_FLOOR_DRAWS (999)
        Number of random reassignment draws when full enumeration is not used.
    seed : int, default=DIAG_SEED (42)
        Seed for the random number generator.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:
        - mode: "enumeration" or "random" (None if insufficient)
        - n_draws_used: number of draws evaluated
        - n_skipped_draws: number of draws skipped due to months without days
        - floor_max_abs_bias: dict with median, p5, p95, rank_pct
        - floor_abs_offset: dict with median, p5, p95, rank_pct
        - floor_amplitude: dict with median, p5, p95, rank_pct
        - real_monthly_bias: 12 monthly biases on validation set
        - real_max_abs_bias: max absolute validation monthly bias
        - real_offset: mean of the 12 monthly validation biases
        - real_abs_offset: absolute value of real_offset
        - real_amplitude: max |bias_m - offset| across months
        - n_cal_years: number of included calibration years
        - n_val_years: number of included validation years
        - status: DIAG_STATUS_OK, DIAG_STATUS_INSUFFICIENT, or
          DIAG_STATUS_MONTH_WITHOUT_DAYS

    Raises
    ------
    ValueError
        If obs and mod indices are not aligned, monotonic, and unique;
        if included_cal_years and included_val_years overlap;
        or if any included year is absent from the data index.
    """
    idx = _require_aligned_index(obs=obs, mod=mod)
    assert isinstance(idx, pd.DatetimeIndex)

    cal_set = set(included_cal_years)
    val_set = set(included_val_years)
    if bool(cal_set & val_set):
        raise ValueError("included_cal_years and included_val_years must not overlap")

    data_years = set(idx.year)
    for y in included_cal_years:
        if y not in data_years:
            raise ValueError(f"included calibration year {y} has no rows in data index")
    for y in included_val_years:
        if y not in data_years:
            raise ValueError(f"included validation year {y} has no rows in data index")

    k_cal = len(included_cal_years)
    k_val = len(included_val_years)

    def _nan_metric() -> Dict[str, float]:
        return {
            "median": float("nan"),
            "p5": float("nan"),
            "p95": float("nan"),
            "rank_pct": float("nan"),
        }

    if k_cal == 0 or k_val == 0:
        # OPEN ITEM v1.10: K.5.2 does not specify the minimum number of included years for D2/D3; NaN returned with status  # noqa: E501
        return {
            "mode": None,
            "n_draws_used": 0,
            "n_skipped_draws": 0,
            "floor_max_abs_bias": _nan_metric(),
            "floor_abs_offset": _nan_metric(),
            "floor_amplitude": _nan_metric(),
            "real_monthly_bias": [float("nan")] * 12,
            "real_max_abs_bias": float("nan"),
            "real_offset": float("nan"),
            "real_abs_offset": float("nan"),
            "real_amplitude": float("nan"),
            "n_cal_years": k_cal,
            "n_val_years": k_val,
            "status": DIAG_STATUS_INSUFFICIENT,
        }

    inc_years = np.array(sorted(cal_set | val_set))
    n = len(inc_years)

    sums, counts = _monthly_diff_sums_counts(obs, mod, inc_years)
    year_to_idx = {y: i for i, y in enumerate(inc_years)}
    cal_row_idx = np.array([year_to_idx[y] for y in included_cal_years], dtype=int)
    val_row_idx = np.array([year_to_idx[y] for y in included_val_years], dtype=int)

    # Real Level 1 and D6 decomposition
    b_cal = _level1_bias_from_years(sums, counts, cal_row_idx)
    v_val = _validation_monthly_bias(sums, counts, val_row_idx, b_cal)

    real_status = DIAG_STATUS_OK
    if np.isnan(v_val).any():
        # OPEN ITEM v1.10: K.5.2 D5/D6 do not specify the handling of a calendar month without paired days; NaN returned with status  # noqa: E501
        real_status = DIAG_STATUS_MONTH_WITHOUT_DAYS
        real_monthly_bias = [float("nan")] * 12
        real_max_abs_bias = float("nan")
        real_offset = float("nan")
        real_abs_offset = float("nan")
        real_amplitude = float("nan")
    else:
        decomp = _decompose(v_val)
        real_monthly_bias = [float(x) for x in v_val]
        real_max_abs_bias = decomp["max_abs_bias"]
        real_offset = decomp["offset"]
        real_abs_offset = decomp["abs_offset"]
        real_amplitude = decomp["amplitude"]

    # NOISE FLOOR simulation
    comb_count = math.comb(n, k_val)
    assignments: List[Tuple[np.ndarray, np.ndarray]] = []
    if comb_count <= n_draws:
        mode: Optional[str] = "enumeration"
        n_draws_used = comb_count
        for p_val in itertools.combinations(inc_years, k_val):
            p_val_set = set(p_val)
            p_cal = [y for y in inc_years if y not in p_val_set]
            p_cal_i = np.array([year_to_idx[y] for y in p_cal], dtype=int)
            p_val_i = np.array([year_to_idx[y] for y in p_val], dtype=int)
            assignments.append((p_cal_i, p_val_i))
    else:
        mode = "random"
        n_draws_used = n_draws
        rng = np.random.default_rng(seed)
        for _ in range(n_draws):
            perm = rng.permutation(inc_years)
            p_cal_rand = perm[:k_cal]
            p_val_rand = perm[k_cal : k_cal + k_val]
            p_cal_i = np.array([year_to_idx[y] for y in p_cal_rand], dtype=int)
            p_val_i = np.array([year_to_idx[y] for y in p_val_rand], dtype=int)
            assignments.append((p_cal_i, p_val_i))

    draw_max_b: List[float] = []
    draw_abs_off: List[float] = []
    draw_seas_amp: List[float] = []
    n_skipped_draws = 0

    for p_cal_i, p_val_i in assignments:
        b_pcal = _level1_bias_from_years(sums, counts, p_cal_i)
        v_pval = _validation_monthly_bias(sums, counts, p_val_i, b_pcal)
        if np.isnan(v_pval).any():
            # OPEN ITEM v1.10: K.5.2 D5/D6 do not specify the handling of a calendar month without paired days; NaN returned with status  # noqa: E501
            n_skipped_draws += 1
            continue
        dec = _decompose(v_pval)
        draw_max_b.append(dec["max_abs_bias"])
        draw_abs_off.append(dec["abs_offset"])
        draw_seas_amp.append(dec["amplitude"])

    def _calc_metric(
        draw_values: List[float],
        real_val: float,
    ) -> Dict[str, float]:
        if len(draw_values) == 0:
            return _nan_metric()
        arr = np.asarray(draw_values, dtype=float)
        med = float(np.median(arr))
        p5 = float(np.percentile(arr, 5))
        p95 = float(np.percentile(arr, 95))
        if np.isnan(real_val):
            rank = float("nan")
        else:
            # OPEN ITEM v1.10: K.5.2 D5 does not specify the percentile-rank operator; <= as in K.4.4 and D4  # noqa: E501
            rank = float(100.0 * np.mean(arr <= real_val))
        return {
            "median": med,
            "p5": p5,
            "p95": p95,
            "rank_pct": rank,
        }

    floor_max_abs_bias = _calc_metric(draw_max_b, real_max_abs_bias)
    floor_abs_offset = _calc_metric(draw_abs_off, real_abs_offset)
    floor_amplitude = _calc_metric(draw_seas_amp, real_amplitude)

    return {
        "mode": mode,
        "n_draws_used": n_draws_used,
        "n_skipped_draws": n_skipped_draws,
        "floor_max_abs_bias": floor_max_abs_bias,
        "floor_abs_offset": floor_abs_offset,
        "floor_amplitude": floor_amplitude,
        "real_monthly_bias": real_monthly_bias,
        "real_max_abs_bias": real_max_abs_bias,
        "real_offset": real_offset,
        "real_abs_offset": real_abs_offset,
        "real_amplitude": real_amplitude,
        "n_cal_years": k_cal,
        "n_val_years": k_val,
        "status": real_status,
    }
