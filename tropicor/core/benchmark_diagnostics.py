"""Benchmark diagnostic helpers under Protocol Appendix K.5.2 (D1, D2, D3).

Provides pure diagnostic functions for Level 1 residual analysis:
- D1: Annual bias and Level 1 residual series
- D2: Level change between calibration and validation partitions
- D3: OLS trend of Level 1 residuals over included years
"""

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from tropicor.core.benchmark_evaluator import _require_aligned_index

DIAG_MIN_PAIRED_DAYS_PER_MONTH: int = 10
DIAG_MIN_VALID_MONTHS_PER_YEAR: int = 9
DIAG_BOOTSTRAP_REPLICAS: int = 10000
DIAG_SEED: int = 42

DIAG_STATUS_OK: str = "ok"
DIAG_STATUS_INSUFFICIENT: str = "insufficient_included_years"


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
