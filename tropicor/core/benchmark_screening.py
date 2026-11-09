"""Screening diagnostic counts and approval masking for benchmark evaluation.

This module provides pure functions implementing:
- Section J.3.3 screening diagnostic counts (outside Section E.4.2 physical
  screening limits, observed Tmax < Tmin, runs of >= 7 consecutive identical
  values, and anomalies beyond 6 calibration monthly standard deviations).
- Section J.3.4 approval masking and label counts (Definitivo record filtering).

All functions are pure, perform no I/O, maintain documentation integrity, and
never raise exceptions based on observation data values.
"""

from dataclasses import dataclass
from typing import Any, Optional, cast

import numpy as np
import pandas as pd

from tropicor.core.benchmark_evaluator import _require_aligned_index

# Screening limits for daily temperature: Section E.4 item 2 (cited as "Section E.4.2" in J.3.3); closed intervals, Tmax in [5, 45] C and Tmin in [-5, 35] C.  # noqa: E501
# OPEN ITEM v1.10 (editorial): J.3.3 cites "Section E.4.2", which is not a heading but item 2 of Section E.4; the limits appear there inside a finding about another station, not as a stated rule  # noqa: E501
E_4_2_TMAX_MIN_C: float = 5.0
E_4_2_TMAX_MAX_C: float = 45.0
E_4_2_TMIN_MIN_C: float = -5.0
E_4_2_TMIN_MAX_C: float = 35.0


@dataclass(frozen=True)
class MonthlyStats:
    """Monthly calibration statistics for calendar months 1-12 (index 0 = January).

    Parameters
    ----------
    mean : tuple[float, ...]
        12 entries with monthly calibration means.
    std : tuple[float, ...]
        12 entries with monthly calibration standard deviations (ddof=1).
    n : tuple[int, ...]
        12 entries with monthly calibration valid sample counts.
    """

    mean: tuple[float, ...]
    std: tuple[float, ...]
    n: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.mean) != 12 or len(self.std) != 12 or len(self.n) != 12:
            raise ValueError("MonthlyStats tuples must have exactly 12 entries.")


@dataclass(frozen=True)
class ScreeningCounts:
    """Screening diagnostic day counts for daily temperature series (Section J.3.3).

    All counts are non-negative integer day counts. Undefined months are reported
    as tuples of 1-based calendar month indices.
    """

    n_valid_tmax: int
    n_valid_tmin: int
    n_paired: int
    n_outside_e42_tmax: int
    n_outside_e42_tmin: int
    n_tmax_below_tmin: int
    n_days_in_runs_ge7_tmax: int
    n_days_in_runs_ge7_tmin: int
    n_beyond_6sigma_tmax: int
    n_beyond_6sigma_tmin: int
    sigma_undefined_months_tmax: tuple[int, ...]
    sigma_undefined_months_tmin: tuple[int, ...]


def monthly_stats(series: pd.Series) -> MonthlyStats:
    """Compute per-calendar-month mean and standard deviation of calibration series.

    Parameters
    ----------
    series : pd.Series
        Calibration temperature series with DatetimeIndex.

    Returns
    -------
    MonthlyStats
        Per-month statistics across calendar months 1-12 (index 0 = January).
    """
    _require_aligned_index(series=series)

    means: list[float] = []
    stds: list[float] = []
    ns: list[int] = []

    dt_index = cast(pd.DatetimeIndex, series.index)
    months = dt_index.month

    # OPEN ITEM v1.10: J.3.3 does not state the SD convention for sigma_m; ddof=1
    for m in range(1, 13):
        m_mask = months == m
        m_vals = series[m_mask].dropna().to_numpy(dtype=np.float64)
        n_m = len(m_vals)
        ns.append(n_m)
        if n_m == 0:
            means.append(float("nan"))
            stds.append(float("nan"))
        elif n_m == 1:
            means.append(float(m_vals[0]))
            stds.append(float("nan"))
        else:
            means.append(float(np.mean(m_vals)))
            stds.append(float(np.std(m_vals, ddof=1)))

    return MonthlyStats(mean=tuple(means), std=tuple(stds), n=tuple(ns))


# OPEN ITEM v1.10: J.3.3 does not state whether a missing calendar date or a NaN breaks a run of consecutive identical values; runs are calendar-consecutive observed days, a missing date or NaN ends the run  # noqa: E501
def _count_days_in_runs_ge7(series: pd.Series) -> int:
    """Count days in maximal runs of >= 7 consecutive days with identical value.

    Parameters
    ----------
    series : pd.Series
        Temperature series with DatetimeIndex.

    Returns
    -------
    int
        Total number of days belonging to runs of >= 7 consecutive identical values.
    """
    if series.empty:
        return 0

    dt_index = cast(pd.DatetimeIndex, series.index)
    vals = series.to_numpy()

    n_days = 0

    current_val: Any = None
    current_len: int = 0
    prev_date: Optional[pd.Timestamp] = None

    for i in range(len(series)):
        val = vals[i]
        curr_date = dt_index[i]

        if pd.isna(val):
            if current_len >= 7:
                n_days += current_len
            current_val = None
            current_len = 0
            prev_date = None
            continue

        if prev_date is not None and (
            curr_date.normalize() - prev_date.normalize()
        ) != pd.Timedelta(days=1):
            if current_len >= 7:
                n_days += current_len
            current_val = val
            current_len = 1
            prev_date = curr_date
        elif current_val is not None and val == current_val:
            current_len += 1
            prev_date = curr_date
        else:
            if current_len >= 7:
                n_days += current_len
            current_val = val
            current_len = 1
            prev_date = curr_date

    if current_len >= 7:
        n_days += current_len

    return n_days


def screening_counts(
    tmax_obs: pd.Series,
    tmin_obs: pd.Series,
    stats_tmax: MonthlyStats,
    stats_tmin: MonthlyStats,
) -> ScreeningCounts:
    """Compute Section J.3.3 diagnostic screening counts for one station partition.

    Parameters
    ----------
    tmax_obs : pd.Series
        Observed daily Tmax series with DatetimeIndex.
    tmin_obs : pd.Series
        Observed daily Tmin series with DatetimeIndex.
    stats_tmax : MonthlyStats
        Calibration monthly statistics for Tmax.
    stats_tmin : MonthlyStats
        Calibration monthly statistics for Tmin.

    Returns
    -------
    ScreeningCounts
        Diagnostic screening counts.
    """
    _require_aligned_index(tmax_obs=tmax_obs, tmin_obs=tmin_obs)

    if not isinstance(stats_tmax, MonthlyStats) or not isinstance(
        stats_tmin, MonthlyStats
    ):
        raise ValueError("stats_tmax and stats_tmin must be MonthlyStats instances.")

    valid_tmax = tmax_obs.notna()
    valid_tmin = tmin_obs.notna()
    paired_mask = valid_tmax & valid_tmin

    n_valid_tmax = int(valid_tmax.sum())
    n_valid_tmin = int(valid_tmin.sum())
    n_paired = int(paired_mask.sum())

    # Section E.4.2 physical screening limits
    n_outside_e42_tmax = int(
        ((tmax_obs < E_4_2_TMAX_MIN_C) | (tmax_obs > E_4_2_TMAX_MAX_C)).sum()
    )
    n_outside_e42_tmin = int(
        ((tmin_obs < E_4_2_TMIN_MIN_C) | (tmin_obs > E_4_2_TMIN_MAX_C)).sum()
    )

    # Observed Tmax < Tmin on paired days (strict inequality)
    n_tmax_below_tmin = int((paired_mask & (tmax_obs < tmin_obs)).sum())

    # Days belonging to runs of >= 7 consecutive identical values
    n_days_in_runs_ge7_tmax = _count_days_in_runs_ge7(tmax_obs)
    n_days_in_runs_ge7_tmin = _count_days_in_runs_ge7(tmin_obs)

    # OPEN ITEM v1.10: J.3.3 does not define the 6-sigma rule for a month with undefined or zero sigma; rule not applied and the month is listed  # noqa: E501
    undefined_m_tmax = tuple(
        m
        for m in range(1, 13)
        if np.isnan(stats_tmax.std[m - 1]) or stats_tmax.std[m - 1] <= 0.0
    )
    undefined_m_tmin = tuple(
        m
        for m in range(1, 13)
        if np.isnan(stats_tmin.std[m - 1]) or stats_tmin.std[m - 1] <= 0.0
    )

    dt_index = cast(pd.DatetimeIndex, tmax_obs.index)
    months = dt_index.month

    n_beyond_6sigma_tmax = 0
    n_beyond_6sigma_tmin = 0

    for m in range(1, 13):
        if m not in undefined_m_tmax:
            mu_max = stats_tmax.mean[m - 1]
            sigma_max = stats_tmax.std[m - 1]
            m_mask_max = (months == m) & valid_tmax
            vals_max = tmax_obs[m_mask_max]
            n_beyond_6sigma_tmax += int(
                (np.abs(vals_max - mu_max) > 6.0 * sigma_max).sum()
            )

        if m not in undefined_m_tmin:
            mu_min = stats_tmin.mean[m - 1]
            sigma_min = stats_tmin.std[m - 1]
            m_mask_min = (months == m) & valid_tmin
            vals_min = tmin_obs[m_mask_min]
            n_beyond_6sigma_tmin += int(
                (np.abs(vals_min - mu_min) > 6.0 * sigma_min).sum()
            )

    return ScreeningCounts(
        n_valid_tmax=n_valid_tmax,
        n_valid_tmin=n_valid_tmin,
        n_paired=n_paired,
        n_outside_e42_tmax=n_outside_e42_tmax,
        n_outside_e42_tmin=n_outside_e42_tmin,
        n_tmax_below_tmin=n_tmax_below_tmin,
        n_days_in_runs_ge7_tmax=n_days_in_runs_ge7_tmax,
        n_days_in_runs_ge7_tmin=n_days_in_runs_ge7_tmin,
        n_beyond_6sigma_tmax=n_beyond_6sigma_tmax,
        n_beyond_6sigma_tmin=n_beyond_6sigma_tmin,
        sigma_undefined_months_tmax=undefined_m_tmax,
        sigma_undefined_months_tmin=undefined_m_tmin,
    )


def approval_label_counts(values: pd.Series, approval: pd.Series) -> dict[str, int]:
    """Count non-NaN values grouped by sanitized approval label.

    Parameters
    ----------
    values : pd.Series
        Observation series with DatetimeIndex.
    approval : pd.Series
        Approval status series aligned with values.

    Returns
    -------
    dict[str, int]
        Counts of non-NaN observations per approval label. Empty string or NaN
        approval is counted under '<missing>'. Leading and trailing whitespace is
        stripped; case is preserved.
    """
    _require_aligned_index(values=values, approval=approval)

    valid_mask = values.notna()
    if not valid_mask.any():
        return {}

    valid_approvals = approval[valid_mask]
    counts: dict[str, int] = {}

    for app in valid_approvals:
        if pd.isna(app):
            key = "<missing>"
        else:
            stripped = str(app).strip()
            key = "<missing>" if stripped == "" else stripped
        counts[key] = counts.get(key, 0) + 1

    return counts


def mask_to_definitive(values: pd.Series, approval: pd.Series) -> pd.Series:
    """Mask observation values whose approval label is not 'Definitivo'.

    Parameters
    ----------
    values : pd.Series
        Observation series with DatetimeIndex.
    approval : pd.Series
        Approval status series aligned with values.

    Returns
    -------
    pd.Series
        A copy of values with NaN wherever the stripped approval label is not
        exactly 'Definitivo'. The input series is not modified, and the series
        name is preserved.
    """
    _require_aligned_index(values=values, approval=approval)

    # J.3.4 secondary analysis uses only the label "Definitivo";
    # the scorecard recomputation itself is done by the caller.
    out = values.copy().astype(float)
    is_definitive = approval.apply(
        lambda x: False if pd.isna(x) else str(x).strip() == "Definitivo"
    )
    out[~is_definitive] = np.nan
    out.name = values.name
    return out
