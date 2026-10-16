"""Pre-specified evaluation protocol verification engine (Protocol v1.0).

Implements the exact verification criteria pre-specified in
docs/science/evaluation-protocol.md, including paired masking, universal
crossing rule, decisive and descriptive hydroclimatic metrics, annual block
bootstrap uncertainty, three-tier state mapping, cross-validation partitioners,
and validation access audit guard.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Pre-specified protocol thresholds and constants
EPSILON: float = 1e-7
PRECIP_DRY_THRESHOLD: float = 1.0  # mm: dry day P < 1.0 mm, wet day P >= 1.0 mm
CDD_MIN_ANNUAL_VALID_DAYS: int = 330
CDD_MAX_MONTHLY_MISSING_FRAC: float = 0.20
CROSSING_RULE_DTR_OFFSET: float = 0.025  # Midpoint +/- 0.025 gives DTR = 0.05 C
CROSSING_ALERT_THRESHOLD: float = 0.01  # > 1% crossing triggers alert
DEFAULT_BOOTSTRAP_REPLICAS: int = 1000
DEFAULT_BOOTSTRAP_SEED: int = 42
MIN_BOOTSTRAP_BLOCKS: int = 7

DEFAULT_LEDGER_PATH: Path = Path("local/reports/validation_openings.jsonl")


# ==============================================================================
# 1. Paired Missingness Mask (Protocol Section 2.4)
# ==============================================================================


def apply_paired_mask(
    observed: pd.Series,
    modeled: pd.Series,
    quality_flags: Optional[pd.Series] = None,
) -> Tuple[pd.Series, pd.Series]:
    """Synchronously mask out missing observation days from modeled series.

    Any date where ground truth observation is missing (NaN or quality flag
    'Faltante') must be synchronously masked out from the modeled series prior
    to evaluating any metric (Section 2.4).

    Args:
        observed: In-situ ground truth observation series.
        modeled: Downscaled or raw model series.
        quality_flags: Optional series containing quality strings (e.g. 'Faltante').

    Returns:
        Tuple of (clean_obs, clean_mod) with identical synchronized indices.
    """
    df = pd.DataFrame({"obs": observed, "mod": modeled})
    if quality_flags is not None:
        faltante_mask = quality_flags.astype(str).str.strip().str.lower() == "faltante"
        df.loc[faltante_mask, "obs"] = np.nan

    clean_df = df.dropna()
    return clean_df["obs"], clean_df["mod"]


# ==============================================================================
# 2. Universal Crossing Rule & Physical Consistency Gates (Sections 3.4, 3.5, 6.1)
# ==============================================================================


@dataclass(frozen=True)
class CrossingDiagnostic:
    """Diagnostic report from applying the universal crossing rule."""

    n_crossings: int
    n_total_days: int
    fraction_crossings: float
    alert_triggered: bool


def apply_universal_crossing_rule(
    tmax: pd.Series,
    tmin: pd.Series,
) -> Tuple[pd.Series, pd.Series, CrossingDiagnostic]:
    """Apply universal deterministic crossing rule (Midpoint +/- 0.025 deg C).

    For every day where Tmin > Tmax:
      Tmid = (Tmax + Tmin) / 2
      Tmax_corr = Tmid + 0.025
      Tmin_corr = Tmid - 0.025
    guaranteeing DTR = 0.05 deg C > 0 while preserving daily mean temperature.
    Reports count and fraction of triggered days, with alert if > 1% (Section 3.5).

    Args:
        tmax: Daily maximum temperature series.
        tmin: Daily minimum temperature series.

    Returns:
        Tuple of (tmax_corrected, tmin_corrected, diagnostic).
    """
    df = pd.DataFrame({"tmax": tmax, "tmin": tmin}).copy()
    valid_mask = df["tmax"].notna() & df["tmin"].notna()
    n_valid = int(valid_mask.sum())

    crossing_mask = valid_mask & (df["tmin"] > df["tmax"])
    n_crossings = int(crossing_mask.sum())
    frac_crossings = float(n_crossings / n_valid) if n_valid > 0 else 0.0

    if n_crossings > 0:
        tmid = (df.loc[crossing_mask, "tmax"] + df.loc[crossing_mask, "tmin"]) / 2.0
        df.loc[crossing_mask, "tmax"] = tmid + CROSSING_RULE_DTR_OFFSET
        df.loc[crossing_mask, "tmin"] = tmid - CROSSING_RULE_DTR_OFFSET

    alert = frac_crossings > CROSSING_ALERT_THRESHOLD
    diag = CrossingDiagnostic(
        n_crossings=n_crossings,
        n_total_days=n_valid,
        fraction_crossings=frac_crossings,
        alert_triggered=alert,
    )
    return df["tmax"], df["tmin"], diag


def check_temperature_physical_gate(tmax: pd.Series, tmin: pd.Series) -> bool:
    """Verify temperature physical consistency gate: Tmin <= Tmax on 100% of days.

    Evaluated on the final series after applying the universal crossing rule.
    Any single day with Tmin > Tmax fails the gate fatally (Section 3.4).
    """
    valid_mask = tmax.notna() & tmin.notna()
    if not valid_mask.any():
        return False
    return bool((tmin[valid_mask] <= tmax[valid_mask]).all())


def check_precipitation_physical_gate(precip: pd.Series) -> bool:
    """Verify precipitation physical gate: P >= 0 and finite on 100% of days.

    Any negative amount or non-finite value (inf) on non-null days fails fatally.
    (Section 6.1).
    """
    valid = precip.dropna()
    if len(valid) == 0:
        return False
    all_finite = np.isfinite(valid.to_numpy()).all()
    all_non_negative = (valid.to_numpy() >= 0.0).all()
    return bool(all_finite and all_non_negative)


# ==============================================================================
# 3. Decisive & Descriptive Precipitation Metrics (Sections 3.2, 3.3)
# ==============================================================================


def compute_wet_day_frequency(
    series: pd.Series,
    threshold: float = PRECIP_DRY_THRESHOLD,
) -> float:
    """Compute frequency of wet days (P >= threshold) over all valid days."""
    valid = series.dropna()
    if len(valid) == 0:
        return float("nan")
    return float((valid >= threshold).sum() / len(valid))


def compute_seasonal_wet_day_frequency(
    series: pd.Series,
    threshold: float = PRECIP_DRY_THRESHOLD,
) -> Dict[str, float]:
    """Compute wet-day frequency for DJF, MAM, JJA, and SON meteorological seasons.

    Returns dict with keys 'DJF', 'MAM', 'JJA', 'SON'.
    """
    valid = series.dropna()
    if not isinstance(valid.index, pd.DatetimeIndex):
        raise ValueError(
            "Series index must be a pd.DatetimeIndex for seasonal metrics."
        )

    month_to_season = {
        12: "DJF",
        1: "DJF",
        2: "DJF",
        3: "MAM",
        4: "MAM",
        5: "MAM",
        6: "JJA",
        7: "JJA",
        8: "JJA",
        9: "SON",
        10: "SON",
        11: "SON",
    }
    seasons = valid.index.month.map(month_to_season)
    result = {}
    for season in ["DJF", "MAM", "JJA", "SON"]:
        s_mask = seasons == season
        if s_mask.sum() == 0:
            result[season] = float("nan")
        else:
            sub = valid[s_mask]
            result[season] = float((sub >= threshold).sum() / len(sub))
    return result


def compute_wet_day_percentiles(
    series: pd.Series,
    percentiles: Sequence[float] = (95.0, 99.0),
    threshold: float = PRECIP_DRY_THRESHOLD,
) -> Dict[float, float]:
    """Compute empirical percentiles strictly on wet days (P >= threshold)."""
    valid = series.dropna()
    wet_days = valid[valid >= threshold]
    if len(wet_days) == 0:
        return {p: float("nan") for p in percentiles}
    return {
        p: float(np.percentile(wet_days.to_numpy(dtype=np.float64), p))
        for p in percentiles
    }


@dataclass(frozen=True)
class CDDDiagnostic:
    """Diagnostic report from CDD streak and year completeness exclusions."""

    n_discarded_streaks_obs: int
    n_discarded_streaks_mod: int
    n_discarded_years_obs: int
    n_discarded_years_mod: int


@dataclass(frozen=True)
class CDDResult:
    """Result of annual CDD_max computation with completeness diagnostics."""

    mean_obs: float
    mean_mod: float
    pbias: float
    diagnostic: CDDDiagnostic
    annual_obs: Dict[int, float] = field(default_factory=dict)
    annual_mod: Dict[int, float] = field(default_factory=dict)

    def __iter__(self):
        return iter((self.mean_obs, self.mean_mod, self.pbias, self.diagnostic))


def compute_annual_cdd_max(
    obs: pd.Series,
    mod: pd.Series,
    threshold: float = PRECIP_DRY_THRESHOLD,
    min_annual_valid_days: int = CDD_MIN_ANNUAL_VALID_DAYS,
    max_monthly_missing_frac: float = CDD_MAX_MONTHLY_MISSING_FRAC,
) -> CDDResult:
    """Compute mean annual maximum consecutive dry days (CDD_max) with completeness.

    Completeness and literal streak rules (Section 3.2):
      (a) Streaks are calculated strictly within each calendar year
          (no Dec 31 crossing).
      (b) Any streak that contains or touches a NaN day or touches an excluded month
          (>20% missing in that month) is DISCARDED, not truncated.
      (c) Annual CDD_max is the maximum of valid remaining streaks (or 0.0 if none).
      (d) A calendar year is evaluated only if it has >= min_annual_valid_days
          (330 days) of valid ground observations; otherwise year is discarded.
      (e) Synchronous paired mask: missing observations are masked in model series.

    Returns:
        CDDResult containing mean_obs, mean_mod, pbias, and diagnostic counts.
    """
    if not isinstance(obs.index, pd.DatetimeIndex):
        raise ValueError("obs index must be a pd.DatetimeIndex.")

    df = pd.DataFrame({"obs": obs, "mod": mod}).sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a pd.DatetimeIndex.")

    # Apply synchronous paired masking: missing obs is masked in mod
    df.loc[df["obs"].isna(), "mod"] = np.nan

    years = sorted(df.index.year.unique())
    annual_cdd_obs: List[float] = []
    annual_cdd_mod: List[float] = []
    annual_obs_map: Dict[int, float] = {}
    annual_mod_map: Dict[int, float] = {}

    n_discarded_years_obs = 0
    n_discarded_years_mod = 0
    n_discarded_streaks_obs = 0
    n_discarded_streaks_mod = 0

    for year in years:
        start_date = pd.Timestamp(year, 1, 1)
        end_date = pd.Timestamp(year, 12, 31)
        full_year_idx = pd.date_range(start_date, end_date, freq="D")
        sub = df.loc[df.index.year == year]
        if sub.index.has_duplicates:
            sub = sub[~sub.index.duplicated(keep="first")]
        df_year = sub.reindex(full_year_idx)
        assert isinstance(df_year.index, pd.DatetimeIndex)

        valid_obs_days = int(df_year["obs"].notna().sum())
        if valid_obs_days < min_annual_valid_days:
            n_discarded_years_obs += 1
            n_discarded_years_mod += 1
            continue

        invalid_months = set()
        for month in range(1, 13):
            df_m = df_year[df_year.index.month == month]
            if len(df_m) == 0:
                invalid_months.add(month)
            else:
                missing_frac = df_m["obs"].isna().sum() / len(df_m)
                if missing_frac > max_monthly_missing_frac:
                    invalid_months.add(month)

        def _evaluate_series_cdd(
            series: pd.Series,
        ) -> Tuple[Optional[float], int, bool]:
            valid_streaks: List[int] = []
            discarded = 0

            curr_streak_len = 0
            touches_invalid = False
            has_dry_day = False

            for idx, val in series.items():
                is_invalid_month = idx.month in invalid_months  # type: ignore
                is_nan = pd.isna(val)
                is_invalid_day = is_invalid_month or is_nan
                is_wet = (not is_invalid_day) and (val >= threshold)

                if is_wet:
                    if has_dry_day:
                        if touches_invalid:
                            discarded += 1
                        else:
                            valid_streaks.append(curr_streak_len)
                    curr_streak_len = 0
                    touches_invalid = False
                    has_dry_day = False
                else:
                    if is_invalid_day:
                        touches_invalid = True
                    else:
                        has_dry_day = True
                        curr_streak_len += 1

            # End of calendar year boundary (Dec 31)
            if has_dry_day:
                if touches_invalid:
                    discarded += 1
                else:
                    valid_streaks.append(curr_streak_len)

            if valid_streaks:
                return float(max(valid_streaks)), discarded, False
            if discarded > 0:
                # All candidate dry streaks were discarded -> year is excluded
                return None, discarded, True
            # Year had zero dry days (all days were wet >= threshold) -> valid 0.0
            return 0.0, 0, False

        max_obs, disc_obs, year_disc_obs = _evaluate_series_cdd(df_year["obs"])
        max_mod, disc_mod, year_disc_mod = _evaluate_series_cdd(df_year["mod"])

        n_discarded_streaks_obs += disc_obs
        n_discarded_streaks_mod += disc_mod

        if year_disc_obs:
            n_discarded_years_obs += 1
        else:
            assert max_obs is not None
            annual_cdd_obs.append(max_obs)
            annual_obs_map[year] = max_obs

        if year_disc_mod:
            n_discarded_years_mod += 1
        else:
            assert max_mod is not None
            annual_cdd_mod.append(max_mod)
            annual_mod_map[year] = max_mod

    diag = CDDDiagnostic(
        n_discarded_streaks_obs=n_discarded_streaks_obs,
        n_discarded_streaks_mod=n_discarded_streaks_mod,
        n_discarded_years_obs=n_discarded_years_obs,
        n_discarded_years_mod=n_discarded_years_mod,
    )

    if len(annual_cdd_obs) == 0 or len(annual_cdd_mod) == 0:
        m_obs = (
            float(np.mean(annual_cdd_obs)) if len(annual_cdd_obs) > 0 else float("nan")
        )
        m_mod = (
            float(np.mean(annual_cdd_mod)) if len(annual_cdd_mod) > 0 else float("nan")
        )
        return CDDResult(
            mean_obs=m_obs,
            mean_mod=m_mod,
            pbias=float("nan"),
            diagnostic=diag,
            annual_obs=annual_obs_map,
            annual_mod=annual_mod_map,
        )

    mean_obs = float(np.mean(annual_cdd_obs))
    mean_mod = float(np.mean(annual_cdd_mod))

    if mean_obs < EPSILON:
        pbias = float("nan")
    else:
        pbias = float(100.0 * (mean_mod - mean_obs) / mean_obs)

    return CDDResult(
        mean_obs=mean_obs,
        mean_mod=mean_mod,
        pbias=pbias,
        diagnostic=diag,
        annual_obs=annual_obs_map,
        annual_mod=annual_mod_map,
    )


def compute_precipitation_decisive_metrics(
    obs: pd.Series,
    mod: pd.Series,
    raw: Optional[pd.Series] = None,
) -> Dict[str, float]:
    """Compute all 6 decisive precipitation metrics and Volume Outperformance.

    Metrics computed (Section 3.2):
      - f_wet_total_pbias: Relative bias of wet-day frequency (%)
      - f_wet_seasonal_max_pbias: Max seasonal relative bias of wet-day frequency (%)
      - p95_wet_pbias: Relative bias of wet-day 95th percentile (%)
      - p99_wet_pbias: Relative bias of wet-day 99th percentile (%)
      - cdd_max_pbias: Relative bias of mean annual CDD_max (%)
      - pbias_monthly: Percent bias of mean monthly accumulation (%)
      - delta_pbias_volume: Outperformance Delta |PBIAS| = |PBIAS_raw| - |PBIAS_mod|

    Returns:
        Dictionary of point estimate metric values.
    """
    clean_obs, clean_mod = apply_paired_mask(obs, mod)

    # 1. f_wet total
    f_obs = compute_wet_day_frequency(clean_obs)
    f_mod = compute_wet_day_frequency(clean_mod)
    f_wet_pbias = (
        float(100.0 * (f_mod - f_obs) / f_obs) if f_obs > EPSILON else float("nan")
    )

    # 2. f_wet seasonal
    s_obs = compute_seasonal_wet_day_frequency(clean_obs)
    s_mod = compute_seasonal_wet_day_frequency(clean_mod)
    s_biases = []
    for s in ["DJF", "MAM", "JJA", "SON"]:
        if s_obs[s] > EPSILON:
            s_biases.append(abs(100.0 * (s_mod[s] - s_obs[s]) / s_obs[s]))
    f_wet_seasonal_max = float(max(s_biases)) if s_biases else float("nan")

    # 3. P95 & P99 wet days
    pct_obs = compute_wet_day_percentiles(clean_obs, percentiles=(95.0, 99.0))
    pct_mod = compute_wet_day_percentiles(clean_mod, percentiles=(95.0, 99.0))

    p95_obs, p95_mod = pct_obs[95.0], pct_mod[95.0]
    p99_obs, p99_mod = pct_obs[99.0], pct_mod[99.0]

    p95_pbias = (
        float(100.0 * (p95_mod - p95_obs) / p95_obs)
        if p95_obs > EPSILON
        else float("nan")
    )
    p99_pbias = (
        float(100.0 * (p99_mod - p99_obs) / p99_obs)
        if p99_obs > EPSILON
        else float("nan")
    )

    # 4. CDD max
    cdd_res = compute_annual_cdd_max(obs, mod)
    cdd_pbias = cdd_res.pbias

    # 5. PBIAS monthly / total accumulation
    sum_obs = float(clean_obs.sum())
    sum_mod = float(clean_mod.sum())
    pbias_monthly = (
        float(100.0 * (sum_mod - sum_obs) / sum_obs)
        if sum_obs > EPSILON
        else float("nan")
    )

    # 6. Volume Outperformance Delta |PBIAS|
    if raw is not None:
        _, clean_raw = apply_paired_mask(obs, raw)
        sum_raw = float(clean_raw.sum())
        pbias_raw = (
            float(100.0 * (sum_raw - sum_obs) / sum_obs)
            if sum_obs > EPSILON
            else float("nan")
        )
        delta_pbias = float(abs(pbias_raw) - abs(pbias_monthly))
    else:
        delta_pbias = float("nan")

    return {
        "f_wet_total_pbias": f_wet_pbias,
        "f_wet_seasonal_max_pbias": f_wet_seasonal_max,
        "p95_wet_pbias": p95_pbias,
        "p99_wet_pbias": p99_pbias,
        "cdd_max_pbias": cdd_pbias,
        "pbias_monthly": pbias_monthly,
        "delta_pbias_volume": delta_pbias,
    }


# Descriptive precipitation metrics
def compute_rx1day(series: pd.Series) -> float:
    """Mean annual maximum 1-day precipitation accumulation."""
    valid = series.dropna()
    if len(valid) == 0:
        return float("nan")
    annual_max = valid.groupby(valid.index.year).max()  # type: ignore
    return float(annual_max.mean())


def compute_rx5day(series: pd.Series) -> float:
    """Mean annual maximum consecutive 5-day precipitation accumulation."""
    valid = series.dropna()
    if len(valid) < 5:
        return float("nan")
    roll5 = valid.rolling(window=5, min_periods=5).sum()
    annual_max = roll5.groupby(roll5.index.year).max()  # type: ignore
    return float(annual_max.mean())


def compute_sdii(series: pd.Series, threshold: float = PRECIP_DRY_THRESHOLD) -> float:
    """Simple Daily Intensity Index on wet days: sum(P_wet) / N_wet."""
    valid = series.dropna()
    wet_days = valid[valid >= threshold]
    if len(wet_days) == 0:
        return float("nan")
    return float(wet_days.sum() / len(wet_days))


def compute_precipitation_contingency(
    obs: pd.Series,
    mod: pd.Series,
    threshold: float = PRECIP_DRY_THRESHOLD,
) -> Dict[str, float]:
    """Compute POD and FAR for wet day detection (P >= threshold)."""
    clean_obs, clean_mod = apply_paired_mask(obs, mod)
    obs_wet = clean_obs >= threshold
    mod_wet = clean_mod >= threshold

    hits = int((mod_wet & obs_wet).sum())
    false_alarms = int((mod_wet & ~obs_wet).sum())
    misses = int((~mod_wet & obs_wet).sum())

    pod = float(hits / (hits + misses)) if (hits + misses) > 0 else float("nan")
    far = (
        float(false_alarms / (hits + false_alarms))
        if (hits + false_alarms) > 0
        else float("nan")
    )

    return {"pod": pod, "far": far}


def compute_daily_pearson(obs: pd.Series, mod: pd.Series) -> float:
    """Compute daily Pearson correlation coefficient between synchronized series."""
    clean_obs, clean_mod = apply_paired_mask(obs, mod)
    if len(clean_obs) < 3:
        return float("nan")
    r, _ = stats.pearsonr(clean_obs.to_numpy(), clean_mod.to_numpy())
    return float(r)


# ==============================================================================
# 4. Decisive & Descriptive Temperature Metrics (Section 3.4)
# ==============================================================================


def compute_max_monthly_temperature_bias(
    tmax_obs: pd.Series,
    tmax_mod: pd.Series,
    tmin_obs: pd.Series,
    tmin_mod: pd.Series,
) -> Tuple[float, float, float]:
    """Compute maximum monthly mean temperature bias across calendar months (m=1..12).

    Evaluates max(|Bias_m(Tmax)|) and max(|Bias_m(Tmin)|) across all months.
    Section 3.4 decisive metric: max(max_m |Bias_m(Tmax)|, max_m |Bias_m(Tmin)|).

    Returns:
        Tuple of (max_bias_tmax, max_bias_tmin, max_combined_bias).
    """
    clean_tmax_obs, clean_tmax_mod = apply_paired_mask(tmax_obs, tmax_mod)
    clean_tmin_obs, clean_tmin_mod = apply_paired_mask(tmin_obs, tmin_mod)

    months = range(1, 13)
    tmax_biases = []
    tmin_biases = []

    for m in months:
        mask_max = clean_tmax_obs.index.month == m  # type: ignore
        mask_min = clean_tmin_obs.index.month == m  # type: ignore

        if mask_max.sum() > 0:
            b_max = float(
                clean_tmax_mod[mask_max].mean() - clean_tmax_obs[mask_max].mean()
            )
            tmax_biases.append(abs(b_max))

        if mask_min.sum() > 0:
            b_min = float(
                clean_tmin_mod[mask_min].mean() - clean_tmin_obs[mask_min].mean()
            )
            tmin_biases.append(abs(b_min))

    max_tmax = float(max(tmax_biases)) if tmax_biases else float("nan")
    max_tmin = float(max(tmin_biases)) if tmin_biases else float("nan")
    max_comb = float(max(max_tmax, max_tmin))

    return max_tmax, max_tmin, max_comb


def compute_dtr_mean_bias(
    tmax_obs: pd.Series,
    tmax_mod: pd.Series,
    tmin_obs: pd.Series,
    tmin_mod: pd.Series,
) -> float:
    """Compute absolute mean DTR bias: |mean(DTR_mod) - mean(DTR_obs)|.

    Note: Section 3.4 specifies this as mean DTR bias, NOT a daily MAE.
    """
    df = pd.DataFrame(
        {
            "tmax_obs": tmax_obs,
            "tmax_mod": tmax_mod,
            "tmin_obs": tmin_obs,
            "tmin_mod": tmin_mod,
        }
    ).dropna()

    if len(df) == 0:
        return float("nan")

    dtr_obs = df["tmax_obs"] - df["tmin_obs"]
    dtr_mod = df["tmax_mod"] - df["tmin_mod"]

    return float(abs(dtr_mod.mean() - dtr_obs.mean()))


def compute_extreme_percentile_temperature_bias(
    tmax_obs: pd.Series,
    tmax_mod: pd.Series,
    tmin_obs: pd.Series,
    tmin_mod: pd.Series,
) -> Tuple[float, float, float]:
    """Compute extreme percentile biases: P5(Tmin) and P95(Tmax).

    Section 3.4: max(|Delta P5(Tmin)|, |Delta P95(Tmax)|).

    Returns:
        Tuple of (abs_delta_p5_tmin, abs_delta_p95_tmax, max_combined_extreme).
    """
    clean_tmax_obs, clean_tmax_mod = apply_paired_mask(tmax_obs, tmax_mod)
    clean_tmin_obs, clean_tmin_mod = apply_paired_mask(tmin_obs, tmin_mod)

    if len(clean_tmin_obs) == 0 or len(clean_tmax_obs) == 0:
        return float("nan"), float("nan"), float("nan")

    p5_obs = float(np.percentile(clean_tmin_obs.to_numpy(), 5.0))
    p5_mod = float(np.percentile(clean_tmin_mod.to_numpy(), 5.0))
    delta_p5 = float(abs(p5_mod - p5_obs))

    p95_obs = float(np.percentile(clean_tmax_obs.to_numpy(), 95.0))
    p95_mod = float(np.percentile(clean_tmax_mod.to_numpy(), 95.0))
    delta_p95 = float(abs(p95_mod - p95_obs))

    max_comb = float(max(delta_p5, delta_p95))
    return delta_p5, delta_p95, max_comb


def compute_temperature_decisive_metrics(
    tmax_obs: pd.Series,
    tmax_mod: pd.Series,
    tmin_obs: pd.Series,
    tmin_mod: pd.Series,
) -> Dict[str, float]:
    """Compute all 3 decisive temperature metrics (Section 3.4).

    Metrics computed:
      - max_monthly_bias: max(max_m |Bias_m(Tmax)|, max_m |Bias_m(Tmin)|)
      - dtr_mean_bias: |mean(DTR_mod) - mean(DTR_obs)|
      - extreme_percentiles_bias: max(|Delta P5(Tmin)|, |Delta P95(Tmax)|)
      - delta_p5_tmin: |Delta P5(Tmin)|
      - delta_p95_tmax: |Delta P95(Tmax)|
    """
    _, _, max_monthly = compute_max_monthly_temperature_bias(
        tmax_obs, tmax_mod, tmin_obs, tmin_mod
    )
    dtr_bias = compute_dtr_mean_bias(tmax_obs, tmax_mod, tmin_obs, tmin_mod)
    d_p5, d_p95, max_ext = compute_extreme_percentile_temperature_bias(
        tmax_obs, tmax_mod, tmin_obs, tmin_mod
    )

    return {
        "max_monthly_bias": max_monthly,
        "dtr_mean_bias": dtr_bias,
        "extreme_percentiles_bias": max_ext,
        "delta_p5_tmin": d_p5,
        "delta_p95_tmax": d_p95,
    }


# ==============================================================================
# 5. Annual Block Bootstrap Uncertainty Engine (Section 5.1)
# ==============================================================================


@dataclass(frozen=True)
class BootstrapConfidenceInterval:
    """Statistical uncertainty report from annual block bootstrap."""

    point_estimate: float
    ci_lower: float
    ci_upper: float
    n_replicas: int
    seed: int
    n_discarded_replicas: int = 0
    insufficient_blocks: bool = False


def compute_annual_block_bootstrap_precipitation(
    obs: pd.Series,
    mod: pd.Series,
    raw: Optional[pd.Series] = None,
    n_replicas: int = DEFAULT_BOOTSTRAP_REPLICAS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> Dict[str, BootstrapConfidenceInterval]:
    """Compute 95% CIs for decisive precip metrics via 1-year block bootstrap.

    Resamples whole calendar years with replacement synchronously across series.
    """
    if not isinstance(obs.index, pd.DatetimeIndex):
        raise ValueError("obs index must be a pd.DatetimeIndex.")

    df_dict = {"obs": obs, "mod": mod}
    if raw is not None:
        df_dict["raw"] = raw
    df = pd.DataFrame(df_dict).sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a pd.DatetimeIndex.")

    # Synchronous paired masking: missing obs is masked in mod/raw
    df.loc[df["obs"].isna(), "mod"] = np.nan
    if "raw" in df.columns:
        df.loc[df["obs"].isna(), "raw"] = np.nan

    # Identify calendar years with valid paired observations
    valid_years = np.array([
        y
        for y in sorted(df.index.year.unique())
        if df.loc[df.index.year == y, "obs"].notna().any()
    ])
    n_years = len(valid_years)

    # Point estimates
    point_raw = df["raw"] if raw is not None else None
    point_estimates = compute_precipitation_decisive_metrics(
        df["obs"], df["mod"], raw=point_raw
    )
    base_cdd = compute_annual_cdd_max(df["obs"], df["mod"])

    if n_years < MIN_BOOTSTRAP_BLOCKS:
        early_precip_results: Dict[str, BootstrapConfidenceInterval] = {}
        for k, pt in point_estimates.items():
            early_precip_results[k] = BootstrapConfidenceInterval(
                point_estimate=pt,
                ci_lower=float("nan"),
                ci_upper=float("nan"),
                n_replicas=0,
                seed=seed,
                n_discarded_replicas=0,
                insufficient_blocks=True,
            )
        return early_precip_results

    # Pre-index years into dataframes for fast lookup
    year_dfs = {y: df[df.index.year == y] for y in valid_years}

    rng = np.random.default_rng(seed)
    sampled_metrics: Dict[str, List[float]] = {k: [] for k in point_estimates}

    # Check valid CDD annual blocks
    cdd_valid_years_obs = len(base_cdd.annual_obs)
    cdd_valid_years_mod = len(base_cdd.annual_mod)
    cdd_insufficient = (
        cdd_valid_years_obs < MIN_BOOTSTRAP_BLOCKS
        or cdd_valid_years_mod < MIN_BOOTSTRAP_BLOCKS
    )

    for _ in range(n_replicas):
        resampled_years = rng.choice(valid_years, size=n_years, replace=True)
        # Concatenate sampled years
        sampled_df = pd.concat([year_dfs[y] for y in resampled_years], axis=0)
        s_raw = sampled_df["raw"] if raw is not None else None
        rep_m = compute_precipitation_decisive_metrics(
            sampled_df["obs"], sampled_df["mod"], raw=s_raw
        )

        # Override cdd_max_pbias with the exact annual resample mean
        if not cdd_insufficient:
            res_obs = [
                base_cdd.annual_obs[y]
                for y in resampled_years
                if y in base_cdd.annual_obs
            ]
            res_mod = [
                base_cdd.annual_mod[y]
                for y in resampled_years
                if y in base_cdd.annual_mod
            ]
            if len(res_obs) > 0 and len(res_mod) > 0:
                m_o = float(np.mean(res_obs))
                m_m = float(np.mean(res_mod))
                if m_o > EPSILON:
                    rep_m["cdd_max_pbias"] = float(100.0 * (m_m - m_o) / m_o)
                else:
                    rep_m["cdd_max_pbias"] = float("nan")
            else:
                rep_m["cdd_max_pbias"] = float("nan")
        else:
            rep_m["cdd_max_pbias"] = float("nan")

        for k, v in rep_m.items():
            if np.isfinite(v):
                sampled_metrics[k].append(v)

    results: Dict[str, BootstrapConfidenceInterval] = {}
    for k, pt in point_estimates.items():
        if k == "cdd_max_pbias" and cdd_insufficient:
            results[k] = BootstrapConfidenceInterval(
                point_estimate=pt,
                ci_lower=float("nan"),
                ci_upper=float("nan"),
                n_replicas=0,
                seed=seed,
                n_discarded_replicas=0,
                insufficient_blocks=True,
            )
            continue

        vals = sampled_metrics[k]
        n_disc = n_replicas - len(vals)
        if len(vals) == 0:
            results[k] = BootstrapConfidenceInterval(
                point_estimate=pt,
                ci_lower=float("nan"),
                ci_upper=float("nan"),
                n_replicas=n_replicas,
                seed=seed,
                n_discarded_replicas=n_disc,
                insufficient_blocks=False,
            )
        else:
            c_low = float(np.percentile(vals, 2.5))
            c_high = float(np.percentile(vals, 97.5))
            results[k] = BootstrapConfidenceInterval(
                point_estimate=pt,
                ci_lower=c_low,
                ci_upper=c_high,
                n_replicas=n_replicas,
                seed=seed,
                n_discarded_replicas=n_disc,
                insufficient_blocks=False,
            )

    return results


def compute_annual_block_bootstrap_temperature(
    tmax_obs: pd.Series,
    tmax_mod: pd.Series,
    tmin_obs: pd.Series,
    tmin_mod: pd.Series,
    n_replicas: int = DEFAULT_BOOTSTRAP_REPLICAS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> Dict[str, BootstrapConfidenceInterval]:
    """Compute 95% CIs for decisive temp metrics via 1-year block bootstrap."""
    df = pd.DataFrame(
        {
            "tmax_obs": tmax_obs,
            "tmax_mod": tmax_mod,
            "tmin_obs": tmin_obs,
            "tmin_mod": tmin_mod,
        }
    ).sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a pd.DatetimeIndex.")

    # Identify calendar years with valid paired observations for both tmax and tmin
    valid_years = np.array([
        y
        for y in sorted(df.index.year.unique())
        if (
            df.loc[df.index.year == y, "tmax_obs"].notna().any()
            and df.loc[df.index.year == y, "tmin_obs"].notna().any()
        )
    ])
    n_years = len(valid_years)

    point_estimates = compute_temperature_decisive_metrics(
        df["tmax_obs"], df["tmax_mod"], df["tmin_obs"], df["tmin_mod"]
    )

    if n_years < MIN_BOOTSTRAP_BLOCKS:
        early_temp_results: Dict[str, BootstrapConfidenceInterval] = {}
        for k, pt in point_estimates.items():
            early_temp_results[k] = BootstrapConfidenceInterval(
                point_estimate=pt,
                ci_lower=float("nan"),
                ci_upper=float("nan"),
                n_replicas=0,
                seed=seed,
                n_discarded_replicas=0,
                insufficient_blocks=True,
            )
        return early_temp_results

    year_dfs = {y: df[df.index.year == y] for y in valid_years}
    rng = np.random.default_rng(seed)
    sampled_metrics: Dict[str, List[float]] = {k: [] for k in point_estimates}

    for _ in range(n_replicas):
        resampled_years = rng.choice(valid_years, size=n_years, replace=True)
        sampled_df = pd.concat([year_dfs[y] for y in resampled_years], axis=0)
        rep_m = compute_temperature_decisive_metrics(
            sampled_df["tmax_obs"],
            sampled_df["tmax_mod"],
            sampled_df["tmin_obs"],
            sampled_df["tmin_mod"],
        )
        for k, v in rep_m.items():
            if np.isfinite(v):
                sampled_metrics[k].append(v)

    results: Dict[str, BootstrapConfidenceInterval] = {}
    for k, pt in point_estimates.items():
        vals = sampled_metrics[k]
        n_disc = n_replicas - len(vals)
        if len(vals) == 0:
            results[k] = BootstrapConfidenceInterval(
                point_estimate=pt,
                ci_lower=float("nan"),
                ci_upper=float("nan"),
                n_replicas=n_replicas,
                seed=seed,
                n_discarded_replicas=n_disc,
                insufficient_blocks=False,
            )
        else:
            c_low = float(np.percentile(vals, 2.5))
            c_high = float(np.percentile(vals, 97.5))
            results[k] = BootstrapConfidenceInterval(
                point_estimate=pt,
                ci_lower=c_low,
                ci_upper=c_high,
                n_replicas=n_replicas,
                seed=seed,
                n_discarded_replicas=n_disc,
                insufficient_blocks=False,
            )

    return results


# ==============================================================================
# 6. Three-Tier Outcome Mapping & Global State Concordance (Sections 4.3, 5.2-5.4)
# ==============================================================================


class MetricOutcome(str, Enum):
    """Three-tier metric outcome status (Section 5.2)."""

    PASS = "Pass"
    INCONCLUSIVE = "Inconclusive"
    FAIL = "Fail"


class GlobalOutcome(str, Enum):
    """Overall method qualification state (Sections 5.3 and 5.4)."""

    SUCCESS = "Success"
    NOT_ESTABLISHED = "Not Established"
    FALSIFIED = "Falsified"


def evaluate_symmetric_metric_outcome(
    ci_lower: float,
    ci_upper: float,
    threshold: float,
    is_two_sided: bool = True,
) -> MetricOutcome:
    """Evaluate symmetric three-tier metric outcome mapping (Section 5.2).

    For metric threshold T:
      - Pass: Entire 95% CI lies within the acceptable zone.
      - Fail: Entire 95% CI lies strictly outside the acceptable zone.
      - Inconclusive: CI crosses the threshold boundary.

    Args:
        ci_lower: 2.5th bootstrap percentile.
        ci_upper: 97.5th bootstrap percentile.
        threshold: Absolute tolerance boundary (e.g. 5.0 for 5%, 0.5 for 0.5 C).
        is_two_sided: If True, acceptable zone is [-threshold, threshold].
                      If False, acceptable zone is [0, threshold].
    """
    if np.isnan(ci_lower) or np.isnan(ci_upper):
        return MetricOutcome.INCONCLUSIVE

    min_acceptable = -abs(threshold) if is_two_sided else 0.0
    max_acceptable = abs(threshold)

    # Pass: full interval inside [min_acceptable, max_acceptable]
    if ci_lower >= min_acceptable and ci_upper <= max_acceptable:
        return MetricOutcome.PASS

    # Fail: full interval outside acceptable zone
    if is_two_sided:
        if ci_lower > max_acceptable or ci_upper < min_acceptable:
            return MetricOutcome.FAIL
    else:
        if ci_lower > max_acceptable:
            return MetricOutcome.FAIL

    return MetricOutcome.INCONCLUSIVE


def compute_required_passes(n_decisive: int) -> int:
    """Compute 70% threshold using integer arithmetic: ceil(7*N/10) = -(-7*N // 10)."""
    return -(-7 * n_decisive // 10)


@dataclass(frozen=True)
class EvaluationScorecard:
    """Comprehensive evaluation scorecard for a downscaling level."""

    station_id: str
    variable: str
    level: str
    decisive_outcomes: Dict[str, MetricOutcome]
    decisive_cis: Dict[str, BootstrapConfidenceInterval]
    n_passes: int
    n_decisive: int
    physical_gate_passed: bool
    volume_outperformance_passed: bool
    global_outcome: GlobalOutcome
    crossing_diagnostic: Optional[CrossingDiagnostic] = None
    cdd_diagnostic: Optional[CDDDiagnostic] = None


def evaluate_precipitation_scorecard(
    station_id: str,
    level: str,
    boot_results: Dict[str, BootstrapConfidenceInterval],
    physical_gate_passed: bool,
    cdd_diagnostic: Optional[CDDDiagnostic] = None,
) -> EvaluationScorecard:
    """Evaluate precipitation decisive criteria and map to global outcome (Section 5.3).

    $N_{\text{decisive}} = 6$ metrics:
      1. f_wet_total_pbias (|PBIAS| <= 5%)
      2. f_wet_seasonal_max_pbias (|PBIAS| <= 10%)
      3. p95_wet_pbias (|PBIAS| <= 15%)
      4. p99_wet_pbias (|PBIAS| <= 25%)
      5. cdd_max_pbias (|PBIAS| <= 15%)
      6. pbias_monthly (|PBIAS| <= 10%)

    Required passes: ceil(7*6/10) = 5 passes.
    Volume Outperformance Gate: CI_95_upper(Delta |PBIAS|) >= 0.
    """
    thresholds = {
        "f_wet_total_pbias": (5.0, True),
        "f_wet_seasonal_max_pbias": (10.0, False),  # absolute max
        "p95_wet_pbias": (15.0, True),
        "p99_wet_pbias": (25.0, True),
        "cdd_max_pbias": (15.0, True),
        "pbias_monthly": (10.0, True),
    }

    outcomes: Dict[str, MetricOutcome] = {}
    for metric, (thresh, two_sided) in thresholds.items():
        ci = boot_results[metric]
        outcomes[metric] = evaluate_symmetric_metric_outcome(
            ci.ci_lower, ci.ci_upper, thresh, is_two_sided=two_sided
        )

    # Volume outperformance gate
    vol_ci = boot_results.get("delta_pbias_volume")
    vol_passed = bool(vol_ci is not None and vol_ci.ci_upper >= 0.0)

    n_decisive = 6
    n_passes = sum(1 for m in thresholds if outcomes[m] == MetricOutcome.PASS)
    n_fails = sum(1 for m in thresholds if outcomes[m] == MetricOutcome.FAIL)
    required_passes = compute_required_passes(n_decisive)  # 5

    # Determine state
    if not physical_gate_passed or not vol_passed or n_fails > 0:
        global_state = GlobalOutcome.FALSIFIED
    elif n_passes >= required_passes:
        global_state = GlobalOutcome.SUCCESS
    else:
        global_state = GlobalOutcome.NOT_ESTABLISHED

    return EvaluationScorecard(
        station_id=station_id,
        variable="precipitation",
        level=level,
        decisive_outcomes=outcomes,
        decisive_cis=boot_results,
        n_passes=n_passes,
        n_decisive=n_decisive,
        physical_gate_passed=physical_gate_passed,
        volume_outperformance_passed=vol_passed,
        global_outcome=global_state,
        cdd_diagnostic=cdd_diagnostic,
    )


def evaluate_temperature_scorecard(
    station_id: str,
    level: str,
    boot_results: Dict[str, BootstrapConfidenceInterval],
    physical_gate_passed: bool,
    crossing_diag: Optional[CrossingDiagnostic] = None,
) -> EvaluationScorecard:
    """Evaluate temperature decisive criteria and map to global outcome (Section 5.3).

    $N_{\text{decisive}} = 3$ metrics:
      1. max_monthly_bias (<= 0.5 C)
      2. dtr_mean_bias (<= 0.8 C)
      3. extreme_percentiles_bias (<= 0.8 C) - Both components must satisfy.

    Required passes: ceil(7*3/10) = 3 passes (all 3 must pass).
    """
    outcomes: Dict[str, MetricOutcome] = {}

    # 1. max monthly bias (<= 0.5 C)
    ci_month = boot_results["max_monthly_bias"]
    outcomes["max_monthly_bias"] = evaluate_symmetric_metric_outcome(
        ci_month.ci_lower, ci_month.ci_upper, 0.5, is_two_sided=False
    )

    # 2. dtr mean bias (<= 0.8 C)
    ci_dtr = boot_results["dtr_mean_bias"]
    outcomes["dtr_mean_bias"] = evaluate_symmetric_metric_outcome(
        ci_dtr.ci_lower, ci_dtr.ci_upper, 0.8, is_two_sided=False
    )

    # 3. extreme percentiles bias: aggregated by max, but both components must pass
    ci_ext = boot_results["extreme_percentiles_bias"]
    ci_p5 = boot_results.get("delta_p5_tmin")
    ci_p95 = boot_results.get("delta_p95_tmax")

    out_ext = evaluate_symmetric_metric_outcome(
        ci_ext.ci_lower, ci_ext.ci_upper, 0.8, is_two_sided=False
    )
    if ci_p5 is not None and ci_p95 is not None:
        out_p5 = evaluate_symmetric_metric_outcome(
            ci_p5.ci_lower, ci_p5.ci_upper, 0.8, is_two_sided=False
        )
        out_p95 = evaluate_symmetric_metric_outcome(
            ci_p95.ci_lower, ci_p95.ci_upper, 0.8, is_two_sided=False
        )
        if out_p5 == MetricOutcome.FAIL or out_p95 == MetricOutcome.FAIL:
            out_ext = MetricOutcome.FAIL
        elif out_ext == MetricOutcome.PASS and (
            out_p5 != MetricOutcome.PASS or out_p95 != MetricOutcome.PASS
        ):
            out_ext = MetricOutcome.INCONCLUSIVE

    outcomes["extreme_percentiles_bias"] = out_ext

    n_decisive = 3
    n_passes = sum(1 for m in outcomes.values() if m == MetricOutcome.PASS)
    n_fails = sum(1 for m in outcomes.values() if m == MetricOutcome.FAIL)
    required_passes = compute_required_passes(n_decisive)  # 3

    if not physical_gate_passed or n_fails > 0:
        global_state = GlobalOutcome.FALSIFIED
    elif n_passes >= required_passes:
        global_state = GlobalOutcome.SUCCESS
    else:
        global_state = GlobalOutcome.NOT_ESTABLISHED

    return EvaluationScorecard(
        station_id=station_id,
        variable="temperature",
        level=level,
        decisive_outcomes=outcomes,
        decisive_cis=boot_results,
        n_passes=n_passes,
        n_decisive=n_decisive,
        physical_gate_passed=physical_gate_passed,
        volume_outperformance_passed=True,
        global_outcome=global_state,
        crossing_diagnostic=crossing_diag,
    )


def apply_weaker_of_two_concordance(
    outcome_primary: GlobalOutcome,
    outcome_robustness: GlobalOutcome,
) -> GlobalOutcome:
    """Apply weaker-of-the-two concordance rule (Section 4.3).

    Ordinal ranking: Falsified < Not Established < Success.
    """
    if (
        outcome_primary == GlobalOutcome.FALSIFIED
        or outcome_robustness == GlobalOutcome.FALSIFIED
    ):
        return GlobalOutcome.FALSIFIED
    if (
        outcome_primary == GlobalOutcome.NOT_ESTABLISHED
        or outcome_robustness == GlobalOutcome.NOT_ESTABLISHED
    ):
        return GlobalOutcome.NOT_ESTABLISHED
    return GlobalOutcome.SUCCESS


# ==============================================================================
# 7. Cross-Validation Partitioners & Tuning Losses (Sections 2.2, 4.1, 4.2)
# ==============================================================================


BENCHMARK_PARTITIONS: Dict[str, Dict[str, Tuple[Tuple[int, int], Tuple[int, int]]]] = {
    "precipitation": {
        "21205710": (
            (1975, 1994),
            (1995, 2015),
        ),  # Jardín Botánico (20 yr cal, 21 yr val)
        "23215030": ((1975, 1987), (1988, 2000)),  # Aguas Claras (13 yr cal, 13 yr val)
    },
    "temperature": {
        "21205710": (
            (1990, 2004),
            (2005, 2020),
        ),  # Jardín Botánico (15 yr cal, 16 yr val)
        "23215030": ((1990, 2000), (2001, 2011)),  # Aguas Claras (11 yr cal, 11 yr val)
    },
}


def get_benchmark_partitions(
    variable: str,
    station_id: str,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Retrieve pre-specified calibration and validation year ranges (Section 2.2)."""
    var_key = variable.lower()
    if var_key not in BENCHMARK_PARTITIONS:
        raise ValueError(
            f"Unknown variable '{variable}'. Expected precipitation/temperature."
        )
    st_dict = BENCHMARK_PARTITIONS[var_key]
    if station_id not in st_dict:
        raise ValueError(f"Unknown benchmark station '{station_id}'.")
    return st_dict[station_id]


def split_calibration_validation(
    series: pd.Series,
    cal_range: Tuple[int, int],
    val_range: Tuple[int, int],
    open_validation: bool = False,
    station_id: str = "unknown",
    variable: str = "unknown",
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> Tuple[pd.Series, pd.Series]:
    """Split historical time series into calibration and validation partitions.

    Accessing the validation partition strictly requires open_validation=True
    and automatically registers the opening in the append-only ledger.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Series index must be a pd.DatetimeIndex.")

    cal_mask = (series.index.year >= cal_range[0]) & (series.index.year <= cal_range[1])
    cal_series = series[cal_mask]

    val_mask = (series.index.year >= val_range[0]) & (series.index.year <= val_range[1])

    if val_mask.any():
        validate_validation_access(
            open_validation=open_validation,
            purpose=f"split_calibration_validation for {variable} station {station_id}",
            station_id=station_id,
            variable=variable,
            ledger_path=ledger_path,
        )
        val_series = series[val_mask]
    else:
        val_series = pd.Series(dtype=series.dtype)

    return cal_series, val_series


# Section 4.1 Tuning losses
def precipitation_tuning_loss(obs: pd.Series, mod: pd.Series) -> float:
    """Multi-objective tuning loss for precipitation Level 3 hyperparameter grid.

    L_tuning = |PBIAS(f_wet)| + |PBIAS(P95)| + |PBIAS(P99)| (Section 4.1).
    """
    m = compute_precipitation_decisive_metrics(obs, mod)
    f_p = abs(m["f_wet_total_pbias"])
    p95_p = abs(m["p95_wet_pbias"])
    p99_p = abs(m["p99_wet_pbias"])
    return float(f_p + p95_p + p99_p)


def temperature_tuning_loss(
    tmax_obs: pd.Series,
    tmax_mod: pd.Series,
    tmin_obs: pd.Series,
    tmin_mod: pd.Series,
) -> float:
    """Distributional tuning loss for temperature Level 3 hyperparameter grid.

    Mean absolute error between quantiles 5 to 95 across Tmax, Tmin, and DTR:
    L_tuning = sum_{V} 1/19 sum_{q} |Q_q(V_mod) - Q_q(V_obs)| (Section 4.1).
    """
    clean_tmax_obs, clean_tmax_mod = apply_paired_mask(tmax_obs, tmax_mod)
    clean_tmin_obs, clean_tmin_mod = apply_paired_mask(tmin_obs, tmin_mod)

    dtr_obs = clean_tmax_obs - clean_tmin_obs
    dtr_mod = clean_tmax_mod - clean_tmin_mod

    quantiles = np.arange(5, 100, 5)  # 5, 10, ..., 95 (19 quantiles)
    total_loss = 0.0

    for obs_v, mod_v in [
        (clean_tmax_obs, clean_tmax_mod),
        (clean_tmin_obs, clean_tmin_mod),
        (dtr_obs, dtr_mod),
    ]:
        q_obs = np.percentile(obs_v.to_numpy(), quantiles)
        q_mod = np.percentile(mod_v.to_numpy(), quantiles)
        total_loss += float(np.mean(np.abs(q_mod - q_obs)))

    return float(total_loss)


def nested_annual_block_cv(
    cal_series: pd.Series,
) -> Generator[Tuple[pd.Series, pd.Series], None, None]:
    """Generate folds for nested annual block CV inside calibration (Section 4.1).

    Splits by whole calendar years (Leave-One-Year-Out block CV) to prevent
    synoptic autocorrelation leakage across folds.
    """
    if not isinstance(cal_series.index, pd.DatetimeIndex):
        raise ValueError("cal_series index must be a pd.DatetimeIndex.")

    years = sorted(cal_series.index.year.unique())
    for val_year in years:
        val_mask = cal_series.index.year == val_year
        train_mask = ~val_mask
        yield cal_series[train_mask], cal_series[val_mask]


def contiguous_5year_block_cv(
    series: pd.Series,
    open_validation: bool = False,
    station_id: str = "unknown",
    variable: str = "unknown",
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> Generator[Tuple[pd.Series, pd.Series, Tuple[int, int]], None, None]:
    """Generate 5-year contiguous block cross-validation folds across the full record.

    Contiguous blocks from year 1. Remainder < 5 is merged into the final block.
    Running this track requires open_validation=True and registers in the ledger.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("series index must be a pd.DatetimeIndex.")

    validate_validation_access(
        open_validation=open_validation,
        purpose=f"5-year contiguous block CV for {variable} station {station_id}",
        station_id=station_id,
        variable=variable,
        ledger_path=ledger_path,
    )

    years = sorted(series.index.year.unique())
    n_years = len(years)
    if n_years < 5:
        raise ValueError(f"5-year block CV requires at least 5 years, found {n_years}.")

    # Partition years into 5-year blocks, merging remainder < 5 into the last block
    blocks: List[List[int]] = []
    i = 0
    while i < n_years:
        rem = n_years - i
        if rem < 10 and rem != 5:
            # If remainder < 10 and != 5, make this block take all remaining years
            blocks.append(years[i:])
            break
        else:
            blocks.append(years[i : i + 5])
            i += 5

    for test_block in blocks:
        test_years_set = set(test_block)
        test_mask = series.index.year.isin(test_years_set)
        train_mask = ~test_mask
        test_range = (min(test_block), max(test_block))
        yield series[train_mask], series[test_mask], test_range


# ==============================================================================
# 8. Validation Access Guard & Audit Ledger (Section 2.2 and requirement g)
# ==============================================================================


def get_current_git_commit() -> str:
    """Retrieve the current Git commit hash from the repository."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown_or_uncommitted"


def record_validation_opening(
    purpose: str,
    station_id: str,
    variable: str,
    seed: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> Dict[str, Any]:
    """Append a cryptographically auditable record of validation data access.

    Appends a JSON record with timestamp (ISO 8601 UTC), git commit hash,
    configuration, seed, purpose, station, and variable to the audit ledger.
    """
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "commit_hash": get_current_git_commit(),
        "purpose": purpose,
        "station_id": station_id,
        "variable": variable,
        "seed": seed,
        "config": config or {},
    }
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return record


def validate_validation_access(
    open_validation: bool,
    purpose: str,
    station_id: str = "unknown",
    variable: str = "unknown",
    seed: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> Dict[str, Any]:
    """Verify authorization before allowing validation partition evaluation.

    Raises PermissionError if open_validation is False.
    Registers access in the append-only ledger if open_validation is True.
    """
    if not open_validation:
        raise PermissionError(
            f"Validation access blocked for {variable} station {station_id}. "
            "Pre-specified protocol rules require explicit open_validation=True "
            "to prevent accidental data snooping or informational leakage."
        )

    return record_validation_opening(
        purpose=purpose,
        station_id=station_id,
        variable=variable,
        seed=seed,
        config=config,
        ledger_path=ledger_path,
    )
