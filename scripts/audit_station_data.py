"""Generic, parametrizable station data auditing script.

Performs strict, non-statistical data quality and completeness audits
on ground meteorological observations and ERA5 reanalysis archives for a given station.

Strict non-leakage constraint:
Under NO circumstances does this script compute, print, or leak:
- means
- medians
- quantiles
- global data minimums or maximums
- wet-day frequency (f_wet)
- any comparisons between station observations and ERA5 values.

Controls executed:
1. File structure, station ID matching, SHA-256, date parsing, ordering, duplicates, calendar completeness.
2. Non-numeric values, sentinel codes (-999, 9999, "S/D"), quality flag frequencies.
3. Physical rules violations (< 0, > 500, Tmax outside [5, 45], Tmin outside [-5, 35],
   Tmax < Tmin, streaks >= 5 days for P != 0, streaks >= 10 days for T), observed resolution.
4. Completeness: valid days per year/month, months with > 20% missing, years >= 330 valid days,
   simultaneous valid Tmax and Tmin years, and 3-variable overlap (PR + Tmax + Tmin).
5. Station metadata, nearest ERA5 grid cell, DEM elevation, delta elevation, Haversine distance,
   and Land-Sea Mask (LSM) for nearest and 4 surrounding cardinal cells (without data).
6. ERA5 inventory: variables, units, temporal coverage, expected vs present hourly timesteps,
   and missing years analysis for evaluation (1990-2012) and full observed extension (through 2025).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from tropicor.downscale.orography import STANDARD_GRAVITY
from tropicor.io.stations import StationCatalog

# Standard sentinels to identify in ground data
KNOWN_SENTINELS = [
    "-999",
    "-999.0",
    "9999",
    "9999.0",
    "-9999",
    "-9999.0",
    "S/D",
    "s/d",
    "Nulo",
    "nulo",
    "nan",
    "NaN",
    "NAN",
]


@dataclass
class FileStructureResult:
    """Structure audit metrics for an individual observation file."""

    filepath: str
    file_sha256: str
    station_codes: List[str]
    station_names: List[str]
    station_code_matches_expected: bool
    total_raw_rows: int
    parseable_dates_count: int
    unparseable_dates_count: int
    unparseable_samples: List[str]
    duplicate_dates_count: int
    duplicate_dates_samples: List[str]
    is_monotonic_increasing: bool
    date_inversion_count: int
    earliest_date: Optional[str]
    latest_date: Optional[str]
    expected_start: str
    expected_end: str
    expected_calendar_days: int
    present_calendar_days_in_range: int
    missing_calendar_days_count: int
    missing_calendar_ranges: List[Dict[str, Any]]


@dataclass
class SentinelAuditResult:
    """Non-numeric and sentinel token audit results."""

    filepath: str
    variable_label: str
    sentinel_counts: Dict[str, int]
    non_numeric_counts: Dict[str, int]
    quality_flag_counts: Dict[str, int]


@dataclass
class PhysicalRulesResult:
    """Physical consistency rules and streak audit results."""

    variable_label: str
    below_min_count: int
    below_min_records: List[Dict[str, Any]]
    above_max_count: int
    above_max_records: List[Dict[str, Any]]
    streaks_count: int
    streaks: List[Dict[str, Any]]
    decimal_places_counts: Dict[int, int]
    inferred_resolution_mm_or_deg: str


@dataclass
class TemperatureConsistencyResult:
    """Consistency audit between daily Tmax and Tmin."""

    inversion_count: int
    inversion_records: List[Dict[str, Any]]


@dataclass
class CompletenessResult:
    """Annual and monthly completeness metrics."""

    variable_label: str
    evaluation_scope: str  # "Protocol Range" or "Full Observed Range"
    expected_start: str
    expected_end: str
    total_expected_days: int
    total_valid_days: int
    total_missing_days: int
    annual_summary: List[Dict[str, Any]]
    years_ge_330_count: int
    years_ge_330_list: List[int]
    months_gt_20pct_missing_count: int
    months_gt_20pct_missing_list: List[Dict[str, Any]]


@dataclass
class SimultaneousTemperatureResult:
    """Completeness metrics for simultaneous valid Tmax and Tmin."""

    total_simultaneous_valid_days: int
    annual_simultaneous_valid: List[Dict[str, Any]]
    years_simultaneous_ge_330_count: int
    years_simultaneous_ge_330_list: List[int]


@dataclass
class TripletOverlapResult:
    """Annual overlap metrics where PR, Tmax, and Tmin are simultaneously valid."""

    total_triplet_valid_days: int
    annual_triplet_summary: List[Dict[str, Any]]
    years_triplet_ge_330_count: int
    years_triplet_ge_330_list: List[int]
    years_triplet_any_count: int
    years_triplet_any_list: List[int]


@dataclass
class MetadataAuditResult:
    """Station coordinates, catalog elevation, and ERA5 cell comparison."""

    station_code: str
    station_name: str
    latitude: Optional[float]
    longitude: Optional[float]
    elevation_masl: Optional[float]
    cne_elevation_masl: Optional[float]
    department: Optional[str]
    municipality: Optional[str]
    natural_region: Optional[str]
    era5_nearest_latitude: Optional[float]
    era5_nearest_longitude: Optional[float]
    distance_to_nearest_era5_km: Optional[float]
    nearest_era5_lsm: Optional[float]
    surrounding_era5_lsm: Dict[str, Dict[str, float]] = field(default_factory=dict)
    era5_geopotential_z: Optional[float] = None
    era5_dem_elevation_m: Optional[float] = None
    elevation_difference_vs_catalog_m: Optional[float] = None
    elevation_difference_vs_cne_m: Optional[float] = None


@dataclass
class Era5InventoryResult:
    """ERA5 reanalysis archive inventory and missing years analysis."""

    era5_dir: str
    static_file_present: bool
    static_variables: List[Dict[str, Any]]
    time_series_variables: List[Dict[str, Any]]
    missing_for_1990_2012_window: Dict[str, List[int]] = field(default_factory=dict)
    missing_for_2025_extension: Dict[str, List[int]] = field(default_factory=dict)


def compute_file_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file on disk."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_haversine_distance_km(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Calculate great-circle distance between two geographic points in km."""
    r_earth = 6371.0088
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    return float(2.0 * r_earth * np.arcsin(np.sqrt(a)))


def group_contiguous_dates(
    missing_dates: List[pd.Timestamp],
) -> List[Dict[str, Any]]:
    """Group sorted missing calendar timestamps into contiguous intervals."""
    if not missing_dates:
        return []

    ranges: List[Dict[str, Any]] = []
    start_date = missing_dates[0]
    prev_date = missing_dates[0]
    count = 1

    for d in missing_dates[1:]:
        if (d - prev_date).days == 1:
            count += 1
            prev_date = d
        else:
            ranges.append(
                {
                    "start": start_date.strftime("%Y-%m-%d"),
                    "end": prev_date.strftime("%Y-%m-%d"),
                    "days": count,
                }
            )
            start_date = d
            prev_date = d
            count = 1

    ranges.append(
        {
            "start": start_date.strftime("%Y-%m-%d"),
            "end": prev_date.strftime("%Y-%m-%d"),
            "days": count,
        }
    )
    return ranges


def audit_file_structure(
    filepath: Path,
    expected_station_id: str = "54085010",
    expected_start: str = "1990-01-01",
    expected_end: str = "2011-12-31",
) -> Tuple[FileStructureResult, pd.DataFrame]:
    """Audit Control 1: File structure, dates, duplicates, ordering, SHA-256."""
    sha256_hash = compute_file_sha256(filepath)

    if filepath.suffix.lower() in (".xlsx", ".xls"):
        df_raw = pd.read_excel(filepath, dtype=str)
    else:
        df_raw = pd.read_csv(filepath, dtype=str)

    total_rows = len(df_raw)

    # Station code and name
    st_codes = (
        sorted(df_raw["CodigoEstacion"].dropna().unique().tolist())
        if "CodigoEstacion" in df_raw.columns
        else []
    )
    st_names = (
        sorted(df_raw["NombreEstacion"].dropna().unique().tolist())
        if "NombreEstacion" in df_raw.columns
        else []
    )

    clean_codes = [str(c).strip().replace(".0", "") for c in st_codes]
    code_match = (
        all(c == expected_station_id for c in clean_codes) if clean_codes else False
    )

    # Date parsing
    date_col = "Fecha" if "Fecha" in df_raw.columns else df_raw.columns[0]
    parsed_dates = pd.to_datetime(df_raw[date_col], errors="coerce")
    df_raw["_parsed_date"] = parsed_dates

    unparseable_mask = parsed_dates.isna()
    unparseable_count = int(unparseable_mask.sum())
    unparseable_samples = (
        df_raw.loc[unparseable_mask, date_col].head(5).tolist()
        if unparseable_count > 0
        else []
    )
    parseable_count = total_rows - unparseable_count

    # Duplicates in parsed dates
    valid_dates_df = df_raw.loc[~unparseable_mask].copy()
    valid_dates_df["_norm_date"] = valid_dates_df["_parsed_date"].dt.normalize()
    dupes_mask = valid_dates_df.duplicated(subset=["_norm_date"], keep=False)
    duplicate_count = int(dupes_mask.sum())
    duplicate_dates_samples = (
        valid_dates_df.loc[dupes_mask, "_norm_date"]
        .dt.strftime("%Y-%m-%d")
        .unique()[:5]
        .tolist()
        if duplicate_count > 0
        else []
    )

    # Temporal monotonicity
    date_series = valid_dates_df["_norm_date"].reset_index(drop=True)
    is_monotonic = bool(date_series.is_monotonic_increasing)
    inversions = 0
    if not is_monotonic and len(date_series) > 1:
        dates_array = pd.DatetimeIndex(date_series).to_numpy(dtype="datetime64[D]")
        diffs = np.diff(dates_array).astype(int)
        inversions = int((diffs < 0).sum())

    earliest_date = (
        date_series.min().strftime("%Y-%m-%d") if not date_series.empty else None
    )
    latest_date = (
        date_series.max().strftime("%Y-%m-%d") if not date_series.empty else None
    )

    # Expected calendar range
    full_calendar = pd.date_range(expected_start, expected_end, freq="D")
    expected_calendar_days = len(full_calendar)

    # Days in range present in observation
    in_range_dates = valid_dates_df[
        (valid_dates_df["_norm_date"] >= expected_start)
        & (valid_dates_df["_norm_date"] <= expected_end)
    ]["_norm_date"].unique()

    present_days_set = set(pd.to_datetime(in_range_dates))
    present_count = len(present_days_set)
    missing_dates = sorted(list(set(full_calendar) - present_days_set))
    missing_count = len(missing_dates)
    missing_ranges = group_contiguous_dates(missing_dates)

    res = FileStructureResult(
        filepath=str(filepath),
        file_sha256=sha256_hash,
        station_codes=[str(c) for c in st_codes],
        station_names=[str(n) for n in st_names],
        station_code_matches_expected=code_match,
        total_raw_rows=total_rows,
        parseable_dates_count=parseable_count,
        unparseable_dates_count=unparseable_count,
        unparseable_samples=[str(s) for s in unparseable_samples],
        duplicate_dates_count=duplicate_count,
        duplicate_dates_samples=[str(s) for s in duplicate_dates_samples],
        is_monotonic_increasing=is_monotonic,
        date_inversion_count=inversions,
        earliest_date=earliest_date,
        latest_date=latest_date,
        expected_start=expected_start,
        expected_end=expected_end,
        expected_calendar_days=expected_calendar_days,
        present_calendar_days_in_range=present_count,
        missing_calendar_days_count=missing_count,
        missing_calendar_ranges=missing_ranges,
    )
    return res, df_raw


def audit_sentinels_and_flags(
    df_raw: pd.DataFrame,
    variable_label: str,
    filepath: str,
    value_col: str = "Valor",
    flag_col: Optional[str] = "NivelAprobacion",
) -> SentinelAuditResult:
    """Audit Control 2: Non-numeric values, sentinels, and quality flag distributions."""
    raw_strings = (
        df_raw[value_col].fillna("").astype(str).str.strip()
        if value_col in df_raw.columns
        else pd.Series([], dtype=str)
    )

    sentinel_counts: Dict[str, int] = {}
    for sentinel in KNOWN_SENTINELS:
        count = int((raw_strings == sentinel).sum())
        if count > 0:
            sentinel_counts[sentinel] = count

    non_numeric_counts: Dict[str, int] = {}
    for val in raw_strings:
        if not val or val in KNOWN_SENTINELS:
            continue
        try:
            float(val.replace(",", "."))
        except ValueError:
            non_numeric_counts[val] = non_numeric_counts.get(val, 0) + 1

    flag_counts: Dict[str, int] = {}
    if flag_col and flag_col in df_raw.columns:
        counts_series = (
            df_raw[flag_col].fillna("MISSING").astype(str).str.strip().value_counts()
        )
        for flag_val, count in counts_series.items():
            flag_counts[str(flag_val)] = int(count)

    return SentinelAuditResult(
        filepath=filepath,
        variable_label=variable_label,
        sentinel_counts=sentinel_counts,
        non_numeric_counts=non_numeric_counts,
        quality_flag_counts=flag_counts,
    )


def detect_streaks(
    dates: pd.DatetimeIndex,
    values: np.ndarray,
    min_length: int,
    ignore_zero: bool = False,
) -> List[Dict[str, Any]]:
    """Detect streaks of consecutive calendar days with identical non-NaN value."""
    streaks: List[Dict[str, Any]] = []
    n = len(dates)
    if n < min_length:
        return streaks

    curr_val = values[0]
    curr_start_idx = 0
    curr_len = 1 if not np.isnan(curr_val) else 0

    for i in range(1, n):
        val = values[i]
        date_step = (dates[i] - dates[i - 1]).days

        if date_step == 1 and not np.isnan(val) and not np.isnan(curr_val):
            if np.isclose(val, curr_val, atol=1e-5):
                curr_len += 1
            else:
                if curr_len >= min_length:
                    if not (ignore_zero and np.isclose(curr_val, 0.0, atol=1e-5)):
                        streaks.append(
                            {
                                "start_date": dates[curr_start_idx].strftime(
                                    "%Y-%m-%d"
                                ),
                                "end_date": dates[i - 1].strftime("%Y-%m-%d"),
                                "length": curr_len,
                                "value": float(curr_val),
                            }
                        )
                curr_val = val
                curr_start_idx = i
                curr_len = 1
        else:
            if curr_len >= min_length:
                if not (ignore_zero and np.isclose(curr_val, 0.0, atol=1e-5)):
                    streaks.append(
                        {
                            "start_date": dates[curr_start_idx].strftime("%Y-%m-%d"),
                            "end_date": dates[i - 1].strftime("%Y-%m-%d"),
                            "length": curr_len,
                            "value": float(curr_val),
                        }
                    )
            curr_val = val
            curr_start_idx = i
            curr_len = 1 if not np.isnan(val) else 0

    if curr_len >= min_length:
        if not (ignore_zero and np.isclose(curr_val, 0.0, atol=1e-5)):
            streaks.append(
                {
                    "start_date": dates[curr_start_idx].strftime("%Y-%m-%d"),
                    "end_date": dates[-1].strftime("%Y-%m-%d"),
                    "length": curr_len,
                    "value": float(curr_val),
                }
            )

    return streaks


def audit_physical_rules(
    series: pd.Series,
    raw_strings: pd.Series,
    variable_label: str,
    lower_bound: float,
    upper_bound: float,
    min_streak_days: int,
    ignore_zero_streaks: bool,
) -> PhysicalRulesResult:
    """Audit Control 3: Physical bound violations, streaks, and observed resolution.

    Strict rule: Count and list date+value ONLY for flagged records.
    Never calculate mean, median, min, max, or quantiles of the series.
    """
    valid_series = series.dropna().sort_index()

    below_mask = valid_series < lower_bound
    below_records = [
        {
            "date": pd.Timestamp(str(idx)).strftime("%Y-%m-%d"),
            "value": float(val),
        }
        for idx, val in valid_series[below_mask].items()
    ]

    above_mask = valid_series > upper_bound
    above_records = [
        {
            "date": pd.Timestamp(str(idx)).strftime("%Y-%m-%d"),
            "value": float(val),
        }
        for idx, val in valid_series[above_mask].items()
    ]

    dates = pd.DatetimeIndex(valid_series.index)
    vals = valid_series.to_numpy(dtype=float)
    streaks = detect_streaks(
        dates,
        vals,
        min_length=min_streak_days,
        ignore_zero=ignore_zero_streaks,
    )

    clean_raw = raw_strings.fillna("").astype(str).str.strip().str.replace(",", ".")
    decimal_counts: Dict[int, int] = {}
    for s in clean_raw:
        if not s or s in KNOWN_SENTINELS:
            continue
        try:
            float(s)
            parts = s.split(".")
            decimals = len(parts[1]) if len(parts) > 1 else 0
            decimal_counts[decimals] = decimal_counts.get(decimals, 0) + 1
        except ValueError:
            continue

    if decimal_counts:
        max_dec = max(decimal_counts.keys())
        inferred = (
            f"10^-{max_dec} (up to {max_dec} decimal places)"
            if max_dec > 0
            else "Integer (0 decimals)"
        )
    else:
        inferred = "Indeterminate"

    return PhysicalRulesResult(
        variable_label=variable_label,
        below_min_count=len(below_records),
        below_min_records=below_records,
        above_max_count=len(above_records),
        above_max_records=above_records,
        streaks_count=len(streaks),
        streaks=streaks,
        decimal_places_counts=dict(sorted(decimal_counts.items())),
        inferred_resolution_mm_or_deg=inferred,
    )


def audit_temperature_inversions(
    tmax_series: pd.Series,
    tmin_series: pd.Series,
) -> TemperatureConsistencyResult:
    """Audit Control 3: Inversions where Tmax < Tmin on the same calendar date."""
    common_dates = tmax_series.dropna().index.intersection(tmin_series.dropna().index)
    if common_dates.empty:
        return TemperatureConsistencyResult(inversion_count=0, inversion_records=[])

    s_max = tmax_series.loc[common_dates]
    s_min = tmin_series.loc[common_dates]
    inv_mask = s_max < s_min

    inversion_records = [
        {
            "date": pd.Timestamp(str(idx)).strftime("%Y-%m-%d"),
            "tmax": float(s_max.loc[idx]),
            "tmin": float(s_min.loc[idx]),
        }
        for idx in common_dates[inv_mask]
    ]

    return TemperatureConsistencyResult(
        inversion_count=len(inversion_records),
        inversion_records=inversion_records,
    )


def audit_completeness(
    series: pd.Series,
    variable_label: str,
    expected_start: str = "1990-01-01",
    expected_end: str = "2011-12-31",
    evaluation_scope: str = "Protocol Range",
) -> CompletenessResult:
    """Audit Control 4: Annual and monthly completeness, missing > 20%, years >= 330 days."""
    full_dates = pd.date_range(expected_start, expected_end, freq="D")
    cal_df = pd.DataFrame(index=full_dates)
    cal_df["year"] = full_dates.year
    cal_df["month"] = full_dates.month
    cal_df["year_month"] = full_dates.to_period("M")

    s_reindexed = series.reindex(full_dates)
    cal_df["is_valid"] = s_reindexed.notna()

    annual_summary: List[Dict[str, Any]] = []
    years_ge_330: List[int] = []

    for y, grp in cal_df.groupby("year"):
        tot_days = len(grp)
        v_days = int(grp["is_valid"].to_numpy().sum())
        m_days = tot_days - v_days
        v_pct = (v_days / tot_days) * 100.0
        ge_330 = bool(v_days >= 330)
        y_int = int(str(y))
        if ge_330:
            years_ge_330.append(y_int)

        annual_summary.append(
            {
                "year": y_int,
                "total_days": tot_days,
                "valid_days": v_days,
                "missing_days": m_days,
                "valid_pct": round(v_pct, 2),
                "ge_330": ge_330,
            }
        )

    months_gt_20pct: List[Dict[str, Any]] = []
    for ym, grp in cal_df.groupby("year_month"):
        tot_days = len(grp)
        v_days = int(grp["is_valid"].to_numpy().sum())
        m_days = tot_days - v_days
        pct_missing = (m_days / tot_days) * 100.0

        if pct_missing > 20.0:
            months_gt_20pct.append(
                {
                    "year_month": str(ym),
                    "total_days": tot_days,
                    "valid_days": v_days,
                    "missing_days": m_days,
                    "pct_missing": round(pct_missing, 2),
                }
            )

    tot_expected = len(full_dates)
    tot_valid = int(cal_df["is_valid"].sum())
    tot_missing = tot_expected - tot_valid

    return CompletenessResult(
        variable_label=variable_label,
        evaluation_scope=evaluation_scope,
        expected_start=expected_start,
        expected_end=expected_end,
        total_expected_days=tot_expected,
        total_valid_days=tot_valid,
        total_missing_days=tot_missing,
        annual_summary=annual_summary,
        years_ge_330_count=len(years_ge_330),
        years_ge_330_list=years_ge_330,
        months_gt_20pct_missing_count=len(months_gt_20pct),
        months_gt_20pct_missing_list=months_gt_20pct,
    )


def audit_simultaneous_temperature(
    tmax_series: pd.Series,
    tmin_series: pd.Series,
    expected_start: str = "1990-01-01",
    expected_end: str = "2011-12-31",
) -> SimultaneousTemperatureResult:
    """Audit Control 4: Years with Tmax and Tmin valid simultaneously."""
    full_dates = pd.date_range(expected_start, expected_end, freq="D")
    cal_df = pd.DataFrame(index=full_dates)
    cal_df["year"] = full_dates.year

    s_max = tmax_series.reindex(full_dates)
    s_min = tmin_series.reindex(full_dates)
    cal_df["both_valid"] = s_max.notna() & s_min.notna()

    annual_res: List[Dict[str, Any]] = []
    years_ge_330: List[int] = []

    for y, grp in cal_df.groupby("year"):
        tot_days = len(grp)
        v_days = int(grp["both_valid"].to_numpy().sum())
        ge_330 = bool(v_days >= 330)
        y_int = int(str(y))
        if ge_330:
            years_ge_330.append(y_int)
        annual_res.append(
            {
                "year": y_int,
                "total_days": tot_days,
                "simultaneous_valid_days": v_days,
                "valid_pct": round((v_days / tot_days) * 100.0, 2),
                "ge_330": ge_330,
            }
        )

    return SimultaneousTemperatureResult(
        total_simultaneous_valid_days=int(cal_df["both_valid"].sum()),
        annual_simultaneous_valid=annual_res,
        years_simultaneous_ge_330_count=len(years_ge_330),
        years_simultaneous_ge_330_list=years_ge_330,
    )


def audit_triplet_overlap(
    pr_series: pd.Series,
    tmax_series: pd.Series,
    tmin_series: pd.Series,
    start_date: str = "1990-01-01",
    end_date: str = "2011-12-31",
) -> TripletOverlapResult:
    """Audit Control 2d: Overlap where PR, Tmax, and Tmin are simultaneously valid."""
    full_dates = pd.date_range(start_date, end_date, freq="D")
    cal_df = pd.DataFrame(index=full_dates)
    cal_df["year"] = full_dates.year

    s_pr = pr_series.reindex(full_dates)
    s_max = tmax_series.reindex(full_dates)
    s_min = tmin_series.reindex(full_dates)

    cal_df["all_three_valid"] = s_pr.notna() & s_max.notna() & s_min.notna()

    annual_res: List[Dict[str, Any]] = []
    years_ge_330: List[int] = []
    years_any: List[int] = []

    for y, grp in cal_df.groupby("year"):
        tot_days = len(grp)
        v_days = int(grp["all_three_valid"].to_numpy().sum())
        ge_330 = bool(v_days >= 330)
        has_any = bool(v_days > 0)
        y_int = int(str(y))
        if ge_330:
            years_ge_330.append(y_int)
        if has_any:
            years_any.append(y_int)
        annual_res.append(
            {
                "year": y_int,
                "total_days": tot_days,
                "triplet_valid_days": v_days,
                "valid_pct": round((v_days / tot_days) * 100.0, 2),
                "ge_330": ge_330,
            }
        )

    return TripletOverlapResult(
        total_triplet_valid_days=int(cal_df["all_three_valid"].sum()),
        annual_triplet_summary=annual_res,
        years_triplet_ge_330_count=len(years_ge_330),
        years_triplet_ge_330_list=years_ge_330,
        years_triplet_any_count=len(years_any),
        years_triplet_any_list=years_any,
    )


def audit_metadata(
    station_id: str,
    cne_path: Optional[Path] = None,
    era5_dir: Optional[Path] = None,
) -> MetadataAuditResult:
    """Audit Control 5: Station coordinates/elevation and nearest ERA5 cell DEM elevation."""
    catalog = StationCatalog.from_benchmark()
    st_meta = catalog.get(station_id)

    name = st_meta.name if st_meta else "UNKNOWN"
    lat = st_meta.latitude if st_meta else None
    lon = st_meta.longitude if st_meta else None
    elev = st_meta.elevation if st_meta else None
    dept = st_meta.department if st_meta else None
    muni = st_meta.municipality if st_meta else None
    region = st_meta.region.value if st_meta else None

    cne_elev: Optional[float] = None
    if cne_path and cne_path.exists():
        try:
            df_cne = pd.read_excel(cne_path)
            code_col = (
                "CODIGO"
                if "CODIGO" in df_cne.columns
                else [c for c in df_cne.columns if "cod" in c.lower()][0]
            )
            df_cne["_clean_code"] = (
                df_cne[code_col]
                .astype(str)
                .str.strip()
                .str.replace(r"\.0$", "", regex=True)
            )
            match = df_cne[df_cne["_clean_code"] == str(station_id)]
            if not match.empty:
                row = match.iloc[0]
                if "ALTITUD" in row and pd.notna(row["ALTITUD"]):
                    cne_elev = float(row["ALTITUD"])
                if lat is None and "LATITUD" in row:
                    lat_str = str(row["LATITUD"]).replace(",", ".")
                    lat = float(lat_str)
                if lon is None and "LONGITUD" in row:
                    lon_str = str(row["LONGITUD"]).replace(",", ".")
                    lon = float(lon_str)
                if dept is None and "DEPARTAMENTO" in row:
                    dept = str(row["DEPARTAMENTO"])
                if muni is None and "MUNICIPIO" in row:
                    muni = str(row["MUNICIPIO"])
        except Exception:  # noqa: BLE001
            pass

    nearest_lat: Optional[float] = None
    nearest_lon: Optional[float] = None
    dist_km: Optional[float] = None
    nearest_lsm: Optional[float] = None
    surrounding_lsm: Dict[str, Dict[str, float]] = {}
    z_val: Optional[float] = None
    era5_dem_elev: Optional[float] = None
    delta_catalog: Optional[float] = None
    delta_cne: Optional[float] = None

    if era5_dir and era5_dir.exists() and lat is not None and lon is not None:
        static_file = era5_dir / "era5_static.nc"
        if static_file.exists():
            try:
                ds_static = xr.open_dataset(static_file)
                lats = ds_static["latitude"].values
                lons = ds_static["longitude"].values

                lat_idx = int(np.abs(lats - lat).argmin())
                lon_idx = int(np.abs(lons - lon).argmin())

                nearest_lat = float(lats[lat_idx])
                nearest_lon = float(lons[lon_idx])
                dist_km = compute_haversine_distance_km(
                    lat, lon, nearest_lat, nearest_lon
                )

                if "lsm" in ds_static:
                    lsm_da = ds_static["lsm"]
                    if "valid_time" in lsm_da.dims:
                        lsm_da = lsm_da.isel(valid_time=0)
                    nearest_lsm = round(
                        float(lsm_da.isel(latitude=lat_idx, longitude=lon_idx).values),
                        4,
                    )

                    cardinals = [
                        ("North", lat_idx - 1, lon_idx),
                        ("South", lat_idx + 1, lon_idx),
                        ("West", lat_idx, lon_idx - 1),
                        ("East", lat_idx, lon_idx + 1),
                    ]
                    for d_name, r, c in cardinals:
                        if 0 <= r < len(lats) and 0 <= c < len(lons):
                            surrounding_lsm[d_name] = {
                                "latitude": float(lats[r]),
                                "longitude": float(lons[c]),
                                "lsm": round(
                                    float(lsm_da.isel(latitude=r, longitude=c).values),
                                    4,
                                ),
                            }

                if "z" in ds_static:
                    z_da = ds_static["z"]
                    if "valid_time" in z_da.dims:
                        z_val = float(
                            z_da.isel(
                                valid_time=0, latitude=lat_idx, longitude=lon_idx
                            ).values
                        )
                    else:
                        z_val = float(
                            z_da.isel(latitude=lat_idx, longitude=lon_idx).values
                        )

                    era5_dem_elev = float(z_val / STANDARD_GRAVITY)

                    if elev is not None:
                        delta_catalog = float(era5_dem_elev - elev)
                    if cne_elev is not None:
                        delta_cne = float(era5_dem_elev - cne_elev)
            except Exception:  # noqa: BLE001
                pass

    return MetadataAuditResult(
        station_code=station_id,
        station_name=name,
        latitude=lat,
        longitude=lon,
        elevation_masl=elev,
        cne_elevation_masl=cne_elev,
        department=dept,
        municipality=muni,
        natural_region=region,
        era5_nearest_latitude=nearest_lat,
        era5_nearest_longitude=nearest_lon,
        distance_to_nearest_era5_km=round(dist_km, 2) if dist_km is not None else None,
        nearest_era5_lsm=nearest_lsm,
        surrounding_era5_lsm=surrounding_lsm,
        era5_geopotential_z=round(z_val, 2) if z_val is not None else None,
        era5_dem_elevation_m=(
            round(era5_dem_elev, 2) if era5_dem_elev is not None else None
        ),
        elevation_difference_vs_catalog_m=(
            round(delta_catalog, 2) if delta_catalog is not None else None
        ),
        elevation_difference_vs_cne_m=(
            round(delta_cne, 2) if delta_cne is not None else None
        ),
    )


def audit_era5_inventory(
    era5_dir: Path,
    expected_start_year: int = 1990,
    expected_end_year: int = 2011,
) -> Era5InventoryResult:
    """Audit Control 6: ERA5 inventory (variables, units, coverage, hourly steps)."""
    static_path = era5_dir / "era5_static.nc"
    static_present = static_path.exists()
    static_vars_info: List[Dict[str, Any]] = []

    if static_present:
        try:
            ds_static = xr.open_dataset(static_path)
            for vname, da in ds_static.data_vars.items():
                static_vars_info.append(
                    {
                        "variable": str(vname),
                        "long_name": da.attrs.get("long_name", str(vname)),
                        "units": da.attrs.get("units", "dimensionless"),
                        "shape": list(da.shape),
                        "dims": list(da.dims),
                    }
                )
        except Exception as e:  # noqa: BLE001
            static_vars_info.append({"error": str(e)})

    variable_specs = [
        ("total_precipitation", "era5_total_precipitation_", "tp"),
        ("2m_temperature", "era5_2m_temperature_", "t2m"),
    ]

    time_series_vars: List[Dict[str, Any]] = []
    missing_1990_2012: Dict[str, List[int]] = {}
    missing_through_2025: Dict[str, List[int]] = {}

    for var_id, prefix, default_short in variable_specs:
        files = sorted(era5_dir.glob(f"{prefix}*.nc"))
        if not files:
            continue

        var_units = "unknown"
        var_long_name = var_id
        short_name = default_short

        try:
            ds_sample = xr.open_dataset(files[0])
            for dname, da in ds_sample.data_vars.items():
                if dname != "expver":
                    short_name = str(dname)
                    var_units = da.attrs.get("units", "unknown")
                    var_long_name = da.attrs.get("long_name", var_id)
                    break
        except Exception:  # noqa: BLE001
            pass

        years_detail: List[Dict[str, Any]] = []
        tot_expected = 0
        tot_present = 0
        missing_years: List[int] = []

        for y in range(expected_start_year, expected_end_year + 1):
            is_leap = (y % 4 == 0 and y % 100 != 0) or (y % 400 == 0)
            exp_hours = 8784 if is_leap else 8760
            tot_expected += exp_hours

            fn = era5_dir / f"{prefix}{y}.nc"
            if not fn.exists():
                missing_years.append(y)
                years_detail.append(
                    {
                        "year": y,
                        "expected_hours": exp_hours,
                        "present_hours": 0,
                        "status": "MISSING",
                    }
                )
            else:
                try:
                    ds_y = xr.open_dataset(fn)
                    n_hours = len(ds_y["valid_time"])
                    tot_present += n_hours
                    status = "OK" if n_hours == exp_hours else "MISMATCH"
                    years_detail.append(
                        {
                            "year": y,
                            "expected_hours": exp_hours,
                            "present_hours": n_hours,
                            "status": status,
                        }
                    )
                except Exception as ex:  # noqa: BLE001
                    years_detail.append(
                        {
                            "year": y,
                            "expected_hours": exp_hours,
                            "present_hours": 0,
                            "status": f"ERROR: {ex}",
                        }
                    )

        # Missing years analysis for evaluation window 1990..2012
        # (Window 12:00 UTC 31-Dec-2011 to 12:00 UTC 01-Jan-2012 requires 2012)
        missing_12 = [
            y for y in range(1990, 2013) if not (era5_dir / f"{prefix}{y}.nc").exists()
        ]
        missing_1990_2012[var_id] = missing_12

        # Missing years analysis through 2025
        missing_25 = [
            y for y in range(1990, 2026) if not (era5_dir / f"{prefix}{y}.nc").exists()
        ]
        missing_through_2025[var_id] = missing_25

        time_series_vars.append(
            {
                "variable_id": var_id,
                "short_name": short_name,
                "long_name": var_long_name,
                "units": var_units,
                "total_files": len(files),
                "expected_total_hours": tot_expected,
                "present_total_hours": tot_present,
                "missing_years": missing_years,
                "years_detail": years_detail,
            }
        )

    return Era5InventoryResult(
        era5_dir=str(era5_dir),
        static_file_present=static_present,
        static_variables=static_vars_info,
        time_series_variables=time_series_vars,
        missing_for_1990_2012_window=missing_1990_2012,
        missing_for_2025_extension=missing_through_2025,
    )


def extract_clean_numeric_series(
    df_raw: pd.DataFrame,
    value_col: str = "Valor",
    date_col: str = "_norm_date",
) -> Tuple[pd.Series, pd.Series]:
    """Extract clean float series and raw string series indexed by normalized date."""
    if date_col not in df_raw.columns:
        df_raw[date_col] = (
            pd.to_datetime(df_raw["Fecha"], errors="coerce").dt.normalize()
            if "Fecha" in df_raw.columns
            else pd.date_range("1990-01-01", periods=len(df_raw))
        )

    clean_df = df_raw.dropna(subset=[date_col]).copy()

    if "NivelAprobacion" in clean_df.columns:
        prio_map = {"Definitivo": 1, "Preliminar": 2}
        clean_df["_prio"] = clean_df["NivelAprobacion"].map(
            lambda x: prio_map.get(str(x).strip(), 99)
        )
        clean_df = clean_df.sort_values(by=[date_col, "_prio"])
        clean_df = clean_df.drop_duplicates(subset=[date_col], keep="first")
    else:
        clean_df = clean_df.drop_duplicates(subset=[date_col], keep="first")

    raw_vals = (
        clean_df[value_col].fillna("").astype(str).str.strip()
        if value_col in clean_df.columns
        else pd.Series([], dtype=str)
    )

    float_vals = pd.to_numeric(
        raw_vals.str.replace(",", "."),
        errors="coerce",
    )
    for s in ["-999", "-999.0", "9999", "9999.0", "-9999", "nan", "NaN"]:
        float_vals[raw_vals == s] = np.nan

    float_series = pd.Series(
        float_vals.to_numpy(),
        index=pd.DatetimeIndex(clean_df[date_col]),
        name=value_col,
    ).sort_index()

    raw_series = pd.Series(
        raw_vals.to_numpy(),
        index=pd.DatetimeIndex(clean_df[date_col]),
        name=value_col,
    ).sort_index()

    return float_series, raw_series


def inspect_quality_flags_only(filepath: Path) -> Dict[str, int]:
    """Read ONLY the NivelAprobacion column and return distinct value counts."""
    df = pd.read_csv(filepath, usecols=["NivelAprobacion"])
    counts = df["NivelAprobacion"].value_counts(dropna=False)
    return {str(k): int(v) for k, v in counts.items()}


def generate_markdown_report(
    station_id: str,
    structure_reports: List[FileStructureResult],
    sentinel_reports: List[SentinelAuditResult],
    physical_reports: List[PhysicalRulesResult],
    temp_consistency: Optional[TemperatureConsistencyResult],
    completeness_reports: List[CompletenessResult],
    simultaneous_temp: Optional[SimultaneousTemperatureResult],
    triplet_overlap: Optional[TripletOverlapResult],
    metadata_res: MetadataAuditResult,
    era5_inv: Optional[Era5InventoryResult],
    jb_ac_quality_flags: Optional[Dict[str, Dict[str, int]]] = None,
) -> str:
    """Format audit results into a structured, readable Markdown document."""
    lines: List[str] = [
        f"# INFORME DE AUDITORÍA DE DATOS DE ESTACIÓN: {station_id}",
        "",
        f"- **Fecha de generación:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "- **Protocolo:** TROPICOR Non-Leakage Data Audit Protocol",
        "- **Restricción estricta:** Tarea de solo lectura. Sin estadísticas globales (medias, medianas, cuantiles, min/max) ni comparaciones con ERA5.",
        "",
        "---",
        "",
        "## 1. Estructura de Archivos y Calendario",
        "",
    ]

    for s in structure_reports:
        match_str = (
            "SÍ (54085010)"
            if s.station_code_matches_expected
            else f"NO (Encontrado: {s.station_codes})"
        )
        lines.extend(
            [
                f"### Archivo: `{Path(s.filepath).name}`",
                f"- **Ruta completa:** `{s.filepath}`",
                f"- **SHA-256:** `{s.file_sha256}`",
                f"- **Código(s) en archivo:** `{', '.join(s.station_codes)}` (Coincide con esperado: {match_str})",
                f"- **Nombre(s) en archivo:** `{', '.join(s.station_names)}`",
                f"- **Filas totales:** {s.total_raw_rows}",
                f"- **Fechas parseables:** {s.parseable_dates_count} | **No parseables:** {s.unparseable_dates_count}",
                f"- **Fechas duplicadas:** {s.duplicate_dates_count}",
                f"- **Orden temporal monótono creciente:** {'SÍ' if s.is_monotonic_increasing else 'NO (inversiones: ' + str(s.date_inversion_count) + ')'}",
                f"- **Rango observado en archivo:** {s.earliest_date} a {s.latest_date}",
                f"- **Rango esperado:** {s.expected_start} a {s.expected_end} ({s.expected_calendar_days} días)",
                f"- **Días presentes en rango esperado:** {s.present_calendar_days_in_range}",
                f"- **Días calendario ausentes en rango esperado:** {s.missing_calendar_days_count}",
                "",
            ]
        )
        if s.missing_calendar_ranges:
            lines.append("#### Intervalos de días calendario ausentes:")
            lines.append("| Inicio | Fin | Días Ausentes |")
            lines.append("| :--- | :--- | :--- |")
            for r in s.missing_calendar_ranges[:25]:
                lines.append(f"| {r['start']} | {r['end']} | {r['days']} |")
            if len(s.missing_calendar_ranges) > 25:
                lines.append(
                    f"| ... | *(y {len(s.missing_calendar_ranges) - 25} intervalos más)* | |"
                )
            lines.append("")

    lines.extend(
        [
            "---",
            "",
            "## 2. Valores No Numéricos, Códigos Centinela y Banderas de Calidad",
            "",
        ]
    )

    for sent in sentinel_reports:
        lines.append(
            f"### Variable: **{sent.variable_label}** (`{Path(sent.filepath).name}`)"
        )
        lines.append("")
        lines.append("**Códigos Centinela detectados:**")
        if sent.sentinel_counts:
            lines.append("| Código Centinela | Ocurrencias |")
            lines.append("| :--- | :--- |")
            for code, cnt in sent.sentinel_counts.items():
                lines.append(f"| `{code}` | {cnt} |")
        else:
            lines.append("*Ninguno detectado.*")
        lines.append("")

        lines.append("**Valores no numéricos no estándar:**")
        if sent.non_numeric_counts:
            lines.append("| Token no numérico | Ocurrencias |")
            lines.append("| :--- | :--- |")
            for tok, cnt in sent.non_numeric_counts.items():
                lines.append(f"| `{tok}` | {cnt} |")
        else:
            lines.append("*Ninguno detectado.*")
        lines.append("")

        lines.append("**Distribución de Banderas de Calidad (`NivelAprobacion`):**")
        if sent.quality_flag_counts:
            lines.append("| Nivel de Aprobación | Registros |")
            lines.append("| :--- | :--- |")
            for flag, cnt in sent.quality_flag_counts.items():
                lines.append(f"| {flag} | {cnt} |")
        else:
            lines.append("*Columna no presente.*")
        lines.append("")

    if jb_ac_quality_flags:
        lines.extend(
            [
                "### Inspección de Banderas de Calidad en Benchmark (JB y AC):",
                "*(Leído exclusivamente desde la columna `NivelAprobacion`)*",
                "",
            ]
        )
        for name_bench, q_counts in jb_ac_quality_flags.items():
            lines.append(f"#### `{name_bench}`:")
            lines.append("| Nivel de Aprobación | Conteo |")
            lines.append("| :--- | :--- |")
            for k, v in q_counts.items():
                lines.append(f"| {k} | {v} |")
            lines.append("")

    lines.extend(
        [
            "---",
            "",
            "## 3. Reglas Físicas, Rachas y Resolución Observada",
            "",
            "*(Solo se listan fecha y valor de los registros marcados)*",
            "",
        ]
    )

    for phys in physical_reports:
        lines.append(f"### Variable: **{phys.variable_label}**")
        lines.append(
            f"- **Valores bajo límite físico inferior:** {phys.below_min_count}"
        )
        if phys.below_min_records:
            lines.append("  | Fecha | Valor Marcado |")
            lines.append("  | :--- | :--- |")
            for r in phys.below_min_records:
                lines.append(f"  | {r['date']} | {r['value']} |")
        lines.append(
            f"- **Valores sobre límite físico superior:** {phys.above_max_count}"
        )
        if phys.above_max_records:
            lines.append("  | Fecha | Valor Marcado |")
            lines.append("  | :--- | :--- |")
            for r in phys.above_max_records:
                lines.append(f"  | {r['date']} | {r['value']} |")

        lines.append(f"- **Rachas detectadas (duración mínima):** {phys.streaks_count}")
        if phys.streaks:
            lines.append("  | Inicio | Fin | Duración (días) | Valor Constante |")
            lines.append("  | :--- | :--- | :--- | :--- |")
            for stk in phys.streaks:
                lines.append(
                    f"  | {stk['start_date']} | {stk['end_date']} | {stk['length']} | {stk['value']} |"
                )

        lines.append("- **Distribución de decimales en registros numéricos:**")
        lines.append("  | Decimales | Conteo de Registros |")
        lines.append("  | :--- | :--- |")
        for dec, cnt in phys.decimal_places_counts.items():
            lines.append(f"  | {dec} | {cnt} |")
        lines.append(
            f"- **Resolución inferida:** `{phys.inferred_resolution_mm_or_deg}`"
        )
        lines.append("")

    if temp_consistency:
        lines.extend(
            [
                "### Consistencia Térmica (Tmax < Tmin el mismo día):",
                f"- **Inversiones detectadas:** {temp_consistency.inversion_count}",
                "",
            ]
        )
        if temp_consistency.inversion_records:
            lines.append("| Fecha | Tmax (°C) | Tmin (°C) |")
            lines.append("| :--- | :--- | :--- |")
            for inv in temp_consistency.inversion_records:
                lines.append(f"| {inv['date']} | {inv['tmax']} | {inv['tmin']} |")
            lines.append("")

    lines.extend(
        [
            "---",
            "",
            "## 4. Completitud de Series y Solapamiento",
            "",
        ]
    )

    for comp in completeness_reports:
        lines.extend(
            [
                f"### Variable: **{comp.variable_label}** [{comp.evaluation_scope}]",
                f"- **Rango evaluado:** {comp.expected_start} a {comp.expected_end} ({comp.total_expected_days} días)",
                f"- **Días válidos totales:** {comp.total_valid_days}",
                f"- **Días faltantes totales:** {comp.total_missing_days}",
                f"- **Años con >= 330 días válidos:** {comp.years_ge_330_count} / {len(comp.annual_summary)} años ({comp.years_ge_330_list})",
                f"- **Meses con > 20% faltante:** {comp.months_gt_20pct_missing_count} meses",
                "",
                "#### Completitud Anual:",
                "| Año | Días Calendario | Días Válidos | Días Faltantes | % Válido | >= 330 Días |",
                "| :--- | :--- | :--- | :--- | :--- | :--- |",
            ]
        )
        for ann in comp.annual_summary:
            flag_ge = "SÍ" if ann["ge_330"] else "NO"
            lines.append(
                f"| {ann['year']} | {ann['total_days']} | {ann['valid_days']} | {ann['missing_days']} | {ann['valid_pct']:.2f}% | {flag_ge} |"
            )
        lines.append("")

        if comp.months_gt_20pct_missing_list:
            lines.extend(
                [
                    f"#### Meses con > 20% de datos faltantes ({comp.variable_label}):",
                    "| Mes | Días del Mes | Días Válidos | Días Faltantes | % Faltante |",
                    "| :--- | :--- | :--- | :--- | :--- |",
                ]
            )
            for m in comp.months_gt_20pct_missing_list:
                lines.append(
                    f"| {m['year_month']} | {m['total_days']} | {m['valid_days']} | {m['missing_days']} | {m['pct_missing']:.2f}% |"
                )
            lines.append("")

    if simultaneous_temp:
        lines.extend(
            [
                "### Completitud Simultánea de Temperatura (Tmax y Tmin válidos el mismo día):",
                f"- **Días simultáneos válidos:** {simultaneous_temp.total_simultaneous_valid_days}",
                f"- **Años con >= 330 días simultáneos:** {simultaneous_temp.years_simultaneous_ge_330_count} ({simultaneous_temp.years_simultaneous_ge_330_list})",
                "",
                "| Año | Total Días | Días Válidos Simultáneos | % Válido | >= 330 Días |",
                "| :--- | :--- | :--- | :--- | :--- |",
            ]
        )
        for ann in simultaneous_temp.annual_simultaneous_valid:
            flag_ge = "SÍ" if ann["ge_330"] else "NO"
            lines.append(
                f"| {ann['year']} | {ann['total_days']} | {ann['simultaneous_valid_days']} | {ann['valid_pct']:.2f}% | {flag_ge} |"
            )
        lines.append("")

    if triplet_overlap:
        lines.extend(
            [
                "### Solapamiento de Tres Variables (Precipitación + Tmax + Tmin):",
                f"- **Días totales con coincidencia triple:** {triplet_overlap.total_triplet_valid_days}",
                f"- **Años con al menos 1 día con datos triples:** {triplet_overlap.years_triplet_any_count} / {len(triplet_overlap.annual_triplet_summary)}",
                f"- **Años con >= 330 días triples válidos:** {triplet_overlap.years_triplet_ge_330_count} ({triplet_overlap.years_triplet_ge_330_list})",
                "",
                "| Año | Total Días | Días Triples Válidos | % Válido | >= 330 Días |",
                "| :--- | :--- | :--- | :--- | :--- |",
            ]
        )
        for ann in triplet_overlap.annual_triplet_summary:
            flag_ge = "SÍ" if ann["ge_330"] else "NO"
            lines.append(
                f"| {ann['year']} | {ann['total_days']} | {ann['triplet_valid_days']} | {ann['valid_pct']:.2f}% | {flag_ge} |"
            )
        lines.append("")

    lines.extend(
        [
            "---",
            "",
            "## 5. Metadatos de Estación y Comparación con Cuadrícula ERA5",
            "",
            "*(Sin datos observacionales ni meteorológicos)*",
            "",
            "| Parámetro | Valor |",
            "| :--- | :--- |",
            f"| **Código Estación** | {metadata_res.station_code} |",
            f"| **Nombre Estación** | {metadata_res.station_name} |",
            f"| **Departamento / Municipio** | {metadata_res.department} / {metadata_res.municipality} |",
            f"| **Región Natural** | {metadata_res.natural_region} |",
            f"| **Latitud Estación** | {metadata_res.latitude:.5f}° N |"
            if metadata_res.latitude is not None
            else "| **Latitud Estación** | Desconocida |",
            f"| **Longitud Estación** | {metadata_res.longitude:.5f}° W |"
            if metadata_res.longitude is not None
            else "| **Longitud Estación** | Desconocida |",
            f"| **Elevación Catálogo (BENCHMARK)** | {metadata_res.elevation_masl:.1f} msnm |"
            if metadata_res.elevation_masl is not None
            else "| **Elevación Catálogo** | No registrada |",
            f"| **Elevación CNE (IDEAM)** | {metadata_res.cne_elevation_masl:.1f} msnm |"
            if metadata_res.cne_elevation_masl is not None
            else "| **Elevación CNE** | No registrada |",
            f"| **Celda ERA5 más cercana** | Lat: {metadata_res.era5_nearest_latitude:.2f}° N, Lon: {metadata_res.era5_nearest_longitude:.2f}° W |"
            if metadata_res.era5_nearest_latitude is not None
            else "| **Celda ERA5 más cercana** | N/A |",
            f"| **Distancia a celda más cercana** | {metadata_res.distance_to_nearest_era5_km:.2f} km |"
            if metadata_res.distance_to_nearest_era5_km is not None
            else "| **Distancia a celda más cercana** | N/A |",
            f"| **Land-Sea Mask (LSM) celda más cercana** | {metadata_res.nearest_era5_lsm:.4f} |"
            if metadata_res.nearest_era5_lsm is not None
            else "| **LSM celda más cercana** | N/A |",
        ]
    )

    if metadata_res.surrounding_era5_lsm:
        lines.append("### LSM de las 4 Celdas ERA5 Circundantes:")
        lines.append("| Dirección | Latitud | Longitud | Land-Sea Mask (LSM) |")
        lines.append("| :--- | :--- | :--- | :--- |")
        for card, info in metadata_res.surrounding_era5_lsm.items():
            lines.append(
                f"| {card} | {info['latitude']:.2f}° N | {info['longitude']:.2f}° W | {info['lsm']:.4f} |"
            )
        lines.append("")

    lines.extend(
        [
            f"| **Geopotencial ERA5 (z)** | {metadata_res.era5_geopotential_z} m²/s² |"
            if metadata_res.era5_geopotential_z is not None
            else "| **Geopotencial ERA5 (z)** | N/A |",
            f"| **Elevación DEM ERA5 (h = z/g)** | {metadata_res.era5_dem_elevation_m} m |"
            if metadata_res.era5_dem_elevation_m is not None
            else "| **Elevación DEM ERA5** | N/A |",
            f"| **Diferencia de Elevación (DEM - Catálogo)** | {metadata_res.elevation_difference_vs_catalog_m:+.2f} m |"
            if metadata_res.elevation_difference_vs_catalog_m is not None
            else "| **Diferencia de Elevación (DEM - Catálogo)** | N/A |",
            f"| **Diferencia de Elevación (DEM - CNE)** | {metadata_res.elevation_difference_vs_cne_m:+.2f} m |"
            if metadata_res.elevation_difference_vs_cne_m is not None
            else "| **Diferencia de Elevación (DEM - CNE)** | N/A |",
            "",
            "---",
            "",
            "## 6. Inventario de Archivos ERA5 Reanálisis y Años Faltantes",
            "",
            "*(Sin estadísticas descriptivas de los campos meteorológicos)*",
            "",
        ]
    )

    if era5_inv:
        lines.append(f"- **Directorio:** `{era5_inv.era5_dir}`")
        lines.append(
            f"- **Archivo estático presente (`era5_static.nc`):** {'SÍ' if era5_inv.static_file_present else 'NO'}"
        )
        if era5_inv.static_variables:
            lines.append("  - Variables estáticas:")
            for sv in era5_inv.static_variables:
                lines.append(
                    f"    - `{sv.get('variable')}` ({sv.get('long_name')}): unidades `{sv.get('units')}`, forma {sv.get('shape')}"
                )
        lines.append("")

        if era5_inv.time_series_variables:
            lines.append("### Variables Horarias Reanálisis:")
            for tvar in era5_inv.time_series_variables:
                lines.extend(
                    [
                        f"#### `{tvar['variable_id']}` (`{tvar['short_name']}`):",
                        f"- **Nombre descriptivo:** {tvar['long_name']}",
                        f"- **Unidades:** `{tvar['units']}`",
                        f"- **Archivos anuales presentes:** {tvar['total_files']}",
                        f"- **Pasos horarios esperados (1990-2011):** {tvar['expected_total_hours']}",
                        f"- **Pasos horarios presentes:** {tvar['present_total_hours']}",
                        f"- **Años ausentes en 1990-2011:** {tvar['missing_years'] if tvar['missing_years'] else 'Ninguno (100% completo)'}",
                        "",
                    ]
                )

        lines.extend(
            [
                "### Análisis de Años Faltantes de ERA5:",
                "",
                "#### Para cubrir 1990-01-01 a 2012-01-01 (Ventana 12-12 UTC requiere 01-ene-2012):",
                f"- **`total_precipitation` (tp):** Años faltantes = {era5_inv.missing_for_1990_2012_window.get('total_precipitation', [])} ({len(era5_inv.missing_for_1990_2012_window.get('total_precipitation', []))} años: 2011 y 2012)",
                f"- **`2m_temperature` (t2m):** Años faltantes = {era5_inv.missing_for_1990_2012_window.get('2m_temperature', [])} ({len(era5_inv.missing_for_1990_2012_window.get('2m_temperature', []))} año: 2012)",
                "",
                "#### Para cubrir la extensión completa observada (hasta 2025):",
                f"- **`total_precipitation` (tp):** Años faltantes = {era5_inv.missing_for_2025_extension.get('total_precipitation', [])} ({len(era5_inv.missing_for_2025_extension.get('total_precipitation', []))} años: 2011 a 2025)",
                f"- **`2m_temperature` (t2m):** Años faltantes = {era5_inv.missing_for_2025_extension.get('2m_temperature', [])} ({len(era5_inv.missing_for_2025_extension.get('2m_temperature', []))} años: 2012 a 2025)",
                "",
            ]
        )
    else:
        lines.append("*Directorio ERA5 no especificado o inexistente.*")

    lines.append("")
    return "\n".join(lines)


def run_audit(
    station_id: str,
    pr_path: Optional[Path] = None,
    tmax_path: Optional[Path] = None,
    tmin_path: Optional[Path] = None,
    cne_path: Optional[Path] = None,
    era5_dir: Optional[Path] = None,
    start_date: str = "1990-01-01",
    end_date: str = "2011-12-31",
    output_dir: Path = Path("local/reports"),
    check_jb_ac_flags: bool = True,
) -> Tuple[Path, Path]:
    """Execute generic station audit and export markdown and JSON reports."""
    output_dir.mkdir(parents=True, exist_ok=True)

    structure_results: List[FileStructureResult] = []
    sentinel_results: List[SentinelAuditResult] = []
    physical_results: List[PhysicalRulesResult] = []
    completeness_results: List[CompletenessResult] = []

    pr_full_series: Optional[pd.Series] = None
    tmax_full_series: Optional[pd.Series] = None
    tmin_full_series: Optional[pd.Series] = None

    # Audit Precipitation if provided
    if pr_path and pr_path.exists():
        struct_pr, df_pr = audit_file_structure(
            pr_path,
            expected_station_id=station_id,
            expected_start=start_date,
            expected_end=end_date,
        )
        structure_results.append(struct_pr)

        sent_pr = audit_sentinels_and_flags(
            df_pr, variable_label="Precipitación", filepath=str(pr_path)
        )
        sentinel_results.append(sent_pr)

        s_pr, raw_pr = extract_clean_numeric_series(df_pr)
        pr_full_series = s_pr

        # Protocol range evaluation
        s_pr_eval = s_pr.loc[start_date:end_date]
        raw_pr_eval = raw_pr.loc[start_date:end_date]
        phys_pr = audit_physical_rules(
            s_pr_eval,
            raw_pr_eval,
            variable_label="Precipitación (mm) [1990-2011]",
            lower_bound=0.0,
            upper_bound=500.0,
            min_streak_days=5,
            ignore_zero_streaks=True,
        )
        physical_results.append(phys_pr)

        comp_pr_proto = audit_completeness(
            s_pr_eval,
            variable_label="Precipitación",
            expected_start=start_date,
            expected_end=end_date,
            evaluation_scope="Rango Protocolo (1990-2011)",
        )
        completeness_results.append(comp_pr_proto)

        # Full observed range evaluation
        if not s_pr.empty:
            pr_min_yr = s_pr.index.min().year
            pr_max_yr = s_pr.index.max().year
            if str(pr_min_yr) != start_date[:4] or str(pr_max_yr) != end_date[:4]:
                comp_pr_full = audit_completeness(
                    s_pr,
                    variable_label="Precipitación",
                    expected_start=f"{pr_min_yr}-01-01",
                    expected_end=f"{pr_max_yr}-12-31",
                    evaluation_scope=f"Rango Completo Observado ({pr_min_yr}-{pr_max_yr})",
                )
                completeness_results.append(comp_pr_full)

    # Audit Tmax if provided
    if tmax_path and tmax_path.exists():
        struct_tmax, df_tmax = audit_file_structure(
            tmax_path,
            expected_station_id=station_id,
            expected_start=start_date,
            expected_end=end_date,
        )
        structure_results.append(struct_tmax)

        sent_tmax = audit_sentinels_and_flags(
            df_tmax,
            variable_label="Temperatura Máxima",
            filepath=str(tmax_path),
        )
        sentinel_results.append(sent_tmax)

        s_tmax, raw_tmax = extract_clean_numeric_series(df_tmax)
        tmax_full_series = s_tmax

        s_tmax_eval = s_tmax.loc[start_date:end_date]
        raw_tmax_eval = raw_tmax.loc[start_date:end_date]
        phys_tmax = audit_physical_rules(
            s_tmax_eval,
            raw_tmax_eval,
            variable_label="Temperatura Máxima (°C) [1990-2011]",
            lower_bound=5.0,
            upper_bound=45.0,
            min_streak_days=10,
            ignore_zero_streaks=False,
        )
        physical_results.append(phys_tmax)

        comp_tmax_proto = audit_completeness(
            s_tmax_eval,
            variable_label="Temperatura Máxima",
            expected_start=start_date,
            expected_end=end_date,
            evaluation_scope="Rango Protocolo (1990-2011)",
        )
        completeness_results.append(comp_tmax_proto)

        if not s_tmax.empty:
            tmax_min_yr = s_tmax.index.min().year
            tmax_max_yr = s_tmax.index.max().year
            if str(tmax_min_yr) != start_date[:4] or str(tmax_max_yr) != end_date[:4]:
                comp_tmax_full = audit_completeness(
                    s_tmax,
                    variable_label="Temperatura Máxima",
                    expected_start=f"{tmax_min_yr}-01-01",
                    expected_end=f"{tmax_max_yr}-12-31",
                    evaluation_scope=f"Rango Completo Observado ({tmax_min_yr}-{tmax_max_yr})",
                )
                completeness_results.append(comp_tmax_full)

    # Audit Tmin if provided
    if tmin_path and tmin_path.exists():
        struct_tmin, df_tmin = audit_file_structure(
            tmin_path,
            expected_station_id=station_id,
            expected_start=start_date,
            expected_end=end_date,
        )
        structure_results.append(struct_tmin)

        sent_tmin = audit_sentinels_and_flags(
            df_tmin,
            variable_label="Temperatura Mínima",
            filepath=str(tmin_path),
        )
        sentinel_results.append(sent_tmin)

        s_tmin, raw_tmin = extract_clean_numeric_series(df_tmin)
        tmin_full_series = s_tmin

        s_tmin_eval = s_tmin.loc[start_date:end_date]
        raw_tmin_eval = raw_tmin.loc[start_date:end_date]
        phys_tmin = audit_physical_rules(
            s_tmin_eval,
            raw_tmin_eval,
            variable_label="Temperatura Mínima (°C) [1990-2011]",
            lower_bound=-5.0,
            upper_bound=35.0,
            min_streak_days=10,
            ignore_zero_streaks=False,
        )
        physical_results.append(phys_tmin)

        comp_tmin_proto = audit_completeness(
            s_tmin_eval,
            variable_label="Temperatura Mínima",
            expected_start=start_date,
            expected_end=end_date,
            evaluation_scope="Rango Protocolo (1990-2011)",
        )
        completeness_results.append(comp_tmin_proto)

        if not s_tmin.empty:
            tmin_min_yr = s_tmin.index.min().year
            tmin_max_yr = s_tmin.index.max().year
            if str(tmin_min_yr) != start_date[:4] or str(tmin_max_yr) != end_date[:4]:
                comp_tmin_full = audit_completeness(
                    s_tmin,
                    variable_label="Temperatura Mínima",
                    expected_start=f"{tmin_min_yr}-01-01",
                    expected_end=f"{tmin_max_yr}-12-31",
                    evaluation_scope=f"Rango Completo Observado ({tmin_min_yr}-{tmin_max_yr})",
                )
                completeness_results.append(comp_tmin_full)

    # Temperature consistency (inversions and simultaneous)
    temp_consistency: Optional[TemperatureConsistencyResult] = None
    simultaneous_temp: Optional[SimultaneousTemperatureResult] = None
    if tmax_full_series is not None and tmin_full_series is not None:
        temp_consistency = audit_temperature_inversions(
            tmax_full_series, tmin_full_series
        )
        simultaneous_temp = audit_simultaneous_temperature(
            tmax_full_series,
            tmin_full_series,
            expected_start=start_date,
            expected_end=end_date,
        )

    # Triplet Overlap (Precip + Tmax + Tmin)
    triplet_overlap: Optional[TripletOverlapResult] = None
    if (
        pr_full_series is not None
        and tmax_full_series is not None
        and tmin_full_series is not None
    ):
        triplet_overlap = audit_triplet_overlap(
            pr_series=pr_full_series,
            tmax_series=tmax_full_series,
            tmin_series=tmin_full_series,
            start_date=start_date,
            end_date=end_date,
        )

    # Metadata audit
    meta_result = audit_metadata(
        station_id=station_id,
        cne_path=cne_path,
        era5_dir=era5_dir,
    )

    # ERA5 inventory
    era5_inv: Optional[Era5InventoryResult] = None
    if era5_dir and era5_dir.exists():
        start_yr = int(start_date.split("-")[0])
        end_yr = int(end_date.split("-")[0])
        era5_inv = audit_era5_inventory(
            era5_dir=era5_dir,
            expected_start_year=start_yr,
            expected_end_year=end_yr,
        )

    # Benchmark JB and AC quality flag check if requested
    jb_ac_flags: Optional[Dict[str, Dict[str, int]]] = None
    if check_jb_ac_flags:
        jb_ac_flags = {}
        jb_p = Path("local/newFiles/DiaPluviometrico_JardinBotanico_21205710.csv")
        ac_p = Path("local/newFiles/DiaPluviometrico_AguasClaras.csv")
        if jb_p.exists():
            jb_ac_flags["Jardín Botánico (21205710)"] = inspect_quality_flags_only(jb_p)
        if ac_p.exists():
            jb_ac_flags["Aguas Claras (23215030)"] = inspect_quality_flags_only(ac_p)

    # Generate Markdown
    md_content = generate_markdown_report(
        station_id=station_id,
        structure_reports=structure_results,
        sentinel_reports=sentinel_results,
        physical_reports=physical_results,
        temp_consistency=temp_consistency,
        completeness_reports=completeness_results,
        simultaneous_temp=simultaneous_temp,
        triplet_overlap=triplet_overlap,
        metadata_res=meta_result,
        era5_inv=era5_inv,
        jb_ac_quality_flags=jb_ac_flags,
    )

    md_path = output_dir / f"audit_{station_id}.md"
    md_path.write_text(md_content, encoding="utf-8")

    # Generate JSON
    json_data = {
        "station_id": station_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "structure": [asdict(s) for s in structure_results],
        "sentinels": [asdict(s) for s in sentinel_results],
        "physical_rules": [asdict(p) for p in physical_results],
        "temperature_consistency": asdict(temp_consistency)
        if temp_consistency
        else None,
        "completeness": [asdict(c) for c in completeness_results],
        "simultaneous_temperature": asdict(simultaneous_temp)
        if simultaneous_temp
        else None,
        "triplet_overlap": asdict(triplet_overlap) if triplet_overlap else None,
        "metadata": asdict(meta_result),
        "era5_inventory": asdict(era5_inv) if era5_inv else None,
        "jb_ac_quality_flags": jb_ac_flags,
    }
    json_path = output_dir / f"audit_{station_id}.json"
    json_path.write_text(json.dumps(json_data, indent=2), encoding="utf-8")

    return md_path, json_path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Audit station meteorological observations and ERA5 reanalysis archives (Read-Only)."
    )
    parser.add_argument(
        "--station-id",
        type=str,
        required=True,
        help="Station code / identifier (e.g. 54085010)",
    )
    parser.add_argument(
        "--pr-path",
        type=Path,
        default=None,
        help="Path to daily precipitation CSV / Excel file",
    )
    parser.add_argument(
        "--tmax-path",
        type=Path,
        default=None,
        help="Path to daily Tmax CSV / Excel file",
    )
    parser.add_argument(
        "--tmin-path",
        type=Path,
        default=None,
        help="Path to daily Tmin CSV / Excel file",
    )
    parser.add_argument(
        "--cne-path",
        type=Path,
        default=Path("local/CNE_IDEAM.xls"),
        help="Path to CNE_IDEAM.xls catalog",
    )
    parser.add_argument(
        "--era5-dir",
        type=Path,
        default=None,
        help="Path to local ERA5 directory for the station",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="1990-01-01",
        help="Expected calendar start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2011-12-31",
        help="Expected calendar end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("local/reports"),
        help="Directory to save audit reports",
    )
    parser.add_argument(
        "--no-jb-ac-check",
        action="store_true",
        help="Skip checking JB and AC quality flags",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    md_path, json_path = run_audit(
        station_id=args.station_id,
        pr_path=args.pr_path,
        tmax_path=args.tmax_path,
        tmin_path=args.tmin_path,
        cne_path=args.cne_path,
        era5_dir=args.era5_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir,
        check_jb_ac_flags=not args.no_jb_ac_check,
    )
    print(md_path.read_text(encoding="utf-8"))
    print("\nAudit completed successfully.")
    print(f"Markdown report: {md_path}")
    print(f"JSON report:     {json_path}")


if __name__ == "__main__":
    main()
