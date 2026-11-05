"""Full-record observation and ERA5 loaders for benchmark evaluation.

This module performs file I/O and reads full observation and ERA5 series,
including validation years; it never reads the ledger file and never imports or
calls any ledger function; the orchestrator reads the ledger, verifies that the
records are in it, and passes them in.

Includes Section J.2 station constants, byte-level SHA-256 file hashing,
file discovery, and daily ERA5 aggregation mirroring Phase 3. The Section E.6.1
plausibility check is applied to the requested years (J.3.1), and J.3.3 screening
counts are computed elsewhere.
"""

import hashlib
import re
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union, cast

import numpy as np
import pandas as pd

from tropicor.downscale.temperature import (
    MAX_PLAUSIBLE_TEMP_C,
    MIN_PLAUSIBLE_TEMP_C,
)
from tropicor.io.era5 import (
    ERA5Adapter,
    aggregate_hourly_to_daily_temperature,
)


@dataclass(frozen=True)
class StationConstants:
    """Fixed audited station geographic and ERA5 static parameters (Section J.2)."""

    latitude: float
    longitude: float
    elevation_m: float
    cell_lat: float
    cell_lon: float
    z_era5_m: float
    delta_z_m: float


# J.2 audited constants; station catalog is deliberately not used.
# Convention: delta_z_m = elevation_m - z_era5_m.
BENCHMARK_STATION_CONSTANTS: Mapping[str, StationConstants] = {
    "21205710": StationConstants(
        latitude=4.667867,
        longitude=-74.101034,
        elevation_m=2552.0,
        cell_lat=4.75,
        cell_lon=-74.00,
        z_era5_m=2753.4,
        delta_z_m=-201.4,
    ),
    "23215030": StationConstants(
        latitude=8.228889,
        longitude=-73.602778,
        elevation_m=133.8,
        cell_lat=8.25,
        cell_lon=-73.50,
        z_era5_m=762.4,
        delta_z_m=-628.6,
    ),
}


@dataclass(frozen=True, eq=False)
class ERA5DailyRecord:
    """Container for daily ERA5 aggregated temperature series and metadata."""

    station_id: str
    tmax: pd.Series = field(repr=False)
    tmin: pd.Series = field(repr=False)
    files_used: tuple[str, ...]


@dataclass(frozen=True, eq=False)
class ObservationRecord:
    """Container for full-record station observation series and metadata.

    The series and approval attributes are excluded from representation to
    prevent exposure of observation values or data labels in strings or logs.
    Equality is identity-based (eq=False) to avoid ambiguous truth-value errors
    from underlying pandas Series.
    """

    station_id: str
    series: pd.Series = field(repr=False)
    n_station_rows: int
    n_identical_duplicates_dropped: int
    approval: Optional[pd.Series] = field(default=None, repr=False)


def _normalize_station_id(station_id: str) -> str:
    """Normalize a station identifier by stripping whitespace and .0 suffix."""
    return re.sub(r"\.0$", "", str(station_id).strip())


def _enforce_opening_guard(
    station_id: str,
    last_year: int,
    calibration_last_year: int,
    opening_records: Optional[Sequence[Mapping[str, Any]]],
    opening_purpose: Optional[str],
) -> None:
    """Enforce authorization guard before any validation-period data access.

    Shared by every loader of this module that can read validation-period data;
    the caller (orchestrator) has read the ledger and verified the records are
    in it. Performs no file access.
    """
    clean_id = _normalize_station_id(station_id)

    if last_year > calibration_last_year:
        if (
            opening_purpose is None
            or not isinstance(opening_purpose, str)
            or not opening_purpose.strip()
        ):
            raise ValueError(
                "opening_purpose must be a non-empty string when validation "
                "years are requested."
            )

        # OPEN ITEM v1.10: K.8.2 and J.10 do not name the ledger purpose label of the benchmark formal opening  # noqa: E501
        authorized = False
        if opening_records is not None:
            for rec in opening_records:
                if isinstance(rec, Mapping):
                    rec_st = _normalize_station_id(str(rec.get("station_id", "")))
                    rec_purpose = rec.get("purpose")
                    if rec_st == clean_id and rec_purpose == opening_purpose:
                        authorized = True
                        break

        if not authorized:
            raise PermissionError(
                f"Validation loading blocked for station '{clean_id}': "
                f"no matching opening record found for purpose '{opening_purpose}'."
            )


def read_observation_record(
    path: Union[str, Path],
    station_id: str,
    first_year: int,
    last_year: int,
    calibration_last_year: int,
    opening_records: Optional[Sequence[Mapping[str, Any]]] = None,
    opening_purpose: Optional[str] = None,
    read_approval: bool = False,
) -> ObservationRecord:
    """Read all rows of a station from an observation CSV, validate and slice.

    Validates date formats, numeric values, and duplicate dates over all rows
    of the target station before trimming to the requested year range and
    reindexing to a complete daily DatetimeIndex.

    When validation years are requested (last_year > calibration_last_year),
    an opening authorization record matching the target station and purpose
    must be supplied in opening_records before any file access takes place.
    The module never reads the ledger file directly.

    Args:
        path: Path to the raw observation CSV file.
        station_id: Station identifier (e.g. '21205710').
        first_year: First year of the requested evaluation period.
        last_year: Last year of the requested evaluation period.
        calibration_last_year: Last year of the station's calibration partition.
        opening_records: Sequence of opening authorization mappings from ledger.
        opening_purpose: Purpose string expected in the opening record.
        read_approval: If True, also extract and align NivelAprobacion flags.

    Returns:
        ObservationRecord containing daily series, row counts, and optional
        approval flags.

    Raises:
        ValueError: If first_year > last_year, opening_purpose is invalid,
            required columns are missing, station is absent, dates or values
            cannot be parsed, duplicate dates have distinct values, or values
            fall outside E.6.1 plausibility limits in the requested years.
        PermissionError: If validation years are requested without a matching
            opening record in opening_records.
        FileNotFoundError: If the observation file path does not exist.
    """
    if first_year > last_year:
        raise ValueError(
            f"Invalid year range: first_year ({first_year}) > last_year ({last_year})."
        )

    _enforce_opening_guard(
        station_id=station_id,
        last_year=last_year,
        calibration_last_year=calibration_last_year,
        opening_records=opening_records,
        opening_purpose=opening_purpose,
    )

    clean_id = _normalize_station_id(station_id)

    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Observation file not found: {path_obj}")

    read_category: Optional[str] = None
    try:
        df_raw = pd.read_csv(path_obj)
    except pd.errors.ParserError:
        read_category = "parser_error"
    except UnicodeDecodeError:
        read_category = "decode_error"
    except pd.errors.EmptyDataError:
        read_category = "empty_file"

    if read_category is not None:
        raise ValueError(f"Could not read observation file ({read_category}).")

    required_cols = ["CodigoEstacion", "Fecha", "Valor"]
    if read_approval:
        required_cols.append("NivelAprobacion")
    missing = [c for c in required_cols if c not in df_raw.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s) in observation file: {', '.join(missing)}"
        )

    st_codes = (
        df_raw["CodigoEstacion"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )
    if clean_id not in st_codes.values:
        raise ValueError(f"Station ID '{clean_id}' not found in observation file.")

    df_st = df_raw[st_codes == clean_id].copy()
    n_station_rows = int(len(df_st))

    parsed_dt = pd.to_datetime(df_st["Fecha"], format="ISO8601", errors="coerce")
    unparseable_mask = (
        parsed_dt.isna()
        & df_st["Fecha"].notna()
        & (df_st["Fecha"].astype(str).str.strip() != "")
    )
    unparseable_count = int(unparseable_mask.sum())
    if unparseable_count > 0:
        raise ValueError(
            f"Station '{clean_id}': found {unparseable_count} unparseable date(s)."
        )

    raw_val_str = df_st["Valor"].astype(str).str.strip().str.replace(",", ".")
    num_vals = pd.to_numeric(raw_val_str, errors="coerce")
    non_numeric_mask = (
        num_vals.isna()
        & df_st["Valor"].notna()
        & ~raw_val_str.str.upper().isin(["", "NAN", "NONE", "NULL"])
    )
    non_numeric_count = int(non_numeric_mask.sum())
    if non_numeric_count > 0:
        raise ValueError(
            f"Station '{clean_id}': found {non_numeric_count} non-numeric value(s) "
            "in column 'Valor'."
        )

    df_st["dt_norm"] = parsed_dt.dt.normalize()
    df_st["clean_val"] = num_vals
    dup_mask = df_st["dt_norm"].duplicated(keep=False)
    if dup_mask.any():
        dup_rows = df_st[dup_mask]
        affected_dates: list[pd.Timestamp] = []
        affected_rows_count = 0
        for dt_val, group in dup_rows.groupby("dt_norm"):
            valid_distinct = group["clean_val"].dropna().unique()
            if len(valid_distinct) > 1:
                affected_dates.append(pd.Timestamp(cast(Any, dt_val)))
                affected_rows_count += len(group)

        if affected_dates:
            affected_dates.sort()
            first_iso = affected_dates[0].strftime("%Y-%m-%d")
            raise ValueError(
                f"Station '{clean_id}': found {len(affected_dates)} duplicate "
                f"date(s) with distinct values; first affected date: {first_iso} "
                f"({affected_rows_count} rows involved)."
            )

    df_st = df_st.dropna(subset=["dt_norm"]).sort_values("dt_norm")
    rows_before_dedup = len(df_st)
    df_st = df_st.drop_duplicates(subset=["dt_norm"], keep="first")
    n_identical_duplicates_dropped = rows_before_dedup - len(df_st)

    s_full = pd.Series(
        df_st["clean_val"].values,
        index=pd.DatetimeIndex(df_st["dt_norm"]),
        dtype="float64",
    )

    s_approval: Optional[pd.Series] = None
    if read_approval:
        approval_vals = df_st["NivelAprobacion"].where(
            df_st["NivelAprobacion"].isna(),
            df_st["NivelAprobacion"].astype(str).str.strip(),
        )
        s_approval = pd.Series(
            approval_vals.values,
            index=pd.DatetimeIndex(df_st["dt_norm"]),
            dtype="object",
        )

    full_daily_index = pd.date_range(
        f"{first_year}-01-01", f"{last_year}-12-31", freq="D"
    )
    s_idx = cast(pd.DatetimeIndex, s_full.index)
    trim_mask = (s_idx.year >= first_year) & (s_idx.year <= last_year)

    s_trimmed = s_full[trim_mask]
    series_reindexed = s_trimmed.reindex(full_daily_index)

    # Section E.6.1 plausibility check on requested years (J.3.1; NaN ignored)
    valid_vals = series_reindexed.dropna()
    if not valid_vals.empty:
        implausible_mask = (valid_vals < MIN_PLAUSIBLE_TEMP_C) | (
            valid_vals > MAX_PLAUSIBLE_TEMP_C
        )
        n_implausible = int(implausible_mask.sum())
        if n_implausible > 0:
            raise ValueError(
                f"Station '{clean_id}': found {n_implausible} value(s) "
                "outside the E.6.1 plausibility limits in the requested years."
            )

    approval_reindexed: Optional[pd.Series] = None
    if s_approval is not None:
        s_app_trimmed = s_approval[trim_mask]
        approval_reindexed = s_app_trimmed.reindex(full_daily_index)

    return ObservationRecord(
        station_id=clean_id,
        series=series_reindexed,
        n_station_rows=n_station_rows,
        n_identical_duplicates_dropped=n_identical_duplicates_dropped,
        approval=approval_reindexed,
    )


def sha256_file(path: Union[str, Path]) -> str:
    """Compute the lowercase SHA-256 hexadecimal digest of a file's raw bytes.

    Reads in binary chunks without decoding, parsing, or interpreting data values.
    """
    path_obj = Path(path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"File not found for hashing: {path_obj}")
    hasher = hashlib.sha256()
    with open(path_obj, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def list_era5_files(
    era5_dir: Union[str, Path],
    years: Sequence[int],
) -> list[Path]:
    r"""Find and return sorted ERA5 NetCDF files for requested years.

    Matches files in era5_dir matching glob '*2m_temperature*.nc' whose filename
    contains each requested year as a whole number (regex (?<!\d)YYYY(?!\d)).

    Args:
        era5_dir: Directory containing ERA5 NetCDF files.
        years: Sequence of integer calendar years required.

    Returns:
        Sorted list of Path objects for the matched files.

    Raises:
        FileNotFoundError: If era5_dir does not exist or is not a directory.
        ValueError: If any requested year has no matching file, listing missing years.
    """
    dir_path = Path(era5_dir)
    if not dir_path.is_dir():
        raise FileNotFoundError(f"ERA5 directory not found: {dir_path}")

    all_nc_files = sorted(dir_path.glob("*2m_temperature*.nc"))

    matched_files: set[Path] = set()
    missing_years: list[int] = []

    for y in sorted(set(years)):
        pattern = re.compile(rf"(?<!\d){y}(?!\d)")
        year_files = [f for f in all_nc_files if pattern.search(f.name)]
        if not year_files:
            missing_years.append(y)
        else:
            matched_files.update(year_files)

    if missing_years:
        missing_str = ", ".join(str(y) for y in sorted(missing_years))
        raise ValueError(f"Missing ERA5 file(s) for year(s): {missing_str}")

    return sorted(matched_files)


def read_era5_daily(
    era5_dir: Union[str, Path],
    station_id: str,
    first_year: int,
    last_year: int,
    calibration_last_year: int,
    opening_records: Optional[Sequence[Mapping[str, Any]]] = None,
    opening_purpose: Optional[str] = None,
    *,
    window_start_utc: int,
    lag_days: int,
    min_hours: int,
) -> ERA5DailyRecord:
    """Read hourly ERA5 NetCDF files, aggregate to daily extremes, and validate.

    Order of verification:
    1. Validates requested year range (first_year <= last_year).
    2. Enforces opening authorization guard before any file access.
    3. Verifies station is registered in J.2 constants table.
    4. Lists required NetCDF files for years first_year..last_year+1.
    5. Loads hourly 2m temperature via ERA5Adapter, verifies grid cell,
       checks units ('K'), aggregates to daily 05:00-05:00 UTC, slices to
       first_year..last_year, and enforces NaN prohibition and E.6.1
       plausibility limits.

    Args:
        era5_dir: Directory containing ERA5 NetCDF files.
        station_id: Station identifier (e.g. '21205710').
        first_year: First year of the requested evaluation period.
        last_year: Last year of the requested evaluation period.
        calibration_last_year: Last year of the station's calibration partition.
        opening_records: Sequence of opening authorization mappings from ledger.
        opening_purpose: Purpose string expected in the opening record.
        window_start_utc: Starting hour of the 24h window in UTC (required).
        lag_days: Lag offset in days (required).
        min_hours: Minimum valid hours required in the 24h window (required).

    Returns:
        ERA5DailyRecord containing daily tmax, tmin, and files_used.

    Raises:
        ValueError: If first_year > last_year, station is unregistered,
            required files are missing, grid cell mismatches J.2 constants,
            units are not 'K', missing values occur in daily series, or
            values violate E.6.1 plausibility limits.
        PermissionError: If validation years are requested without authorization.
        FileNotFoundError: If era5_dir does not exist.
        TypeError: If keyword-only arguments are missing.
    """
    if first_year > last_year:
        raise ValueError(
            f"Invalid year range: first_year ({first_year}) > last_year ({last_year})."
        )

    _enforce_opening_guard(
        station_id=station_id,
        last_year=last_year,
        calibration_last_year=calibration_last_year,
        opening_records=opening_records,
        opening_purpose=opening_purpose,
    )

    clean_id = _normalize_station_id(station_id)

    if clean_id not in BENCHMARK_STATION_CONSTANTS:
        raise ValueError(f"Station '{clean_id}' is not in BENCHMARK_STATION_CONSTANTS.")
    constants = BENCHMARK_STATION_CONSTANTS[clean_id]

    needed_years = list(range(first_year, last_year + 2))
    files = list_era5_files(era5_dir, needed_years)
    files_used = tuple(sorted(f.name for f in files))

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*Dask is not installed.*", category=UserWarning
        )
        adapter = ERA5Adapter(files)

    lat, lon = constants.latitude, constants.longitude
    var_name = adapter._resolve_variable_name("t2m")
    ds_pt = adapter.extract_point(lat, lon, method="nearest")

    grid_lat = float(ds_pt["latitude"].values)
    grid_lon = float(ds_pt["longitude"].values)
    exp_lat = constants.cell_lat
    exp_lon = constants.cell_lon
    if not (
        np.isclose(grid_lat, exp_lat, atol=1e-3)
        and np.isclose(grid_lon, exp_lon, atol=1e-3)
    ):
        raise ValueError(
            f"ERA5 grid cell ({grid_lat:.2f}, {grid_lon:.2f}) does not match "
            f"registered cell ({exp_lat:.2f}, {exp_lon:.2f}) "
            f"for station '{clean_id}'."
        )

    da = ds_pt[var_name]
    if "step" in da.dims:
        da = da.squeeze("step")
    if "surface" in da.dims:
        da = da.squeeze("surface")
    if "number" in da.dims:
        da = da.isel(number=0)

    units = str(da.attrs.get("units", "")).strip()
    if units != "K":
        raise ValueError(
            f"ERA5 variable '{var_name}' must have units 'K' (Kelvin), found '{units}'."
        )

    s_hourly_raw = da.to_series()
    s_hourly_raw.index = pd.DatetimeIndex(s_hourly_raw.index)

    # Stricter than Phase 3: duplicated hourly timestamps are never silently kept.
    n_dup = int(s_hourly_raw.index.duplicated().sum())
    if n_dup > 0:
        raise ValueError(
            f"Station '{clean_id}': ERA5 hourly series contains {n_dup} "
            "duplicated timestamp(s); check for overlapping ERA5 files."
        )

    s_hourly_c = s_hourly_raw - 273.15

    df_daily_mod = aggregate_hourly_to_daily_temperature(
        s_hourly_c,
        window_start_utc=window_start_utc,
        min_hours=min_hours,
        lag_days=lag_days,
        convert_to_celsius=False,
    )

    mod_idx = cast(pd.DatetimeIndex, df_daily_mod.index)
    target_mask = (mod_idx.year >= first_year) & (mod_idx.year <= last_year)
    full_target_index = pd.date_range(
        f"{first_year}-01-01", f"{last_year}-12-31", freq="D"
    )
    df_daily_mod_sliced = df_daily_mod.loc[target_mask].reindex(full_target_index)

    mod_tmax = df_daily_mod_sliced["tmax"]
    mod_tmin = df_daily_mod_sliced["tmin"]

    n_nan = int(mod_tmax.isna().sum() + mod_tmin.isna().sum())
    if n_nan > 0:
        raise ValueError(
            f"Station '{clean_id}': found {n_nan} missing value(s) "
            "in ERA5 daily series."
        )

    tmax_implausible = int(
        ((mod_tmax < MIN_PLAUSIBLE_TEMP_C) | (mod_tmax > MAX_PLAUSIBLE_TEMP_C)).sum()
    )
    tmin_implausible = int(
        ((mod_tmin < MIN_PLAUSIBLE_TEMP_C) | (mod_tmin > MAX_PLAUSIBLE_TEMP_C)).sum()
    )
    n_implausible = tmax_implausible + tmin_implausible
    if n_implausible > 0:
        raise ValueError(
            f"Station '{clean_id}': found {n_implausible} value(s) "
            "outside the E.6.1 plausibility limits in the requested years."
        )

    mod_tmax.name = "mod_tmax"
    mod_tmin.name = "mod_tmin"

    return ERA5DailyRecord(
        station_id=clean_id,
        tmax=mod_tmax,
        tmin=mod_tmin,
        files_used=files_used,
    )
