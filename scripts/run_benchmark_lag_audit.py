#!/usr/bin/env python3
"""Primary benchmark observational lag audit for daily temperature (Protocol Appendix J.5 / Section 8).

Implements the pre-validation observational lag audit for the two primary benchmark
stations defined in Protocol Section 2.1 and Appendix J.2:
- 21205710 (Jardín Botánico)
- 23215030 (Aguas Claras)

Protocol Specifications:
- Appendix J.2: Uses fixed audited coordinates and elevations; validates cell and
  z_era5 against ERA5 static NetCDF (aborts on mismatch or |diff| > 0.1 m).
- Appendix J.5 / Section 8.2: Evaluates cross-correlation curves, lag correlations
  (lags -2, -1, 0, +1, +2 days), extrema timing, and paired 30-day block bootstrap
  of Delta r(k) = r(k) - r(0) for k in {-1, +1} (1,000 replicas, seed 42).
- Strict Metric Boundary (Section 8.2): ZERO bias, MAE, mean, SD, or dispersion
  statistic anywhere in outputs or logs.
- Appendix J.5 item 4: Validation access is a pre-declared access with purpose="lag_audit"
  registered in the ledger before any validation observation is read, requiring explicit
  CLI authorization flag --confirm-written-authorization.

Modes:
- --mode calibration : Runs strictly on calibration partitions; local/era5_hourly/<id>/;
  verifies exact equivalence of observed series against load_calibration_temperature_pair;
  no ledger interaction.
- --mode validation  : Runs on validation partitions; local/era5_ts/<id>/; runs preflight
  gates for all requested stations before first ledger access; requires
  --confirm-written-authorization and verifies git cleanliness before recording validation
  opening in ledger.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
import pandas as pd
import xarray as xr

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from scripts.audit_andean_calibration_lag import (  # noqa: E402
    AUDIT_LAGS,
    CANDIDATE_LAGS,
    DEFAULT_BLOCK_SIZE,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_REPLICAS,
    audit_variable_lag_calibration,
    compute_file_sha256,
    compute_safe_pearson,
    write_report_json,
)
from tropicor.core.protocol import (  # noqa: E402
    DEFAULT_LEDGER_PATH,
    get_current_git_commit,
    validate_validation_access,
)
from tropicor.downscale.temperature import (  # noqa: E402
    load_calibration_temperature_pair,
)
from tropicor.io.era5 import (  # noqa: E402
    ERA5Adapter,
    aggregate_hourly_to_daily_temperature,
)

# Appendix J.2 Station Constants
BENCHMARK_STATIONS: Dict[str, Dict[str, Any]] = {
    "21205710": {
        "name": "Jardín Botánico",
        "latitude": 4.667867,
        "longitude": -74.101034,
        "elevation_m": 2552.0,
        "expected_cell_lat": 4.75,
        "expected_cell_lon": -74.00,
        "expected_z_era5_m": 2753.4,
        "expected_delta_z_m": -201.4,
        "cal_years": (1990, 2004),
        "val_years": (2005, 2020),
        "obs_tmax_csv": REPO_ROOT
        / "local"
        / "newFiles"
        / "TempMaxima_JardinBotanico.csv",
        "obs_tmin_csv": REPO_ROOT
        / "local"
        / "newFiles"
        / "TempMinima_JardinBotanico.csv",
        "era5_hourly_dir": REPO_ROOT / "local" / "era5_hourly" / "21205710",
        "era5_ts_dir": REPO_ROOT / "local" / "era5_ts" / "21205710",
        "static_file": REPO_ROOT
        / "local"
        / "era5_hourly"
        / "21205710"
        / "era5_static.nc",
    },
    "23215030": {
        "name": "Aguas Claras",
        "latitude": 8.228889,
        "longitude": -73.602778,
        "elevation_m": 133.8,
        "expected_cell_lat": 8.25,
        "expected_cell_lon": -73.50,
        "expected_z_era5_m": 762.4,
        "expected_delta_z_m": -628.6,
        "cal_years": (1990, 2000),
        "val_years": (2001, 2011),
        "obs_tmax_csv": REPO_ROOT / "local" / "newFiles" / "TempMaxima_AguasClaras.csv",
        "obs_tmin_csv": REPO_ROOT / "local" / "newFiles" / "TempMinima_AguasClaras.csv",
        "era5_hourly_dir": REPO_ROOT / "local" / "era5_hourly" / "23215030",
        "era5_ts_dir": REPO_ROOT / "local" / "era5_ts" / "23215030",
        "static_file": REPO_ROOT
        / "local"
        / "era5_hourly"
        / "23215030"
        / "era5_static.nc",
    },
}

DEFAULT_OUTPUT_DIR = REPO_ROOT / "local" / "reports"


def verify_station_static_parameters(
    station_id: str,
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    """Verify ERA5 static geopotential and cell coordinates against Protocol Appendix J.2.

    Protocol Specifications:
    - Extracts static point using nearest-neighbor on ERA5 static file.
    - Resolves surface geopotential variable 'z' and converts to height: z_m = z / 9.80665.
    - Validates grid cell latitude and longitude match expected Table J.2 coordinates.
    - Validates z_era5_m equals Table J.2 audited value within 0.1 m tolerance.
    - Computes static file cryptographic hash (SHA-256).
    """
    static_file: Path = meta["static_file"]
    if not static_file.exists():
        raise FileNotFoundError(
            f"Static NetCDF file not found for station {station_id}: {static_file}"
        )

    static_sha = compute_file_sha256(static_file)

    with xr.open_dataset(static_file) as ds_static:
        lat_target = meta["latitude"]
        lon_target = meta["longitude"]

        if "latitude" in ds_static.coords and "longitude" in ds_static.coords:
            pt = ds_static.sel(
                latitude=lat_target, longitude=lon_target, method="nearest"
            )
            cell_lat = float(pt["latitude"].values)
            cell_lon = float(pt["longitude"].values)
        elif "lat" in ds_static.coords and "lon" in ds_static.coords:
            pt = ds_static.sel(lat=lat_target, lon=lon_target, method="nearest")
            cell_lat = float(pt["lat"].values)
            cell_lon = float(pt["lon"].values)
        else:
            raise ValueError(
                f"Static dataset coordinates not recognized in {static_file}: {list(ds_static.coords.keys())}"
            )

        if not (
            np.isclose(cell_lat, meta["expected_cell_lat"], atol=1e-3)
            and np.isclose(cell_lon, meta["expected_cell_lon"], atol=1e-3)
        ):
            raise ValueError(
                f"Extracted cell ({cell_lat:.2f}, {cell_lon:.2f}) does not match Table J.2 expected cell "
                f"({meta['expected_cell_lat']:.2f}, {meta['expected_cell_lon']:.2f}) for station {station_id}."
            )

        z_var_name = None
        for cand in ["z", "Z", "geopotential"]:
            if cand in pt.data_vars:
                z_var_name = cand
                break
        if z_var_name is None:
            raise ValueError(
                f"Geopotential variable 'z' not found in static file {static_file}. Variables: {list(pt.data_vars.keys())}"
            )

        z_val = float(np.squeeze(pt[z_var_name].values))
        g0 = 9.80665
        z_era5_m = z_val / g0
        delta_z_m = meta["elevation_m"] - z_era5_m

        diff_z = abs(z_era5_m - meta["expected_z_era5_m"])
        if diff_z > 0.1:
            raise ValueError(
                f"Computed z_era5 ({z_era5_m:.2f} m) differs from Table J.2 value "
                f"({meta['expected_z_era5_m']:.2f} m) by {diff_z:.2f} m > 0.1 m for station {station_id}."
            )

    return {
        "cell_latitude": cell_lat,
        "cell_longitude": cell_lon,
        "delta_z_m": float(delta_z_m),
        "elevation_m": float(meta["elevation_m"]),
        "expected_cell_lat": float(meta["expected_cell_lat"]),
        "expected_cell_lon": float(meta["expected_cell_lon"]),
        "expected_delta_z_m": float(meta["expected_delta_z_m"]),
        "expected_z_era5_m": float(meta["expected_z_era5_m"]),
        "static_file": (
            static_file.relative_to(REPO_ROOT).as_posix()
            if static_file.is_relative_to(REPO_ROOT)
            else static_file.as_posix()
        ),
        "static_sha256": static_sha,
        "station_id": station_id,
        "station_name": meta["name"],
        "z_era5_m": float(z_era5_m),
    }


def check_era5_completeness(
    era5_dir: Path,
    station_id: str,
    required_years: Sequence[int],
) -> Tuple[List[Path], Dict[str, str]]:
    """Verify that all required ERA5 NetCDF files exist and return paths and hashes.

    Aborts with FileNotFoundError listing every missing year if incomplete.
    Never proceeds with a partial set.
    """
    if not era5_dir.is_dir():
        raise FileNotFoundError(f"ERA5 directory not found: {era5_dir}")

    missing_years: List[int] = []
    found_files: List[Path] = []
    for y in required_years:
        pattern = f"*2m_temperature*{y}*.nc"
        matches = [
            f for f in era5_dir.glob(pattern) if re.search(rf"(?<!\d){y}(?!\d)", f.name)
        ]
        if not matches:
            missing_years.append(y)
        elif len(matches) > 1:
            raise ValueError(
                f"Multiple ERA5 files found for year {y} in {era5_dir}: {[f.name for f in matches]}"
            )
        else:
            found_files.append(matches[0])

    if missing_years:
        raise FileNotFoundError(
            f"Missing required ERA5 file(s) for station {station_id} in {era5_dir}. "
            f"Missing years: {sorted(missing_years)}. Required years: {list(required_years)}."
        )

    found_files.sort(key=lambda p: p.name)
    hashes: Dict[str, str] = {f.name: compute_file_sha256(f) for f in found_files}
    return found_files, hashes


def verify_era5_source_equivalence(
    station_id: str,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Verify hourly t2m center-cell equivalence between local/era5_hourly/ and local/era5_ts/.

    Evaluates strictly over the station's CALIBRATION years (model data only, no observations).
    Extracts center-cell hourly t2m from both sources and aborts if max |diff| > 1e-3 K
    or if timestamps differ.
    """
    clean_id = re.sub(r"\.0$", "", str(station_id).strip())
    if meta is None:
        if clean_id not in BENCHMARK_STATIONS:
            raise ValueError(f"Unknown benchmark station '{clean_id}'.")
        meta = BENCHMARK_STATIONS[clean_id]

    cal_start, cal_end = meta["cal_years"]
    cal_years = list(range(cal_start, cal_end + 1))

    dir_hourly = meta["era5_hourly_dir"]
    dir_ts = meta["era5_ts_dir"]

    files_h, hashes_h = check_era5_completeness(dir_hourly, clean_id, cal_years)
    files_ts, hashes_ts = check_era5_completeness(dir_ts, clean_id, cal_years)

    lat_target = meta["latitude"]
    lon_target = meta["longitude"]

    # Extract from era5_hourly
    adapter_h = ERA5Adapter(files_h)
    ds_h = adapter_h.extract_point(lat_target, lon_target, method="nearest")
    da_h = ds_h[adapter_h._resolve_variable_name("t2m")]
    for d in ["step", "surface", "number"]:
        if d in da_h.dims:
            da_h = da_h.squeeze(d) if d != "number" else da_h.isel(number=0)
    s_h = da_h.to_series()
    s_h.index = pd.DatetimeIndex(s_h.index)

    # Extract from era5_ts
    adapter_ts = ERA5Adapter(files_ts)
    ds_ts = adapter_ts.extract_point(lat_target, lon_target, method="nearest")
    da_ts = ds_ts[adapter_ts._resolve_variable_name("t2m")]
    for d in ["step", "surface", "number"]:
        if d in da_ts.dims:
            da_ts = da_ts.squeeze(d) if d != "number" else da_ts.isel(number=0)
    s_ts = da_ts.to_series()
    s_ts.index = pd.DatetimeIndex(s_ts.index)

    # Verify timestamps match identically
    if not s_h.index.equals(s_ts.index):
        raise ValueError(
            f"Timestamps differ between era5_hourly and era5_ts for station {clean_id} "
            f"over calibration years {cal_start}-{cal_end}: "
            f"hourly length={len(s_h)}, ts length={len(s_ts)}."
        )

    # Abort if either series contains NaN
    if s_h.isna().any() or s_ts.isna().any():
        n_h_nan = int(s_h.isna().sum())
        n_ts_nan = int(s_ts.isna().sum())
        raise ValueError(
            f"NaN values detected in ERA5 series during source equivalence check for "
            f"station {clean_id}: era5_hourly NaNs={n_h_nan}, era5_ts NaNs={n_ts_nan}."
        )

    # Verify max absolute difference <= 1e-3 K
    abs_diff = np.abs(s_h.values - s_ts.values)
    max_diff_k = float(np.max(abs_diff))

    if not np.isfinite(max_diff_k):
        raise ValueError(
            f"Non-finite max absolute difference ({max_diff_k}) computed during ERA5 "
            f"source equivalence check for station {clean_id}."
        )

    if max_diff_k > 1e-3:
        raise ValueError(
            f"ERA5 source equivalence failed for station {clean_id}: "
            f"max |difference| = {max_diff_k:.6e} K (> 1e-3 K)."
        )

    return {
        "calibration_years": [cal_start, cal_end],
        "equivalent": True,
        "era5_hourly_hashes": {f"{clean_id}/{k}": v for k, v in hashes_h.items()},
        "era5_ts_hashes": {f"{clean_id}/{k}": v for k, v in hashes_ts.items()},
        "max_abs_diff_k": max_diff_k,
        "n_hours": int(len(s_h)),
        "station_id": clean_id,
    }


def read_and_slice_benchmark_observations(
    obs_path: Path,
    station_id: str,
    start_year: int,
    end_year: int,
    var_name: str,
) -> pd.Series:
    """Read DHIME daily observations and immediately slice to the target partition span.

    Enforces:
    - Required columns 'CodigoEstacion', 'Fecha', 'Valor'.
    - Strict ISO8601 parseable dates.
    - Duplicate date validation: rejects distinct values and duplicate groups mixing
      missing and non-missing raw entries; preserves official deduplication order.
    - Non-empty numeric values.
    - Immediate partition year trimming before any metric calculation.
    - Physical range check [-60, 60] °C.
    """
    if not obs_path.exists():
        raise FileNotFoundError(f"Observation file not found: {obs_path}")

    df_raw = pd.read_csv(obs_path)
    clean_id = re.sub(r"\.0$", "", str(station_id).strip())

    if "CodigoEstacion" not in df_raw.columns:
        raise ValueError(f"Required column 'CodigoEstacion' not found in '{obs_path}'.")
    if "Fecha" not in df_raw.columns or "Valor" not in df_raw.columns:
        raise ValueError(
            f"Required columns 'Fecha' and 'Valor' not found in '{obs_path}'."
        )

    st_codes = (
        df_raw["CodigoEstacion"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )
    if clean_id not in st_codes.values:
        raise ValueError(
            f"Station ID '{clean_id}' not found in observation file '{obs_path}'."
        )

    df_st = df_raw[st_codes == clean_id].copy()

    # Parse and validate dates
    parsed_dt = pd.to_datetime(df_st["Fecha"], format="ISO8601", errors="coerce")
    if parsed_dt.isna().any():
        n_unp = int(parsed_dt.isna().sum())
        raise ValueError(f"Found {n_unp} unparseable date(s) in '{obs_path}'.")
    df_st["dt_norm"] = parsed_dt.dt.normalize()

    # Duplicate date handling
    dup_mask = df_st["dt_norm"].duplicated(keep=False)
    if dup_mask.any():
        for _, grp in df_st[dup_mask].groupby("dt_norm"):
            distinct_vals = grp["Valor"].dropna().unique()
            if len(distinct_vals) > 1:
                raise ValueError(
                    f"Duplicate date with distinct values found in '{obs_path}' for station {clean_id}."
                )

        mixed_dup_dates: List[str] = []
        for dt_val, grp in df_st[dup_mask].groupby("dt_norm"):
            val_missing = grp["Valor"].isna() | grp["Valor"].astype(
                str
            ).str.strip().isin(["", "nan", "NaN", "None", "NULL"])
            if val_missing.any() and (~val_missing).any():
                dt_str = (
                    dt_val.strftime("%Y-%m-%d")
                    if hasattr(dt_val, "strftime")
                    else str(dt_val)[:10]
                )
                mixed_dup_dates.append(dt_str)

        if mixed_dup_dates:
            sample_dates = mixed_dup_dates[:5]
            raise ValueError(
                f"Duplicate date group(s) mixing missing and non-missing raw Valor entries "
                f"found in '{obs_path}' for station {clean_id}: "
                f"{len(mixed_dup_dates)} group(s) affected (dates: {sample_dates})."
            )

    df_st = (
        df_st.dropna(subset=["dt_norm"])
        .sort_values("dt_norm")
        .drop_duplicates(subset=["dt_norm"], keep="first")
    )

    clean_num = pd.to_numeric(
        df_st["Valor"].astype(str).str.strip().str.replace(",", "."),
        errors="coerce",
    )
    s_full = pd.Series(clean_num.values, index=pd.DatetimeIndex(df_st["dt_norm"]))

    # Slice strictly to partition years
    s_idx = cast(pd.DatetimeIndex, s_full.index)
    part_mask = (s_idx.year >= start_year) & (s_idx.year <= end_year)
    s_part = s_full[part_mask]

    # Verify physical validity [-60, 60] °C
    valid_vals = s_part.dropna()
    if not valid_vals.empty:
        min_v = float(valid_vals.min())
        max_v = float(valid_vals.max())
        if min_v < -60.0 or max_v > 60.0:
            raise ValueError(
                f"Physically implausible temperature in '{obs_path}' for {var_name}: "
                f"min={min_v:.2f} °C, max={max_v:.2f} °C outside [-60, 60] °C."
            )

    full_part_index = pd.date_range(
        f"{start_year}-01-01", f"{end_year}-12-31", freq="D"
    )
    s_res = s_part.reindex(full_part_index)
    s_res.name = var_name
    return s_res


def inspect_observation_file_quality(
    obs_path: Path,
    station_id: str,
    start_year: int,
    end_year: int,
) -> Dict[str, int]:
    """Inspect observation CSV for duplicate dates and non-numeric entries within window."""
    if not obs_path.exists():
        raise FileNotFoundError(f"Observation file not found: {obs_path}")

    df_raw = pd.read_csv(obs_path)
    clean_id = re.sub(r"\.0$", "", str(station_id).strip())

    if "CodigoEstacion" not in df_raw.columns:
        raise ValueError(f"Required column 'CodigoEstacion' not found in '{obs_path}'.")
    if "Fecha" not in df_raw.columns or "Valor" not in df_raw.columns:
        raise ValueError(
            f"Required columns 'Fecha' and 'Valor' not found in '{obs_path}'."
        )

    st_codes = (
        df_raw["CodigoEstacion"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )
    if clean_id not in st_codes.values:
        raise ValueError(
            f"Station ID '{clean_id}' not found in observation file '{obs_path}'."
        )

    df_st = df_raw[st_codes == clean_id].copy()
    parsed_dt = pd.to_datetime(df_st["Fecha"], format="ISO8601", errors="coerce")
    df_st["dt_norm"] = parsed_dt.dt.normalize()

    # Restrict strictly to window [start_year, end_year]
    window_mask = (df_st["dt_norm"].dt.year >= start_year) & (
        df_st["dt_norm"].dt.year <= end_year
    )
    df_win = df_st[window_mask]

    duplicate_dates_count = int(df_win["dt_norm"].duplicated(keep="first").sum())

    raw_val_str = df_win["Valor"].astype(str).str.strip().str.replace(",", ".")
    num_vals = pd.to_numeric(raw_val_str, errors="coerce")
    non_numeric_mask = (
        num_vals.isna()
        & df_win["Valor"].notna()
        & ~raw_val_str.str.upper().isin(["", "NAN", "NONE", "NULL"])
    )
    non_numeric_count = int(non_numeric_mask.sum())

    return {
        "duplicate_dates_count": duplicate_dates_count,
        "non_numeric_valor_count": non_numeric_count,
    }


def extract_and_aggregate_benchmark_era5(
    era5_files: List[Path],
    meta: Dict[str, Any],
    start_year: int,
    end_year: int,
    window_start_utc: int = 5,
    min_hours: int = 24,
    audit_lags: Sequence[int] = AUDIT_LAGS,
) -> Tuple[Dict[int, pd.Series], Dict[int, pd.Series]]:
    """Extract hourly ERA5 at station cell and aggregate daily series across all audit lags.

    For each lag k in audit_lags, aggregation uses the loader's lag_days parameter:
        aggregate_hourly_to_daily_temperature(..., lag_days=k)
    """
    adapter = ERA5Adapter(era5_files)
    lat_target = meta["latitude"]
    lon_target = meta["longitude"]
    ds_pt = adapter.extract_point(lat_target, lon_target, method="nearest")

    var_name = adapter._resolve_variable_name("t2m")
    da = ds_pt[var_name]
    for dim_name in ["step", "surface", "number"]:
        if dim_name in da.dims:
            da = da.squeeze(dim_name) if dim_name != "number" else da.isel(number=0)

    units = str(da.attrs.get("units", "")).strip()
    if units != "K":
        raise ValueError(
            f"ERA5 variable '{var_name}' must have units 'K', found '{units}'."
        )

    # Explicit Kelvin to Celsius conversion
    s_hourly_c = da.to_series() - 273.15
    s_hourly_c.index = pd.DatetimeIndex(s_hourly_c.index)

    part_index = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="D")
    mod_tx_by_lag: Dict[int, pd.Series] = {}
    mod_tn_by_lag: Dict[int, pd.Series] = {}

    for k in audit_lags:
        df_daily_k = aggregate_hourly_to_daily_temperature(
            s_hourly_c,
            window_start_utc=window_start_utc,
            min_hours=min_hours,
            lag_days=k,
            convert_to_celsius=False,
        )
        mod_tx_by_lag[k] = df_daily_k["tmax"].reindex(part_index)
        mod_tn_by_lag[k] = df_daily_k["tmin"].reindex(part_index)

    return mod_tx_by_lag, mod_tn_by_lag


def preflight_validation_checks(
    station_id: str,
    meta: Optional[Dict[str, Any]] = None,
    confirm_written_authorization: bool = False,
) -> Dict[str, Any]:
    """Execute all pre-validation verification gates for a benchmark station.

    Strictly executes in order:
    1. CLI authorization flag check (--confirm-written-authorization).
    2. Git-clean guard covering scripts/run_benchmark_lag_audit.py,
       scripts/audit_andean_calibration_lag.py, and tropicor/.
    3. Static coordinate and elevation verification against Protocol Appendix J.2.
    4. ERA5 completeness and hashing over (start_year - 1 .. end_year + 1) in local/era5_ts/<id>/.
    5. ERA5 source equivalence between hourly and ts sources over calibration years.
    6. Observation CSV files existence and SHA-256 computation.

    Returns preflight context dictionary with verified file paths, hashes, and metadata.
    Aborts on any failure without touching ledger or reading observation values.
    """
    clean_id = re.sub(r"\.0$", "", str(station_id).strip())
    if meta is None:
        if clean_id not in BENCHMARK_STATIONS:
            raise ValueError(f"Unsupported station ID '{clean_id}'.")
        meta = BENCHMARK_STATIONS[clean_id]

    # 1. Authorization flag check
    if not confirm_written_authorization:
        raise PermissionError(
            "Validation mode execution blocked: requires explicit CLI flag "
            "--confirm-written-authorization."
        )

    # 2. Git-clean guard covering scripts/run_benchmark_lag_audit.py,
    #    scripts/audit_andean_calibration_lag.py, and tropicor/
    git_check = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--",
            "scripts/run_benchmark_lag_audit.py",
            "scripts/audit_andean_calibration_lag.py",
            "tropicor/",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if git_check.stdout.strip():
        raise RuntimeError(
            "Uncommitted or untracked changes detected in scripts/run_benchmark_lag_audit.py, "
            "scripts/audit_andean_calibration_lag.py, or tropicor/:\n"
            f"{git_check.stdout}"
        )

    # 3. Static parameter verification
    static_info = verify_station_static_parameters(clean_id, meta)

    # 4. ERA5 completeness and hashing (validation uses local/era5_ts/<id>/)
    start_year, end_year = meta["val_years"]
    required_era5_years = list(range(start_year - 1, end_year + 2))
    era5_dir = meta["era5_ts_dir"]
    era5_files, era5_hashes = check_era5_completeness(
        era5_dir, clean_id, required_era5_years
    )

    # 5. Source equivalence between era5_hourly and era5_ts over calibration
    source_equiv = verify_era5_source_equivalence(clean_id, meta)

    # 6. Observation files existence and hashing
    if not meta["obs_tmax_csv"].exists():
        raise FileNotFoundError(
            f"Tmax observation file not found: {meta['obs_tmax_csv']}"
        )
    if not meta["obs_tmin_csv"].exists():
        raise FileNotFoundError(
            f"Tmin observation file not found: {meta['obs_tmin_csv']}"
        )

    obs_tmax_hash = compute_file_sha256(meta["obs_tmax_csv"])
    obs_tmin_hash = compute_file_sha256(meta["obs_tmin_csv"])

    return {
        "era5_dir": era5_dir,
        "era5_files": era5_files,
        "era5_hashes": era5_hashes,
        "obs_tmax_hash": obs_tmax_hash,
        "obs_tmin_hash": obs_tmin_hash,
        "required_era5_years": required_era5_years,
        "source_equiv": source_equiv,
        "start_year": start_year,
        "end_year": end_year,
        "static_info": static_info,
        "station_id": clean_id,
    }


def audit_benchmark_station(
    station_id: str,
    mode: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    block_size: int = DEFAULT_BLOCK_SIZE,
    replicas: int = DEFAULT_REPLICAS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    window_start_utc: int = 5,
    min_hours: int = 24,
    confirm_written_authorization: bool = False,
    ledger_path: Path = DEFAULT_LEDGER_PATH,
    preflight_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute lag audit for a single benchmark station under specified mode."""
    clean_id = re.sub(r"\.0$", "", str(station_id).strip())
    if clean_id not in BENCHMARK_STATIONS:
        raise ValueError(
            f"Unsupported station ID '{clean_id}'. Supported stations: {list(BENCHMARK_STATIONS.keys())}."
        )

    meta = BENCHMARK_STATIONS[clean_id]

    if mode == "validation":
        # Step 3: Use preflight_context if provided, else execute preflight checks
        if preflight_context is None:
            preflight_context = preflight_validation_checks(
                clean_id,
                meta,
                confirm_written_authorization=confirm_written_authorization,
            )

        static_info = preflight_context["static_info"]
        start_year = preflight_context["start_year"]
        end_year = preflight_context["end_year"]
        required_era5_years = preflight_context["required_era5_years"]
        era5_dir = preflight_context["era5_dir"]
        era5_files = preflight_context["era5_files"]
        era5_hashes = preflight_context["era5_hashes"]
        source_equiv = preflight_context["source_equiv"]
        obs_tmax_hash = preflight_context["obs_tmax_hash"]
        obs_tmin_hash = preflight_context["obs_tmin_hash"]

        # Resolve ledger_path against REPO_ROOT when it is not absolute
        effective_ledger = (
            ledger_path if ledger_path.is_absolute() else (REPO_ROOT / ledger_path)
        )

        # 7. ONLY THEN: call validate_validation_access before reading observations
        opening_record = validate_validation_access(
            open_validation=True,
            purpose="lag_audit",
            station_id=clean_id,
            variable="temperature",
            seed=seed,
            config={
                "block_size_days": block_size,
                "bootstrap_replicas": replicas,
                "bootstrap_seed": seed,
                "min_hours": min_hours,
                "mode": "validation",
                "protocol_section": "Appendix J.5 / Section 8",
                "window_start_utc": window_start_utc,
            },
            ledger_path=effective_ledger,
        )

        # 8. Read validation observations
        obs_tx = read_and_slice_benchmark_observations(
            meta["obs_tmax_csv"], clean_id, start_year, end_year, "obs_tmax"
        )
        obs_tn = read_and_slice_benchmark_observations(
            meta["obs_tmin_csv"], clean_id, start_year, end_year, "obs_tmin"
        )

    elif mode == "calibration":
        opening_record = None
        # Calibration mode uses local/era5_hourly/<id>/ and calibration span
        start_year, end_year = meta["cal_years"]
        # Required years matches Appendix J.4 reports: cal_start .. cal_end + 1
        required_era5_years = list(range(start_year, end_year + 2))
        era5_dir = meta["era5_hourly_dir"]

        static_info = verify_station_static_parameters(clean_id, meta)
        era5_files, era5_hashes = check_era5_completeness(
            era5_dir, clean_id, required_era5_years
        )
        source_equiv = verify_era5_source_equivalence(clean_id, meta)

        obs_tmax_hash = compute_file_sha256(meta["obs_tmax_csv"])
        obs_tmin_hash = compute_file_sha256(meta["obs_tmin_csv"])

        obs_tx = read_and_slice_benchmark_observations(
            meta["obs_tmax_csv"], clean_id, start_year, end_year, "obs_tmax"
        )
        obs_tn = read_and_slice_benchmark_observations(
            meta["obs_tmin_csv"], clean_id, start_year, end_year, "obs_tmin"
        )

        # Step 4: Verification against official loader load_calibration_temperature_pair
        obs_tx_official, obs_tn_official, _, _ = load_calibration_temperature_pair(
            station_id=clean_id,
            obs_tmax_path=meta["obs_tmax_csv"],
            obs_tmin_path=meta["obs_tmin_csv"],
            era5_target=era5_dir,
            window_start_utc=window_start_utc,
            lag_days=0,
            min_hours=min_hours,
        )

        # Check exact equivalence of Tmax
        if not obs_tx.index.equals(obs_tx_official.index):
            raise ValueError(
                f"Index mismatch between read_and_slice_benchmark_observations and "
                f"load_calibration_temperature_pair for station {clean_id} Tmax."
            )
        if not (obs_tx.isna() == obs_tx_official.isna()).all():
            raise ValueError(
                f"NaN mask mismatch between read_and_slice_benchmark_observations and "
                f"load_calibration_temperature_pair for station {clean_id} Tmax."
            )
        valid_tx = obs_tx.dropna()
        valid_tx_off = obs_tx_official.dropna()
        if not np.array_equal(valid_tx.to_numpy(), valid_tx_off.to_numpy()):
            raise ValueError(
                f"Value mismatch between read_and_slice_benchmark_observations and "
                f"load_calibration_temperature_pair for station {clean_id} Tmax."
            )

        # Check exact equivalence of Tmin
        if not obs_tn.index.equals(obs_tn_official.index):
            raise ValueError(
                f"Index mismatch between read_and_slice_benchmark_observations and "
                f"load_calibration_temperature_pair for station {clean_id} Tmin."
            )
        if not (obs_tn.isna() == obs_tn_official.isna()).all():
            raise ValueError(
                f"NaN mask mismatch between read_and_slice_benchmark_observations and "
                f"load_calibration_temperature_pair for station {clean_id} Tmin."
            )
        valid_tn = obs_tn.dropna()
        valid_tn_off = obs_tn_official.dropna()
        if not np.array_equal(valid_tn.to_numpy(), valid_tn_off.to_numpy()):
            raise ValueError(
                f"Value mismatch between read_and_slice_benchmark_observations and "
                f"load_calibration_temperature_pair for station {clean_id} Tmin."
            )

        # Report quality metrics within calibration window
        quality_tx = inspect_observation_file_quality(
            meta["obs_tmax_csv"], clean_id, start_year, end_year
        )
        quality_tn = inspect_observation_file_quality(
            meta["obs_tmin_csv"], clean_id, start_year, end_year
        )

    else:
        raise ValueError(
            f"Invalid mode '{mode}'. Must be 'calibration' or 'validation'."
        )

    # Extract ERA5 hourly and aggregate daily series across all audit lags
    mod_tx_by_lag, mod_tn_by_lag = extract_and_aggregate_benchmark_era5(
        era5_files=era5_files,
        meta=meta,
        start_year=start_year,
        end_year=end_year,
        window_start_utc=window_start_utc,
        min_hours=min_hours,
        audit_lags=AUDIT_LAGS,
    )

    # Collect cryptographic hashes of all input files
    input_hashes: Dict[str, str] = {
        f"{clean_id}/{meta['obs_tmax_csv'].name}": obs_tmax_hash,
        f"{clean_id}/{meta['obs_tmin_csv'].name}": obs_tmin_hash,
        f"{clean_id}/{meta['static_file'].name}": static_info["static_sha256"],
    }
    for fname, fhash in era5_hashes.items():
        input_hashes[f"{clean_id}/{fname}"] = fhash

    # Execute lag audit for Tmax and Tmin
    res_tx = audit_variable_lag_calibration(
        obs=obs_tx,
        mod_by_lag=mod_tx_by_lag,
        candidate_lags=CANDIDATE_LAGS,
        audit_lags=AUDIT_LAGS,
        block_size=block_size,
        n_replicas=replicas,
        seed=seed,
    )
    res_tn = audit_variable_lag_calibration(
        obs=obs_tn,
        mod_by_lag=mod_tn_by_lag,
        candidate_lags=CANDIDATE_LAGS,
        audit_lags=AUDIT_LAGS,
        block_size=block_size,
        n_replicas=replicas,
        seed=seed,
    )

    # Consistency check for r(0) on paired masked daily series at lag 0
    mask_tx0 = obs_tx.notna() & mod_tx_by_lag[0].notna()
    direct_tx_r0 = float(
        np.corrcoef(obs_tx[mask_tx0].to_numpy(), mod_tx_by_lag[0][mask_tx0].to_numpy())[
            0, 1
        ]
    )
    audit_tx_r0 = compute_safe_pearson(
        obs_tx[mask_tx0].to_numpy(), mod_tx_by_lag[0][mask_tx0].to_numpy()
    )
    diff_tx = abs(direct_tx_r0 - audit_tx_r0)

    mask_tn0 = obs_tn.notna() & mod_tn_by_lag[0].notna()
    direct_tn_r0 = float(
        np.corrcoef(obs_tn[mask_tn0].to_numpy(), mod_tn_by_lag[0][mask_tn0].to_numpy())[
            0, 1
        ]
    )
    audit_tn_r0 = compute_safe_pearson(
        obs_tn[mask_tn0].to_numpy(), mod_tn_by_lag[0][mask_tn0].to_numpy()
    )
    diff_tn = abs(direct_tn_r0 - audit_tn_r0)

    if diff_tx >= 1e-12:
        raise ValueError(
            f"Consistency check failed for Tmax r(0) on station {clean_id}: "
            f"direct={direct_tx_r0}, audit={audit_tx_r0}, diff={diff_tx} >= 1e-12."
        )
    if diff_tn >= 1e-12:
        raise ValueError(
            f"Consistency check failed for Tmin r(0) on station {clean_id}: "
            f"direct={direct_tn_r0}, audit={audit_tn_r0}, diff={diff_tn} >= 1e-12."
        )

    # Activation decision rule (Section 8.3 / Appendix J.5 item 5)
    all_triggered = sorted(
        list(set(res_tx["triggered_lags"] + res_tn["triggered_lags"]))
    )
    lag_sensitivity_activated = bool(len(all_triggered) > 0)

    git_commit = get_current_git_commit()
    is_dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    )

    payload: Dict[str, Any] = {
        "activated_lags": all_triggered,
        "commit_hash": git_commit,
        "config": {
            "audit_lags": AUDIT_LAGS,
            "block_size_days": block_size,
            "bootstrap_replicas": replicas,
            "bootstrap_seed": seed,
            "bootstrap_unit": "calendar_30d_moving_block",
            "candidate_lags": CANDIDATE_LAGS,
            "era5_dir": (
                era5_dir.relative_to(REPO_ROOT).as_posix()
                if era5_dir.is_relative_to(REPO_ROOT)
                else era5_dir.as_posix()
            ),
            "min_hours": min_hours,
            "mode": mode,
            "numpy_version": str(np.__version__),
            "pandas_version": str(pd.__version__),
            "protocol_section": "Appendix J.5 / Section 8",
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "required_era5_years": required_era5_years,
            "window_start_utc": window_start_utc,
        },
        "consistency_check_r0": {
            "tmax": {
                "audit_r0": audit_tx_r0,
                "difference": diff_tx,
                "direct_r0": direct_tx_r0,
                "passed": bool(diff_tx < 1e-12),
            },
            "tmin": {
                "audit_r0": audit_tn_r0,
                "difference": diff_tn,
                "direct_r0": direct_tn_r0,
                "passed": bool(diff_tn < 1e-12),
            },
        },
        "era5_source_equivalence": source_equiv,
        "input_hashes": {k: input_hashes[k] for k in sorted(input_hashes.keys())},
        "is_dirty": is_dirty,
        "lag_sensitivity_activated": lag_sensitivity_activated,
        "partition_years": [start_year, end_year],
        "station_id": clean_id,
        "station_name": meta["name"],
        "static_verification": static_info,
        "tmax": res_tx,
        "tmin": res_tn,
    }

    if mode == "calibration":
        payload["observation_data_quality"] = {
            "tmax": {
                "duplicate_dates_count": quality_tx["duplicate_dates_count"],
                "non_numeric_valor_count": quality_tx["non_numeric_valor_count"],
                "matches_official_loader": True,
            },
            "tmin": {
                "duplicate_dates_count": quality_tn["duplicate_dates_count"],
                "non_numeric_valor_count": quality_tn["non_numeric_valor_count"],
                "matches_official_loader": True,
            },
        }

    if opening_record is not None:
        payload["validation_opening_record"] = opening_record

    # Write deterministic JSON
    out_file = output_dir / f"benchmark_lag_audit_{mode}_{clean_id}.json"
    write_report_json(payload, out_file)

    # Strict logging without any bias, MAE, or dispersion statistics
    print(f"Station {clean_id} ({meta['name']}) [{mode} {start_year}-{end_year}]:")
    print(
        f"  Tmax lag 0 correlation: direct={direct_tx_r0:.10f}, audit={audit_tx_r0:.10f}, diff={diff_tx:.2e}"
    )
    print(
        f"  Tmin lag 0 correlation: direct={direct_tn_r0:.10f}, audit={audit_tn_r0:.10f}, diff={diff_tn:.2e}"
    )
    print(
        f"  Tmax optimal lag: {res_tx['extrema_timing']['optimal_lag']} (max r={res_tx['extrema_timing']['max_r']:.6f})"
    )
    print(
        f"  Tmin optimal lag: {res_tn['extrema_timing']['optimal_lag']} (max r={res_tn['extrema_timing']['max_r']:.6f})"
    )
    for var_label, res_var in [("Tmax", res_tx), ("Tmin", res_tn)]:
        for k_str, cand in res_var["candidate_lags"].items():
            print(
                f"  {var_label} Delta r({k_str}): point={cand['point_delta_r']:+.6f}, "
                f"95% CI=[{cand['ci_95_lower']:+.6f}, {cand['ci_95_upper']:+.6f}], "
                f"excludes_zero_in_favor_of_k={cand['excludes_zero_in_favor_of_k']}"
            )
    print(
        f"  ERA5 source equivalence: max |diff|={source_equiv['max_abs_diff_k']:.6e} K, "
        f"n_hours={source_equiv['n_hours']} (verified <= 1e-3 K)"
    )
    if mode == "calibration":
        print("  Observation validation against load_calibration_temperature_pair:")
        print(
            f"    Tmax: matches official loader=True, duplicate_dates={quality_tx['duplicate_dates_count']}, "
            f"non_numeric_valor={quality_tx['non_numeric_valor_count']}"
        )
        print(
            f"    Tmin: matches official loader=True, duplicate_dates={quality_tn['duplicate_dates_count']}, "
            f"non_numeric_valor={quality_tn['non_numeric_valor_count']}"
        )
    print(
        f"  Sensitivity activated: {lag_sensitivity_activated} (activated lags: {all_triggered})"
    )
    report_disp = (
        out_file.relative_to(REPO_ROOT)
        if out_file.is_relative_to(REPO_ROOT)
        else out_file
    )
    print(f"  Report written to: {report_disp}")

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run primary benchmark observational lag audit (Protocol Appendix J.5 / Section 8)."
    )
    parser.add_argument(
        "--mode",
        choices=["calibration", "validation"],
        default="calibration",
        help="Evaluation mode: 'calibration' (no ledger) or 'validation' (requires authorization).",
    )
    parser.add_argument(
        "--station",
        choices=["21205710", "23215030", "both"],
        default="both",
        help="Target benchmark station ID or 'both'.",
    )
    parser.add_argument(
        "--confirm-written-authorization",
        action="store_true",
        default=False,
        help="Explicit written authorization flag required for --mode validation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write output JSON reports.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        help="Calendar block size in days for moving block bootstrap (default: 30).",
    )
    parser.add_argument(
        "--replicas",
        type=int,
        default=DEFAULT_REPLICAS,
        help="Number of bootstrap replicas (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
        help="Random number generator seed (default: 42).",
    )
    parser.add_argument(
        "--window-start-utc",
        type=int,
        default=5,
        help="Aggregation window start hour in UTC (default: 5).",
    )
    parser.add_argument(
        "--min-hours",
        type=int,
        default=24,
        help="Minimum valid hours per day (default: 24).",
    )
    args = parser.parse_args()

    station_list = (
        ["21205710", "23215030"] if args.station == "both" else [args.station]
    )

    preflight_contexts: Dict[str, Dict[str, Any]] = {}
    if args.mode == "validation":
        # Step 3: Run preflight for ALL requested stations before the first validate_validation_access call
        for st_id in station_list:
            preflight_contexts[st_id] = preflight_validation_checks(
                station_id=st_id,
                meta=BENCHMARK_STATIONS[st_id],
                confirm_written_authorization=args.confirm_written_authorization,
            )

    for st_id in station_list:
        audit_benchmark_station(
            station_id=st_id,
            mode=args.mode,
            output_dir=args.output_dir,
            block_size=args.block_size,
            replicas=args.replicas,
            seed=args.seed,
            window_start_utc=args.window_start_utc,
            min_hours=args.min_hours,
            confirm_written_authorization=args.confirm_written_authorization,
            preflight_context=preflight_contexts.get(st_id),
        )


if __name__ == "__main__":
    main()
