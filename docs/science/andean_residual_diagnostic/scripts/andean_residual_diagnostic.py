"""Diagnostic of the Andean residual on the five ALREADY-OPENED Andean stations.

Evaluates calibration-to-validation level change versus seasonal structure,
with a sampling-noise floor (Appendix K diagnostic).

Stations analyzed:
  - 37015020 (Berlin)
  - 24025040 (Mogotes)
  - 26135040 (Matecaña)
  - 27015070 (Olaya Herrera)
  - 52045010 (Obonuco)

Methodology & Protocol Citations:
  - Observation reading and block slicing: reproduced from tropicor/core/phase3_evaluator.py,
    lines 627-715 (_read_obs_block).
  - ERA5 aggregation to 05-05 UTC daily extremes: reproduced from tropicor/core/phase3_evaluator.py,
    lines 719-825 (_load_era5_daily_block) using ERA5Adapter and aggregate_hourly_to_daily_temperature.
  - Level 1 fitting & monthly biases: reproduced from tropicor/core/phase3_evaluator.py,
    lines 1361-1380 (compute_monthly_biases) and tropicor.downscale.temperature (Level1MonthlyAdditiveTemperature).

RULES:
  - Commit NOTHING.
  - Read NO observations of JB (21205710) or AC (23215030).
  - Do NOT touch local/reports/validation_openings.jsonl.
  - Do NOT import tropicor.core.phase3_evaluator.
  - Arithmetic and raw metrics only; no interpretation.
"""

from __future__ import annotations

import datetime
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from tropicor.downscale.temperature import (
    Level1MonthlyAdditiveTemperature,
    _validate_temperature_range,
    load_calibration_temperature_pair,
)
from tropicor.io.era5 import ERA5Adapter, aggregate_hourly_to_daily_temperature

# Execution parameters
BOOTSTRAP_SEED = 42
BOOTSTRAP_REPLICAS = 1000
NOISE_FLOOR_DRAWS = 999
MIN_DAYS_PER_MONTH = 10
MIN_MONTHS_PER_YEAR = 9
WINDOW_START_UTC = 5
LAG_DAYS = 0
MIN_HOURS = 24

OPENED_STATION_IDS = [
    "37015020",
    "24025040",
    "26135040",
    "27015070",
    "52045010",
]


def compute_sha256(filepath: Path) -> str:
    """Compute lowercase SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()


def get_git_head(repo_root: Path = Path(".")) -> str:
    """Get current git HEAD commit hash."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root.resolve()),
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN"


def read_obs_series(
    path: Path,
    station_id: str,
    start_year: int,
    end_year: int,
    valid_years: Sequence[int],
) -> pd.Series:
    """Read observation extremes from CSV, parse dates, slice block, and mask invalid years.

    Reproduces logic of tropicor/core/phase3_evaluator.py lines 627-715 (_read_obs_block).
    """
    clean_id = re.sub(r"\.0$", "", str(station_id).strip())
    df_raw = pd.read_csv(path)
    st_codes = (
        df_raw["CodigoEstacion"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )
    df_st = df_raw[st_codes == clean_id].copy()

    parsed_dt = pd.to_datetime(df_st["Fecha"], format="ISO8601", errors="coerce")
    df_st["dt_norm"] = parsed_dt.dt.normalize()
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

    full_block_index = pd.date_range(
        f"{start_year}-01-01", f"{end_year}-12-31", freq="D"
    )
    block_mask = (s_full.index.year >= start_year) & (s_full.index.year <= end_year)
    s_block = s_full[block_mask].reindex(full_block_index)

    if valid_years is not None:
        valid_set = set(valid_years)
        invalid_years_mask = ~s_block.index.year.isin(valid_set)
        s_block.loc[invalid_years_mask] = np.nan

    _validate_temperature_range(s_block, f"obs ({clean_id})")
    return s_block


def load_era5_daily_extremes(
    era5_dir: Path,
    lat_cne: float,
    lon_cne: float,
    exp_lat: float,
    exp_lon: float,
    needed_years: Sequence[int],
    start_year: int,
    target_end_year: int,
) -> Tuple[pd.Series, pd.Series]:
    """Load hourly ERA5 NetCDFs, extract cell, aggregate to daily 05-05 UTC, slice block.

    Reproduces logic of tropicor/core/phase3_evaluator.py lines 719-825 (_load_era5_daily_block).
    """
    nc_files = sorted(glob.glob(str(era5_dir / "*2m_temperature*.nc")))
    target_nc_files = [
        f
        for f in nc_files
        if any(re.search(rf"(?<!\d){y}(?!\d)", Path(f).name) for y in needed_years)
    ]
    if len(target_nc_files) < len(needed_years):
        raise FileNotFoundError(
            f"Missing ERA5 NetCDF files for years {needed_years} in {era5_dir}"
        )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        adapter = ERA5Adapter(target_nc_files)

    var_name = adapter._resolve_variable_name("t2m")
    ds_pt = adapter.extract_point(lat_cne, lon_cne, method="nearest")

    # Cell verification
    grid_lat = float(ds_pt["latitude"].values)
    grid_lon = float(ds_pt["longitude"].values)
    if not (
        np.isclose(grid_lat, exp_lat, atol=1e-3)
        and np.isclose(grid_lon, exp_lon, atol=1e-3)
    ):
        raise ValueError(
            f"ERA5 grid cell ({grid_lat:.2f}, {grid_lon:.2f}) does not match "
            f"registered ({exp_lat:.2f}, {exp_lon:.2f})"
        )

    da = ds_pt[var_name]
    if "step" in da.dims:
        da = da.squeeze("step")
    if "surface" in da.dims:
        da = da.squeeze("surface")
    if "number" in da.dims:
        da = da.isel(number=0)

    s_hourly_raw = da.to_series()
    s_hourly_raw.index = pd.DatetimeIndex(s_hourly_raw.index)
    s_hourly_c = s_hourly_raw - 273.15

    df_daily_mod = aggregate_hourly_to_daily_temperature(
        s_hourly_c,
        window_start_utc=WINDOW_START_UTC,
        min_hours=MIN_HOURS,
        lag_days=LAG_DAYS,
        convert_to_celsius=False,
    )

    mod_idx = pd.DatetimeIndex(df_daily_mod.index)
    target_mask = (mod_idx.year >= start_year) & (mod_idx.year <= target_end_year)
    full_target_index = pd.date_range(
        f"{start_year}-01-01", f"{target_end_year}-12-31", freq="D"
    )
    df_daily_mod_sliced = df_daily_mod.loc[target_mask].reindex(full_target_index)

    mod_tmax = df_daily_mod_sliced["tmax"]
    mod_tmin = df_daily_mod_sliced["tmin"]

    if mod_tmax.isna().any() or mod_tmin.isna().any():
        raise ValueError(f"Found NaNs in reanalysis daily series for years {start_year}-{target_end_year}")

    return mod_tmax, mod_tmin


def compute_calendar_monthly_biases(
    obs: pd.Series, mod: pd.Series
) -> Dict[int, Optional[float]]:
    """Compute 12-month calendar mean biases: Bias_m = mean(Mod) - mean(Obs)."""
    valid_mask = obs.notna() & mod.notna()
    c_obs = obs[valid_mask]
    c_mod = mod[valid_mask]
    res: Dict[int, Optional[float]] = {}
    for m in range(1, 13):
        m_mask = c_obs.index.month == m
        if m_mask.any():
            res[m] = float(c_mod[m_mask].mean() - c_obs[m_mask].mean())
        else:
            res[m] = None
    return res


def main() -> None:
    repo_root = Path(".")
    out_dir = Path("local/analysis_k/diagnostic")
    out_dir.mkdir(parents=True, exist_ok=True)

    git_head = get_git_head(repo_root)

    print("=" * 80)
    print("APPENDIX K: DIAGNOSTIC OF THE ANDEAN RESIDUAL (5 OPENED STATIONS)")
    print(f"Git commit HEAD: {git_head}")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # Step 0: Self-Check 2 (Inputs verification against phase3_expected_inputs.json)
    # --------------------------------------------------------------------------
    expected_inputs_path = repo_root / "tropicor" / "data" / "phase3_expected_inputs.json"
    if not expected_inputs_path.is_file():
        print(f"[FATAL ERROR] Missing {expected_inputs_path}", file=sys.stderr)
        sys.exit(1)

    with open(expected_inputs_path, "r", encoding="utf-8") as f:
        exp_inputs = json.load(f)

    with open(repo_root / "tropicor" / "data" / "andean_station_registry.json", "r", encoding="utf-8") as f:
        registry_data = json.load(f)["stations"]

    print("\n" + "=" * 80)
    print("STEP 0: SELF-CHECK 2 (INPUT FILE HASH VERIFICATION)")
    print("=" * 80)

    input_hashes_manifest: Dict[str, Dict[str, Any]] = {}

    def verify_hash(path: Path, expected: str, desc: str) -> bool:
        if not path.is_file():
            print(f"  [MISMATCH] {desc}: File not found at {path}")
            return False
        actual = compute_sha256(path)
        match = (actual == expected.lower())
        input_hashes_manifest[path.as_posix()] = {
            "description": desc,
            "size": path.stat().st_size,
            "sha256": actual,
            "expected_sha256": expected.lower(),
            "match": match,
        }
        status_str = "MATCH" if match else "MISMATCH"
        print(f"  [{status_str}] {desc} ({path.as_posix()}): {actual}")
        if not match:
            print(f"      Expected: {expected.lower()}", file=sys.stderr)
        return match

    all_inputs_pass = True

    # 1. Registry & Lag audit
    all_inputs_pass &= verify_hash(
        repo_root / "tropicor" / "data" / "andean_station_registry.json",
        exp_inputs["registry"],
        "Andean Station Registry",
    )
    all_inputs_pass &= verify_hash(
        repo_root / "docs" / "science" / "appendix_g" / "lag_audit_andean_calibration.json",
        exp_inputs["lag_audit"],
        "Lag Audit Calibration",
    )

    # 2. Calibration reports and station files from scorecards
    station_metadata: Dict[str, Any] = {}
    for st in OPENED_STATION_IDS:
        cal_path = repo_root / exp_inputs["archived_paths"]["calibration"][st]
        all_inputs_pass &= verify_hash(
            cal_path,
            exp_inputs["calibration"][st],
            f"Calibration Report ({st})",
        )

        sc_path = repo_root / "docs" / "science" / "appendix_g" / "phase3_andes" / f"phase3_scorecard_{st}.json"
        with open(sc_path, "r", encoding="utf-8") as f:
            sc = json.load(f)
        ih = sc["input_hashes"]

        st_dir = repo_root / "local" / "era5_ts" / st
        tmax_list = sorted(list(st_dir.glob("TempMaxima_*.csv")))
        tmin_list = sorted(list(st_dir.glob("TempMinima_*.csv")))

        if len(tmax_list) != 1 or len(tmin_list) != 1:
            print(f"  [ERROR] Expected 1 Tmax and 1 Tmin CSV in {st_dir}", file=sys.stderr)
            all_inputs_pass = False
            continue

        tmax_p = tmax_list[0]
        tmin_p = tmin_list[0]

        all_inputs_pass &= verify_hash(tmax_p, ih["obs_tmax_csv"], f"Tmax Obs CSV ({st})")
        all_inputs_pass &= verify_hash(tmin_p, ih["obs_tmin_csv"], f"Tmin Obs CSV ({st})")

        # ERA5 files
        for fn, exp_h in ih["era5_calibration_files"].items():
            all_inputs_pass &= verify_hash(st_dir / fn, exp_h, f"ERA5 Cal ({st}/{fn})")
        for fn, exp_h in ih["era5_validation_files"].items():
            all_inputs_pass &= verify_hash(st_dir / fn, exp_h, f"ERA5 Val ({st}/{fn})")

        station_metadata[st] = {
            "scorecard": sc,
            "st_dir": st_dir,
            "tmax_csv": tmax_p,
            "tmin_csv": tmin_p,
            "reg": registry_data[st],
        }

    if not all_inputs_pass:
        print("\n[CRITICAL ERROR] Self-Check 2 FAILED: Input hash mismatch detected. Aborting.", file=sys.stderr)
        sys.exit(1)
    print("\nSelf-Check 2: ALL INPUT FILE HASHES MATCH (PASS).")

    # --------------------------------------------------------------------------
    # Step 1: Self-Check 1 (Mandatory Reproduction of Level 1 Monthly Biases)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 1: SELF-CHECK 1 (REPRODUCTION OF LEVEL 1 MONTHLY VALIDATION BIASES)")
    print("=" * 80)

    self_check_1_results: Dict[str, Any] = {}
    reproduction_all_pass = True

    station_loaded_data: Dict[str, Any] = {}

    for st in OPENED_STATION_IDS:
        meta = station_metadata[st]
        st_dir = meta["st_dir"]
        reg_info = meta["reg"]
        cal_start, cal_end = reg_info["cal_block"]
        val_start, val_end = reg_info["val_block"]
        valid_years = reg_info["valid_years"]
        cal_years = [y for y in valid_years if cal_start <= y <= cal_end]
        val_years = [y for y in valid_years if val_start <= y <= val_end]
        min_yr = min(valid_years)
        max_yr = max(valid_years)

        # 1. Load Calibration Pair via official temperature loader
        obs_tmax_cal, obs_tmin_cal, mod_tmax_cal, mod_tmin_cal = load_calibration_temperature_pair(
            station_id=st,
            obs_tmax_path=meta["tmax_csv"],
            obs_tmin_path=meta["tmin_csv"],
            era5_target=st_dir,
            window_start_utc=WINDOW_START_UTC,
            lag_days=LAG_DAYS,
            min_hours=MIN_HOURS,
        )

        m1 = Level1MonthlyAdditiveTemperature()
        m1.fit(obs_tmax_cal, obs_tmin_cal, mod_tmax_cal, mod_tmin_cal)

        # 2. Load Validation observations and ERA5 daily series
        obs_tmax_val = read_obs_series(meta["tmax_csv"], st, val_start, val_end, valid_years)
        obs_tmin_val = read_obs_series(meta["tmin_csv"], st, val_start, val_end, valid_years)
        mod_tmax_val, mod_tmin_val = load_era5_daily_extremes(
            st_dir,
            reg_info["latitud_cne"],
            reg_info["longitud_cne"],
            reg_info["celda_era5"]["lat"],
            reg_info["celda_era5"]["lon"],
            range(val_start, val_end + 2),
            val_start,
            val_end,
        )

        # 3. Transform validation ERA5 with fitted Level 1
        l1_tmax_val, l1_tmin_val = m1.transform(mod_tmax_val, mod_tmin_val)

        # 4. Compute monthly biases on validation
        b_val_tmax = compute_calendar_monthly_biases(obs_tmax_val, l1_tmax_val)
        b_val_tmin = compute_calendar_monthly_biases(obs_tmin_val, l1_tmin_val)

        # 5. Compare against archived scorecard
        sc_level1 = meta["scorecard"]["descriptive_metrics"]["level1"]
        sc_b_tmax = sc_level1["monthly_biases_tmax"]
        sc_b_tmin = sc_level1["monthly_biases_tmin"]

        diff_tmax = max(abs(b_val_tmax[m] - sc_b_tmax[str(m)]) for m in range(1, 13))
        diff_tmin = max(abs(b_val_tmin[m] - sc_b_tmin[str(m)]) for m in range(1, 13))

        pass_tmax = (diff_tmax <= 1e-6)
        pass_tmin = (diff_tmin <= 1e-6)
        st_pass = pass_tmax and pass_tmin
        reproduction_all_pass &= st_pass

        self_check_1_results[st] = {
            "station_id": st,
            "station_name": reg_info["nombre"],
            "max_abs_diff_tmax": diff_tmax,
            "max_abs_diff_tmin": diff_tmin,
            "pass_tmax": pass_tmax,
            "pass_tmin": pass_tmin,
            "pass": st_pass,
        }

        status_str = "PASS" if st_pass else "FAIL"
        print(
            f"  Station {st} ({reg_info['nombre'][:25]}): "
            f"Max |Diff| Tmax = {diff_tmax:.2e}, Tmin = {diff_tmin:.2e} -> {status_str}"
        )

        # 6. Load full union series for Steps 2-5
        obs_tmax_all = read_obs_series(meta["tmax_csv"], st, min_yr, max_yr, valid_years)
        obs_tmin_all = read_obs_series(meta["tmin_csv"], st, min_yr, max_yr, valid_years)
        mod_tmax_all, mod_tmin_all = load_era5_daily_extremes(
            st_dir,
            reg_info["latitud_cne"],
            reg_info["longitud_cne"],
            reg_info["celda_era5"]["lat"],
            reg_info["celda_era5"]["lon"],
            range(min_yr, max_yr + 2),
            min_yr,
            max_yr,
        )

        station_loaded_data[st] = {
            "m1": m1,
            "cal_years": cal_years,
            "val_years": val_years,
            "valid_years": valid_years,
            "obs_tmax": obs_tmax_all,
            "obs_tmin": obs_tmin_all,
            "mod_tmax": mod_tmax_all,
            "mod_tmin": mod_tmin_all,
            "obs_tmax_val": obs_tmax_val,
            "obs_tmin_val": obs_tmin_val,
            "mod_tmax_val": mod_tmax_val,
            "mod_tmin_val": mod_tmin_val,
        }

    if not reproduction_all_pass:
        print("\n[CRITICAL ERROR] Self-Check 1 FAILED: Could not reproduce Level 1 monthly biases. Aborting.", file=sys.stderr)
        sys.exit(1)
    print("\nSelf-Check 1: EXACT REPRODUCTION OF ARCHIVED LEVEL 1 BIASES (PASS).")

    # --------------------------------------------------------------------------
    # Extra Self-Check: Valid paired days count against registry/scorecards
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EXTRA SELF-CHECK: VALID PAIRED DAYS COUNTS AGAINST ARCHIVED SCORECARDS")
    print("=" * 80)
    days_check_all_pass = True
    for st in OPENED_STATION_IDS:
        ld = station_loaded_data[st]
        sc_days = station_metadata[st]["scorecard"]["valid_paired_days"]
        n_obs_tmax = int((ld["obs_tmax_val"].notna() & ld["mod_tmax_val"].notna()).sum())
        n_obs_tmin = int((ld["obs_tmin_val"].notna() & ld["mod_tmin_val"].notna()).sum())
        pass_d_tmax = (n_obs_tmax == sc_days["tmax"])
        pass_d_tmin = (n_obs_tmin == sc_days["tmin"])
        days_check_all_pass &= (pass_d_tmax and pass_d_tmin)
        status_d = "PASS" if (pass_d_tmax and pass_d_tmin) else "FAIL"
        print(
            f"  Station {st}: Tmax Val Paired Days = {n_obs_tmax} (exp {sc_days['tmax']}) | "
            f"Tmin Val Paired Days = {n_obs_tmin} (exp {sc_days['tmin']}) -> {status_d}"
        )

    # --------------------------------------------------------------------------
    # Steps 2 - 5: Analysis Per Station and Variable
    # --------------------------------------------------------------------------
    summary_results: Dict[str, Any] = {
        "git_head": git_head,
        "script_path": "local/analysis_k/diagnostic/andean_residual_diagnostic.py",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "parameters": {
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicas": BOOTSTRAP_REPLICAS,
            "noise_floor_draws": NOISE_FLOOR_DRAWS,
            "min_days_per_month": MIN_DAYS_PER_MONTH,
            "min_months_per_year": MIN_MONTHS_PER_YEAR,
            "window_start_utc": WINDOW_START_UTC,
            "min_hours": MIN_HOURS,
            "lag_days": LAG_DAYS,
        },
        "self_check_0_inputs": {"status": "PASS" if all_inputs_pass else "FAIL"},
        "self_check_1_reproduction": self_check_1_results,
        "extra_self_check_days": {"status": "PASS" if days_check_all_pass else "FAIL"},
        "stations": {},
    }

    station_csv_paths: List[Path] = []

    for st in OPENED_STATION_IDS:
        meta = station_metadata[st]
        st_name = meta["reg"]["nombre"]
        ld = station_loaded_data[st]
        cal_years = ld["cal_years"]
        val_years = ld["val_years"]
        valid_years = ld["valid_years"]
        m1 = ld["m1"]

        st_out_dict: Dict[str, Any] = {
            "station_id": st,
            "station_name": st_name,
            "cal_years": cal_years,
            "val_years": val_years,
            "variables": {},
        }

        records_station_csv: List[Dict[str, Any]] = []

        print("\n" + "=" * 80)
        print(f"STATION {st} ({st_name}) - ANNUAL SERIES & RESIDUAL DIAGNOSTICS")
        print("=" * 80)

        for var_name, obs_s, mod_s, b_cal_dict, sc_mb_key in [
            ("Tmax", ld["obs_tmax"], ld["mod_tmax"], m1.tmax_biases_, "monthly_biases_tmax"),
            ("Tmin", ld["obs_tmin"], ld["mod_tmin"], m1.tmin_biases_, "monthly_biases_tmin"),
        ]:
            print(f"\n--- Variable: {var_name} ---")

            # ------------------------------------------------------------------
            # Step 2: Annual Series
            # ------------------------------------------------------------------
            ann_rows: List[Dict[str, Any]] = []
            for y in valid_years:
                mask_y = obs_s.index.year == y
                part = "calibration" if y in cal_years else "validation"
                n_days = int((obs_s[mask_y].notna() & mod_s[mask_y].notna()).sum())
                m_deltas = []
                m_resids = []
                for m in range(1, 13):
                    m_mask = mask_y & (obs_s.index.month == m)
                    v = obs_s[m_mask].notna() & mod_s[m_mask].notna()
                    if v.sum() >= MIN_DAYS_PER_MONTH:
                        delta_m = float(mod_s[m_mask][v].mean() - obs_s[m_mask][v].mean())
                        m_deltas.append(delta_m)
                        m_resids.append(delta_m - b_cal_dict[m])

                n_valid_m = len(m_deltas)
                is_included = (n_valid_m >= MIN_MONTHS_PER_YEAR)
                d_y = float(np.mean(m_deltas)) if is_included else np.nan
                r_y = float(np.mean(m_resids)) if is_included else np.nan

                ann_record = {
                    "station_id": st,
                    "station_name": st_name,
                    "variable": var_name,
                    "year": y,
                    "partition": part,
                    "n_valid_days": n_days,
                    "n_valid_months": n_valid_m,
                    "raw_bias_d_y": d_y,
                    "level1_residual_r_y": r_y,
                    "included_in_stats": is_included,
                }
                ann_rows.append(ann_record)
                records_station_csv.append(ann_record)

            df_ann = pd.DataFrame(ann_rows)

            print(f"Annual Series Summary (Total {len(df_ann)} years in registry):")
            print(f"{'Year':5s} | {'Part':11s} | {'Days':5s} | {'Months':6s} | {'Raw Bias d_y (K)':17s} | {'L1 Resid r_y (K)':17s} | {'Included'}")
            print("-" * 75)
            for _, r in df_ann.iterrows():
                d_str = f"{r['raw_bias_d_y']:17.4f}" if not np.isnan(r['raw_bias_d_y']) else f"{'NA':>17s}"
                r_str = f"{r['level1_residual_r_y']:17.4f}" if not np.isnan(r['level1_residual_r_y']) else f"{'NA':>17s}"
                print(f"{int(r['year']):5d} | {r['partition']:11s} | {int(r['n_valid_days']):5d} | {int(r['n_valid_months']):6d} | {d_str} | {r_str} | {r['included_in_stats']}")

            # OLS trend of r_y over included years
            df_inc = df_ann[df_ann["included_in_stats"]].copy()
            years_arr = df_inc["year"].values.astype(float)
            r_arr = df_inc["level1_residual_r_y"].values.astype(float)

            slope, intercept = np.polyfit(years_arr, r_arr, 1)
            trend_decade = float(slope * 10.0)

            rng_step2 = np.random.default_rng(BOOTSTRAP_SEED)
            boot_trends: List[float] = []
            for _ in range(BOOTSTRAP_REPLICAS):
                resample_idx = rng_step2.choice(len(years_arr), size=len(years_arr), replace=True)
                s_b, _ = np.polyfit(years_arr[resample_idx], r_arr[resample_idx], 1)
                boot_trends.append(float(s_b * 10.0))

            ci_trend = [
                float(np.percentile(boot_trends, 2.5)),
                float(np.percentile(boot_trends, 97.5)),
            ]

            print("-" * 75)
            print(f"Linear trend of r_y ({len(df_inc)} included years): {trend_decade:+.4f} K/decade | 95% CI: [{ci_trend[0]:+.4f}, {ci_trend[1]:+.4f}] K/decade")

            # ------------------------------------------------------------------
            # Step 3: Level Change (Calibration -> Validation)
            # ------------------------------------------------------------------
            cal_dy = df_inc[df_inc["partition"] == "calibration"]["raw_bias_d_y"].values.astype(float)
            val_dy = df_inc[df_inc["partition"] == "validation"]["raw_bias_d_y"].values.astype(float)

            mean_cal_dy = float(np.mean(cal_dy))
            mean_val_dy = float(np.mean(val_dy))
            delta_level = float(mean_val_dy - mean_cal_dy)

            rng_step3 = np.random.default_rng(BOOTSTRAP_SEED)
            boot_delta_level: List[float] = []
            for _ in range(BOOTSTRAP_REPLICAS):
                c_b = rng_step3.choice(cal_dy, size=len(cal_dy), replace=True)
                v_b = rng_step3.choice(val_dy, size=len(val_dy), replace=True)
                boot_delta_level.append(float(np.mean(v_b) - np.mean(c_b)))

            ci_level = [
                float(np.percentile(boot_delta_level, 2.5)),
                float(np.percentile(boot_delta_level, 97.5)),
            ]

            print(
                f"Level change (mean(d_val) - mean(d_cal)): {delta_level:+.4f} K "
                f"(Val mean = {mean_val_dy:+.4f} K, Cal mean = {mean_cal_dy:+.4f} K) | "
                f"95% CI: [{ci_level[0]:+.4f}, {ci_level[1]:+.4f}] K"
            )

            # ------------------------------------------------------------------
            # Step 4: Noise Floor (999 Random Reassignments)
            # ------------------------------------------------------------------
            inc_years = df_inc["year"].values
            k_cal = len(cal_dy)
            k_val = len(val_dy)

            # Pre-extract data points per (year, month) for fast draw evaluation
            data_by_ym: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
            for y in inc_years:
                mask_y = obs_s.index.year == y
                for m in range(1, 13):
                    m_mask = mask_y & (obs_s.index.month == m)
                    v = obs_s[m_mask].notna() & mod_s[m_mask].notna()
                    if v.sum() > 0:
                        data_by_ym[(int(y), m)] = (mod_s[m_mask][v].values, obs_s[m_mask][v].values)
                    else:
                        data_by_ym[(int(y), m)] = (np.array([]), np.array([]))

            rng_step4 = np.random.default_rng(BOOTSTRAP_SEED)
            noise_max_b: List[float] = []
            noise_abs_offset: List[float] = []
            noise_seas_amp: List[float] = []
            skipped_months_count = 0

            for _ in range(NOISE_FLOOR_DRAWS):
                perm = rng_step4.permutation(inc_years)
                p_cal_yrs = set(perm[:k_cal])
                p_val_yrs = set(perm[k_cal:])

                # Fit pseudo-calibration monthly biases
                b_pcal: Dict[int, Optional[float]] = {}
                for m in range(1, 13):
                    m_mods = [data_by_ym[(int(y), m)][0] for y in p_cal_yrs if len(data_by_ym[(int(y), m)][0]) > 0]
                    m_obss = [data_by_ym[(int(y), m)][1] for y in p_cal_yrs if len(data_by_ym[(int(y), m)][1]) > 0]
                    if len(m_mods) > 0:
                        all_m = np.concatenate(m_mods)
                        all_o = np.concatenate(m_obss)
                        b_pcal[m] = float(np.mean(all_m) - np.mean(all_o))
                    else:
                        b_pcal[m] = None

                # Evaluate on pseudo-validation
                b_pval: List[float] = []
                for m in range(1, 13):
                    if b_pcal[m] is None:
                        skipped_months_count += 1
                        continue
                    m_mods = [data_by_ym[(int(y), m)][0] for y in p_val_yrs if len(data_by_ym[(int(y), m)][0]) > 0]
                    m_obss = [data_by_ym[(int(y), m)][1] for y in p_val_yrs if len(data_by_ym[(int(y), m)][1]) > 0]
                    if len(m_mods) == 0:
                        skipped_months_count += 1
                        continue
                    all_m = np.concatenate(m_mods)
                    all_o = np.concatenate(m_obss)
                    bias_m = float(np.mean(all_m - b_pcal[m]) - np.mean(all_o))
                    b_pval.append(bias_m)

                b_pval_arr = np.array(b_pval)
                draw_max_b = float(np.max(np.abs(b_pval_arr)))
                draw_off = float(np.mean(b_pval_arr))
                draw_s_amp = float(np.max(np.abs(b_pval_arr - draw_off)))

                noise_max_b.append(draw_max_b)
                noise_abs_offset.append(abs(draw_off))
                noise_seas_amp.append(draw_s_amp)

            # Archived real Level 1 validation monthly biases
            real_b_dict = meta["scorecard"]["descriptive_metrics"]["level1"][sc_mb_key]
            real_b_vals = [float(real_b_dict[str(m)]) for m in range(1, 13)]
            real_max_b = float(max(abs(v) for v in real_b_vals))
            real_offset = float(np.mean(real_b_vals))
            real_abs_offset = float(abs(real_offset))
            real_seas_amp = float(max(abs(v - real_offset) for v in real_b_vals))

            rank_max_b = float(np.mean(np.array(noise_max_b) <= real_max_b) * 100.0)

            med_noise_max_b = float(np.median(noise_max_b))
            p5_noise_max_b = float(np.percentile(noise_max_b, 5))
            p95_noise_max_b = float(np.percentile(noise_max_b, 95))

            med_noise_abs_off = float(np.median(noise_abs_offset))
            p95_noise_abs_off = float(np.percentile(noise_abs_offset, 95))

            med_noise_seas_amp = float(np.median(noise_seas_amp))
            p95_noise_seas_amp = float(np.percentile(noise_seas_amp, 95))

            print("-" * 75)
            print("Step 4 Noise Floor & Step 5 Decomposition Comparison:")
            print(f"  Real Archived Level 1 Max |Bias_m|: {real_max_b:.4f} K")
            print(f"  Noise Floor Max |Bias_m| (999 draws): Median = {med_noise_max_b:.4f} K | P5 = {p5_noise_max_b:.4f} K | P95 = {p95_noise_max_b:.4f} K")
            print(f"  Percentile Rank of Real Value:       {rank_max_b:.2f}% (Skipped months: {skipped_months_count})")
            print(f"  Offset Decomposition:")
            print(f"    Offset: Real = {real_offset:+.4f} K | |Offset|: Real = {real_abs_offset:.4f} K | Noise Floor |Offset|: Median = {med_noise_abs_off:.4f} K, P95 = {p95_noise_abs_off:.4f} K")
            print(f"    Seasonal Amplitude: Real = {real_seas_amp:.4f} K | Noise Floor Seas Amp: Median = {med_noise_seas_amp:.4f} K, P95 = {p95_noise_seas_amp:.4f} K")

            var_res = {
                "n_included_years": len(df_inc),
                "trend_decade_k": trend_decade,
                "trend_95ci_k": ci_trend,
                "level_change_k": delta_level,
                "level_change_95ci_k": ci_level,
                "noise_floor": {
                    "real_max_monthly_bias_k": real_max_b,
                    "median_k": med_noise_max_b,
                    "p5_k": p5_noise_max_b,
                    "p95_k": p95_noise_max_b,
                    "percentile_rank": rank_max_b,
                    "skipped_months_count": skipped_months_count,
                },
                "decomposition": {
                    "real_offset_k": real_offset,
                    "real_abs_offset_k": real_abs_offset,
                    "real_seasonal_amplitude_k": real_seas_amp,
                    "noise_abs_offset_median_k": med_noise_abs_off,
                    "noise_abs_offset_p95_k": p95_noise_abs_off,
                    "noise_seasonal_amplitude_median_k": med_noise_seas_amp,
                    "noise_seasonal_amplitude_p95_k": p95_noise_seas_amp,
                },
            }
            st_out_dict["variables"][var_name] = var_res

        summary_results["stations"][st] = st_out_dict

        # Save CSV per station
        df_station_csv = pd.DataFrame(records_station_csv)
        csv_path = out_dir / f"diagnostic_{st}.csv"
        df_station_csv.to_csv(csv_path, index=False)
        station_csv_paths.append(csv_path)
        print(f"\nSaved station CSV: {csv_path.as_posix()}")

    # --------------------------------------------------------------------------
    # Save summary.json & Report Provenance
    # --------------------------------------------------------------------------
    script_file = Path("local/analysis_k/diagnostic/andean_residual_diagnostic.py")
    summary_results["script_sha256"] = compute_sha256(script_file) if script_file.is_file() else "PENDING"
    summary_results["input_files"] = input_hashes_manifest

    summary_json_path = out_dir / "summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)

    script_sha = compute_sha256(script_file)
    summary_json_sha = compute_sha256(summary_json_path)

    print("\n" + "=" * 80)
    print("OUTPUT FILES & SHA-256 HASHES")
    print("=" * 80)
    print(f"Script:               {script_file.as_posix()}")
    print(f"  SHA-256:            {script_sha}")
    for csv_p in station_csv_paths:
        print(f"Output CSV ({csv_p.stem}): {csv_p.as_posix()}")
        print(f"  SHA-256:            {compute_sha256(csv_p)}")
    print(f"Summary JSON:         {summary_json_path.as_posix()}")
    print(f"  SHA-256:            {summary_json_sha}")
    print("=" * 80)


if __name__ == "__main__":
    main()

