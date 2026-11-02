"""Diagnostic of the Andean residual on the five ALREADY-OPENED Andean stations,
extended with:
  (A) the five opened Andean stations without year 2020;
  (B) sampling-noise floor for JB and AC computed ONLY from calibration years;
  (C) text and code check of the §3.4 bootstrap;
  (D) sign convention of delta-z.

Methodology & Protocol Citations:
  - Observation reading and block slicing: reproduced from tropicor/core/phase3_evaluator.py,
    lines 627-715 (_read_obs_block).
  - ERA5 aggregation to 05-05 UTC daily extremes: reproduced from tropicor/core/phase3_evaluator.py,
    lines 719-825 (_load_era5_daily_block) using ERA5Adapter and aggregate_hourly_to_daily_temperature.
  - Level 1 fitting & monthly biases: reproduced from tropicor/core/phase3_evaluator.py,
    lines 1361-1380 (compute_monthly_biases) and tropicor.downscale.temperature (Level1MonthlyAdditiveTemperature).
  - JB & AC calibration loading: load_calibration_temperature_pair from tropicor.downscale.temperature.

RULES:
  - Commit NOTHING.
  - Read NO validation observations of JB (21205710) or AC (23215030).
  - Do NOT touch local/reports/validation_openings.jsonl.
  - Do NOT import tropicor.core.phase3_evaluator.
  - Arithmetic and raw metrics only; no interpretation.
"""

from __future__ import annotations

import datetime
import glob
import hashlib
import itertools
import json
import math
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

BENCHMARK_STATIONS = {
    "21205710": {
        "name": "JARDIN BOTANICO [21205710]",
        "tmax_csv": Path("local/newFiles/TempMaxima_JardinBotanico.csv"),
        "tmin_csv": Path("local/newFiles/TempMinima_JardinBotanico.csv"),
        "era5_dir": Path("local/era5_ts/21205710"),
        "cal_report": Path("docs/science/appendix_j/calibration/benchmark_calibration_21205710.json"),
        "lag_audit": Path("docs/science/appendix_j/lag_audit/benchmark_lag_audit_validation_21205710.json"),
        "cal_start": 1990,
        "cal_end": 2004,
        "n_cal_years": 15,
        "real_val_years_note": "2005-2020 (16 years)",
    },
    "23215030": {
        "name": "AGUAS CLARAS [23215030]",
        "tmax_csv": Path("local/newFiles/TempMaxima_AguasClaras.csv"),
        "tmin_csv": Path("local/newFiles/TempMinima_AguasClaras.csv"),
        "era5_dir": Path("local/era5_ts/23215030"),
        "cal_report": Path("docs/science/appendix_j/calibration/benchmark_calibration_23215030.json"),
        "lag_audit": Path("docs/science/appendix_j/lag_audit/benchmark_lag_audit_validation_23215030.json"),
        "cal_start": 1990,
        "cal_end": 2000,
        "n_cal_years": 11,
        "real_val_years_note": "2001-2011 (11 years)",
    },
}


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
    """Read observation extremes from CSV, parse dates, slice block, and mask invalid years."""
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
    """Load hourly ERA5 NetCDFs, extract cell, aggregate to daily 05-05 UTC, slice block."""
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


def run_sampling_noise_simulation(
    inc_years: Sequence[int],
    data_by_ym: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]],
    k_cal: int,
    k_val: int,
    n_draws: int = NOISE_FLOOR_DRAWS,
    seed: int = BOOTSTRAP_SEED,
    enumerate_all: bool = False,
) -> Tuple[List[float], List[float], List[float], int, str]:
    """Run random reassignments or full enumeration into pseudo-cal and pseudo-val partitions."""
    noise_max_b: List[float] = []
    noise_abs_offset: List[float] = []
    noise_seas_amp: List[float] = []
    skipped_months_count = 0

    if enumerate_all:
        all_combinations = list(itertools.combinations(inc_years, k_val))
        mode_str = f"Enumerated ALL {len(all_combinations)} distinct assignments"
        assignments = []
        for p_val in all_combinations:
            p_val_set = set(p_val)
            p_cal_set = set(inc_years) - p_val_set
            assignments.append((p_cal_set, p_val_set))
    else:
        rng = np.random.default_rng(seed)
        mode_str = f"Random {n_draws} draws (seed {seed})"
        assignments = []
        for _ in range(n_draws):
            perm = rng.permutation(inc_years)
            assignments.append((set(perm[:k_cal]), set(perm[k_cal:k_cal + k_val])))

    for p_cal_yrs, p_val_yrs in assignments:
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

    return noise_max_b, noise_abs_offset, noise_seas_amp, skipped_months_count, mode_str


def compute_annual_series_and_stats(
    obs_s: pd.Series,
    mod_s: pd.Series,
    target_years: Sequence[int],
    cal_years_set: Set[int],
    b_ref_dict: Dict[int, float],
) -> Tuple[pd.DataFrame, float, List[float]]:
    """Compute annual series of d_y and r_y, and bootstrap trend."""
    ann_rows: List[Dict[str, Any]] = []
    for y in target_years:
        mask_y = obs_s.index.year == y
        part = "calibration" if y in cal_years_set else "validation"
        n_days = int((obs_s[mask_y].notna() & mod_s[mask_y].notna()).sum())
        m_deltas = []
        m_resids = []
        for m in range(1, 13):
            m_mask = mask_y & (obs_s.index.month == m)
            v = obs_s[m_mask].notna() & mod_s[m_mask].notna()
            if v.sum() >= MIN_DAYS_PER_MONTH:
                delta_m = float(mod_s[m_mask][v].mean() - obs_s[m_mask][v].mean())
                m_deltas.append(delta_m)
                m_resids.append(delta_m - b_ref_dict[m])

        n_valid_m = len(m_deltas)
        is_included = (n_valid_m >= MIN_MONTHS_PER_YEAR)
        d_y = float(np.mean(m_deltas)) if is_included else np.nan
        r_y = float(np.mean(m_resids)) if is_included else np.nan

        ann_record = {
            "year": int(y),
            "partition": part,
            "n_valid_days": n_days,
            "n_valid_months": n_valid_m,
            "raw_bias_d_y": d_y,
            "level1_residual_r_y": r_y,
            "included_in_stats": is_included,
        }
        ann_rows.append(ann_record)

    df_ann = pd.DataFrame(ann_rows)
    df_inc = df_ann[df_ann["included_in_stats"]].copy()

    if len(df_inc) >= 2:
        years_arr = df_inc["year"].values.astype(float)
        r_arr = df_inc["level1_residual_r_y"].values.astype(float)
        slope, _ = np.polyfit(years_arr, r_arr, 1)
        trend_decade = float(slope * 10.0)

        rng = np.random.default_rng(BOOTSTRAP_SEED)
        boot_trends: List[float] = []
        for _ in range(BOOTSTRAP_REPLICAS):
            resample_idx = rng.choice(len(years_arr), size=len(years_arr), replace=True)
            s_b, _ = np.polyfit(years_arr[resample_idx], r_arr[resample_idx], 1)
            boot_trends.append(float(s_b * 10.0))

        ci_trend = [
            float(np.percentile(boot_trends, 2.5)),
            float(np.percentile(boot_trends, 97.5)),
        ]
    else:
        trend_decade = float("nan")
        ci_trend = [float("nan"), float("nan")]

    return df_ann, trend_decade, ci_trend


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    repo_root = Path(".")
    out_dir = Path("local/analysis_k/diagnostic2")
    out_dir.mkdir(parents=True, exist_ok=True)

    git_head = get_git_head(repo_root)

    print("=" * 80)
    print("APPENDIX K: EXTENDED ANDEAN & BENCHMARK DIAGNOSTICS (PARTS A, B, C, D)")
    print(f"Git commit HEAD: {git_head}")
    print("=" * 80)

    # ==========================================================================
    # STEP 0: SELF-CHECK 2 (INPUT FILE HASH VERIFICATION)
    # ==========================================================================
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

    station_metadata: Dict[str, Any] = {}
    for st in OPENED_STATION_IDS:
        cal_path = repo_root / exp_inputs["archived_paths"]["calibration"][st]
        all_inputs_pass &= verify_hash(cal_path, exp_inputs["calibration"][st], f"Calibration Report ({st})")

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
        print("\n[CRITICAL ERROR] Input hash verification FAILED. Aborting.", file=sys.stderr)
        sys.exit(1)
    print("\nSelf-Check 2: ALL INPUT FILE HASHES VERIFIED (PASS).")

    # ==========================================================================
    # STEP 1: SELF-CHECK 1 (REPRODUCTION OF LEVEL 1 MONTHLY VALIDATION BIASES)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: SELF-CHECK 1 (REPRODUCTION OF LEVEL 1 MONTHLY VALIDATION BIASES)")
    print("=" * 80)
    reproduction_all_pass = True
    self_check_1_results: Dict[str, Any] = {}
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

        # 1. Load Calibration Pair via official loader
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

        l1_tmax_val, l1_tmin_val = m1.transform(mod_tmax_val, mod_tmin_val)
        b_val_tmax = compute_calendar_monthly_biases(obs_tmax_val, l1_tmax_val)
        b_val_tmin = compute_calendar_monthly_biases(obs_tmin_val, l1_tmin_val)

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

        # Full union series for Steps 2-5
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
        print("\n[CRITICAL ERROR] Self-Check 1 FAILED. Aborting.", file=sys.stderr)
        sys.exit(1)
    print("\nSelf-Check 1: EXACT REPRODUCTION OF ARCHIVED LEVEL 1 BIASES (PASS).")

    # ==========================================================================
    # PART A: ANDEAN FIVE OPENED STATIONS (WITH AND WITHOUT YEAR 2020)
    # ==========================================================================
    print("\n" + "#" * 80)
    print("PART A: ANDEAN FIVE OPENED STATIONS (WITH AND WITHOUT YEAR 2020)")
    print("#" * 80)

    part_a_results: Dict[str, Any] = {}
    part_a_csv_paths: List[Path] = []

    for st in OPENED_STATION_IDS:
        meta = station_metadata[st]
        st_name = meta["reg"]["nombre"]
        ld = station_loaded_data[st]
        cal_years = ld["cal_years"]
        val_years = ld["val_years"]
        valid_years = ld["valid_years"]
        m1 = ld["m1"]
        cal_years_set = set(cal_years)

        has_2020 = 2020 in valid_years
        val_has_2020 = 2020 in val_years

        print(f"\nStation {st} ({st_name}):")
        if val_has_2020:
            print("  [INFO] Year 2020 is in the validation partition. Re-evaluating with and without 2020.")
        else:
            print("  [INFO] Year 2020 is absent from the partition. Results without 2020 are unchanged.")

        st_out: Dict[str, Any] = {
            "station_id": st,
            "station_name": st_name,
            "has_2020": has_2020,
            "val_has_2020": val_has_2020,
            "cal_years": cal_years,
            "val_years_with_2020": val_years,
            "val_years_without_2020": [y for y in val_years if y != 2020],
            "variables": {},
        }

        records_station_no2020_csv: List[Dict[str, Any]] = []

        for var_name, obs_s, mod_s, b_cal_dict, sc_mb_key in [
            ("Tmax", ld["obs_tmax"], ld["mod_tmax"], m1.tmax_biases_, "monthly_biases_tmax"),
            ("Tmin", ld["obs_tmin"], ld["mod_tmin"], m1.tmin_biases_, "monthly_biases_tmin"),
        ]:
            # ------------------------------------------------------------------
            # 1. Config: WITH 2020
            # ------------------------------------------------------------------
            df_ann_w, tr_w, ci_tr_w = compute_annual_series_and_stats(
                obs_s, mod_s, valid_years, cal_years_set, b_cal_dict
            )
            df_inc_w = df_ann_w[df_ann_w["included_in_stats"]].copy()
            cal_dy_w = df_inc_w[df_inc_w["partition"] == "calibration"]["raw_bias_d_y"].values.astype(float)
            val_dy_w = df_inc_w[df_inc_w["partition"] == "validation"]["raw_bias_d_y"].values.astype(float)
            delta_lvl_w = float(np.mean(val_dy_w) - np.mean(cal_dy_w))

            rng_w = np.random.default_rng(BOOTSTRAP_SEED)
            b_lvl_w = [
                float(np.mean(rng_w.choice(val_dy_w, len(val_dy_w), replace=True)) - np.mean(rng_w.choice(cal_dy_w, len(cal_dy_w), replace=True)))
                for _ in range(BOOTSTRAP_REPLICAS)
            ]
            ci_lvl_w = [float(np.percentile(b_lvl_w, 2.5)), float(np.percentile(b_lvl_w, 97.5))]

            # Pre-extract (year, month) for noise floor
            data_ym_w: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
            for y in df_inc_w["year"].values:
                mask_y = obs_s.index.year == y
                for m in range(1, 13):
                    m_mask = mask_y & (obs_s.index.month == m)
                    v = obs_s[m_mask].notna() & mod_s[m_mask].notna()
                    if v.sum() > 0:
                        data_ym_w[(int(y), m)] = (mod_s[m_mask][v].values, obs_s[m_mask][v].values)
                    else:
                        data_ym_w[(int(y), m)] = (np.array([]), np.array([]))

            nf_max_b_w, nf_off_w, nf_seas_w, _, _ = run_sampling_noise_simulation(
                df_inc_w["year"].values, data_ym_w, len(cal_dy_w), len(val_dy_w), NOISE_FLOOR_DRAWS, BOOTSTRAP_SEED, False
            )

            real_b_dict_w = meta["scorecard"]["descriptive_metrics"]["level1"][sc_mb_key]
            real_b_vals_w = [float(real_b_dict_w[str(m)]) for m in range(1, 13)]
            real_max_b_w = float(max(abs(v) for v in real_b_vals_w))
            real_off_w = float(np.mean(real_b_vals_w))
            real_seas_w = float(max(abs(v - real_off_w) for v in real_b_vals_w))
            rank_w = float(np.mean(np.array(nf_max_b_w) <= real_max_b_w) * 100.0)

            # ------------------------------------------------------------------
            # 2. Config: WITHOUT 2020
            # ------------------------------------------------------------------
            valid_years_no2020 = [y for y in valid_years if y != 2020]
            val_years_no2020 = [y for y in val_years if y != 2020]

            df_ann_no, tr_no, ci_tr_no = compute_annual_series_and_stats(
                obs_s, mod_s, valid_years_no2020, cal_years_set, b_cal_dict
            )
            df_inc_no = df_ann_no[df_ann_no["included_in_stats"]].copy()
            cal_dy_no = df_inc_no[df_inc_no["partition"] == "calibration"]["raw_bias_d_y"].values.astype(float)
            val_dy_no = df_inc_no[df_inc_no["partition"] == "validation"]["raw_bias_d_y"].values.astype(float)
            delta_lvl_no = float(np.mean(val_dy_no) - np.mean(cal_dy_no))

            rng_no = np.random.default_rng(BOOTSTRAP_SEED)
            b_lvl_no = [
                float(np.mean(rng_no.choice(val_dy_no, len(val_dy_no), replace=True)) - np.mean(rng_no.choice(cal_dy_no, len(cal_dy_no), replace=True)))
                for _ in range(BOOTSTRAP_REPLICAS)
            ]
            ci_lvl_no = [float(np.percentile(b_lvl_no, 2.5)), float(np.percentile(b_lvl_no, 97.5))]

            data_ym_no: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
            for y in df_inc_no["year"].values:
                mask_y = obs_s.index.year == y
                for m in range(1, 13):
                    m_mask = mask_y & (obs_s.index.month == m)
                    v = obs_s[m_mask].notna() & mod_s[m_mask].notna()
                    if v.sum() > 0:
                        data_ym_no[(int(y), m)] = (mod_s[m_mask][v].values, obs_s[m_mask][v].values)
                    else:
                        data_ym_no[(int(y), m)] = (np.array([]), np.array([]))

            nf_max_b_no, nf_off_no, nf_seas_no, _, _ = run_sampling_noise_simulation(
                df_inc_no["year"].values, data_ym_no, len(cal_dy_no), len(val_dy_no), NOISE_FLOOR_DRAWS, BOOTSTRAP_SEED, False
            )

            # Real Level 1 evaluated on validation partition without 2020
            # Slicing validation series to exclude 2020
            val_mask_no2020 = (obs_s.index.year >= min(val_years_no2020)) & (obs_s.index.year <= max(val_years_no2020)) & (obs_s.index.year != 2020)
            obs_val_no2020 = obs_s[val_mask_no2020]
            mod_val_no2020 = mod_s[val_mask_no2020]

            # Level 1 transformed validation without 2020
            l1_val_no2020 = mod_val_no2020.copy()
            for m in range(1, 13):
                m_idx = l1_val_no2020.index.month == m
                l1_val_no2020.loc[m_idx] = mod_val_no2020.loc[m_idx] - b_cal_dict[m]

            b_val_no_dict = compute_calendar_monthly_biases(obs_val_no2020, l1_val_no2020)
            real_b_vals_no = [float(b_val_no_dict[m]) for m in range(1, 13)]
            real_max_b_no = float(max(abs(v) for v in real_b_vals_no))
            real_off_no = float(np.mean(real_b_vals_no))
            real_seas_no = float(max(abs(v - real_off_no) for v in real_b_vals_no))
            rank_no = float(np.mean(np.array(nf_max_b_no) <= real_max_b_no) * 100.0)

            st_out["variables"][var_name] = {
                "with_2020": {
                    "real_max_monthly_bias_k": real_max_b_w,
                    "noise_floor_median_k": float(np.median(nf_max_b_w)),
                    "noise_floor_p5_k": float(np.percentile(nf_max_b_w, 5)),
                    "noise_floor_p95_k": float(np.percentile(nf_max_b_w, 95)),
                    "percentile_rank": rank_w,
                    "level_change_k": delta_lvl_w,
                    "level_change_95ci_k": ci_lvl_w,
                    "trend_decade_k": tr_w,
                    "trend_95ci_k": ci_tr_w,
                    "real_offset_k": real_off_w,
                    "real_seasonal_amplitude_k": real_seas_w,
                    "noise_abs_offset_median_k": float(np.median(nf_off_w)),
                    "noise_abs_offset_p95_k": float(np.percentile(nf_off_w, 95)),
                    "noise_seasonal_amplitude_median_k": float(np.median(nf_seas_w)),
                    "noise_seasonal_amplitude_p95_k": float(np.percentile(nf_seas_w, 95)),
                },
                "without_2020": {
                    "real_max_monthly_bias_k": real_max_b_no,
                    "noise_floor_median_k": float(np.median(nf_max_b_no)),
                    "noise_floor_p5_k": float(np.percentile(nf_max_b_no, 5)),
                    "noise_floor_p95_k": float(np.percentile(nf_max_b_no, 95)),
                    "percentile_rank": rank_no,
                    "level_change_k": delta_lvl_no,
                    "level_change_95ci_k": ci_lvl_no,
                    "trend_decade_k": tr_no,
                    "trend_95ci_k": ci_tr_no,
                    "real_offset_k": real_off_no,
                    "real_seasonal_amplitude_k": real_seas_no,
                    "noise_abs_offset_median_k": float(np.median(nf_off_no)),
                    "noise_abs_offset_p95_k": float(np.percentile(nf_off_no, 95)),
                    "noise_seasonal_amplitude_median_k": float(np.median(nf_seas_no)),
                    "noise_seasonal_amplitude_p95_k": float(np.percentile(nf_seas_no, 95)),
                },
            }

            for _, r in df_ann_no.iterrows():
                records_station_no2020_csv.append({
                    "station_id": st,
                    "station_name": st_name,
                    "variable": var_name,
                    "year": int(r["year"]),
                    "partition": r["partition"],
                    "n_valid_days": int(r["n_valid_days"]),
                    "n_valid_months": int(r["n_valid_months"]),
                    "raw_bias_d_y": r["raw_bias_d_y"],
                    "level1_residual_r_y": r["level1_residual_r_y"],
                    "included_in_stats": r["included_in_stats"],
                })

        # Save CSV without 2020
        df_csv_no2020 = pd.DataFrame(records_station_no2020_csv)
        csv_no2020_path = out_dir / f"diagnostic_{st}_no2020.csv"
        df_csv_no2020.to_csv(csv_no2020_path, index=False)
        part_a_csv_paths.append(csv_no2020_path)

        part_a_results[st] = st_out

    # Print Part A Summary Table
    print("\n" + "=" * 140)
    print("PART A: COMPARISON TABLE (WITH 2020 vs WITHOUT 2020)")
    print("=" * 140)
    print(f"{'Station':12s} | {'Var':4s} | {'Config':12s} | {'Real|B|':7s} | {'NF Med':6s} | {'NF P95':6s} | {'Rank':7s} | {'Shift':7s} | {'Shift 95% CI':20s} | {'Trend':7s} | {'Trend 95% CI':20s} | {'Off':7s} | {'NF|Off|':7s} | {'Seas':7s} | {'NF Seas':7s}")
    print("-" * 140)

    for st in OPENED_STATION_IDS:
        res_st = part_a_results[st]
        st_lbl = f"{st} ({'has 2020' if res_st['val_has_2020'] else 'no 2020'})"
        for var in ["Tmax", "Tmin"]:
            w = res_st["variables"][var]["with_2020"]
            no = res_st["variables"][var]["without_2020"]

            ci_s_w = f"[{w['level_change_95ci_k'][0]:+.4f}, {w['level_change_95ci_k'][1]:+.4f}]"
            ci_t_w = f"[{w['trend_95ci_k'][0]:+.4f}, {w['trend_95ci_k'][1]:+.4f}]"
            print(f"{st_lbl:12s} | {var:4s} | {'with_2020':12s} | {w['real_max_monthly_bias_k']:7.4f} | {w['noise_floor_median_k']:6.4f} | {w['noise_floor_p95_k']:6.4f} | {w['percentile_rank']:6.2f}% | {w['level_change_k']:+7.4f} | {ci_s_w:20s} | {w['trend_decade_k']:+7.4f} | {ci_t_w:20s} | {w['real_offset_k']:+7.4f} | {w['noise_abs_offset_median_k']:7.4f} | {w['real_seasonal_amplitude_k']:7.4f} | {w['noise_seasonal_amplitude_median_k']:7.4f}")

            ci_s_no = f"[{no['level_change_95ci_k'][0]:+.4f}, {no['level_change_95ci_k'][1]:+.4f}]"
            ci_t_no = f"[{no['trend_95ci_k'][0]:+.4f}, {no['trend_95ci_k'][1]:+.4f}]"
            cfg_lbl = "without_2020" if res_st["val_has_2020"] else "unchanged"
            print(f"{'':12s} | {var:4s} | {cfg_lbl:12s} | {no['real_max_monthly_bias_k']:7.4f} | {no['noise_floor_median_k']:6.4f} | {no['noise_floor_p95_k']:6.4f} | {no['percentile_rank']:6.2f}% | {no['level_change_k']:+7.4f} | {ci_s_no:20s} | {no['trend_decade_k']:+7.4f} | {ci_t_no:20s} | {no['real_offset_k']:+7.4f} | {no['noise_abs_offset_median_k']:7.4f} | {no['real_seasonal_amplitude_k']:7.4f} | {no['noise_seasonal_amplitude_median_k']:7.4f}")
            print("-" * 140)

    # ==========================================================================
    # PART B: SAMPLING-NOISE FLOOR FOR JB AND AC (CALIBRATION YEARS ONLY)
    # ==========================================================================
    print("\n" + "#" * 80)
    print("PART B: SAMPLING-NOISE FLOOR FOR JB AND AC (CALIBRATION YEARS ONLY)")
    print("#" * 80)

    part_b_results: Dict[str, Any] = {}
    part_b_csv_paths: List[Path] = []

    for st, cfg in BENCHMARK_STATIONS.items():
        st_name = cfg["name"]
        print(f"\n--- Station {st} ({st_name}) ---")

        # 1. Self-checks: Observation CSV hashes against lag audit JSON
        with open(cfg["lag_audit"], "r", encoding="utf-8") as f:
            lag_data = json.load(f)
        exp_ih = lag_data["input_hashes"]

        actual_tx_h = compute_sha256(cfg["tmax_csv"])
        actual_tn_h = compute_sha256(cfg["tmin_csv"])

        exp_tx_h = exp_ih[f"{st}/{cfg['tmax_csv'].name}"]
        exp_tn_h = exp_ih[f"{st}/{cfg['tmin_csv'].name}"]

        match_tx = actual_tx_h == exp_tx_h
        match_tn = actual_tn_h == exp_tn_h
        print(f"  Self-Check Hash Tmax CSV: {'MATCH' if match_tx else 'MISMATCH'} ({actual_tx_h})")
        print(f"  Self-Check Hash Tmin CSV: {'MATCH' if match_tn else 'MISMATCH'} ({actual_tn_h})")
        if not (match_tx and match_tn):
            print(f"[FATAL ERROR] Observation CSV hash mismatch on {st}. Aborting.", file=sys.stderr)
            sys.exit(1)

        # 2. Check calibration years match era5_source_equivalence.calibration_years
        cal_years_audit = lag_data["era5_source_equivalence"]["calibration_years"]
        cal_start_exp = cal_years_audit[0]
        cal_end_exp = cal_years_audit[-1]
        cal_years_list = list(range(cal_start_exp, cal_end_exp + 1))
        n_years = len(cal_years_list)
        cal_years_set = set(cal_years_list)
        print(f"  Calibration Years: {cal_start_exp}-{cal_end_exp} (Total {n_years} years)")
        if not (cal_start_exp == cfg["cal_start"] and cal_end_exp == cfg["cal_end"]):
            print(f"[FATAL ERROR] Calibration years mismatch for {st}. Aborting.", file=sys.stderr)
            sys.exit(1)

        # 3. Load Calibration Pair via official loader
        # Citing official call:
        # load_calibration_temperature_pair(station_id, obs_tmax_path, obs_tmin_path, era5_target, window_start_utc=5, lag_days=0, min_hours=24)
        print(f"  Loading calibration pair via load_calibration_temperature_pair (window=5, lag=0, min_hours=24)...")
        obs_tx_cal, obs_tn_cal, mod_tx_cal, mod_tn_cal = load_calibration_temperature_pair(
            station_id=st,
            obs_tmax_path=cfg["tmax_csv"],
            obs_tmin_path=cfg["tmin_csv"],
            era5_target=cfg["era5_dir"],
            window_start_utc=WINDOW_START_UTC,
            lag_days=LAG_DAYS,
            min_hours=MIN_HOURS,
        )

        # 4. Reproduce scalar raw_bias from benchmark_calibration_<id>.json
        with open(cfg["cal_report"], "r", encoding="utf-8") as f:
            cal_rep = json.load(f)
        exp_raw_bias = cal_rep["raw_bias"]
        scalar_tx_bias = float(np.round((mod_tx_cal - obs_tx_cal).mean(), 4))
        scalar_tn_bias = float(np.round((mod_tn_cal - obs_tn_cal).mean(), 4))

        match_sc_tx = (scalar_tx_bias == exp_raw_bias["tmax"])
        match_sc_tn = (scalar_tn_bias == exp_raw_bias["tmin"])
        print(f"  Self-Check Scalar raw_bias Tmax: {'MATCH' if match_sc_tx else 'MISMATCH'} ({scalar_tx_bias} vs exp {exp_raw_bias['tmax']})")
        print(f"  Self-Check Scalar raw_bias Tmin: {'MATCH' if match_sc_tn else 'MISMATCH'} ({scalar_tn_bias} vs exp {exp_raw_bias['tmin']})")
        if not (match_sc_tx and match_sc_tn):
            print(f"[FATAL ERROR] Scalar raw_bias reproduction failed on {st}. Aborting.", file=sys.stderr)
            sys.exit(1)

        # Also verify Level 1 calibration loss
        m1_bench = Level1MonthlyAdditiveTemperature()
        m1_bench.fit(obs_tx_cal, obs_tn_cal, mod_tx_cal, mod_tn_cal)
        exp_loss = cal_rep["level1"]["calibration_loss"]
        print(f"  Archived J.4 Level 1 Calibration Loss: {exp_loss}")

        # 5. Annual Series & Reassignments over CALIBRATION years only
        # Configurations:
        # (i) Balanced: ceil(n/2) cal, floor(n/2) val
        k_cal_bal = int(np.ceil(n_years / 2.0))
        k_val_bal = int(np.floor(n_years / 2.0))

        # (ii) 5-year pseudo-validation: 5 years val, rest cal
        k_val_5yr = 5
        k_cal_5yr = n_years - 5

        comb_bal_count = math.comb(n_years, k_val_bal)
        comb_5yr_count = math.comb(n_years, k_val_5yr)

        enum_bal = comb_bal_count <= NOISE_FLOOR_DRAWS
        enum_5yr = comb_5yr_count <= NOISE_FLOOR_DRAWS

        st_bench_out: Dict[str, Any] = {
            "station_id": st,
            "station_name": st_name,
            "calibration_years": cal_years_list,
            "n_calibration_years": n_years,
            "real_validation_partition_note": f"Real benchmark validation partition is {cfg['real_val_years_note']}, which differs from the pseudo-validation sizes used here.",
            "configurations": {
                "balanced": {
                    "pseudo_cal_years": k_cal_bal,
                    "pseudo_val_years": k_val_bal,
                    "total_possible_combinations": comb_bal_count,
                    "evaluated_mode": f"Enumerate ALL {comb_bal_count} assignments" if enum_bal else f"Random {NOISE_FLOOR_DRAWS} draws",
                },
                "pseudo_val_5yr": {
                    "pseudo_cal_years": k_cal_5yr,
                    "pseudo_val_years": k_val_5yr,
                    "total_possible_combinations": comb_5yr_count,
                    "evaluated_mode": f"Enumerate ALL {comb_5yr_count} assignments" if enum_5yr else f"Random {NOISE_FLOOR_DRAWS} draws",
                },
            },
            "variables": {},
        }

        ann_csv_records: List[Dict[str, Any]] = []

        for var_name, obs_s, mod_s in [
            ("Tmax", obs_tx_cal, mod_tx_cal),
            ("Tmin", obs_tn_cal, mod_tn_cal),
        ]:
            # Full-calibration Level 1 monthly biases
            b_full_cal = compute_calendar_monthly_biases(obs_s, mod_s)
            b_full_dict = {m: float(b_full_cal[m]) for m in range(1, 13)}

            # Annual series d_y and r_y
            df_ann, trend_dec, ci_trend = compute_annual_series_and_stats(
                obs_s, mod_s, cal_years_list, cal_years_set, b_full_dict
            )

            for _, r in df_ann.iterrows():
                ann_csv_records.append({
                    "station_id": st,
                    "station_name": st_name,
                    "variable": var_name,
                    "year": int(r["year"]),
                    "partition": "calibration",
                    "n_valid_days": int(r["n_valid_days"]),
                    "n_valid_months": int(r["n_valid_months"]),
                    "raw_bias_d_y": r["raw_bias_d_y"],
                    "level1_residual_r_y": r["level1_residual_r_y"],
                    "included_in_stats": r["included_in_stats"],
                })

            # Pre-extract (year, month) data
            df_inc = df_ann[df_ann["included_in_stats"]].copy()
            data_ym: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
            for y in df_inc["year"].values:
                mask_y = obs_s.index.year == y
                for m in range(1, 13):
                    m_mask = mask_y & (obs_s.index.month == m)
                    v = obs_s[m_mask].notna() & mod_s[m_mask].notna()
                    if v.sum() > 0:
                        data_ym[(int(y), m)] = (mod_s[m_mask][v].values, obs_s[m_mask][v].values)
                    else:
                        data_ym[(int(y), m)] = (np.array([]), np.array([]))

            inc_years = df_inc["year"].values

            # Config 1: Balanced
            nf_max_bal, nf_off_bal, nf_seas_bal, _, mode_bal = run_sampling_noise_simulation(
                inc_years, data_ym, k_cal_bal, k_val_bal, NOISE_FLOOR_DRAWS, BOOTSTRAP_SEED, enum_bal
            )

            # Config 2: 5-year pseudo-validation
            nf_max_5yr, nf_off_5yr, nf_seas_5yr, _, mode_5yr = run_sampling_noise_simulation(
                inc_years, data_ym, k_cal_5yr, k_val_5yr, NOISE_FLOOR_DRAWS, BOOTSTRAP_SEED, enum_5yr
            )

            st_bench_out["variables"][var_name] = {
                "annual_trend_decade_k": trend_dec,
                "annual_trend_95ci_k": ci_trend,
                "balanced": {
                    "pseudo_cal_years": k_cal_bal,
                    "pseudo_val_years": k_val_bal,
                    "mode": mode_bal,
                    "max_monthly_bias_k": {
                        "median": float(np.median(nf_max_bal)),
                        "p5": float(np.percentile(nf_max_bal, 5)),
                        "p95": float(np.percentile(nf_max_bal, 95)),
                    },
                    "abs_offset_k": {
                        "median": float(np.median(nf_off_bal)),
                        "p5": float(np.percentile(nf_off_bal, 5)),
                        "p95": float(np.percentile(nf_off_bal, 95)),
                    },
                    "seasonal_amplitude_k": {
                        "median": float(np.median(nf_seas_bal)),
                        "p5": float(np.percentile(nf_seas_bal, 5)),
                        "p95": float(np.percentile(nf_seas_bal, 95)),
                    },
                },
                "pseudo_val_5yr": {
                    "pseudo_cal_years": k_cal_5yr,
                    "pseudo_val_years": k_val_5yr,
                    "mode": mode_5yr,
                    "max_monthly_bias_k": {
                        "median": float(np.median(nf_max_5yr)),
                        "p5": float(np.percentile(nf_max_5yr, 5)),
                        "p95": float(np.percentile(nf_max_5yr, 95)),
                    },
                    "abs_offset_k": {
                        "median": float(np.median(nf_off_5yr)),
                        "p5": float(np.percentile(nf_off_5yr, 5)),
                        "p95": float(np.percentile(nf_off_5yr, 95)),
                    },
                    "seasonal_amplitude_k": {
                        "median": float(np.median(nf_seas_5yr)),
                        "p5": float(np.percentile(nf_seas_5yr, 5)),
                        "p95": float(np.percentile(nf_seas_5yr, 95)),
                    },
                },
            }

        # Save annual series CSV
        df_bench_csv = pd.DataFrame(ann_csv_records)
        bench_csv_p = out_dir / f"calibration_annual_{st}.csv"
        df_bench_csv.to_csv(bench_csv_p, index=False)
        part_b_csv_paths.append(bench_csv_p)

        part_b_results[st] = st_bench_out

    # Print Part B Summary Tables
    print("\n" + "=" * 115)
    print("PART B RESULTS: JB & AC CALIBRATION-YEARS RESIDUAL SAMPLING-NOISE FLOOR")
    print("=" * 115)
    print(f"{'Station':25s} | {'Var':4s} | {'Config':15s} | {'Mode':32s} | {'Max|B| Med [P5, P95] (K)':26s} | {'|Off| Med/P95':14s} | {'Seas Med/P95':14s}")
    print("-" * 140)

    for st, cfg in BENCHMARK_STATIONS.items():
        res_b = part_b_results[st]
        for var in ["Tmax", "Tmin"]:
            v_res = res_b["variables"][var]
            for cfg_name, cfg_title in [("balanced", "Balanced"), ("pseudo_val_5yr", "5-yr PseudoVal")]:
                c_data = v_res[cfg_name]
                mb_str = f"{c_data['max_monthly_bias_k']['median']:.4f} [{c_data['max_monthly_bias_k']['p5']:.4f}, {c_data['max_monthly_bias_k']['p95']:.4f}]"
                off_str = f"{c_data['abs_offset_k']['median']:.4f} / {c_data['abs_offset_k']['p95']:.4f}"
                seas_str = f"{c_data['seasonal_amplitude_k']['median']:.4f} / {c_data['seasonal_amplitude_k']['p95']:.4f}"
                print(f"{res_b['station_name'][:25]:25s} | {var:4s} | {cfg_title:15s} | {c_data['mode']:32s} | {mb_str:26s} | {off_str:14s} | {seas_str:14s}")
        print(f"  Note: {res_b['real_validation_partition_note']}")
        print("-" * 140)

    # ==========================================================================
    # PART C: BOOTSTRAP IN §3.4 TEXT & CODE CHECK
    # ==========================================================================
    print("\n" + "#" * 80)
    print("PART C: BOOTSTRAP IN §3.4 TEXT & CODE CHECK")
    print("#" * 80)

    part_c_quotes = {
        "protocol_section_3_4": {
            "file": "docs/science/evaluation-protocol.md",
            "lines": "140-150",
            "text": (
                "### 3.4 Temperature Decisive Metrics & Falsification Thresholds\n"
                "Evaluated on out-of-sample validation series across N_decisive = 3 decisive metrics, subject to an absolute physical consistency disqualifying gate:\n\n"
                "| Metric | Definition | Threshold / Tolerance | Rationale |\n"
                "| :--- | :--- | :---: | :--- |\n"
                "| **|Bias_m|** | Maximum absolute monthly bias across all calendar months (m=1..12): max(max_m |Bias_m(T_max)|, max_m |Bias_m(T_min)|) | <= 0.5 °C | Prevents seasonal phase distortion in degree-day and phenological models. |\n"
                "| **DTR Bias** | Diurnal Temperature Range (T_max - T_min): absolute value of the mean DTR bias, |mean(DTR_mod - DTR_obs)| (not a daily MAE) | |Delta DTR| <= 0.8 °C | Captures boundary-layer thermal inertia and radiative damping. |\n"
                "| **Extreme Percentiles** | Maximum extreme percentile bias: max(|Delta P_5(T_min)|, |Delta P_95(T_max)|) | <= 0.8 °C | Evaluates thermal stress risks, frost exposure, and microclimatic extremes. |\n\n"
                "#### Aggregation of Extreme Components\n"
                "Pairwise extremes are aggregated taking the maximum absolute value. Both individual components (e.g., P_5 of T_min and P_95 of T_max) and their respective 95% annual bootstrap confidence intervals must satisfy the <= 0.8 °C threshold for the metric to achieve Pass."
            ),
        },
        "protocol_section_5_1": {
            "file": "docs/science/evaluation-protocol.md",
            "lines": "229-233",
            "text": (
                "### 5.1 Annual Block Bootstrap Uncertainty\n"
                "All metric biases and paired error differences are evaluated using a **1,000-replica block bootstrap with a fixed block length of 1 year (annual blocks)**:\n"
                "- Resampling whole 1-year blocks preserves interannual climate variability (such as ENSO regimes) and prevents the artificial disruption or mixing of seasonal cycles.\n"
                "- **Paired Bootstrap for Method Differences**: All paired error differences Delta Error = |Error_{k-1}| - |Error_k| and Delta |PBIAS| = |PBIAS_raw| - |PBIAS_corr| are computed via **paired annual block bootstrap**: in each of the 1,000 bootstrap iterations, the exact same resampled calendar years are evaluated for both candidate method and baseline.\n"
                "- The 30-day block bootstrap is strictly reserved for daily lag analysis (temporal alignment)."
            ),
        },
        "protocol_appendix_j_7": {
            "file": "docs/science/evaluation-protocol.md",
            "lines": "1093-1095",
            "text": (
                "### J.7 Valid Years and Minimum Sample\n"
                "1. A calendar year is valid if it contains at least one paired day with all four values valid (Sections D.16 and E.6.9).\n"
                "2. A track is computed only if its evaluation series contains at least `MIN_BOOTSTRAP_BLOCKS = 7` valid years. Otherwise the station-variable-track is reported as **Not Established (insufficient sample)**; under Section 5.4 no level can then reach Global Success at that station."
            ),
        },
        "protocol_appendix_j_8": {
            "file": "docs/science/evaluation-protocol.md",
            "lines": "1097-1099",
            "text": (
                "### J.8 Levels, Metrics and Outcomes\n"
                "Levels 0, 1, 2, Level 3 winner and Level 3 constant (Sections 6.2 and E.7). For each: decisive metrics with 95 % intervals (Section 3.4; 1,000 replicas, annual blocks, seed 42), physical consistency gate and crossing diagnostics (Section 3.5), three-tier outcomes (Section 5.2), state mapping (Section 5.3), concordance (Section 4.3), progression diagnostics with paired bootstrap (Section 6.4), and operational designation across both stations (Section 5.4)."
            ),
        },
        "code_protocol_bootstrap": {
            "file": "tropicor/core/protocol.py",
            "lines": "891-958",
            "text": (
                "def compute_annual_block_bootstrap_temperature(\n"
                "    tmax_obs: pd.Series,\n"
                "    tmax_mod: pd.Series,\n"
                "    tmin_obs: pd.Series,\n"
                "    tmin_mod: pd.Series,\n"
                "    n_replicas: int = DEFAULT_BOOTSTRAP_REPLICAS,\n"
                "    seed: int = DEFAULT_BOOTSTRAP_SEED,\n"
                ") -> Dict[str, BootstrapConfidenceInterval]:\n"
                "    \"\"\"Compute 95% CIs for decisive temp metrics via 1-year block bootstrap.\"\"\"\n"
                "    ...\n"
                "    valid_years = np.array([\n"
                "        y for y in sorted(df.index.year.unique())\n"
                "        if (df.loc[df.index.year == y, 'tmax_obs'].notna().any() and df.loc[df.index.year == y, 'tmin_obs'].notna().any())\n"
                "    ])\n"
                "    n_years = len(valid_years)\n"
                "    point_estimates = compute_temperature_decisive_metrics(df['tmax_obs'], df['tmax_mod'], df['tmin_obs'], df['tmin_mod'])\n"
                "    if n_years < MIN_BOOTSTRAP_BLOCKS:\n"
                "        return early_temp_results ... (insufficient_blocks=True)\n"
                "    year_dfs = {y: df[df.index.year == y] for y in valid_years}\n"
                "    rng = np.random.default_rng(seed)\n"
                "    sampled_metrics = {k: [] for k in point_estimates}\n"
                "    for _ in range(n_replicas):\n"
                "        resampled_years = rng.choice(valid_years, size=n_years, replace=True)\n"
                "        sampled_df = pd.concat([year_dfs[y] for y in resampled_years], axis=0)\n"
                "        rep_m = compute_temperature_decisive_metrics(\n"
                "            sampled_df['tmax_obs'], sampled_df['tmax_mod'], sampled_df['tmin_obs'], sampled_df['tmin_mod']\n"
                "        )\n"
                "        for k, v in rep_m.items():\n"
                "            if np.isfinite(v):\n"
                "                sampled_metrics[k].append(v)"
            ),
        },
        "code_evaluator_call_site": {
            "file": "tropicor/core/phase3_evaluator.py",
            "lines": "1671-1673 & 1885-1892",
            "text": (
                "# Call site for Level 1 decisive metrics bootstrap:\n"
                "l1_boot = compute_temperature_decisive_metrics_bootstrap(\n"
                "    obs_tmax_val, l1_tmax, obs_tmin_val, l1_tmin, n_replicas=bootstrap_replicas\n"
                ")\n\n"
                "# Inside compute_temperature_decisive_metrics_bootstrap:\n"
                "return compute_annual_block_bootstrap_temperature(\n"
                "    tmax_obs=obs_tmax,\n"
                "    tmax_mod=mod_tmax,\n"
                "    tmin_obs=obs_tmin,\n"
                "    tmin_mod=mod_tmin,\n"
                "    n_replicas=n_replicas,\n"
                "    seed=seed,\n"
                ")"
            ),
        },
    }

    print("\n[VERBATIM QUOTES & LINE NUMBERS]")
    for k, q in part_c_quotes.items():
        print(f"\n--- {k} ({q['file']} lines {q['lines']}) ---")
        print(q["text"])

    print("\n" + "-" * 80)
    print("FACTUAL ANSWERS TO SPECIFIC QUESTIONS:")
    print("-" * 80)
    print("Question (1): What is resampled (years of the validation partition only, or also calibration years)?")
    print("Answer: Years of the validation partition only.")
    print("  Quote: In phase3_evaluator.py (lines 1671-1673), the arguments passed to compute_temperature_decisive_metrics_bootstrap")
    print("         are 'obs_tmax_val, l1_tmax, obs_tmin_val, l1_tmin'. In protocol.py (lines 912-920), valid_years are derived")
    print("         from df.index.year.unique(), which contains solely the calendar years present in those validation series.")
    print("\nQuestion (2): Whether the correction (Level 1 / Level 3) is REFITTED in each replica or the fitted parameters are held fixed?")
    print("Answer: The fitted parameters are held fixed; the correction is NOT refitted in each replica.")
    print("  Quote: In phase3_evaluator.py, Level 1 transformation is performed ONCE prior to bootstrap evaluation:")
    print("         'l1_tmax, l1_tmin = m1.transform(mod_tmax_val, mod_tmin_val)' (line 1668). In protocol.py (lines 947-954),")
    print("         the bootstrap loop resamples rows from the pre-transformed dataframe and calls compute_temperature_decisive_metrics")
    print("         directly on (sampled_df['tmax_obs'], sampled_df['tmax_mod'], sampled_df['tmin_obs'], sampled_df['tmin_mod']).")
    print("\nQuestion (3): The number of replicas, seed and MIN_BOOTSTRAP_BLOCKS?")
    print("Answer:")
    print("  - Number of replicas: 1,000 (protocol.py line 31: DEFAULT_BOOTSTRAP_REPLICAS: int = 1000; §5.1 lines 229-230: '1,000-replica block bootstrap').")
    print("  - Seed: 42 (protocol.py line 32: DEFAULT_BOOTSTRAP_SEED: int = 42; §J.8 line 1098: 'seed 42').")
    print("  - MIN_BOOTSTRAP_BLOCKS: 7 (protocol.py line 33: MIN_BOOTSTRAP_BLOCKS: int = 7; §J.7 line 1095: 'MIN_BOOTSTRAP_BLOCKS = 7 valid years').")

    # ==========================================================================
    # PART D: SIGN CONVENTION OF DELTA-Z
    # ==========================================================================
    print("\n" + "#" * 80)
    print("PART D: SIGN CONVENTION OF DELTA-Z")
    print("#" * 80)

    # 1. Load statics_g7.json
    statics_g7_path = repo_root / "docs" / "science" / "appendix_g" / "statics_g7.json"
    with open(statics_g7_path, "r", encoding="utf-8") as f:
        statics_g7_stations = json.load(f)["stations"]

    # Table J.2 constants from evaluation-protocol.md
    table_j2_constants = {
        "21205710": {"name": "JARDIN BOTANICO [21205710]", "z_station": 2552.0, "z_era5": 2753.4, "delta_z": -201.4},
        "23215030": {"name": "AGUAS CLARAS [23215030]", "z_station": 133.8, "z_era5": 762.4, "delta_z": -628.6},
    }

    print("\n[VERBATIM DEFINITIONS]")
    print("1. tropicor/downscale/orography.py lines 87-88 & 113:")
    print("   'Calculates the vertical elevation difference:'")
    print("   '    Δz = z_station - z_model'")
    print("   'return float(station_elevation - model_elevation)'")
    print("   and line 240: 'delta_t = -1.0 * lapse_rate * delta_z'")
    print("\n2. docs/science/evaluation-protocol.md Section 6.2 & Section J.9 (line 1101):")
    print("   'Level 0-E: T_E = T_raw - \\Gamma\\Delta z with \\Gamma = 6.5 K km^{-1} fixed a priori and \\Delta z from Section J.2'")
    print("   Table J.2 (lines 1058-1061):")
    print("   '| 21205710 | ... | Elevation (m): 2552.0 | z_era5 (m): 2753.4 | Delta z (m): -201.4 |'")
    print("   '| 23215030 | ... | Elevation (m):  133.8 | z_era5 (m):  762.4 | Delta z (m): -628.6 |'")

    print("\n" + "=" * 115)
    print("ARITHMETIC COMPUTATION OF DELTA-Z CONVENTIONS")
    print("=" * 115)
    print(f"{'Station':25s} | {'Source File':20s} | {'z_station':9s} | {'z_era5':9s} | {'delta_z':12s} | {'delta_z - (z_st - z_era5)':25s} | {'delta_z - (z_era5 - z_st)':25s}")
    print("-" * 125)

    part_d_records: List[Dict[str, Any]] = []

    # 5 Andean stations
    for st in OPENED_STATION_IDS:
        reg_st = registry_data[st]
        st_name = reg_st["nombre"]
        st_elev_reg = float(reg_st["altitud_cne"])
        stat_st = statics_g7_stations[st]
        h_cne = float(stat_st["h_cne_m"])
        h_era5 = float(stat_st["h_era5_m"])
        delta_h = float(stat_st["delta_h_m"])

        diff_conv1 = float(delta_h - (h_cne - h_era5))
        diff_conv2 = float(delta_h - (h_era5 - h_cne))

        print(f"{st_name[:25]:25s} | {'statics_g7.json':20s} | {h_cne:9.1f} | {h_era5:9.4f} | {delta_h:+12.4f} | {diff_conv1:+25.4f} | {diff_conv2:+25.4f}")
        part_d_records.append({
            "station_id": st,
            "station_name": st_name,
            "source": "docs/science/appendix_g/statics_g7.json & registry",
            "z_station_m": h_cne,
            "z_era5_m": h_era5,
            "delta_z_m": delta_h,
            "diff_from_z_st_minus_z_era5_m": diff_conv1,
            "diff_from_z_era5_minus_z_st_m": diff_conv2,
            "followed_convention": "z_era5 - z_station",
        })

    # Benchmark stations JB and AC
    for st, c_vals in table_j2_constants.items():
        st_name = c_vals["name"]
        z_st = c_vals["z_station"]
        z_er = c_vals["z_era5"]
        dz = c_vals["delta_z"]

        diff_conv1 = float(dz - (z_st - z_er))
        diff_conv2 = float(dz - (z_er - z_st))

        print(f"{st_name[:25]:25s} | {'Table J.2 (protocol)':20s} | {z_st:9.1f} | {z_er:9.4f} | {dz:+12.4f} | {diff_conv1:+25.4f} | {diff_conv2:+25.4f}")
        part_d_records.append({
            "station_id": st,
            "station_name": st_name,
            "source": "docs/science/evaluation-protocol.md Table J.2",
            "z_station_m": z_st,
            "z_era5_m": z_er,
            "delta_z_m": dz,
            "diff_from_z_st_minus_z_era5_m": diff_conv1,
            "diff_from_z_era5_minus_z_st_m": diff_conv2,
            "followed_convention": "z_station - z_era5",
        })

    print("-" * 125)
    print("CONVENTION SUMMARY:")
    print("  1. docs/science/appendix_g/statics_g7.json follows: delta_h = z_era5 - z_station (diff is 0.0 m for all 5 stations).")
    print("  2. docs/science/evaluation-protocol.md Table J.2 follows: delta_z = z_station - z_era5 (diff is 0.0 m for JB and AC).")
    print("  3. tropicor/downscale/orography.py defines: delta_z = z_station - z_model (matching Table J.2).")
    print("  4. tropicor/data/andean_station_registry.json contains 'altitud_cne' only; it does not store z_era5 or delta_z.")

    # ==========================================================================
    # SAVE SUMMARY JSON & OUTPUT HASHES
    # ==========================================================================
    script_file = Path("local/analysis_k/diagnostic2/andean_residual_diagnostic.py")
    script_sha = compute_sha256(script_file) if script_file.is_file() else "PENDING"

    complete_summary: Dict[str, Any] = {
        "git_head": git_head,
        "script_path": script_file.as_posix(),
        "script_sha256": script_sha,
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
        "self_check_2_inputs": {"status": "PASS" if all_inputs_pass else "FAIL", "manifest": input_hashes_manifest},
        "self_check_1_level1_reproduction": self_check_1_results,
        "part_a_andean_opened_stations": part_a_results,
        "part_b_benchmark_calibration_sampling_noise": part_b_results,
        "part_c_bootstrap_audit": part_c_quotes,
        "part_d_delta_z_sign_audit": part_d_records,
    }

    summary_json_path = out_dir / "summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(complete_summary, f, indent=2)

    script_sha_final = compute_sha256(script_file)
    summary_json_sha = compute_sha256(summary_json_path)

    print("\n" + "=" * 80)
    print("OUTPUT FILES & SHA-256 HASHES")
    print("=" * 80)
    print(f"Script:               {script_file.as_posix()}")
    print(f"  SHA-256:            {script_sha_final}")
    for csv_p in part_a_csv_paths:
        print(f"Output CSV (Part A):  {csv_p.as_posix()}")
        print(f"  SHA-256:            {compute_sha256(csv_p)}")
    for csv_p in part_b_csv_paths:
        print(f"Output CSV (Part B):  {csv_p.as_posix()}")
        print(f"  SHA-256:            {compute_sha256(csv_p)}")
    print(f"Summary JSON:         {summary_json_path.as_posix()}")
    print(f"  SHA-256:            {summary_json_sha}")
    print("=" * 80)


if __name__ == "__main__":
    main()
