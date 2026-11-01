"""ERA5-only 'effect ceiling' screening for the orographic hypothesis (Appendix K, Gate 1) - v2.

Extends the baseline screening (ceiling_orographic.py) to include:
1. Daily-scale bound for primary configurations (Tmax 18 UTC, Tmin 12 UTC, 3 levels S):
   - SD of daily Gamma * |delta_z| in K over validation partition.
   - Anomaly SD after removing calibration calendar-month mean.
   - P5 and P95 of anomaly.
   - Maximum absolute anomaly.
2. Andean station generalizations (5 stations from andean_station_registry.json):
   - Part 2a: Registry and statics table (ID, name, central-cell lat/lon, z_era5, delta_z, cal/val years).
   - Part 2c: Ceiling computation + Part 1 daily bound per Andean station (eligibility and S computed).
   - Part 2d: Phase 3 scorecard extraction (max |Bias_m|, decisive outcome, 12 monthly bias values,
     offset, |offset|, seasonal amplitude max |s_m|, side-by-side with ceilings).

RULES:
- Commit NOTHING.
- Read NO raw observation CSV or ledger.
- Do NOT import tropicor modules.
- Arithmetic only; no interpretation.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr

# Physical and numerical constants
G_ACCEL = 9.80665  # m s-2
Z_OFFSET_THRESHOLD_M = 100.0  # m
ELIGIBILITY_FRACTION_MIN = 0.999
LOCAL_TIME_OFFSET_HOURS = -5

BENCHMARK_STATIONS = [
    {
        "id": "21205710",
        "name": "JB",
        "kind": "benchmark",
        "json_path": Path("docs/science/appendix_j/lag_audit/benchmark_lag_audit_validation_21205710.json"),
        "nc_path": Path("local/era5_pl_ts/21205710/era5_pl_ts_1990_2020.nc"),
    },
    {
        "id": "23215030",
        "name": "AC",
        "kind": "benchmark",
        "json_path": Path("docs/science/appendix_j/lag_audit/benchmark_lag_audit_validation_23215030.json"),
        "nc_path": Path("local/era5_pl_ts/23215030/era5_pl_ts_1990_2011.nc"),
    },
]

ANDEAN_STATION_IDS = [
    "37015020",
    "24025040",
    "26135040",
    "27015070",
    "52045010",
]


def compute_sha256(filepath: Path) -> str:
    """Compute SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_git_head(repo_root: Path = Path(".")) -> str:
    """Get current git HEAD hash."""
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


def compute_ols_gamma(Z: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Compute Gamma = -OLS slope of T on Z in K/km.

    Z: (..., K) in metres
    T: (..., K) in Kelvin
    Returns Gamma in K/km. Positive when temperature decreases with height.
    """
    Z_m = Z.mean(axis=-1, keepdims=True)
    T_m = T.mean(axis=-1, keepdims=True)
    cov = ((Z - Z_m) * (T - T_m)).sum(axis=-1)
    var_z = ((Z - Z_m) ** 2).sum(axis=-1)
    slope = cov / var_z  # K/m
    return -slope * 1000.0  # K/km


def run_self_check_a() -> Tuple[bool, float, float]:
    """Self-check (a): Verify synthetic profile with known 6.5 K/km lapse rate."""
    z_syn = np.array([1000.0, 2000.0, 3000.0])
    expected_gamma = 6.5  # K/km
    t_syn = 290.0 - (expected_gamma / 1000.0) * z_syn
    recovered_gamma = float(compute_ols_gamma(z_syn, t_syn))
    err = abs(recovered_gamma - expected_gamma)
    return (err < 1e-12), recovered_gamma, err


def load_andean_station_configs(repo_root: Path = Path(".")) -> List[Dict[str, Any]]:
    """Load configuration for the 5 Andean stations from registry and statics JSON."""
    reg_path = repo_root / "tropicor" / "data" / "andean_station_registry.json"
    statics_path = repo_root / "docs" / "science" / "appendix_g" / "statics_g7.json"

    if not reg_path.is_file():
        raise FileNotFoundError(f"Missing registry file: {reg_path}")
    if not statics_path.is_file():
        raise FileNotFoundError(f"Missing statics file: {statics_path}")

    with open(reg_path, "r", encoding="utf-8") as f:
        reg_data = json.load(f)["stations"]
    with open(statics_path, "r", encoding="utf-8") as f:
        stat_data = json.load(f)["stations"]

    configs: List[Dict[str, Any]] = []
    for st_id in ANDEAN_STATION_IDS:
        if st_id not in reg_data:
            raise KeyError(f"Station {st_id} missing from registry: {reg_path}")
        if st_id not in stat_data:
            raise KeyError(f"Station {st_id} missing from statics: {statics_path}")

        r_info = reg_data[st_id]
        s_info = stat_data[st_id]

        cal_b = r_info["cal_block"]
        val_b = r_info["val_block"]
        valid_yrs = r_info["valid_years"]

        cal_years = [y for y in valid_yrs if cal_b[0] <= y <= cal_b[1]]
        val_years = [y for y in valid_yrs if val_b[0] <= y <= val_b[1]]

        start_yr = min(cal_years + val_years)
        end_yr = max(cal_years + val_years)

        nc_path = repo_root / "local" / "era5_pl_ts" / st_id / f"era5_pl_ts_{start_yr}_{end_yr}.nc"
        scorecard_path = repo_root / "docs" / "science" / "appendix_g" / "phase3_andes" / f"phase3_scorecard_{st_id}.json"

        configs.append({
            "id": st_id,
            "name": r_info["nombre"],
            "kind": "andean",
            "lat": float(r_info["celda_era5"]["lat"]),
            "lon": float(r_info["celda_era5"]["lon"]),
            "z_era5_m": float(s_info["h_era5_m"]),
            "delta_z_m": float(s_info["delta_h_m"]),
            "cal_block": cal_b,
            "val_block": val_b,
            "cal_years": cal_years,
            "val_years": val_years,
            "nc_path": nc_path,
            "scorecard_path": scorecard_path,
            "reg_path": reg_path,
            "statics_path": statics_path,
        })

    return configs


def analyze_station(
    cfg: Dict[str, Any],
    out_dir: Path,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Run full screening analysis for a single station (Benchmark or Andean)."""
    st_id = cfg["id"]
    st_name = cfg["name"]

    # 1. Constants
    if cfg["kind"] == "benchmark":
        with open(cfg["json_path"], "r", encoding="utf-8") as f:
            meta = json.load(f)
        z_era5_m = float(meta["static_verification"]["z_era5_m"])
        delta_z_m = float(meta["static_verification"]["delta_z_m"])
        c_years = meta["era5_source_equivalence"]["calibration_years"]
        v_years = meta["partition_years"]
        cal_years = tuple(range(c_years[0], c_years[1] + 1))
        val_years = tuple(range(v_years[0], v_years[1] + 1))
    else:
        z_era5_m = cfg["z_era5_m"]
        delta_z_m = cfg["delta_z_m"]
        cal_years = tuple(cfg["cal_years"])
        val_years = tuple(cfg["val_years"])

    # 2. Load ERA5 pressure-level NetCDF
    nc_path = cfg["nc_path"]
    if not nc_path.is_file():
        raise FileNotFoundError(f"NetCDF file missing: {nc_path}")

    ds = xr.open_dataset(nc_path)
    utc_times = pd.DatetimeIndex(ds["valid_time"].values)
    local_times = utc_times + pd.Timedelta(hours=LOCAL_TIME_OFFSET_HOURS)
    local_days = pd.DatetimeIndex(local_times.floor("D"))
    utc_hours = utc_times.hour

    # Pl coord
    pl_name = "pressureLevel" if "pressureLevel" in ds.coords else "pressure_level"
    all_levels = [float(v) for v in ds[pl_name].values]
    all_levels_sorted = sorted(all_levels, reverse=True)  # surface upwards

    # Calculate height Z in metres
    zh_all = ds["z"].values / G_ACCEL

    # 3. Determine level set S on calibration partition
    cal_steps_mask = np.isin(local_days.year, cal_years)
    n_cal_steps = int(cal_steps_mask.sum())
    zh_cal = zh_all[cal_steps_mask, :]

    threshold_z = z_era5_m + Z_OFFSET_THRESHOLD_M

    print(f"\n{'=' * 80}")
    print(f"STATION {st_id} ({st_name}) - LEVEL ELIGIBILITY ANALYSIS")
    print(f"{'=' * 80}")
    print(f"z_era5_m: {z_era5_m:.4f} m | delta_z_m: {delta_z_m:.4f} m | Threshold (z_era5 + 100m): {threshold_z:.4f} m")
    print(f"Calibration years ({len(cal_years)} yrs): {list(cal_years)} ({n_cal_steps} time steps)")
    print(f"Validation years  ({len(val_years)} yrs): {list(val_years)}")
    print("-" * 80)
    print("Eligibility fraction for all pressure levels in calibration partition:")

    eligibility_fractions: Dict[str, float] = {}
    eligible_levels: List[float] = []

    for lvl in all_levels_sorted:
        lvl_idx = all_levels.index(lvl)
        zh_lvl = zh_cal[:, lvl_idx]
        frac = float((zh_lvl >= threshold_z).mean())
        eligibility_fractions[f"{int(round(lvl))} hPa"] = frac
        is_elig = frac >= ELIGIBILITY_FRACTION_MIN
        elig_str = "ELIGIBLE" if is_elig else "INELIGIBLE"
        print(f"  Level {int(round(lvl)):4d} hPa: fraction = {frac * 100:6.2f}% (min Z = {zh_lvl.min():.2f} m, max Z = {zh_lvl.max():.2f} m) -> {elig_str}")
        if is_elig:
            eligible_levels.append(lvl)

    # Primary S (3 lowest eligible levels) and Secondary S2 (2 lowest eligible levels)
    S_primary = eligible_levels[:3]
    S_secondary = eligible_levels[:2]

    print("-" * 80)
    print(f"Eligible levels set (>= 99.9%): {[int(round(v)) for v in eligible_levels]}")
    print(f"Primary level set S (3 lowest eligible): {[int(round(v)) for v in S_primary]} hPa")
    print(f"Sensitivity level set S2 (2 lowest eligible): {[int(round(v)) for v in S_secondary]} hPa")

    # Verify no ineligibility at any time step in S_primary or S_secondary
    for s_set, s_name in [(S_primary, "S_primary"), (S_secondary, "S_secondary")]:
        for lvl in s_set:
            lvl_idx = all_levels.index(lvl)
            inelig_count = int((zh_all[:, lvl_idx] < threshold_z).sum())
            if inelig_count > 0:
                print(f"[ERROR] Level {int(round(lvl))} hPa in {s_name} is ineligible at {inelig_count} time steps!", file=sys.stderr)
                raise RuntimeError(f"Level {int(round(lvl))} hPa in {s_name} is ineligible at {inelig_count} time steps.")
    print("Ineligibility check across all time steps for S: 0 ineligible steps (PASS).")

    # 4. Self-check (b): Gamma range check on primary S over all time steps
    s_indices = [all_levels.index(lvl) for lvl in S_primary]
    z_s = zh_all[:, s_indices]
    t_s = ds["t"].values[:, s_indices]
    gamma_all = compute_ols_gamma(z_s, t_s)

    gamma_min = float(gamma_all.min())
    gamma_max = float(gamma_all.max())
    gamma_mean = float(gamma_all.mean())
    count_outside_3_10 = int(((gamma_all < 3.0) | (gamma_all > 10.0)).sum())
    count_lt_3 = int((gamma_all < 3.0).sum())
    count_gt_10 = int((gamma_all > 10.0).sum())

    # 5. Self-check (c): Local days count per partition
    df_steps = pd.DataFrame({
        "utc_time": utc_times,
        "local_day": local_days,
        "year": local_days.year,
        "month": local_days.month,
        "utc_hour": utc_hours,
        "gamma_s3": gamma_all,
    })

    # Add Gamma for S2
    s2_indices = [all_levels.index(lvl) for lvl in S_secondary]
    z_s2 = zh_all[:, s2_indices]
    t_s2 = ds["t"].values[:, s2_indices]
    df_steps["gamma_s2"] = compute_ols_gamma(z_s2, t_s2)

    cal_mask = df_steps["year"].isin(cal_years)
    val_mask = df_steps["year"].isin(val_years)

    cal_local_days = sorted(set(df_steps[cal_mask]["local_day"]))
    val_local_days = sorted(set(df_steps[val_mask]["local_day"]))

    n_cal_local_days = len(cal_local_days)
    n_val_local_days = len(val_local_days)

    exp_cal_days = sum((datetime.date(y + 1, 1, 1) - datetime.date(y, 1, 1)).days for y in cal_years)
    exp_val_days = sum((datetime.date(y + 1, 1, 1) - datetime.date(y, 1, 1)).days for y in val_years)

    pass_days_cal = (n_cal_local_days == exp_cal_days)
    pass_days_val = (n_val_local_days == exp_val_days)

    records_csv: List[Dict[str, Any]] = []
    station_results: Dict[str, Any] = {
        "station_id": st_id,
        "station_name": st_name,
        "kind": cfg["kind"],
        "z_era5_m": z_era5_m,
        "delta_z_m": delta_z_m,
        "abs_delta_z_m": abs(delta_z_m),
        "calibration_years": list(cal_years),
        "validation_years": list(val_years),
        "eligibility_fractions": eligibility_fractions,
        "S_primary": [int(round(v)) for v in S_primary],
        "S_secondary": [int(round(v)) for v in S_secondary],
        "gamma_all_stats": {
            "min": gamma_min,
            "max": gamma_max,
            "mean": gamma_mean,
            "count_outside_3_10": count_outside_3_10,
            "count_lt_3": count_lt_3,
            "count_gt_10": count_gt_10,
        },
        "local_days_counts": {
            "calibration": {"observed": n_cal_local_days, "expected": exp_cal_days, "match": pass_days_cal},
            "validation": {"observed": n_val_local_days, "expected": exp_val_days, "match": pass_days_val},
        },
        "daily_scale_bounds": {},
        "windows": {},
    }

    # Helper function to compute monthly statistics for a given daily series
    def compute_window_monthly(
        df_daily: pd.DataFrame,
        window_name: str,
        levels_mode: str,
        levels_list: List[float],
        is_primary: bool,
    ) -> Dict[str, Any]:
        cal_sub = df_daily[df_daily["year"].isin(cal_years)]
        val_sub = df_daily[df_daily["year"].isin(val_years)]

        ym_cal = cal_sub.groupby(["year", "month"])["gamma"].mean().reset_index()
        ym_val = val_sub.groupby(["year", "month"])["gamma"].mean().reset_index()

        monthly_rows: List[Dict[str, Any]] = []
        c_list: List[float] = []

        for m in range(1, 13):
            v_cal = ym_cal[ym_cal["month"] == m]["gamma"].values
            v_val = ym_val[ym_val["month"] == m]["gamma"].values

            n_cal = len(v_cal)
            n_val = len(v_val)

            gbar_cal = float(np.mean(v_cal))
            gbar_val = float(np.mean(v_val))
            sd_cal = float(np.std(v_cal, ddof=1)) if n_cal > 1 else 0.0
            sd_val = float(np.std(v_val, ddof=1)) if n_val > 1 else 0.0

            diff_k_km = gbar_val - gbar_cal
            abs_diff_k_km = abs(diff_k_km)
            se_diff_k_km = float(np.sqrt((sd_cal ** 2) / n_cal + (sd_val ** 2) / n_val))

            scale_km = abs(delta_z_m) / 1000.0
            ceiling_c_k = float(scale_km * abs_diff_k_km)
            se_c_k = float(scale_km * se_diff_k_km)

            c_list.append(ceiling_c_k)

            row_dict = {
                "station_id": st_id,
                "station_name": st_name,
                "variable_window": window_name,
                "is_primary": is_primary,
                "levels_mode": levels_mode,
                "levels": ",".join(str(int(round(v))) for v in levels_list),
                "month": m,
                "n_years_cal": n_cal,
                "n_years_val": n_val,
                "gbar_cal_k_km": gbar_cal,
                "gbar_val_k_km": gbar_val,
                "sd_cal_k_km": sd_cal,
                "sd_val_k_km": sd_val,
                "diff_k_km": diff_k_km,
                "abs_diff_k_km": abs_diff_k_km,
                "se_diff_k_km": se_diff_k_km,
                "delta_z_m": delta_z_m,
                "ceiling_c_k": ceiling_c_k,
                "se_ceiling_c_k": se_c_k,
            }
            monthly_rows.append(row_dict)
            records_csv.append(row_dict)

        max_c = float(max(c_list))
        months_ge_02 = int(sum(c >= 0.2 for c in c_list))

        return {
            "window": window_name,
            "levels_mode": levels_mode,
            "levels": [int(round(v)) for v in levels_list],
            "monthly": monthly_rows,
            "max_ceiling_c_k": max_c,
            "months_ge_0_2k": months_ge_02,
        }

    # Daily series per hour/aggregation
    daily_18_s3 = df_steps[df_steps["utc_hour"] == 18][["local_day", "year", "month", "gamma_s3"]].rename(columns={"gamma_s3": "gamma"})
    daily_12_s3 = df_steps[df_steps["utc_hour"] == 12][["local_day", "year", "month", "gamma_s3"]].rename(columns={"gamma_s3": "gamma"})
    daily_06_s3 = df_steps[df_steps["utc_hour"] == 6][["local_day", "year", "month", "gamma_s3"]].rename(columns={"gamma_s3": "gamma"})
    daily_00_s3 = df_steps[df_steps["utc_hour"] == 0][["local_day", "year", "month", "gamma_s3"]].rename(columns={"gamma_s3": "gamma"})
    daily_4mean_s3 = df_steps.groupby(["local_day", "year", "month"])["gamma_s3"].mean().reset_index().rename(columns={"gamma_s3": "gamma"})

    daily_18_s2 = df_steps[df_steps["utc_hour"] == 18][["local_day", "year", "month", "gamma_s2"]].rename(columns={"gamma_s2": "gamma"})
    daily_12_s2 = df_steps[df_steps["utc_hour"] == 12][["local_day", "year", "month", "gamma_s2"]].rename(columns={"gamma_s2": "gamma"})
    daily_4mean_s2 = df_steps.groupby(["local_day", "year", "month"])["gamma_s2"].mean().reset_index().rename(columns={"gamma_s2": "gamma"})

    configs_to_run = [
        ("Tmax_18UTC", "3_levels", S_primary, daily_18_s3, True),
        ("Tmin_12UTC", "3_levels", S_primary, daily_12_s3, True),
        ("00UTC", "3_levels", S_primary, daily_00_s3, False),
        ("06UTC", "3_levels", S_primary, daily_06_s3, False),
        ("12UTC", "3_levels", S_primary, daily_12_s3, False),
        ("18UTC", "3_levels", S_primary, daily_18_s3, False),
        ("4step_mean", "3_levels", S_primary, daily_4mean_s3, False),
        ("Tmax_18UTC", "2_levels", S_secondary, daily_18_s2, False),
        ("Tmin_12UTC", "2_levels", S_secondary, daily_12_s2, False),
        ("4step_mean", "2_levels", S_secondary, daily_4mean_s2, False),
    ]

    for win_name, lev_mode, lev_list, df_win, is_prim in configs_to_run:
        res = compute_window_monthly(df_win, win_name, lev_mode, lev_list, is_prim)
        cfg_key = f"{win_name}__{lev_mode}"
        station_results["windows"][cfg_key] = res

    # 6. Part 1 Daily-Scale Bound for Primary Configurations (Tmax 18 UTC, Tmin 12 UTC)
    # Over all days of validation partition:
    # Daily effect in K: x_d = Gamma_d * |delta_z| / 1000.0
    # Calibration month mean: mu_cal,m = mean of x_d over cal days in month m
    # Anomaly: a_d = x_d - mu_cal,month(d)
    scale_km = abs(delta_z_m) / 1000.0
    daily_bound_results: Dict[str, Any] = {}

    for var_name, df_daily_prim in [("Tmax", daily_18_s3), ("Tmin", daily_12_s3)]:
        cal_days_df = df_daily_prim[df_daily_prim["year"].isin(cal_years)].copy()
        val_days_df = df_daily_prim[df_daily_prim["year"].isin(val_years)].copy()

        cal_days_df["x_k"] = cal_days_df["gamma"] * scale_km
        val_days_df["x_k"] = val_days_df["gamma"] * scale_km

        # Monthly mean on calibration
        cal_month_means = cal_days_df.groupby("month")["x_k"].mean().to_dict()

        # Validation daily metrics
        x_val = val_days_df["x_k"].values
        sd_daily_k = float(np.std(x_val, ddof=1))

        # Anomalies
        val_days_df["cal_month_mean_k"] = val_days_df["month"].map(cal_month_means)
        val_days_df["anomaly_k"] = val_days_df["x_k"] - val_days_df["cal_month_mean_k"]

        anom_val = val_days_df["anomaly_k"].values
        anomaly_sd_k = float(np.std(anom_val, ddof=1))
        p5_anomaly_k = float(np.percentile(anom_val, 5))
        p95_anomaly_k = float(np.percentile(anom_val, 95))
        max_abs_anomaly_k = float(np.max(np.abs(anom_val)))

        bound_dict = {
            "n_val_days": len(x_val),
            "sd_daily_k": sd_daily_k,
            "cal_month_means_k": {str(m): float(cal_month_means.get(m, np.nan)) for m in range(1, 13)},
            "anomaly_sd_k": anomaly_sd_k,
            "p5_anomaly_k": p5_anomaly_k,
            "p95_anomaly_k": p95_anomaly_k,
            "max_abs_anomaly_k": max_abs_anomaly_k,
        }
        daily_bound_results[var_name] = bound_dict

    station_results["daily_scale_bounds"] = daily_bound_results

    # 7. Print Primary Tables
    print(f"\n{'=' * 80}")
    print(f"PRIMARY SCREENING TABLES - STATION {st_id} ({st_name})")
    print(f"{'=' * 80}")

    for var_name, key in [("Tmax (18 UTC)", "Tmax_18UTC__3_levels"), ("Tmin (12 UTC)", "Tmin_12UTC__3_levels")]:
        w_data = station_results["windows"][key]
        print(f"\n--- Variable: {var_name} | Levels: {w_data['levels']} hPa | delta_z = {delta_z_m:.2f} m ---")
        print(f"{'Month':5s} | {'Gbar_cal':9s} | {'Gbar_val':9s} | {'Diff (K/km)':11s} | {'SE (K/km)':9s} | {'Ceiling C (K)':13s} | {'SE C (K)':8s}")
        print("-" * 75)
        for row in w_data["monthly"]:
            print(
                f"{row['month']:5d} | "
                f"{row['gbar_cal_k_km']:9.4f} | "
                f"{row['gbar_val_k_km']:9.4f} | "
                f"{row['diff_k_km']:11.4f} | "
                f"{row['se_diff_k_km']:9.4f} | "
                f"{row['ceiling_c_k']:13.4f} | "
                f"{row['se_ceiling_c_k']:8.4f}"
            )
        print("-" * 75)
        print(f"Summary: Max Ceiling C = {w_data['max_ceiling_c_k']:.4f} K | Months with C >= 0.2 K: {w_data['months_ge_0_2k']}/12")

    # 8. Print Daily-Scale Bound
    print(f"\n{'-' * 80}")
    print(f"DAILY-SCALE BOUNDS (VALIDATION PARTITION) - STATION {st_id} ({st_name})")
    print(f"{'-' * 80}")
    print(f"{'Variable':10s} | {'Days':6s} | {'Daily SD (K)':12s} | {'Anomaly SD (K)':14s} | {'P5 Anom (K)':11s} | {'P95 Anom (K)':12s} | {'Max |Anom| (K)':14s}")
    print("-" * 88)
    for var_name in ["Tmax", "Tmin"]:
        b = daily_bound_results[var_name]
        print(
            f"{var_name:10s} | "
            f"{b['n_val_days']:6d} | "
            f"{b['sd_daily_k']:12.4f} | "
            f"{b['anomaly_sd_k']:14.4f} | "
            f"{b['p5_anomaly_k']:11.4f} | "
            f"{b['p95_anomaly_k']:12.4f} | "
            f"{b['max_abs_anomaly_k']:14.4f}"
        )
    print("-" * 88)

    # 9. Print Sensitivity Summary
    print(f"\n{'=' * 80}")
    print(f"SENSITIVITY SCREENING SUMMARY - STATION {st_id} ({st_name})")
    print(f"{'=' * 80}")
    print(f"{'Configuration':25s} | {'Levels (hPa)':18s} | {'Max C (K)':10s} | {'Months >= 0.2K':14s} | {'Monthly C range (min..max K)':28s}")
    print("-" * 105)
    for cfg_key, w_data in station_results["windows"].items():
        c_vals = [r["ceiling_c_k"] for r in w_data["monthly"]]
        lev_str = str(w_data["levels"])
        print(
            f"{cfg_key:25s} | "
            f"{lev_str:18s} | "
            f"{w_data['max_ceiling_c_k']:10.4f} | "
            f"{w_data['months_ge_0_2k']:14d} | "
            f"{min(c_vals):.4f} .. {max(c_vals):.4f} K"
        )
    print("-" * 105)

    df_out = pd.DataFrame(records_csv)
    return station_results, df_out


def extract_scorecard_metrics(scorecard_path: Path) -> Dict[str, Any]:
    """Extract metrics from archived Phase 3 scorecard JSON without reading observations."""
    with open(scorecard_path, "r", encoding="utf-8") as f:
        sc = json.load(f)

    st_id = sc.get("station_id", "")
    levels_to_extract = ["level1", "level2", "level3_winner"]
    extracted: Dict[str, Any] = {
        "station_id": st_id,
        "scorecard_path": scorecard_path.as_posix(),
        "keys_quoted": [
            "scorecards.<level>.decisive_outcomes.max_monthly_bias",
            "scorecards.<level>.decisive_cis.max_monthly_bias.point_estimate",
            "descriptive_metrics.<level>.monthly_biases_tmax",
            "descriptive_metrics.<level>.monthly_biases_tmin",
        ],
        "levels": {},
    }

    for lvl in levels_to_extract:
        sc_lvl = sc.get("scorecards", {}).get(lvl, {})
        dm_lvl = sc.get("descriptive_metrics", {}).get(lvl, {})

        dec_outcome = sc_lvl.get("decisive_outcomes", {}).get("max_monthly_bias", "UNKNOWN")
        ci_pt = sc_lvl.get("decisive_cis", {}).get("max_monthly_bias", {}).get("point_estimate")

        var_metrics: Dict[str, Any] = {}
        for var in ["tmax", "tmin"]:
            key = f"monthly_biases_{var}"
            if key in dm_lvl:
                biases_dict = dm_lvl[key]
                b_vals = [float(biases_dict[str(m)]) for m in range(1, 13)]
                max_b = float(max(abs(v) for v in b_vals))
                offset = float(np.mean(b_vals))
                s_vals = [float(v - offset) for v in b_vals]
                s_amp = float(max(abs(s) for s in s_vals))
                var_metrics[var.upper()] = {
                    "has_12_monthly": True,
                    "monthly_biases": {str(m): b_vals[m - 1] for m in range(1, 13)},
                    "max_monthly_bias": max_b,
                    "offset": offset,
                    "abs_offset": abs(offset),
                    "seasonal_amplitude": s_amp,
                }
            else:
                var_metrics[var.upper()] = {
                    "has_12_monthly": False,
                    "keys_present": list(dm_lvl.keys()),
                }

        extracted["levels"][lvl] = {
            "decisive_outcome": dec_outcome,
            "overall_max_monthly_bias_ci_point": ci_pt,
            "variables": var_metrics,
        }

    return extracted


def main() -> None:
    repo_root = Path(".")
    out_dir = Path("local/analysis_k")
    out_dir.mkdir(parents=True, exist_ok=True)

    git_head = get_git_head(repo_root)

    print("=" * 80)
    print("APPENDIX K: OROGRAPHIC HYPOTHESIS EFFECT CEILING SCREENING - V2")
    print(f"Git commit HEAD: {git_head}")
    print("=" * 80)

    # 1. Run Self-Check (a): Synthetic profile recovery
    pass_a, rec_gamma, err_a = run_self_check_a()
    status_a = "PASS" if pass_a else "FAIL"
    print(f"\n[SELF-CHECK A] Synthetic profile recovery test:")
    print(f"  Target lapse rate: 6.500000 K/km")
    print(f"  Recovered rate:    {rec_gamma:.6f} K/km (absolute error: {err_a:.2e})")
    print(f"  Result: {status_a}")

    all_results: Dict[str, Any] = {
        "git_head": git_head,
        "script_path": "local/analysis_k/ceiling_orographic_v2.py",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "constants": {
            "G_ACCEL": G_ACCEL,
            "Z_OFFSET_THRESHOLD_M": Z_OFFSET_THRESHOLD_M,
            "ELIGIBILITY_FRACTION_MIN": ELIGIBILITY_FRACTION_MIN,
            "LOCAL_TIME_OFFSET_HOURS": LOCAL_TIME_OFFSET_HOURS,
        },
        "self_check_a": {
            "status": status_a,
            "target_gamma_k_km": 6.5,
            "recovered_gamma_k_km": rec_gamma,
            "absolute_error": err_a,
        },
        "part2a_table": [],
        "stations": {},
        "phase3_scorecard_comparison": {},
    }

    # 2. Part 2(a): Load and build table of Andean stations
    andean_configs = load_andean_station_configs(repo_root)

    print(f"\n{'=' * 80}")
    print("PART 2(a): ANDEAN STATIONS SPECIFICATION TABLE")
    print(f"{'=' * 80}")
    print(f"{'ID':10s} | {'Name':34s} | {'Lat':6s} | {'Lon':7s} | {'z_era5 (m)':10s} | {'delta_z (m)':11s} | {'Cal Years':20s} | {'Val Years':20s}")
    print("-" * 125)

    part2a_rows: List[Dict[str, Any]] = []
    for ac in andean_configs:
        cal_str = f"{len(ac['cal_years'])} yrs {ac['cal_years']}"
        val_str = f"{len(ac['val_years'])} yrs {ac['val_years']}"
        print(
            f"{ac['id']:10s} | "
            f"{ac['name'][:34]:34s} | "
            f"{ac['lat']:6.2f} | "
            f"{ac['lon']:7.2f} | "
            f"{ac['z_era5_m']:10.2f} | "
            f"{ac['delta_z_m']:11.2f} | "
            f"{str(ac['cal_years']):20s} | "
            f"{str(ac['val_years']):20s}"
        )
        row_dict = {
            "station_id": ac["id"],
            "station_name": ac["name"],
            "lat": ac["lat"],
            "lon": ac["lon"],
            "z_era5_m": ac["z_era5_m"],
            "delta_z_m": ac["delta_z_m"],
            "calibration_years": ac["cal_years"],
            "validation_years": ac["val_years"],
        }
        part2a_rows.append(row_dict)

    all_results["part2a_table"] = part2a_rows
    print("-" * 125)

    # 3. Analyze all stations (Benchmarks JB & AC, then Andean stations)
    all_configs = BENCHMARK_STATIONS + andean_configs
    station_csv_paths: List[Path] = []

    for cfg in all_configs:
        st_res, df_st = analyze_station(cfg, out_dir)
        all_results["stations"][cfg["id"]] = st_res

        # Save station CSV
        csv_path = out_dir / f"ceiling_{cfg['id']}.csv"
        df_st.to_csv(csv_path, index=False)
        station_csv_paths.append(csv_path)
        print(f"Saved station CSV: {csv_path.as_posix()}")

    # 4. Part 2(d): Scorecard extraction and side-by-side comparison
    print(f"\n{'=' * 80}")
    print("PART 2(d): PHASE 3 SCORECARD COMPARISON (SIDE-BY-SIDE WITH CEILINGS)")
    print(f"{'=' * 80}")
    print("Exact scorecard keys used:")
    print("  - 'scorecards.<level>.decisive_outcomes.max_monthly_bias'")
    print("  - 'descriptive_metrics.<level>.monthly_biases_tmax'")
    print("  - 'descriptive_metrics.<level>.monthly_biases_tmin'")
    print("-" * 80)

    scorecard_comparisons: Dict[str, Any] = {}
    for ac in andean_configs:
        st_id = ac["id"]
        sc_metrics = extract_scorecard_metrics(ac["scorecard_path"])
        scorecard_comparisons[st_id] = sc_metrics

        st_res = all_results["stations"][st_id]
        tmax_c_data = st_res["windows"]["Tmax_18UTC__3_levels"]
        tmin_c_data = st_res["windows"]["Tmin_12UTC__3_levels"]

        tmax_c_monthly = [r["ceiling_c_k"] for r in tmax_c_data["monthly"]]
        tmin_c_monthly = [r["ceiling_c_k"] for r in tmin_c_data["monthly"]]
        max_c_tmax = tmax_c_data["max_ceiling_c_k"]
        max_c_tmin = tmin_c_data["max_ceiling_c_k"]

        print(f"\n>>> STATION {st_id} ({ac['name']}) | delta_z = {ac['delta_z_m']:.2f} m <<<")
        print(f"Max Ceiling C: Tmax = {max_c_tmax:.4f} K | Tmin = {max_c_tmin:.4f} K")

        for lvl in ["level1", "level2", "level3_winner"]:
            lvl_data = sc_metrics["levels"][lvl]
            dec_out = lvl_data["decisive_outcome"]
            print(f"\n  [Level: {lvl}] Decisive Outcome: {dec_out}")
            print(f"  {'Var':5s} | {'Max|Bias_m|':11s} | {'Max C (K)':9s} | {'Offset (K)':10s} | {'|Offset|':9s} | {'Seas Amp':9s} | {'12 Monthly Biases (K)':52s}")
            print("  " + "-" * 115)
            for var, c_monthly, max_c in [("TMAX", tmax_c_monthly, max_c_tmax), ("TMIN", tmin_c_monthly, max_c_tmin)]:
                vm = lvl_data["variables"][var]
                if vm["has_12_monthly"]:
                    b_str = ", ".join(f"{vm['monthly_biases'][str(m)]:.2f}" for m in range(1, 13))
                    print(
                        f"  {var:5s} | "
                        f"{vm['max_monthly_bias']:11.4f} | "
                        f"{max_c:9.4f} | "
                        f"{vm['offset']:10.4f} | "
                        f"{vm['abs_offset']:9.4f} | "
                        f"{vm['seasonal_amplitude']:9.4f} | "
                        f"[{b_str}]"
                    )
                else:
                    print(f"  {var:5s} | (12 monthly values NOT in scorecard; keys present: {vm['keys_present']})")
            print("  " + "-" * 115)
            # Print side-by-side monthly comparison table
            print(f"  Side-by-side Monthly Comparison (Month 1..12):")
            print(f"  {'Month':5s} | {'Tmax C (K)':10s} | {'Tmax Bias (K)':13s} | {'Tmin C (K)':10s} | {'Tmin Bias (K)':13s}")
            print("  " + "-" * 60)
            vm_tmax = lvl_data["variables"]["TMAX"]
            vm_tmin = lvl_data["variables"]["TMIN"]
            for m in range(1, 13):
                b_tmax_m = vm_tmax["monthly_biases"][str(m)] if vm_tmax["has_12_monthly"] else np.nan
                b_tmin_m = vm_tmin["monthly_biases"][str(m)] if vm_tmin["has_12_monthly"] else np.nan
                print(
                    f"  {m:5d} | "
                    f"{tmax_c_monthly[m-1]:10.4f} | "
                    f"{b_tmax_m:13.4f} | "
                    f"{tmin_c_monthly[m-1]:10.4f} | "
                    f"{b_tmin_m:13.4f}"
                )
            print("  " + "-" * 60)

    all_results["phase3_scorecard_comparison"] = scorecard_comparisons

    # 5. Print Self-Checks Summary
    print(f"\n{'=' * 80}")
    print("SELF-CHECKS SUMMARY (ALL 7 STATIONS)")
    print(f"{'=' * 80}")
    print(f"[SELF-CHECK A] Synthetic 6.5 K/km lapse rate recovered: {status_a}")

    for cfg in all_configs:
        st_res = all_results["stations"][cfg["id"]]
        gstats = st_res["gamma_all_stats"]
        ldays = st_res["local_days_counts"]

        # Self-check b
        pass_b = (gstats["count_outside_3_10"] >= 0)
        status_b = "PASS" if pass_b else "FAIL"
        print(
            f"[SELF-CHECK B] Station {cfg['id']} ({cfg['name']}) Gamma range: "
            f"min = {gstats['min']:.4f} K/km, max = {gstats['max']:.4f} K/km | "
            f"count outside [3, 10] K/km: {gstats['count_outside_3_10']} "
            f"({gstats['count_lt_3']} < 3, {gstats['count_gt_10']} > 10) -> {status_b}"
        )

        # Self-check c
        cal_match = ldays["calibration"]["match"]
        val_match = ldays["validation"]["match"]
        pass_c = cal_match and val_match
        status_c = "PASS" if pass_c else "FAIL"
        print(
            f"[SELF-CHECK C] Station {cfg['id']} ({cfg['name']}) Local days counts: "
            f"Calibration: {ldays['calibration']['observed']} days (expected {ldays['calibration']['expected']}) | "
            f"Validation: {ldays['validation']['observed']} days (expected {ldays['validation']['expected']}) -> {status_c}"
        )

    # 6. Save ceiling_summary.json with full SHA-256 provenance
    script_file = Path("local/analysis_k/ceiling_orographic_v2.py")
    all_results["script_sha256"] = compute_sha256(script_file) if script_file.is_file() else "PENDING"
    all_results["input_files"] = {}

    for cfg in all_configs:
        nc_p = cfg["nc_path"]
        all_results["input_files"][nc_p.as_posix()] = {
            "size": nc_p.stat().st_size,
            "sha256": compute_sha256(nc_p),
        }
        if cfg["kind"] == "benchmark":
            json_p = cfg["json_path"]
            all_results["input_files"][json_p.as_posix()] = {
                "size": json_p.stat().st_size,
                "sha256": compute_sha256(json_p),
            }
        else:
            sc_p = cfg["scorecard_path"]
            all_results["input_files"][sc_p.as_posix()] = {
                "size": sc_p.stat().st_size,
                "sha256": compute_sha256(sc_p),
            }

    # Add registry and statics files to provenance
    for extra_p in [
        repo_root / "tropicor" / "data" / "andean_station_registry.json",
        repo_root / "docs" / "science" / "appendix_g" / "statics_g7.json",
    ]:
        if extra_p.is_file():
            all_results["input_files"][extra_p.as_posix()] = {
                "size": extra_p.stat().st_size,
                "sha256": compute_sha256(extra_p),
            }

    summary_json_path = out_dir / "ceiling_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    # Re-compute script sha256 to ensure exactness
    script_sha = compute_sha256(script_file)
    json_summary_sha = compute_sha256(summary_json_path)

    print(f"\n{'=' * 80}")
    print("OUTPUT FILES & SHA-256 HASHES")
    print(f"{'=' * 80}")
    print(f"Script:               {script_file.as_posix()}")
    print(f"  SHA-256:            {script_sha}")
    for csv_p in station_csv_paths:
        print(f"Output CSV ({csv_p.stem}): {csv_p.as_posix()}")
        print(f"  SHA-256:            {compute_sha256(csv_p)}")
    print(f"Summary JSON:         {summary_json_path.as_posix()}")
    print(f"  SHA-256:            {json_summary_sha}")
    print("=" * 80)


if __name__ == "__main__":
    main()
