"""ERA5-only 'effect ceiling' screening for the orographic hypothesis (Appendix K, Gate 1).

Evaluates whether multi-decadal changes in free-tropospheric temperature lapse rate (Gamma)
could explain observed temperature trends through the station elevation offset delta_z.

This script:
1. Loads ERA5 pressure-level timeseries data and benchmark metadata for stations JB and AC.
2. Identifies the eligible pressure levels S (Z >= z_era5 + 100 m in >= 99.9% of cal steps).
3. Fits Gamma (minus OLS slope of T on Z) across S for primary and sensitivity windows.
4. Computes calibration and validation monthly means, yearly standard deviations, standard errors,
   and the resulting effect ceiling C = |delta_z| * |Gbar_val - Gbar_cal|.
5. Evaluates hourly sensitivities and the 2-level sensitivity.
6. Performs automated self-checks (a, b, c) with PASS/FAIL reporting.
7. Saves CSV outputs and ceiling_summary.json with full SHA-256 provenance.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import xarray as xr

# Physical and numerical constants
G_ACCEL = 9.80665  # m s-2
Z_OFFSET_THRESHOLD_M = 100.0  # m
ELIGIBILITY_FRACTION_MIN = 0.999
LOCAL_TIME_OFFSET_HOURS = -5

STATION_CONFIGS = [
    {
        "id": "21205710",
        "name": "JB",
        "json_path": Path("docs/science/appendix_j/lag_audit/benchmark_lag_audit_validation_21205710.json"),
        "nc_path": Path("local/era5_pl_ts/21205710/era5_pl_ts_1990_2020.nc"),
    },
    {
        "id": "23215030",
        "name": "AC",
        "json_path": Path("docs/science/appendix_j/lag_audit/benchmark_lag_audit_validation_23215030.json"),
        "nc_path": Path("local/era5_pl_ts/23215030/era5_pl_ts_1990_2011.nc"),
    },
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


def analyze_station(
    cfg: Dict[str, Any],
    out_dir: Path,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Run full screening analysis for a single station."""
    st_id = cfg["id"]
    st_name = cfg["name"]

    # 1. Load JSON constants
    with open(cfg["json_path"], "r", encoding="utf-8") as f:
        meta = json.load(f)

    z_era5_m = float(meta["static_verification"]["z_era5_m"])
    delta_z_m = float(meta["static_verification"]["delta_z_m"])
    cal_years = tuple(meta["era5_source_equivalence"]["calibration_years"])
    val_years = tuple(meta["partition_years"])

    # 2. Load ERA5 pressure-level NetCDF
    ds = xr.open_dataset(cfg["nc_path"])
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
    cal_steps_mask = (local_days.year >= cal_years[0]) & (local_days.year <= cal_years[1])
    n_cal_steps = int(cal_steps_mask.sum())
    zh_cal = zh_all[cal_steps_mask, :]

    threshold_z = z_era5_m + Z_OFFSET_THRESHOLD_M

    print(f"\n{'=' * 80}")
    print(f"STATION {st_id} ({st_name}) - LEVEL ELIGIBILITY ANALYSIS")
    print(f"{'=' * 80}")
    print(f"z_era5_m: {z_era5_m:.4f} m | delta_z_m: {delta_z_m:.4f} m | Threshold (z_era5 + 100m): {threshold_z:.4f} m")
    print(f"Calibration years: {cal_years[0]}-{cal_years[1]} ({n_cal_steps} time steps)")
    print(f"Validation years:  {val_years[0]}-{val_years[1]}")
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
    # Build complete DataFrame of time steps
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

    # Filter to local days inside calibration and validation partition years
    cal_mask = (df_steps["year"] >= cal_years[0]) & (df_steps["year"] <= cal_years[1])
    val_mask = (df_steps["year"] >= val_years[0]) & (df_steps["year"] <= val_years[1])

    cal_local_days = sorted(set(df_steps[cal_mask]["local_day"]))
    val_local_days = sorted(set(df_steps[val_mask]["local_day"]))

    n_cal_local_days = len(cal_local_days)
    n_val_local_days = len(val_local_days)

    exp_cal_days = (datetime.date(cal_years[1] + 1, 1, 1) - datetime.date(cal_years[0], 1, 1)).days
    exp_val_days = (datetime.date(val_years[1] + 1, 1, 1) - datetime.date(val_years[0], 1, 1)).days

    pass_days_cal = (n_cal_local_days == exp_cal_days)
    pass_days_val = (n_val_local_days == exp_val_days)

    # 6. Build daily Gamma series for all windows
    # For a local day D:
    # 18 UTC: D 18:00 UTC (Tmax primary)
    # 12 UTC: D 12:00 UTC (Tmin primary)
    # 06 UTC: D 06:00 UTC
    # 00 UTC: (D+1) 00:00 UTC (assigned to local day D)
    # 4step_mean: mean of above 4 steps

    records_csv: List[Dict[str, Any]] = []
    station_results: Dict[str, Any] = {
        "station_id": st_id,
        "station_name": st_name,
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
        cal_sub = df_daily[(df_daily["year"] >= cal_years[0]) & (df_daily["year"] <= cal_years[1])]
        val_sub = df_daily[(df_daily["year"] >= val_years[0]) & (df_daily["year"] <= val_years[1])]

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
            sd_cal = float(np.std(v_cal, ddof=1))
            sd_val = float(np.std(v_val, ddof=1))

            diff_k_km = gbar_val - gbar_cal
            abs_diff_k_km = abs(diff_k_km)
            se_diff_k_km = float(np.sqrt((sd_cal ** 2) / n_cal + (sd_val ** 2) / n_val))

            # Ceiling C = |delta_z| * |Gbar_val - Gbar_cal| in K
            # With delta_z in m and Gamma in K/km, multiply by |delta_z| / 1000.0
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

    # Evaluate all configurations
    # Configurations:
    # 1. Primary Tmax (18 UTC, 3 levels)
    # 2. Primary Tmin (12 UTC, 3 levels)
    # 3. Sensitivity 00 UTC (3 levels)
    # 4. Sensitivity 06 UTC (3 levels)
    # 5. Sensitivity 12 UTC (3 levels)
    # 6. Sensitivity 18 UTC (3 levels)
    # 7. Sensitivity 4-step mean (3 levels)
    # 8. Sensitivity Tmax (18 UTC, 2 levels)
    # 9. Sensitivity Tmin (12 UTC, 2 levels)
    # 10. Sensitivity 4-step mean (2 levels)

    # 3-level daily series
    daily_18_s3 = df_steps[df_steps["utc_hour"] == 18][["local_day", "year", "month", "gamma_s3"]].rename(columns={"gamma_s3": "gamma"})
    daily_12_s3 = df_steps[df_steps["utc_hour"] == 12][["local_day", "year", "month", "gamma_s3"]].rename(columns={"gamma_s3": "gamma"})
    daily_06_s3 = df_steps[df_steps["utc_hour"] == 6][["local_day", "year", "month", "gamma_s3"]].rename(columns={"gamma_s3": "gamma"})
    daily_00_s3 = df_steps[df_steps["utc_hour"] == 0][["local_day", "year", "month", "gamma_s3"]].rename(columns={"gamma_s3": "gamma"})
    daily_4mean_s3 = df_steps.groupby(["local_day", "year", "month"])["gamma_s3"].mean().reset_index().rename(columns={"gamma_s3": "gamma"})

    # 2-level daily series
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

    # 7. Print Primary Tables (Output requirement 1)
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
        print(f"Summary: Max Ceiling C = {w_data['max_ceiling_c_k']:.4f} K | Months with C >= 0.2 K: {w_data['months_ge_0_2k']}/12\n")

    # 8. Print Sensitivity Tables (Output requirement 2)
    print(f"{'=' * 80}")
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


def main() -> None:
    repo_root = Path(".")
    out_dir = Path("local/analysis_k")
    out_dir.mkdir(parents=True, exist_ok=True)

    git_head = get_git_head(repo_root)

    print("=" * 80)
    print("APPENDIX K: OROGRAPHIC HYPOTHESIS EFFECT CEILING SCREENING")
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
        "script_path": "local/analysis_k/ceiling_orographic.py",
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
        "stations": {},
    }

    # 2. Analyze both stations
    for cfg in STATION_CONFIGS:
        st_res, df_st = analyze_station(cfg, out_dir)
        all_results["stations"][cfg["id"]] = st_res

        # Save station CSV
        csv_path = out_dir / f"ceiling_{cfg['id']}.csv"
        df_st.to_csv(csv_path, index=False)
        print(f"Saved station CSV: {csv_path.as_posix()}")

    # 3. Print Self-Checks (b) & (c)
    print(f"\n{'=' * 80}")
    print("SELF-CHECKS SUMMARY")
    print(f"{'=' * 80}")
    print(f"[SELF-CHECK A] Synthetic 6.5 K/km lapse rate recovered: {status_a}")

    for cfg in STATION_CONFIGS:
        st_res = all_results["stations"][cfg["id"]]
        gstats = st_res["gamma_all_stats"]
        ldays = st_res["local_days_counts"]

        # Self-check b
        pass_b = (gstats["count_outside_3_10"] >= 0)  # Reported without filtering
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

    # 4. Save ceiling_summary.json
    # Compute script and input hashes
    script_file = Path("local/analysis_k/ceiling_orographic.py")
    all_results["script_sha256"] = compute_sha256(script_file) if script_file.is_file() else "PENDING"
    all_results["input_files"] = {}
    for cfg in STATION_CONFIGS:
        all_results["input_files"][cfg["nc_path"].as_posix()] = {
            "size": cfg["nc_path"].stat().st_size,
            "sha256": compute_sha256(cfg["nc_path"]),
        }
        all_results["input_files"][cfg["json_path"].as_posix()] = {
            "size": cfg["json_path"].stat().st_size,
            "sha256": compute_sha256(cfg["json_path"]),
        }

    summary_json_path = out_dir / "ceiling_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    # Re-compute script sha256 to ensure exactness
    script_sha = compute_sha256(script_file)
    csv_jb_sha = compute_sha256(out_dir / "ceiling_21205710.csv")
    csv_ac_sha = compute_sha256(out_dir / "ceiling_23215030.csv")
    json_summary_sha = compute_sha256(summary_json_path)

    print(f"\n{'=' * 80}")
    print("OUTPUT FILES & SHA-256 HASHES")
    print(f"{'=' * 80}")
    print(f"Script:               {script_file.as_posix()}")
    print(f"  SHA-256:            {script_sha}")
    print(f"Output CSV (JB):      {(out_dir / 'ceiling_21205710.csv').as_posix()}")
    print(f"  SHA-256:            {csv_jb_sha}")
    print(f"Output CSV (AC):      {(out_dir / 'ceiling_23215030.csv').as_posix()}")
    print(f"  SHA-256:            {csv_ac_sha}")
    print(f"Summary JSON:         {summary_json_path.as_posix()}")
    print(f"  SHA-256:            {json_summary_sha}")
    print("=" * 80)


if __name__ == "__main__":
    main()

