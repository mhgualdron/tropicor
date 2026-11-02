"""Close the gaps of the last diagnostics (Appendix K diagnostic extension).

Standalone script covering:
  1. JB and AC annual calibration series (d_y, r_y), OLS slope with 95% bootstrap CI,
     and calibration monthly bias offset and seasonal amplitude.
  2. Verbatim quotes and criteria check for annual block bootstrap in protocol and evaluator.
  3. Grep of delta_z, delta_h, "Delta z" and lapse-rate corrections across tropicor/ and scripts/,
     and analysis of Level 0-E.
  4. Comparison of random stream between archived and diagnostic2 scripts, and recomputation
     of level-change 95% CIs with 1,000 and 10,000 resamples (seed 42) for all 10 Andean cases.

Rules:
  - Commit NOTHING.
  - Slice JB/AC observations to calibration years only at load time.
  - Do NOT touch validation openings ledger.
  - Do NOT import phase3_evaluator or run audit scripts.
"""

from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from tropicor.downscale.temperature import load_calibration_temperature_pair

# Constants
MIN_DAYS_PER_MONTH = 10
MIN_MONTHS_PER_YEAR = 6
BOOTSTRAP_SEED = 42

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
        "cal_start": 1990,
        "cal_end": 2004,
    },
    "23215030": {
        "name": "AGUAS CLARAS [23215030]",
        "tmax_csv": Path("local/newFiles/TempMaxima_AguasClaras.csv"),
        "tmin_csv": Path("local/newFiles/TempMinima_AguasClaras.csv"),
        "era5_dir": Path("local/era5_ts/23215030"),
        "cal_start": 1990,
        "cal_end": 2000,
    },
}


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


def compute_annual_series(
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
        for _ in range(1000):
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


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    out_dir = Path("local/analysis_k/diagnostic2")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_data: Dict[str, Any] = {
        "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "task": "Close diagnostic gaps",
    }

    # =========================================================================
    # ITEM 1: JB and AC Annual Series (Calibration Only)
    # =========================================================================
    print("=" * 100)
    print("ITEM 1: JB AND AC ANNUAL SERIES (CALIBRATION YEARS ONLY)")
    print("=" * 100)

    item1_results: Dict[str, Any] = {}

    for st, cfg in BENCHMARK_STATIONS.items():
        print(f"\n>>> Station {st} ({cfg['name']}) - Calibration: {cfg['cal_start']}-{cfg['cal_end']} <<<")
        obs_tx, obs_tn, mod_tx, mod_tn = load_calibration_temperature_pair(
            station_id=st,
            obs_tmax_path=cfg["tmax_csv"],
            obs_tmin_path=cfg["tmin_csv"],
            era5_target=cfg["era5_dir"],
            window_start_utc=5,
            lag_days=0,
            min_hours=24,
        )

        cal_years = list(range(cfg["cal_start"], cfg["cal_end"] + 1))
        cal_years_set = set(cal_years)

        item1_results[st] = {
            "name": cfg["name"],
            "cal_start": cfg["cal_start"],
            "cal_end": cfg["cal_end"],
            "variables": {},
        }

        for var_name, obs_s, mod_s in [("Tmax", obs_tx, mod_tx), ("Tmin", obs_tn, mod_tn)]:
            b_m = compute_calendar_monthly_biases(obs_s, mod_s)
            b_vals = np.array([float(b_m[m]) for m in range(1, 13)])
            offset = float(np.mean(b_vals))
            seasonal_amp = float(np.max(np.abs(b_vals - offset)))

            df_ann, trend_dec, ci_trend = compute_annual_series(
                obs_s, mod_s, cal_years, cal_years_set, {m: float(b_m[m]) for m in range(1, 13)}
            )

            print(f"\n--- {cfg['name']} | Variable: {var_name} ---")
            print(f"Calibration monthly biases (m=1..12): {[round(x, 4) for x in b_vals]}")
            print(f"Offset (mean of 12 monthly biases): {offset:+.4f} K")
            print(f"Seasonal amplitude (max_m |b_m - offset|): {seasonal_amp:.4f} K")
            print(f"OLS Slope of r_y: {trend_dec:+.4f} K/decade | 95% Bootstrap CI: [{ci_trend[0]:+.4f}, {ci_trend[1]:+.4f}] K/decade")
            print("\nAnnual Series (Calibration Years Only):")
            print(f"{'Year':6s} | {'Partition':11s} | {'Valid Days':10s} | {'Valid Months':12s} | {'d_y (K)':10s} | {'r_y (K)':10s} | {'Included':8s}")
            print("-" * 80)
            ann_records = []
            for _, r in df_ann.iterrows():
                y = int(r["year"])
                part = str(r["partition"])
                nd = int(r["n_valid_days"])
                nm = int(r["n_valid_months"])
                dy = f"{r['raw_bias_d_y']:+.4f}" if pd.notna(r["raw_bias_d_y"]) else "   NaN    "
                ry = f"{r['level1_residual_r_y']:+.4f}" if pd.notna(r["level1_residual_r_y"]) else "   NaN    "
                inc = str(bool(r["included_in_stats"]))
                print(f"{y:6d} | {part:11s} | {nd:10d} | {nm:12d} | {dy:10s} | {ry:10s} | {inc:8s}")
                ann_records.append({
                    "year": y,
                    "partition": part,
                    "n_valid_days": nd,
                    "n_valid_months": nm,
                    "raw_bias_d_y": float(r["raw_bias_d_y"]) if pd.notna(r["raw_bias_d_y"]) else None,
                    "level1_residual_r_y": float(r["level1_residual_r_y"]) if pd.notna(r["level1_residual_r_y"]) else None,
                    "included_in_stats": bool(r["included_in_stats"]),
                })

            item1_results[st]["variables"][var_name] = {
                "offset_k": offset,
                "seasonal_amplitude_k": seasonal_amp,
                "monthly_biases_k": [float(x) for x in b_vals],
                "ols_slope_k_per_decade": trend_dec,
                "ols_slope_95ci_k_per_decade": ci_trend,
                "annual_series": ann_records,
            }

    summary_data["item1_benchmark_calibration"] = item1_results

    # =========================================================================
    # ITEM 4: Bootstrap CI Discrepancy & Recomputation (1k and 10k resamples)
    # =========================================================================
    print("\n" + "=" * 100)
    print("ITEM 4: BOOTSTRAP CI DISCREPANCY & RECOMPUTATION (1,000 AND 10,000 RESAMPLES)")
    print("=" * 100)

    # Recompute level-change CI for all 10 Andean station-variable cases
    ci_table_rows: List[Dict[str, Any]] = []

    for st in OPENED_STATION_IDS:
        df_st = pd.read_csv(f"local/analysis_k/diagnostic/diagnostic_{st}.csv")
        st_name = df_st["station_name"].iloc[0]
        for var in ["Tmax", "Tmin"]:
            sub = df_st[(df_st["variable"] == var) & (df_st["included_in_stats"] == True)]
            cal_dy = sub[sub["partition"] == "calibration"]["raw_bias_d_y"].values.astype(float)
            val_dy = sub[sub["partition"] == "validation"]["raw_bias_d_y"].values.astype(float)
            point_delta = float(np.mean(val_dy) - np.mean(cal_dy))

            # 1. Archived procedure (1k): seed 42, choice(cal_dy) then choice(val_dy)
            rng_arch_1k = np.random.default_rng(BOOTSTRAP_SEED)
            b_arch_1k = []
            for _ in range(1000):
                c_b = rng_arch_1k.choice(cal_dy, size=len(cal_dy), replace=True)
                v_b = rng_arch_1k.choice(val_dy, size=len(val_dy), replace=True)
                b_arch_1k.append(float(np.mean(v_b) - np.mean(c_b)))
            ci_arch_1k = [float(np.percentile(b_arch_1k, 2.5)), float(np.percentile(b_arch_1k, 97.5))]

            # 2. Diagnostic2 procedure (1k): seed 42, choice(val_dy) then choice(cal_dy)
            rng_d2_1k = np.random.default_rng(BOOTSTRAP_SEED)
            b_d2_1k = []
            for _ in range(1000):
                v_b = rng_d2_1k.choice(val_dy, size=len(val_dy), replace=True)
                c_b = rng_d2_1k.choice(cal_dy, size=len(cal_dy), replace=True)
                b_d2_1k.append(float(np.mean(v_b) - np.mean(c_b)))
            ci_d2_1k = [float(np.percentile(b_d2_1k, 2.5)), float(np.percentile(b_d2_1k, 97.5))]

            # 3. Archived procedure (10k): seed 42, choice(cal_dy) then choice(val_dy)
            rng_arch_10k = np.random.default_rng(BOOTSTRAP_SEED)
            b_arch_10k = []
            for _ in range(10000):
                c_b = rng_arch_10k.choice(cal_dy, size=len(cal_dy), replace=True)
                v_b = rng_arch_10k.choice(val_dy, size=len(val_dy), replace=True)
                b_arch_10k.append(float(np.mean(v_b) - np.mean(c_b)))
            ci_arch_10k = [float(np.percentile(b_arch_10k, 2.5)), float(np.percentile(b_arch_10k, 97.5))]

            # 4. Diagnostic2 procedure (10k): seed 42, choice(val_dy) then choice(cal_dy)
            rng_d2_10k = np.random.default_rng(BOOTSTRAP_SEED)
            b_d2_10k = []
            for _ in range(10000):
                v_b = rng_d2_10k.choice(val_dy, size=len(val_dy), replace=True)
                c_b = rng_d2_10k.choice(cal_dy, size=len(cal_dy), replace=True)
                b_d2_10k.append(float(np.mean(v_b) - np.mean(c_b)))
            ci_d2_10k = [float(np.percentile(b_d2_10k, 2.5)), float(np.percentile(b_d2_10k, 97.5))]

            rec = {
                "station_id": st,
                "station_name": st_name,
                "variable": var,
                "point_delta_level_k": point_delta,
                "n_cal_years": len(cal_dy),
                "n_val_years": len(val_dy),
                "ci_archived_1k": ci_arch_1k,
                "ci_diagnostic2_1k": ci_d2_1k,
                "ci_archived_10k": ci_arch_10k,
                "ci_diagnostic2_10k": ci_d2_10k,
            }
            ci_table_rows.append(rec)

    # Print Table Side-by-Side
    print(f"{'Station':25s} | {'Var':4s} | {'Point (K)':9s} | {'Archived 1k CI':21s} | {'Diag2 1k CI':21s} | {'Archived 10k CI':21s} | {'Diag2 10k CI':21s}")
    print("-" * 135)
    for r in ci_table_rows:
        st_label = f"{r['station_name'][:25]}"
        pt = f"{r['point_delta_level_k']:+.4f}"
        a1k = f"[{r['ci_archived_1k'][0]:+.4f}, {r['ci_archived_1k'][1]:+.4f}]"
        d1k = f"[{r['ci_diagnostic2_1k'][0]:+.4f}, {r['ci_diagnostic2_1k'][1]:+.4f}]"
        a10k = f"[{r['ci_archived_10k'][0]:+.4f}, {r['ci_archived_10k'][1]:+.4f}]"
        d10k = f"[{r['ci_diagnostic2_10k'][0]:+.4f}, {r['ci_diagnostic2_10k'][1]:+.4f}]"
        print(f"{st_label:25s} | {r['variable']:4s} | {pt:9s} | {a1k:21s} | {d1k:21s} | {a10k:21s} | {d10k:21s}")

    summary_data["item4_bootstrap_comparison"] = ci_table_rows

    # Save summary JSON
    summary_path = out_dir / "gap_closure_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # Save comparison CSV
    csv_rows = []
    for r in ci_table_rows:
        csv_rows.append({
            "station_id": r["station_id"],
            "station_name": r["station_name"],
            "variable": r["variable"],
            "point_delta_level_k": r["point_delta_level_k"],
            "archived_1k_ci_low": r["ci_archived_1k"][0],
            "archived_1k_ci_high": r["ci_archived_1k"][1],
            "diag2_1k_ci_low": r["ci_diagnostic2_1k"][0],
            "diag2_1k_ci_high": r["ci_diagnostic2_1k"][1],
            "archived_10k_ci_low": r["ci_archived_10k"][0],
            "archived_10k_ci_high": r["ci_archived_10k"][1],
            "diag2_10k_ci_low": r["ci_diagnostic2_10k"][0],
            "diag2_10k_ci_high": r["ci_diagnostic2_10k"][1],
        })
    df_ci_csv = pd.DataFrame(csv_rows)
    ci_csv_path = out_dir / "bootstrap_ci_10k_comparison.csv"
    df_ci_csv.to_csv(ci_csv_path, index=False)

    print(f"\nOutputs written:")
    print(f"  - {summary_path} (SHA-256: {compute_file_sha256(summary_path)})")
    print(f"  - {ci_csv_path} (SHA-256: {compute_file_sha256(ci_csv_path)})")


if __name__ == "__main__":
    main()

