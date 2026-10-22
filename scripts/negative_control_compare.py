"""Negative control: compare ERA5 annual files between two distinct stations (JB vs AC).

Compares:
- Station 21205710 (Jardín Botánico, Bogotá)
- Station 23215030 (Aguas Claras, Cesar)

Using the exact same comparison logic, metrics, and thresholds as scripts/compare_era5_sources.py:
- 2m_temperature (1990-2011)
- total_precipitation (1975-2001)

Expected result: Complete failure with massive differences (negative control).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from tropicor.io.era5 import (
    ERA5Adapter,
    aggregate_hourly_to_daily_precipitation,
    aggregate_hourly_to_daily_temperature,
)
from tropicor.io.stations import StationCatalog

THRESHOLDS = {
    "2m_temperature": {
        "hourly_max_abs_diff": 0.01,  # K
        "daily_tmax_max_abs_diff": 0.01,  # °C
        "daily_tmin_max_abs_diff": 0.01,  # °C
    },
    "total_precipitation": {
        "hourly_max_abs_diff": 1e-5,  # m (0.01 mm)
        "daily_precip_max_abs_diff": 0.01,  # mm
    },
}


def run_negative_control() -> Dict[str, Any]:
    catalog = StationCatalog.from_benchmark()
    jb_meta = catalog.get("21205710")
    ac_meta = catalog.get("23215030")

    if jb_meta is None or ac_meta is None:
        raise ValueError("Station metadata not found for JB or AC.")

    lat_jb = round(jb_meta.latitude * 4.0) / 4.0
    lon_jb = round(jb_meta.longitude * 4.0) / 4.0
    lat_ac = round(ac_meta.latitude * 4.0) / 4.0
    lon_ac = round(ac_meta.longitude * 4.0) / 4.0

    jb_dir = Path("local/era5_hourly/21205710")
    ac_dir = Path("local/era5_hourly/23215030")

    results: Dict[str, Any] = {
        "test_type": "negative_control_comparison",
        "station_1": {
            "code": "21205710",
            "name": jb_meta.name,
            "center": [lat_jb, lon_jb],
        },
        "station_2": {
            "code": "23215030",
            "name": ac_meta.name,
            "center": [lat_ac, lon_ac],
        },
        "variables": {},
        "overall_status": "PENDING",
    }

    # 1. Temperature: 1990 to 2011
    years_temp = list(range(1990, 2012))
    files_jb_t = [jb_dir / f"era5_2m_temperature_{y}.nc" for y in years_temp]
    files_ac_t = [ac_dir / f"era5_2m_temperature_{y}.nc" for y in years_temp]

    adapter_jb_t = ERA5Adapter(files_jb_t)
    adapter_ac_t = ERA5Adapter(files_ac_t)

    pt_jb_t = adapter_jb_t.extract_point(lat_jb, lon_jb, method="nearest")
    pt_ac_t = adapter_ac_t.extract_point(lat_ac, lon_ac, method="nearest")

    s_jb_t = pt_jb_t["t2m"].to_series()
    s_ac_t = pt_ac_t["t2m"].to_series()

    idx_t = s_jb_t.index.intersection(s_ac_t.index)
    s_jb_t = s_jb_t.loc[idx_t]
    s_ac_t = s_ac_t.loc[idx_t]

    diff_hourly_t = (s_jb_t - s_ac_t).abs()
    max_diff_t = float(diff_hourly_t.max())
    mean_diff_t = float(diff_hourly_t.mean())
    n_hours_t = len(idx_t)
    hours_above_t = int(
        (diff_hourly_t > THRESHOLDS["2m_temperature"]["hourly_max_abs_diff"]).sum()
    )

    # Daily temp aggregation
    df_daily_jb_t = aggregate_hourly_to_daily_temperature(
        s_jb_t, window_start_utc=5, min_hours=24, lag_days=0
    )
    df_daily_ac_t = aggregate_hourly_to_daily_temperature(
        s_ac_t, window_start_utc=5, min_hours=24, lag_days=0
    )
    common_days_t = df_daily_jb_t.dropna().index.intersection(
        df_daily_ac_t.dropna().index
    )

    tmax_diff = (
        df_daily_jb_t.loc[common_days_t, "tmax"]
        - df_daily_ac_t.loc[common_days_t, "tmax"]
    ).abs()
    tmin_diff = (
        df_daily_jb_t.loc[common_days_t, "tmin"]
        - df_daily_ac_t.loc[common_days_t, "tmin"]
    ).abs()

    max_tmax_diff = float(tmax_diff.max())
    max_tmin_diff = float(tmin_diff.max())
    days_tmax_above = int(
        (tmax_diff > THRESHOLDS["2m_temperature"]["daily_tmax_max_abs_diff"]).sum()
    )
    days_tmin_above = int(
        (tmin_diff > THRESHOLDS["2m_temperature"]["daily_tmin_max_abs_diff"]).sum()
    )

    temp_passed = (
        (hours_above_t == 0) and (days_tmax_above == 0) and (days_tmin_above == 0)
    )

    results["variables"]["2m_temperature"] = {
        "period": "1990-2011",
        "n_hours_compared": n_hours_t,
        "max_hourly_abs_diff_K": max_diff_t,
        "mean_hourly_abs_diff_K": mean_diff_t,
        "hourly_threshold_K": THRESHOLDS["2m_temperature"]["hourly_max_abs_diff"],
        "hours_above_threshold": hours_above_t,
        "pct_hours_above_threshold": round(hours_above_t / n_hours_t * 100.0, 2),
        "n_days_compared": len(common_days_t),
        "max_daily_tmax_diff_C": max_tmax_diff,
        "max_daily_tmin_diff_C": max_tmin_diff,
        "days_tmax_above_threshold": days_tmax_above,
        "days_tmin_above_threshold": days_tmin_above,
        "passed": temp_passed,
    }

    # 2. Precipitation: 1975 to 2001
    years_pr = list(range(1975, 2002))
    files_jb_p = [jb_dir / f"era5_total_precipitation_{y}.nc" for y in years_pr]
    files_ac_p = [ac_dir / f"era5_total_precipitation_{y}.nc" for y in years_pr]

    adapter_jb_p = ERA5Adapter(files_jb_p)
    adapter_ac_p = ERA5Adapter(files_ac_p)

    pt_jb_p = adapter_jb_p.extract_point(lat_jb, lon_jb, method="nearest")
    pt_ac_p = adapter_ac_p.extract_point(lat_ac, lon_ac, method="nearest")

    s_jb_p = pt_jb_p["tp"].to_series()
    s_ac_p = pt_ac_p["tp"].to_series()

    idx_p = s_jb_p.index.intersection(s_ac_p.index)
    s_jb_p = s_jb_p.loc[idx_p]
    s_ac_p = s_ac_p.loc[idx_p]

    diff_hourly_p = (s_jb_p - s_ac_p).abs()
    max_diff_p = float(diff_hourly_p.max())
    mean_diff_p = float(diff_hourly_p.mean())
    n_hours_p = len(idx_p)
    hours_above_p = int(
        (diff_hourly_p > THRESHOLDS["total_precipitation"]["hourly_max_abs_diff"]).sum()
    )

    # Convert m to mm for daily precipitation
    s_jb_p_mm = s_jb_p * 1000.0
    s_ac_p_mm = s_ac_p * 1000.0

    s_daily_jb_p = aggregate_hourly_to_daily_precipitation(
        s_jb_p_mm, window_start_utc=12, min_hours=24, lag_days=0
    )
    s_daily_ac_p = aggregate_hourly_to_daily_precipitation(
        s_ac_p_mm, window_start_utc=12, min_hours=24, lag_days=0
    )
    common_days_p = s_daily_jb_p.dropna().index.intersection(
        s_daily_ac_p.dropna().index
    )

    tp_diff = (s_daily_jb_p.loc[common_days_p] - s_daily_ac_p.loc[common_days_p]).abs()
    max_tp_diff = float(tp_diff.max())
    days_tp_above = int(
        (tp_diff > THRESHOLDS["total_precipitation"]["daily_precip_max_abs_diff"]).sum()
    )

    pr_passed = (hours_above_p == 0) and (days_tp_above == 0)

    results["variables"]["total_precipitation"] = {
        "period": "1975-2001",
        "n_hours_compared": n_hours_p,
        "max_hourly_abs_diff_m": max_diff_p,
        "max_hourly_abs_diff_mm": max_diff_p * 1000.0,
        "mean_hourly_abs_diff_m": mean_diff_p,
        "hourly_threshold_m": THRESHOLDS["total_precipitation"]["hourly_max_abs_diff"],
        "hours_above_threshold": hours_above_p,
        "pct_hours_above_threshold": round(hours_above_p / n_hours_p * 100.0, 2),
        "n_days_compared": len(common_days_p),
        "max_daily_precip_diff_mm": max_tp_diff,
        "days_precip_above_threshold": days_tp_above,
        "passed": pr_passed,
    }

    # Negative control expects both to FAIL
    if not temp_passed and not pr_passed:
        results["overall_status"] = "FAILED_AS_EXPECTED_NEGATIVE_CONTROL"
    else:
        results["overall_status"] = "UNEXPECTED_PASS_DEFECTIVE_COMPARATOR"

    out_p = Path("local/reports/negative_control_comparison.json")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(results, indent=2), encoding="utf-8")

    return results


if __name__ == "__main__":
    res = run_negative_control()
    print("=" * 80)
    print("NEGATIVE CONTROL COMPARISON REPORT (JB vs AC)")
    print("=" * 80)
    print(f"Overall status: {res['overall_status']}")
    print("-" * 80)
    t_res = res["variables"]["2m_temperature"]
    print(f"Variable: 2m_temperature ({t_res['period']})")
    print(f"  Hours compared: {t_res['n_hours_compared']:,}")
    print(f"  Max hourly difference: {t_res['max_hourly_abs_diff_K']:.4f} K")
    print(f"  Mean hourly difference: {t_res['mean_hourly_abs_diff_K']:.4f} K")
    print(
        f"  Hours above threshold (> 0.01 K): {t_res['hours_above_threshold']:,} ({t_res['pct_hours_above_threshold']}%)"
    )
    print(f"  Max daily Tmax difference: {t_res['max_daily_tmax_diff_C']:.4f} °C")
    print(f"  Passed equivalence: {t_res['passed']} (Expected False)")
    print("-" * 80)
    p_res = res["variables"]["total_precipitation"]
    print(f"Variable: total_precipitation ({p_res['period']})")
    print(f"  Hours compared: {p_res['n_hours_compared']:,}")
    print(
        f"  Max hourly difference: {p_res['max_hourly_abs_diff_mm']:.4f} mm ({p_res['max_hourly_abs_diff_m']:.6e} m)"
    )
    print(
        f"  Hours above threshold (> 1e-5 m): {p_res['hours_above_threshold']:,} ({p_res['pct_hours_above_threshold']}%)"
    )
    print(
        f"  Max daily precipitation difference: {p_res['max_daily_precip_diff_mm']:.4f} mm"
    )
    print(f"  Passed equivalence: {p_res['passed']} (Expected False)")
    print("=" * 80)
