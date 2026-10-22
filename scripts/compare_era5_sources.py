"""Verification and numerical equivalence comparison between single-levels and timeseries ERA5 sources.

Compares hourly and daily metrics between:
- local/era5_hourly/<station>/ (CDS reanalysis-era5-single-levels)
- local/era5_ts/<station>/     (CDS reanalysis-era5-single-levels-timeseries)

Pre-fixed equivalence criteria:
- 2m_temperature:
    hourly max |dif| <= 0.01 K
    daily Tmax and Tmin |dif| <= 0.01 °C
- total_precipitation:
    hourly max |dif| <= 1e-5 m (0.01 mm)
    daily precipitation |dif| <= 0.01 mm
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

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


def find_available_years(directory: Path, variable: str) -> List[int]:
    """Extract list of sorted years available for a given variable in a directory."""
    files = list(directory.glob(f"era5_{variable}_*.nc"))
    years = []
    for f in files:
        m = re.search(r"(?<!\d)(\d{4})(?!\d)", f.name)
        if m:
            years.append(int(m.group(1)))
    return sorted(years)


def compare_single_station_variable(
    station_code: str,
    variable: str,
    base_sl_dir: Path = Path("local/era5_hourly"),
    base_ts_dir: Path = Path("local/era5_ts"),
) -> Dict[str, Any]:
    """Compare single-levels vs timeseries for a given station and variable."""
    catalog = StationCatalog.from_benchmark()
    st_meta = catalog.get(station_code)
    if st_meta is None:
        raise ValueError(f"Station '{station_code}' not found in benchmark catalog.")

    lat_center = round(st_meta.latitude * 4.0) / 4.0
    lon_center = round(st_meta.longitude * 4.0) / 4.0

    sl_dir = base_sl_dir / station_code
    ts_dir = base_ts_dir / station_code

    years_sl = set(find_available_years(sl_dir, variable))
    years_ts = set(find_available_years(ts_dir, variable))

    # Common continuous range to compare
    common_years = sorted(years_sl.intersection(years_ts))
    if not common_years:
        raise ValueError(
            f"No common years found between single-levels ({sorted(years_sl)}) "
            f"and timeseries ({sorted(years_ts)}) for station {station_code}, variable {variable}."
        )

    # Note: if a boundary year (like end_year+1) exists with only 24 hours in timeseries,
    # we evaluate full years where single-levels also has full hours.
    # Exclude boundary year from full-year comparison if single-levels does not have it.
    eval_years = [
        y
        for y in common_years
        if (sl_dir / f"era5_{variable}_{y}.nc").stat().st_size > 50000
    ]

    sl_files = [sl_dir / f"era5_{variable}_{y}.nc" for y in eval_years]
    ts_files = [ts_dir / f"era5_{variable}_{y}.nc" for y in eval_years]

    adapter_sl = ERA5Adapter(sl_files)
    adapter_ts = ERA5Adapter(ts_files)

    # 4.a: Same cell (lat/lon) in both sources
    pt_sl = adapter_sl.extract_point(lat_center, lon_center, method="nearest")
    pt_ts = adapter_ts.extract_point(lat_center, lon_center, method="nearest")

    lat_sl = float(pt_sl.latitude.values)
    lon_sl = float(pt_sl.longitude.values)
    lat_ts = float(pt_ts.latitude.values)
    lon_ts = float(pt_ts.longitude.values)

    if abs(lat_sl - lat_center) > 1e-6 or abs(lat_ts - lat_center) > 1e-6:
        raise ValueError(
            f"Latitude mismatch with center {lat_center}: sl={lat_sl}, ts={lat_ts}"
        )
    if abs(lon_sl - lon_center) > 1e-6 or abs(lon_ts - lon_center) > 1e-6:
        raise ValueError(
            f"Longitude mismatch with center {lon_center}: sl={lon_sl}, ts={lon_ts}"
        )

    # 4.e: Units and dtypes of both sides
    var_alias = "t2m" if variable == "2m_temperature" else "tp"
    da_sl = pt_sl[var_alias]
    da_ts = pt_ts[var_alias]

    units_sl = str(da_sl.attrs.get("units", "")).strip()
    units_ts = str(da_ts.attrs.get("units", "")).strip()
    dtype_sl = str(da_sl.dtype)
    dtype_ts = str(da_ts.dtype)

    # 4.b: Same hourly grid, same number of hours, zero NaNs
    s_sl = da_sl.to_series()
    s_ts = da_ts.to_series()

    # Align to common index
    common_idx = s_sl.index.intersection(s_ts.index)
    s_sl = s_sl.loc[common_idx]
    s_ts = s_ts.loc[common_idx]

    n_hours = len(common_idx)
    nan_sl = int(s_sl.isnull().sum())
    nan_ts = int(s_ts.isnull().sum())

    if nan_sl > 0 or nan_ts > 0:
        raise ValueError(f"NaNs present in hourly series: sl={nan_sl}, ts={nan_ts}")

    # 4.c: Hourly difference
    diff_hourly = (s_sl - s_ts).abs()
    mean_diff_hourly = float(diff_hourly.mean())
    max_diff_hourly = float(diff_hourly.max())
    p999_diff_hourly = float(np.percentile(np.asarray(diff_hourly), 99.9))

    hourly_thresh = THRESHOLDS[variable]["hourly_max_abs_diff"]
    hours_above_threshold = int((diff_hourly > hourly_thresh).sum())

    # 4.d: Daily aggregation difference
    daily_report: Dict[str, Any] = {}
    if variable == "2m_temperature":
        df_daily_sl = aggregate_hourly_to_daily_temperature(
            s_sl, window_start_utc=5, min_hours=24, lag_days=0
        )
        df_daily_ts = aggregate_hourly_to_daily_temperature(
            s_ts, window_start_utc=5, min_hours=24, lag_days=0
        )
        common_days = df_daily_sl.dropna().index.intersection(
            df_daily_ts.dropna().index
        )
        tmax_diff = (
            df_daily_sl.loc[common_days, "tmax"] - df_daily_ts.loc[common_days, "tmax"]
        ).abs()
        tmin_diff = (
            df_daily_sl.loc[common_days, "tmin"] - df_daily_ts.loc[common_days, "tmin"]
        ).abs()
        tmean_diff = (
            df_daily_sl.loc[common_days, "tmean"]
            - df_daily_ts.loc[common_days, "tmean"]
        ).abs()

        max_tmax_diff = float(tmax_diff.max())
        mean_tmax_diff = float(tmax_diff.mean())
        max_tmin_diff = float(tmin_diff.max())
        mean_tmin_diff = float(tmin_diff.mean())
        max_tmean_diff = float(tmean_diff.max())

        daily_report = {
            "n_days": len(common_days),
            "max_abs_diff_tmax": max_tmax_diff,
            "mean_abs_diff_tmax": mean_tmax_diff,
            "max_abs_diff_tmin": max_tmin_diff,
            "mean_abs_diff_tmin": mean_tmin_diff,
            "max_abs_diff_tmean": max_tmean_diff,
            "daily_tmax_threshold": THRESHOLDS[variable]["daily_tmax_max_abs_diff"],
            "daily_tmin_threshold": THRESHOLDS[variable]["daily_tmin_max_abs_diff"],
            "days_tmax_above_threshold": int(
                (tmax_diff > THRESHOLDS[variable]["daily_tmax_max_abs_diff"]).sum()
            ),
            "days_tmin_above_threshold": int(
                (tmin_diff > THRESHOLDS[variable]["daily_tmin_max_abs_diff"]).sum()
            ),
        }
    else:  # total_precipitation
        # In single-levels, raw tp is in meters. aggregate_hourly_to_daily_precipitation expects mm.
        s_sl_mm = s_sl * 1000.0 if units_sl in ("m", "") else s_sl
        s_ts_mm = s_ts * 1000.0 if units_ts in ("m", "") else s_ts

        s_daily_sl = aggregate_hourly_to_daily_precipitation(
            s_sl_mm, window_start_utc=12, min_hours=24, lag_days=0
        )
        s_daily_ts = aggregate_hourly_to_daily_precipitation(
            s_ts_mm, window_start_utc=12, min_hours=24, lag_days=0
        )
        common_days = s_daily_sl.dropna().index.intersection(s_daily_ts.dropna().index)
        tp_diff = (s_daily_sl.loc[common_days] - s_daily_ts.loc[common_days]).abs()

        max_tp_diff = float(tp_diff.max())
        mean_tp_diff = float(tp_diff.mean())
        p999_tp_diff = float(np.percentile(np.asarray(tp_diff), 99.9))

        daily_report = {
            "n_days": len(common_days),
            "max_abs_diff_mm": max_tp_diff,
            "mean_abs_diff_mm": mean_tp_diff,
            "p999_abs_diff_mm": p999_tp_diff,
            "daily_threshold_mm": THRESHOLDS[variable]["daily_precip_max_abs_diff"],
            "days_above_threshold": int(
                (tp_diff > THRESHOLDS[variable]["daily_precip_max_abs_diff"]).sum()
            ),
        }

    # Verify pre-fixed equivalence criteria
    passed = True
    failure_reasons = []

    if max_diff_hourly > hourly_thresh:
        passed = False
        failure_reasons.append(
            f"Hourly max |diff| {max_diff_hourly} > threshold {hourly_thresh}"
        )

    if variable == "2m_temperature":
        if (
            daily_report["max_abs_diff_tmax"]
            > THRESHOLDS[variable]["daily_tmax_max_abs_diff"]
        ):
            passed = False
            failure_reasons.append(
                f"Daily Tmax max |diff| {daily_report['max_abs_diff_tmax']} > threshold {THRESHOLDS[variable]['daily_tmax_max_abs_diff']}"
            )
        if (
            daily_report["max_abs_diff_tmin"]
            > THRESHOLDS[variable]["daily_tmin_max_abs_diff"]
        ):
            passed = False
            failure_reasons.append(
                f"Daily Tmin max |diff| {daily_report['max_abs_diff_tmin']} > threshold {THRESHOLDS[variable]['daily_tmin_max_abs_diff']}"
            )
    else:
        if (
            daily_report["max_abs_diff_mm"]
            > THRESHOLDS[variable]["daily_precip_max_abs_diff"]
        ):
            passed = False
            failure_reasons.append(
                f"Daily precip max |diff| {daily_report['max_abs_diff_mm']} > threshold {THRESHOLDS[variable]['daily_precip_max_abs_diff']}"
            )

    return {
        "station_code": station_code,
        "station_name": st_meta.name,
        "variable": variable,
        "years_evaluated": eval_years,
        "grid_cell": {
            "latitude_center": lat_center,
            "longitude_center": lon_center,
            "latitude_sl": lat_sl,
            "longitude_sl": lon_sl,
            "latitude_ts": lat_ts,
            "longitude_ts": lon_ts,
            "cell_matches": True,
        },
        "metadata": {
            "units_single_levels": units_sl,
            "units_timeseries": units_ts,
            "dtype_single_levels": dtype_sl,
            "dtype_timeseries": dtype_ts,
        },
        "hourly_comparison": {
            "n_hours": n_hours,
            "nan_single_levels": nan_sl,
            "nan_timeseries": nan_ts,
            "mean_abs_diff": mean_diff_hourly,
            "max_abs_diff": max_diff_hourly,
            "p999_abs_diff": p999_diff_hourly,
            "threshold": hourly_thresh,
            "hours_above_threshold": hours_above_threshold,
        },
        "daily_comparison": daily_report,
        "equivalent": passed,
        "failure_reasons": failure_reasons,
    }


def main() -> None:
    """CLI to run source equivalence comparison across stations and variables."""
    parser = argparse.ArgumentParser(
        description="Verify numerical equivalence between single-levels and timeseries ERA5 datasets."
    )
    parser.add_argument(
        "--stations",
        nargs="+",
        default=["21205710", "23215030"],
        help="Station codes to evaluate (default: 21205710, 23215030).",
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        default=["2m_temperature", "total_precipitation"],
        help="Variables to evaluate (default: 2m_temperature, total_precipitation).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("local/reports/era5_source_equivalence.json"),
        help="Path to output JSON report (default: local/reports/era5_source_equivalence.json).",
    )
    args = parser.parse_args()

    results: Dict[str, Any] = {
        "report_type": "era5_source_equivalence",
        "description": "Numerical equivalence verification between ERA5 single-levels and single-levels-timeseries",
        "stations": {},
        "all_equivalent": True,
    }

    print("=" * 88)
    print("COMPARACIÓN DE EQUIVALENCIA NUMÉRICA: ERA5 SINGLE-LEVELS VS TIMESERIES")
    print("=" * 88)

    for st in args.stations:
        results["stations"][st] = {}
        for var in args.variables:
            print(f"\n--- Estación {st} | Variable: {var} ---")
            try:
                res = compare_single_station_variable(st, var)
                results["stations"][st][var] = res
                print(
                    f"  Años evaluados:    {min(res['years_evaluated'])} - {max(res['years_evaluated'])} ({len(res['years_evaluated'])} años)"
                )
                print(
                    f"  Celda (lat, lon):  ({res['grid_cell']['latitude_center']}, {res['grid_cell']['longitude_center']})"
                )
                print(
                    f"  Horas comparadas:  {res['hourly_comparison']['n_hours']:,} h (cero NaNs)"
                )
                print(
                    f"  Hourly max |dif|:  {res['hourly_comparison']['max_abs_diff']:.6e} (umbral: {res['hourly_comparison']['threshold']})"
                )
                print(
                    f"  Hourly mean |dif|: {res['hourly_comparison']['mean_abs_diff']:.6e}"
                )
                print(
                    f"  Hourly p99.9:      {res['hourly_comparison']['p999_abs_diff']:.6e}"
                )
                print(
                    f"  Horas sobre umbral:{res['hourly_comparison']['hours_above_threshold']}"
                )

                if var == "2m_temperature":
                    print(
                        f"  Daily Tmax max |dif|: {res['daily_comparison']['max_abs_diff_tmax']:.6e} °C (umbral: {res['daily_comparison']['daily_tmax_threshold']})"
                    )
                    print(
                        f"  Daily Tmin max |dif|: {res['daily_comparison']['max_abs_diff_tmin']:.6e} °C (umbral: {res['daily_comparison']['daily_tmin_threshold']})"
                    )
                else:
                    print(
                        f"  Daily precip max |dif|: {res['daily_comparison']['max_abs_diff_mm']:.6e} mm (umbral: {res['daily_comparison']['daily_threshold_mm']})"
                    )

                print(
                    f"  Resultado:         {'EQUIVALENTE (PASS)' if res['equivalent'] else 'FALLÓ (FAIL)'}"
                )

                if not res["equivalent"]:
                    results["all_equivalent"] = False
                    print(f"  ERRORES: {res['failure_reasons']}")
            except Exception as exc:
                results["all_equivalent"] = False
                results["stations"][st][var] = {"error": str(exc), "equivalent": False}
                print(f"  ERROR EN EVALUACIÓN: {exc}")

    # Ensure output report directory exists
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 88)
    print(f"Informe guardado en: {args.report}")
    print(
        f"Estado global de equivalencia: {'PASS - TODAS LAS PRUEBAS EQUIVALENTES' if results['all_equivalent'] else 'FAIL - DISCREPANCIAS DETECTADAS'}"
    )
    print("=" * 88)

    if not results["all_equivalent"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
