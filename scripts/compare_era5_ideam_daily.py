#!/usr/bin/env python3
"""Compare ERA5 hourly precipitation aggregated to daily vs IDEAM DHIME rain gauge.

Evaluates:
- Temporal lags: -1, 0, +1 days
- Spatial sensitivity: Nearest grid cell + 3x3 surrounding spatial neighborhood
- Distributional verification metrics: Bias, RMSE, Pearson r, Spearman rho,
  rainy day frequencies (>=0.1, >=1.0, >=5.0, >=10.0 mm), daily quantiles,
  and monthly breakdown.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from tropicor.io.era5 import ERA5Adapter
from tropicor.io.ideam import read_dhime_daily_precipitation


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare daily ERA5 precipitation (12-12 UTC) vs DHIME gauges."
    )
    parser.add_argument(
        "--csv",
        "--dhime-file",
        dest="dhime_file",
        type=Path,
        required=True,
        help="Ruta obligatoria al archivo CSV/Excel de precipitación diaria DHIME.",
    )
    parser.add_argument(
        "--station",
        "--station-id",
        dest="station_id",
        type=str,
        required=True,
        help="Código obligatorio de la estación a evaluar (ej. 21205710, 23215030).",
    )
    parser.add_argument(
        "--era5-source",
        type=str,
        default=None,
        help="Ruta o patrón glob opcional a archivos NetCDF ERA5 (por defecto: local/era5_hourly/<station>/*.nc).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("local/era5_ideam_daily_comparison.csv"),
        help="Path to output CSV comparison report.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform comparison and print report without writing output CSV.",
    )
    return parser.parse_args(argv)


def resolve_era5_source(
    explicit_source: Optional[str], station_id: Optional[str] = None
) -> str:
    """Resolve ERA5 source path or glob pattern."""
    if explicit_source:
        return explicit_source

    if station_id:
        station_dir = Path("local/era5_hourly") / station_id
        if station_dir.exists() and list(station_dir.glob("*.nc")):
            return str(station_dir / "*.nc")

    hourly_dir = Path("local/era5_hourly")
    nc_files = list(hourly_dir.glob("*.nc"))
    if nc_files:
        return str(hourly_dir / "*.nc")

    msg = (
        f"No se encontraron archivos NetCDF en 'local/era5_hourly/{station_id or ''}/*.nc'. "
        "Descarga los datos con download_era5_years.py o especifica --era5-source."
    )
    raise FileNotFoundError(msg)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate haversine distance in kilometers between two geographic points."""
    r = 6371.0  # Earth radius in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    )
    return float(2.0 * r * np.arcsin(np.sqrt(a)))


def evaluate_synchronous_metrics(obs: pd.Series, mod: pd.Series) -> Dict[str, Any]:
    """Compute hydroclimatic metrics between synchronous observation and model."""
    df = pd.concat([obs.rename("obs"), mod.rename("mod")], axis=1).dropna()
    n = len(df)
    if n < 3:
        return {
            "n": n,
            "r": np.nan,
            "p_val": np.nan,
            "rho": np.nan,
            "p_rho": np.nan,
            "bias": np.nan,
            "pbias": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "sum_obs": float(df["obs"].sum()) if n > 0 else np.nan,
            "sum_mod": float(df["mod"].sum()) if n > 0 else np.nan,
            "mean_obs": float(df["obs"].mean()) if n > 0 else np.nan,
            "mean_mod": float(df["mod"].mean()) if n > 0 else np.nan,
        }

    y_obs = df["obs"].to_numpy(dtype=float)
    y_mod = df["mod"].to_numpy(dtype=float)

    bias = float(np.mean(y_mod - y_obs))
    rmse = float(np.sqrt(np.mean((y_mod - y_obs) ** 2)))
    mae = float(np.mean(np.abs(y_mod - y_obs)))
    sum_obs = float(np.sum(y_obs))
    sum_mod = float(np.sum(y_mod))
    pbias = (
        float(100.0 * (sum_mod - sum_obs) / sum_obs) if abs(sum_obs) > 1e-6 else np.nan
    )

    r, p_val = stats.pearsonr(y_obs, y_mod)
    rho, p_rho = stats.spearmanr(y_obs, y_mod)

    return {
        "n": n,
        "r": float(r),
        "p_val": float(p_val),
        "rho": float(rho),
        "p_rho": float(p_rho),
        "bias": bias,
        "pbias": pbias,
        "rmse": rmse,
        "mae": mae,
        "sum_obs": sum_obs,
        "sum_mod": sum_mod,
        "mean_obs": float(np.mean(y_obs)),
        "mean_mod": float(np.mean(y_mod)),
    }


def run_comparison(
    dhime_path: Path,
    era5_source: str,
    station_id: str,
) -> Dict[str, Any]:
    """Execute full lag, spatial neighborhood, and distributional comparison."""
    # 1. Load DHIME ground truth
    ds_dhime = read_dhime_daily_precipitation(dhime_path)
    if station_id not in ds_dhime.station.values:
        avail = list(ds_dhime.station.values)
        raise KeyError(
            f"Station ID '{station_id}' not found in DHIME dataset. Available: {avail}"
        )

    lat_obs = float(ds_dhime.lat.sel(station=station_id).values)
    lon_obs = float(ds_dhime.lon.sel(station=station_id).values)
    station_name = str(ds_dhime.station_name.sel(station=station_id).values)
    elevation = float(ds_dhime.elevation.sel(station=station_id).values)

    s_obs = ds_dhime["pr"].sel(station=station_id).to_series()

    # 2. Open ERA5
    with ERA5Adapter(era5_source) as era5_adapter:
        ds_era5 = era5_adapter.dataset

        lats = ds_era5.latitude.values
        lons = ds_era5.longitude.values

        # Find nearest indices
        ilat_near = int(np.argmin(np.abs(lats - lat_obs)))
        ilon_near = int(np.argmin(np.abs(lons - lon_obs)))

        lat_center = float(lats[ilat_near])
        lon_center = float(lons[ilon_near])

        # Lag Evaluation (-1, 0, +1)
        lag_results: List[Dict[str, Any]] = []
        for lag in [-1, 0, 1]:
            s_era_lag = era5_adapter.get_daily_precipitation(
                lat_center,
                lon_center,
                lag_days=lag,
                min_hours=24,
            )
            metrics = evaluate_synchronous_metrics(s_obs, s_era_lag)
            metrics["lag_days"] = lag
            metrics["description"] = (
                "ERA5 shifted backward 1 day (assigned D-1)"
                if lag == -1
                else (
                    "Standard WMO/IDEAM [12:00 UTC D -> 12:00 UTC D+1] assigned D"
                    if lag == 0
                    else "ERA5 shifted forward 1 day (assigned D+1)"
                )
            )
            lag_results.append(metrics)

        # Spatial Sensitivity (3x3 neighborhood around station)
        # Verify that all 9 cells are strictly within the downloaded area
        spatial_results: List[Dict[str, Any]] = []
        for dlat in [1, 0, -1]:
            idx_lat = ilat_near + dlat
            if idx_lat < 0 or idx_lat >= len(lats):
                raise ValueError(
                    f"Spatial 3x3 evaluation failed: neighbor cell with offset "
                    f"dlat={dlat} is outside the downloaded latitude range "
                    f"[{float(lats.min()):.2f}, {float(lats.max()):.2f}]."
                )
            cur_lat = float(lats[idx_lat])
            for dlon in [-1, 0, 1]:
                idx_lon = ilon_near + dlon
                if idx_lon < 0 or idx_lon >= len(lons):
                    raise ValueError(
                        f"Spatial 3x3 evaluation failed: neighbor cell with offset "
                        f"dlon={dlon} is outside the downloaded longitude range "
                        f"[{float(lons.min()):.2f}, {float(lons.max()):.2f}]."
                    )
                cur_lon = float(lons[idx_lon])

                s_era_cell = era5_adapter.get_daily_precipitation(
                    cur_lat,
                    cur_lon,
                    lag_days=0,
                    min_hours=24,
                )
                m = evaluate_synchronous_metrics(s_obs, s_era_cell)
                dist = haversine_km(lat_obs, lon_obs, cur_lat, cur_lon)
                spatial_results.append(
                    {
                        "dlat": dlat,
                        "dlon": dlon,
                        "lat": cur_lat,
                        "lon": cur_lon,
                        "distance_km": dist,
                        "is_center": (dlat == 0 and dlon == 0),
                        **m,
                    }
                )

        # Detailed metrics for center cell at Lag 0
        s_era_center = era5_adapter.get_daily_precipitation(
            lat_center,
            lon_center,
            lag_days=0,
            min_hours=24,
        )
        df_sync = pd.concat(
            [s_obs.rename("obs"), s_era_center.rename("era5")], axis=1
        ).dropna()
        obs_vals = df_sync["obs"].to_numpy(dtype=float)
        era_vals = df_sync["era5"].to_numpy(dtype=float)

        # Rainy day frequencies
        thresholds = [0.1, 1.0, 5.0, 10.0]
        rain_freqs: List[Dict[str, Any]] = []
        for th in thresholds:
            n_obs_wet = int(np.sum(obs_vals >= th))
            pct_obs_wet = (
                float(100.0 * n_obs_wet / len(obs_vals)) if len(obs_vals) > 0 else 0.0
            )
            n_era_wet = int(np.sum(era_vals >= th))
            pct_era_wet = (
                float(100.0 * n_era_wet / len(era_vals)) if len(era_vals) > 0 else 0.0
            )
            rain_freqs.append(
                {
                    "threshold_mm": th,
                    "n_obs": n_obs_wet,
                    "pct_obs": pct_obs_wet,
                    "n_era5": n_era_wet,
                    "pct_era5": pct_era_wet,
                    "pct_diff": pct_era_wet - pct_obs_wet,
                }
            )

        # Quantiles
        quantiles = [0.50, 0.75, 0.90, 0.95, 0.99]
        quantile_metrics: List[Dict[str, Any]] = []
        for q in quantiles:
            q_obs = float(np.quantile(obs_vals, q))
            q_era = float(np.quantile(era_vals, q))
            quantile_metrics.append(
                {
                    "quantile": f"P{int(q * 100)}",
                    "prob": q,
                    "obs_val_mm": q_obs,
                    "era5_val_mm": q_era,
                    "diff_mm": q_era - q_obs,
                }
            )

        # Monthly breakdown
        df_sync["year_month"] = df_sync.index.to_period("M").astype(str)
        monthly_breakdown: List[Dict[str, Any]] = []
        for ym, grp in df_sync.groupby("year_month"):
            m_metrics = evaluate_synchronous_metrics(grp["obs"], grp["era5"])
            monthly_breakdown.append(
                {
                    "year_month": ym,
                    "n_days": len(grp),
                    "sum_obs_mm": float(grp["obs"].sum()),
                    "sum_era5_mm": float(grp["era5"].sum()),
                    "bias_mm_day": m_metrics["bias"],
                    "rmse_mm_day": m_metrics["rmse"],
                    "r": m_metrics["r"],
                    "rho": m_metrics["rho"],
                }
            )

    return {
        "metadata": {
            "station_id": station_id,
            "station_name": station_name,
            "station_lat": lat_obs,
            "station_lon": lon_obs,
            "elevation_masl": elevation,
            "era5_center_lat": lat_center,
            "era5_center_lon": lon_center,
            "center_distance_km": haversine_km(
                lat_obs, lon_obs, lat_center, lon_center
            ),
            "source_era5": era5_source,
            "source_dhime": str(dhime_path),
        },
        "lag_results": lag_results,
        "spatial_results": spatial_results,
        "rain_frequencies": rain_freqs,
        "quantiles": quantile_metrics,
        "monthly_breakdown": monthly_breakdown,
    }


def print_report(results: Dict[str, Any]) -> None:
    meta = results["metadata"]
    print("\n" + "=" * 78)
    print("TROPICOR SPRINT 8 - FASE 4: COMPARACIÓN ERA5 HORARIO VS DHIME")
    print("=" * 78)
    print(f"Estación:   {meta['station_name']} [{meta['station_id']}]")
    lat_s = meta["station_lat"]
    lon_s = meta["station_lon"]
    elev_s = meta["elevation_masl"]
    print(f"Ubicación:  Lat {lat_s:.4f}°, Lon {lon_s:.4f}°, Altitud {elev_s:.1f} msnm")
    lat_e = meta["era5_center_lat"]
    lon_e = meta["era5_center_lon"]
    dist_e = meta["center_distance_km"]
    p_era = f"Punto ERA5: Lat {lat_e:.4f}°, Lon {lon_e:.4f}° (Dist: {dist_e:.2f} km)"
    print(p_era)
    print(f"Fuente DHIME: {meta['source_dhime']}")
    print(f"Fuente ERA5:  {meta['source_era5']}")

    print("\n" + "-" * 78)
    print("1. EVALUACIÓN DE DESFASES TEMPORALES (LAGS -1, 0, +1 DÍAS)")
    print("-" * 78)
    h_lag = (
        f"{'Lag':<8} | {'N':<4} | {'Pearson r':<18} | {'Spearman rho':<18} | "
        f"{'Sesgo (mm)':<10} | {'RMSE (mm)':<10}"
    )
    print(h_lag)
    print("-" * 78)
    for r in results["lag_results"]:
        lag_str = f"{r['lag_days']:+d} d"
        r_str = f"{r['r']:+.4f} (p={r['p_val']:.2e})" if not np.isnan(r["r"]) else "N/A"
        rho_str = (
            f"{r['rho']:+.4f} (p={r['p_rho']:.2e})" if not np.isnan(r["rho"]) else "N/A"
        )
        bias_str = f"{r['bias']:+.2f}" if not np.isnan(r["bias"]) else "N/A"
        rmse_str = f"{r['rmse']:.2f}" if not np.isnan(r["rmse"]) else "N/A"
        line = (
            f"{lag_str:<8} | {r['n']:<4} | {r_str:<18} | {rho_str:<18} | "
            f"{bias_str:<10} | {rmse_str:<10}"
        )
        print(line)

    print("\n" + "-" * 78)
    print("2. SENSIBILIDAD ESPACIAL: MATRIZ 3x3 DE VECINOS ALREDEDOR DE LA ESTACIÓN")
    print("-" * 78)
    h_spat = (
        f"{'Malla':<10} | {'Lat':<6} | {'Lon':<7} | {'Dist(km)':<8} | "
        f"{'r':<8} | {'rho':<8} | {'Sesgo':<8} | {'RMSE':<8} | {'Suma(mm)':<9}"
    )
    print(h_spat)
    print("-" * 78)
    for s in results["spatial_results"]:
        tag = "CENTRO" if s["is_center"] else f"({s['dlat']:+d},{s['dlon']:+d})"
        print(
            f"{tag:<10} | {s['lat']:6.2f} | {s['lon']:7.2f} | "
            f"{s['distance_km']:8.1f} | {s['r']:+7.4f} | {s['rho']:+7.4f} | "
            f"{s['bias']:+7.2f} | {s['rmse']:7.2f} | {s['sum_mod']:8.1f}"
        )

    print("\n" + "-" * 78)
    print("3. VERIFICACIÓN DISTRIBUCIONAL: PUNTO MÁS CERCANO (Lag 0)")
    print("-" * 78)
    print("Frecuencia de Días Lluviosos:")
    for f in results["rain_frequencies"]:
        print(
            f"  Umbral >= {f['threshold_mm']:4.1f} mm: "
            f"Obs = {f['pct_obs']:5.1f}% ({f['n_obs']} d) | "
            f"ERA5 = {f['pct_era5']:5.1f}% ({f['n_era5']} d) | "
            f"Diferencia = {f['pct_diff']:+5.1f}%"
        )

    print("\nCuantiles de Precipitación Diaria (mm/día):")
    print(f"  {'Cuantil':<8} | {'Observado':<12} | {'ERA5':<12} | {'Diferencia':<12}")
    for q in results["quantiles"]:
        obs_q = f"{q['obs_val_mm']:10.2f} mm"
        era_q = f"{q['era5_val_mm']:10.2f} mm"
        diff_q = f"{q['diff_mm']:+10.2f} mm"
        print(f"  {q['quantile']:<8} | {obs_q} | {era_q} | {diff_q}")

    print("\nDesglose Mensual:")
    h_m = (
        f"  {'Mes':<8} | {'N':<4} | {'Obs Total':<10} | {'ERA5 Total':<11} | "
        f"{'Sesgo Medio':<12} | {'Pearson r':<10}"
    )
    print(h_m)
    for m in results["monthly_breakdown"]:
        print(
            f"  {m['year_month']:<8} | {m['n_days']:<4} | "
            f"{m['sum_obs_mm']:8.2f} mm | {m['sum_era5_mm']:9.2f} mm | "
            f"{m['bias_mm_day']:+10.2f} mm | {m['r']:+8.4f}"
        )
    print("=" * 78 + "\n")


def export_csv_summary(results: Dict[str, Any], output_path: Path) -> None:
    """Export comparison results as a structured CSV file."""
    rows: List[Dict[str, Any]] = []
    meta = results["metadata"]

    # Section 1: Lags
    for lag in results["lag_results"]:
        rows.append(
            {
                "section": "lag_evaluation",
                "station_id": meta["station_id"],
                "station_name": meta["station_name"],
                "item": f"lag_{lag['lag_days']:+d}",
                "metric_1_name": "pearson_r",
                "metric_1_val": lag["r"],
                "metric_2_name": "spearman_rho",
                "metric_2_val": lag["rho"],
                "metric_3_name": "bias_mm_day",
                "metric_3_val": lag["bias"],
                "metric_4_name": "rmse_mm_day",
                "metric_4_val": lag["rmse"],
                "n_samples": lag["n"],
                "notes": lag["description"],
            }
        )

    # Section 2: Spatial
    for s in results["spatial_results"]:
        tag = "center" if s["is_center"] else f"neighbor_{s['dlat']:+d}_{s['dlon']:+d}"
        notes_str = f"lat={s['lat']:.2f}, lon={s['lon']:.2f}, rmse={s['rmse']:.2f}"
        rows.append(
            {
                "section": "spatial_sensitivity",
                "station_id": meta["station_id"],
                "station_name": meta["station_name"],
                "item": tag,
                "metric_1_name": "distance_km",
                "metric_1_val": s["distance_km"],
                "metric_2_name": "pearson_r",
                "metric_2_val": s["r"],
                "metric_3_name": "bias_mm_day",
                "metric_3_val": s["bias"],
                "metric_4_name": "sum_era5_mm",
                "metric_4_val": s["sum_mod"],
                "n_samples": s["n"],
                "notes": notes_str,
            }
        )

    # Section 3: Rainy frequencies
    for f in results["rain_frequencies"]:
        rows.append(
            {
                "section": "rainy_frequency",
                "station_id": meta["station_id"],
                "station_name": meta["station_name"],
                "item": f"threshold_{f['threshold_mm']}mm",
                "metric_1_name": "pct_obs",
                "metric_1_val": f["pct_obs"],
                "metric_2_name": "pct_era5",
                "metric_2_val": f["pct_era5"],
                "metric_3_name": "pct_diff",
                "metric_3_val": f["pct_diff"],
                "metric_4_name": "n_obs_wet",
                "metric_4_val": f["n_obs"],
                "n_samples": np.nan,
                "notes": f"n_era5_wet={f['n_era5']}",
            }
        )

    # Section 4: Quantiles
    for q in results["quantiles"]:
        rows.append(
            {
                "section": "daily_quantiles",
                "station_id": meta["station_id"],
                "station_name": meta["station_name"],
                "item": q["quantile"],
                "metric_1_name": "obs_val_mm",
                "metric_1_val": q["obs_val_mm"],
                "metric_2_name": "era5_val_mm",
                "metric_2_val": q["era5_val_mm"],
                "metric_3_name": "diff_mm",
                "metric_3_val": q["diff_mm"],
                "metric_4_name": "prob",
                "metric_4_val": q["prob"],
                "n_samples": np.nan,
                "notes": "Quantile comparison at nearest grid cell",
            }
        )

    # Section 5: Monthly breakdown
    for m in results["monthly_breakdown"]:
        rows.append(
            {
                "section": "monthly_breakdown",
                "station_id": meta["station_id"],
                "station_name": meta["station_name"],
                "item": m["year_month"],
                "metric_1_name": "sum_obs_mm",
                "metric_1_val": m["sum_obs_mm"],
                "metric_2_name": "sum_era5_mm",
                "metric_2_val": m["sum_era5_mm"],
                "metric_3_name": "bias_mm_day",
                "metric_3_val": m["bias_mm_day"],
                "metric_4_name": "pearson_r",
                "metric_4_val": m["r"],
                "n_samples": m["n_days"],
                "notes": f"rmse={m['rmse_mm_day']:.2f}, rho={m['rho']:.4f}",
            }
        )

    df_out = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    print(f"Resultados guardados exitosamente en: {output_path}")


def main() -> None:
    args = parse_args()
    era5_source = resolve_era5_source(args.era5_source, args.station_id)
    results = run_comparison(
        dhime_path=args.dhime_file,
        era5_source=era5_source,
        station_id=args.station_id,
    )
    print_report(results)
    if not args.dry_run:
        export_csv_summary(results, args.output_csv)


if __name__ == "__main__":
    main()
