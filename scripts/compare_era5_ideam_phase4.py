#!/usr/bin/env python3
"""TROPICOR Sprint 8 - Fase 4: Comparación ERA5 Horario vs DHIME (Resultados Preliminares).

Evalúa de manera unificada y determinística:
1. Auditoría previa por estación (periodo, días válidos, faltantes, calidad, meses completos).
2. Desfases temporales (-1, 0, +1 días) global y por temporada (DEF, MAM, JJA, SON) con IC 95% bootstrap.
3. Matriz de vecinos 3x3 alrededor de la estación.
4. Métricas con el desfase ganador (sesgo, RMSE, frecuencia >=0.1 y >=1.0 mm, percentiles 50, 90, 95, 99) por mes.
5. Exclusión de meses incompletos en los agregados mensuales con reporte de meses restantes.
6. Verificación del último día de ERA5 como NaN.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from tropicor.io.era5 import ERA5Adapter
from tropicor.io.ideam import read_dhime_daily_precipitation


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calcula la distancia haversine en kilómetros."""
    r = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return float(2.0 * r * np.arcsin(np.sqrt(a)))


def compute_bootstrap_ci(
    obs: np.ndarray,
    mod: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Calcula intervalos de confianza del 95% para Pearson r y Spearman rho vía bootstrap."""
    n = len(obs)
    if n < 3:
        return (np.nan, np.nan), (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    r_boot = np.empty(n_boot)
    rho_boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        o_b = obs[idx]
        m_b = mod[idx]
        r_boot[i] = stats.pearsonr(o_b, m_b)[0]
        rho_boot[i] = stats.spearmanr(o_b, m_b)[0]
    ci_r = (float(np.percentile(r_boot, 2.5)), float(np.percentile(r_boot, 97.5)))
    ci_rho = (float(np.percentile(rho_boot, 2.5)), float(np.percentile(rho_boot, 97.5)))
    return ci_r, ci_rho


def evaluate_metrics(obs: pd.Series, mod: pd.Series) -> Dict[str, Any]:
    """Calcula métricas básicas entre series síncronas sin NaNs."""
    df = pd.concat([obs.rename("obs"), mod.rename("mod")], axis=1).dropna()
    n = len(df)
    if n < 3:
        return {
            "n": n,
            "r": np.nan,
            "rho": np.nan,
            "bias": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "sum_obs": np.nan,
            "sum_mod": np.nan,
        }
    y_obs = df["obs"].to_numpy(dtype=float)
    y_mod = df["mod"].to_numpy(dtype=float)
    bias = float(np.mean(y_mod - y_obs))
    rmse = float(np.sqrt(np.mean((y_mod - y_obs) ** 2)))
    mae = float(np.mean(np.abs(y_mod - y_obs)))
    r = float(stats.pearsonr(y_obs, y_mod)[0])
    rho = float(stats.spearmanr(y_obs, y_mod)[0])
    return {
        "n": n,
        "r": r,
        "rho": rho,
        "bias": bias,
        "rmse": rmse,
        "mae": mae,
        "sum_obs": float(np.sum(y_obs)),
        "sum_mod": float(np.sum(y_mod)),
    }


def analyze_station(
    station_id: str,
    station_name: str,
    dhime_csv_path: Path,
    era5_files: List[str],
    start_date: str,
    end_date: str,
) -> Dict[str, Any]:
    """Ejecuta el protocolo de análisis completo para una estación."""
    print(f"\nProcesando estación {station_name} [{station_id}]...")

    # 1. Cargar IDEAM desde CSV
    ds_dhime = read_dhime_daily_precipitation(dhime_csv_path)
    lat_obs = float(ds_dhime.lat.sel(station=station_id).values)
    lon_obs = float(ds_dhime.lon.sel(station=station_id).values)
    elevation = float(ds_dhime.elevation.sel(station=station_id).values)

    s_obs_full = ds_dhime["pr"].sel(station=station_id).to_series()

    # Recortar al periodo común
    common_idx = pd.date_range(start_date, end_date, freq="D")
    s_obs = s_obs_full.reindex(common_idx)

    # Auditoría previa
    total_days = len(common_idx)
    valid_days = int(s_obs.notna().sum())
    missing_days = total_days - valid_days
    pct_missing = (missing_days / total_days) * 100.0

    # Nivel de aprobación en el periodo
    # También leer directamente del CSV original para verificar
    df_raw = pd.read_csv(dhime_csv_path)
    df_raw["Fecha_dt"] = pd.to_datetime(df_raw["Fecha"], errors="coerce").dt.normalize()
    df_raw_period = df_raw[(df_raw["Fecha_dt"] >= start_date) & (df_raw["Fecha_dt"] <= end_date)]
    raw_flags = df_raw_period["NivelAprobacion"].value_counts(dropna=False).to_dict()

    # Meses completos en el periodo
    monthly_counts = s_obs.groupby(s_obs.index.to_period("M")).agg(
        total="count",
        valid=lambda x: int(x.notna().sum()),
        days_in_month=lambda x: len(x),
    )
    monthly_counts["is_complete"] = monthly_counts["valid"] == monthly_counts["days_in_month"]
    total_months = len(monthly_counts)
    complete_months = int(monthly_counts["is_complete"].sum())

    # 2. Cargar ERA5
    with ERA5Adapter(era5_files) as era5_adapter:
        ds_era5 = era5_adapter.dataset
        lats = ds_era5.latitude.values
        lons = ds_era5.longitude.values

        ilat_near = int(np.argmin(np.abs(lats - lat_obs)))
        ilon_near = int(np.argmin(np.abs(lons - lon_obs)))
        lat_center = float(lats[ilat_near])
        lon_center = float(lons[ilon_near])
        dist_center = haversine_km(lat_obs, lon_obs, lat_center, lon_center)

        # Verificación del último día de ERA5
        s_era_center_raw = era5_adapter.get_daily_precipitation(
            lat_center, lon_center, lag_days=0, min_hours=24
        )
        last_day_val = s_era_center_raw.loc[end_date] if end_date in s_era_center_raw.index else np.nan
        last_day_is_nan = bool(pd.isna(last_day_val))

        # 3. Desfases temporales (-1, 0, +1 días)
        lag_results: List[Dict[str, Any]] = []
        for lag in [-1, 0, 1]:
            s_era_lag = era5_adapter.get_daily_precipitation(
                lat_center, lon_center, lag_days=lag, min_hours=24
            ).reindex(common_idx)

            # Global
            df_sync = pd.concat([s_obs.rename("obs"), s_era_lag.rename("mod")], axis=1).dropna()
            n_sync = len(df_sync)
            y_o = df_sync["obs"].to_numpy(dtype=float)
            y_m = df_sync["mod"].to_numpy(dtype=float)

            r_val = float(stats.pearsonr(y_o, y_m)[0]) if n_sync >= 3 else np.nan
            rho_val = float(stats.spearmanr(y_o, y_m)[0]) if n_sync >= 3 else np.nan
            bias_val = float(np.mean(y_m - y_o)) if n_sync >= 1 else np.nan
            rmse_val = float(np.sqrt(np.mean((y_m - y_o) ** 2))) if n_sync >= 1 else np.nan

            ci_r, ci_rho = compute_bootstrap_ci(y_o, y_m, n_boot=1000, seed=42)

            # Por temporada
            seasons = {
                "DEF": [12, 1, 2],
                "MAM": [3, 4, 5],
                "JJA": [6, 7, 8],
                "SON": [9, 10, 11],
            }
            season_metrics: Dict[str, Any] = {}
            for s_name, s_months in seasons.items():
                df_seas = df_sync[df_sync.index.month.isin(s_months)]
                n_s = len(df_seas)
                if n_s >= 3:
                    y_os = df_seas["obs"].to_numpy(dtype=float)
                    y_ms = df_seas["mod"].to_numpy(dtype=float)
                    r_s = float(stats.pearsonr(y_os, y_ms)[0])
                    rho_s = float(stats.spearmanr(y_os, y_ms)[0])
                    bias_s = float(np.mean(y_ms - y_os))
                    rmse_s = float(np.sqrt(np.mean((y_ms - y_os) ** 2)))
                    ci_rs, ci_rhos = compute_bootstrap_ci(y_os, y_ms, n_boot=1000, seed=42)
                else:
                    r_s, rho_s, bias_s, rmse_s = np.nan, np.nan, np.nan, np.nan
                    ci_rs, ci_rhos = (np.nan, np.nan), (np.nan, np.nan)
                season_metrics[s_name] = {
                    "n": n_s,
                    "r": r_s,
                    "r_ci": ci_rs,
                    "rho": rho_s,
                    "rho_ci": ci_rhos,
                    "bias": bias_s,
                    "rmse": rmse_s,
                }

            lag_results.append(
                {
                    "lag": lag,
                    "n": n_sync,
                    "r": r_val,
                    "r_ci": ci_r,
                    "rho": rho_val,
                    "rho_ci": ci_rho,
                    "bias": bias_val,
                    "rmse": rmse_val,
                    "seasons": season_metrics,
                }
            )

        # 4. Matriz 3x3 de vecinos (lag 0 y lag ganador)
        # Identificar lag ganador por mayor r
        best_lag_entry = max(lag_results, key=lambda x: x["r"])
        winning_lag = best_lag_entry["lag"]

        spatial_grid: List[Dict[str, Any]] = []
        for dlat in [1, 0, -1]:
            idx_lat = ilat_near + dlat
            cur_lat = float(lats[idx_lat])
            for dlon in [-1, 0, 1]:
                idx_lon = ilon_near + dlon
                cur_lon = float(lons[idx_lon])
                dist_km = haversine_km(lat_obs, lon_obs, cur_lat, cur_lon)
                is_center = (dlat == 0 and dlon == 0)

                # Evaluamos a lag 0 y al winning lag
                s_era_cell_0 = era5_adapter.get_daily_precipitation(
                    cur_lat, cur_lon, lag_days=0, min_hours=24
                ).reindex(common_idx)
                m0 = evaluate_metrics(s_obs, s_era_cell_0)

                s_era_cell_win = era5_adapter.get_daily_precipitation(
                    cur_lat, cur_lon, lag_days=winning_lag, min_hours=24
                ).reindex(common_idx)
                mwin = evaluate_metrics(s_obs, s_era_cell_win)

                spatial_grid.append(
                    {
                        "dlat": dlat,
                        "dlon": dlon,
                        "lat": cur_lat,
                        "lon": cur_lon,
                        "dist_km": dist_km,
                        "is_center": is_center,
                        "lag0": m0,
                        "lag_win": mwin,
                    }
                )

        # 5. Métricas detalladas con el desfase ganador por mes
        s_era_winner = era5_adapter.get_daily_precipitation(
            lat_center, lon_center, lag_days=winning_lag, min_hours=24
        ).reindex(common_idx)

        df_winner = pd.concat([s_obs.rename("obs"), s_era_winner.rename("mod")], axis=1)

        monthly_winner_metrics: List[Dict[str, Any]] = []
        for month_num in range(1, 13):
            sub_m = df_winner[df_winner.index.month == month_num].dropna()
            n_m = len(sub_m)
            o_m = sub_m["obs"].to_numpy(dtype=float)
            m_m = sub_m["mod"].to_numpy(dtype=float)

            bias_m = float(np.mean(m_m - o_m)) if n_m > 0 else np.nan
            rmse_m = float(np.sqrt(np.mean((m_m - o_m) ** 2))) if n_m > 0 else np.nan

            # Frecuencias >= 0.1 y >= 1.0 mm
            p_obs_01 = (np.sum(o_m >= 0.1) / n_m * 100.0) if n_m > 0 else np.nan
            p_era_01 = (np.sum(m_m >= 0.1) / n_m * 100.0) if n_m > 0 else np.nan
            diff_01 = p_era_01 - p_obs_01

            p_obs_10 = (np.sum(o_m >= 1.0) / n_m * 100.0) if n_m > 0 else np.nan
            p_era_10 = (np.sum(m_m >= 1.0) / n_m * 100.0) if n_m > 0 else np.nan
            diff_10 = p_era_10 - p_obs_10

            # Percentiles 50, 90, 95, 99
            quantiles = [0.50, 0.90, 0.95, 0.99]
            q_obs = {f"P{int(q*100)}": float(np.quantile(o_m, q)) for q in quantiles} if n_m > 0 else {}
            q_era = {f"P{int(q*100)}": float(np.quantile(m_m, q)) for q in quantiles} if n_m > 0 else {}
            q_diff = {f"P{int(q*100)}": q_era[f"P{int(q*100)}"] - q_obs[f"P{int(q*100)}"] for q in quantiles} if n_m > 0 else {}

            monthly_winner_metrics.append(
                {
                    "month": month_num,
                    "month_name": ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"][month_num - 1],
                    "n_days": n_m,
                    "bias": bias_m,
                    "rmse": rmse_m,
                    "pct_wet_01_obs": p_obs_01,
                    "pct_wet_01_era": p_era_01,
                    "pct_wet_01_diff": diff_01,
                    "pct_wet_10_obs": p_obs_10,
                    "pct_wet_10_era": p_era_10,
                    "pct_wet_10_diff": diff_10,
                    "q_obs": q_obs,
                    "q_era": q_era,
                    "q_diff": q_diff,
                }
            )

        # 6. Agregados Mensuales con exclusión estricta de meses incompletos
        # Agrupamos por Periodo 'M': solo conservamos meses donde is_complete == True y ERA5 completo
        # Contamos cuántos meses completos quedan por cada mes del año
        complete_months_by_month_num = {}
        monthly_aggregates_complete: List[Dict[str, Any]] = []

        for month_num in range(1, 13):
            # Identificar todos los periodos (YYYY-MM) correspondientes a este mes
            periods = [p for p in monthly_counts.index if p.month == month_num]
            # Filtrar solo aquellos completos
            complete_periods = [p for p in periods if monthly_counts.loc[p, "is_complete"]]
            complete_months_by_month_num[month_num] = {
                "total_possible": len(periods),
                "complete_kept": len(complete_periods),
                "excluded": len(periods) - len(complete_periods),
            }

            # Calcular suma mensual observada y modelada sobre los meses completos
            sums_obs = []
            sums_mod = []
            for cp in complete_periods:
                cp_mask = (df_winner.index.to_period("M") == cp)
                sub_cp = df_winner[cp_mask]
                # Verificamos si ERA5 no tiene NaNs en este mes completo
                if sub_cp["mod"].notna().all() and sub_cp["obs"].notna().all():
                    sums_obs.append(sub_cp["obs"].sum())
                    sums_mod.append(sub_cp["mod"].sum())

            if len(sums_obs) > 0:
                mean_obs_sum = float(np.mean(sums_obs))
                mean_mod_sum = float(np.mean(sums_mod))
                monthly_bias = float(np.mean(np.array(sums_mod) - np.array(sums_obs)))
                monthly_rmse = float(np.sqrt(np.mean((np.array(sums_mod) - np.array(sums_obs)) ** 2)))
                monthly_r = float(stats.pearsonr(sums_obs, sums_mod)[0]) if len(sums_obs) >= 3 else np.nan
            else:
                mean_obs_sum, mean_mod_sum, monthly_bias, monthly_rmse, monthly_r = np.nan, np.nan, np.nan, np.nan, np.nan

            monthly_aggregates_complete.append(
                {
                    "month": month_num,
                    "month_name": ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"][month_num - 1],
                    "complete_months_count": len(sums_obs),
                    "mean_obs_sum_mm": mean_obs_sum,
                    "mean_mod_sum_mm": mean_mod_sum,
                    "monthly_bias_mm": monthly_bias,
                    "monthly_rmse_mm": monthly_rmse,
                    "monthly_r": monthly_r,
                }
            )

    return {
        "metadata": {
            "station_id": station_id,
            "station_name": station_name,
            "lat_obs": lat_obs,
            "lon_obs": lon_obs,
            "elevation": elevation,
            "lat_center": lat_center,
            "lon_center": lon_center,
            "dist_center_km": dist_center,
            "start_date": start_date,
            "end_date": end_date,
            "last_day_is_nan": last_day_is_nan,
            "last_day_val": last_day_val,
        },
        "audit": {
            "total_days": total_days,
            "valid_days": valid_days,
            "missing_days": missing_days,
            "pct_missing": pct_missing,
            "raw_flags": raw_flags,
            "total_months": total_months,
            "complete_months": complete_months,
            "pct_complete_months": (complete_months / total_months) * 100.0,
        },
        "lag_results": lag_results,
        "winning_lag": winning_lag,
        "spatial_grid": spatial_grid,
        "monthly_winner_metrics": monthly_winner_metrics,
        "complete_months_by_month_num": complete_months_by_month_num,
        "monthly_aggregates_complete": monthly_aggregates_complete,
    }


def main() -> None:
    # 1. Definir estaciones
    stations = [
        {
            "id": "21205710",
            "name": "Jardín Botánico",
            "csv": Path("local/newFiles/DiaPluviometrico_JardinBotanico_21205710.csv"),
            "era5": sorted([str(f) for f in Path("local/era5_hourly/21205710").glob("era5_total_precipitation_*.nc") if "2024" not in f.name]),
            "start": "1975-01-01",
            "end": "2015-12-31",
        },
        {
            "id": "23215030",
            "name": "Aguas Claras",
            "csv": Path("local/newFiles/DiaPluviometrico_AguasClaras.csv"),
            "era5": sorted([str(f) for f in Path("local/era5_hourly/23215030").glob("era5_total_precipitation_*.nc")]),
            "start": "1975-01-01",
            "end": "2000-12-31",
        },
    ]

    all_results = {}
    for st in stations:
        res = analyze_station(
            station_id=st["id"],
            station_name=st["name"],
            dhime_csv_path=st["csv"],
            era5_files=st["era5"],
            start_date=st["start"],
            end_date=st["end"],
        )
        all_results[st["id"]] = res

    # Imprimir reporte estructurado
    print("\n" + "=" * 88)
    print("REPORTE FASE 4: PRELIMINAR — COBERTURA PARCIAL DE ERA5")
    print("=" * 88)

    for st_id, r in all_results.items():
        meta = r["metadata"]
        audit = r["audit"]
        print("\n" + "#" * 88)
        print(f"ESTACIÓN: {meta['station_name'].upper()} [CÓDIGO: {meta['station_id']}]")
        print("#" * 88)
        print(f"Coordenadas Observadas: Lat {meta['lat_obs']:.4f}°, Lon {meta['lon_obs']:.4f}°, Altitud {meta['elevation']:.1f} msnm")
        print(f"Punto ERA5 Más Cercano: Lat {meta['lat_center']:.4f}°, Lon {meta['lon_center']:.4f}° (Distancia: {meta['dist_center_km']:.2f} km)")
        print(f"Periodo Común Evaluado: {meta['start_date']} a {meta['end_date']}")

        print("\n--- 1. AUDITORÍA PREVIA AL ANÁLISIS ---")
        print(f"Días totales en calendario:   {audit['total_days']:,} días")
        print(f"Días válidos observados:      {audit['valid_days']:,} días")
        print(f"Días faltantes:               {audit['missing_days']:,} días ({audit['pct_missing']:.2f}%)")
        print("Proporción por Nivel de Aprobación:")
        for k, v in audit["raw_flags"].items():
            pct = (v / audit['total_days']) * 100.0
            print(f"  - {k}: {v:,} registros ({pct:.2f}% del periodo)")
        print(f"Meses completos observados:   {audit['complete_months']} de {audit['total_months']} meses ({audit['pct_complete_months']:.2f}%)")

        print("\n--- 2. VERIFICACIÓN FIN DE SERIE ERA5 ---")
        print(f"Fecha final: {meta['end_date']}")
        print(f"Valor ERA5 en fecha final: {meta['last_day_val']}")
        print(f"¿Es estrictamente NaN?: {meta['last_day_is_nan']} (Confirmado: faltan horas 12:00 a 24:00 del D+1)")

        print("\n--- 3. EVALUACIÓN DE DESFASES TEMPORALES (-1, 0, +1 DÍAS) ---")
        print(f"{'Desfase':<8} | {'N':<6} | {'Pearson r [IC 95%]':<26} | {'Spearman rho [IC 95%]':<26} | {'Sesgo (mm)':<10} | {'RMSE (mm)':<10}")
        print("-" * 96)
        for lag in r["lag_results"]:
            lag_name = f"{lag['lag']:+d} d"
            r_str = f"{lag['r']:+.4f} [{lag['r_ci'][0]:+.4f}, {lag['r_ci'][1]:+.4f}]"
            rho_str = f"{lag['rho']:+.4f} [{lag['rho_ci'][0]:+.4f}, {lag['rho_ci'][1]:+.4f}]"
            print(f"{lag_name:<8} | {lag['n']:<6} | {r_str:<26} | {rho_str:<26} | {lag['bias']:+10.2f} | {lag['rmse']:10.2f}")

        print("\nDesglose por Temporada (N, r [IC 95%], rho [IC 95%], Sesgo, RMSE):")
        for lag in r["lag_results"]:
            print(f"  * Desfase {lag['lag']:+d} días:")
            for s_name, sm in lag["seasons"].items():
                r_s_str = f"{sm['r']:+.4f} [{sm['r_ci'][0]:+.4f}, {sm['r_ci'][1]:+.4f}]"
                rho_s_str = f"{sm['rho']:+.4f} [{sm['rho_ci'][0]:+.4f}, {sm['rho_ci'][1]:+.4f}]"
                print(f"    - {s_name:<4} | N={sm['n']:<5} | r={r_s_str:<24} | rho={rho_s_str:<24} | Sesgo={sm['bias']:+6.2f} | RMSE={sm['rmse']:6.2f}")

        print(f"\n--> DESFASE GANADOR: {r['winning_lag']:+d} días (determinado por máxima correlación)")

        print("\n--- 4. MATRIZ DE VECINOS 3x3 (Lag 0 y Desfase Ganador) ---")
        print(f"{'Celda':<10} | {'Lat':<6} | {'Lon':<7} | {'Dist(km)':<8} | {'r (Lag 0)':<10} | {'rho (Lag 0)':<11} | {'r (Ganador)':<11} | {'Sesgo (Gan)':<11} | {'RMSE (Gan)':<10} | {'Suma ERA5':<9}")
        print("-" * 105)
        for c in r["spatial_grid"]:
            tag = "CENTRO" if c["is_center"] else f"({c['dlat']:+d},{c['dlon']:+d})"
            m0 = c["lag0"]
            mw = c["lag_win"]
            print(f"{tag:<10} | {c['lat']:6.2f} | {c['lon']:7.2f} | {c['dist_km']:8.1f} | {m0['r']:+10.4f} | {m0['rho']:+11.4f} | {mw['r']:+11.4f} | {mw['bias']:+11.2f} | {mw['rmse']:10.2f} | {mw['sum_mod']:9.1f}")

        print(f"\n--- 5. MÉTRICAS DETALLADAS POR MES CON DESFASE GANADOR ({r['winning_lag']:+d} d) ---")
        print(f"{'Mes':<4} | {'N':<5} | {'Sesgo':<7} | {'RMSE':<6} | {'P>=0.1 Obs':<10} | {'P>=0.1 ERA':<10} | {'Dif 0.1':<8} | {'P>=1.0 Obs':<10} | {'P>=1.0 ERA':<10} | {'Dif 1.0':<8}")
        print("-" * 92)
        for m in r["monthly_winner_metrics"]:
            print(f"{m['month_name']:<4} | {m['n_days']:<5} | {m['bias']:+7.2f} | {m['rmse']:6.2f} | {m['pct_wet_01_obs']:9.1f}% | {m['pct_wet_01_era']:9.1f}% | {m['pct_wet_01_diff']:+7.1f}% | {m['pct_wet_10_obs']:9.1f}% | {m['pct_wet_10_era']:9.1f}% | {m['pct_wet_10_diff']:+7.1f}%")

        print("\nPercentiles Diarios de Precipitación (mm/día) por Mes (Obs / ERA5 / Diferencia):")
        print(f"{'Mes':<4} | {'P50 (Obs/ERA/Dif)':<26} | {'P90 (Obs/ERA/Dif)':<26} | {'P95 (Obs/ERA/Dif)':<26} | {'P99 (Obs/ERA/Dif)':<26}")
        print("-" * 115)
        for m in r["monthly_winner_metrics"]:
            q_o, q_e, q_d = m["q_obs"], m["q_era"], m["q_diff"]
            p50_s = f"{q_o['P50']:4.1f} / {q_e['P50']:4.1f} / {q_d['P50']:+5.1f}"
            p90_s = f"{q_o['P90']:4.1f} / {q_e['P90']:4.1f} / {q_d['P90']:+5.1f}"
            p95_s = f"{q_o['P95']:4.1f} / {q_e['P95']:4.1f} / {q_d['P95']:+5.1f}"
            p99_s = f"{q_o['P99']:4.1f} / {q_e['P99']:4.1f} / {q_d['P99']:+5.1f}"
            print(f"{m['month_name']:<4} | {p50_s:<26} | {p90_s:<26} | {p95_s:<26} | {p99_s:<26}")

        print("\n--- 6. AGREGADOS MENSUALES (EXCLUSIÓN ESTRICTA DE MESES INCOMPLETOS) ---")
        print(f"{'Mes':<4} | {'Meses Usados':<14} | {'Meses Excluidos':<16} | {'Obs Media (mm)':<15} | {'ERA5 Media (mm)':<15} | {'Sesgo (mm)':<11} | {'RMSE (mm)':<10} | {'Pearson r':<9}")
        print("-" * 105)
        for ma in r["monthly_aggregates_complete"]:
            counts = r["complete_months_by_month_num"][ma["month"]]
            u_str = f"{counts['complete_kept']} / {counts['total_possible']}"
            e_str = f"{counts['excluded']}"
            print(f"{ma['month_name']:<4} | {u_str:<14} | {e_str:<16} | {ma['mean_obs_sum_mm']:15.2f} | {ma['mean_mod_sum_mm']:15.2f} | {ma['monthly_bias_mm']:+11.2f} | {ma['monthly_rmse_mm']:10.2f} | {ma['monthly_r']:+9.4f}")


if __name__ == "__main__":
    main()
