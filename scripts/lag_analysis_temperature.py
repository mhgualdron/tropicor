#!/usr/bin/env python3
"""Empirical Lag, Window, and Orographic Analysis for Daily Temperature (ERA5 vs DHIME).

Tests daily lag offsets (-1, 0, +1 days) and candidate sampling windows:
- candidata 00–24 UTC (calendar day UTC)
- candidata 12–12 UTC (07:00–07:00 local cycle)
- candidata 05–05 UTC (día civil local Colombia UTC-5)
separately for daily maximum temperature (Tmax) and daily minimum temperature (Tmin).

Evaluates:
1. Primary lag and window metrics across full period and seasons (DEF, MAM, JJA, SON)
   with 95% bootstrap confidence intervals (1000 resamples).
2. Paired correlation differences (lag 0 vs lag -1, lag 0 vs lag +1) with 30-day block
   bootstrap (1000 resamples) on raw values and monthly climatological anomalies.
3. 3x3 spatial neighborhood analysis (cell elevation, delta_z, raw/adjusted bias).
4. ERA5 internal lapse rate regression and sensitivity analysis (Gamma 4.0 to 8.0 °C/km).

Outputs:
    local/reports/lag_analysis_temperature.csv
    local/reports/temperature_paired_lag_bootstrap.csv
    local/reports/temperature_grid_3x3.csv
    local/reports/temperature_lapse_rate_sensitivity.csv
    local/reports/temperature_window_diff_bootstrap.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from tropicor.downscale.orography import (
    STANDARD_GRAVITY,
    STANDARD_LAPSE_RATE,
    geopotential_to_height,
    lapse_rate_temperature_correction,
)
from tropicor.io.era5 import ERA5Adapter
from tropicor.io.ideam import read_dhime_daily_temperature

# Fixed seed for the paired window-difference bootstrap (recorded in the output).
WINDOW_DIFF_SEED = 42


def compute_bootstrap_ci(
    obs: np.ndarray,
    mod: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Compute 95% bootstrap confidence intervals for Pearson r and Spearman rho."""
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
        r_boot[i] = (
            stats.pearsonr(o_b, m_b)[0]
            if np.std(o_b) > 1e-12 and np.std(m_b) > 1e-12
            else 0.0
        )
        rho_boot[i] = (
            stats.spearmanr(o_b, m_b)[0]
            if np.std(o_b) > 1e-12 and np.std(m_b) > 1e-12
            else 0.0
        )
    ci_r = (float(np.percentile(r_boot, 2.5)), float(np.percentile(r_boot, 97.5)))
    ci_rho = (float(np.percentile(rho_boot, 2.5)), float(np.percentile(rho_boot, 97.5)))
    return ci_r, ci_rho


def block_bootstrap_corr_diff(
    obs: np.ndarray,
    mod_0: np.ndarray,
    mod_other: np.ndarray,
    block_size: int = 30,
    n_boot: int = 1000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute paired correlation difference r(lag 0) - r(other) with 30-day block bootstrap CI.

    Returns:
        (mean_diff, ci_lower, ci_upper)
    """
    n = len(obs)
    if n < block_size or np.std(obs) < 1e-12:
        return 0.0, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))
    diffs = np.empty(n_boot)
    max_start = max(1, n - block_size)

    for b in range(n_boot):
        starts = rng.integers(0, max_start, size=n_blocks)
        indices = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]
        o_b = obs[indices]
        m0_b = mod_0[indices]
        mo_b = mod_other[indices]

        std_o = np.std(o_b)
        std_m0 = np.std(m0_b)
        std_mo = np.std(mo_b)

        r0 = stats.pearsonr(o_b, m0_b)[0] if (std_o > 1e-12 and std_m0 > 1e-12) else 0.0
        ro = stats.pearsonr(o_b, mo_b)[0] if (std_o > 1e-12 and std_mo > 1e-12) else 0.0
        diffs[b] = r0 - ro

    mean_diff = float(np.mean(diffs))
    ci_lower = float(np.percentile(diffs, 2.5))
    ci_upper = float(np.percentile(diffs, 97.5))
    return round(mean_diff, 4), round(ci_lower, 4), round(ci_upper, 4)


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r returning 0.0 for zero-variance inputs."""
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def block_bootstrap_window_diff(
    obs: np.ndarray,
    mod_a: np.ndarray,
    mod_b: np.ndarray,
    block_size: int = 30,
    n_boot: int = 1000,
    seed: int = WINDOW_DIFF_SEED,
) -> Tuple[float, float, float]:
    """Paired difference r(obs, mod_a) - r(obs, mod_b) with block bootstrap CI.

    Both windows are evaluated on the same common sample and the same resampled
    blocks of ``block_size`` consecutive observations (30-day blocks). The point
    estimate is computed on the original data (not the bootstrap mean); the CI is
    the 2.5-97.5 percentile interval of the bootstrap differences.

    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    n = len(obs)
    point = _safe_pearson(obs, mod_a) - _safe_pearson(obs, mod_b)
    if n < block_size:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))
    max_start = max(1, n - block_size)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, max_start, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]
        o_b = obs[idx]
        diffs[b] = _safe_pearson(o_b, mod_a[idx]) - _safe_pearson(o_b, mod_b[idx])
    return (
        point,
        float(np.percentile(diffs, 2.5)),
        float(np.percentile(diffs, 97.5)),
    )


def compute_monthly_anomalies(s: pd.Series) -> pd.Series:
    """Subtract calendar monthly mean from series to yield anomalies."""
    monthly_means = s.groupby(s.index.month).transform("mean")
    return s - monthly_means


def get_season(month: int) -> str:
    """Map calendar month to climatological season (DEF, MAM, JJA, SON)."""
    if month in (12, 1, 2):
        return "DEF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def evaluate_single_station(
    station_id: str,
    station_name: str,
    dhime_ds: Any,
    era5_files: List[str],
    static_file: Path,
    start_date: str,
    end_date: str,
    audit_elevation: float,
    audit_lat: float,
    audit_lon: float,
    lapse_rate: float = STANDARD_LAPSE_RATE,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    """Evaluate candidate lags, windows, block bootstrap differences, 3x3 grid, and sensitivity."""
    candidate_windows = [
        ("candidata 00–24 UTC", 0),
        ("candidata 12–12 UTC", 12),
        ("candidata 05–05 UTC (día civil local)", 5),
    ]
    candidate_lags = [-1, 0, 1]

    common_idx = pd.date_range(start_date, end_date, freq="D")

    s_obs_tmax = (
        dhime_ds["tasmax"].sel(station=station_id).to_series().reindex(common_idx)
    )
    s_obs_tmin = (
        dhime_ds["tasmin"].sel(station=station_id).to_series().reindex(common_idx)
    )

    if not static_file.exists():
        raise FileNotFoundError(f"Static file not found: {static_file}")

    # Read static grid
    with ERA5Adapter(static_file) as static_adapter:
        ds_stat = static_adapter.dataset
        lats_stat = ds_stat.latitude.values
        lons_stat = ds_stat.longitude.values
        ilat_c = int(np.argmin(np.abs(lats_stat - audit_lat)))
        ilon_c = int(np.argmin(np.abs(lons_stat - audit_lon)))
        grid_lat = float(lats_stat[ilat_c])
        grid_lon = float(lons_stat[ilon_c])

        z_var = static_adapter._resolve_variable_name("z")
        z_raw = float(
            ds_stat[z_var].sel(latitude=grid_lat, longitude=grid_lon).squeeze().values
        )
        elev_model_val = float(geopotential_to_height(z_raw, STANDARD_GRAVITY))
        lsm_val = static_adapter.get_land_sea_mask(grid_lat, grid_lon)

    delta_z = audit_elevation - elev_model_val

    primary_records: List[Dict[str, Any]] = []
    paired_diff_records: List[Dict[str, Any]] = []
    window_diff_records: List[Dict[str, Any]] = []
    grid_3x3_records: List[Dict[str, Any]] = []
    sensitivity_records: List[Dict[str, Any]] = []
    seasonal_center_records: List[Dict[str, Any]] = []

    with ERA5Adapter(era5_files) as era5_adapter:
        # Pre-extract series for all candidate windows and lags at center cell
        daily_dfs: Dict[Tuple[int, int], pd.DataFrame] = {}
        for _, win_start in candidate_windows:
            for lag in candidate_lags:
                df_win = era5_adapter.get_daily_temperature(
                    latitude=grid_lat,
                    longitude=grid_lon,
                    window_start_utc=win_start,
                    min_hours=24,
                    lag_days=lag,
                    convert_to_celsius=True,
                ).reindex(common_idx)
                daily_dfs[(win_start, lag)] = df_win

        # 1. Primary evaluation across windows, lags, and seasons
        for win_label, win_start in candidate_windows:
            for lag in candidate_lags:
                df_era5_daily = daily_dfs[(win_start, lag)]

                for var_key, s_obs in [("Tmax", s_obs_tmax), ("Tmin", s_obs_tmin)]:
                    s_mod_raw = df_era5_daily[var_key.lower()]
                    df_sync = pd.concat(
                        [s_obs.rename("obs"), s_mod_raw.rename("mod_raw")],
                        axis=1,
                    ).dropna()

                    if df_sync.empty:
                        continue

                    temporal_groups = [("Todo", df_sync)]
                    for season_name in ["DEF", "MAM", "JJA", "SON"]:
                        season_mask = df_sync.index.map(lambda dt: get_season(dt.month))
                        sub_season = df_sync[season_mask == season_name]
                        temporal_groups.append((season_name, sub_season))

                    for subset_name, df_sub in temporal_groups:
                        n_sync = len(df_sub)
                        if n_sync < 3:
                            continue

                        y_o = df_sub["obs"].to_numpy(dtype=float)
                        y_m_raw = df_sub["mod_raw"].to_numpy(dtype=float)

                        y_m_adj = lapse_rate_temperature_correction(
                            y_m_raw,
                            station_elevation=audit_elevation,
                            model_elevation=elev_model_val,
                            lapse_rate=lapse_rate,
                        )

                        r_raw = (
                            float(stats.pearsonr(y_o, y_m_raw)[0])
                            if np.std(y_o) > 1e-12 and np.std(y_m_raw) > 1e-12
                            else 0.0
                        )
                        rho_raw = (
                            float(stats.spearmanr(y_o, y_m_raw)[0])
                            if np.std(y_o) > 1e-12 and np.std(y_m_raw) > 1e-12
                            else 0.0
                        )
                        ci_r, ci_rho = compute_bootstrap_ci(y_o, y_m_raw, n_boot=1000)

                        raw_bias = float(np.mean(y_m_raw - y_o))
                        raw_rmse = float(np.sqrt(np.mean((y_m_raw - y_o) ** 2)))
                        adj_bias = float(np.mean(y_m_adj - y_o))
                        adj_rmse = float(np.sqrt(np.mean((y_m_adj - y_o) ** 2)))
                        r_adj = (
                            float(stats.pearsonr(y_o, y_m_adj)[0])
                            if np.std(y_o) > 1e-12 and np.std(y_m_adj) > 1e-12
                            else 0.0
                        )

                        primary_records.append(
                            {
                                "station_name": station_name,
                                "station_id": station_id,
                                "variable": var_key,
                                "window": win_label,
                                "window_start_utc": win_start,
                                "lag_days": lag,
                                "season": subset_name,
                                "n_days": n_sync,
                                "grid_lat": round(grid_lat, 3),
                                "grid_lon": round(grid_lon, 3),
                                "elevation_obs_masl": round(audit_elevation, 1),
                                "elevation_model_masl": round(elev_model_val, 2),
                                "delta_z_m": round(delta_z, 2),
                                "land_sea_mask": lsm_val,
                                "pearson_r": round(r_raw, 4),
                                "pearson_ci_lower": round(ci_r[0], 4),
                                "pearson_ci_upper": round(ci_r[1], 4),
                                "spearman_rho": round(rho_raw, 4),
                                "spearman_ci_lower": round(ci_rho[0], 4),
                                "spearman_ci_upper": round(ci_rho[1], 4),
                                "bias_raw_degC": round(raw_bias, 3),
                                "rmse_raw_degC": round(raw_rmse, 3),
                                "pearson_r_adj": round(r_adj, 4),
                                "bias_adjusted_degC": round(adj_bias, 3),
                                "rmse_adjusted_degC": round(adj_rmse, 3),
                            }
                        )

        # 2. Paired correlation differences with 30-day block bootstrap
        for win_label, win_start in candidate_windows:
            df_lag0 = daily_dfs[(win_start, 0)]
            df_lag_neg = daily_dfs[(win_start, -1)]
            df_lag_pos = daily_dfs[(win_start, 1)]

            for var_key, s_obs in [("Tmax", s_obs_tmax), ("Tmin", s_obs_tmin)]:
                col = var_key.lower()
                df_paired = pd.concat(
                    [
                        s_obs.rename("obs"),
                        df_lag0[col].rename("mod_0"),
                        df_lag_neg[col].rename("mod_neg1"),
                        df_lag_pos[col].rename("mod_pos1"),
                    ],
                    axis=1,
                ).dropna()

                if len(df_paired) < 60:
                    continue

                y_obs_arr = df_paired["obs"].to_numpy()
                y_m0_arr = df_paired["mod_0"].to_numpy()
                y_m_neg_arr = df_paired["mod_neg1"].to_numpy()
                y_m_pos_arr = df_paired["mod_pos1"].to_numpy()

                r0_raw = float(stats.pearsonr(y_obs_arr, y_m0_arr)[0])
                r_neg_raw = float(stats.pearsonr(y_obs_arr, y_m_neg_arr)[0])
                r_pos_raw = float(stats.pearsonr(y_obs_arr, y_m_pos_arr)[0])

                # Raw block bootstrap
                diff_neg_raw, ci_neg_l, ci_neg_u = block_bootstrap_corr_diff(
                    y_obs_arr, y_m0_arr, y_m_neg_arr, block_size=30, n_boot=1000
                )
                diff_pos_raw, ci_pos_l, ci_pos_u = block_bootstrap_corr_diff(
                    y_obs_arr, y_m0_arr, y_m_pos_arr, block_size=30, n_boot=1000
                )

                # Monthly anomalies block bootstrap
                s_obs_anom = compute_monthly_anomalies(df_paired["obs"])
                s_m0_anom = compute_monthly_anomalies(df_paired["mod_0"])
                s_m_neg_anom = compute_monthly_anomalies(df_paired["mod_neg1"])
                s_m_pos_anom = compute_monthly_anomalies(df_paired["mod_pos1"])

                yo_anom = s_obs_anom.to_numpy()
                ym0_anom = s_m0_anom.to_numpy()
                ym_neg_anom = s_m_neg_anom.to_numpy()
                ym_pos_anom = s_m_pos_anom.to_numpy()

                r0_anom = float(stats.pearsonr(yo_anom, ym0_anom)[0])
                r_neg_anom = float(stats.pearsonr(yo_anom, ym_neg_anom)[0])
                r_pos_anom = float(stats.pearsonr(yo_anom, ym_pos_anom)[0])

                diff_neg_anom, ci_neg_l_anom, ci_neg_u_anom = block_bootstrap_corr_diff(
                    yo_anom, ym0_anom, ym_neg_anom, block_size=30, n_boot=1000
                )
                diff_pos_anom, ci_pos_l_anom, ci_pos_u_anom = block_bootstrap_corr_diff(
                    yo_anom, ym0_anom, ym_pos_anom, block_size=30, n_boot=1000
                )

                paired_diff_records.append(
                    {
                        "station_name": station_name,
                        "station_id": station_id,
                        "variable": var_key,
                        "window": win_label,
                        "data_type": "Crudo",
                        "n_days": len(df_paired),
                        "r_lag0": round(r0_raw, 4),
                        "r_lag_minus1": round(r_neg_raw, 4),
                        "diff_r_minus1 (r0 - r_neg)": diff_neg_raw,
                        "ci_95_diff_minus1": f"[{ci_neg_l:+.4f}, {ci_neg_u:+.4f}]",
                        "r_lag_plus1": round(r_pos_raw, 4),
                        "diff_r_plus1 (r0 - r_pos)": diff_pos_raw,
                        "ci_95_diff_plus1": f"[{ci_pos_l:+.4f}, {ci_pos_u:+.4f}]",
                    }
                )
                paired_diff_records.append(
                    {
                        "station_name": station_name,
                        "station_id": station_id,
                        "variable": var_key,
                        "window": win_label,
                        "data_type": "Anomalías mensuales",
                        "n_days": len(df_paired),
                        "r_lag0": round(r0_anom, 4),
                        "r_lag_minus1": round(r_neg_anom, 4),
                        "diff_r_minus1 (r0 - r_neg)": diff_neg_anom,
                        "ci_95_diff_minus1": (
                            f"[{ci_neg_l_anom:+.4f}, {ci_neg_u_anom:+.4f}]"
                        ),
                        "r_lag_plus1": round(r_pos_anom, 4),
                        "diff_r_plus1 (r0 - r_pos)": diff_pos_anom,
                        "ci_95_diff_plus1": (
                            f"[{ci_pos_l_anom:+.4f}, {ci_pos_u_anom:+.4f}]"
                        ),
                    }
                )

        # 2b. Paired window difference (00-24 UTC vs 05-05 UTC), 30-day block bootstrap
        for var_key, s_obs in [("Tmax", s_obs_tmax), ("Tmin", s_obs_tmin)]:
            col = var_key.lower()
            for lag in candidate_lags:
                df_w = pd.concat(
                    [
                        s_obs.rename("obs"),
                        daily_dfs[(0, lag)][col].rename("mod_00_24"),
                        daily_dfs[(5, lag)][col].rename("mod_05_05"),
                    ],
                    axis=1,
                ).dropna()
                season_idx = df_w.index.map(lambda dt: get_season(dt.month))
                w_groups = [("Todo", df_w)]
                for season_name in ["DEF", "MAM", "JJA", "SON"]:
                    w_groups.append((season_name, df_w[season_idx == season_name]))

                for season_name, df_s in w_groups:
                    if len(df_s) < 60:
                        continue
                    o_arr = df_s["obs"].to_numpy(dtype=float)
                    a_arr = df_s["mod_00_24"].to_numpy(dtype=float)
                    b_arr = df_s["mod_05_05"].to_numpy(dtype=float)
                    point, ci_lo, ci_hi = block_bootstrap_window_diff(
                        o_arr, a_arr, b_arr, block_size=30, n_boot=1000
                    )
                    window_diff_records.append(
                        {
                            "station_name": station_name,
                            "station_id": station_id,
                            "variable": var_key,
                            "lag_days": lag,
                            "season": season_name,
                            "n_days_common": len(df_s),
                            "r_00_24": round(_safe_pearson(o_arr, a_arr), 4),
                            "r_05_05": round(_safe_pearson(o_arr, b_arr), 4),
                            "delta_r_point (r_00-24 - r_05-05)": round(point, 4),
                            "ci95_lower": round(ci_lo, 4),
                            "ci95_upper": round(ci_hi, 4),
                            "ci_excludes_zero": bool(ci_lo > 0 or ci_hi < 0),
                            "block_size_days": 30,
                            "n_boot": 1000,
                            "seed": WINDOW_DIFF_SEED,
                        }
                    )

        # 3. 3x3 Grid Neighborhood Analysis & Center Cell Seasonal Breakdown
        station_grid_recs: List[Dict[str, Any]] = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                ilat = ilat_c + di
                ilon = ilon_c + dj
                lat_cell = float(lats_stat[ilat])
                lon_cell = float(lons_stat[ilon])

                z_cell_raw = float(
                    ds_stat[z_var]
                    .sel(latitude=lat_cell, longitude=lon_cell)
                    .squeeze()
                    .values
                )
                h_cell = float(geopotential_to_height(z_cell_raw, STANDARD_GRAVITY))
                dz_cell = audit_elevation - h_cell
                lsm_cell = static_adapter.get_land_sea_mask(lat_cell, lon_cell)

                # Daily temperature using civil day window (05-05 UTC, lag 0)
                df_cell_daily = era5_adapter.get_daily_temperature(
                    latitude=lat_cell,
                    longitude=lon_cell,
                    window_start_utc=5,
                    min_hours=24,
                    lag_days=0,
                    convert_to_celsius=True,
                ).reindex(common_idx)

                df_sync_all = pd.concat(
                    [
                        s_obs_tmax.rename("obs_max"),
                        s_obs_tmin.rename("obs_min"),
                        df_cell_daily["tmax"].rename("mod_max"),
                        df_cell_daily["tmin"].rename("mod_min"),
                    ],
                    axis=1,
                ).dropna()

                y_omax = df_sync_all["obs_max"].to_numpy(dtype=float)
                y_omin = df_sync_all["obs_min"].to_numpy(dtype=float)
                y_mmax = df_sync_all["mod_max"].to_numpy(dtype=float)
                y_mmin = df_sync_all["mod_min"].to_numpy(dtype=float)

                y_odtr = y_omax - y_omin
                y_mdtr = y_mmax - y_mmin

                y_mmax_adj = lapse_rate_temperature_correction(
                    y_mmax, audit_elevation, h_cell, lapse_rate=lapse_rate
                )
                y_mmin_adj = lapse_rate_temperature_correction(
                    y_mmin, audit_elevation, h_cell, lapse_rate=lapse_rate
                )
                y_mdtr_adj = y_mmax_adj - y_mmin_adj

                bias_tmax_raw = float(np.mean(y_mmax - y_omax))
                bias_tmax_adj = float(np.mean(y_mmax_adj - y_omax))
                rmse_tmax_raw = float(np.sqrt(np.mean((y_mmax - y_omax) ** 2)))
                rmse_tmax_adj = float(np.sqrt(np.mean((y_mmax_adj - y_omax) ** 2)))
                r_tmax = (
                    float(stats.pearsonr(y_omax, y_mmax)[0])
                    if len(y_omax) > 2
                    and np.std(y_omax) > 1e-12
                    and np.std(y_mmax) > 1e-12
                    else 0.0
                )

                bias_tmin_raw = float(np.mean(y_mmin - y_omin))
                bias_tmin_adj = float(np.mean(y_mmin_adj - y_omin))
                rmse_tmin_raw = float(np.sqrt(np.mean((y_mmin - y_omin) ** 2)))
                rmse_tmin_adj = float(np.sqrt(np.mean((y_mmin_adj - y_omin) ** 2)))
                r_tmin = (
                    float(stats.pearsonr(y_omin, y_mmin)[0])
                    if len(y_omin) > 2
                    and np.std(y_omin) > 1e-12
                    and np.std(y_mmin) > 1e-12
                    else 0.0
                )

                bias_dtr_raw = float(np.mean(y_mdtr - y_odtr))
                bias_dtr_adj = float(np.mean(y_mdtr_adj - y_odtr))
                rmse_dtr_raw = float(np.sqrt(np.mean((y_mdtr - y_odtr) ** 2)))
                rmse_dtr_adj = float(np.sqrt(np.mean((y_mdtr_adj - y_odtr) ** 2)))
                r_dtr = (
                    float(stats.pearsonr(y_odtr, y_mdtr)[0])
                    if len(y_odtr) > 2
                    and np.std(y_odtr) > 1e-12
                    and np.std(y_mdtr) > 1e-12
                    else 0.0
                )

                is_center = di == 0 and dj == 0

                # Seasonal breakdown in center cell
                if is_center:
                    seasonal_groups = [("Todo", df_sync_all)]
                    for season_name in ["DEF", "MAM", "JJA", "SON"]:
                        mask = df_sync_all.index.map(lambda dt: get_season(dt.month))
                        sub_s = df_sync_all[mask == season_name]
                        seasonal_groups.append((season_name, sub_s))

                    for s_lbl, df_sub in seasonal_groups:
                        if len(df_sub) < 3:
                            continue
                        s_omax = df_sub["obs_max"].to_numpy(dtype=float)
                        s_omin = df_sub["obs_min"].to_numpy(dtype=float)
                        s_mmax = df_sub["mod_max"].to_numpy(dtype=float)
                        s_mmin = df_sub["mod_min"].to_numpy(dtype=float)
                        s_odtr = s_omax - s_omin
                        s_mdtr = s_mmax - s_mmin
                        s_mmax_adj = lapse_rate_temperature_correction(
                            s_mmax, audit_elevation, h_cell, lapse_rate=lapse_rate
                        )
                        s_mmin_adj = lapse_rate_temperature_correction(
                            s_mmin, audit_elevation, h_cell, lapse_rate=lapse_rate
                        )
                        s_mdtr_adj = s_mmax_adj - s_mmin_adj

                        seasonal_center_records.append(
                            {
                                "station_name": station_name,
                                "station_id": station_id,
                                "season": s_lbl,
                                "n_days": len(df_sub),
                                "tmax_bias_raw_degC": round(
                                    float(np.mean(s_mmax - s_omax)), 3
                                ),
                                "tmax_bias_adj_degC": round(
                                    float(np.mean(s_mmax_adj - s_omax)), 3
                                ),
                                "tmax_rmse_raw_degC": round(
                                    float(np.sqrt(np.mean((s_mmax - s_omax) ** 2))), 3
                                ),
                                "tmax_rmse_adj_degC": round(
                                    float(np.sqrt(np.mean((s_mmax_adj - s_omax) ** 2))),
                                    3,
                                ),
                                "tmin_bias_raw_degC": round(
                                    float(np.mean(s_mmin - s_omin)), 3
                                ),
                                "tmin_bias_adj_degC": round(
                                    float(np.mean(s_mmin_adj - s_omin)), 3
                                ),
                                "tmin_rmse_raw_degC": round(
                                    float(np.sqrt(np.mean((s_mmin - s_omin) ** 2))), 3
                                ),
                                "tmin_rmse_adj_degC": round(
                                    float(np.sqrt(np.mean((s_mmin_adj - s_omin) ** 2))),
                                    3,
                                ),
                                "dtr_bias_raw_degC": round(
                                    float(np.mean(s_mdtr - s_odtr)), 3
                                ),
                                "dtr_bias_adj_degC": round(
                                    float(np.mean(s_mdtr_adj - s_odtr)), 3
                                ),
                                "dtr_rmse_raw_degC": round(
                                    float(np.sqrt(np.mean((s_mdtr - s_odtr) ** 2))), 3
                                ),
                                "dtr_rmse_adj_degC": round(
                                    float(np.sqrt(np.mean((s_mdtr_adj - s_odtr) ** 2))),
                                    3,
                                ),
                            }
                        )

                station_grid_recs.append(
                    {
                        "station_name": station_name,
                        "station_id": station_id,
                        "station_elevation_masl": round(audit_elevation, 1),
                        "di": di,
                        "dj": dj,
                        "grid_lat": round(lat_cell, 3),
                        "grid_lon": round(lon_cell, 3),
                        "is_center": is_center,
                        "cell_elevation_masl": round(h_cell, 1),
                        "delta_z_m": round(dz_cell, 1),
                        "abs_delta_z_m": round(abs(dz_cell), 1),
                        "land_sea_mask": lsm_cell,
                        "tmax_r": round(r_tmax, 4),
                        "tmax_bias_raw_degC": round(bias_tmax_raw, 3),
                        "tmax_bias_adj_degC": round(bias_tmax_adj, 3),
                        "tmax_rmse_raw_degC": round(rmse_tmax_raw, 3),
                        "tmax_rmse_adj_degC": round(rmse_tmax_adj, 3),
                        "tmin_r": round(r_tmin, 4),
                        "tmin_bias_raw_degC": round(bias_tmin_raw, 3),
                        "tmin_bias_adj_degC": round(bias_tmin_adj, 3),
                        "tmin_rmse_raw_degC": round(rmse_tmin_raw, 3),
                        "tmin_rmse_adj_degC": round(rmse_tmin_adj, 3),
                        "dtr_r": round(r_dtr, 4),
                        "dtr_bias_raw_degC": round(bias_dtr_raw, 3),
                        "dtr_bias_adj_degC": round(bias_dtr_adj, 3),
                        "dtr_rmse_raw_degC": round(rmse_dtr_raw, 3),
                        "dtr_rmse_adj_degC": round(rmse_dtr_adj, 3),
                    }
                )

        # Identify cell minimizing |Δz| and add selection warning
        min_dz_rec = min(station_grid_recs, key=lambda r: r["abs_delta_z_m"])
        for r in station_grid_recs:
            is_min = r["di"] == min_dz_rec["di"] and r["dj"] == min_dz_rec["dj"]
            r["is_min_abs_delta_z"] = is_min
            if is_min and not r["is_center"]:
                r["warning_selection"] = (
                    "AVISO: Celda minimiza |Δz| pero NO coincide con la celda más cercana (centro). "
                    "No se selecciona automáticamente."
                )
            elif r["is_center"]:
                r["warning_selection"] = (
                    "Celda más cercana / centro (referencia estándar)."
                    if is_min
                    else f"Celda más cercana / centro. No minimiza |Δz| (menor |Δz| en di={min_dz_rec['di']}, dj={min_dz_rec['dj']})."
                )
            else:
                r["warning_selection"] = ""

        grid_3x3_records.extend(station_grid_recs)

        # 4. Sensitivity of adjusted bias to lapse rate (Gamma: 4.0 to 8.0 °C/km)
        # Using center grid cell, lag 0, window 05-05 UTC
        df_center_0 = daily_dfs[(5, 0)]
        for var_key, s_obs in [("Tmax", s_obs_tmax), ("Tmin", s_obs_tmin)]:
            s_mod_raw = df_center_0[var_key.lower()]
            df_sync = pd.concat(
                [s_obs.rename("obs"), s_mod_raw.rename("mod")], axis=1
            ).dropna()
            y_o = df_sync["obs"].to_numpy()
            y_m = df_sync["mod"].to_numpy()
            raw_b = float(np.mean(y_m - y_o))

            gamma_values = [4.0, 4.5, 5.0, 5.24, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
            for g_km in gamma_values:
                g_m = g_km / 1000.0
                y_adj = lapse_rate_temperature_correction(
                    y_m, audit_elevation, elev_model_val, lapse_rate=g_m
                )
                b_adj = float(np.mean(y_adj - y_o))
                rmse_adj = float(np.sqrt(np.mean((y_adj - y_o) ** 2)))
                delta_t = -g_m * delta_z

                sensitivity_records.append(
                    {
                        "station_name": station_name,
                        "station_id": station_id,
                        "variable": var_key,
                        "lapse_rate_degC_per_km": g_km,
                        "delta_z_m": round(delta_z, 1),
                        "delta_T_degC": round(delta_t, 3),
                        "bias_raw_degC": round(raw_b, 3),
                        "bias_adjusted_degC": round(b_adj, 3),
                        "rmse_adjusted_degC": round(rmse_adj, 3),
                    }
                )

    return (
        primary_records,
        paired_diff_records,
        grid_3x3_records,
        sensitivity_records,
        seasonal_center_records,
        window_diff_records,
    )


def compute_era5_spatial_lapse_rate(
    static_file: Path,
    era5_files: List[str],
) -> Dict[str, Any]:
    """Compute intrinsic lapse rate across ERA5 grid cells by linear regression for Tmax, Tmin, Tmean."""
    with ERA5Adapter(static_file) as static_adapter:
        ds_stat = static_adapter.dataset
        z_var = static_adapter._resolve_variable_name("z")
        z_grid = (ds_stat[z_var].squeeze().values) / STANDARD_GRAVITY  # shape (5, 5)

    with ERA5Adapter(era5_files) as era5_adapter:
        ds_t = era5_adapter.dataset
        var_t = era5_adapter._resolve_variable_name("t2m")
        tdim = "time" if "time" in ds_t.dims else "valid_time"
        t2m_c = ds_t[var_t] - 273.15
        t_mean_grid = t2m_c.mean(dim=tdim).values
        tmax_mean_grid = t2m_c.resample({tdim: "1D"}).max().mean(dim=tdim).values
        tmin_mean_grid = t2m_c.resample({tdim: "1D"}).min().mean(dim=tdim).values

    z_flat = z_grid.ravel()
    out: Dict[str, Any] = {
        "z_min_masl": round(float(z_flat.min()), 1),
        "z_max_masl": round(float(z_flat.max()), 1),
        "z_range_m": round(float(z_flat.max() - z_flat.min()), 1),
    }

    for key, arr in [
        ("Tmax", tmax_mean_grid),
        ("Tmin", tmin_mean_grid),
        ("Tmean", t_mean_grid),
    ]:
        arr_flat = arr.ravel()
        slope, intercept, r_val, p_val, std_err = stats.linregress(z_flat, arr_flat)
        gamma = -slope * 1000.0
        out[key] = {
            "mean_min_degC": round(float(arr_flat.min()), 2),
            "mean_max_degC": round(float(arr_flat.max()), 2),
            "slope_degC_per_m": slope,
            "empirical_lapse_rate_degC_per_km": round(float(gamma), 2),
            "pearson_r": round(float(r_val), 4),
            "r_squared": round(float(r_val**2), 4),
            "p_value": float(p_val),
            "standard_error": float(std_err),
        }

    return out


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analizador empírico de temperatura (desfases, ventanas, celdas y gradiente)."
    )
    parser.add_argument(
        "--station",
        type=str,
        default="all",
        help="Código de estación ('21205710', '23215030', o 'all').",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="1990-01-01",
        help="Fecha inicial (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="1994-12-31",
        help="Fecha final (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("local/reports/lag_analysis_temperature.csv"),
        help="Ruta destino para tabla de resultados primarios.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspecciona el plan de análisis sin ejecutar cómputos.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()

    if args.dry_run:
        print("DISEÑO DE ANÁLISIS DE TEMPERATURA (DRY RUN)")
        print("=" * 88)
        print(f"Estación:            {args.station}")
        print(f"Periodo:             {args.start_date} a {args.end_date}")
        print("Ventanas:            00-24 UTC, 12-12 UTC, 05-05 UTC (día civil local)")
        print("Desfases:            -1 día, 0 días, +1 día")
        print("Bootstrap pareado:   Bloques de 30 días (1000 réplicas)")
        print("Evaluación espacial: Matriz 3x3 de celdas por estación")
        print("Regresión ERA5:      Gradiente intrínseco de celda")
        print("Sensibilidad:        Gradientes Gamma 4.0 a 8.0 °C/km")
        print("=" * 88)
        return

    stations_config = [
        {
            "id": "21205710",
            "name": "Jardín Botánico",
            "csv_max": Path("local/newFiles/TempMaxima_JardinBotanico.csv"),
            "csv_min": Path("local/newFiles/TempMinima_JardinBotanico.csv"),
            "era5_dir": Path("local/era5_hourly/21205710"),
            "static_file": Path("local/era5_hourly/21205710/era5_static.nc"),
        },
        {
            "id": "23215030",
            "name": "Aguas Claras",
            "csv_max": Path("local/newFiles/TempMaxima_AguasClaras.csv"),
            "csv_min": Path("local/newFiles/TempMinima_AguasClaras.csv"),
            "era5_dir": Path("local/era5_hourly/23215030"),
            "static_file": Path("local/era5_hourly/23215030/era5_static.nc"),
        },
    ]

    if args.station != "all":
        stations_config = [s for s in stations_config if s["id"] == args.station]
        if not stations_config:
            print(f"Error: Estación '{args.station}' no reconocida.", file=sys.stderr)
            sys.exit(1)

    audit_df = pd.read_csv("local/station_coords_audit.csv")

    all_primary: List[Dict[str, Any]] = []
    all_paired: List[Dict[str, Any]] = []
    all_grid_3x3: List[Dict[str, Any]] = []
    all_sens: List[Dict[str, Any]] = []
    all_seasonal: List[Dict[str, Any]] = []
    all_window_diff: List[Dict[str, Any]] = []
    era5_lapse_reports: Dict[str, Any] = {}

    for cfg in stations_config:
        st_id = str(cfg["id"])
        st_name = str(cfg["name"])
        print(f"\nProcesando estación: {st_name} [{st_id}]...")

        audit_row = audit_df[audit_df["codigo"] == int(st_id)].iloc[0]
        audit_elevation = float(audit_row["altitud_final"])
        audit_lat = float(audit_row["lat_cne"])
        audit_lon = float(audit_row["lon_cne"])

        ds_dhime = read_dhime_daily_temperature(
            filepath=cfg["csv_max"],
            filepath_min=cfg["csv_min"],
        )

        era5_dir = Path(cfg["era5_dir"])
        era5_files = sorted(str(p) for p in era5_dir.glob("era5_2m_temperature_*.nc"))
        if not era5_files:
            print(
                f"Error: No se encontraron archivos de temperatura en {era5_dir}",
                file=sys.stderr,
            )
            sys.exit(1)

        static_file = Path(cfg["static_file"])

        p_recs, pair_recs, g3_recs, sens_recs, seas_recs, win_recs = (
            evaluate_single_station(
                station_id=st_id,
                station_name=st_name,
                dhime_ds=ds_dhime,
                era5_files=era5_files,
                static_file=static_file,
                start_date=args.start_date,
                end_date=args.end_date,
                audit_elevation=audit_elevation,
                audit_lat=audit_lat,
                audit_lon=audit_lon,
            )
        )
        all_primary.extend(p_recs)
        all_paired.extend(pair_recs)
        all_grid_3x3.extend(g3_recs)
        all_sens.extend(sens_recs)
        all_seasonal.extend(seas_recs)
        all_window_diff.extend(win_recs)

        # Intrinsic lapse rate in ERA5 domain
        lapse_info = compute_era5_spatial_lapse_rate(static_file, era5_files)
        era5_lapse_reports[st_name] = lapse_info

    # Save reports
    reports_dir = Path("local/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    df_primary = pd.DataFrame(all_primary)
    df_primary.to_csv(args.output_csv, index=False)
    print(f"Reporte primario guardado en: {args.output_csv} ({len(df_primary)} filas)")

    df_paired = pd.DataFrame(all_paired)
    paired_csv = reports_dir / "temperature_paired_lag_bootstrap.csv"
    df_paired.to_csv(paired_csv, index=False)
    print(f"Reporte pareado bootstrap guardado en: {paired_csv}")

    df_grid = pd.DataFrame(all_grid_3x3)
    grid_csv = reports_dir / "temperature_grid_3x3.csv"
    df_grid.to_csv(grid_csv, index=False)
    print(f"Reporte 3x3 guardado en: {grid_csv}")

    df_sens = pd.DataFrame(all_sens)
    sens_csv = reports_dir / "temperature_lapse_rate_sensitivity.csv"
    df_sens.to_csv(sens_csv, index=False)
    print(f"Reporte de sensibilidad guardado en: {sens_csv}")

    df_seasonal = pd.DataFrame(all_seasonal)
    seasonal_csv = reports_dir / "temperature_center_cell_seasonal_biases.csv"
    df_seasonal.to_csv(seasonal_csv, index=False)
    print(f"Reporte estacional de celda central guardado en: {seasonal_csv}")

    df_wdiff = pd.DataFrame(all_window_diff)
    wdiff_csv = reports_dir / "temperature_window_diff_bootstrap.csv"
    df_wdiff.to_csv(wdiff_csv, index=False)
    print(f"Reporte de diferencia pareada entre ventanas guardado en: {wdiff_csv}")

    if not df_wdiff.empty:
        print("\n" + "=" * 88)
        print("DIFERENCIA PAREADA ENTRE VENTANAS: r(00-24 UTC) - r(05-05 UTC), IC95%")
        print(
            f"Bloques de 30 dias, 1000 replicas, semilla fija = {WINDOW_DIFF_SEED}; "
            "estimacion puntual sobre los datos originales (no media del bootstrap)"
        )
        print("=" * 88)
        lag0_all = df_wdiff[df_wdiff["lag_days"] == 0]
        for _, row in lag0_all[lag0_all["season"] == "Todo"].iterrows():
            print(
                f"  {row['station_name']} [{row['variable']}] lag 0, Todo: "
                f"N={row['n_days_common']}, r(00-24)={row['r_00_24']:.4f}, "
                f"r(05-05)={row['r_05_05']:.4f}, "
                f"delta_r={row['delta_r_point (r_00-24 - r_05-05)']:+.4f}, "
                f"IC95% [{row['ci95_lower']:+.4f}, {row['ci95_upper']:+.4f}]"
            )
        excl = df_wdiff[df_wdiff["ci_excludes_zero"]]
        print(
            f"\nCombinaciones (estacion x variable x lag x temporada) con IC que excluye cero: "
            f"{len(excl)} de {len(df_wdiff)}"
        )
        for _, row in excl.iterrows():
            print(
                f"  {row['station_name']} [{row['variable']}] lag {row['lag_days']:+d}, "
                f"{row['season']}: delta_r={row['delta_r_point (r_00-24 - r_05-05)']:+.4f}, "
                f"IC95% [{row['ci95_lower']:+.4f}, {row['ci95_upper']:+.4f}]"
            )
        max_abs_delta = float(df_wdiff["delta_r_point (r_00-24 - r_05-05)"].abs().max())
        max_abs_delta_lag0 = float(
            lag0_all["delta_r_point (r_00-24 - r_05-05)"].abs().max()
        )
        print(
            f"\nMax |delta_r| (lag 0, todas las temporadas): {max_abs_delta_lag0:.4f}"
        )
        print(f"Max |delta_r| (todos los lags y temporadas): {max_abs_delta:.4f}")

        # Consistency check against the r values of the primary (unpaired) table.
        keys = ["station_name", "variable", "lag_days", "season"]
        prim0 = df_primary[df_primary["window_start_utc"] == 0][keys + ["pearson_r"]]
        prim5 = df_primary[df_primary["window_start_utc"] == 5][keys + ["pearson_r"]]
        chk = df_wdiff.merge(prim0, on=keys).merge(
            prim5, on=keys, suffixes=("_p00", "_p05")
        )
        chk["d00"] = (chk["r_00_24"] - chk["pearson_r_p00"]).abs()
        chk["d05"] = (chk["r_05_05"] - chk["pearson_r_p05"]).abs()
        print(
            "\nVerificacion de consistencia con la tabla primaria (muestras no pareadas): "
            f"max |r_00-24 - r_primario| = {chk['d00'].max():.4f}, "
            f"max |r_05-05 - r_primario| = {chk['d05'].max():.4f}"
        )
        ac = chk[
            (chk["station_name"] == "Aguas Claras")
            & (chk["variable"] == "Tmin")
            & (chk["lag_days"] == 0)
            & (chk["season"] == "Todo")
        ]
        for _, row in ac.iterrows():
            print(
                "  Aguas Claras Tmin lag 0 Todo: "
                f"r_pareado(00-24)={row['r_00_24']:.4f} vs primario={row['pearson_r_p00']:.4f}; "
                f"r_pareado(05-05)={row['r_05_05']:.4f} vs primario={row['pearson_r_p05']:.4f}"
            )

    # Comparison of candidate windows: 00-24 UTC vs 05-05 UTC (local civil day)
    w0 = df_primary[df_primary["window_start_utc"] == 0]
    w5 = df_primary[df_primary["window_start_utc"] == 5]
    if not w0.empty and not w5.empty:
        merged_w = pd.merge(
            w0,
            w5,
            on=["station_name", "variable", "lag_days", "season"],
            suffixes=("_00_24", "_05_05"),
        )
        merged_w["pearson_diff"] = (
            merged_w["pearson_r_00_24"] - merged_w["pearson_r_05_05"]
        ).abs()
        max_diff_overall = float(merged_w["pearson_diff"].max())
        max_diff_lag0_todo = float(
            merged_w[(merged_w["lag_days"] == 0) & (merged_w["season"] == "Todo")][
                "pearson_diff"
            ].max()
        )
        max_diff_lag0_all = float(
            merged_w[merged_w["lag_days"] == 0]["pearson_diff"].max()
        )

        win_comp_csv = reports_dir / "temperature_window_comparison.csv"
        merged_w.to_csv(win_comp_csv, index=False)

        print("\n" + "=" * 88)
        print("COMPARACION ENTRE VENTANAS: 00-24 UTC vs 05-05 UTC (DIA CIVIL LOCAL)")
        print("=" * 88)
        print("Diferencia maxima de Pearson (|r_00_24 - r_05_05|):")
        print(f"  - Lag 0 (Todo el año): {max_diff_lag0_todo:.4f}")
        print(f"  - Lag 0 (todas las temporadas): {max_diff_lag0_all:.4f}")
        print(
            f"  - Maxima global (todos los desfases y temporadas): {max_diff_overall:.4f}"
        )
        print("\nDetalle para Lag 0 (Todo el año):")
        for _, row in merged_w[
            (merged_w["lag_days"] == 0) & (merged_w["season"] == "Todo")
        ].iterrows():
            print(
                f"  {row['station_name']} [{row['variable']}]: "
                f"r(00-24)={row['pearson_r_00_24']:.4f}, "
                f"r(05-05)={row['pearson_r_05_05']:.4f} | "
                f"|Delta r|={row['pearson_diff']:.4f}"
            )
        print(f"\nReporte comparativo de ventanas guardado en: {win_comp_csv}")

    print("\n" + "=" * 88)
    print("ANALISIS DE VECINDARIO 3x3 Y SELECCION DE CELDAS (AVISO MENOR |delta_z|)")
    print("=" * 88)
    for cfg in stations_config:
        st_name = cfg["name"]
        sub_g = [r for r in all_grid_3x3 if r["station_name"] == st_name]
        min_r = [r for r in sub_g if r.get("is_min_abs_delta_z")][0]
        center_r = [r for r in sub_g if r["is_center"]][0]
        print(
            f"\nEstacion: {st_name} (altitud auditada = {center_r['station_elevation_masl']} msnm)"
        )
        print(
            f"  Celda central / mas cercana: ({center_r['grid_lat']}°N, {center_r['grid_lon']}°W), "
            f"z={center_r['cell_elevation_masl']} m, delta_z={center_r['delta_z_m']:+.1f} m | "
            f"Sesgo Tmax crudo/adj: {center_r['tmax_bias_raw_degC']:+.2f}/{center_r['tmax_bias_adj_degC']:+.2f} °C, "
            f"Sesgo Tmin crudo/adj: {center_r['tmin_bias_raw_degC']:+.2f}/{center_r['tmin_bias_adj_degC']:+.2f} °C, "
            f"Sesgo DTR crudo/adj: {center_r['dtr_bias_raw_degC']:+.2f}/{center_r['dtr_bias_adj_degC']:+.2f} °C"
        )
        print(
            f"  Celda con menor |delta_z|: ({min_r['grid_lat']}°N, {min_r['grid_lon']}°W, di={min_r['di']}, dj={min_r['dj']}), "
            f"z={min_r['cell_elevation_masl']} m, delta_z={min_r['delta_z_m']:+.1f} m | "
            f"Sesgo Tmax crudo/adj: {min_r['tmax_bias_raw_degC']:+.2f}/{min_r['tmax_bias_adj_degC']:+.2f} °C, "
            f"Sesgo Tmin crudo/adj: {min_r['tmin_bias_raw_degC']:+.2f}/{min_r['tmin_bias_adj_degC']:+.2f} °C, "
            f"Sesgo DTR crudo/adj: {min_r['dtr_bias_raw_degC']:+.2f}/{min_r['dtr_bias_adj_degC']:+.2f} °C"
        )
        if not min_r["is_center"]:
            print(
                "  --> AVISO: La celda de menor |delta_z| NO coincide con la celda mas cercana (centro). "
                "NO se selecciona automaticamente."
            )
        else:
            print("  --> OK: La celda central coincide con la de menor |delta_z|.")

    print("\n" + "=" * 88)
    print(
        "SESGOS POR TEMPORADA EN LA CELDA CENTRAL (CRUDO Y AJUSTADO: Tmax, Tmin, DTR)"
    )
    print("=" * 88)
    for cfg in stations_config:
        st_name = cfg["name"]
        print(f"\nEstacion: {st_name}")
        sub_s = [r for r in all_seasonal if r["station_name"] == st_name]
        for r in sub_s:
            print(
                f"  [{r['season']:4s}] N={r['n_days']:4d} | "
                f"Tmax sesgo: {r['tmax_bias_raw_degC']:+.2f} -> {r['tmax_bias_adj_degC']:+.2f} °C | "
                f"Tmin sesgo: {r['tmin_bias_raw_degC']:+.2f} -> {r['tmin_bias_adj_degC']:+.2f} °C | "
                f"DTR sesgo: {r['dtr_bias_raw_degC']:+.2f} -> {r['dtr_bias_adj_degC']:+.2f} °C"
            )

    lapse_rows = []
    print("\n" + "=" * 88)
    print("GRADIENTE TERMICO INTRINSECO DE ERA5 (REGRESION EN EL DOMINIO 5x5)")
    print("=" * 88)
    for st_name, info in era5_lapse_reports.items():
        print(f"\nEstacion / Dominio: {st_name}")
        print(
            f"  Rango de elevacion: {info['z_min_masl']} a {info['z_max_masl']} msnm (span: {info['z_range_m']} m)"
        )
        for var_k in ["Tmax", "Tmin", "Tmean"]:
            v_info = info[var_k]
            print(
                f"  {var_k}: Gradiente empirico ERA5 = {v_info['empirical_lapse_rate_degC_per_km']} °C/km "
                f"(r = {v_info['pearson_r']}, R2 = {v_info['r_squared']}, p = {v_info['p_value']:.2e}, "
                f"rango: {v_info['mean_min_degC']} a {v_info['mean_max_degC']} °C)"
            )
            lapse_rows.append(
                {
                    "domain": st_name,
                    "variable": var_k,
                    "z_min_masl": info["z_min_masl"],
                    "z_max_masl": info["z_max_masl"],
                    "z_range_m": info["z_range_m"],
                    "empirical_lapse_rate_degC_per_km": v_info[
                        "empirical_lapse_rate_degC_per_km"
                    ],
                    "pearson_r": v_info["pearson_r"],
                    "r_squared": v_info["r_squared"],
                    "p_value": v_info["p_value"],
                    "standard_error": v_info["standard_error"],
                }
            )

    df_lapse = pd.DataFrame(lapse_rows)
    lapse_csv = reports_dir / "temperature_internal_lapse_rates.csv"
    df_lapse.to_csv(lapse_csv, index=False)
    print(f"\nReporte de gradientes internos guardado en: {lapse_csv}")


if __name__ == "__main__":
    main()
