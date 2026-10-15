#!/usr/bin/env python3
"""Empirical Lag Analysis between IDEAM DHIME Daily Precipitation and ERA5.

Generates the canonical, reproducible lag table for -1, 0, and +1 days across
evaluated stations, with 95% bootstrap confidence intervals (1000 resamples).
Coordinates and elevations are retrieved strictly from the audited StationCatalog.

Outputs:
    local/reports/lag_analysis.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from tropicor.io.era5 import ERA5Adapter
from tropicor.io.ideam import read_dhime_daily_precipitation


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
        r_boot[i] = stats.pearsonr(o_b, m_b)[0]
        rho_boot[i] = stats.spearmanr(o_b, m_b)[0]
    ci_r = (float(np.percentile(r_boot, 2.5)), float(np.percentile(r_boot, 97.5)))
    ci_rho = (float(np.percentile(rho_boot, 2.5)), float(np.percentile(rho_boot, 97.5)))
    return ci_r, ci_rho


def run_lag_analysis() -> pd.DataFrame:
    """Execute lag analysis across benchmark stations and export CSV."""
    stations_config = [
        {
            "id": "21205710",
            "name": "Jardín Botánico",
            "csv": Path("local/newFiles/DiaPluviometrico_JardinBotanico_21205710.csv"),
            "era5_dir": Path("local/era5_hourly/21205710"),
            "start": "1975-01-01",
            "end": "2015-12-31",
        },
        {
            "id": "23215030",
            "name": "Aguas Claras",
            "csv": Path("local/newFiles/DiaPluviometrico_AguasClaras.csv"),
            "era5_dir": Path("local/era5_hourly/23215030"),
            "start": "1975-01-01",
            "end": "2000-12-31",
        },
    ]

    records: List[Dict[str, Any]] = []

    for cfg in stations_config:
        st_id = cfg["id"]
        st_name = cfg["name"]
        print(f"Analyzing station: {st_name} [{st_id}]...")

        # 1. Read DHIME daily precipitation (uses audited StationCatalog)
        ds_dhime = read_dhime_daily_precipitation(cfg["csv"])
        lat_obs = float(ds_dhime.lat.sel(station=st_id).values)
        lon_obs = float(ds_dhime.lon.sel(station=st_id).values)
        elevation = float(ds_dhime.elevation.sel(station=st_id).values)

        common_idx = pd.date_range(cfg["start"], cfg["end"], freq="D")
        s_obs = ds_dhime["pr"].sel(station=st_id).to_series().reindex(common_idx)

        # 2. Gather ERA5 files
        era5_files = sorted(
            [str(f) for f in cfg["era5_dir"].glob("*.nc") if "2024" not in f.name]
        )
        if not era5_files:
            raise FileNotFoundError(f"No ERA5 NetCDF files found in {cfg['era5_dir']}")

        with ERA5Adapter(era5_files) as era5_adapter:
            ds_era5 = era5_adapter.dataset
            lats = ds_era5.latitude.values
            lons = ds_era5.longitude.values

            # Find nearest grid node
            ilat = int(np.argmin(np.abs(lats - lat_obs)))
            ilon = int(np.argmin(np.abs(lons - lon_obs)))
            lat_center = float(lats[ilat])
            lon_center = float(lons[ilon])

            for lag in [-1, 0, 1]:
                s_era_lag = era5_adapter.get_daily_precipitation(
                    lat_center, lon_center, lag_days=lag, min_hours=24
                ).reindex(common_idx)

                df_sync = pd.concat(
                    [s_obs.rename("obs"), s_era_lag.rename("mod")], axis=1
                ).dropna()
                n_sync = len(df_sync)
                y_o = df_sync["obs"].to_numpy(dtype=float)
                y_m = df_sync["mod"].to_numpy(dtype=float)

                r_val = float(stats.pearsonr(y_o, y_m)[0])
                rho_val = float(stats.spearmanr(y_o, y_m)[0])
                bias_val = float(np.mean(y_m - y_o))
                rmse_val = float(np.sqrt(np.mean((y_m - y_o) ** 2)))

                ci_r, ci_rho = compute_bootstrap_ci(y_o, y_m, n_boot=1000, seed=42)

                records.append(
                    {
                        "station_name": st_name,
                        "station_id": st_id,
                        "elevation_masl": elevation,
                        "period": f"{cfg['start'][:4]}–{cfg['end'][:4]}",
                        "lag_days": lag,
                        "n_days": n_sync,
                        "pearson_r": round(r_val, 4),
                        "pearson_ci_lower": round(ci_r[0], 4),
                        "pearson_ci_upper": round(ci_r[1], 4),
                        "spearman_rho": round(rho_val, 4),
                        "spearman_ci_lower": round(ci_rho[0], 4),
                        "spearman_ci_upper": round(ci_rho[1], 4),
                        "bias_mm_day": round(bias_val, 2),
                        "rmse_mm_day": round(rmse_val, 2),
                    }
                )

    df_out = pd.DataFrame(records)

    # Save to local/reports/lag_analysis.csv
    reports_dir = Path("local/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_csv = reports_dir / "lag_analysis.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"\nWrote results to: {out_csv}")

    # Print markdown table
    print("\n" + "=" * 92)
    print("CANONICAL LAG ANALYSIS TABLE (OUTPUT FOR DOCUMENTATION)")
    print("=" * 92)
    header = (
        "| Station | Lag (Days) | $N$ (Days) | Pearson $r$ [95% CI] "
        "| Spearman $\\rho$ [95% CI] | Bias (mm/day) | RMSE (mm/day) |"
    )
    print(header)
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for _, row in df_out.iterrows():
        lag_str = f"{row['lag_days']:+d} d"
        st_cell = (
            f"**{row['station_name']}**<br/>({row['station_id']}, "
            f"{int(row['elevation_masl'])} masl)"
            if row["lag_days"] == -1
            else ""
        )
        r_str = (
            f"{row['pearson_r']:+.4f} "
            f"[{row['pearson_ci_lower']:+.4f}, {row['pearson_ci_upper']:+.4f}]"
        )
        rho_str = (
            f"{row['spearman_rho']:+.4f} "
            f"[{row['spearman_ci_lower']:+.4f}, {row['spearman_ci_upper']:+.4f}]"
        )
        if row["lag_days"] == 0:
            lag_str = f"**{lag_str}**"
            n_str = f"**{int(row['n_days']):,}**"
            r_str = f"**{r_str}**"
            rho_str = f"**{rho_str}**"
            bias_str = f"**{row['bias_mm_day']:+.2f}**"
            rmse_str = f"**{row['rmse_mm_day']:.2f}**"
        else:
            n_str = f"{int(row['n_days']):,}"
            bias_str = f"{row['bias_mm_day']:+.2f}"
            rmse_str = f"{row['rmse_mm_day']:.2f}"
        print(
            f"| {st_cell} | {lag_str} | {n_str} | {r_str} | {rho_str} | "
            f"{bias_str} | {rmse_str} |"
        )

    return df_out


if __name__ == "__main__":
    run_lag_analysis()
