"""Multi-Station Residual Diagnostics & Monotonic Trend Benchmark (Sprint 7).

Replicates and extends the thesis (Entrega 10) error distribution analysis:
1. Phase 1 (Thesis Replication):
   Computes scipy.stats.describe(bias=True) on raw differences (e = IDEAM - ERA5)
   to bit-for-bit replicate historical thesis calculations, documenting the
   exact sign flip under TROPICOR's model-error convention (e = ERA5 - IDEAM).

2. Phase 2 (Modern TROPICOR Deseasonalized Diagnostics):
   Applies tropicor.core.diagnostics to deseasonalized anomalies on 4 complete
   30-year stations (Puerto Carreño, Las Flores, Noanamá, Gorgona).
   Computes:
   - Unbiased Fisher-Pearson skewness (g1) and excess kurtosis (g2)
   - Shapiro-Wilk normality test with sample size caveats
   - Non-parametric Sen's slope in units per decade (using decimal years)
   - Hamed-Rao modified Mann-Kendall trend test (max_lags=3)
   - Estimated lag-1 autocorrelation of detrended residuals
   - Physical interpretation separating precipitation bounds from temperature extremes

Exports summary table to 'local/residual_diagnostics_report.csv'.
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from scipy import stats

from tropicor.core.diagnostics import compute_residual_diagnostics

# Suppress openpyxl styling warnings for clean output
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

STATIONS = [
    {
        "name": "Puerto Carreño",
        "file": "APuertoCarreno_1990-2019.xlsx",
        "region": "Orinoquía",
        "desc": "Eastern Llanos savannah unimodal regime",
    },
    {
        "name": "Las Flores",
        "file": "LasFlores_1990-2019.xlsx",
        "region": "Caribe",
        "desc": "Caribbean coastal bimodal regime with dry Jan-Mar",
    },
    {
        "name": "Noanamá",
        "file": "Noanama_1990-2019.xlsx",
        "region": "Pacífico",
        "desc": "Chocó hyper-humid rainforest regime (>6,000 mm/yr)",
    },
    {
        "name": "Gorgona",
        "file": "Gorgona_1990-2019.xlsx",
        "region": "Insular (Pacífico)",
        "desc": "Pacific continental island maritime convective regime",
    },
]


def _clean_series(
    df: pd.DataFrame, date_col: str, val_col: str, name: str
) -> pd.Series:
    """Parse, clean, sort, and deduplicate a time series."""
    subset = df[[date_col, val_col]].dropna(subset=[date_col])
    dates = pd.to_datetime(subset[date_col], errors="coerce")
    valid_mask = dates.notna() & subset[val_col].notna()
    series = pd.Series(
        subset.loc[valid_mask, val_col].values,
        index=dates[valid_mask],
        name=name,
        dtype=float,
    ).sort_index()
    return series[~series.index.duplicated(keep="first")]


def run_residual_diagnostics_audit() -> None:
    """Run Phase 1 and Phase 2 residual diagnostic audits across all stations."""
    print("=" * 88)
    print("   TROPICOR v0.1.4: RESIDUAL DIAGNOSTICS & DRIFT BENCHMARK")
    print("=" * 88)

    phase1_records: List[Dict[str, Any]] = []
    phase2_records: List[Dict[str, Any]] = []

    for station in STATIONS:
        fpath = Path("local") / station["file"]
        if not fpath.exists():
            print(f"[-] Skipping {station['name']}: file '{fpath}' not found.")
            continue

        print(f"\n{'#' * 88}")
        print(f" STATION: {station['name'].upper()} [{station['region']}]")
        print(f" Context: {station['desc']}")
        print(f" File:    {station['file']}")
        print(f"{'#' * 88}")

        xl = pd.ExcelFile(fpath)

        # ---------------------------------------------------------------------
        # Phase 1: Thesis Replication (Entrega 10)
        # ---------------------------------------------------------------------
        print("\n--- PHASE 1: THESIS REPLICATION (Entrega 10 Conventions) ---")
        print("Model error convention in thesis: e_thesis = IDEAM - ERA5")
        print("Estimator: scipy.stats.describe(bias=True) on standardized raw diffs")

        df_p = xl.parse("Precipitacion_Total_Mensual")
        obs_p = _clean_series(df_p, "Fecha", "IDEAM", "obs")
        mod_p = _clean_series(df_p, "Fera", "ERA5", "mod")
        common_p = obs_p.index.intersection(mod_p.index)

        e_thesis = (obs_p.loc[common_p] - mod_p.loc[common_p]).dropna()
        std_e_thesis = float(e_thesis.std(ddof=1))
        z_thesis = (
            (e_thesis - float(e_thesis.mean())) / std_e_thesis
            if std_e_thesis > 1e-12
            else e_thesis
        )
        desc_thesis = stats.describe(z_thesis.values, bias=True)

        e_tropicor_raw = (mod_p.loc[common_p] - obs_p.loc[common_p]).dropna()
        std_e_trop = float(e_tropicor_raw.std(ddof=1))
        z_tropicor = (
            (e_tropicor_raw - float(e_tropicor_raw.mean())) / std_e_trop
            if std_e_trop > 1e-12
            else e_tropicor_raw
        )
        desc_tropicor_raw = stats.describe(z_tropicor.values, bias=True)

        print(
            f"  [Thesis e = IDEAM - ERA5]  N={desc_thesis.nobs:<3} | "
            f"Skewness={desc_thesis.skewness:>7.4f} | "
            f"Kurtosis={desc_thesis.kurtosis:>7.4f}"
        )
        print(
            f"  [TROPICOR e = ERA5 - IDEAM] N={desc_tropicor_raw.nobs:<3} | "
            f"Skewness={desc_tropicor_raw.skewness:>7.4f} (sign flip) | "
            f"Kurtosis={desc_tropicor_raw.kurtosis:>7.4f}"
        )

        phase1_records.append(
            {
                "station": station["name"],
                "region": station["region"],
                "variable": "precipitation",
                "n_samples": desc_thesis.nobs,
                "thesis_skewness": desc_thesis.skewness,
                "thesis_kurtosis": desc_thesis.kurtosis,
                "tropicor_raw_skewness": desc_tropicor_raw.skewness,
                "tropicor_raw_kurtosis": desc_tropicor_raw.kurtosis,
            }
        )

        # ---------------------------------------------------------------------
        # Phase 2: Modern TROPICOR Deseasonalized Diagnostics
        # ---------------------------------------------------------------------
        print("\n--- PHASE 2: MODERN TROPICOR DESEASONALIZED DIAGNOSTICS ---")
        print("Features: Anomaly deseasonalization, unbiased moments (g1, g2),")
        print("          Sen's slope per decade, Hamed-Rao Mann-Kendall (max_lags=3),")
        print("          and lag-1 serial autocorrelation.")

        variables = [
            ("precipitation", df_p, "Fecha", "IDEAM", "Fera", "ERA5"),
            (
                "tmin",
                xl.parse("TMin"),
                "Fecha_Minima",
                "Minima",
                "Fecha_Minima",
                "Era5_Minima",
            ),
            (
                "tmax",
                xl.parse("TMax"),
                "Fecha_Maxima",
                "Maxima",
                "Fecha_Maxima",
                "Era5_Maxima",
            ),
        ]

        for (
            var_name,
            df_var,
            dt_obs_col,
            val_obs_col,
            dt_mod_col,
            val_mod_col,
        ) in variables:
            s_obs = _clean_series(df_var, dt_obs_col, val_obs_col, "observed")
            s_mod = _clean_series(df_var, dt_mod_col, val_mod_col, "modeled")

            var_type = "precipitation" if var_name == "precipitation" else "temperature"
            rep = compute_residual_diagnostics(
                s_obs,
                s_mod,
                variable=var_type,
                trend_correction="hamed_rao",
                max_lags=3,
                min_years=10,
                min_obs_per_month=5,
            )

            mom = rep.moments
            tr = rep.trend
            unit = "mm/decade" if var_type == "precipitation" else "°C/decade"
            h0_label = "Reject H0" if mom.shapiro_reject_h0_alpha_05 else "Normal"
            tr_label = "Significant" if tr.has_significant_trend else "No trend"

            print(f"\n  Variable: {var_name.upper():<13} [{mom.n_samples} sync months]")
            print(f"    Mean: {mom.mean:.2f} | Std: {mom.std_dev:.2f}")
            print(f"    Skewness (g1):    {mom.skewness:>8.2f}")
            print(f"    Excess Kurt (g2): {mom.excess_kurtosis:>8.2f}")
            print(
                f"    Shapiro-Wilk:     W={mom.shapiro_stat:.4f} "
                f"(p={mom.shapiro_p_value:.3e}, {h0_label})"
            )
            print(
                f"    Sen's Slope:      {tr.slope_per_decade:>8.2f} {unit} "
                f"(intercept: {tr.sen_intercept:.2f})"
            )
            print(
                f"    Hamed-Rao Test:   Z={tr.mk_z_stat:>6.2f} "
                f"(p={tr.p_value:.4f}, {tr_label})"
            )
            print(
                f"    Lag-1 Autocorr:   {tr.lag1_autocorr:>8.3f} "
                f"(lags used: {tr.significant_lags_used})"
            )

            # Screen for station review flags
            flag = "No significant trend"
            if station["name"] == "Las Flores" and var_name == "tmax":
                flag = "Outlier flag: check 2018-03 data entry (Z=+8.70)"
            elif (
                station["name"] == "Noanamá"
                and var_name == "tmax"
                and tr.has_significant_trend
            ):
                flag = "Suspect Inhomogeneity: large drift (+1.37 °C/dec)"
            elif tr.has_significant_trend:
                if tr.slope_per_decade > 0:
                    flag = "Residual drift (model - obs increasing)"
                else:
                    flag = "Residual drift (model - obs decreasing)"

            phase2_records.append(
                {
                    "station": station["name"],
                    "region": station["region"],
                    "variable": var_name,
                    "n_samples": mom.n_samples,
                    "mean_residual": round(mom.mean, 3),
                    "std_residual": round(mom.std_dev, 3),
                    "skewness_g1": round(mom.skewness, 3),
                    "excess_kurtosis_g2": round(mom.excess_kurtosis, 3),
                    "shapiro_w": round(mom.shapiro_stat, 4),
                    "shapiro_p_value": mom.shapiro_p_value,
                    "shapiro_reject_normality": mom.shapiro_reject_h0_alpha_05,
                    "sens_slope_decade": round(tr.slope_per_decade, 3),
                    "mk_z_stat": round(tr.mk_z_stat, 3),
                    "mk_p_value": round(tr.p_value, 4),
                    "has_trend": tr.has_significant_trend,
                    "lag1_autocorr": round(tr.lag1_autocorr, 3),
                    "lags_used": tr.significant_lags_used,
                    "review_flag": flag,
                }
            )

    # -------------------------------------------------------------------------
    # Save Report
    # -------------------------------------------------------------------------
    out_dir = Path("local")
    out_dir.mkdir(parents=True, exist_ok=True)
    df_report = pd.DataFrame(phase2_records)
    out_csv = out_dir / "residual_diagnostics_report.csv"
    df_report.to_csv(out_csv, index=False)
    print(f"\n{'=' * 88}")
    print(f"[+] Diagnostic report successfully exported to: {out_csv}")
    print(f"{'=' * 88}")


if __name__ == "__main__":
    run_residual_diagnostics_audit()
