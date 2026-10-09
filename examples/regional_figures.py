"""Generate publication-grade figures from Colombian station benchmark datasets.

Demonstrates TROPICOR's visualization suite (tropicor.viz) on empirical 30-year
observational records from UNAL thesis research across distinct climatic regimes:
1. Puerto Carreño (Orinoquía): Unimodal savannah regime.
2. Las Flores (Caribe): Caribbean bimodal regime with dry winter.
3. Noanamá (Pacífico): Hyper-humid Chocó rainforest regime (>6,000 mm/year).
4. Gorgona (Insular): Eastern Pacific continental island convective regime.

Outputs generated in 'local/figures/':
- fig1_station_network_map.png: National benchmark network with dual insets.
- fig2_precipitation_regimes_map.png: Cartographic map of precipitation regimes.
- fig3_regional_climatologies.png: Small multiples grid of 12-month climatologies.
- fig4_taylor_diagram_eval.png: Polar Taylor diagram evaluating ERA5 reanalysis.
- stations_map_interactive.html: Offline interactive station explorer (go.Scattergeo).
"""

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tropicor.core.alignment import align_common_period
from tropicor.core.climatology import (
    classify_rainfall_regime,
    compute_monthly_climatology,
)
from tropicor.core.metrics import taylor_statistics
from tropicor.io.ideam import IdeamAdapter
from tropicor.io.stations import StationCatalog
from tropicor.viz import (
    FIGURE_DPI,
    plot_climatology_multiples,
    plot_regime_map,
    plot_station_map,
    plot_station_map_interactive,
    plot_taylor_diagram,
)

OUTPUT_DIR = Path("local") / "figures"


def load_regional_series() -> Dict[str, Tuple[pd.Series, pd.Series]]:
    """Load empirical paired observational and ERA5 series from local Excel files."""
    station_files = [
        ("Puerto Carreño (Orinoquía)", Path("local") / "APuertoCarreno_1990-2019.xlsx"),
        ("Las Flores (Caribe)", Path("local") / "LasFlores_1990-2019.xlsx"),
        ("Noanamá (Pacífico)", Path("local") / "Noanama_1990-2019.xlsx"),
        ("Gorgona (Insular)", Path("local") / "Gorgona_1990-2019.xlsx"),
    ]

    series_data = {}

    for label, fpath in station_files:
        if fpath.exists():
            df = pd.read_excel(fpath, sheet_name="Precipitacion_Total_Mensual")
            obs_df = df[["Fecha", "IDEAM"]].dropna(subset=["Fecha"])
            obs_dates = pd.to_datetime(obs_df["Fecha"], errors="coerce")
            obs_valid = obs_dates.notna() & obs_df["IDEAM"].notna()
            obs_s = pd.Series(
                obs_df.loc[obs_valid, "IDEAM"].values,
                index=obs_dates[obs_valid],
                name="observed",
            ).sort_index()

            mod_df = df[["Fera", "ERA5"]].dropna(subset=["Fera"])
            mod_dates = pd.to_datetime(mod_df["Fera"], errors="coerce")
            mod_valid = mod_dates.notna() & mod_df["ERA5"].notna()
            mod_s = pd.Series(
                mod_df.loc[mod_valid, "ERA5"].values,
                index=mod_dates[mod_valid],
                name="modeled",
            ).sort_index()

            obs_sync, mod_sync = align_common_period(obs_s, mod_s, drop_na=True)
            series_data[label] = (obs_sync, mod_sync)

    # Fallback to synthetic if local empirical files are not present
    if not series_data:
        print(
            "[!] Local Excel files not found; generating synthetic paired benchmarks."
        )
        rng = np.random.default_rng(42)
        idx = pd.date_range("1990-01-01", periods=360, freq="MS")

        profiles = {
            "Puerto Carreño (Orinoquía)": np.array(
                [20, 30, 80, 180, 320, 420, 410, 310, 210, 160, 90, 30]
            ),
            "Las Flores (Caribe)": np.array(
                [5, 5, 10, 35, 110, 100, 75, 115, 155, 170, 70, 15]
            ),
            "Noanamá (Pacífico)": np.array(
                [450, 420, 480, 560, 680, 610, 580, 620, 670, 690, 630, 510]
            ),
            "Gorgona (Insular)": np.array(
                [280, 260, 310, 400, 510, 460, 430, 470, 530, 580, 520, 360]
            ),
        }

        for name, p in profiles.items():
            base = np.tile(p, 30)
            obs = pd.Series(np.maximum(0, base + rng.normal(0, p * 0.15)), index=idx)
            mod = pd.Series(
                np.maximum(0, base * 0.95 + rng.normal(10, p * 0.2)), index=idx
            )
            series_data[name] = (obs, mod)

    return series_data


def generate_all_figures() -> None:
    """Generate all 4 publication figures and interactive HTML explorer."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    catalog = StationCatalog.from_benchmark()

    # -------------------------------------------------------------------------
    # Figure 1: National Benchmark Station Network Map with Dual Insets
    # -------------------------------------------------------------------------
    print("[1/5] Generating Figure 1: Station Network Map with Dual Insets...")
    ax1 = plot_station_map(
        catalog,
        color_by_region=True,
        show_insets=True,
        title="TROPICOR Benchmark Meteorological Stations (Colombia)",
        figsize=(8.5, 9.5),
    )
    fig1_path = OUTPUT_DIR / "fig1_station_network_map.png"
    ax1.figure.savefig(fig1_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(ax1.figure)
    print(f"      Saved: {fig1_path}")

    # -------------------------------------------------------------------------
    # Figure 2: Precipitation Regimes Map (Strictly Computed Regimes Only)
    # -------------------------------------------------------------------------
    print("[2/5] Generating Figure 2: Precipitation Regimes Map...")
    # Compute regimes strictly for stations where empirical data exists
    regimes: Dict[str, str] = {}
    disagreements: List[str] = []

    regional_files = {
        "38015030": Path("local/APuertoCarreno_1990-2019.xlsx"),
        "29045120": Path("local/LasFlores_1990-2019.xlsx"),
        "54085010": Path("local/Noanama_1990-2019.xlsx"),
        "57025020": Path("local/Gorgona_1990-2019.xlsx"),
    }

    for code, fpath in regional_files.items():
        if fpath.exists():
            df_reg = pd.read_excel(fpath, sheet_name="Precipitacion_Total_Mensual")
            obs_df = df_reg[["Fecha", "IDEAM"]].dropna(subset=["Fecha", "IDEAM"])
            obs_dates = pd.to_datetime(obs_df["Fecha"])
            s_obs = pd.Series(obs_df["IDEAM"].values, index=obs_dates)
            clim_obs = compute_monthly_climatology(s_obs)
            reg_obs = classify_rainfall_regime(clim_obs)
            regimes[code] = reg_obs

            mod_df = df_reg[["Fera", "ERA5"]].dropna(subset=["Fera", "ERA5"])
            mod_dates = pd.to_datetime(mod_df["Fera"])
            s_mod = pd.Series(mod_df["ERA5"].values, index=mod_dates)
            clim_mod = compute_monthly_climatology(s_mod)
            reg_mod = classify_rainfall_regime(clim_mod)

            if reg_obs != reg_mod:
                disagreements.append(code)

    # Bogotá benchmark station (21205710)
    bogota_file = Path("local/21205710.xlsx")
    if bogota_file.exists():
        s_bogota = IdeamAdapter(bogota_file).get_series()
        clim_bogota = compute_monthly_climatology(s_bogota)
        regimes["21205710"] = classify_rainfall_regime(clim_bogota)

    ax2 = plot_regime_map(
        stations=catalog,
        regimes=regimes,
        disagreements=disagreements,
        show_insets=True,
        title="Colombian Rainfall Regimes by Station (IDEAM Ground Truth)",
        figsize=(8.5, 9.5),
    )
    fig2_path = OUTPUT_DIR / "fig2_precipitation_regimes_map.png"
    ax2.figure.savefig(
        fig2_path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
        bbox_extra_artists=ax2.artists,
    )
    plt.close(ax2.figure)
    print(f"      Saved: {fig2_path}")

    # -------------------------------------------------------------------------
    # Figure 3: Regional Climatology Comparisons (with P10-P90 & Peaks)
    # -------------------------------------------------------------------------
    print("[3/5] Generating Figure 3: Regional Climatology Comparisons...")
    series_data = load_regional_series()
    fig3, _ = plot_climatology_multiples(
        stations_data=series_data,
        variable="precipitation",
        model_name="ERA5",
        ncols=2,
        show_regime=True,
        figsize=(9.0, 7.0),
    )
    fig3_path = OUTPUT_DIR / "fig3_regional_climatologies.png"
    fig3.savefig(fig3_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig3)
    print(f"      Saved: {fig3_path}")

    # -------------------------------------------------------------------------
    # Figure 4: Taylor Diagram Model Evaluation (12 Climatological Means)
    # -------------------------------------------------------------------------
    print("[4/5] Generating Figure 4: Taylor Diagram Model Evaluation...")
    taylor_stats = {}
    for name, (obs, mod) in series_data.items():
        short_name = name.split(" (")[0]
        # Compute 12-month climatologies for annual cycle verification
        obs_c = compute_monthly_climatology(obs, min_years=5)
        mod_c = compute_monthly_climatology(mod, min_years=5)
        taylor_stats[short_name] = taylor_statistics(obs_c, mod_c, normalize=True)

    ax4 = plot_taylor_diagram(
        stats_list=taylor_stats,
        normalize=True,
        title=(
            "Taylor Diagram - Climatological Annual Cycle (12 Monthly Means)\n"
            "ERA5 Reanalysis vs IDEAM Ground Truth"
        ),
        figsize=(7.5, 7.5),
    )
    fig4_path = OUTPUT_DIR / "fig4_taylor_diagram_eval.png"
    ax4.figure.savefig(fig4_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(ax4.figure)
    print(f"      Saved: {fig4_path}")

    # -------------------------------------------------------------------------
    # Interactive HTML Map (Offline Scattergeo)
    # -------------------------------------------------------------------------
    print("[5/5] Generating Interactive Station Map HTML (Offline Scattergeo)...")
    fig_html = plot_station_map_interactive(
        catalog,
        color_by_region=True,
        title="TROPICOR Meteorological Station Network (Colombia)",
    )
    html_path = OUTPUT_DIR / "stations_map_interactive.html"
    fig_html.write_html(str(html_path))
    print(f"      Saved: {html_path}")

    print(
        "\n[+] All regional publication figures successfully generated in"
        " local/figures/"
    )


if __name__ == "__main__":
    generate_all_figures()
