"""Generate publication-grade gallery assets for TROPICOR documentation.

Produces synthetic demonstration figures with generic labels ('Synthetic Station A',
'Synthetic Station B', etc.) with zero real station names or proprietary datasets.
Ensures 100% offline reproducibility and showcases cartographic, climatological,
and diagnostic visualization features.

Usage:
    uv run python scripts/generate_gallery.py [--output-dir docs/assets/gallery]
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Ensure repository root is on Python module search path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tropicor.core.climatology import compute_monthly_climatology  # noqa: E402
from tropicor.core.metrics import taylor_statistics  # noqa: E402
from tropicor.io.stations import (  # noqa: E402
    NaturalRegion,
    StationCatalog,
)
from tropicor.viz import (  # noqa: E402
    REGION_PALETTE,
    plot_climatology_multiples,
    plot_regime_map,
    plot_station_map,
    plot_station_map_interactive,
    plot_taylor_diagram,
)

DEFAULT_GALLERY_DIR = REPO_ROOT / "docs" / "assets" / "gallery"


def generate_synthetic_monthly_series() -> Tuple[
    Dict[str, Tuple[pd.Series, pd.Series]],
    Dict[str, Tuple[pd.Series, pd.Series]],
    Dict[str, NaturalRegion],
]:
    """Generate 10-year monthly synthetic precipitation series for 4 stations.

    Returns:
        Tuple of (multi_year_dict, climatology_dict, region_map).
    """
    dates = pd.date_range("2010-01-01", "2019-12-31", freq="MS")
    rng = np.random.default_rng(42)
    n_years = 10

    # Base 12-month precipitation shapes (mm/month)
    # 1. Synthetic Station J (Orinoquía): Classic unimodal cycle (wet in Jun/Jul)
    base_ori_obs = np.array([30, 45, 95, 230, 390, 470, 460, 340, 250, 180, 110, 40])
    base_ori_mod = np.array([45, 60, 110, 210, 360, 430, 420, 310, 230, 160, 95, 50])

    # 2. Synthetic Station D (Caribe): Dry early year, peak in October
    base_car_obs = np.array([5, 8, 12, 35, 90, 85, 75, 120, 150, 175, 70, 15])
    base_car_mod = np.array([12, 15, 20, 45, 100, 95, 85, 130, 165, 185, 80, 22])

    # 3. Synthetic Station G (Pacífico): Very high precipitation (Jun, Oct peaks)
    # Model exhibits unimodal mismatch (peak only in Oct)
    base_pac_obs = np.array(
        [510, 480, 560, 680, 770, 840, 720, 750, 810, 880, 730, 590]
    )
    base_pac_mod = np.array(
        [430, 390, 460, 570, 620, 680, 640, 670, 730, 790, 680, 520]
    )

    # 4. Synthetic Station P (Insular): Tropical maritime bimodal (Jun, Oct/Dec)
    base_ins_obs = np.array(
        [280, 240, 270, 410, 590, 710, 540, 580, 640, 780, 620, 430]
    )
    base_ins_mod = np.array(
        [310, 270, 290, 430, 580, 680, 550, 600, 660, 730, 610, 450]
    )

    def tile_with_noise(base: np.ndarray, noise_std: float) -> pd.Series:
        tiled = np.tile(base, n_years)
        noise = rng.normal(0, noise_std, size=len(tiled))
        values = np.clip(tiled + noise, 0.0, None)
        return pd.Series(values, index=dates)

    series_data = {
        "Synthetic Station J": (
            tile_with_noise(base_ori_obs, 25.0),
            tile_with_noise(base_ori_mod, 25.0),
        ),
        "Synthetic Station D": (
            tile_with_noise(base_car_obs, 15.0),
            tile_with_noise(base_car_mod, 15.0),
        ),
        "Synthetic Station G": (
            tile_with_noise(base_pac_obs, 45.0),
            tile_with_noise(base_pac_mod, 45.0),
        ),
        "Synthetic Station P": (
            tile_with_noise(base_ins_obs, 35.0),
            tile_with_noise(base_ins_mod, 35.0),
        ),
    }

    clim_data = {
        name: (
            compute_monthly_climatology(obs),
            compute_monthly_climatology(mod),
        )
        for name, (obs, mod) in series_data.items()
    }

    region_map = {
        "Synthetic Station J": NaturalRegion.ORINOQUIA,
        "Synthetic Station D": NaturalRegion.CARIBE,
        "Synthetic Station G": NaturalRegion.PACIFICO,
        "Synthetic Station P": NaturalRegion.INSULAR,
    }

    return series_data, clim_data, region_map


def generate_all_gallery_figures(output_dir: Path) -> List[Path]:
    """Generate all gallery figures using real catalog and synthetic profiles.

    Args:
        output_dir: Directory where figures and interactive HTML will be written.

    Returns:
        List of generated file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: List[Path] = []

    print("[1/5] Generating Station Network Map (48 Real Stations)...")
    catalog = StationCatalog.from_benchmark()
    ax_map = plot_station_map(
        stations=catalog,
        color_by_region=True,
        show_insets=True,
        title="TROPICOR station catalog",
        figsize=(8.5, 9.5),
    )
    p1 = output_dir / "gallery_fig1_station_network_map.png"
    ax_map.figure.savefig(p1, dpi=300, bbox_inches="tight")
    generated.append(p1)

    print("[2/5] Generating Synthetic Precipitation Regimes Map (48 Stations)...")
    # 5 stations with illustrative synthetic regimes, 1 with model disagreement,
    # remainder marked as 'No data'
    regimes = {
        "38015030": "unimodal",  # Puerto Carreño (Orinoquía)
        "29045120": "bimodal",  # Las Flores (Caribe)
        "54085010": "bimodal",  # Noanamá (Pacífico)
        "57025020": "bimodal",  # Gorgona (Insular)
        "21205710": "bimodal",  # Bogotá (Andina)
    }
    disagreements = ["54085010"]

    ax_reg = plot_regime_map(
        stations=catalog,
        regimes=regimes,
        disagreements=disagreements,
        disagreement_label="Obs/model disagreement",
        show_insets=True,
        title=(
            "Precipitation Regimes by Station (Synthetic Demonstration)\n"
            "Illustrative demonstration data — not empirical ground truth"
        ),
        figsize=(8.5, 9.5),
    )
    footnote = ax_reg.figure.text(
        0.5,
        0.01,
        (
            "* Synthetic regimes and model disagreement markers "
            "for illustration purposes only."
        ),
        ha="center",
        fontsize=8.5,
        color="#555555",
        style="italic",
    )
    p2 = output_dir / "gallery_fig2_precipitation_regimes_map.png"
    ax_reg.figure.savefig(
        p2,
        dpi=300,
        bbox_inches="tight",
        bbox_extra_artists=[*ax_reg.artists, footnote],
    )
    generated.append(p2)

    print("[3/5] Generating Synthetic Regional Climatologies (Small Multiples)...")
    series_data, clim_data, region_map = generate_synthetic_monthly_series()

    fig_multi, _ = plot_climatology_multiples(
        stations_data=series_data,
        variable="precipitation",
        obs_name="synthetic",
        model_name="Model",
        obs_label="Observed (synthetic)",
        model_label="Modeled (synthetic)",
        ncols=2,
        show_regime=True,
        figsize=(9.5, 7.5),
    )
    p3 = output_dir / "gallery_fig3_regional_climatologies.png"
    fig_multi.savefig(p3, dpi=300, bbox_inches="tight")
    generated.append(p3)

    print("[4/5] Generating Synthetic Taylor Diagram...")
    stats_dict = {}
    colors_dict = {}
    for st_name, (obs_clim, mod_clim) in clim_data.items():
        stats_dict[st_name] = taylor_statistics(
            observed=obs_clim,
            modeled=mod_clim,
            normalize=True,
        )
        colors_dict[st_name] = REGION_PALETTE[region_map[st_name]]

    ax_taylor = plot_taylor_diagram(
        stats_list=stats_dict,
        normalize=True,
        colors=colors_dict,
        title=(
            "Taylor Diagram - Synthetic Climatological Annual Cycle "
            "(12 Monthly Means)\nSynthetic Model Evaluation vs Reference"
        ),
        figsize=(8.0, 7.5),
    )
    p4 = output_dir / "gallery_fig4_taylor_diagram_eval.png"
    ax_taylor.figure.savefig(p4, dpi=300, bbox_inches="tight")
    generated.append(p4)

    print("[5/5] Generating Interactive Web Map (48 Real Stations)...")
    fig_html = plot_station_map_interactive(
        stations=catalog,
        color_by_region=True,
        title="TROPICOR station catalog",
    )
    p5 = output_dir / "gallery_interactive_map.html"
    fig_html.write_html(str(p5))
    generated.append(p5)

    print(f"[+] All {len(generated)} gallery assets generated in: {output_dir}")
    return generated


def main() -> None:
    """Entry point for gallery generation script."""
    parser = argparse.ArgumentParser(
        description="Generate TROPICOR synthetic documentation gallery figures."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_GALLERY_DIR,
        help=f"Target directory for generated figures (default: {DEFAULT_GALLERY_DIR})",
    )
    args = parser.parse_args()
    generate_all_gallery_figures(args.output_dir)


if __name__ == "__main__":
    main()
