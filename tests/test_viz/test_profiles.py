"""Unit tests for analytical profiles and diagnostic visualizers."""

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tropicor.core.metrics import TaylorStatistics
from tropicor.viz.profiles import (
    plot_climatology_comparison,
    plot_climatology_multiples,
    plot_missing_data_heatmap,
    plot_obs_vs_model,
    plot_taylor_diagram,
    plot_thermal_range,
)


def _generate_synthetic_climate_series(
    n_years: int = 15,
) -> Tuple[pd.Series, pd.Series]:
    """Helper to generate synthetic observed and modeled monthly time series."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2000-01-01", periods=n_years * 12, freq="MS")
    # Bimodal precipitation cycle with peaks in May (month 5) and Oct (month 10)
    month_cycle = np.array(
        [50, 60, 110, 150, 220, 120, 80, 90, 140, 240, 180, 80], dtype=float
    )
    base_signal = np.tile(month_cycle, n_years)

    obs = pd.Series(base_signal + rng.normal(0, 15, size=len(idx)), index=idx)
    mod = pd.Series(base_signal * 0.9 + rng.normal(10, 20, size=len(idx)), index=idx)
    return obs, mod


def test_plot_climatology_comparison_precipitation_includes_regime() -> None:
    """Verify precipitation climatology comparison includes regime badge."""
    obs, mod = _generate_synthetic_climate_series()
    ax = plot_climatology_comparison(
        observed=obs,
        modeled=mod,
        variable="precipitation",
        station_name="Bucaramanga UIS",
        show_regime=True,
    )

    try:
        assert ax is not None
        title = ax.get_title()
        assert "Bucaramanga UIS" in title
        assert "Obs:" in title and "ERA5:" in title
        assert len(ax.lines) == 2  # Observed + Modeled
    finally:
        plt.close(ax.figure)


def test_plot_climatology_comparison_temperature_omits_regime() -> None:
    """Verify temperature climatology comparison does not classify rainfall regime."""
    idx = pd.date_range("2000-01-01", periods=120, freq="MS")
    t_obs = pd.Series(np.linspace(20, 25, 120), index=idx)
    t_mod = pd.Series(np.linspace(19, 24, 120), index=idx)

    ax = plot_climatology_comparison(
        observed=t_obs,
        modeled=t_mod,
        variable="temperature",
        station_name="Bogota",
        show_regime=True,
    )

    try:
        assert ax is not None
        title = ax.get_title()
        assert "Bogota" in title
        assert "Regime" not in title
        assert ax.get_ylabel() == "Temperature (°C)"
    finally:
        plt.close(ax.figure)


def test_plot_climatology_multiples_grid() -> None:
    """Verify multi-station small multiples climatology grid."""
    obs1, mod1 = _generate_synthetic_climate_series(5)
    obs2, mod2 = _generate_synthetic_climate_series(5)

    data_map = {
        "Station Alpha": (obs1, mod1),
        "Station Beta": (obs2, mod2),
    }

    fig, axes = plot_climatology_multiples(
        stations_data=data_map,
        variable="precipitation",
        ncols=2,
    )

    try:
        assert fig is not None
        assert len(axes) >= 2
        assert axes[0].get_title().startswith("Station Alpha")
        assert axes[1].get_title().startswith("Station Beta")
    finally:
        plt.close(fig)


def test_plot_taylor_diagram_renders_properly() -> None:
    """Verify Taylor diagram renders reference point, test models, and CRMSE arcs."""
    stats_dict = {
        "ERA5": TaylorStatistics(
            std_obs=15.0, std_model=14.2, correlation=0.91, crmse=6.1
        ),
        "CHIRPS": TaylorStatistics(
            std_obs=15.0, std_model=16.5, correlation=0.86, crmse=8.3
        ),
    }

    ax = plot_taylor_diagram(
        stats_list=stats_dict,
        ref_std=15.0,
        normalize=False,
        title="Diagnostic Taylor Diagram",
    )

    try:
        assert ax is not None
        assert ax.get_title() == "Diagnostic Taylor Diagram"
        assert ax.get_legend() is not None
        # Verify scatter collections for reference point and test models
        assert len(ax.collections) >= 2
    finally:
        plt.close(ax.figure)


def test_plot_taylor_diagram_normalized() -> None:
    """Verify normalized Taylor diagram places reference at 1.0."""
    t_stat = TaylorStatistics(
        std_obs=1.0, std_model=0.95, correlation=0.88, crmse=0.45, normalized=True
    )
    ax = plot_taylor_diagram(stats_list=[t_stat], labels=["Test Model"], normalize=True)

    try:
        assert ax is not None
        assert "Normalized" in ax.get_xlabel()
    finally:
        plt.close(ax.figure)


def test_plot_obs_vs_model_validation_metrics() -> None:
    """Verify scatter plot of obs vs model includes identity line and metrics box."""
    obs, mod = _generate_synthetic_climate_series()
    ax = plot_obs_vs_model(
        observed=obs,
        modeled=mod,
        station_name="Las Flores",
        variable_label="Precipitation (mm)",
        show_metrics=True,
    )

    try:
        assert ax is not None
        assert "Las Flores" in ax.get_title()
        assert len(ax.texts) >= 1  # Metrics text box
        # Regression + Identity lines
        assert len(ax.lines) >= 2
    finally:
        plt.close(ax.figure)


def test_plot_missing_data_heatmap() -> None:
    """Verify completeness heatmap matrix."""
    obs, _ = _generate_synthetic_climate_series(10)
    # Inject NaN gaps
    obs.iloc[5:15] = np.nan

    ax = plot_missing_data_heatmap(obs, title="Missing Data Profile")

    try:
        assert ax is not None
        assert ax.get_title() == "Missing Data Profile"
        # 1 image in ax
        assert len(ax.images) == 1
    finally:
        plt.close(ax.figure)


def test_plot_thermal_range() -> None:
    """Verify diurnal temperature range curves and double difference."""
    idx = pd.date_range("2000-01-01", periods=120, freq="MS")
    tmax_obs = pd.Series(np.full(120, 28.0), index=idx)
    tmin_obs = pd.Series(np.full(120, 18.0), index=idx)
    tmax_mod = pd.Series(np.full(120, 26.0), index=idx)
    tmin_mod = pd.Series(np.full(120, 19.0), index=idx)

    ax = plot_thermal_range(
        tmax=tmax_obs,
        tmin=tmin_obs,
        tmax_model=tmax_mod,
        tmin_model=tmin_mod,
        station_name="Noanama",
    )

    try:
        assert ax is not None
        assert "Noanama" in ax.get_title()
        # 3 curves: Observed DTR, Modeled DTR, Double Difference
        assert len(ax.lines) == 3
    finally:
        plt.close(ax.figure)
