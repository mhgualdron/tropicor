"""Unit tests for static station mapping engine and insets."""

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from tropicor.io.stations import StationCatalog
from tropicor.viz.maps import (
    _partition_by_bounding_box,
    plot_regime_map,
    plot_station_map,
)


def test_partition_by_bounding_box_routes_islands_accurately() -> None:
    """Verify dynamic bounding box routes San Andrés, Malpelo, and Mainland."""
    catalog = StationCatalog.from_benchmark()
    df = catalog.to_dataframe()

    df_main, df_sa, df_mal = _partition_by_bounding_box(df)

    # San Andrés & Providencia
    assert len(df_sa) == 2
    sa_codes = set(df_sa["code"])
    assert "17015010" in sa_codes  # SESQUICENTENARIO
    assert "17025020" in sa_codes  # EL EMBRUJO

    # Malpelo
    assert len(df_mal) == 1
    assert "57015010" in set(df_mal["code"])  # MALPELO - AUT

    # Mainland stations
    assert len(df_main) == len(df) - 3
    # Verify Gorgona island (Cauca Pacific coast) remains in mainland regional extent
    assert "57025020" in set(df_main["code"])


def test_plot_station_map_basic_by_region() -> None:
    """Verify plot_station_map renders successfully with regional color palette."""
    catalog = StationCatalog.from_benchmark()
    ax = plot_station_map(catalog, color_by_region=True, show_insets=True)

    try:
        assert ax is not None
        assert len(ax.collections) >= 2
        # Check legend exists
        assert ax.get_legend() is not None
        assert ax.get_title() == "Colombian Meteorological Station Network"
    finally:
        plt.close(ax.figure)


def test_plot_station_map_with_continuous_values() -> None:
    """Verify plot_station_map renders continuous values with an external colorbar."""
    catalog = StationCatalog.from_benchmark()
    df = catalog.to_dataframe()

    # Color stations by elevation
    ax = plot_station_map(
        df,
        values="elevation",
        cmap="plasma",
        cbar_label="Elevation (masl)",
        show_insets=True,
    )

    try:
        assert ax is not None
        fig = ax.figure
        # Colorbar adds an additional axis to the figure
        assert len(fig.axes) >= 2
    finally:
        plt.close(ax.figure)


def test_plot_station_map_existing_axes_without_insets() -> None:
    """Verify rendering onto an existing Axes object with insets disabled."""
    fig, custom_ax = plt.subplots(figsize=(6, 7))
    try:
        catalog = StationCatalog.from_benchmark()
        ret_ax = plot_station_map(catalog, ax=custom_ax, show_insets=False)
        assert ret_ax is custom_ax
        # Only 1 axis in figure (no inset child axes created)
        assert len(fig.axes) == 1
    finally:
        plt.close(fig)


def test_plot_regime_map_renders_dual_legends() -> None:
    """Verify plot_regime_map renders symbols and regional colors with dual legends."""
    catalog = StationCatalog.from_benchmark()
    df = catalog.to_dataframe()

    # Mock regime mapping across stations
    mock_regimes = {
        st_code: "bimodal"
        if idx % 3 == 0
        else ("unimodal" if idx % 3 == 1 else "multimodal")
        for idx, st_code in enumerate(df["code"])
    }

    ax = plot_regime_map(
        stations=df,
        regimes=mock_regimes,
        show_insets=True,
        title="Benchmark Rainfall Regimes",
    )

    try:
        assert ax is not None
        assert ax.get_title() == "Benchmark Rainfall Regimes"
        # Verify legends are attached
        assert ax.get_legend() is not None
        # Verify proxy handles were added to axes artists
        assert len(ax.artists) >= 1
    finally:
        plt.close(ax.figure)


def test_plot_station_map_invalid_dataframe_raises_value_error() -> None:
    """Verify ValueError when dataframe lacks mandatory spatial columns."""
    bad_df = pd.DataFrame({"station_id": ["123"], "temp": [25.0]})
    with pytest.raises(ValueError, match="missing mandatory columns"):
        plot_station_map(bad_df)
