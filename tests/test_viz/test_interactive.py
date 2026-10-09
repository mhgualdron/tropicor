"""Unit tests for interactive Plotly station mapping engine."""

import pytest

from tropicor.io.stations import StationCatalog
from tropicor.viz.interactive import plot_station_map_interactive


def test_plot_station_map_interactive_basic() -> None:
    """Verify interactive map generates traces with corrected department metadata."""
    catalog = StationCatalog.from_benchmark()
    fig = plot_station_map_interactive(catalog, color_by_region=True)

    assert fig is not None
    # 6 traces corresponding to the 6 natural regions
    assert len(fig.data) == 6

    # Extract all customdata rows across traces
    all_custom = []
    for tr in fig.data:
        if tr.customdata is not None:
            all_custom.extend(tr.customdata)

    custom_by_code = {row[1]: row for row in all_custom}

    # Verify corrected departments appear in interactive hover card metadata
    # CASERI (27045020): Antioquia
    assert custom_by_code["27045020"][3] == "Antioquia"
    assert custom_by_code["27045020"][4] == "Caucasia"

    # MELLITO EL (12025030): Antioquia
    assert custom_by_code["12025030"][3] == "Antioquia"
    assert custom_by_code["12025030"][4] == "Necoclí"

    # COOPERATIVA LA (32075060): Meta
    assert custom_by_code["32075060"][3] == "Meta"
    assert custom_by_code["32075060"][4] == "Fuente De Oro"

    # LAS GAVIOTAS (34015010): Vichada
    assert custom_by_code["34015010"][3] == "Vichada"
    assert custom_by_code["34015010"][4] == "Cumaribo"


def test_plot_station_map_interactive_continuous_values() -> None:
    """Verify interactive map with continuous metric creates trace with colorbar."""
    catalog = StationCatalog.from_benchmark()
    fig = plot_station_map_interactive(
        catalog,
        values="elevation",
        cbar_label="Elevation (masl)",
    )

    assert fig is not None
    assert len(fig.data) == 1
    trace = fig.data[0]
    assert trace.marker.showscale is True
    assert trace.marker.colorbar.title.text == "Elevation (masl)"


def test_plot_station_map_interactive_layout_options() -> None:
    """Verify custom layout properties (height, width, center, zoom)."""
    catalog = StationCatalog.from_benchmark()
    custom_center = {"lat": 5.0, "lon": -73.0}
    fig = plot_station_map_interactive(
        catalog,
        title="Custom Interactive Map",
        zoom=6.0,
        center=custom_center,
        height=800,
        width=1000,
    )

    assert fig.layout.title.text == "Custom Interactive Map"
    assert fig.layout.height == 800
    assert fig.layout.width == 1000


def test_plot_station_map_interactive_invalid_column() -> None:
    """Verify ValueError when requested values column is absent."""
    catalog = StationCatalog.from_benchmark()
    df = catalog.to_dataframe()
    with pytest.raises(ValueError, match="Values column 'non_existent' not found"):
        plot_station_map_interactive(df, values="non_existent")


def test_plot_station_map_interactive_offline_scattergeo_and_bounds() -> None:
    """Verify offline Scattergeo usage and that stations fall within map extent."""
    catalog = StationCatalog.from_benchmark()
    fig = plot_station_map_interactive(catalog, color_by_region=True)

    # Must use Scattergeo (not tile-dependent Scattermap/Scattermapbox)
    for trace in fig.data:
        assert trace.type == "scattergeo"
        assert trace.marker.line.color == "black"

    # Geo layout boundaries
    geo = fig.layout.geo
    assert geo.scope == "south america"
    lat_min, lat_max = geo.lataxis.range
    lon_min, lon_max = geo.lonaxis.range

    # Verify all marker coordinates fall strictly within the map extent
    for trace in fig.data:
        for lat, lon in zip(trace.lat, trace.lon, strict=True):
            assert lat_min <= lat <= lat_max, (
                f"Latitude {lat} outside [{lat_min}, {lat_max}]"
            )
            assert lon_min <= lon <= lon_max, (
                f"Longitude {lon} outside [{lon_min}, {lon_max}]"
            )
