"""Unit tests for bundled cartographic boundary loading and basemap rendering."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from tropicor.viz._boundaries import (
    MAINLAND_EXTENT,
    MALPELO_EXTENT,
    SAN_ANDRES_EXTENT,
    draw_cartographic_basemap,
    load_boundary_geojson,
    load_colombia_mainland_polygons,
    load_islands_polygons,
    load_neighbor_countries_polygons,
)


def test_load_boundary_geojson_cached() -> None:
    """Verify GeoJSON boundary loading and LRU cache memoization."""
    data1 = load_boundary_geojson("colombia_boundary_50m.geojson")
    data2 = load_boundary_geojson("colombia_boundary_50m.geojson")

    assert data1 is data2  # Same cached instance
    assert "features" in data1
    assert len(data1["features"]) >= 1


def test_load_boundary_geojson_missing_raises_error() -> None:
    """Verify FileNotFoundError on non-existent boundary resource."""
    with pytest.raises(
        FileNotFoundError, match="Boundary resource 'non_existent.geojson'"
    ):
        load_boundary_geojson("non_existent.geojson")


def test_colombia_mainland_polygons_structure() -> None:
    """Verify mainland Colombia polygons are parsed into 2D float coordinate arrays."""
    polys = load_colombia_mainland_polygons()
    assert len(polys) >= 1
    for poly in polys:
        assert isinstance(poly, np.ndarray)
        assert poly.ndim == 2
        assert poly.shape[1] == 2  # (lon, lat)

    # Check approximate mainland Colombia bounds: [-79.5, -66.5] lon, [-4.5, 12.5] lat
    all_lons = np.concatenate([p[:, 0] for p in polys])
    all_lats = np.concatenate([p[:, 1] for p in polys])
    assert -80.0 <= all_lons.min() <= -78.0
    assert -68.0 <= all_lons.max() <= -66.0
    assert -5.0 <= all_lats.min() <= -3.0
    assert 12.0 <= all_lats.max() <= 13.5


def test_neighbor_countries_polygons() -> None:
    """Verify neighboring South American countries are loaded."""
    neighbors = load_neighbor_countries_polygons()
    assert len(neighbors) >= 5
    for poly in neighbors:
        assert isinstance(poly, np.ndarray)
        assert poly.ndim == 2
        assert poly.shape[1] == 2


def test_islands_polygons_contain_san_andres_and_malpelo() -> None:
    """Verify high-resolution island boundaries contain San Andrés and Malpelo."""
    islands = load_islands_polygons()
    assert len(islands) >= 2

    # Verify Malpelo island exists (~ -81.6°W, ~ 3.98°N)
    has_malpelo = any(
        (-81.7 <= p[:, 0].mean() <= -81.5) and (3.9 <= p[:, 1].mean() <= 4.1)
        for p in islands
    )
    assert has_malpelo, "Malpelo island polygon missing from islands dataset"

    # Verify San Andrés & Providencia exists (~ -81.7° to -81.3°W, ~ 12.5° to 13.5°N)
    has_san_andres = any(
        (-82.0 <= p[:, 0].mean() <= -80.0) and (12.0 <= p[:, 1].mean() <= 14.0)
        for p in islands
    )
    assert has_san_andres, "San Andrés / Providencia polygon missing from dataset"


def test_draw_cartographic_basemap_renders_on_axes() -> None:
    """Verify drawing basemap onto a Matplotlib Axes."""
    fig, ax = plt.subplots(figsize=(6, 6))
    try:
        draw_cartographic_basemap(
            ax,
            extent=MAINLAND_EXTENT,
            draw_neighbors=True,
            draw_islands=True,
        )
        assert len(ax.collections) >= 2  # Neighbor collection + Colombia collection
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        assert xlim[0] <= MAINLAND_EXTENT[0]
        assert xlim[1] >= MAINLAND_EXTENT[1]
        assert ylim[0] <= MAINLAND_EXTENT[2]
        assert ylim[1] >= MAINLAND_EXTENT[3]
    finally:
        plt.close(fig)


def test_spatial_extent_constants() -> None:
    """Verify spatial extent constants are well-formed (min < max)."""
    for ext in (MAINLAND_EXTENT, SAN_ANDRES_EXTENT, MALPELO_EXTENT):
        min_lon, max_lon, min_lat, max_lat = ext
        assert min_lon < max_lon
        assert min_lat < max_lat
