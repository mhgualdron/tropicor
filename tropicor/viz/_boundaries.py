"""Cartographic boundary loaders and basemap rendering engine.

Loads bundled public-domain vector geometries from Natural Earth and draws
publication-quality geographic basemaps for Colombia and its insular territories.
"""

import functools
import importlib.resources
import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    import matplotlib.axes

# Spatial extents [min_lon, max_lon, min_lat, max_lat]
MAINLAND_EXTENT: Tuple[float, float, float, float] = (-83.0, -66.5, -4.5, 13.0)
SAN_ANDRES_EXTENT: Tuple[float, float, float, float] = (-82.0, -81.1, 12.3, 13.6)
MALPELO_EXTENT: Tuple[float, float, float, float] = (-81.8, -81.4, 3.8, 4.3)

# Cartographic styling tokens
OCEAN_COLOR: str = "#EBF4F9"
LAND_COLOR: str = "#F8F9FA"
NEIGHBOR_COLOR: str = "#ECEFF1"
BORDER_COLOR: str = "#607D8B"
COASTLINE_COLOR: str = "#455A64"


@functools.lru_cache(maxsize=8)
def load_boundary_geojson(filename: str) -> Dict[str, Any]:
    """Load a bundled GeoJSON file from package data with in-memory caching.

    Args:
        filename: Name of the GeoJSON file in tropicor/data/boundaries/.

    Returns:
        Parsed GeoJSON dictionary.

    Raises:
        FileNotFoundError: If the GeoJSON boundary file is missing from package data.
    """
    boundaries_pkg = importlib.resources.files("tropicor.data.boundaries")
    resource_path = boundaries_pkg.joinpath(filename)
    if not resource_path.is_file():
        msg = f"Boundary resource '{filename}' not found in 'tropicor.data.boundaries'."
        raise FileNotFoundError(msg)

    content = resource_path.read_text(encoding="utf-8")
    return json.loads(content)


def extract_polygon_rings(
    geometry: Dict[str, Any],
) -> List[np.ndarray]:
    """Extract exterior coordinate rings from a GeoJSON geometry.

    Args:
        geometry: GeoJSON geometry dictionary (Polygon or MultiPolygon).

    Returns:
        List of NumPy float arrays of shape (N, 2) containing [lon, lat] coordinates.
    """
    geom_type = geometry.get("type", "")
    coords = geometry.get("coordinates", [])
    polygons: List[np.ndarray] = []

    if geom_type == "Polygon":
        if coords:
            # Exterior ring is first element
            polygons.append(np.array(coords[0], dtype=np.float64))
    elif geom_type == "MultiPolygon":
        for poly in coords:
            if poly:
                polygons.append(np.array(poly[0], dtype=np.float64))

    return polygons


def load_colombia_mainland_polygons() -> List[np.ndarray]:
    """Retrieve polygon coordinates for mainland Colombia at 1:50m scale."""
    data = load_boundary_geojson("colombia_boundary_50m.geojson")
    polys: List[np.ndarray] = []
    for feat in data.get("features", []):
        polys.extend(extract_polygon_rings(feat.get("geometry", {})))
    return polys


def load_neighbor_countries_polygons() -> List[np.ndarray]:
    """Retrieve polygon coordinates for neighboring countries at 1:50m scale."""
    data = load_boundary_geojson("colombia_and_neighbors_50m.geojson")
    polys: List[np.ndarray] = []
    for feat in data.get("features", []):
        props = feat.get("properties", {})
        name = props.get("name") or props.get("NAME") or ""
        adm0 = props.get("adm0_a3") or ""
        # Exclude Colombia itself from neighbor collection
        if name.lower() != "colombia" and adm0.upper() != "COL":
            polys.extend(extract_polygon_rings(feat.get("geometry", {})))
    return polys


def load_islands_polygons() -> List[np.ndarray]:
    """Retrieve polygon coordinates for San Andrés, Providencia, and Malpelo."""
    data = load_boundary_geojson("colombia_islands_10m.geojson")
    polys: List[np.ndarray] = []
    for feat in data.get("features", []):
        polys.extend(extract_polygon_rings(feat.get("geometry", {})))
    return polys


def draw_cartographic_basemap(
    ax: "matplotlib.axes.Axes",
    extent: Optional[Tuple[float, float, float, float]] = None,
    draw_neighbors: bool = True,
    draw_islands: bool = False,
    ocean_color: str = OCEAN_COLOR,
    land_color: str = LAND_COLOR,
    neighbor_color: str = NEIGHBOR_COLOR,
    border_color: str = BORDER_COLOR,
    coastline_color: str = COASTLINE_COLOR,
) -> None:
    """Render cartographic boundaries and oceans onto a Matplotlib Axes.

    Args:
        ax: Matplotlib Axes object.
        extent: Optional (min_lon, max_lon, min_lat, max_lat) bounds.
        draw_neighbors: Whether to render adjacent South American countries.
        draw_islands: Whether to draw high-resolution insular coastline polygons.
        ocean_color: Background color for marine surfaces.
        land_color: Fill color for Colombian sovereign land.
        neighbor_color: Fill color for neighboring land masses.
        border_color: Stroke color for national borders.
        coastline_color: Stroke color for coastlines.
    """
    from matplotlib.collections import PolyCollection

    # 1. Fill ocean background
    ax.set_facecolor(ocean_color)

    # 2. Draw neighbor countries
    if draw_neighbors:
        neighbor_polys = load_neighbor_countries_polygons()
        if neighbor_polys:
            neighbor_coll = PolyCollection(
                neighbor_polys,
                facecolors=neighbor_color,
                edgecolors=coastline_color,
                linewidths=0.6,
                zorder=1,
            )
            ax.add_collection(neighbor_coll)

    # 3. Draw Colombia mainland
    colombia_polys = load_colombia_mainland_polygons()
    if colombia_polys:
        col_coll = PolyCollection(
            colombia_polys,
            facecolors=land_color,
            edgecolors=border_color,
            linewidths=0.9,
            zorder=2,
        )
        ax.add_collection(col_coll)

    # 4. Draw high-resolution island polygons if requested
    if draw_islands:
        island_polys = load_islands_polygons()
        if island_polys:
            island_coll = PolyCollection(
                island_polys,
                facecolors=land_color,
                edgecolors=border_color,
                linewidths=0.9,
                zorder=2,
            )
            ax.add_collection(island_coll)

    # 5. Apply spatial bounding limits
    if extent is not None:
        min_lon, max_lon, min_lat, max_lat = extent
        ax.set_xlim(min_lon, max_lon)
        ax.set_ylim(min_lat, max_lat)

    # Set 1:1 aspect ratio and subtle reference grid
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5, color="#CFD8DC", zorder=3)
    ax.tick_params(labelsize=8, direction="out", color="#90A4AE")
