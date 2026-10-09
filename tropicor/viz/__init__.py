"""Visualization and cartographic mapping suite for TROPICOR.

Provides publication-grade static figures, Taylor diagrams, climatological profiles,
and interactive geospatial station maps for tropical complex terrain.
"""

from tropicor.viz._boundaries import (
    MAINLAND_EXTENT,
    MALPELO_EXTENT,
    SAN_ANDRES_EXTENT,
    draw_cartographic_basemap,
    load_boundary_geojson,
)
from tropicor.viz.interactive import plot_station_map_interactive
from tropicor.viz.maps import plot_regime_map, plot_station_map
from tropicor.viz.palettes import (
    FIGURE_DPI,
    OKABE_ITO,
    REGIME_MARKERS,
    REGION_COLORS,
    get_region_color,
)
from tropicor.viz.profiles import (
    plot_climatology_comparison,
    plot_climatology_multiples,
    plot_missing_data_heatmap,
    plot_obs_vs_model,
    plot_taylor_diagram,
    plot_thermal_range,
)

__all__ = [
    "FIGURE_DPI",
    "MAINLAND_EXTENT",
    "MALPELO_EXTENT",
    "OKABE_ITO",
    "REGIME_MARKERS",
    "REGION_COLORS",
    "SAN_ANDRES_EXTENT",
    "draw_cartographic_basemap",
    "get_region_color",
    "load_boundary_geojson",
    "plot_climatology_comparison",
    "plot_climatology_multiples",
    "plot_missing_data_heatmap",
    "plot_obs_vs_model",
    "plot_regime_map",
    "plot_station_map",
    "plot_station_map_interactive",
    "plot_taylor_diagram",
    "plot_thermal_range",
]
