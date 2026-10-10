# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.3] - Unreleased

### Fixed
- **Station Catalog Corrections** (`tropicor.io.stations`):
  - Corrected department for station `32075060` (COOPERATIVA LA) from `Arauca` to `Meta` (municipality: Fuente de Oro).
  - Corrected department for station `34015010` (LAS GAVIOTAS) from `Arauca` to `Vichada` (municipality: Cumaribo).
  - Corrected department for station `27045020` (CASERI) from `Cauca` to `Antioquia`, and fixed municipality spelling to `Caucasia`. Documented ecotone transition between Andina and Caribe natural regions.
  - Corrected department for station `12025030` (MELLITO EL) from `Chocó` to `Antioquia` (municipality: Necoclí).
  - Fixed character encoding for station `37055010` (`AEROPUERTO SANTIAGO PÉREZ`).

### Added
- **Scientific Visualization Suite** (`tropicor.viz`):
  - Added `tropicor.viz.maps`: `plot_station_map()` with automatic dual insets for San Andrés/Providencia and Isla Malpelo, clean boundary-free insets positioned in the Pacific Ocean, and high-contrast marker styling (`edgecolors="black"`).
  - Added `plot_regime_map()`: maps classified rainfall regimes (bimodal, unimodal, multimodal, indeterminate) strictly for stations with verified calculations, displaying neutral grey markers for uncomputed stations ("No data") and overlaying black `x` crosses over the observed marker for stations showing model/observation disagreement.
  - Added `tropicor.viz.profiles`: `plot_climatology_comparison()` supporting multi-year P10–P90 interannual variability bands, automatic peak detection markers (stars for observations, triangles for models), and detected regime badges in titles.
  - Added `plot_climatology_multiples()`: multi-station small multiples grid with a single unified shared figure legend positioned outside the panels.
  - Added `plot_taylor_diagram()`: Cartesian quarter-plane Taylor diagram evaluating normalized or dimensional standard deviation, Pearson correlation rays, and centered RMS error (CRMSE) green concentric arcs, with custom regional coloring via `REGION_PALETTE`.
  - Added diagnostic plots: `plot_missing_data_heatmap()`, `plot_obs_vs_model()`, and `plot_thermal_range()`.
  - Added `tropicor.viz.interactive`: `plot_station_map_interactive()` rendering offline interactive Colombian station networks via `plotly.graph_objects.Scattergeo` with rich hovercards and outside legends.
  - Added `tropicor.viz.palettes`: colorblind-safe Okabe & Ito (2008) palette, `REGION_PALETTE` (`REGION_COLORS`), and `REGIME_MARKERS`.
  - Added `tropicor.viz._boundaries`: lightweight pure-Python GeoJSON boundary loader and cartographic drawer without external spatial GIS dependencies (no GDAL, Cartopy, or GeoPandas).
- **Core Verification Metrics** (`tropicor.core.metrics`):
  - Added `TaylorStatistics` dataclass and `taylor_statistics()` function for computing reference standard deviation, test standard deviation, Pearson correlation, and Centered Root Mean Square Error (CRMSE) following Taylor (2001) geometric closure. Fully decoupled from matplotlib and graphic libraries.
- **Cartographic Boundary Assets** (`tropicor.data.boundaries`):
  - Bundled public-domain Natural Earth 1:50m boundary files (`colombia_boundary_50m.geojson`, `colombia_and_neighbors_50m.geojson`) and dedicated 1:10m island coastline GeoJSON (`colombia_islands_10m.geojson`) for high-resolution dual insets of San Andrés, Providencia, and Malpelo.
- **Documentation & Examples**:
  - Added comprehensive User Guide for publication visualizations and mapping in MkDocs.
  - Added `examples/regional_figures.py` reproducing the 4 thesis validation figures on empirical Colombian benchmarks.
  - Added automated figure generation script in `scripts/generate_gallery.py`.
