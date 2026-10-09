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
- **Core Verification Metrics** (`tropicor.core.metrics`):
  - Added `TaylorStatistics` dataclass and `taylor_statistics()` function for computing reference standard deviation, test standard deviation, Pearson correlation, and Centered Root Mean Square Error (CRMSE) following Taylor (2001) geometric closure. Fully decoupled from matplotlib and graphic libraries.
- **Cartographic Boundary Assets** (`tropicor.data.boundaries`):
  - Bundled public-domain Natural Earth 1:50m boundary files (`colombia_boundary_50m.geojson`, `colombia_and_neighbors_50m.geojson`) and dedicated 1:10m island coastline GeoJSON (`colombia_islands_10m.geojson`) for high-resolution dual insets of San Andrés, Providencia, and Malpelo.
