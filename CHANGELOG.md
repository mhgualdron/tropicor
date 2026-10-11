# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.4] - 2026-10-11

### Added
- **Standardized Residual Diagnostics & Trend Drift Engine** (`tropicor.core.diagnostics`):
  - Added `DistributionMoments`: computes sample mean, variance, sample standard deviation, unbiased Fisher-Pearson skewness ($g_1$), and excess kurtosis ($g_2$) with optional `bias=True` flag for historical thesis replication. Evaluates Shapiro-Wilk $W$ normality testing with physical boundary and extreme-value guards.
  - Added `TrendDiagnostics`: non-parametric Sen's slope evaluated against decimal calendar years (units per year and per decade, invariant to missing months), Mann-Kendall score $S$, standardized $Z$, and two-sided p-value.
  - Added Hamed & Rao (1998) variance correction with `max_lags=3` empirical calibration: controls Type I false positive inflation under autoregressive serial dependence while maintaining ~89% detection power.
  - Added `lag1_autocorr` field to `TrendDiagnostics` reporting the sample lag-1 autocorrelation of detrended residuals on the calendar grid.
  - Added automatic `UserWarning` when lag-1 autocorrelation exceeds 0.70, notifying users that corrected p-values may still be liberal (~20% false positives at $\phi=0.85$).
  - Added `ResidualDiagnosticsReport`: unified immutable dataclass providing full series (`raw_residuals`, `monthly_climatology_residuals`, `deseasonalized_residuals`, `standardized_residuals`) and `.summary()` tabular DataFrame exporter.
  - Added `compute_deseasonalized_residuals()`: isolates residual drift from annual climatology with sample size guards (`min_years=10`, `min_obs_per_month=5`) and optional `monthly_standardize=True` parameter to mitigate precipitation heteroscedasticity.
  - Added `standardize_residuals()` for scaling series to zero mean and unit variance.
  - Added top-level package exports in `tropicor` and `tropicor.core`.
- **Validation Script & Real-Data Benchmarking** (`examples/validate_residual_diagnostics.py`):
  - Phase 1: Replicates historical UNAL thesis calculations (Entrega 10) on raw standardized differences ($e_{\text{thesis}} = \text{IDEAM} - \text{ERA5}$), documenting the exact sign flip under modern convention ($e_{\text{tropicor}} = \text{ERA5} - \text{IDEAM}$).
  - Phase 2: Comprehensive diagnostic audit across 4 Colombian benchmark stations (Puerto Carreño, Las Flores, Noanamá, Gorgona) for precipitation, TMin, and TMax, printing lag-1 autocorrelation for every series and exporting `local/residual_diagnostics_report.csv`.
- **Documentation**:
  - Added `docs/guides/residual-diagnostics.md`: mathematical formulations, empirical false positive calibration sweep table, statistical power preservation table, documentation of `max_lags=3` as an empirical choice, and Colombian benchmark results.
  - Added API reference in `docs/api/core.md` and updated navigation in `mkdocs.yml`.

## [0.1.3] - 2026-10-09

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
