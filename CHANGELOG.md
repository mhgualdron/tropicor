# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.5] - 2026-10-27

### Added
- **Statistical Bias Correction Engine** (`tropicor.core.bias`):
  - Added Empirical Quantile Mapping (EQM) and Linear Scaling correction tailored for monthly and multi-annual climate series in complex tropical terrain (`EmpiricalQuantileMapping`, `LinearScalingCorrection`, `linear_scale_series`, `quantile_map_series`).
- **Pre-Specified Evaluation Protocol Verification Engine** (`tropicor.core.protocol`):
  - Added verification engine implementing criteria pre-specified in `docs/science/evaluation-protocol.md`: paired masking, universal crossing rule, decisive and descriptive hydroclimatic metrics, annual block bootstrap uncertainty, three-tier state mapping, cross-validation partitioners, and validation access audit guard.
- **Station Selection & Partitioning Engine** (`tropicor.core.station_selection`):
  - Added `compute_optimal_partition()` for partitioning candidate station records into non-overlapping calibration and validation blocks with tie-breaking rules.
- **Andean Station Registry** (`tropicor.core.andean_registry`):
  - Added data-driven Andean station registry with `tropicor/data/andean_station_registry.json` providing validated metadata, coordinates, and partition ranges with strict block separation.
- **Phase 3 Evaluator** (`tropicor.core.phase3_evaluator`):
  - Added Phase 3 evaluation engine with `tropicor/data/phase3_expected_inputs.json` for Andean temperature validation openings under Protocol v1.7, enforcing pre-opening verification of repository state, input hashes, single opening via audit ledger, and deterministic scorecard generation.
- **Daily Temperature Downscaling Hierarchy** (`tropicor.downscale.temperature`):
  - Added daily temperature downscaling hierarchy (Levels 1 to 3: Level 1 monthly additive bias correction, Level 2 mean state additive + DTR multiplicative scaling, Level 3 empirical quantile mapping with universal crossing rule).
  - Added `load_andean_calibration()` utility for loading frozen Andean calibration configurations and tuning parameters.
- **Daily Precipitation Downscaling Hierarchy** (`tropicor.downscale.precipitation`):
  - Added daily precipitation downscaling hierarchy (Levels 1 to 3, calibration-only: Level 1 Linear Scaling, Level 2 Threshold Matching & Scaling, Level 3 wet-day EQM with nested cross-validation hyperparameter tuning).

### Changed
- `tropicor.downscale.orography`: added `apply_orographic_correction_daily()` for lapse-rate vertical adjustments on daily temperature time series.
- `tropicor.io.era5`: added `ERA5DailyTemperatureAdapter`, `read_era5_daily_temperature()`, and `aggregate_hourly_to_daily_temperature()` for daily temperature extraction and hourly aggregation.
- `tropicor.io.ideam`: added `DhimeDailyTemperatureAdapter` and `read_dhime_daily_temperature()` for parsing and normalizing daily maximum and minimum temperature observations into CF-compliant datasets.
- `tropicor.io.stations`: added documentation clarification comment regarding Aguachica station metadata.
- `tropicor.core`: extended with bias correction, evaluation protocol, Andean registry, and Phase 3 evaluator exports.
- `tropicor.downscale`: extended with temperature downscaling, precipitation downscaling, and calibration loader exports.
- `tropicor.io`: extended with daily temperature adapter exports for IDEAM and ERA5.

### Documentation
- Evaluation protocol versions v1.0 to v1.7 and technical appendices in `docs/science/evaluation-protocol.md`.
- Appendix G pre-validation archive in `docs/science/appendix_g/README.md`.
- Archived Phase 3 reports, scorecards, and summary (commit 9450057).

### Notes
- Andean Phase 3 results are anchored to evaluator commit 9b2dcf24 and archived at commit 9450057. Relative to commit 9b2dcf24, this release changes tropicor/ only in the version string.
- Archived Phase 3 files are now checked out byte-stable on every platform (`.gitattributes`): the five calibration reports and the validation opening ledger with CRLF line endings; the lag audit, the ERA5 statics summary, the Phase 3 scorecards and summary with LF. These are the conventions their frozen SHA-256 values were computed with.

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
- **Reproducible Monte Carlo Calibration Script** (`scripts/calibrate_mann_kendall.py`):
  - Deterministic benchmarking engine across 5 seeds (`[42, 101, 2024, 777, 9999]`), $n \in \{120, 360\}$, and $\phi \in \{0.30, 0.65, 0.85\}$ (1,000 runs per cell). Serves as the canonical single source of truth for documentation calibration tables, reflecting the finalized detrended rank autocorrelation and significance filtering implementation.
- **Station Inhomogeneity Screening & Outlier Audit**:
  - Identified isolated data entry error in Las Flores $T_{\max}$ at March 2018 ($Z = +8.70$, recorded after 32-month station gap), explaining extreme kurtosis ($g_2 = +25.08$). Documented that removing this single unviable point restores a distribution close to normal ($g_1 = -0.32, g_2 = +0.37, p = 0.123$).
  - Flagged Noanamá $T_{\max}$ (+1.37 °C/decade, $p < 0.0001$) as a candidate in-situ station inhomogeneity to review before attributing drift to ERA5.
- **Validation Script & Real-Data Benchmarking** (`examples/validate_residual_diagnostics.py`):
  - Phase 1: Replicates historical UNAL thesis calculations (Entrega 10) to the printed precision (~6 decimal places), documenting the sign flip under model error convention ($e = \text{ERA5} - \text{IDEAM}$) and reproducing the thesis bug of evaluating `stats.describe` on `df['Diff']` for $T_{\max}$.
  - Phase 2: Comprehensive diagnostic audit across 4 Colombian benchmark stations (Puerto Carreño, Las Flores, Noanamá, Gorgona) for precipitation, TMin, and TMax, printing lag-1 autocorrelation for every series and exporting `local/residual_diagnostics_report.csv` with explicit exclusion audit for Las Flores.
- **Documentation**:
  - Added `docs/guides/residual-diagnostics.md`: mathematical formulations, empirical false positive calibration sweep table, statistical power preservation table, documentation of `max_lags=3` as an empirical choice, floating-point ERA5 verification, and Colombian benchmark results.
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
