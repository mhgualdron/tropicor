# Scientific Visualization & Cartographic Mapping

This guide explains how to generate publication-grade cartographic maps, climatological profiles, diagnostic charts, and interactive web maps using `tropicor.viz`.

---

## Architectural Philosophy & Design Principles

The visualization suite in TROPICOR is engineered around five fundamental principles:

1. **Strict Modular Decoupling**: Core mathematical kernels (`tropicor.core`, `tropicor.downscale`, `tropicor.io`) have zero dependencies on graphical libraries. Matplotlib is included via `tropicor[viz]` and Plotly via `tropicor[viz-interactive]`. Top-level `tropicor` exports no plotting functions.
2. **Zero-GDAL / Pure-Python Mapping**: Unlike heavyweight GIS stacks (Cartopy, GeoPandas, GDAL, Fiona, Rasterio) which frequently break during installation or CI, TROPICOR renders geographic maps using bundled public-domain Natural Earth GeoJSON boundaries (`tropicor/data/boundaries/`) parsed directly with Python's standard library.
3. **Colorblind Accessibility**: Colors adhere strictly to the **Okabe & Ito (2008)** universal color design palette, providing maximal distinction for deuteranopia, protanopia, and tritanopia. Neighboring natural regions (e.g. Andina `#0072B2` and Pacífico `#F0E442`) are assigned maximally contrasting hues.
4. **Offline Autonomy**: Both static Matplotlib figures and interactive Plotly maps run 100% offline without requiring internet access, map tile downloads, or external API tokens.
5. **Reproducibility & Scientific Truth**: Maps and profiles graph strictly calculated values. Stations without processed data are explicitly rendered as neutral grey ("No data"), and model disagreements are highlighted with designated markers.

---

## Color Palette & Visual Tokens

TROPICOR defines standard cartographic and categorical styling tokens in `tropicor.viz.palettes`:

```python
from tropicor.io.stations import NaturalRegion
from tropicor.viz import OKABE_ITO, REGION_PALETTE, REGIME_MARKERS
from tropicor.viz.palettes import get_region_color

# Retrieve hex color for a region
andina_color = REGION_PALETTE[NaturalRegion.ANDINA]  # #0072B2 (Blue)
pacifico_color = REGION_PALETTE[NaturalRegion.PACIFICO]  # #F0E442 (Yellow)
caribe_color = REGION_PALETTE[NaturalRegion.CARIBE]  # #E69F00 (Orange)
orinoquia_color = REGION_PALETTE[NaturalRegion.ORINOQUIA]  # #CC79A7 (Reddish purple)
amazonia_color = REGION_PALETTE[NaturalRegion.AMAZONIA]  # #009E73 (Bluish green)
insular_color = REGION_PALETTE[NaturalRegion.INSULAR]  # #D55E00 (Vermilion)
```

### Regime Marker Tokens
Precipitation regimes map to distinct geometric symbols:
- `bimodal`: Circle (`"o"`)
- `unimodal`: Triangle (`"^"`)
- `multimodal`: Diamond (`"D"`)
- `indeterminate`: Square (`"s"`)
- `disagreement`: Bold cross (`"x"` overlaid on observed regime)

---

## Cartographic Maps

### 1. National Benchmark Station Network (`plot_station_map`)

Draws the national Colombian meteorological network with automatic dual insets for offshore insular territories:
- **San Andrés & Providencia** (Caribbean Sea, $\sim 12.5^\circ\text{N}, -81.7^\circ\text{W}$)
- **Isla Malpelo** (Eastern Pacific Ocean, $\sim 3.98^\circ\text{N}, -81.60^\circ\text{W}$)

Insets are positioned in open Pacific waters to ensure mainland stations (such as Urabá and the Pacific coast) remain completely visible.

```python
import matplotlib.pyplot as plt
from tropicor.io.stations import StationCatalog
from tropicor.viz import plot_station_map

catalog = StationCatalog.from_benchmark()

ax = plot_station_map(
    stations=catalog,
    color_by_region=True,
    show_insets=True,
    title="TROPICOR Benchmark Meteorological Stations (Colombia)",
    figsize=(8.5, 9.5),
)
plt.savefig("station_network_map.png", dpi=300, bbox_inches="tight")
plt.show()
```

### 2. Precipitation Regimes & Disagreement Map (`plot_regime_map`)

Visualizes annual rainfall regimes across the network. Only stations with explicit calculated regimes are drawn with classification symbols; uncomputed stations render in neutral grey (`#B0BEC5`) as "No data".

Stations where models (e.g., ERA5) disagree with ground observations maintain their observed regime symbol with a bold black $\mathbf{X}$ overlaid on top.

```python
from tropicor.core.climatology import (
    classify_rainfall_regime,
    compute_monthly_climatology,
)
from tropicor.viz import plot_regime_map

# Map of calculated regimes
regimes = {
    "38015030": "unimodal",  # Puerto Carreño (Orinoquía)
    "29045120": "bimodal",  # Las Flores (Caribe)
    "54085010": "bimodal",  # Noanamá (Pacífico observed)
    "57025020": "bimodal",  # Gorgona (Insular)
    "21205710": "bimodal",  # Bogotá (Andina)
}

# Flag stations where ERA5 exhibits a regime mismatch (Noanamá is bimodal in Obs but unimodal in ERA5)
disagreements = ["54085010"]

ax = plot_regime_map(
    stations=catalog,
    regimes=regimes,
    disagreements=disagreements,
    show_insets=True,
    title="Colombian Rainfall Regimes by Station (IDEAM Ground Truth)",
    figsize=(8.5, 9.5),
)
plt.savefig("regimes_map.png", dpi=300, bbox_inches="tight")
plt.show()
```

---

## Climatological Profiles & Diagnostic Figures

### 1. Climatological Comparison (`plot_climatology_comparison`)

Compares 12-month observational normals against climate model reanalyses.
- **P10–P90 Shaded Bands**: Displays interannual variability envelopes when multi-year `DatetimeIndex` series are provided.
- **Circular Peak Detection**: Marks observed peaks with stars ($\star$) and model peaks with triangles ($\blacktriangledown$).
- **Regime Classification**: Formats subplot titles with detected regimes for both sources.

```python
from tropicor.viz import plot_climatology_comparison

ax = plot_climatology_comparison(
    observed=series_obs,
    modeled=series_era5,
    variable="precipitation",
    station_name="Noanamá (Pacífico)",
    model_name="ERA5",
    show_regime=True,
)
plt.show()
```

### 2. Multi-Station Small Multiples (`plot_climatology_multiples`)

Generates a coordinated grid of annual climatology comparisons across multiple stations, featuring a single unified figure legend positioned outside the panels.

```python
from tropicor.viz import plot_climatology_multiples

stations_dict = {
    "Puerto Carreño": (obs_pc, mod_pc),
    "Las Flores": (obs_lf, mod_lf),
    "Noanamá": (obs_no, mod_no),
    "Gorgona": (obs_go, mod_go),
}

fig, axes = plot_climatology_multiples(
    stations_data=stations_dict,
    variable="precipitation",
    model_name="ERA5",
    ncols=2,
    show_regime=True,
    figsize=(9.2, 7.2),
)
fig.savefig("climatologies_grid.png", dpi=300, bbox_inches="tight")
plt.show()
```

### 3. Polar Taylor Diagram (`plot_taylor_diagram`)

Synthesizes model performance following Karl E. Taylor (2001) in Cartesian space:
- **Radial Distance**: Normalized or dimensional standard deviation $\sigma_{\text{model}} / \sigma_{\text{obs}}$.
- **Angular Coordinate**: Pearson correlation coefficient $r = \cos(\theta)$.
- **Green Concentric Arcs**: Centered Root Mean Square Error ($E' / \text{CRMSE}$).
- **Regional Colors**: Points colored using `REGION_PALETTE` tokens.

```python
from tropicor.core.metrics import taylor_statistics
from tropicor.io.stations import NaturalRegion
from tropicor.viz import REGION_PALETTE, plot_taylor_diagram

stats = {
    "Puerto Carreño": taylor_statistics(obs_clim_pc, mod_clim_pc, normalize=True),
    "Las Flores": taylor_statistics(obs_clim_lf, mod_clim_lf, normalize=True),
    "Noanamá": taylor_statistics(obs_clim_no, mod_clim_no, normalize=True),
    "Gorgona": taylor_statistics(obs_clim_go, mod_clim_go, normalize=True),
}

colors = {
    "Puerto Carreño": REGION_PALETTE[NaturalRegion.ORINOQUIA],
    "Las Flores": REGION_PALETTE[NaturalRegion.CARIBE],
    "Noanamá": REGION_PALETTE[NaturalRegion.PACIFICO],
    "Gorgona": REGION_PALETTE[NaturalRegion.INSULAR],
}

ax = plot_taylor_diagram(
    stats_list=stats,
    normalize=True,
    colors=colors,
    title="Taylor Diagram - Climatological Annual Cycle (12 Monthly Means)\nERA5 Reanalysis vs IDEAM Ground Truth",
    figsize=(7.5, 7.5),
)
plt.savefig("taylor_diagram.png", dpi=300, bbox_inches="tight")
plt.show()
```

### 4. Thermal Range Diagnostic (`plot_thermal_range`)

Generates a two-panel diagnostic of Diurnal Temperature Range (DTR):
- **Panel 1**: Monthly mean $T_{\max}$ and $T_{\min}$ curves for both observed and modeled data.
- **Panel 2**: Double difference metric $\Delta\text{DTR} = \text{DTR}_{\text{ERA5}} - \text{DTR}_{\text{IDEAM}}$, quantifying reanalysis thermal dampening.

```python
from tropicor.viz import plot_thermal_range

ax = plot_thermal_range(
    tmax=tmax_obs,
    tmin=tmin_obs,
    tmax_model=tmax_era5,
    tmin_model=tmin_era5,
    station_name="Bogotá - El Dorado",
    model_name="ERA5",
)
plt.show()
```

---

## Interactive Maps via Plotly (`plot_station_map_interactive`)

Requires `pip install tropicor[viz-interactive]` (which installs `plotly>=5.20`).
Uses Plotly's native `go.Scattergeo` vector basemap to guarantee 100% offline functionality without external web tile dependencies.

Features:
- Categorical region filtering with interactive legend toggling.
- Continuous metric mapping (elevation, RMSE, KGE).
- Rich station hovercards displaying station code, department, municipality, elevation, coordinates, and metric values.
- Legend placed cleanly outside the map viewing area.

```python
from tropicor.io.stations import StationCatalog
from tropicor.viz import plot_station_map_interactive

catalog = StationCatalog.from_benchmark()

fig = plot_station_map_interactive(
    stations=catalog,
    color_by_region=True,
    title="TROPICOR Meteorological Station Network (Colombia)",
)

# Export standalone offline HTML file
fig.write_html("stations_map_interactive.html")

# In Jupyter / Colab
fig.show()
```
