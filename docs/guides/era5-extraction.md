# ERA5 NetCDF Extraction Guide

The `tropicor.io.era5.ERA5Adapter` class provides a performant, memory-efficient spatial slicing engine for Copernicus ECMWF ERA5 reanalysis NetCDF datasets.

It automatically handles common NetCDF pitfalls encountered in tropical climate workflows, including Copernicus Climate Data Store (CDS) dimension quirks, coordinate standardizations, and elevation extraction.

---

## 1. Quickstart

```python
from tropicor.io.era5 import ERA5Adapter

# Open an ERA5 monthly dataset using a context manager
with ERA5Adapter("data/era5/era5_colombia_monthly.nc") as era5:
    # Extract 2-meter temperature at Bucaramanga coordinates
    temp_series = era5.get_series(
        latitude=7.139,
        longitude=-73.120,
        variable="t2m",
        method="bilinear",
    )
    print(temp_series.head())
```

---

## 2. Copernicus CDS Harmonization

ERA5 files downloaded from the Copernicus CDS API frequently suffer from structural inconsistencies:
- **`valid_time` vs `time`**: Modern CDS downloads name the temporal dimension `valid_time`. `ERA5Adapter` standardizes this to `time`.
- **`expver` dimension**: Multi-version CDS downloads bundle both consolidated data (`expver=1`) and preliminary near-real-time updates (`expver=5`). `ERA5Adapter` resolves this by prioritizing `expver=1` and dropping the redundant coordinate.
- **Descending Latitudes**: ECMWF grid definitions store latitudes in descending order ($+90^\circ \to -90^\circ$). `ERA5Adapter` calls `.sortby(["latitude", "longitude"])` so spatial interpolation and monotonic slicing execute reliably.
- **Longitude Normalization**: If longitudes are encoded in $[0, 360^\circ]$, they are mapped to $[-180^\circ, 180^\circ]$.

---

## 3. Metadata-Driven Unit Conversions

To ensure scientific integrity and eliminate silent bugs in extreme tropical microclimates (such as freezing Andean páramos or Chocó monthly precipitation exceeding $1000\,\text{mm}$), `ERA5Adapter` **never uses numeric thresholds** to guess units.

Instead, unit conversions read the NetCDF variable's `attrs['units']` attribute:

| Variable | NetCDF Attribute `units` | Output Unit | Conversion Applied |
| :--- | :--- | :--- | :--- |
| **Temperature** (`t2m`) | `K`, `Kelvin` | $^\circ\text{C}$ | $T_{^\circ\text{C}} = T_{\text{K}} - 273.15$ |
| **Temperature** (`t2m`) | `°C`, `C`, `Celsius` | $^\circ\text{C}$ | Identity (no change) |
| **Precipitation** (`tp`) | `m`, `meter`, `metres` | $\text{mm}$ | $P_{\text{mm}} = P_{\text{m}} \times 1000$ |
| **Precipitation** (`tp`) | `mm`, `millimeter` | $\text{mm}$ | Identity (no change) |

> [!WARNING]
> If a variable's `units` attribute is missing or contains an unrecognized string, a `UserWarning` is raised and raw data values are passed through unconverted.

---

## 4. Collocating with Station Catalogs

You can query ERA5 directly using `StationMetadata` instances from `StationCatalog`:

```python
from tropicor.io.era5 import ERA5Adapter
from tropicor.io.stations import StationCatalog

catalog = StationCatalog()
station = catalog.get("21205710")  # JARDIN BOTANICO (Bogotá)

with ERA5Adapter("data/era5/era5_monthly.nc") as era5:
    era5_precip = era5.get_station_series(
        station=station,
        variable="tp",
        method="bilinear",
    )
```

---

## 5. Model Ground Elevation from Surface Geopotential

In ERA5, surface topography is archived as surface geopotential $z$ with units $\text{m}^2/\text{s}^2$. The model grid surface elevation $h_{\text{model}}$ in meters above sea level (masl) is derived using the standard acceleration due to gravity $g_0 = 9.80665\,\text{m}/\text{s}^2$:

$$h_{\text{model}} = \frac{z}{g_0}$$

```python
with ERA5Adapter("data/era5/era5_surface_orography.nc") as era5:
    model_elev = era5.get_model_elevation(latitude=7.139, longitude=-73.120)
    print(f"ERA5 Model Elevation: {model_elev} masl")
```

Comparing $h_{\text{model}}$ against the station's actual altitude $h_{\text{obs}}$ allows quantifying the **orographic height deficit** ($\Delta h = h_{\text{obs}} - h_{\text{model}}$) responsible for systemic temperature biases across the Andes.

---

## 6. Daily Precipitation Aggregation & WMO/DHIME Convention

By standard WMO and IDEAM convention (*Día Pluviométrico convencional*), the precipitation attributed to calendar day $D$ is the 24-hour total accumulated between 07:00 local time on day $D$ and 07:00 local time on day $D+1$. Across continental Colombia ($\text{UTC}-5$, without daylight saving time), this window corresponds strictly to:

$$\text{Día Pluviométrico } D = \sum_{t = \text{12:00 UTC}(D)}^{\text{12:00 UTC}(D+1)} P(t)$$

`ERA5Adapter.get_daily_precipitation()` aggregates hourly reanalysis records according to this exact window (`window_start_utc=12, min_hours=24`). If the final day of a multi-year download lacks the first 12 hours of the subsequent day ($D+1$), it is strictly coerced to `NaN` (never imputed as zero).

### Empirical Validation of Temporal Lag (Lag 0 Confirmation)

The assignment of the 12:00–12:00 UTC window to day $D$ (Lag 0) vs $D-1$ (Lag -1) or $D+1$ (Lag +1) was tested empirically against independent DHIME rain gauge records across two contrasting climatic regimes: an Andean highland plateau station (**Jardín Botánico**, 2553 masl, 1975–2015, $N = 12,362$) and an inter-Andean valley station (**Aguas Claras**, Aguachica, Cesar, 134 masl, 1975–2000, $N = 8,979$). Coordinates and elevations are drawn strictly from the audited station catalog (`StationCatalog`).

Reproducible results generated by `scripts/lag_analysis.py` and saved to `local/reports/lag_analysis.csv`:

| Station | Lag (Days) | $N$ (Days) | Pearson $r$ [95% CI] | Spearman $\rho$ [95% CI] | Bias (mm/day) | RMSE (mm/day) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Jardín Botánico**<br/>(21205710, 2553 masl) | -1 d | 12,362 | +0.2530 [+0.2251, +0.2802] | +0.3327 [+0.3167, +0.3487] | -0.04 | 7.08 |
| | **+0 d** | **12,362** | **+0.3195 [+0.2907, +0.3491]** | **+0.4268 [+0.4116, +0.4426]** | **-0.04** | **6.78** |
| | +1 d | 12,361 | +0.1969 [+0.1731, +0.2185] | +0.3038 [+0.2873, +0.3199] | -0.05 | 7.33 |
| **Aguas Claras**<br/>(23215030, 134 masl) | -1 d | 8,978 | +0.2135 [+0.1886, +0.2378] | +0.3734 [+0.3560, +0.3903] | +4.37 | 13.15 |
| | **+0 d** | **8,979** | **+0.2528 [+0.2270, +0.2791]** | **+0.4377 [+0.4218, +0.4540]** | **+4.37** | **12.86** |
| | +1 d | 8,979 | +0.1524 [+0.1303, +0.1748] | +0.3467 [+0.3279, +0.3646] | +4.37 | 13.59 |

In both stations and across all individual seasons (DEF, MAM, JJA, SON), **Lag 0 achieves the highest Pearson $r$ and Spearman $\rho$**, with non-overlapping 95% bootstrap confidence intervals relative to Lag +1, empirically confirming the operational WMO/IDEAM convention.

