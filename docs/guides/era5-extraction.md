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

---

## 7. Daily Temperature Aggregation & Temporal Conventions

Temporal aggregation conventions in `tropicor` are configured per variable via `get_variable_convention()`:

- **Precipitation (`precipitation`)**: 12:00–12:00 UTC (07:00–07:00 local time), Lag 0.
- **Temperature (`temperature`)**: 05:00–05:00 UTC (00:00–24:00 local civil day, $\text{UTC}-5$), Lag 0.

### Synthetic Validation of Civil Day Extremes

The local civil day window (05:00 UTC day $D$ to 05:00 UTC day $D+1$) ensures that diurnal extremes observed during calendar day $D$ in Colombia are assigned to day $D$:

- Diurnal maximum at 19:00 UTC (14:00 local time) on day $D$ $\rightarrow$ assigned to day $D$.
- Diurnal minimum at 11:00 UTC (06:00 local time) on day $D$ $\rightarrow$ assigned to day $D$.

Both conditions are verified by unit tests (`test_synthetic_temperature_civil_day_convention`).

### Empirical Validation of Temperature Lag & Scope

The temperature convention was evaluated across two benchmark stations during the period **1990–1994**:
- **Jardín Botánico** (Station ID `21205710`, audited elevation 2552.0 masl)
- **Aguas Claras** (Station ID `23215030`, audited elevation 133.8 masl)

> **Scope Note**: The empirical analysis covers 2 benchmark stations over the 1990–1994 common period. The window comparison is the paired difference $\Delta r = r_{00\text{--}24} - r_{05\text{--}05}$ with a 95% interval from a 30-day block bootstrap (1000 resamples, fixed seed 42), produced by `scripts/lag_analysis_temperature.py` in `local/reports/temperature_window_diff_bootstrap.csv`. At lag 0 over the full year, $|\Delta r| \le 0.0057$ in the four station × variable combinations (up to 0.0112 including seasonal subsets). Of the 60 combinations (station × variable × lag × season), 10 have an interval that excludes zero. At lag 0 these are: Aguas Claras $T_{\min}$ (full year, $\Delta r = -0.0057$, interval $[-0.0129, -0.0001]$, whose upper bound is at the edge of zero) in favor of 05:00–05:00 UTC, and Jardín Botánico $T_{\min}$ in MAM ($\Delta r = +0.0098$, $[+0.0027, +0.0151]$) and SON ($\Delta r = +0.0072$, $[+0.0019, +0.0141]$) in favor of 00:00–24:00 UTC. The direction is therefore not consistent across combinations. The 05:00–05:00 UTC window is configured as the default because it corresponds to the Colombian local civil day ($\text{UTC}-5$), not because of a consistent empirical advantage.

Reproducible results generated by `scripts/lag_analysis_temperature.py` and saved to `local/reports/lag_analysis_temperature.csv`:

| Station | Variable | Window | Lag | $N$ | Pearson $r$ [95% CI] | Spearman $\rho$ [95% CI] | Bias (°C) | RMSE (°C) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Jardín Botánico**<br/>(21205710, 2552.0 masl) | $T_{\text{max}}$ | 05–05 UTC | -1 d | 1422 | +0.1661 [+0.1166, +0.2140] | +0.1529 [+0.1021, +0.2036] | -1.65 | 2.48 |
| **Jardín Botánico**<br/>(21205710, 2552.0 masl) | $T_{\text{max}}$ | 05–05 UTC | **+0 d** | 1422 | **+0.2582 [+0.2089, +0.3043]** | +0.2512 [+0.2019, +0.2994] | -1.65 | 2.40 |
| **Jardín Botánico**<br/>(21205710, 2552.0 masl) | $T_{\text{max}}$ | 05–05 UTC | +1 d | 1421 | +0.1462 [+0.1018, +0.1901] | +0.1407 [+0.0941, +0.1890] | -1.65 | 2.49 |
| **Jardín Botánico**<br/>(21205710, 2552.0 masl) | $T_{\text{max}}$ | 00–24 UTC | -1 d | 1422 | +0.1670 [+0.1174, +0.2144] | +0.1530 [+0.1022, +0.2036] | -1.65 | 2.47 |
| **Jardín Botánico**<br/>(21205710, 2552.0 masl) | $T_{\text{max}}$ | 00–24 UTC | **+0 d** | 1422 | **+0.2582 [+0.2089, +0.3043]** | +0.2512 [+0.2018, +0.2994] | -1.65 | 2.40 |
| **Jardín Botánico**<br/>(21205710, 2552.0 masl) | $T_{\text{max}}$ | 00–24 UTC | +1 d | 1421 | +0.1461 [+0.1016, +0.1901] | +0.1407 [+0.0941, +0.1890] | -1.65 | 2.49 |
| **Jardín Botánico**<br/>(21205710, 2552.0 masl) | $T_{\text{min}}$ | 05–05 UTC | -1 d | 1434 | +0.1878 [+0.1224, +0.2422] | +0.1590 [+0.1002, +0.2083] | +0.61 | 1.91 |
| **Jardín Botánico**<br/>(21205710, 2552.0 masl) | $T_{\text{min}}$ | 05–05 UTC | **+0 d** | 1434 | **+0.4158 [+0.3640, +0.4594]** | +0.4090 [+0.3616, +0.4507] | +0.60 | 1.68 |
| **Jardín Botánico**<br/>(21205710, 2552.0 masl) | $T_{\text{min}}$ | 05–05 UTC | +1 d | 1433 | +0.3068 [+0.2573, +0.3557] | +0.2964 [+0.2490, +0.3441] | +0.61 | 1.79 |
| **Jardín Botánico**<br/>(21205710, 2552.0 masl) | $T_{\text{min}}$ | 00–24 UTC | -1 d | 1434 | +0.1888 [+0.1236, +0.2447] | +0.1583 [+0.1009, +0.2057] | +0.63 | 1.92 |
| **Jardín Botánico**<br/>(21205710, 2552.0 masl) | $T_{\text{min}}$ | 00–24 UTC | **+0 d** | 1434 | **+0.4180 [+0.3669, +0.4619]** | +0.4118 [+0.3644, +0.4531] | +0.63 | 1.69 |
| **Jardín Botánico**<br/>(21205710, 2552.0 masl) | $T_{\text{min}}$ | 00–24 UTC | +1 d | 1433 | +0.3002 [+0.2521, +0.3504] | +0.2904 [+0.2428, +0.3369] | +0.63 | 1.81 |
| **Aguas Claras**<br/>(23215030, 133.8 masl) | $T_{\text{max}}$ | 05–05 UTC | -1 d | 1804 | +0.4989 [+0.4630, +0.5359] | +0.4817 [+0.4428, +0.5188] | -5.54 | 5.84 |
| **Aguas Claras**<br/>(23215030, 133.8 masl) | $T_{\text{max}}$ | 05–05 UTC | **+0 d** | 1805 | **+0.6813 [+0.6547, +0.7085]** | +0.6657 [+0.6355, +0.6947] | -5.55 | 5.75 |
| **Aguas Claras**<br/>(23215030, 133.8 masl) | $T_{\text{max}}$ | 05–05 UTC | +1 d | 1805 | +0.5426 [+0.5079, +0.5773] | +0.5571 [+0.5202, +0.5898] | -5.55 | 5.83 |
| **Aguas Claras**<br/>(23215030, 133.8 masl) | $T_{\text{max}}$ | 00–24 UTC | -1 d | 1805 | +0.4998 [+0.4636, +0.5357] | +0.4820 [+0.4444, +0.5207] | -5.54 | 5.84 |
| **Aguas Claras**<br/>(23215030, 133.8 masl) | $T_{\text{max}}$ | 00–24 UTC | **+0 d** | 1806 | **+0.6815 [+0.6541, +0.7094]** | +0.6658 [+0.6347, +0.6947] | -5.54 | 5.74 |
| **Aguas Claras**<br/>(23215030, 133.8 masl) | $T_{\text{max}}$ | 00–24 UTC | +1 d | 1805 | +0.5431 [+0.5083, +0.5780] | +0.5573 [+0.5203, +0.5899] | -5.55 | 5.82 |
| **Aguas Claras**<br/>(23215030, 133.8 masl) | $T_{\text{min}}$ | 05–05 UTC | -1 d | 1806 | +0.4748 [+0.4348, +0.5100] | +0.4393 [+0.3985, +0.4736] | -2.94 | 3.13 |
| **Aguas Claras**<br/>(23215030, 133.8 masl) | $T_{\text{min}}$ | 05–05 UTC | **+0 d** | 1807 | **+0.5186 [+0.4817, +0.5507]** | +0.4768 [+0.4365, +0.5119] | -2.94 | 3.12 |
| **Aguas Claras**<br/>(23215030, 133.8 masl) | $T_{\text{min}}$ | 05–05 UTC | +1 d | 1807 | +0.4058 [+0.3616, +0.4472] | +0.3754 [+0.3312, +0.4152] | -2.94 | 3.15 |
| **Aguas Claras**<br/>(23215030, 133.8 masl) | $T_{\text{min}}$ | 00–24 UTC | -1 d | 1807 | +0.4731 [+0.4353, +0.5092] | +0.4381 [+0.3989, +0.4760] | -2.92 | 3.12 |
| **Aguas Claras**<br/>(23215030, 133.8 masl) | $T_{\text{min}}$ | 00–24 UTC | **+0 d** | 1808 | **+0.5131 [+0.4761, +0.5456]** | +0.4731 [+0.4319, +0.5057] | -2.92 | 3.11 |
| **Aguas Claras**<br/>(23215030, 133.8 masl) | $T_{\text{min}}$ | 00–24 UTC | +1 d | 1807 | +0.3952 [+0.3513, +0.4355] | +0.3637 [+0.3195, +0.4026] | -2.92 | 3.15 |

### Seasonal Biases in Center Cell ($T_{\max}$, $T_{\min}$, and DTR)

Biases for raw and orographically adjusted temperatures ($\Gamma = 6.5\text{ °C/km}$), along with diurnal temperature range ($\text{DTR} = T_{\max} - T_{\min}$), recorded in `local/reports/temperature_center_cell_seasonal_biases.csv`:

| Station | Season | $N$ | $T_{\max}$ Bias Raw / Adj (°C) | $T_{\min}$ Bias Raw / Adj (°C) | DTR Bias Raw / Adj (°C) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Jardín Botánico** | Todo | 1416 | -1.66 / -0.35 | +0.60 / +1.91 | -2.26 / -2.26 |
| | DEF | 358 | -0.99 / +0.32 | +1.40 / +2.71 | -2.39 / -2.39 |
| | MAM | 330 | -1.75 / -0.44 | +0.39 / +1.70 | -2.14 / -2.14 |
| | JJA | 357 | -2.39 / -1.08 | +0.01 / +1.32 | -2.40 / -2.40 |
| | SON | 371 | -1.51 / -0.20 | +0.60 / +1.91 | -2.11 / -2.11 |
| **Aguas Claras** | Todo | 1788 | -5.54 / -1.46 | -2.94 / +1.15 | -2.61 / -2.61 |
| | DEF | 442 | -5.91 / -1.83 | -2.92 / +1.16 | -2.99 / -2.99 |
| | MAM | 447 | -5.73 / -1.64 | -3.16 / +0.93 | -2.57 / -2.57 |
| | JJA | 449 | -5.37 / -1.28 | -2.82 / +1.26 | -2.54 / -2.54 |
| | SON | 450 | -5.16 / -1.08 | -2.83 / +1.25 | -2.33 / -2.33 |

*Note: Because the same uniform linear lapse rate is subtracted from both $T_{\max}$ and $T_{\min}$, the adjustment terms cancel identically for the diurnal temperature range: $(T_{\max} - \Gamma\Delta z) - (T_{\min} - \Gamma\Delta z) = T_{\max} - T_{\min}$. Consequently, DTR bias is identical in raw and adjusted forms.*

### 3x3 Spatial Grid Neighborhood Analysis & Cell Selection Warning

The 9 grid cells surrounding each station evaluated in `local/reports/temperature_grid_3x3.csv`:

#### Jardín Botánico (Station: 4.684°N, -74.108°W, 2552.0 masl)

| Grid $(\Delta i, \Delta j)$ | Coord (Lat, Lon) | $z_{\text{cell}}$ (m) | $\Delta z$ (m) | $\|\Delta z\|$ (m) | $T_{\max}$ Bias Raw / Adj (°C) | $T_{\min}$ Bias Raw / Adj (°C) | DTR Bias Raw / Adj (°C) | Note |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $(0, 0)$ | 4.75°N, -74.00°W | 2753.4 | -201.4 | 201.4 | -1.66 / -0.35 | +0.60 / +1.91 | -2.26 / -2.26 | Closest cell (center) |
| $(-1, 1)$ | 4.50°N, -73.75°W | 2549.1 | +2.9 | 2.9 | -3.18 / -3.20 | +1.13 / +1.11 | -4.31 / -4.31 | Minimizes $\|\Delta z\|$ |
| $(-1, -1)$ | 4.50°N, -74.25°W | 2600.3 | -48.3 | 48.3 | -1.18 / -0.87 | +2.45 / +2.77 | -3.63 / -3.63 | Neighbor |
| $(-1, 0)$ | 4.50°N, -74.00°W | 2726.2 | -174.2 | 174.2 | -2.81 / -1.68 | +0.57 / +1.71 | -3.38 / -3.38 | Neighbor |
| $(0, -1)$ | 4.75°N, -74.25°W | 2430.6 | +121.4 | 121.4 | -0.64 / -1.43 | +3.13 / +2.34 | -3.76 / -3.76 | Neighbor |
| $(0, 1)$ | 4.75°N, -73.75°W | 2808.8 | -256.8 | 256.8 | -3.79 / -2.12 | -0.30 / +1.38 | -3.49 / -3.49 | Neighbor |
| $(1, -1)$ | 5.00°N, -74.25°W | 2230.7 | +321.3 | 321.3 | -0.37 / -2.46 | +3.74 / +1.65 | -4.11 / -4.11 | Neighbor |
| $(1, 0)$ | 5.00°N, -74.00°W | 2696.8 | -144.8 | 144.8 | -1.43 / -0.49 | +0.76 / +1.70 | -2.20 / -2.20 | Neighbor |
| $(1, 1)$ | 5.00°N, -73.75°W | 2846.5 | -294.5 | 294.5 | -3.71 / -1.80 | -0.96 / +0.96 | -2.76 / -2.76 | Neighbor |

#### Aguas Claras (Station: 8.232°N, -73.607°W, 133.8 masl)

| Grid $(\Delta i, \Delta j)$ | Coord (Lat, Lon) | $z_{\text{cell}}$ (m) | $\Delta z$ (m) | $\|\Delta z\|$ (m) | $T_{\max}$ Bias Raw / Adj (°C) | $T_{\min}$ Bias Raw / Adj (°C) | DTR Bias Raw / Adj (°C) | Note |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $(0, 0)$ | 8.25°N, -73.50°W | 762.4 | -628.6 | 628.6 | -5.54 / -1.46 | -2.94 / +1.15 | -2.61 / -2.61 | Closest cell (center) |
| $(0, -1)$ | 8.25°N, -73.75°W | 155.0 | -21.2 | 21.2 | -2.67 / -2.53 | +0.23 / +0.36 | -2.89 / -2.89 | Minimizes $\|\Delta z\|$ |
| $(-1, -1)$ | 8.00°N, -73.75°W | 46.0 | +87.8 | 87.8 | -2.50 / -3.07 | +0.53 / -0.04 | -3.03 / -3.03 | Neighbor |
| $(-1, 0)$ | 8.00°N, -73.50°W | 527.6 | -393.8 | 393.8 | -5.46 / -2.90 | -2.58 / -0.02 | -2.89 / -2.89 | Neighbor |
| $(-1, 1)$ | 8.00°N, -73.25°W | 1411.5 | -1277.7 | 1277.7 | -9.08 / -0.77 | -7.46 / +0.84 | -1.61 / -1.61 | Neighbor |
| $(0, 1)$ | 8.25°N, -73.25°W | 1321.8 | -1188.0 | 1188.0 | -8.40 / -0.68 | -6.44 / +1.28 | -1.96 / -1.96 | Neighbor |
| $(1, -1)$ | 8.50°N, -73.75°W | 173.5 | -39.7 | 39.7 | -2.82 / -2.56 | -0.19 / +0.07 | -2.63 / -2.63 | Neighbor |
| $(1, 0)$ | 8.50°N, -73.50°W | 778.6 | -644.8 | 644.8 | -5.17 / -0.98 | -3.12 / +1.07 | -2.05 / -2.05 | Neighbor |
| $(1, 1)$ | 8.50°N, -73.25°W | 941.6 | -807.8 | 807.8 | -6.20 / -0.95 | -4.62 / +0.63 | -1.59 / -1.59 | Neighbor |

!!! warning "Spatial Cell Selection Notice"
    In both benchmark stations, the grid cell that minimizes elevation difference $|\Delta z|$ does **not** coincide with the geographically closest cell (center).
    
    - **Jardín Botánico**: Minimum $|\Delta z| = 2.9\text{ m}$ is at cell $(-1, 1)$, whereas the center cell has $|\Delta z| = 201.4\text{ m}$.
    - **Aguas Claras**: Minimum $|\Delta z| = 21.2\text{ m}$ is at cell $(0, -1)$, whereas the center cell has $|\Delta z| = 628.6\text{ m}$.
    
    `tropicor` **does not automatically select** the cell of minimum $|\Delta z|$. By default, spatial extraction uses the closest cell by geographic coordinate distance to preserve local horizontal alignment.

