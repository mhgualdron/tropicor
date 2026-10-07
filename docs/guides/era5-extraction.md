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
