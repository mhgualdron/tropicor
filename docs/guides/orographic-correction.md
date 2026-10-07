# Orographic Lapse-Rate Correction

This guide explains how to correct vertical elevation discrepancies between global atmospheric reanalyses (such as ECMWF ERA5) and ground-truth meteorological stations using physics-based environmental lapse rates.

---

## The Orographic Discrepancy Problem

Global reanalysis models compute atmospheric variables over a continuous grid box ($\sim 31\,\text{km}$ horizontal resolution in ERA5). In mountainous regions such as the Colombian Andes, complex terrain is spatially smoothed:

- High-altitude valleys (e.g., Bucaramanga, Medellín, Cali) are represented as significantly higher than reality because their elevations are averaged with adjacent cordillera peaks.
- High mountain summits (e.g., paramos, volcanic peaks) are represented as lower than reality because their crests are smoothed into the surrounding topography.

Because temperature decreases with altitude in the troposphere, a vertical offset between model ground and true station altitude produces systematic thermal bias:

$$\Delta z = z_{\text{station}} - z_{\text{model}}$$

$$\Delta T = -\Gamma \cdot \Delta z = -\Gamma \cdot (z_{\text{station}} - z_{\text{model}})$$

---

## Basic Usage

The `tropicor.downscale` module provides `lapse_rate_temperature_correction` to perform this physical adjustment.

```python
import pandas as pd
from tropicor.downscale import (
    STANDARD_LAPSE_RATE,
    lapse_rate_temperature_correction,
)

# Example: Bucaramanga UIS Station (Station: 898 masl, ERA5: 2118 masl)
station_elevation = 898.0
era5_elevation = 2118.0

# Monthly ERA5 temperature series (in °C)
era5_temp = pd.Series(
    [16.2, 16.5, 16.8, 17.1, 16.9],
    index=pd.date_range("2010-01-01", periods=5, freq="MS"),
    name="t2m_raw",
)

# Apply lapse rate adjustment
corrected_temp = lapse_rate_temperature_correction(
    temperature=era5_temp,
    station_elevation=station_elevation,
    model_elevation=era5_elevation,
    lapse_rate=STANDARD_LAPSE_RATE,  # 0.0065 °C/m (6.5 °C/km)
)

print(corrected_temp)
```

Output:
```
2010-01-01    24.13
2010-02-01    24.43
2010-03-01    24.73
2010-04-01    25.03
2010-05-01    24.83
Freq: MS, Name: t2m_raw, dtype: float64
```

---

## Working with ERA5Adapter and StationCatalog

You can seamlessly integrate orographic correction with `tropicor.io`:

```python
from tropicor.io.era5 import ERA5Adapter
from tropicor.io.stations import StationCatalog
from tropicor.downscale import lapse_rate_temperature_correction

# 1. Load station metadata
catalog = StationCatalog.from_benchmark()
station = catalog.get("23195040")  # Bucaramanga UIS

# 2. Extract series and model ground elevation from ERA5 NetCDF
with ERA5Adapter("data/era5_monthly.nc") as adapter:
    raw_temp = adapter.get_station_series(station, variable="t2m")
    model_elevation = adapter.get_model_elevation(station.latitude, station.longitude)

# 3. Apply vertical adjustment
if model_elevation is not None:
    corrected_temp = lapse_rate_temperature_correction(
        temperature=raw_temp,
        station_elevation=station.elevation,
        model_elevation=model_elevation,
    )
```

---

## Supported Input Data Formats

`lapse_rate_temperature_correction` supports multiple data structures while preserving indexes, metadata, and column names:

1. **`pandas.Series`**: Retains `DatetimeIndex`, frequency, and series name.
2. **`pandas.DataFrame`**: Adjusts multi-column temperature data (e.g., columns `['t_min', 't_max', 't_mean']`) simultaneously.
3. **`numpy.ndarray`**: Adjusts 1D or 2D NumPy arrays without altering array shape.
4. **`float`**: Direct scalar temperature adjustment.

---

## Choosing a Lapse Rate

TROPICOR defines three physical reference constants:

| Constant | Value | Description |
| :--- | :--- | :--- |
| `STANDARD_LAPSE_RATE` | $0.0065\,^\circ\text{C}/\text{m}$ ($6.5\,^\circ\text{C}/\text{km}$) | Standard environmental lapse rate (ICAO / ISA troposphere). Recommended default. |
| `DRY_ADIABATIC_LAPSE_RATE` | $0.0098\,^\circ\text{C}/\text{m}$ ($9.8\,^\circ\text{C}/\text{km}$) | Theoretical dry adiabatic lapse rate ($g / c_p$). |
| `MOIST_ADIABATIC_LAPSE_RATE` | $0.0050\,^\circ\text{C}/\text{m}$ ($5.0\,^\circ\text{C}/\text{km}$) | Saturated pseudoadiabatic lapse rate in humid tropical regimes. |

You can also pass a custom scalar or a monthly time-varying `pd.Series` of calibrated regional lapse rates.
