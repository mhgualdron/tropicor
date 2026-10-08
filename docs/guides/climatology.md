# Annual Climatology & Seasonal Curves

This guide explains how to extract 12-month annual climatological normal profiles, compute geometric curve separation metrics, detect circular seasonal peaks, and classify tropical rainfall regimes using `tropicor.core.climatology`.

---

## Tropical Seasonality and the Annual Cycle

In tropical latitudes, seasonal dynamics differ fundamentally from the four-season thermal regimes of mid-latitude climates:

1. **Thermal Seasonality**: Annual temperature variation in equatorial regions is extremely muted (typically $1 - 3\,^\circ\text{C}$ between the warmest and coldest months). The Diurnal Temperature Range (DTR) routinely exceeds the annual thermal amplitude ("the night is the winter of the tropics").
2. **Pluviometric Seasonality**: Precipitation is the dominant driver of tropical seasons, dictated by the annual migration of the **Intertropical Convergence Zone (ITCZ / ZCIT)**:
   - **Bimodal Regimes (Colombian Andes)**: As the ITCZ crosses Colombia twice a year (moving north in March–May and returning south in September–November), intramontane Andean basins experience two distinct rainy seasons separated by two dry/drier seasons (*veranillos*).
   - **Unimodal Regimes (Orinoquía, Amazonía, Caribe)**: Lowland plains and coastal basins experience a single protracted rainy season and a pronounced dry season.

---

## Mathematical Formulation

### 1. 12-Month Climatological Profile

For a multi-year monthly observation series $X(y, m)$ across years $y$ and calendar months $m \in \{1, \dots, 12\}$:

$$C(m) = \frac{1}{N_m} \sum_{k=1}^{N_m} X_m(k)$$

Where $N_m$ is the count of valid, non-null observations for calendar month $m$.

### 2. Climatological Curve Distance ($d_{\text{clim}}$)

Introduced and evaluated in UNAL thesis Entregas 7 & 10, the geometric Euclidean distance between two 12-month annual cycles ($C_{\text{obs}}$ and $C_{\text{mod}}$) quantifies the total separation in seasonal geometry:

$$d_{\text{clim}} = \sqrt{\sum_{m=1}^{12} \left( C_{\text{mod}}(m) - C_{\text{obs}}(m) \right)^2}$$

Unlike the Pearson correlation coefficient ($r$), which measures only shape alignment and phase synchronization ($r \approx 1.0$ even under massive scale errors), $d_{\text{clim}}$ directly penalizes both systematic offsets and seasonal amplitude distortions.

### 3. Normalized Curve Distance ($d_{\text{norm}}$)

Because raw Euclidean distance inherits the physical units of the variable ($\text{mm}/\text{month}$ or $^\circ\text{C}$), comparing errors between dry basins (e.g., La Guajira, $\sim 30\,\text{mm}/\text{month}$) and hyper-humid basins (e.g., Chocó, $\sim 700\,\text{mm}/\text{month}$) requires scale normalization:

$$d_{\text{mean}} = \frac{d_{\text{clim}}}{\left| \bar{C}_{\text{obs}} \right|}, \quad d_{\text{amp}} = \frac{d_{\text{clim}}}{\max(C_{\text{obs}}) - \min(C_{\text{obs}})}, \quad d_{\text{std}} = \frac{d_{\text{clim}}}{\sigma(C_{\text{obs}})}$$

### 4. Circular Boundary Peak Detection

A calendar year is inherently periodic: December ($m=12$) is immediately adjacent to January ($m=1$). Evaluating peaks with standard linear window filters artificially clips maxima located near calendar boundaries.

`annual_cycle_peaks` applies **$3\times$ circular padding** (tiling the 12-month cycle to 36 months) before running `scipy.signal.find_peaks`. Maxima located in the central replica ($m \in [1, 12]$) preserve their full topographic prominence without boundary truncation.

---

## Basic Usage

```python
import numpy as np
import pandas as pd
from tropicor.core.climatology import (
    annual_cycle_amplitude,
    annual_cycle_peaks,
    annual_cycle_phase,
    classify_rainfall_regime,
    compute_monthly_climatology,
)

# 1. Generate synthetic 15-year bimodal precipitation series
dates = pd.date_range("2005-01-01", periods=180, freq="MS")
# Two wet peaks: April (m=4) and October (m=10)
precip_values = [30, 45, 80, 140, 110, 55, 40, 50, 75, 160, 130, 50] * 15
series = pd.Series(precip_values, index=dates, name="precipitacion")

# 2. Extract 12-month climatology profile
clim = compute_monthly_climatology(series, min_years=10, min_obs_per_month=5)
print(clim)

# 3. Compute seasonal metrics
amp = annual_cycle_amplitude(clim)
phase = annual_cycle_phase(clim, mode="peak")
peaks = annual_cycle_peaks(clim)
regime = classify_rainfall_regime(clim)

print(f"Cycle Amplitude: {amp:.1f} mm/month")
print(f"Global Peak:     Month {phase}")
print(f"Detected Peaks:  {peaks}")
print(f"Rainfall Regime: {regime}")
```

Output:
```
Cycle Amplitude: 130.0 mm/month
Global Peak:     Month 10
Detected Peaks:  [4, 10]
Rainfall Regime: bimodal
```

---

## Observational Completeness and Filter Interactions

!!! note "Interaction between Completeness Filtering and Curve Functions"
    `compute_monthly_climatology` enforces two quality thresholds:
    
    1. `min_years` (default: `10`): Ensures the dataset contains sufficient historical span to establish a climatological baseline (aligned with WMO guidelines).
    2. `min_obs_per_month` (default: `5`): Any calendar month with fewer valid records is assigned `NaN` and emits a warning.
    
    Downstream curve metrics (`climatological_curve_distance`, `annual_cycle_peaks`) strictly require all 12 calendar months to be valid numbers. If a month contains `NaN` due to observational gaps, they will raise a descriptive `ValueError`. In such cases, inspect the station record or adjust `min_obs_per_month` to allow partial months.

!!! warning "Exclusivity of `classify_rainfall_regime` for Precipitation"
    `classify_rainfall_regime` is designed **exclusively for rainfall data**. In tropical South America, annual temperature cycles are so flat ($1 - 3\,^\circ\text{C}$) that applying peak prominence classification to temperature is physically uninformative and will return noise or `indeterminate`.

---

## Evaluating Curve Separation Between Observations and ERA5

```python
import pandas as pd
from tropicor.core.climatology import climatological_curve_distance

# Station climatology (Observed)
obs_clim = pd.Series(
    [30, 45, 80, 140, 110, 55, 40, 50, 75, 160, 130, 50],
    index=range(1, 13),
)

# Reanalysis climatology with flattened second peak (Modeled)
mod_clim = pd.Series(
    [35, 50, 75, 130, 100, 60, 45, 55, 70, 110, 95, 55],
    index=range(1, 13),
)

# Raw Euclidean distance (mm/month)
d_raw = climatological_curve_distance(obs_clim, mod_clim)

# Normalized distance by observed mean
d_norm = climatological_curve_distance(obs_clim, mod_clim, normalize="mean")

print(f"Raw Curve Distance:        {d_raw:.2f} mm/month")
print(f"Mean-Normalized Distance:   {d_norm:.4f}")
```

---

## Canonical Sanity Check: Bogotá Jardín Botánico Station

As a canonical sanity check against historical ground observations, TROPICOR was evaluated on 36 years (441 monthly records, 1990–2026, 379 valid non-null observations) from IDEAM station **JARDIN BOTANICO [21205710]** (Bogotá, D.C., 2,552 masl):

```
Month         | Mean Precip (mm/month) | Regional Context (Literature Ref)
--------------|------------------------|----------------------------------
01 - January  |               55.91 mm | 1st Dry Season (Early-year veranillo)
02 - February |               72.38 mm | 1st Dry Season
03 - March    |              113.34 mm | Seasonal Transition
04 - April    |              138.00 mm | 1st Rainy Season (Peak 1 / ITCZ North)
05 - May      |              127.51 mm | 1st Rainy Season
06 - June     |               75.96 mm | Seasonal Transition
07 - July     |               51.98 mm | 2nd Dry Season (Mid-year veranillo)
08 - August   |               50.27 mm | 2nd Dry Season (Global Trough)
09 - September|               69.73 mm | Seasonal Transition
10 - October  |              124.83 mm | 2nd Rainy Season
11 - November |              144.92 mm | 2nd Rainy Season (Global Peak / ITCZ South)
12 - December |               82.49 mm | Seasonal Transition
```

*Note: The "Regional Context" column denotes documented meteorological literature for the Bogotá savannah, provided as reference context rather than as an algorithmic classification output.*

### Algorithmic Output:
- **Annual Amplitude**: $94.65\,\text{mm}/\text{month}$
- **Global Extremes**: Peak in November ($144.92\,\text{mm}$), trough in August ($50.27\,\text{mm}$)
- **Circular Peaks Detected**: Month 4 (April: $138.0\,\text{mm}$) and Month 11 (November: $144.9\,\text{mm}$)
- **Classified Regime**: **`BIMODAL`**

This confirms that the circular boundary algorithm reliably identifies the canonical bimodal structure of Bogotá with the default $10\%$ prominence threshold on noisy historical data. Full empirical validation across diverse tropical microclimates (unimodal Amazonian and Orinoquian basins, Pacific regimes, and arid Caribbean zones) is evaluated across the complete station catalog network.
