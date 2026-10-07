# TROPICOR

**Tropical Correction and Orographic Resolution**  
*Scientific Python framework for physical vertical downscaling and statistical bias correction of atmospheric reanalyses across complex tropical terrain.*

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.1.1-green.svg)](https://pypi.org/project/tropicor/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23219419.svg)](https://doi.org/10.5281/zenodo.23219419)
[![Documentation](https://img.shields.io/badge/docs-mkdocs--material-teal.svg)](https://mhgualdron.github.io/tropicor/)

---

## Overview

Global atmospheric reanalyses like **ECMWF ERA5** represent topography on smoothed grid scales ($\sim 31\,\text{km}$). In steep mountainous regions such as the Colombian Andes, this horizontal averaging introduces severe vertical discrepancies ($\Delta z$ up to $\pm 1,500\,\text{m}$) between the model surface and ground-truth stations. 

Because temperature decreases with altitude at the environmental lapse rate ($\approx 6.5\,^\circ\text{C}/\text{km}$), smoothed model elevations report temperatures with systemic errors exceeding $8^\circ\text{C}$ in intramontane valleys.

**TROPICOR** operationalizes physical downscaling methods and validation engines into a modular, production-ready Python package.

---

## Core Features (v0.1.1 MVP)

- **Topographic Lapse-Rate Downscaling (`tropicor.downscale`)**: Physics-based vertical temperature adjustment resolving elevation discrepancies ($\Delta z = z_{\text{station}} - z_{\text{model}}$) using standard, dry, or moist adiabatic lapse rates.
- **ERA5 NetCDF Adapter (`tropicor.io.era5`)**: Memory-efficient spatial extraction from multi-file NetCDF reanalyses with automated unit harmonization (Kelvin $\to$ °C, meters $\to$ mm) and geopotential elevation extraction.
- **In-Situ Meteorological Ingestion (`tropicor.io.ideam`)**: Native parser for Colombian IDEAM DHIME meteorological time series.
- **Canonical Station Catalog (`tropicor.io.stations`)**: Built-in metadata catalog of 48 Colombian benchmark stations across all 6 natural regions (Andina, Caribe, Pacífico, Orinoquía, Amazonía, Insular).
- **Comprehensive Validation Suite (`tropicor.core.metrics`)**: Immutable `ValidationReport` computing Pearson $r$, RMSE, MAE, Mean Bias, PBIAS, Kling-Gupta Efficiency (KGE), and Euclidean curve distance with epsilon near-zero guards.

---

## Installation

```bash
pip install tropicor
```

Or install with all extras (documentation and visualization):

```bash
pip install "tropicor[viz,docs]"
```

---

## Quickstart

```python
import pandas as pd
from tropicor import (
    STANDARD_LAPSE_RATE,
    StationCatalog,
    compute_validation_metrics,
    lapse_rate_temperature_correction,
)

# 1. Retrieve benchmark station metadata (Bucaramanga UIS Station)
catalog = StationCatalog.from_benchmark()
station = catalog.get("23195040")

# 2. Correct raw ERA5 temperature for orographic offset (898m vs 2118m)
raw_era5_temp = pd.Series(
    [16.1, 16.4, 16.2], index=pd.date_range("2010-01-01", periods=3, freq="MS")
)
corrected_temp = lapse_rate_temperature_correction(
    temperature=raw_era5_temp,
    station_elevation=station.elevation,  # 898.0 masl
    model_elevation=2118.0,  # ERA5 smoothed grid
    lapse_rate=STANDARD_LAPSE_RATE,  # 0.0065 °C/m
)

# 3. Evaluate hydroclimatic metrics against ground observations
obs_temp = pd.Series(
    [24.0, 24.3, 24.1], index=pd.date_range("2010-01-01", periods=3, freq="MS")
)
report = compute_validation_metrics(observed=obs_temp, modeled=corrected_temp)

print(report.to_series())
```

---

## Academic Citation & Lineage

TROPICOR is an independent software implementation based on algorithms originally developed during an undergraduate geology thesis at **Universidad Nacional de Colombia** (Bogotá). The author gratefully acknowledges the past academic guidance of **PhD Germán Andrés Prieto Gómez** and **PhD Daniel Hernández Deckers** during that original research.

If you use TROPICOR in your research, please cite it using the metadata in [`CITATION.cff`](CITATION.cff) or as follows:

```bibtex
@software{hernandez_gualdron_tropicor_2026,
  author = {Hernández Gualdrón, Mateo},
  title = {TROPICOR: Tropical Correction and Orographic Resolution},
  version = {0.1.1},
  year = {2026},
  doi = {10.5281/zenodo.23219419},
  url = {https://doi.org/10.5281/zenodo.23219419}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
