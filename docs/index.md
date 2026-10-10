# TROPICOR: Tropical Correction and Orographic Resolution

[![PyPI version](https://img.shields.io/pypi/v/tropicor.svg)](https://pypi.org/project/tropicor/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23219418.svg)](https://doi.org/10.5281/zenodo.23219418)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mhgualdron/tropicor/blob/main/notebooks/tropicor_quickstart.ipynb)

**TROPICOR** is a Python framework designed for bias correction, elevation lapse-rate adjustment, and topographic machine learning downscaling of global atmospheric reanalyses (such as **ERA5**) across complex tropical orography.

---

## 🏔️ The Core Problem

Global reanalysis models like **ERA5** provide comprehensive 30-km resolution atmospheric fields worldwide. While they perform remarkably well in flat terrain (e.g., Orinoquía and Caribbean lowlands), their smoothed orography introduces **severe systematic biases in tropical mountainous terrain** (such as the Colombian Andes):

- **Elevation Mismatches**: Grid-cell average elevation often deviates by over $1,000\,\text{m}$ from true meteorological stations.
- **Thermal Distortion**: Unresolved elevation discrepancies trigger $-6^\circ\text{C}$ to $-10^\circ\text{C}$ cold biases if left uncorrected.
- **Precipitation Phase Errors**: Complex ridge-valley microclimates are smoothed out.

TROPICOR bridges this gap by combining **canonical ground-truth station observations** (IDEAM DHIME), **physical environmental lapse-rate adjustments**, and **topography-aware machine learning regressors**.

---

## ⚡ Quick Installation

Install the latest release from PyPI:

```bash
pip install tropicor
```

Or install with documentation and development extras:

```bash
pip install "tropicor[docs,dev]"
```

---

## 🧭 Architecture at a Glance

TROPICOR is structured around a strict Directed Acyclic Graph (DAG):

- **Layer 0 (Leaf Modules)**: Canonical `StationCatalog` and natural regions.
- **Layer 1 (Ingestion Engines)**: Fault-tolerant `IdeamAdapter` for DHIME `.xlsx` files and `ERA5Adapter` for NetCDF extraction.
- **Layer 2 (Physics & Diagnostics)**: Orographic lapse-rate vertical adjustments ($\Delta z \cdot \Gamma$) and climatological cycle geometry.
- **Layer 3 (Machine Learning)**: Topographic feature extraction and supervised downscaling regressors.
- **Layer 4 (Facade Pipeline)**: High-level `TropicorPipeline` for end-to-end processing.

---

## 📖 Getting Started

Ready to begin? Head to our [Getting Started Guide](getting-started.md) to explore station metadata and parse native IDEAM exports in under 5 minutes.

---

## 📚 Academic Citation

If you use TROPICOR in your research, please cite:

```bibtex
@software{hernandez_gualdron_tropicor_2026,
  author = {Hernández Gualdrón, Mateo},
  title = {TROPICOR: Tropical Correction and Orographic Resolution},
  version = {0.1.3},
  year = {2026},
  doi = {10.5281/zenodo.23219418},
  url = {https://doi.org/10.5281/zenodo.23219418}
}
```
