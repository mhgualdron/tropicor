# Getting Started with TROPICOR

In this 5-minute guide, you will learn how to:

1. Query Colombia's ground-truth meteorological stations using `StationCatalog`.
2. Parse a native IDEAM DHIME Excel export using `IdeamAdapter`.
3. Extract clean, monthly-aligned Pandas time series.

---

## 1. Exploring the Station Catalog

TROPICOR includes a canonical catalog of 48 benchmark stations across Colombia's 6 natural regions:

```python
from tropicor.io import StationCatalog, NaturalRegion

# Load benchmark catalog
catalog = StationCatalog.from_benchmark()
print(f"Total benchmark stations: {len(catalog)}")

# Lookup a station by its 8-digit IDEAM code
uis = catalog.get_by_code("23195040")
print(f"Station: {uis.name}")
print(f"Elevation: {uis.elevation} masl")
print(f"Coordinates: {uis.latitude}°N, {uis.longitude}°W")
print(f"Department: {uis.department} ({uis.region.value})")
```

You can also filter stations by natural region:

```python
# Filter all Andean stations
andina_stations = catalog.filter_by_region(NaturalRegion.ANDINA)
for st in andina_stations[:3]:
    print(f"- {st.name} ({st.code}): {st.elevation} m")
```

---

## 2. Ingesting Native IDEAM DHIME Exports

IDEAM's DHIME portal exports meteorological records as formatted Excel (`.xlsx`) spreadsheets with a 7-row metadata header block and data tables.

TROPICOR's `IdeamAdapter` reads these files directly without requiring manual cleaning or header deletion:

```python
from tropicor.io import IdeamAdapter

# Initialize adapter from downloaded DHIME file
adapter = IdeamAdapter("21205710.xlsx")

# Inspect parsed metadata
print(f"Station: {adapter.name} [{adapter.code}]")
print(f"Variable: {adapter.variable} ({adapter.unit})")
print(f"Elevation: {adapter.elevation} masl")
print(f"Department: {adapter.department}")

# Extract standardized time series
series = adapter.get_series()
print(series.head())
```

Output:
```text
Station: JARDIN BOTANICO [21205710]
Variable: PRECIPITACION (mm)
Elevation: 2552.0 masl
Department: Bogotá

1990-01-01     62.5
1990-02-01     44.9
1990-03-01     58.2
1990-04-01    143.3
Freq: 1MS, Name: precipitacion, dtype: float64
```

### Key Guarantees:
- **Strict DatetimeIndex**: Normalized to Start-of-Month (`1MS`) frequency.
- **Sentinel Coercion**: `-9999.0`, `-999.0`, and empty cells are automatically mapped to `numpy.nan`.
- **Zero Truncation**: Preserves periods and values faithfully for climatological analysis.
