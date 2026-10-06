# Ingesting Native IDEAM DHIME Exports

The **Instituto de Hidrología, Meteorología y Estudios Ambientales (IDEAM)** provides ground-truth observation records through its **DHIME** (*Datos Hidrológicos y Meteorológicos*) platform.

---

## File Format & Structure

Native DHIME exports (`.xlsx`) contain two key sections:

### 1. Metadata Block (Rows 1–7)
```text
Row 1: Logo | Report title
Row 2: "Nombre estacion:" | <name> [<code>] | "Categoría de estación:" | ...
Row 3: "Latitud:" | <lat> | "Longitud:" | <lon> | "Elevación:" | <elev>
Row 4: "Entidad:" | ... | "Departamento:" | <department>
Row 5: "Municipio:" | <municipality>
Row 6: "Variable:" | <variable> | "Frecuencia:" | <frequency>
Row 7: "Parametro:" | <parameter> | "Unidad medida:" | <unit>
```

### 2. Data Table (Row 9 onwards)
- **Column 0 (`Fecha`)**: Timestamp formatted as `YYYY-MM-DD HH:MM`.
- **Column 2 (`Valor`)**: Observation measurement (using `.` as decimal separator).
- **Column 4 (`Nivel de Aprobación`)**: Quality flag (`Preliminar`, `Definitivo`).

---

## Working with `IdeamAdapter`

### Basic Usage

```python
from pathlib import Path
from tropicor.io import IdeamAdapter

file_path = Path("local/21205710.xlsx")
adapter = IdeamAdapter(file_path)

# Access metadata properties
print("Code:", adapter.code)
print("Name:", adapter.name)
print("Latitude:", adapter.latitude)
print("Longitude:", adapter.longitude)
print("Elevation:", adapter.elevation)
print("Variable:", adapter.variable)
print("Unit:", adapter.unit)
```

### Extracting Time Series

Call `adapter.get_series()` to retrieve a clean Pandas `Series`:

```python
series = adapter.get_series()

# Summary statistics
print(series.describe())

# Check for missing months
print(f"Total missing values: {series.isna().sum()}")
```

### Missing Value Handling

IDEAM often marks missing or uncalibrated readings with sentinel numbers such as `-9999.0` or `-999.0`. `IdeamAdapter` converts these automatically to `np.nan`:

```python
assert not (series == -9999.0).any()
```

### Combining Multi-Variable Stations

In IDEAM DHIME, precipitation and temperature are typically exported into separate Excel workbooks. You can load both and align them along their common `DatetimeIndex`:

```python
import pandas as pd
from tropicor.io import IdeamAdapter

# Load precipitation and temperature files
p_adapter = IdeamAdapter("path/to/precip.xlsx")
t_adapter = IdeamAdapter("path/to/temp.xlsx")

# Merge into a single station dataframe
df_station = pd.DataFrame({
    "precipitation": p_adapter.get_series(),
    "temperature": t_adapter.get_series(),
})

print(df_station.head())
```
