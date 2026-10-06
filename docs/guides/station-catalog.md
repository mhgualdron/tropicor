# Working with the Station Catalog

The `StationCatalog` provides access to the 48 benchmark stations evaluated across Colombia's 6 natural regions.

---

## The `NaturalRegion` Taxonomy

Colombia exhibits sharp climatic contrasts partitioned into 6 natural geographic domains:

- `NaturalRegion.ANDINA`: High-altitude mountain ranges (Cordilleras Occidental, Central, Oriental).
- `NaturalRegion.CARIBE`: Lowland coastal plains and Caribbean river basins.
- `NaturalRegion.PACIFICO`: Super-humid tropical rainforest and coastal margins.
- `NaturalRegion.ORINOQUIA`: Eastern savannah flatlands (Llanos Orientales).
- `NaturalRegion.AMAZONIA`: Tropical rainforest biome.
- `NaturalRegion.INSULAR`: Oceanic islands including San Andrés, Providencia, Gorgona, and Malpelo.

---

## Querying and Filtering Stations

### Inspecting Bounding Boxes

```python
from tropicor.io import StationCatalog

catalog = StationCatalog()

min_lat, max_lat, min_lon, max_lon = catalog.bounding_box()
print(f"Domain Extent: Lat [{min_lat:.2f}, {max_lat:.2f}], Lon [{min_lon:.2f}, {max_lon:.2f}]")
```

### Exporting to Pandas DataFrame

Convert the entire catalog to a `pandas.DataFrame` for spatial analysis or plotting:

```python
df_stations = catalog.to_dataframe()

# Count stations per natural region
print(df_stations["region"].value_counts())

# Find highest elevation stations
highest = df_stations.sort_values(by="elevation", ascending=False)
print(highest[["name", "elevation", "region"]].head(5))
```

### Custom Station Catalogs

You can instantiate a custom catalog by passing a list of `StationMetadata` instances:

```python
from tropicor.io import StationCatalog, StationMetadata, NaturalRegion

custom_station = StationMetadata(
    code="99990001",
    name="PARAMO CHINGAZA",
    latitude=4.512,
    longitude=-73.745,
    elevation=3650.0,
    region=NaturalRegion.ANDINA,
    department="Cundinamarca",
    municipality="Fómeque",
)

custom_catalog = StationCatalog([custom_station])
print(len(custom_catalog))
```
