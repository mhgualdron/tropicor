"""Input/Output module for station metadata, IDEAM DHIME, and ERA5 ingestion."""

from tropicor.io.era5 import ERA5Adapter
from tropicor.io.ideam import IdeamAdapter
from tropicor.io.stations import (
    BENCHMARK_STATIONS,
    NaturalRegion,
    StationCatalog,
    StationMetadata,
)

__all__ = [
    "ERA5Adapter",
    "IdeamAdapter",
    "NaturalRegion",
    "StationMetadata",
    "StationCatalog",
    "BENCHMARK_STATIONS",
]
