"""Input/Output module for station metadata and IDEAM DHIME ingestion."""

from tropicor.io.ideam import IdeamAdapter
from tropicor.io.stations import (
    BENCHMARK_STATIONS,
    NaturalRegion,
    StationCatalog,
    StationMetadata,
)

__all__ = [
    "IdeamAdapter",
    "NaturalRegion",
    "StationMetadata",
    "StationCatalog",
    "BENCHMARK_STATIONS",
]
