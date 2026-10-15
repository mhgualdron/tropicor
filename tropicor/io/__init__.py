"""Input/Output module for station metadata, IDEAM DHIME, and ERA5 ingestion."""

from tropicor.io.era5 import (
    ERA5Adapter,
    aggregate_hourly_to_daily_precipitation,
)
from tropicor.io.ideam import (
    DhimeDailyPrecipitationAdapter,
    IdeamAdapter,
    read_dhime_daily_precipitation,
)
from tropicor.io.stations import (
    BENCHMARK_STATIONS,
    NaturalRegion,
    StationCatalog,
    StationMetadata,
)

__all__ = [
    "ERA5Adapter",
    "aggregate_hourly_to_daily_precipitation",
    "IdeamAdapter",
    "DhimeDailyPrecipitationAdapter",
    "read_dhime_daily_precipitation",
    "NaturalRegion",
    "StationMetadata",
    "StationCatalog",
    "BENCHMARK_STATIONS",
]
