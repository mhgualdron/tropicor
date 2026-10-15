"""Input/Output module for station metadata, IDEAM DHIME, and ERA5 ingestion."""

from tropicor.io.era5 import (
    VARIABLE_CONVENTIONS,
    ERA5Adapter,
    VariableAggregationConvention,
    aggregate_hourly_to_daily_precipitation,
    aggregate_hourly_to_daily_temperature,
    get_variable_convention,
)
from tropicor.io.ideam import (
    DhimeDailyPrecipitationAdapter,
    DhimeDailyTemperatureAdapter,
    IdeamAdapter,
    read_dhime_daily_precipitation,
    read_dhime_daily_temperature,
)
from tropicor.io.stations import (
    BENCHMARK_STATIONS,
    NaturalRegion,
    StationCatalog,
    StationMetadata,
)

__all__ = [
    "ERA5Adapter",
    "VariableAggregationConvention",
    "VARIABLE_CONVENTIONS",
    "get_variable_convention",
    "aggregate_hourly_to_daily_precipitation",
    "aggregate_hourly_to_daily_temperature",
    "IdeamAdapter",
    "DhimeDailyPrecipitationAdapter",
    "DhimeDailyTemperatureAdapter",
    "read_dhime_daily_precipitation",
    "read_dhime_daily_temperature",
    "NaturalRegion",
    "StationMetadata",
    "StationCatalog",
    "BENCHMARK_STATIONS",
]
