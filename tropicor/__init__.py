"""TROPICOR: Tropical Correction and Orographic Resolution.

AI framework for bias correction and spatial downscaling
of global climate reanalyses across tropical complex terrains.
"""

from tropicor.core.metrics import ValidationReport, compute_validation_metrics
from tropicor.downscale.orography import (
    DRY_ADIABATIC_LAPSE_RATE,
    MOIST_ADIABATIC_LAPSE_RATE,
    STANDARD_LAPSE_RATE,
    compute_elevation_offset,
    lapse_rate_temperature_correction,
)
from tropicor.io.era5 import ERA5Adapter
from tropicor.io.ideam import IdeamAdapter
from tropicor.io.stations import (
    BENCHMARK_STATIONS,
    NaturalRegion,
    StationCatalog,
    StationMetadata,
)

__version__ = "0.1.1"

__all__ = [
    "BENCHMARK_STATIONS",
    "DRY_ADIABATIC_LAPSE_RATE",
    "ERA5Adapter",
    "IdeamAdapter",
    "MOIST_ADIABATIC_LAPSE_RATE",
    "NaturalRegion",
    "STANDARD_LAPSE_RATE",
    "StationCatalog",
    "StationMetadata",
    "ValidationReport",
    "__version__",
    "compute_elevation_offset",
    "compute_validation_metrics",
    "lapse_rate_temperature_correction",
]
