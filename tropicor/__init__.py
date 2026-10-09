"""TROPICOR: Tropical Correction and Orographic Resolution.

AI framework for bias correction and spatial downscaling
of global climate reanalyses across tropical complex terrains.
"""

from tropicor.core.alignment import align_common_period
from tropicor.core.climatology import (
    annual_cycle_amplitude,
    annual_cycle_peaks,
    annual_cycle_phase,
    classify_rainfall_regime,
    climatological_curve_distance,
    compute_monthly_climatology,
)
from tropicor.core.dtr import (
    compute_double_difference,
    compute_dtr,
    compute_monthly_extreme_range,
)
from tropicor.core.metrics import (
    TaylorStatistics,
    ValidationReport,
    compute_validation_metrics,
    taylor_statistics,
)
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

__version__ = "0.1.3"

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
    "TaylorStatistics",
    "ValidationReport",
    "__version__",
    "align_common_period",
    "annual_cycle_amplitude",
    "annual_cycle_peaks",
    "annual_cycle_phase",
    "classify_rainfall_regime",
    "climatological_curve_distance",
    "compute_double_difference",
    "compute_dtr",
    "compute_elevation_offset",
    "compute_monthly_climatology",
    "compute_monthly_extreme_range",
    "compute_validation_metrics",
    "lapse_rate_temperature_correction",
    "taylor_statistics",
]
