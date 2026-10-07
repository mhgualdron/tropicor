"""Downscaling and orographic resolution engines for TROPICOR."""

from tropicor.downscale.orography import (
    DRY_ADIABATIC_LAPSE_RATE,
    MOIST_ADIABATIC_LAPSE_RATE,
    STANDARD_LAPSE_RATE,
    compute_elevation_offset,
    lapse_rate_temperature_correction,
)

__all__ = [
    "DRY_ADIABATIC_LAPSE_RATE",
    "MOIST_ADIABATIC_LAPSE_RATE",
    "STANDARD_LAPSE_RATE",
    "compute_elevation_offset",
    "lapse_rate_temperature_correction",
]
