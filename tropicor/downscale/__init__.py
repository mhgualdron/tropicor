"""Downscaling and orographic resolution engines for TROPICOR."""

from tropicor.downscale.orography import (
    DRY_ADIABATIC_LAPSE_RATE,
    MOIST_ADIABATIC_LAPSE_RATE,
    STANDARD_GRAVITY,
    STANDARD_LAPSE_RATE,
    altitude_temperature_adjustment,
    compute_elevation_offset,
    compute_orographic_bias_report,
    geopotential_to_height,
    lapse_rate_temperature_correction,
)

__all__ = [
    "DRY_ADIABATIC_LAPSE_RATE",
    "MOIST_ADIABATIC_LAPSE_RATE",
    "STANDARD_GRAVITY",
    "STANDARD_LAPSE_RATE",
    "altitude_temperature_adjustment",
    "compute_elevation_offset",
    "compute_orographic_bias_report",
    "geopotential_to_height",
    "lapse_rate_temperature_correction",
]
