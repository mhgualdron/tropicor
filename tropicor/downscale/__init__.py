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
from tropicor.downscale.precipitation import (
    BasePrecipitationDownscaler,
    Level1LinearScalingPrecipitation,
    Level2ThresholdMatchingPrecipitation,
    Level3QuantileMappingPrecipitation,
    Level3TuningResult,
    date_to_leap_doy,
    leap_doy_circular_distance,
    load_calibration_precipitation_pair,
    select_best_level3_result,
    tune_level3_precipitation_hyperparameters,
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
    "BasePrecipitationDownscaler",
    "Level1LinearScalingPrecipitation",
    "Level2ThresholdMatchingPrecipitation",
    "Level3QuantileMappingPrecipitation",
    "Level3TuningResult",
    "date_to_leap_doy",
    "leap_doy_circular_distance",
    "load_calibration_precipitation_pair",
    "select_best_level3_result",
    "tune_level3_precipitation_hyperparameters",
]
