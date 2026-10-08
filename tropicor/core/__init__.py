"""Core scientific validation metrics and statistical algorithms."""

from tropicor.core.climatology import (
    annual_cycle_amplitude,
    annual_cycle_peaks,
    annual_cycle_phase,
    classify_rainfall_regime,
    climatological_curve_distance,
    compute_monthly_climatology,
)
from tropicor.core.metrics import (
    ValidationReport,
    compute_validation_metrics,
)

__all__ = [
    "ValidationReport",
    "annual_cycle_amplitude",
    "annual_cycle_peaks",
    "annual_cycle_phase",
    "classify_rainfall_regime",
    "climatological_curve_distance",
    "compute_monthly_climatology",
    "compute_validation_metrics",
]
