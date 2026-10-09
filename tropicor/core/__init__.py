"""Core scientific validation metrics and statistical algorithms."""

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
    ValidationReport,
    compute_validation_metrics,
)

__all__ = [
    "ValidationReport",
    "align_common_period",
    "annual_cycle_amplitude",
    "annual_cycle_peaks",
    "annual_cycle_phase",
    "classify_rainfall_regime",
    "climatological_curve_distance",
    "compute_double_difference",
    "compute_dtr",
    "compute_monthly_climatology",
    "compute_monthly_extreme_range",
    "compute_validation_metrics",
]
