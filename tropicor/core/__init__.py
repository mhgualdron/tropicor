"""Core scientific validation metrics and statistical algorithms."""

from tropicor.core.alignment import align_common_period
from tropicor.core.andean_registry import (
    AndeanStation,
    get_andean_partition,
    get_andean_station,
    load_andean_registry,
    request_andean_validation_partition,
)
from tropicor.core.bias import (
    EmpiricalQuantileMapping,
    LinearScalingCorrection,
    linear_scale_series,
    quantile_map_series,
)
from tropicor.core.climatology import (
    annual_cycle_amplitude,
    annual_cycle_peaks,
    annual_cycle_phase,
    classify_rainfall_regime,
    climatological_curve_distance,
    compute_monthly_climatology,
)
from tropicor.core.diagnostics import (
    DistributionMoments,
    ResidualDiagnosticsReport,
    TrendDiagnostics,
    compute_deseasonalized_residuals,
    compute_distribution_moments,
    compute_mann_kendall_trend,
    compute_residual_diagnostics,
    standardize_residuals,
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
from tropicor.core.station_selection import (
    compute_optimal_partition,
    evaluate_partition_eligibility,
    evaluate_slot_status,
)

__all__ = [
    "AndeanStation",
    "DistributionMoments",
    "EmpiricalQuantileMapping",
    "LinearScalingCorrection",
    "ResidualDiagnosticsReport",
    "TrendDiagnostics",
    "ValidationReport",
    "align_common_period",
    "annual_cycle_amplitude",
    "annual_cycle_peaks",
    "annual_cycle_phase",
    "classify_rainfall_regime",
    "climatological_curve_distance",
    "compute_deseasonalized_residuals",
    "compute_distribution_moments",
    "compute_double_difference",
    "compute_dtr",
    "compute_mann_kendall_trend",
    "compute_monthly_climatology",
    "compute_monthly_extreme_range",
    "compute_optimal_partition",
    "compute_residual_diagnostics",
    "compute_validation_metrics",
    "evaluate_partition_eligibility",
    "evaluate_slot_status",
    "get_andean_partition",
    "get_andean_station",
    "linear_scale_series",
    "load_andean_registry",
    "quantile_map_series",
    "request_andean_validation_partition",
    "standardize_residuals",
]
