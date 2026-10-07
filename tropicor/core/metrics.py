"""Statistical verification metrics and validation reporting for climate reanalyses."""

import warnings
from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats

EPSILON: float = 1e-6


@dataclass(frozen=True)
class ValidationReport:
    """Immutable report of scientific validation metrics."""

    pearson_r: float
    p_value: float
    rmse: float
    mae: float
    bias: float
    pbias: float
    kge: float
    euclidean_distance: float
    n_samples: int

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to a Python dictionary."""
        return asdict(self)

    def to_series(self) -> pd.Series:
        """Convert metrics to a pandas Series for display and export."""
        return pd.Series(self.to_dict(), name="validation_metrics")


def compute_validation_metrics(
    observed: pd.Series,
    modeled: pd.Series,
    min_valid_samples: int = 3,
) -> ValidationReport:
    """Compute hydroclimatic validation metrics between observed and modeled series.

    Synchronizes indices via inner join, dropping any non-matching or NaN entries.
    Computes Pearson r, p-value, RMSE, MAE, mean bias, PBIAS, KGE, and Euclidean dist.

    Args:
        observed: In-situ ground truth observation time series.
        modeled: Reanalysis or downscaled model time series.
        min_valid_samples: Minimum number of valid synchronous samples required.

    Returns:
        ValidationReport containing all computed metrics.

    Raises:
        ValueError: If valid synchronous samples are fewer than min_valid_samples.
    """
    # Synchronize timestamps and drop mismatched NaNs
    df_sync = pd.concat(
        [observed.rename("obs"), modeled.rename("mod")], axis=1
    ).dropna()
    n_samples = len(df_sync)

    if n_samples < min_valid_samples:
        raise ValueError(
            f"Insufficient valid synchronous samples: found {n_samples}, "
            f"minimum required is {min_valid_samples}."
        )

    y_obs = df_sync["obs"].to_numpy(dtype=np.float64)
    y_mod = df_sync["mod"].to_numpy(dtype=np.float64)

    # Core error metrics
    rmse = float(np.sqrt(np.mean((y_mod - y_obs) ** 2)))
    mae = float(np.mean(np.abs(y_mod - y_obs)))
    bias = float(np.mean(y_mod) - np.mean(y_obs))
    euclidean_dist = float(np.sqrt(np.sum((y_mod - y_obs) ** 2)))

    # Percent Bias (PBIAS)
    sum_obs = float(np.sum(y_obs))
    if abs(sum_obs) < EPSILON:
        pbias = float("nan")
        warnings.warn(
            "PBIAS is undefined when sum of observed values is near zero.",
            UserWarning,
            stacklevel=2,
        )
    else:
        pbias = float(100.0 * np.sum(y_mod - y_obs) / sum_obs)

    # Pearson correlation r and p-value
    std_obs = float(np.std(y_obs, ddof=1)) if n_samples > 1 else 0.0
    std_mod = float(np.std(y_mod, ddof=1)) if n_samples > 1 else 0.0

    if std_obs < EPSILON or std_mod < EPSILON:
        pearson_r = 0.0
        p_val = 1.0
        warnings.warn(
            "Pearson correlation is undefined for zero-variance series. Returning r=0.",
            UserWarning,
            stacklevel=2,
        )
    else:
        r, p = stats.pearsonr(y_obs, y_mod)
        pearson_r = float(r)
        p_val = float(p)

    # Kling-Gupta Efficiency (KGE)
    mean_obs = float(np.mean(y_obs))
    mean_mod = float(np.mean(y_mod))

    if abs(mean_obs) < EPSILON:
        kge = float("nan")
        warnings.warn(
            "KGE is undefined when observed mean is near zero (zero bias ratio).",
            UserWarning,
            stacklevel=2,
        )
    elif std_obs < EPSILON:
        kge = float("nan")
        warnings.warn(
            "KGE is undefined when observed variance is near zero (zero alpha ratio).",
            UserWarning,
            stacklevel=2,
        )
    else:
        alpha = std_mod / std_obs
        beta = mean_mod / mean_obs
        kge = float(
            1.0
            - np.sqrt((pearson_r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
        )

    return ValidationReport(
        pearson_r=pearson_r,
        p_value=p_val,
        rmse=rmse,
        mae=mae,
        bias=bias,
        pbias=pbias,
        kge=kge,
        euclidean_distance=euclidean_dist,
        n_samples=n_samples,
    )
